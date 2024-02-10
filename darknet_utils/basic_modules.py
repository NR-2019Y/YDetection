from .parse_darknet_cfg import parse_darknet_cfg, Section
import operator
import numpy as np
import torch
import functools
from torch import nn
from typing import Dict, Tuple, Optional, NamedTuple, Type, List, Iterable, Sequence, Union, Callable
from collections import defaultdict

shape_type = Tuple[int, ...]

ACTIVATION_DICT: Dict[str, Optional[nn.Module]] = {
    'logistic': nn.Sigmoid(),
    'relu': nn.ReLU(inplace=True),
    'leaky': nn.LeakyReLU(0.1, inplace=True),
    'linear': None
}

TYPE2MODULE: Dict[str, Callable] = dict()


class RegisterModule:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, module_cls: Type[nn.Module]) -> Type[nn.Module]:
        assert self.name not in TYPE2MODULE
        TYPE2MODULE[self.name] = module_cls
        return module_cls


def compute_numel(shape: shape_type) -> int:
    return functools.reduce(operator.mul, shape)


@RegisterModule("[convolutional]")
class DarknetConvolution(nn.Sequential):
    @staticmethod
    def compute_output_shape(input_shape: shape_type,
                             ksize: int, stride: int, pad: int, out_ch: int) -> shape_type:
        _, in_h, in_w = input_shape
        out_h = None if in_h is None \
            else (in_h + 2 * pad - ksize) // stride + 1
        out_w = None if in_w is None \
            else (in_w + 2 * pad - ksize) // stride + 1
        return out_ch, out_h, out_w

    def __init__(self, input_shape: shape_type, options: Dict[str, str]):
        super(DarknetConvolution, self).__init__()
        in_ch: int = input_shape[0]
        out_ch: int = int(options.get('filters', 1))
        ksize: int = int(options.get('size', 1))
        stride: int = int(options.get('stride', 1))
        use_pad: int = int(options.get('pad', 0))
        padding: int = int(options.get('padding', 0))
        groups: int = int(options.get('groups', 1))
        assert groups == 1
        if use_pad:
            padding = ksize // 2
        activation_type = options.get('activation', 'logistic')
        use_bn = int(options.get('batch_normalize', 0))

        self.output_shape: shape_type = self.compute_output_shape(input_shape,
                                                                  ksize, stride, padding, out_ch)
        # print("conv:", input_shape, self.output_shape)
        self.conv_kernel_shape = (out_ch, in_ch, ksize, ksize)
        self.out_ch = out_ch

        self.conv = nn.Conv2d(in_ch, out_ch, (ksize, ksize),
                              (stride, stride), padding, bias=(not use_bn))
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_ch, eps=1e-10)
        self.act = ACTIVATION_DICT[activation_type]

    def load_darknet_weights(self, fp):
        out_ch = self.out_ch
        if self.use_bn:
            shapes = ((out_ch,), (out_ch,), (out_ch,),
                      (out_ch,), self.conv_kernel_shape)
            wdatas = [
                np.fromfile(fp, np.float32, count=compute_numel(shape)).reshape(shape) for shape in shapes
            ]
            bn_bias, bn_weights, bn_running_mean, bn_running_var, conv_weight = wdatas
            self.bn.bias.data.copy_(torch.from_numpy(bn_bias))
            self.bn.weight.data.copy_(torch.from_numpy(bn_weights))
            self.bn.running_mean.data.copy_(torch.from_numpy(bn_running_mean))
            self.bn.running_var.data.copy_(torch.from_numpy(bn_running_var))
            self.conv.weight.data.copy_(torch.from_numpy(conv_weight))
            #
            # inv_bn_std = 1. / np.sqrt(bn_running_var + 1e-20)
            # gamma_mul_inv_bn_std = bn_weights * inv_bn_std
            #
            # BIAS = bn_bias - bn_running_mean * gamma_mul_inv_bn_std
            # WEIGHTS = conv_weight * gamma_mul_inv_bn_std[:, None, None, None]

        else:
            BIAS = np.fromfile(fp, np.float32, count=out_ch)
            WEIGHTS = np.fromfile(fp, np.float32, count=compute_numel(self.conv_kernel_shape)).reshape(
                self.conv_kernel_shape
            )

            self.conv.bias.data.copy_(torch.from_numpy(BIAS))
            self.conv.weight.data.copy_(torch.from_numpy(WEIGHTS))

    def fuse_bn(self):
        if not self.use_bn:
            assert not hasattr(self, "bn")
            return
        self.use_bn = False
        assert self.conv.bias is None
        bn_bias = self.bn.bias.data
        bn_weights = self.bn.weight.data
        bn_running_mean = self.bn.running_mean
        bn_running_var = self.bn.running_var
        eps = self.bn.eps

        inv_std = 1. / bn_running_var.add(eps).sqrt()
        gamma_mul_inv_std = bn_weights.mul(inv_std)
        new_bias = bn_bias.sub(bn_running_mean.mul(gamma_mul_inv_std))
        self.conv.bias = nn.Parameter(new_bias)
        self.conv.weight.data.mul_(gamma_mul_inv_std[:, None, None, None])
        del self.bn


@RegisterModule("[maxpool]")
class DarknetMaxPool(nn.Sequential):
    @staticmethod
    def compute_output_shape(input_shape: shape_type,
                             ksize: int, stride: int, padding: int) -> shape_type:
        in_ch, in_h, in_w = input_shape
        out_h = None if in_h is None \
            else (in_h - ksize + padding) // stride + 1
        out_w = None if in_w is None \
            else (in_w - ksize + padding) // stride + 1
        return in_ch, out_h, out_w

    def __init__(self, input_shape: shape_type, options: Dict[str, str]):
        super(DarknetMaxPool, self).__init__()
        stride: int = int(options.get('stride', 1))
        ksize: int = int(options.get('size', stride))
        padding: int = int(options.get('padding', ksize - 1))
        self.output_shape: shape_type = self.compute_output_shape(
            input_shape, ksize, stride, padding)
        if padding:
            p1 = padding // 2
            p2 = padding - p1
            self.pad = nn.ConstantPad2d([p1, p2, p1, p2], float('-inf'))
        self.pool = nn.MaxPool2d(ksize, stride)


@RegisterModule("[connected]")
class DarknetConnected(nn.Sequential):
    def __init__(self, input_shape: Tuple[int, ...], options: Dict[str, str]):
        super(DarknetConnected, self).__init__()
        if len(input_shape) > 1:
            self.flat = nn.Flatten()
        input_numel: int = compute_numel(input_shape)
        output_numel: int = int(options.get('output', 1))
        activation_type = options.get('activation', 'logistic')
        assert 'batch_normalize' not in options
        # use_bn = int(options.get('batch_normalize', 0))
        self.output_shape: shape_type = (output_numel,)
        self.connected = nn.Linear(input_numel, output_numel)
        self.act = ACTIVATION_DICT[activation_type]

    def load_darknet_weights(self, fp):
        for param in (self.connected.bias, self.connected.weight):
            param.data.copy_(torch.from_numpy(np.fromfile(fp, count=param.numel(), dtype=np.float32))
                             .view(param.shape))


@RegisterModule("[region]")
class DarknetRegion(nn.Module):
    def __init__(self, input_shape: shape_type, options: Dict[str, str]):
        super().__init__()
        self.output_shape: shape_type = input_shape
        self.coords: int = int(options.get("coords", 4))
        assert self.coords == 4
        self.classes: int = int(options.get("classes", 20))
        self.num: int = int(options.get("num", 1))
        biases = options.get("anchors", None)
        if biases is not None:
            biases = torch.tensor(
                [float(f) for f in biases.split(",")], dtype=torch.float32)
        else:
            biases = torch.full((self.num * 2,),
                                fill_value=0.5, dtype=torch.float32)
        self.register_buffer("biases", biases)

        self.softmax = int(options.get("softmax", 0))
        background = int(options.get("background", 0))
        assert not background

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        b, ic, ih, iw = X.shape
        X = X.view(b * self.num, -1, ih, iw)
        assert X.shape[1] == self.classes + self.coords + 1
        coord_cxcy = X[:, 0:2].sigmoid()
        coord_wh = X[:, 2:4]
        obj_conf = X[:, 4:5].sigmoid()
        if self.softmax:
            cls_conf = torch.softmax(X[:, 5:], dim=1)
        else:
            cls_conf = X[:, 5:].sigmoid()
        return torch.cat((coord_cxcy, coord_wh, obj_conf, cls_conf),
                         dim=1).view(b, ic, ih, iw)


@RegisterModule("[route]")
class DarknetRoute(nn.Module):
    multi_input = True

    def __init__(self, shapes: Sequence[shape_type], current_index: int, options: Dict[str, str]):
        super().__init__()
        layers = [int(l) for l in options["layers"].split(",")]
        c, h, w = 0, 0, 0
        for i, l in enumerate(layers):
            if l < 0:
                layers[i] += current_index
            shape = shapes[layers[i]]
            if i == 0:
                c, h, w = shape
            else:
                ci, hi, wi = shape
                assert hi == h and wi == w, "h or w dismatch"
                c += ci
        # print("cat:", [shapes[l] for l in layers])
        self.layers = layers
        self.output_shape: shape_type = (c, h, w)

    def forward(self, XList: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        if len(XList) == 1:
            return XList[0]
        return torch.cat(XList, 1)


# src/parser.c
# src/blas.c
@RegisterModule("[reorg]")
class DarknetReorg(nn.Module):
    def __init__(self, input_shape: shape_type, options: Dict[str, str]):
        super().__init__()
        self.stride: int = int(options.get("stride", 1))
        assert "reverse" not in options
        assert "flatten" not in options
        assert "extra" not in options
        in_ch, in_h, in_w = input_shape
        self.output_shape: shape_type = (in_ch * self.stride * self.stride, in_h // self.stride, in_w // self.stride)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        b, c, h, w = X.shape
        s = self.stride
        return X.view(b, c // (s * s), h, s, w, s).permute(0, 3, 5, 1, 2, 4).reshape(b, c * s * s, h // s, w // s)


@RegisterModule("[detection]")
class DarknetDetection(nn.Module):
    def __init__(self, input_shape: shape_type, options: Dict[str, str]):
        super().__init__()
        use_softmax = int(options.get("softmax", 0))
        assert use_softmax == 0
        use_reorg = int(options.get("reorg", 0))
        assert use_reorg == 0
        self.output_shape: shape_type = input_shape

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X


@RegisterModule("[upsample]")
class DarknetUpsample(nn.Module):
    def __init__(self, input_shape: shape_type, options: Dict[str, str]):
        super().__init__()
        stride = int(options.get("stride", 2))
        scale = float(options.get("scale", 1.0))
        assert stride > 0
        in_ch, in_h, in_w = input_shape
        self.output_shape: shape_type = in_ch, in_h * stride, in_w * stride
        self.upsample = nn.Upsample(scale_factor=float(stride), mode='nearest')
        self.scale = scale

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.upsample(X).mul_(self.scale)


@RegisterModule("[yolo]")
class DarknetYolo(nn.Module):
    net_h: int
    net_w: int

    def __init__(self, input_shape: shape_type, options: Dict[str, str]):
        super().__init__()
        self.output_shape: shape_type = input_shape
        self.classes = int(options["classes"])
        anchors = torch.tensor([float(f) for f in options["anchors"].split(',')],
                               dtype=torch.float32).view(-1, 2)
        mask = [int(m) for m in options["mask"].split(',')]
        for i, m in enumerate(mask):
            assert m == mask[0] + i
        current_anchors = anchors[mask[0]:mask[0] + len(mask)]
        self.register_buffer("anchors", anchors)
        self.register_buffer("current_anchors", current_anchors)
        self.mask = mask
        self.ignore_thresh = float(options.get("ignore_thresh", 0.5))
        self.truth_thresh = float(options.get("truth_thresh", 1.0))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        b, c, h, w = X.shape
        nmask = len(self.mask)
        assert c // nmask == self.classes + 5
        X = X.view(b, nmask, c // nmask, h, w)
        X[:, :, :2].sigmoid_()
        X[:, :, 4:].sigmoid_()
        return X.view(b, c, h, w)


class TensorWithCnt(NamedTuple):
    tensor: torch.Tensor
    cnt: int


class TensorDict:
    def __init__(self):
        self.dic: Dict[int, TensorWithCnt] = dict()

    def add_tensor(self, key: int, ctensor: TensorWithCnt):
        self.dic[key] = ctensor

    def get(self, key: int):
        assert key in self.dic
        cx: TensorWithCnt = self.dic[key]
        if cx.cnt == 1:
            del self.dic[key]
        else:
            assert cx.cnt > 1
            cx.cnt -= 1
        return cx.tensor


class DarknetInferenceModel(nn.Module):
    def __init__(self, input_shape: shape_type, section_iter: Iterable[Section], fp, fuse_bn: bool):
        super().__init__()
        tensor_shapes = []
        layer_cnts = defaultdict(int)
        for i, section in enumerate(section_iter):
            layer_type_cls = TYPE2MODULE[section.stype]
            if getattr(layer_type_cls, "multi_input", False):
                curr_layer = layer_type_cls(tensor_shapes, i, section.options)
                for l in curr_layer.layers:
                    if l != i - 1:
                        layer_cnts[l] += 1
            else:
                curr_layer = layer_type_cls(input_shape, section.options)
            self.add_module(f"{layer_type_cls.__name__}_L{i}", curr_layer)
            input_shape = curr_layer.output_shape
            tensor_shapes.append(input_shape)
            if hasattr(curr_layer, "load_darknet_weights"):
                curr_layer.load_darknet_weights(fp)
            if fuse_bn and isinstance(curr_layer, DarknetConvolution):
                curr_layer.fuse_bn()
        self.layer_cnts = layer_cnts

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        layer_cnts = self.layer_cnts
        saved_tensors = TensorDict()
        for i, layer in enumerate(self.children()):
            if getattr(layer.__class__, "multi_input", False):
                inputs = []
                for l in layer.layers:
                    if l == i - 1:
                        inputs.append(X)
                    else:
                        inputs.append(saved_tensors.get(l))
                X = layer(inputs)
            else:
                X = layer(X)
            cnt: int = layer_cnts.get(i, 0)
            if cnt:
                saved_tensors.add_tensor(i, TensorWithCnt(tensor=X, cnt=cnt))
        return X


class DarknetModel(NamedTuple):
    model: nn.Module
    input_height: int
    input_width: int
    input_channels: int


def BuildDarknetInferenceModel(cfg_file: str, weight_file: str, fuse_bn: bool = True) -> DarknetModel:
    all_sections = parse_darknet_cfg(cfg_file)
    with open(weight_file, 'rb') as fp:
        major, minor, revision = np.fromfile(fp, np.int32, 3)
        # print(major, minor, revision)
        transpose = (major > 1000) or (minor > 1000)
        assert not transpose
        if (major * 10 + minor >= 2) and (major < 1000) and (minor < 1000):
            fp.read(8)
        else:
            fp.read(4)

        section_iter = iter(all_sections)
        net_section = next(section_iter)
        assert net_section.stype == '[net]'
        input_height: int = int(net_section.options['height'])
        input_width: int = int(net_section.options['width'])
        input_channels: int = int(net_section.options['channels'])

        input_shape = (input_channels, input_height, input_width)
        model = DarknetInferenceModel(input_shape, section_iter, fp, fuse_bn=fuse_bn)
        model.eval()
        assert len(fp.read()) == 0

    return DarknetModel(model=model,
                        input_height=input_height,
                        input_width=input_width,
                        input_channels=input_channels)
