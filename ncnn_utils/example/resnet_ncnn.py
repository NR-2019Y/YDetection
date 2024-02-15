import subprocess
import ncnn
import numpy as np
import torch
from torch import nn
import torchvision
from ncnn_utils import graph_utils, F, Tensor, Operation, NCNNModel
from typing import Union, Tuple, List, Optional, Type, Sequence, Callable, Set, Dict, NamedTuple
from collections import defaultdict


class Conv2d:
    def __init__(self, weight_data: np.ndarray, bias_data: Optional[np.ndarray] = None,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1):
        self.wdatas: Tuple[np.ndarray, Optional[np.ndarray]] = (weight_data, bias_data)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    @staticmethod
    def from_torch_module(conv: nn.Conv2d) -> 'Conv2d':
        assert conv.groups == 1
        assert not conv.transposed
        assert (not isinstance(conv.padding, str)) and (len(conv.padding) == 2)
        padding = conv._reversed_padding_repeated_twice
        return Conv2d(conv.weight.data.numpy(),
                      None if conv.bias is None else conv.bias.data.numpy(),
                      stride=conv.stride, padding=padding, dilation=conv.dilation)  # type: ignore

    def __call__(self, x: Tensor) -> Tensor:
        weight_data, bias_data = self.wdatas
        oc, ic, kh, kw = weight_data.shape
        if bias_data is not None:
            assert bias_data.shape == (oc,)
        return F.convolution(x, weight_data, bias_data, (kh, kw), self.stride, self.padding, self.dilation)


class BatchNorm2d:
    def __init__(self, bn_weight: np.ndarray, bn_bias: np.ndarray,
                 running_mean: np.ndarray, running_var: np.ndarray, eps: float):
        self.wdatas = (bn_weight, bn_bias, running_mean, running_var)
        self.eps = eps

    @staticmethod
    def from_torch_module(bn: nn.BatchNorm2d) -> 'BatchNorm2d':
        return BatchNorm2d(bn.weight.data.numpy(), bn.bias.data.numpy(),
                           bn.running_mean.data.numpy(), bn.running_var.data.numpy(),
                           eps=bn.eps)

    def __call__(self, x: Tensor) -> Tensor:
        bn_weight, bn_bias, running_mean, running_var = self.wdatas
        return F.batch_norm(x, bn_weight, bn_bias, running_mean, running_var, eps=self.eps)


def fuse_torch_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Conv2d:
    conv_weight = conv.weight.data.numpy()
    conv_bias = None if conv.bias is None else conv.bias.data.numpy()
    bn_weight, bn_bias, running_mean, running_var = \
        bn.weight.data.numpy(), bn.bias.data.numpy(), bn.running_mean.data.numpy(), bn.running_var.data.numpy()
    eps = bn.eps
    # eps = 1e-20
    # gamma_mul_inv_std = bn_weight / (np.sqrt(running_var) + eps)
    gamma_mul_inv_std = bn_weight / np.sqrt(running_var + eps)
    new_bias = bn_bias - running_mean * gamma_mul_inv_std
    if conv_bias is not None:
        new_bias += conv_bias
    new_weight = conv_weight * gamma_mul_inv_std[:, None, None, None]
    assert conv.groups == 1
    assert not conv.transposed
    assert (not isinstance(conv.padding, str)) and (len(conv.padding) == 2)
    padding = conv._reversed_padding_repeated_twice
    return Conv2d(new_weight, new_bias, conv.stride, padding, conv.dilation)  # type: ignore


class Linear:
    def __init__(self, weight_data: np.ndarray, bias_data: Optional[np.ndarray] = None):
        self.wdatas: Tuple[np.ndarray, Optional[np.ndarray]] = (weight_data, bias_data)

    @staticmethod
    def from_torch_module(l: nn.Linear) -> 'Linear':
        return Linear(l.weight.data.numpy(), None if l.bias is None else l.bias.data.numpy())

    def __call__(self, x: Tensor) -> Tensor:
        weight_data, bias_data = self.wdatas
        return F.linear(x, weight_data, bias_data)


class Sequential:
    def __init__(self, modules: Sequence[Callable[[Tensor], Tensor]]):
        self.modules = modules

    def __call__(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x


def r18_pool_fn(x: Tensor) -> Tensor:
    return F.maxpool2d(x, ksize=(3, 3), stride=(2, 2), padding=(1, 1, 1, 1))


class BasicBlock:
    def __init__(self,
                 conv1: Callable[[Tensor], Tensor],
                 conv2: Callable[[Tensor], Tensor],
                 downsample: Optional[Callable[[Tensor], Tensor]]):
        self.conv1 = conv1
        self.conv2 = conv2
        self.downsample = downsample

    @staticmethod
    def from_torch_module(block: nn.Module, fuse_bn: bool = True):
        if fuse_bn:
            conv1 = fuse_torch_conv_bn(block.conv1, block.bn1)
            conv2 = fuse_torch_conv_bn(block.conv2, block.bn2)
        else:
            conv1 = Sequential([
                Conv2d.from_torch_module(block.conv1),
                BatchNorm2d.from_torch_module(block.bn1)
            ])
            conv2 = Sequential([
                Conv2d.from_torch_module(block.conv2),
                BatchNorm2d.from_torch_module(block.bn2)
            ])
        downsample = None
        if block.downsample is not None:
            if fuse_bn:
                downsample = fuse_torch_conv_bn(block.downsample[0], block.downsample[1])
            else:
                downsample = Sequential([
                    Conv2d.from_torch_module(block.downsample[0]),
                    BatchNorm2d.from_torch_module(block.downsample[1])
                ])
        return BasicBlock(conv1, conv2, downsample)

    def __call__(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x)


def make_resnet18(r18_model: nn.Module, fuse_bn: bool) -> Callable[[Tensor], Tensor]:
    if fuse_bn:
        conv1 = fuse_torch_conv_bn(r18_model.conv1, r18_model.bn1)
    else:
        conv1 = Sequential([
            Conv2d.from_torch_module(r18_model.conv1),
            BatchNorm2d.from_torch_module(r18_model.bn1)
        ])
    layers = [
        Sequential([BasicBlock.from_torch_module(l, fuse_bn=fuse_bn) for l in r18_model_layer])
        for r18_model_layer in (r18_model.layer1, r18_model.layer2, r18_model.layer3, r18_model.layer4)
    ]
    fc = Linear.from_torch_module(r18_model.fc)
    return Sequential([conv1, F.relu, r18_pool_fn, *layers, F.global_avgpool2d, fc])
    # return Sequential([conv1, F.relu, r18_pool_fn, *layers])
    # return Sequential([conv1, F.relu, r18_pool_fn])


def main():
    r18_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()

    # r18_torch = nn.Sequential(r18_model.conv1, r18_model.bn1, r18_model.relu, r18_model.maxpool,
    #                           r18_model.layer1, r18_model.layer2, r18_model.layer3, r18_model.layer4).eval()
    # r18_torch = nn.Sequential(r18_model.conv1, r18_model.bn1, r18_model.relu, r18_model.maxpool).eval()
    r18_torch = r18_model

    # r18_ncnn = make_resnet18(r18_model, fuse_bn=False)
    r18_ncnn = make_resnet18(r18_model, fuse_bn=True)
    with graph_utils.graph_scope(graph_utils.Graph()) as g:
        X_ = F.placeholder((3, 224, 224))
        out = r18_ncnn(X_)

    def eval_ncnn(x: np.ndarray, net: ncnn.Net) -> np.ndarray:
        # print(net.input_names(), net.output_names())
        iname = net.input_names()[0]
        oname = net.output_names()[0]
        ex = net.create_extractor()
        ex.input(iname, ncnn.Mat(x))
        _, nout = ex.extract(oname)
        return nout.numpy()

    @torch.no_grad()
    def eval_torch(x: np.ndarray) -> np.ndarray:
        return r18_torch(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()

    ncnn_model = NCNNModel.from_graph(g, outputs=[out])
    save_prefix = "/home/a/Desktop/tmp/r18"
    ncnn_model.to_file(save_prefix)

    net = ncnn.Net()
    net.load_param(f"{save_prefix}.param")
    net.load_model(f"{save_prefix}.bin")

    torch.onnx.export(r18_torch, torch.rand(1, 3, 224, 224), f"{save_prefix}.onnx")
    subprocess.call(["onnx2ncnn", f"{save_prefix}.onnx", f"{save_prefix}.2.param", f"{save_prefix}.2.bin"])
    net2 = ncnn.Net()
    net2.load_param(f"{save_prefix}.2.param")
    net2.load_model(f"{save_prefix}.2.bin")

    for _ in range(10):
        x = np.random.rand(3, 224, 224).astype(np.float32)

        y1 = eval_ncnn(x, net)
        y2 = eval_ncnn(x, net2)
        y3 = eval_torch(x)

        r = y1 / (y3 + 1e-20)
        print("13", np.abs(y1 - y3).max(), r.max(), r.min(), np.allclose(y1, y3))
        r = y2 / (y3 + 1e-20)
        print("23", np.abs(y2 - y3).max(), r.max(), r.min(), np.allclose(y2, y3))
        r = y1 / (y2 + 1e-20)
        print("12", np.abs(y1 - y2).max(), r.max(), r.min(), np.allclose(y1, y2))


if __name__ == '__main__':
    main()
