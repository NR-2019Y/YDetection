import torch
from .. import parse_darknet_cfg, Section
from collections import defaultdict
from ..basic_modules import DarknetYolo, DarknetConvolution, TYPE2MODULE, TensorDict, TensorWithCnt
from ..basic_modules import shape_type
from typing import List, Sequence, Tuple, Dict, Optional
from .post_process import yolov3_dets
from .loss import yolov3_dets_and_loss
from torch import nn
import numpy as np


class YoloLayer(DarknetYolo):
    def __init__(self, input_shape: shape_type, options: Dict[str, str], net_h: int, net_w: int):
        super(YoloLayer, self).__init__(input_shape, options)  # noqa
        self.inference_only: bool = False
        net_wh_tensor = torch.tensor([net_w, net_h], dtype=self.anchors.dtype, device=self.anchors.device)
        self.register_buffer("anchors_scale", self.anchors.div(net_wh_tensor))
        self.register_buffer("current_anchors_scale", self.current_anchors.div(net_wh_tensor))

    def forward(self, preds: torch.Tensor, target_list: Sequence[torch.Tensor] = None):
        if target_list is None:
            assert self.inference_only
            return yolov3_dets(preds, self.current_anchors_scale)
        assert not self.inference_only
        return yolov3_dets_and_loss(preds=preds,
                                    target_list=target_list,
                                    anchors_scale=self.anchors_scale,
                                    current_anchors_scale=self.current_anchors_scale,
                                    valid_index_start=self.mask[0],
                                    ignore_thresh=self.ignore_thresh,
                                    truth_thresh=self.truth_thresh,
                                    valid_index_end=self.mask[-1] + 1)


class YoloV3Model(nn.Module):
    input_height: int
    input_width: int
    input_channels: int

    def __init__(self, cfg_file: str, weight_file: Optional[str], num_classes: int, na: int = 3):
        super().__init__()
        self._inference_only: bool = False
        all_sections: List[Section] = parse_darknet_cfg(cfg_file)
        fp = None
        if weight_file is not None:
            fp = open(weight_file, "rb")
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
                if section.stype == "[yolo]":
                    curr_layer = YoloLayer(input_shape, section.options, net_h=input_height, net_w=input_width)
                else:
                    curr_layer = layer_type_cls(input_shape, section.options)
            if fp is not None and hasattr(curr_layer, "load_darknet_weights"):
                curr_layer.load_darknet_weights(fp)
            if isinstance(curr_layer, DarknetConvolution) and (not curr_layer.use_bn):
                ori_conv_kernel_shape = curr_layer.conv_kernel_shape
                assert ori_conv_kernel_shape[2:] == (1, 1)
                assert curr_layer.act is None
                in_ch = ori_conv_kernel_shape[1]
                out_ch = na * (num_classes + 5)
                curr_layer.output_shape = (out_ch,) + curr_layer.output_shape[1:]
                curr_layer.conv_kernel_shape = (out_ch,) + ori_conv_kernel_shape[1:]
                curr_layer.out_ch = out_ch
                curr_layer.conv = nn.Conv2d(in_ch, out_ch, (1, 1), (1, 1), 0, bias=True)
            self.add_module(f"{curr_layer.__class__.__name__}_L{i}", curr_layer)
            input_shape = curr_layer.output_shape
            tensor_shapes.append(input_shape)
        if fp is not None:
            assert len(fp.read()) == 0
            fp.close()
        self.layer_cnts = layer_cnts
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

    def set_inference_mode(self):
        self._inference_only = True
        for layer in self.children():
            if isinstance(layer, DarknetConvolution):
                layer.fuse_bn()
            elif isinstance(layer, YoloLayer):
                layer.inference_only = True

    @torch.no_grad()
    def inference_forward(self, X: torch.Tensor) -> torch.Tensor:
        layer_cnts = self.layer_cnts
        saved_tensors = TensorDict()
        all_dets: List[torch.Tensor] = []
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
                if isinstance(layer, YoloLayer):
                    dets = layer(X)
                    all_dets.append(dets)
                    X = None
                else:
                    X = layer(X)
            cnt: int = layer_cnts.get(i, 0)
            if cnt:
                saved_tensors.add_tensor(i, TensorWithCnt(tensor=X, cnt=cnt))
        return torch.cat(all_dets, dim=1)

    def forward(self, X: torch.Tensor, target_list: Optional[Sequence[torch.Tensor]] = None):
        if target_list is None:
            assert self._inference_only
            return self.inference_forward(X)
        assert not self._inference_only
        layer_cnts = self.layer_cnts
        saved_tensors = TensorDict()

        dets_list = []
        loss_box_list = []
        loss_obj_list = []
        loss_cls_list = []

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
                if not isinstance(layer, YoloLayer):
                    X = layer(X)
                else:
                    dets, (loss_box, loss_obj, loss_cls) = layer(X, target_list)
                    X = None
                    dets_list.append(dets)
                    loss_box_list.append(loss_box)
                    loss_obj_list.append(loss_obj)
                    loss_cls_list.append(loss_cls)
            cnt: int = layer_cnts.get(i, 0)
            if cnt:
                saved_tensors.add_tensor(i, TensorWithCnt(tensor=X, cnt=cnt))
        return torch.cat(dets_list, dim=1), (
            torch.stack(loss_box_list).sum(),
            torch.stack(loss_obj_list).sum(),
            torch.stack(loss_cls_list).sum()
        )

# if __name__ == '__main__':
#     print(TYPE2MODULE["[yolo]"])
#     l = YoloLayer((10, 200, 200), {"classes": "80", "anchors": "1,2,3,4,5,6", "mask": "0,1,2"})
#     print(YoloLayer.mro())
