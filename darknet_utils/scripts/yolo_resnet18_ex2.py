import math
import torch
from torch import nn
from darknet_utils.training.models import get_resnet18_backbone, _CBL
from darknet_utils.training.utils import cxcywh2xyxy2, broadcast_compute_box_iou, broadcast_compute_wh_iou
from darknet_utils.training.loss import _encode_yolov1_targets
from typing import Sequence, List


class YoloResnet18(nn.Module):
    def __init__(self, num: int, num_classes: int):
        super().__init__()
        self.backbone = get_resnet18_backbone()
        num_features = num_classes + num * 5
        self.head = nn.Sequential(
            _CBL(512, 1024, 3, 1, 1),
            _CBL(1024, 512, 3, 1, 1),
            nn.Conv2d(512, num_features, (1, 1), (1, 1), 0)
        )
        self.num = num
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        batch_size, h, w, ch = x.shape
        box = x[..., self.num_classes + self.num:].reshape(batch_size, h, w, self.num, 4)
        box = torch.cat((box[..., 0:2].sigmoid(), box[..., 2:4]), dim=-1).reshape(batch_size, h, w, self.num * 4)
        return torch.cat((x[..., :self.num_classes + self.num].sigmoid(), box), dim=-1)


@torch.no_grad()
def yolo_dets_ex(preds: torch.Tensor, num_classes: int, wh_scales: List[float]) -> torch.Tensor:
    batch_size, h, w, _ = preds.shape
    # wh_tensor = torch.tensor([w, h], dtype=torch.float32, device=preds.device)
    wh_scales_tensor = torch.tensor(wh_scales, dtype=torch.float32, device=preds.device).view(-1, 2)
    num: int = len(wh_scales_tensor)

    preds_cls = preds[..., :num_classes]
    preds_cls_conf, preds_cls_id = preds_cls.max(-1)  # [batch_size, h, w]
    preds_obj = preds[..., num_classes:num_classes + num]  # [batch_size, h, w, num]
    preds_box = preds[..., num_classes + num:].reshape(batch_size, h, w, num, 4)

    preds_cls_idf = torch.broadcast_to(preds_cls_id[..., None], (batch_size, h, w, num)).float()
    preds_scores = preds_cls_conf[..., None] * preds_obj  # [batch_size, h, w, num]

    rg_y, rg_x = torch.meshgrid(
        torch.arange(h, dtype=preds.dtype, device=preds.device),
        torch.arange(w, dtype=preds.dtype, device=preds.device),
        indexing='ij')
    grid_hw = torch.stack((rg_x, rg_y), dim=-1)  # [h, w, 2]
    wh_tensor = torch.tensor([w, h], dtype=preds.dtype, device=preds.device)
    preds_cxcy = preds_box[..., :2].add(grid_hw[:, :, None, :]).div(wh_tensor)
    preds_wh = preds_box[..., 2:].exp().mul(wh_scales_tensor)
    preds_xyxy = cxcywh2xyxy2(preds_cxcy, preds_wh)
    return torch.cat((
        preds_xyxy.view(batch_size, h * w * num, 4),
        preds_scores.reshape(batch_size, h * w * num, 1),
        preds_cls_idf.reshape(batch_size, h * w * num, 1)
    ), dim=-1)


# src/detection_layer.c
def yolo_loss_ex(preds: torch.Tensor, target_list: Sequence[torch.Tensor],
                 num_classes: int, wh_scales: List[float],
                 obj_scale: float, no_obj_scale: float, cls_scale: float, coord_scale: float,
                 rescore: bool):
    batch_size, h, w, _ = preds.shape
    # wh_tensor = torch.tensor([w, h], dtype=torch.float32, device=preds.device)
    wh_scales_tensor = torch.tensor(wh_scales, dtype=torch.float32, device=preds.device).view(-1, 2)
    num: int = len(wh_scales_tensor)
    # preds: [batch_size, h, w, num_classes + num + num * 4]
    # targets: [batch_size, h, w, obj + cls + coords(4 = cx, cy, w, h)]

    sqrt_no_obj_scale = math.sqrt(no_obj_scale)
    sqrt_obj_scale = math.sqrt(obj_scale)

    preds = preds.reshape(batch_size * h * w, num_classes + num + num * 4)
    targets = _encode_yolov1_targets(target_list, h=h, w=w, num_classes=num_classes).view(-1, num_classes + 5)
    obj_mask = targets[:, 0] != 0  # [batch_size * h * w]

    # delta_obj: [batch_size * h * w, 2]
    delta_obj = sqrt_no_obj_scale * (-preds[:, num_classes:num_classes + num])

    preds_filobj = preds[obj_mask]  # [nobj, num_classes + num * 5]
    targets_filobj = targets[obj_mask]  # [nobj, num_classes + 5]

    loss_cls = (0.5 * cls_scale) * \
               ((targets_filobj[:, 1:1 + num_classes] - preds_filobj[:, :num_classes]).square().sum())

    preds_filobj_box = preds_filobj[:, num_classes + num:].reshape(-1, num, 4)
    with torch.no_grad():
        # _preds_filobj_xy = preds_filobj_box[..., 0:2]
        # _preds_filobj_wh = preds_filobj_box[..., 2:4].exp().mul(wh_scales_tensor)  # [nobj, num, 2]
        # _preds_filobj_xyxy = cxcywh2xyxy2(_preds_filobj_xy.div(wh_tensor), _preds_filobj_wh)
        # # preds_filobj_xyxy: [nobj, num, 4]
        # _targets_filobj_xyxy = cxcywh2xyxy2(
        #     targets_filobj[:, 1 + num_classes:3 + num_classes].div(wh_tensor),  # [nobj, 2]
        #     targets_filobj[:, 3 + num_classes:]  # [nobj, 2]
        # )  # [nobj, 4]
        # iou_mat = broadcast_compute_box_iou(_preds_filobj_xyxy, _targets_filobj_xyxy[:, None, :])  # [nobj, num]
        # iou_best_val, iou_best_n = iou_mat.max(-1)

        _targets_filobj_wh = targets_filobj[:, 3 + num_classes:]  # [nobj, 2]
        iou_mat = broadcast_compute_wh_iou(_targets_filobj_wh[:, None, :], wh_scales_tensor)  # [nobj, num]
        iou_best_val, iou_best_n = iou_mat.max(-1)

    _range = torch.arange(len(preds_filobj), dtype=torch.long, device=preds.device)
    preds_filobj_obj = preds_filobj[:, num_classes:num_classes + num]  # [nobj, num]
    if rescore:
        delta_obj[_range, iou_best_n] = \
            sqrt_obj_scale * (iou_best_val - preds_filobj_obj[_range, iou_best_n])
    else:
        delta_obj[_range, iou_best_n] = \
            sqrt_obj_scale * (1.0 - preds_filobj_obj[_range, iou_best_n])

    loss_obj = 0.5 * (delta_obj.square().sum())
    loss_box_cxcy = (0.5 * coord_scale) * \
                    ((targets_filobj[:, 1 + num_classes:3 + num_classes] -
                      preds_filobj_box[_range, iou_best_n, 0:2]).square().sum())
    wh_scales_best = wh_scales_tensor[iou_best_n]
    loss_box_wh = (0.5 * coord_scale) * \
                  ((targets_filobj[:, 3 + num_classes:].div(wh_scales_best).log() -
                    preds_filobj_box[_range, iou_best_n, 2:4]).square().sum())
    loss_box = loss_box_cxcy + loss_box_wh
    batch_scale = 1.0 / batch_size
    return batch_scale * loss_box, batch_scale * loss_obj, batch_scale * loss_cls
