import torch
from typing import Sequence, Tuple

eps = 1e-8


def cxcywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    # boxes: [N..., 4]
    # return : [N..., 4]
    cxcy = boxes[..., 0:2]
    half_wh = 0.5 * boxes[..., 2:4]
    return torch.cat((cxcy - half_wh, cxcy + half_wh), dim=-1)


def cxcywh2xyxy2(cxcy: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
    # cxcy: [N..., 2]
    # wh: [N..., 2]
    half_wh = wh * 0.5
    return torch.cat((cxcy - half_wh, cxcy + half_wh), dim=-1)


def cxcywhcls2xyxycls(target: torch.Tensor):
    # target: [n_target, 5 = (cx, cy, w, h, cls_id)]
    assert target.ndim == 2
    assert target.shape[-1] == 5
    cls_id = target[:, -1]
    cxcy = target[:, 0:2]
    half_wh = 0.5 * target[:, 2:4]
    return torch.cat((
        cxcy - half_wh, cxcy + half_wh, cls_id[:, None]
    ), dim=-1)


def broadcast_compute_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # boxes1 and boxes2: [Nb1..., 4], [Nb2..., 4], xyxy format
    # can boardcast to [N..., 4] (bitwise op)
    # return: [N...]
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4

    def compute_area(wh: torch.Tensor) -> torch.Tensor:
        return wh[..., 0] * wh[..., 1]

    boxes1_wh = boxes1[..., 2:4] - boxes1[..., 0:2]
    boxes2_wh = boxes2[..., 2:4] - boxes2[..., 0:2]
    overlap_xymin = torch.maximum(boxes1[..., 0:2], boxes2[..., 0:2])
    overlap_xymax = torch.minimum(boxes1[..., 2:4], boxes2[..., 2:4])
    overlap_wh = (overlap_xymax - overlap_xymin).clamp(0.)

    boxes1_area = compute_area(boxes1_wh)
    boxes2_area = compute_area(boxes2_wh)
    overlap_area = compute_area(overlap_wh)

    return overlap_area / (boxes1_area + boxes2_area - overlap_area + eps)


def broadcast_compute_wh_iou(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
    wh_overlap = torch.minimum(wh1, wh2)
    area1 = wh1[..., 0] * wh1[..., 1]
    area2 = wh2[..., 0] * wh2[..., 1]
    area_overlap = wh_overlap[..., 0] * wh_overlap[..., 1]
    return area_overlap / (area1 + area2 - area_overlap + eps)
