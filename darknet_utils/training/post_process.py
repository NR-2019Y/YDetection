import torch
from .utils import cxcywh2xyxy2
import torchvision


@torch.no_grad()
def yolov1_dets(preds: torch.Tensor, num: int, num_classes: int, use_sqrt: bool) -> torch.Tensor:
    batch_size, h, w, _ = preds.shape
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
    if use_sqrt:
        preds_wh = preds_box[..., 2:].square()
    else:
        preds_wh = preds_box[..., 2:]
    preds_xyxy = cxcywh2xyxy2(preds_cxcy, preds_wh)
    return torch.cat((
        preds_xyxy.view(batch_size, h * w * num, 4),
        preds_scores.reshape(batch_size, h * w * num, 1),
        preds_cls_idf.reshape(batch_size, h * w * num, 1)
    ), dim=-1)


@torch.no_grad()
def yolov3_dets(preds: torch.Tensor, current_anchors_scale: torch.Tensor) -> torch.Tensor:
    # return: [batch_size, num_dets, 6=(x1, y1, x2, y2, conf, cls)]
    na = len(current_anchors_scale)
    # preds: [batch_size, na*(5+nc), h, w]
    batch_size, _, h, w = preds.size()
    preds = preds.permute(0, 2, 3, 1).view(batch_size, h, w, na, -1)

    wh_tensor = torch.tensor([w, h], dtype=preds.dtype, device=preds.device)
    rg_y, rg_x = torch.meshgrid(
        torch.arange(h, dtype=preds.dtype, device=preds.device),
        torch.arange(w, dtype=preds.dtype, device=preds.device),
        indexing='ij')
    grid_hw = torch.stack((rg_x, rg_y), dim=-1)  # [h, w, 2]
    preds_cxcy = preds[..., 0:2].sigmoid().add(grid_hw[:, :, None, :]).div(wh_tensor)
    preds_wh = preds[..., 2:4].exp().mul(current_anchors_scale)
    preds_cls_conf, preds_cls_ids = preds[..., 5:].sigmoid().max(-1)
    preds_scores = preds[..., 4].sigmoid() * preds_cls_conf
    dets = torch.cat((cxcywh2xyxy2(preds_cxcy.reshape(batch_size, -1, 2), preds_wh.reshape(batch_size, -1, 2)),
                      preds_scores.reshape(batch_size, -1, 1),
                      preds_cls_ids.float().reshape(batch_size, -1, 1)), dim=-1)
    return dets  # [batch_size, num_dets, 6]


@torch.no_grad()
def nms_dets(dets: torch.Tensor, conf_threshold: float = 0.001, iou_threshold: float = 0.6) -> torch.Tensor:
    # dets: [num_dets, 6]
    assert dets.dim() == 2
    scores = dets[:, 4]
    mask = scores > conf_threshold
    dets = dets[mask]
    keep = torchvision.ops.batched_nms(dets[:, :4], dets[:, 4], dets[:, 5].long(), iou_threshold)
    return dets[keep]
