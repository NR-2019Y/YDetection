import math

import torch
from typing import Sequence, Tuple, Dict
from .utils import broadcast_compute_box_iou, broadcast_compute_wh_iou
from .utils import cxcywh2xyxy, cxcywh2xyxy2


# src/data.c
# fill_truth_region
def _encode_yolov1_targets(target_list: Sequence[torch.Tensor], h: int, w: int, num_classes: int) -> torch.Tensor:
    device = target_list[0].device
    batch_size: int = len(target_list)
    repeats = torch.tensor([len(t) for t in target_list], dtype=torch.long, device=device)
    bid = torch.arange(batch_size, device=device).repeat_interleave(repeats, 0)  # [n_targets]
    btargets = torch.cat(target_list, dim=0)  # type: ignore
    btargets_x = btargets[:, 0].mul(w)
    btargets_y = btargets[:, 1].mul(h)
    btargets_xidx = btargets_x.long().clamp_(0, w - 1)
    btargets_yidx = btargets_y.long().clamp_(0, h - 1)
    btargets_xy_to_center = torch.stack((btargets_x, btargets_y), dim=-1) % 1.0
    targets = torch.zeros((batch_size, h, w, 5 + num_classes), dtype=torch.float32, device=device)
    # targets: [batch_size, h, w, 5 + num_classes] 5+nc: obj + cls + cx, cy, w, h
    targets[bid, btargets_yidx, btargets_xidx, 0] = 1.0
    targets[bid, btargets_yidx, btargets_xidx, 1 + btargets[..., 4].long()] = 1.0
    targets[bid, btargets_yidx, btargets_xidx, 1 + num_classes:3 + num_classes] = btargets_xy_to_center
    targets[bid, btargets_yidx, btargets_xidx, 3 + num_classes:] = btargets[:, 2:4]
    return targets


def _encode_targets(target_list: Sequence[torch.Tensor]) -> torch.Tensor:
    # target_list: list of [ni, 5 = (cx, cy, w, h, clsid)]
    num_targets_batch = max(max(len(x) for x in target_list), 1)
    batch_size = len(target_list)
    targets = torch.zeros((batch_size, num_targets_batch, 5), dtype=torch.float32, device=target_list[0].device)
    for i, target in enumerate(target_list):
        targets[i, :len(target)] = target
    return targets


# src/detection_layer.c
def yolov1_loss(preds: torch.Tensor, target_list: Sequence[torch.Tensor],
                num: int, num_classes: int,
                obj_scale: float, no_obj_scale: float, cls_scale: float, coord_scale: float,
                use_sqrt: bool, rescore: bool):
    batch_size, h, w, _ = preds.shape
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

    wh_tensor = torch.tensor([w, h], dtype=torch.float32, device=preds.device)
    preds_filobj_box = preds_filobj[:, num_classes + num:].reshape(-1, num, 4)
    if use_sqrt:
        preds_filobj_wh = preds_filobj_box[..., 2:4].square()
    else:
        preds_filobj_wh = preds_filobj_box[..., 2:4]
    preds_filobj_xyxy = cxcywh2xyxy2(preds_filobj_box[..., 0:2].div(wh_tensor), preds_filobj_wh)
    # preds_filobj_xyxy: [nobj, num, 4]
    targets_filobj_xyxy = cxcywh2xyxy2(
        targets_filobj[:, 1 + num_classes:3 + num_classes].div(wh_tensor),  # [nobj, 2]
        targets_filobj[:, 3 + num_classes:]  # [nobj, 2]
    )  # [nobj, 4]
    with torch.no_grad():
        iou_mat = broadcast_compute_box_iou(preds_filobj_xyxy, targets_filobj_xyxy[:, None, :])  # [nobj, num]
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
    if use_sqrt:
        loss_box_cxcy = (0.5 * coord_scale) * \
                        ((targets_filobj[:, 1 + num_classes:3 + num_classes] -
                          preds_filobj_box[_range, iou_best_n, 0:2]).square().sum())
        loss_box_wh = (0.5 * coord_scale) * \
                      ((targets_filobj[:, 3 + num_classes:].sqrt() -
                        preds_filobj_box[_range, iou_best_n, 2:4]).square().sum())
        loss_box = loss_box_cxcy + loss_box_wh
    else:
        loss_box = (0.5 * coord_scale) * \
                   ((targets_filobj[:, 1 + num_classes:] - preds_filobj_box[_range, iou_best_n]).square().sum())
    batch_scale = 1.0 / batch_size
    return batch_scale * loss_box, batch_scale * loss_obj, batch_scale * loss_cls


# # src/detection_layer.c
# def yolov1_loss(preds: torch.Tensor, target_list: Sequence[torch.Tensor],
#                 num: int, num_classes: int,
#                 obj_scale: float, no_obj_scale: float, cls_scale: float, coord_scale: float,
#                 use_sqrt: bool, rescore: bool):
#     batch_size, h, w, _ = preds.shape
#     # preds: [batch_size, h, w, num_classes + num + num * 4]
#     # targets: [batch_size, h, w, obj + cls + coords(4 = cx, cy, w, h)]
#
#     sqrt_no_obj_scale = math.sqrt(no_obj_scale)
#     sqrt_obj_scale = math.sqrt(obj_scale)
#
#     preds = preds.reshape(batch_size * h * w, num_classes + num + num * 4)
#     targets = _encode_yolov1_targets(target_list, h=h, w=w, num_classes=num_classes).view(-1, num_classes + 5)
#     obj_mask = targets[:, 0] != 0  # [batch_size * h * w]
#
#     preds_noobj = preds[~obj_mask]
#     loss_noobj = (0.5 * no_obj_scale) * (preds_noobj[:, num_classes:num_classes + num].square().sum())
#
#     preds_filobj = preds[obj_mask]  # [nobj, num_classes + num * 5]
#     targets_filobj = targets[obj_mask]  # [nobj, num_classes + 5]
#
#     loss_cls = (0.5 * cls_scale) * \
#                ((targets_filobj[:, 1:1 + num_classes] - preds_filobj[:, :num_classes]).square().sum())
#
#     wh_tensor = torch.tensor([w, h], dtype=torch.float32, device=preds.device)
#     preds_filobj_box = preds_filobj[:, num_classes + num:].reshape(-1, num, 4)
#     if use_sqrt:
#         preds_filobj_wh = preds_filobj_box[..., 2:4].square()
#     else:
#         preds_filobj_wh = preds_filobj_box[..., 2:4]
#     preds_filobj_xyxy = cxcywh2xyxy2(preds_filobj_box[..., 0:2].div(wh_tensor), preds_filobj_wh)
#     # preds_filobj_xyxy: [nobj, num, 4]
#     targets_filobj_xyxy = cxcywh2xyxy2(
#         targets_filobj[:, 1 + num_classes:3 + num_classes].div(wh_tensor),  # [nobj, 2]
#         targets_filobj[:, 3 + num_classes:]  # [nobj, 2]
#     )  # [nobj, 4]
#     with torch.no_grad():
#         iou_mat = broadcast_compute_box_iou(preds_filobj_xyxy, targets_filobj_xyxy[:, None, :])  # [nobj, num]
#         iou_best_val, iou_best_n = iou_mat.max(-1)
#
#     _range = torch.arange(len(preds_filobj), dtype=torch.long, device=preds.device)
#     preds_filobj_obj = preds_filobj[:, num_classes:num_classes + num]  # [nobj, num]
#
#     if rescore:
#         loss_hasobj = (0.5 * obj_scale) * \
#                       ((iou_best_val - preds_filobj_obj[_range, iou_best_n]).square().sum())
#     else:
#         loss_hasobj = (0.5 * obj_scale) * \
#                       ((1.0 - preds_filobj_obj[_range, iou_best_n]).square().sum())
#
#     loss_obj = loss_noobj + loss_hasobj
#     if use_sqrt:
#         loss_box_cxcy = (0.5 * coord_scale) * \
#                         ((targets_filobj[:, 1 + num_classes:3 + num_classes] -
#                           preds_filobj_box[_range, iou_best_n, 0:2]).square().sum())
#         loss_box_wh = (0.5 * coord_scale) * \
#                       ((targets_filobj[:, 3 + num_classes:].sqrt() -
#                         preds_filobj_box[_range, iou_best_n, 2:4]).square().sum())
#         loss_box = loss_box_cxcy + loss_box_wh
#     else:
#         loss_box = (0.5 * coord_scale) * \
#                    ((targets_filobj[:, 1 + num_classes:] - preds_filobj_box[_range, iou_best_n]).square().sum())
#     batch_scale = 1.0 / batch_size
#     return batch_scale * loss_box, batch_scale * loss_obj, batch_scale * loss_cls


@torch.no_grad()
def _assign_targets_to_anchors(anchors_scale: torch.Tensor, target_list: Sequence[torch.Tensor],
                               h: int, w: int, valid_index_start: int, valid_index_end: int):
    # anchors: [tna, 2]
    device = anchors_scale.device
    batch_size: int = len(target_list)
    repeats = torch.tensor([len(t) for t in target_list], dtype=torch.long, device=device)
    bid = torch.arange(batch_size, device=device).repeat_interleave(repeats, 0)  # [n_targets]
    btargets = torch.cat(target_list, dim=0)  # type: ignore
    # btargets: [n_targets, 5 = (cx, cy, w, h, clsid)]

    iou_scores = broadcast_compute_wh_iou(btargets[:, None, 2:4], anchors_scale)  # [n_targets, tna]
    iou_scores_max_index = iou_scores.argmax(-1)  # [n_targets]
    mask = (iou_scores_max_index >= valid_index_start) & (iou_scores_max_index < valid_index_end)

    bid_fil = bid[mask]
    btargets_fil = btargets[mask]
    btargets_xidx_fil = (btargets_fil[:, 0] * w).long().clamp_(0, w - 1)
    btargets_yidx_fil = (btargets_fil[:, 1] * h).long().clamp_(0, h - 1)
    iou_scores_max_index_fil = iou_scores_max_index[mask].sub(valid_index_start)
    return (bid_fil, btargets_yidx_fil, btargets_xidx_fil, iou_scores_max_index_fil), btargets_fil


# src/yolo_layer.c
def yolov3_dets_and_loss(preds: torch.Tensor, target_list: Sequence[torch.Tensor],
                         anchors_scale: torch.Tensor, current_anchors_scale: torch.Tensor,
                         ignore_thresh: float, truth_thresh: float,
                         valid_index_start: int, valid_index_end: int) \
        -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    assert ignore_thresh <= truth_thresh
    na = len(current_anchors_scale)
    # preds: [batch_size, na*(5+nc), h, w]
    batch_size, _, h, w = preds.size()
    wh_tensor = torch.tensor([w, h], dtype=preds.dtype, device=preds.device)
    preds = preds.permute(0, 2, 3, 1).view(batch_size, h, w, na, -1)
    nc = preds.size(-1) - 5

    rg_y, rg_x = torch.meshgrid(
        torch.arange(h, dtype=preds.dtype, device=preds.device),
        torch.arange(w, dtype=preds.dtype, device=preds.device),
        indexing='ij')
    grid_hw = torch.stack((rg_x, rg_y), dim=-1)  # [h, w, 2]
    preds_cxcy = preds[..., 0:2].sigmoid().add(grid_hw[:, :, None, :]).div(wh_tensor)
    preds_obj = preds[..., 4].sigmoid()
    preds_cls = preds[..., 5:].sigmoid()
    with torch.no_grad():
        preds_xyxy = cxcywh2xyxy2(preds_cxcy,
                                  preds[..., 2:4].exp().mul(current_anchors_scale))  # [batch_size, h, w, na, 4]
        targets = _encode_targets(target_list)  # [b, n_max_gt, 5]
        targets_xyxy = cxcywh2xyxy(targets[..., :4])  # [b, n_max_gt, 4]
        iou_scores = broadcast_compute_box_iou(preds_xyxy[:, :, :, :, None, :],
                                               targets_xyxy[:, None, None, None, :, :])
        # iou_scores: [batch_size, h, w, na, n_max_gt, 4]
        iou_scores_max_val, iou_scores_max_idx = iou_scores.max(-1)  # [batch_size, h, w, na]
        noobj_indices = torch.where(iou_scores_max_val <= ignore_thresh)
        truth_indices = torch.where(iou_scores_max_val > truth_thresh)
        targets_match = torch.take_along_dim(
            targets[:, None, None, None, :, :],  # [batch_size, 1, 1, 1, n_max_gt, 5]
            iou_scores_max_idx[..., None, None],  # [batch_size, h, w, na, 1, 1]
            dim=-2
        ).squeeze(-2)  # [batch_size, h, w, na, 5]
        targets_match_tw_th = targets_match[..., 2:4].div(current_anchors_scale).log()  # [batch_size, h, w, na, 2]
        targets_match_fil_truth = targets_match[truth_indices]  # [n_fil_truth, 2]

        preds_cls_conf, preds_cls_ids = preds_cls.max(-1)
        preds_scores = preds_obj * preds_cls_conf
        dets = torch.cat((preds_xyxy.view(batch_size, -1, 4),
                          preds_scores.reshape(batch_size, -1, 1),
                          preds_cls_ids.float().view(batch_size, -1, 1)), dim=-1)
        # dets: [batch_size, num_dets, 6=(x1, y1, x2, y2, conf, cls)]

    delta_obj = torch.zeros((batch_size, h, w, na), dtype=preds.dtype, device=preds.device)
    delta_obj[noobj_indices] = -preds_obj[noobj_indices]
    delta_obj[truth_indices] = 1.0 - preds_obj[truth_indices]

    delta_box = torch.zeros((batch_size, h, w, na, 4), dtype=preds.dtype, device=preds.device)
    delta_box_fil_truth_scale = 2.0 - targets_match_fil_truth[:, 2] * targets_match_fil_truth[:, 3]
    delta_box[(*truth_indices, slice(0, 2))] = (targets_match_fil_truth[..., :2] - preds_cxcy[truth_indices]). \
        mul_(wh_tensor).mul_(delta_box_fil_truth_scale[:, None])  # [n_fil_truth, 2]
    delta_box[(*truth_indices, slice(2, 4))] = \
        (targets_match_tw_th[truth_indices] - preds[(*truth_indices, slice(2, 4))]) \
            .mul_(delta_box_fil_truth_scale[:, None])  # [n_fil_truth, 2]

    (bid, btargets_yidx, btargets_xidx, iou_scores_max_index), btargets = \
        _assign_targets_to_anchors(anchors_scale, target_list, h, w, valid_index_start, valid_index_end)
    delta_obj[bid, btargets_yidx, btargets_xidx, iou_scores_max_index] = \
        1.0 - preds_obj[bid, btargets_yidx, btargets_xidx, iou_scores_max_index]

    delta_box_target_assign_scale = 2.0 - btargets[..., 2] * btargets[..., 3]
    delta_box[bid, btargets_yidx, btargets_xidx, iou_scores_max_index, :2] = \
        (btargets[:, :2] - preds_cxcy[bid, btargets_yidx, btargets_xidx, iou_scores_max_index]). \
            mul_(wh_tensor).mul_(delta_box_target_assign_scale[:, None])
    btargets_tw_th = btargets[..., 2:4]. \
        div(current_anchors_scale[iou_scores_max_index]).log()
    delta_box[bid, btargets_yidx, btargets_xidx, iou_scores_max_index, 2:] = \
        (btargets_tw_th - preds[bid, btargets_yidx, btargets_xidx, iou_scores_max_index, 2:4]) \
            .mul_(delta_box_target_assign_scale[:, None])

    delta_cls = torch.zeros((batch_size, h, w, na, nc), dtype=preds.dtype, device=preds.device)
    delta_cls[(*truth_indices, targets_match_fil_truth[..., 4].long())] = 1.0
    delta_cls[bid, btargets_yidx, btargets_xidx, iou_scores_max_index, btargets[..., 4].long()] = 1.0
    delta_cls[truth_indices] -= preds_cls[truth_indices]
    delta_cls[bid, btargets_yidx, btargets_xidx, iou_scores_max_index] \
        -= preds_cls[bid, btargets_yidx, btargets_xidx, iou_scores_max_index]

    scale = 0.5 / batch_size
    loss_box = delta_box.square().sum().mul(scale)
    loss_obj = delta_obj.square().sum().mul(scale)
    loss_cls = delta_cls.square().sum().mul(scale)
    return dets, (loss_box, loss_obj, loss_cls)


class YoloV3LossMetric:
    def __init__(self):
        self.metric_dict = {
            "loss_box": 0.0,
            "loss_obj": 0.0,
            "loss_cls": 0.0
        }
        self.total: int = 0

    def reset(self):
        for k, v in self.metric_dict.items():
            self.metric_dict[k] = 0.0
        self.total = 0

    def update(self, losses: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_size: int):
        for k, l in zip(self.metric_dict, losses):
            self.metric_dict[k] += l.detach()
        self.total += batch_size

    def get_metrics(self) -> Dict[str, float]:
        return {k: float(v) / self.total for k, v in self.metric_dict.items()}


YoloV1LossMetric = YoloV3LossMetric
