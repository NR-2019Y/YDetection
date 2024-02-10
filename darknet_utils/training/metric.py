from typing import Sequence
import numpy as np


def computeIOU(pred_xyxy: np.ndarray, target_xyxy: np.ndarray, *, eps=1e-7) -> np.ndarray:
    # pred_xyxy: [N or 1, 4] target_xyxy: [N or 1, 4]
    pred_wh = pred_xyxy[:, 2:4] - pred_xyxy[:, 0:2]
    target_wh = target_xyxy[:, 2:4] - target_xyxy[:, 0:2]
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    target_area = target_wh[:, 0] * target_wh[:, 1]
    overlap_xymin = np.maximum(pred_xyxy[:, 0:2], target_xyxy[:, 0:2])
    overlap_xymax = np.maximum(pred_xyxy[:, 2:4], target_xyxy[:, 2:4])
    overlap_wh = np.maximum(overlap_xymax - overlap_xymin, 0)
    overlap_area = overlap_wh[:, 0] * overlap_wh[:, 1]
    iou = overlap_area / (pred_area + target_area - overlap_area + eps)
    return iou


def calc_ap(recall: np.ndarray, precison: np.ndarray):
    mrec = np.concatenate([[0.], recall, [1.]])
    mpre = np.concatenate([[1.], precison, [0.]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.arange(len(mrec) - 1)
    # i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# # reference: pycocotools
# def calc_ap(recall: np.ndarray, precision: np.ndarray, recThrs: Optional[np.ndarray] = None):
#     assert len(recall) == len(precision)
#     if not len(recall):
#         return 0
#     if recThrs is None:
#         recThrs = np.linspace(.0, 1.00, round(
#             (1.00 - .0) / .01) + 1, endpoint=True)
#         # recThrs = np.linspace(.0, 1.00, round(
#         #     (1.00 - .0) / .1) + 1, endpoint=True)

#     R = recall[-1]
#     mrec = np.concatenate([recall, [1.]])
#     mpre = np.concatenate([precision, [0.]])
#     mpre = np.maximum.accumulate(mpre[::-1])[::-1]
#     # print(mpre.tolist())
#     inds = np.searchsorted(mrec, recThrs, side='left')
#     # print(inds.tolist())
#     P = mpre[inds]
#     return np.mean(P)


# 参考: Object-Detection-Metrics
# https://github.com/rafaelpadilla/Object-Detection-Metrics
def evaluate_np(pred_bboxes: Sequence[np.ndarray],
                target_bboxes: Sequence[np.ndarray],
                num_classes: int,
                iou_threshold: float,
                topk: int = 100) -> np.ndarray:
    """
    pred_bboxes: List of [N, 6=(x1, y1, x2, y2, conf, cls_id)]
    target_bboxes: List of [N, 6=(x1, y1, x2, y2, cls_id)]
    """
    # print([len(x) for x in pred_bboxes])
    num_samples = len(pred_bboxes)
    assert len(target_bboxes) == num_samples
    pred_bboxes_with_file_index = np.concatenate([
        np.concatenate((bbox, np.full((len(bbox), 1), i)), axis=1) for i, bbox in enumerate(pred_bboxes)
    ], axis=0)

    target_bboxes_by_classes = [
        [bbox[bbox[:, -1] == c, :4] for c in range(num_classes)] for bbox in target_bboxes
    ]
    target_bboxes_visited = [
        [np.zeros(len(cbbox), dtype=bool) for cbbox in bbox] for bbox in target_bboxes_by_classes
    ]

    all_ap = []
    for c in range(num_classes):
        npos = sum(len(bbox[c]) for bbox in target_bboxes_by_classes)
        c_pred_bboxes = pred_bboxes_with_file_index[pred_bboxes_with_file_index[:, 5] == c]
        if len(c_pred_bboxes) <= topk:
            bbox_order = np.argsort(-c_pred_bboxes[:, 4])
        else:
            neg_scores = -c_pred_bboxes[:, 4]
            bbox_order = np.argpartition(neg_scores, topk - 1)[:topk]
            bbox_order = bbox_order[np.argsort(neg_scores[bbox_order])]
        c_pred_bboxes = c_pred_bboxes[bbox_order]
        TP = np.zeros(len(c_pred_bboxes), dtype=np.int32)
        for i, pred in enumerate(c_pred_bboxes):
            file_id = int(pred[-1])
            t_bboxes = target_bboxes_by_classes[file_id][c]
            if not len(t_bboxes):
                continue
            iou = computeIOU(pred[None, :4], t_bboxes)
            iou_maxindex = iou.argmax()
            iou_maxval = iou[iou_maxindex]
            if iou_maxval >= iou_threshold and (not target_bboxes_visited[file_id][c][iou_maxindex]):
                TP[i] = 1
                target_bboxes_visited[file_id][c][iou_maxindex] = True
        FP = 1 - TP
        tp_cumsum = np.cumsum(TP)
        fp_cumsum = np.cumsum(FP)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / npos
        ap = calc_ap(recall, precision)
        all_ap.append(ap)
    return np.asarray(all_ap, dtype=np.float32)
