import os
import os.path as osp
import shutil
import sys
import importlib
import argparse
import torch
from typing import Tuple, List
from torch.utils.data import DataLoader
from darknet_utils.training.utils import cxcywhcls2xyxycls
from darknet_utils.training.metric import evaluate_np
from darknet_utils.training.loss import YoloV1LossMetric as LossMetric
from darknet_utils.training.post_process import nms_dets
from yolo_resnet18_ex2 import YoloResnet18, yolo_dets_ex, yolo_loss_ex
from darknet_utils.training.detection_dataset_aug import (
    Compose, YoloDataset, ImageTransform, RandomFlipLR, RandomResizePad, collate_fn)
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm


def get_module(module_path: str):
    assert module_path.endswith(".py")
    module_dir = osp.dirname(osp.abspath(module_path))
    module_name, _ = osp.splitext(osp.basename(module_path))
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module


def get_module_from_args():
    xparser = argparse.ArgumentParser()
    xparser.add_argument("--cfg", type=str, default="cfg/yolo_resnet18_ex1_cfg.py")
    args = xparser.parse_args()
    return get_module(args.cfg)


def make_dataloader(cfg) -> Tuple[DataLoader, DataLoader]:
    input_height = cfg.input_height
    input_width = cfg.input_width
    mean = getattr(cfg, "mean", None)
    std = getattr(cfg, "std", None)
    # device: torch.device = torch.device(cfg.device)  # type: ignore
    cache_dir = getattr(cfg, "cache_dir", None)
    val_cache_dir = getattr(cfg, "val_cache_dir", None)
    image_device = torch.device(getattr(cfg, "image_device", 'cpu'))
    val_image_device = torch.device(getattr(cfg, "val_image_device", image_device))
    train_num_expand = getattr(cfg, "train_num_expand", 1)
    train_dataset = YoloDataset(image_dir=cfg.image_dir["train"],
                                label_dir=cfg.label_dir["train"],
                                input_height=input_height,
                                input_width=input_width,
                                mean=mean, std=std,
                                image_only_transform=ImageTransform(),
                                image_label_transform=Compose([
                                    RandomFlipLR(prob=0.25),
                                    RandomResizePad(ratio_min=0.750, ratio_max=1.333,
                                                    input_height=input_height,
                                                    input_width=input_width)]),
                                image_device=image_device,
                                cache_dir=cache_dir,
                                num_expand=train_num_expand)
    val_dataset = YoloDataset(image_dir=cfg.image_dir["val"],
                              label_dir=cfg.label_dir["val"],
                              input_height=input_height,
                              input_width=input_width,
                              mean=mean, std=std,
                              image_only_transform=None,
                              image_label_transform=None,
                              image_device=val_image_device,
                              cache_dir=val_cache_dir)
    batch_size = cfg.batch_size
    val_batch_size = getattr(cfg, "val_batch_size", batch_size)
    num_workers = cfg.num_workers
    val_num_workers = getattr(cfg, "val_num_workers", num_workers)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                  num_workers=num_workers, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=collate_fn,
                                num_workers=val_num_workers, shuffle=False, drop_last=False)
    # pin_memory=True, pin_memory_device=str(device))
    return train_dataloader, val_dataloader


# https://github.com/yjh0410/new-YOLOv1_PyTorch/blob/master/train.py
class LRUpdater:
    def __init__(self, optim: torch.optim.Optimizer,
                 base_lr: float, warm_up_epochs: int, lr_steps: List[int],
                 num_batch_each_epoch: int):
        self.optim = optim
        self.base_lr = base_lr
        self.warm_up_epochs = warm_up_epochs
        self.num_batch_each_epoch = num_batch_each_epoch
        self.lr_steps = lr_steps
        self._last_lr = base_lr

    def set_lr(self, lr: float):
        for param_group in self.optim.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group["lr"]

    def update(self, epoch: int, batch_index: int):
        if epoch < self.warm_up_epochs:
            lr = self.base_lr * pow((batch_index + epoch * self.num_batch_each_epoch) /
                                    (self.warm_up_epochs * self.num_batch_each_epoch), 4)
            self.set_lr(lr)
        elif epoch == self.warm_up_epochs and batch_index == 0:
            self.set_lr(self.base_lr)
        elif batch_index == 0 and epoch in self.lr_steps:
            self._last_lr *= 0.1
            self.set_lr(self._last_lr)


# https://github.com/ultralytics/yolov3 (utils/torch_utils.py)
def get_sgd_optim(model: torch.nn.Module, lr: float, weight_decay: float):
    g = [], [], []
    bn_cls = tuple(v for k, v in torch.nn.__dict__.items() if 'Norm' in k)
    for m in model.modules():
        for name, param in m.named_parameters(recurse=False):
            if name == "bias":
                g[2].append(param)
            elif name == "weight" and isinstance(m, bn_cls):
                g[1].append(param)
            else:
                g[0].append(param)
    params = [
        {"params": g[0], "weight_decay": weight_decay},
        {"params": g[1], "weight_decay": 0.0},
        {"params": g[2]}
    ]
    optim = torch.optim.SGD(params=params, lr=lr, momentum=0.9, nesterov=True)
    return optim


def main():
    # print(torch.multiprocessing.get_all_sharing_strategies())
    torch.multiprocessing.set_start_method('spawn', force=True)
    cfg = get_module_from_args()
    train_dataloader, val_dataloader = make_dataloader(cfg)  # type: ignore

    NCOLS = 130
    output_dir: str = cfg.output_dir  # type: ignore
    if osp.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    device: torch.device = torch.device(cfg.device)  # type: ignore
    epochs: int = cfg.epochs  # type: ignore

    save_freq: int = cfg.save_freq  # type: ignore
    class_names: List[str] = cfg.class_names  # type: ignore
    num_classes: int = len(class_names)
    wh_scales: List[float] = cfg.wh_scales  # type: ignore
    num: int = len(wh_scales) // 2

    loss_kw = dict(
        num_classes=num_classes,
        wh_scales=wh_scales,  # type: ignore
        obj_scale=cfg.obj_scale,  # type: ignore
        no_obj_scale=cfg.no_obj_scale,  # type: ignore
        cls_scale=cfg.cls_scale,  # type: ignore
        coord_scale=cfg.coord_scale,  # type: ignore
        rescore=cfg.rescore  # type: ignore
    )
    dets_kw = dict(num_classes=num_classes, wh_scales=wh_scales)  # type: ignore

    yolo_model = YoloResnet18(num=num, num_classes=num_classes).to(device)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("#### %(asctime)s ####\n%(message)s")
    log_file_handler = logging.FileHandler(osp.join(output_dir, "output.log"))
    log_file_handler.setFormatter(fmt)
    log_console_handler = logging.StreamHandler(stream=sys.stdout)
    log_console_handler.setFormatter(fmt)
    logger.addHandler(log_file_handler)
    logger.addHandler(log_console_handler)
    writer = SummaryWriter(log_dir=output_dir)

    base_lr: float = cfg.base_lr  # type: ignore
    lr_steps: List[int] = cfg.lr_steps  # type: ignore
    warm_up_epochs: int = cfg.warm_up_epochs  # type: ignore
    weight_decay: float = cfg.weight_decay  # type: ignore
    # optim = torch.optim.Adam(yolo_model.parameters(), lr=1e-4)
    optim = get_sgd_optim(yolo_model, lr=1e-5, weight_decay=weight_decay)
    lr_updater = LRUpdater(optim, base_lr=base_lr, warm_up_epochs=warm_up_epochs,
                           lr_steps=lr_steps, num_batch_each_epoch=len(train_dataloader))

    train_metric = LossMetric()
    val_metric = LossMetric()

    for epoch in range(epochs):
        yolo_model.train()
        train_metric.reset()
        train_pred_bboxes, train_target_bboxes = [], []
        for i, (imgs, target_list) in enumerate(tqdm(train_dataloader, ncols=NCOLS)):
            lr_updater.update(epoch, i)
            optim.zero_grad()
            imgs = imgs.to(device)
            target_list_device = tuple(t.to(device) for t in target_list)
            preds = yolo_model(imgs)
            losses = yolo_loss_ex(preds, target_list_device, **loss_kw)
            (losses[0] + losses[1] + losses[2]).backward()
            optim.step()
            train_metric.update(losses, len(target_list))
            batch_dets = yolo_dets_ex(preds.detach(), **dets_kw)
            for dets, target in zip(batch_dets.detach(), target_list):
                dets = nms_dets(dets)
                train_pred_bboxes.append(dets.cpu().numpy())
                train_target_bboxes.append(cxcywhcls2xyxycls(target).numpy())
        train_mAP = float(evaluate_np(train_pred_bboxes, train_target_bboxes,
                                      num_classes=num_classes, iou_threshold=0.5).mean())
        writer.add_scalar("train/mAP", train_mAP, global_step=epoch)
        train_metric_dic = train_metric.get_metrics()
        for k, v in train_metric_dic.items():
            writer.add_scalar(f"train/{k}", v, global_step=epoch)
        train_msg = f"epoch:{epoch}  [TRAIN]  mAP:{train_mAP:.4f}  " + \
                    str.join("  ", (f"{k}:{v:.4f}" for k, v in train_metric_dic.items()))
        logger.info(train_msg)

        yolo_model.eval()
        val_metric.reset()
        val_pred_bboxes, val_target_bboxes = [], []
        with torch.no_grad():
            for imgs, target_list in tqdm(val_dataloader, ncols=NCOLS):
                imgs = imgs.to(device)
                target_list_device = tuple(t.to(device) for t in target_list)
                preds = yolo_model(imgs)
                losses = yolo_loss_ex(preds, target_list_device, **loss_kw)
                batch_dets = yolo_dets_ex(preds.detach(), **dets_kw)
                val_metric.update(losses, len(target_list))
                for dets, target in zip(batch_dets.detach(), target_list):
                    dets = nms_dets(dets)
                    val_pred_bboxes.append(dets.cpu().numpy())
                    val_target_bboxes.append(cxcywhcls2xyxycls(target).numpy())
        val_mAP = float(evaluate_np(val_pred_bboxes, val_target_bboxes,
                                    num_classes=num_classes, iou_threshold=0.5).mean())
        writer.add_scalar("val/mAP", val_mAP, global_step=epoch)
        val_metric_dic = val_metric.get_metrics()
        for k, v in val_metric_dic.items():
            writer.add_scalar(f"val/{k}", v, global_step=epoch)
        val_msg = f"epoch:{epoch}  [ VAL ]  mAP:{val_mAP:.4f}  " + \
                  str.join("  ", (f"{k}:{v:.4f}" for k, v in val_metric_dic.items()))
        logger.info(val_msg)

        if (epoch + 1) % save_freq == 0 or (epoch + 1 == epochs):
            torch.save({
                "model": yolo_model.state_dict(),
                "optim": optim.state_dict()
            }, osp.join(output_dir, f"{osp.splitext(osp.basename(__file__))[0]}-{epoch}.pt"))


if __name__ == '__main__':
    main()
