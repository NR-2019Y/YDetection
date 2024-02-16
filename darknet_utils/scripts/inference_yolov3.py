import os.path as osp
import sys
import importlib
import argparse
import cv2
import torch
import glob
from typing import List
from darknet_utils.training.post_process import nms_dets
from darknet_utils.training.build_model import YoloV3Model
import numpy as np


def get_module(module_path: str):
    assert module_path.endswith(".py")
    module_dir = osp.dirname(osp.abspath(module_path))
    module_name, _ = osp.splitext(osp.basename(module_path))
    sys.path.insert(0, module_dir)
    module = importlib.import_module(module_name)
    sys.path.pop(0)
    return module


def parse_args():
    xparser = argparse.ArgumentParser()
    xparser.add_argument("--cfg", type=str, default='cfg/yolov3_cfg.py')
    xparser.add_argument("--ckpt", type=str)
    xparser.add_argument("--input-dir", type=str)
    return xparser.parse_args()


def preprocess_image(img: np.ndarray, input_h: int, input_w: int):
    img_h, img_w = img.shape[:2]
    ratio = min(input_h / img_h, input_w / img_w)
    new_h = round(img_h * ratio)
    new_w = round(img_w * ratio)
    pad_h = input_h - new_h
    pad_w = input_w - new_w
    pad_top, pad_left = pad_h // 2, pad_w // 2
    img = img[..., ::-1].astype(np.float32) / 255.
    img_resize = cv2.resize(img, (new_w, new_h))
    img_resize_pad = np.pad(img_resize,
                            [[pad_top, pad_h - pad_top], [pad_left, pad_w - pad_left], [0, 0]],
                            mode='constant', constant_values=0.)
    img_blob = np.ascontiguousarray(np.transpose(img_resize_pad, (2, 0, 1))[None])
    img_tensor = torch.from_numpy(img_blob)
    return img_tensor, ratio, (pad_left, pad_top)


def main():
    args = parse_args()
    cfg = get_module(args.cfg)

    num_anchors: int = cfg.num_anchors  # type: ignore
    class_names: List[str] = cfg.class_names  # type: ignore
    num_classes: int = len(class_names)

    cfg_file: str = cfg.cfg_file  # type: ignore
    yolo_model = YoloV3Model(cfg_file=cfg_file, weight_file=None,
                             num_classes=num_classes, na=num_anchors)
    yolo_model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    yolo_model.eval()
    yolo_model.set_inference_mode()

    input_h = yolo_model.input_height
    input_w = yolo_model.input_width
    image_paths = glob.glob(osp.join(args.input_dir, "*.jpg"))

    cv2.namedWindow("W", cv2.WINDOW_NORMAL)
    for imgp in image_paths:
        img = cv2.imread(imgp, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_tensor, ratio, (pad_left, pad_top) = preprocess_image(img, input_h=input_h, input_w=input_w)
        dets = yolo_model(img_tensor).squeeze(0)
        dets = nms_dets(dets, conf_threshold=0.25, iou_threshold=0.45).numpy()

        FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 1.2
        FONT_THICKNESS = 2
        COLOR = [0, 255, 0]
        for x1, y1, x2, y2, conf, clsid in dets.tolist():
            clsid = int(clsid)
            x1 = round((x1 * input_w - pad_left) / ratio)
            x2 = round((x2 * input_w - pad_left) / ratio)
            y1 = round((y1 * input_h - pad_top) / ratio)
            y2 = round((y2 * input_h - pad_top) / ratio)
            text = f"{class_names[clsid]}:{conf:.4f}"
            (fw, fh), fb = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(img, (x1, y1 - fh - fb), (x1 + fw, y1), COLOR, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR, 2)
            cv2.putText(img, text, (x1, y1 - fb), FONT_FACE, FONT_SCALE, [0, 0, 0], FONT_THICKNESS)
        cv2.imshow("W", img)
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
