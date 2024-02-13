import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms
import torchvision.transforms.functional as Fv
import glob
from typing import Sequence, List, Optional
from torch.utils.data import DataLoader, Dataset
import os.path as osp


def get_labels(label_paths: Sequence[str]) -> List[torch.Tensor]:
    labels: List[torch.Tensor] = []
    for labp in label_paths:
        with open(labp, "r") as f:
            curr_label = []
            for sline in f:
                clsid, cx, cy, w, h = map(float, sline.split())
                curr_label.append(torch.tensor([
                    cx, cy, w, h, clsid
                ], dtype=torch.float32))
            if not len(curr_label):
                curr_label = torch.zeros((0, 5), dtype=torch.float32)
            else:
                curr_label = torch.stack(curr_label, dim=0)
            labels.append(curr_label)
    return labels


class YoloDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str,
                 input_height: int, input_width: int,
                 mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        super().__init__()
        image_paths = sorted(glob.glob(osp.join(image_dir, "*.jpg")))
        label_paths = sorted(glob.glob(osp.join(label_dir, "*.txt")))
        assert len(image_paths) == len(label_paths)
        for imgp, labp in zip(image_paths, label_paths):
            img_fn = osp.splitext(osp.basename(imgp))[0]
            lab_fn = osp.splitext(osp.basename(labp))[0]
            assert img_fn == lab_fn
        self.input_height = input_height
        self.input_width = input_width
        self.image_paths = image_paths
        self.labels = get_labels(label_paths)
        self.normalize = None
        if mean is None:
            assert std is None
        else:
            assert std is not None
            self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        imgp: str = self.image_paths[item]
        label_tensor: torch.Tensor = self.labels[item].clone()
        img = cv2.imread(imgp, cv2.IMREAD_COLOR)
        img_h, img_w = img.shape[:2]
        img = np.ascontiguousarray(img[..., ::-1])
        img_tensor = Fv.to_tensor(img)

        if self.normalize is not None:
            img_tensor = self.normalize(img_tensor)

        ratio = min(self.input_height / img_h, self.input_width / img_w)
        new_h = round(img_h * ratio)
        new_w = round(img_w * ratio)
        pad_h = self.input_height - new_h
        pad_w = self.input_width - new_w

        img_tensor = Fv.resize(img_tensor, [new_h, new_w])
        if pad_h != 0 or pad_w != 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            img_tensor = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom))
            if pad_h != 0:
                label_tensor[:, 1] = (label_tensor[:, 1] * new_h + pad_top) / self.input_height
                label_tensor[:, 3] *= (new_h / self.input_height)
            if pad_w != 0:
                label_tensor[:, 0] = (label_tensor[:, 0] * new_w + pad_left) / self.input_width
                label_tensor[:, 2] *= (new_w / self.input_width)

        return img_tensor, label_tensor


def tensor_to_bgr(img_tensor: torch.Tensor) -> np.ndarray:
    return img_tensor.flip(0).permute(1, 2, 0).mul(255).round().to(torch.uint8).contiguous().numpy()


def collate_fn(L):
    images, labels = zip(*L)
    return torch.stack(images, 0), labels


def make_dataloader(batch_size: int, num_workers: int = 0):
    input_size = 416
    train_dataset = YoloDataset(image_dir='/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/images/train',
                                label_dir='/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/labels/train',
                                input_height=input_size, input_width=input_size)
    val_dataset = YoloDataset(image_dir='/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/images/val',
                              label_dir='/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/labels/val',
                              input_height=input_size, input_width=input_size)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, drop_last=False,
                                collate_fn=collate_fn, num_workers=num_workers)
    return train_dataloader, val_dataloader


def _check_dataset():
    class_names = ['bird', 'cat', 'dog', 'horse', 'person']
    ds = YoloDataset(image_dir='/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/images/train',
                     label_dir='/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/labels/train',
                     input_height=300, input_width=400)

    cv2.namedWindow("W", cv2.WINDOW_NORMAL)
    for img_tensor, label_tensor in ds:
        img: np.ndarray = tensor_to_bgr(img_tensor)
        # print(img.shape, img.dtype)
        img_h, img_w = img.shape[:2]
        label = label_tensor.tolist()

        FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.6
        FONT_THICKNESS = 1
        COLOR = [0, 255, 0]

        for cx, cy, w, h, clsid in label:
            clsid = int(clsid)
            x1 = round((cx - 0.5 * w) * img_w)
            y1 = round((cy - 0.5 * h) * img_h)
            x2 = round((cx + 0.5 * w) * img_w)
            y2 = round((cy + 0.5 * h) * img_h)
            text = class_names[clsid]
            (fw, fh), fb = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(img, (x1, y1 - fh - fb), (x1 + fw, y1), COLOR, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR, 2)
            cv2.putText(img, text, (x1, y1 - fb), FONT_FACE, FONT_SCALE, [0, 0, 0], FONT_THICKNESS)
        cv2.imshow("W", img)
        if cv2.waitKey(0) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _check_dataset()
