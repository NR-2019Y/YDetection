import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms
import torchvision.transforms.functional as Fv
import glob
from typing import Sequence, List, Optional, Tuple, Callable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os.path as osp
import random

_NCOLS = 130


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


def adjust_contrast(img: torch.Tensor, factor: float) -> torch.Tensor:
    mean = Fv.rgb_to_grayscale(img).mean()
    return (factor * img + (1 - factor) * mean).clamp(0, 1)


def adjust_saturation(img: torch.Tensor, factor: float) -> torch.Tensor:
    gray_img = Fv.rgb_to_grayscale(img)
    return (factor * img + (1 - factor) * gray_img).clamp(0, 1)


def adjust_brightness(img: torch.Tensor, factor: float) -> torch.Tensor:
    return img.mul(factor).clamp(0, 1)


def adjust_gamma(img: torch.Tensor, factor: float) -> torch.Tensor:
    return img.pow(factor).clamp(0, 1)


def equalize(img: torch.Tensor) -> torch.Tensor:
    return Fv.equalize(img.mul(255).round().to(torch.uint8)).div(255)


class ImageTransform:
    def __init__(self, factor_contrast: Tuple[float] = (0.90, 1.11),
                 factor_saturation: Tuple[float, float] = (0.90, 1.11),
                 factor_brightness: Tuple[float, float] = (0.90, 1.11),
                 factor_gamma: Tuple[float, float] = (0.90, 1.11),
                 prob_equalize: float = 0.5):
        self.factor_contrast = factor_contrast
        self.factor_saturation = factor_saturation
        self.factor_brightness = factor_brightness
        self.factor_gamma = factor_gamma
        self.prob_equalize = prob_equalize

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_tensor = adjust_contrast(img_tensor, random.uniform(self.factor_contrast[0], self.factor_contrast[1]))
        img_tensor = adjust_saturation(img_tensor, random.uniform(self.factor_saturation[0], self.factor_saturation[1]))
        img_tensor = adjust_brightness(img_tensor, random.uniform(self.factor_brightness[0], self.factor_brightness[1]))
        img_tensor = adjust_gamma(img_tensor, random.uniform(self.factor_gamma[0], self.factor_gamma[1]))
        if random.uniform(0, 1) > self.prob_equalize:
            img_tensor = equalize(img_tensor)
        return img_tensor


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args if len(args) > 1 else args[0]


class RandomFlipLR:
    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, img_tensor: torch.Tensor, label_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.uniform(0, 1) > self.prob:
            img_tensor = img_tensor.flip(-1)
            label_tensor[:, 0] = 1.0 - label_tensor[:, 0]
        return img_tensor, label_tensor


class RandomFlipUD:
    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, img_tensor: torch.Tensor, label_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.uniform(0, 1) > self.prob:
            img_tensor = img_tensor.flip(-2)
            label_tensor[:, 1] = 1.0 - label_tensor[:, 1]
        return img_tensor, label_tensor


class ResizePad:
    def __init__(self, input_height, input_width, pad_value: float = 114. / 255.):
        self.input_height = input_height
        self.input_width = input_width
        self.pad_value = pad_value

    def __call__(self, img_tensor: torch.Tensor, label_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_h, img_w = img_tensor.shape[-2:]
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


class RandomResizePad:
    def __init__(self, ratio_min: float, ratio_max: float, input_height, input_width, pad_value: float = 114. / 255.):
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.input_height = input_height
        self.input_width = input_width
        self.pad_value = pad_value

    def __call__(self, img_tensor: torch.Tensor, label_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_h, img_w = img_tensor.shape[-2:]
        wh_ratio = (img_w / img_h) * random.uniform(self.ratio_min, self.ratio_max)  # resize后的宽高比
        # resize后 new_h: new_w = 1: wh_ratio
        ratio_h = float(self.input_height)
        ratio_w = float(self.input_width / wh_ratio)
        if ratio_h > ratio_w:
            new_w = self.input_width
            new_h = round(new_w / wh_ratio)
        else:
            new_h = self.input_height
            new_w = round(new_h * wh_ratio)
        img_tensor = Fv.resize(img_tensor, [new_h, new_w])
        pad_h = self.input_height - new_h
        pad_w = self.input_width - new_w
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


class YoloDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str,
                 input_height: int, input_width: int,
                 mean: Optional[List[float]] = None, std: Optional[List[float]] = None,
                 image_only_transform: Optional[Callable] = None,
                 image_label_transform: Optional[Callable] = None,
                 image_device: torch.device = torch.device('cpu'),
                 cache_dir: Optional[str] = None,
                 num_expand: int = 1):
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

        self.image_only_transform = image_only_transform
        self.image_label_transform = image_label_transform if image_label_transform is not None \
            else ResizePad(input_height=input_height, input_width=input_width)
        self.image_device = image_device

        if cache_dir:
            cache_dir = osp.expandvars(osp.expanduser(cache_dir))
            self.cache_dir = cache_dir
            self.cache_images(cache_dir, image_paths)
        else:
            self.cache_dir = None
        self.n = len(self.image_paths)
        self.num_expand = num_expand

    @staticmethod
    def cache_images(cache_dir: str, image_paths: List[str]):
        print(f"cache images to {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        for i, imgp in enumerate(tqdm(image_paths, ncols=_NCOLS)):
            save_path = osp.join(cache_dir, f"img-{i}.pt")
            if osp.isfile(save_path):
                continue
            img = cv2.imread(imgp, cv2.IMREAD_COLOR)
            img = np.ascontiguousarray(img[..., ::-1])
            img_tensor = Fv.to_tensor(img)
            torch.save(img_tensor, save_path)

    def __len__(self):
        return self.n * self.num_expand

    def __getitem__(self, item):
        item %= self.n
        if self.cache_dir:
            img_tensor = torch.load(osp.join(self.cache_dir, f"img-{item}.pt"), map_location=self.image_device)
        else:
            imgp: str = self.image_paths[item]
            img = cv2.imread(imgp, cv2.IMREAD_COLOR)
            img = np.ascontiguousarray(img[..., ::-1])
            img_tensor = Fv.to_tensor(img).to(self.image_device)
        label_tensor: torch.Tensor = self.labels[item].clone()

        if self.image_only_transform is not None:
            img_tensor = self.image_only_transform(img_tensor)

        if self.normalize is not None:
            img_tensor = self.normalize(img_tensor)

        img_tensor, label_tensor = self.image_label_transform(img_tensor, label_tensor)

        return img_tensor, label_tensor


def tensor_to_bgr(img_tensor: torch.Tensor) -> np.ndarray:
    return img_tensor.flip(0).permute(1, 2, 0).mul(255).round().to(torch.uint8).contiguous().cpu().numpy()


def collate_fn(L):
    images, labels = zip(*L)
    return torch.stack(images, 0), labels


def make_dataloader(batch_size: int, num_workers: int = 0):
    input_size = 416
    train_dataset = YoloDataset(image_dir='/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/images/train',
                                label_dir='/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/labels/train',
                                input_height=input_size, input_width=input_size,
                                cache_dir=f"~/DATA_CACHE/_v5c_cache_{input_size}_train")
    val_dataset = YoloDataset(image_dir='/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/images/val',
                              label_dir='/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/labels/val',
                              input_height=input_size, input_width=input_size,
                              cache_dir=f"~/DATA_CACHE/_v5c_cache_{input_size}_val")
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, drop_last=False,
                                collate_fn=collate_fn, num_workers=num_workers)
    return train_dataloader, val_dataloader


def _check_dataset():
    class_names = ['bird', 'cat', 'dog', 'horse', 'person']
    input_height, input_width = 600, 800
    image_dir = '/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/images/train'
    label_dir = '/home/a/MY_PROJ/CV_LEARN/Detection/YOLOV1/voc07-5c/labels/train'
    ds = YoloDataset(image_dir=image_dir, label_dir=label_dir,
                     input_height=input_height, input_width=input_width,
                     image_only_transform=ImageTransform(),
                     # image_label_transform=None)
                     image_label_transform=Compose([
                         RandomFlipLR(0.5),
                         # RandomFlipUD(0.5),
                         RandomResizePad(ratio_min=0.750, ratio_max=1.333,
                                         input_height=input_height, input_width=input_width)]),
                     cache_dir=f"~/DATA_CACHE/_v5c_cache_{input_height}x{input_width}_train")

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
