import numpy as np
import ctypes
from utils import size_2_t, get_pair

cpool = ctypes.cdll.LoadLibrary("libpool.so")


def basic_maxpool2d(image: np.ndarray, kh: int, kw: int, ceil_mode: bool = False):
    b, ic, ih, iw = image.shape
    # assert ih % kh == 0 and iw % kw == 0
    new_h, new_w = ih, iw
    if ih % kh != 0 or iw % kw != 0:
        if not ceil_mode:
            new_h = ih // kh * kh
            new_w = iw // kw * kw
            image = image[:, :, :new_h, :new_w]
        else:
            new_h = (ih + kh - 1) // kh * kh
            new_w = (iw + kw - 1) // kw * kw
            image = np.pad(image, [[0, 0], [0, 0], [0, new_h - ih], [0, new_w - iw]],
                           mode='constant', constant_values=float('-inf'))
    x = image.reshape((b, ic, new_h // kh, kh, new_w // kw, kw))
    x = np.transpose(x, (0, 1, 2, 4, 3, 5)).reshape(b, ic, new_h // kh, new_w // kw, kh * kw).max(-1)
    return x


def maxpool2d_nchw(img: np.ndarray, ksize: size_2_t, stride: size_2_t = 1, dilation: size_2_t = 1,
                   pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0) -> np.ndarray:
    batch_size, ic, ih, iw = img.shape
    kh, kw = get_pair(ksize)
    sh, sw = get_pair(stride)
    dh, dw = get_pair(dilation)
    dkh = kh + (kh - 1) * (dh - 1)
    dkw = kw + (kw - 1) * (dw - 1)
    oh = (ih + pad_top + pad_bottom - dkh) // sh + 1
    ow = (iw + pad_left + pad_right - dkw) // sw + 1
    img = np.pad(img, [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]],
                 mode='constant', constant_values=float('-inf'))
    istepb, istepc, isteph, istepw = img.strides
    x = np.lib.stride_tricks.as_strided(img, (batch_size, ic, oh, ow, kh, kw),
                                        (istepb, istepc, isteph * sh, istepw * sw, isteph * dh, istepw * dw))
    x = x.reshape(batch_size, ic, oh, ow, kh * kw)
    return x.max(axis=-1)


def maxpool2d_cpp(img: np.ndarray, ksize: size_2_t, stride: size_2_t = 1, dilation: size_2_t = 1,
                  pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0) -> np.ndarray:
    img = np.ascontiguousarray(img)
    batch_size, ic, ih, iw = img.shape
    kh, kw = get_pair(ksize)
    sh, sw = get_pair(stride)
    dh, dw = get_pair(dilation)
    dkh = kh + (kh - 1) * (dh - 1)
    dkw = kw + (kw - 1) * (dw - 1)
    oh = (ih + pad_top + pad_bottom - dkh) // sh + 1
    ow = (iw + pad_left + pad_right - dkw) // sw + 1
    result = np.empty((batch_size, ic, oh, ow), dtype=np.float32)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    cpool.maxpool2d(img.ctypes.data_as(c_float_p), result.ctypes.data_as(c_float_p),
                    ctypes.c_int(batch_size * ic), ctypes.c_int(ih), ctypes.c_int(iw),
                    ctypes.c_int(kh), ctypes.c_int(kw),
                    ctypes.c_int(sh), ctypes.c_int(sw),
                    ctypes.c_int(dh), ctypes.c_int(dw),
                    ctypes.c_int(pad_top), ctypes.c_int(pad_bottom), ctypes.c_int(pad_left), ctypes.c_int(pad_right))
    return result


def _check():
    import time
    import torch
    import torch.nn.functional as F

    for _ in range(3):
        x = np.random.rand(2, 500, 100, 102).astype(np.float32)
        # ceil_mode = False

        tic = time.time()
        y1 = F.max_pool2d(torch.from_numpy(x), (3, 5), (2, 7), padding=(1, 2)).numpy()
        print(f"torch: {time.time() - tic} sec")

        tic = time.time()
        y2 = maxpool2d_nchw(x, (3, 5), (2, 7), pad_top=1, pad_bottom=1, pad_left=2, pad_right=2)
        print(f"numpy: {time.time() - tic} sec")

        tic = time.time()
        y3 = maxpool2d_cpp(x, (3, 5), (2, 7), pad_top=1, pad_bottom=1, pad_left=2, pad_right=2)
        print(f"cpp: {time.time() - tic} sec")

        print(np.abs(y1 - y2).max())
        print(np.abs(y1 - y3).max())

    # for _ in range(3):
    #     x = np.random.rand(2, 512, 97, 102).astype(np.float32)
    #     ceil_mode = False
    #     tic = time.time()
    #     y1 = basic_maxpool2d(x, 3, 4, ceil_mode)
    #     print(f"numpy: {time.time() - tic} sec")
    #     tic = time.time()
    #     y2 = F.max_pool2d(torch.from_numpy(x), (3, 4), ceil_mode=ceil_mode).numpy()
    #     print(f"torch: {time.time() - tic} sec")

    #     print(np.abs(y1 - y2).max())


if __name__ == '__main__':
    _check()
