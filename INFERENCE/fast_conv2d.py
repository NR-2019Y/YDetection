import numpy as np
import ctypes
from utils import size_2_t, get_pair
from typing import Optional

cconv = ctypes.cdll.LoadLibrary("libfastconv.so")


def fast_conv2d_np(img: np.ndarray, kernel: np.ndarray, bias: Optional[np.ndarray],
                   stride: size_2_t = 1, dilation: size_2_t = 1,
                   pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0) -> np.ndarray:
    if pad_top != 0 or pad_bottom != 0 or pad_left != 0 or pad_right != 0:
        img = np.pad(img, [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]],
                     mode='constant', constant_values=0.)
    batch_size, _ic, ih, iw = img.shape
    oc, ic, kh, kw = kernel.shape
    assert _ic == ic
    sh, sw = get_pair(stride)
    dh, dw = get_pair(dilation)
    dkh = kh + (kh - 1) * (dh - 1)
    dkw = kw + (kw - 1) * (dw - 1)
    oh = (ih - dkh) // sh + 1
    ow = (iw - dkw) // sw + 1
    istepb, istepc, isteph, istepw = img.strides
    x = np.lib.stride_tricks.as_strided(img, (batch_size, ic, kh, kw, oh, ow),
                                        (istepb, istepc, isteph * dh, istepw * dw, isteph * sh, istepw * sw),
                                        writeable=False)
    conv_result = np.matmul(
        kernel.reshape(1, oc, ic * kh * kw),
        x.reshape(batch_size, ic * kh * kw, oh * ow)).reshape(batch_size, oc, oh, ow)
    return conv_result if bias is None else conv_result + bias[None, :, None, None]


def fast_conv2d_cpp(img: np.ndarray, kernel: np.ndarray, bias: Optional[np.ndarray],
                    stride: size_2_t = 1, dilation: size_2_t = 1,
                    pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0) -> np.ndarray:
    batch_size, _ic, ih, iw = img.shape
    oc, ic, kh, kw = kernel.shape
    assert _ic == ic
    sh, sw = get_pair(stride)
    dh, dw = get_pair(dilation)
    dkh = kh + (kh - 1) * (dh - 1)
    dkw = kw + (kw - 1) * (dw - 1)
    oh = (ih + pad_top + pad_bottom - dkh) // sh + 1
    ow = (iw + pad_left + pad_right - dkw) // sw + 1

    result = np.empty((batch_size, oc, oh, ow), dtype=np.float32)
    c_float_p = ctypes.POINTER(ctypes.c_float)

    img = np.ascontiguousarray(img)
    kernel = np.ascontiguousarray(kernel)
    if bias is not None:
        bias = np.ascontiguousarray(bias)
        bias_ptr = bias.ctypes.data_as(c_float_p)
    else:
        bias_ptr = c_float_p()

    cconv.conv2d_cpu(img.ctypes.data_as(c_float_p), kernel.ctypes.data_as(c_float_p),
                     bias_ptr, result.ctypes.data_as(c_float_p),
                     ctypes.c_int(batch_size), ctypes.c_int(ic), ctypes.c_int(ih), ctypes.c_int(iw),
                     ctypes.c_int(oc), ctypes.c_int(kh), ctypes.c_int(kw),
                     ctypes.c_int(sh), ctypes.c_int(sw), ctypes.c_int(dh), ctypes.c_int(dw),
                     ctypes.c_int(pad_top), ctypes.c_int(pad_bottom),
                     ctypes.c_int(pad_left), ctypes.c_int(pad_right))
    return result


def _check_conv():
    import time
    import torch
    from torch import nn
    import torch.nn.functional as F

    b, ic, ih, iw = 16, 3, 640, 640
    # b, ic, ih, iw = 1, 32, 640, 640
    # b, ic, ih, iw = 1, 128, 340, 300
    oc, kh, kw = 256, 7, 7
    # b, ic, ih, iw = 32, 32, 200, 203
    # oc, kh, kw = 64, 7, 7
    stride = 2
    pad = 3

    for _ in range(10):
        kernel = np.random.rand(oc, ic, kh, kw).astype(np.float32)
        bias = np.random.rand(oc).astype(np.float32)
        # bias = None
        X = np.random.rand(b, ic, ih, iw).astype(np.float32)
        tic = time.time()
        p1 = F.conv2d(torch.from_numpy(X),
                      torch.from_numpy(kernel),
                      None if bias is None else torch.from_numpy(bias),
                      stride, pad).numpy()
        print(f"torch: {time.time() - tic} sec")

        tic = time.time()
        p2 = fast_conv2d_np(X, kernel, bias, stride, pad_top=pad, pad_bottom=pad, pad_left=pad, pad_right=pad)
        print(f"numpy: {time.time() - tic} sec")
        ratio = p1 / p2
        print(np.abs(p1 - p2).max(), ratio.max(), ratio.min(), np.allclose(p1, p2))

        tic = time.time()
        p2 = fast_conv2d_cpp(X, kernel, bias, stride, pad_top=pad, pad_bottom=pad, pad_left=pad, pad_right=pad)
        print(f"cpp: {time.time() - tic} sec")
        ratio = p1 / p2
        print(np.abs(p1 - p2).max(), ratio.max(), ratio.min(), np.allclose(p1, p2))


if __name__ == '__main__':
    _check_conv()
