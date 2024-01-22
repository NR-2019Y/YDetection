import numpy as np
import time
import ctypes

cpool = ctypes.cdll.LoadLibrary("libpool.so")


def im2col_cf_np(img: np.ndarray, kh: int, kw: int, sh: int, sw: int, dh: int, dw: int,
                 pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0) -> np.ndarray:
    if pad_top != 0 or pad_bottom != 0 or pad_left != 0 or pad_right != 0:
        img = np.pad(img, [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]],
                     mode='constant', constant_values=0.)
    batch_size, ic, ih, iw = img.shape
    dkh = kh + (kh - 1) * (dh - 1)
    dkw = kw + (kw - 1) * (dw - 1)
    oh = (ih - dkh) // sh + 1
    ow = (iw - dkw) // sw + 1
    istepb, istepc, isteph, istepw = img.strides
    return np.lib.stride_tricks.as_strided(img, (batch_size, ic, oh, ow, kh, kw),
                                           (istepb, istepc, isteph * sh, istepw * sw, isteph * dh, istepw * dw),
                                           writeable=False).copy()


def im2col_cf_cpp(img: np.ndarray, kh: int, kw: int, sh: int, sw: int, dh: int, dw: int,
                  pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0) -> np.ndarray:
    img = np.ascontiguousarray(img)
    batch_size, ic, ih, iw = img.shape
    dkh = kh + (kh - 1) * (dh - 1)
    dkw = kw + (kw - 1) * (dw - 1)
    oh = (ih + pad_top + pad_bottom - dkh) // sh + 1
    ow = (iw + pad_left + pad_right - dkw) // sw + 1
    result = np.zeros((batch_size, ic, oh, ow, kh, kw), dtype=np.float32)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    cpool.im2col_cfirst(img.ctypes.data_as(c_float_p), result.ctypes.data_as(c_float_p),
                        ctypes.c_int(batch_size * ic), ctypes.c_int(ih), ctypes.c_int(iw),
                        ctypes.c_int(kh), ctypes.c_int(kw),
                        ctypes.c_int(sh), ctypes.c_int(sw),
                        ctypes.c_int(dh), ctypes.c_int(dw),
                        ctypes.c_int(pad_top), ctypes.c_int(pad_bottom),
                        ctypes.c_int(pad_left), ctypes.c_int(pad_right))
    return result


def main():
    B, IC, IH, IW = 4, 64, 200, 202
    KH, KW = 3, 7
    SH, SW = 2, 3
    DH, DW = 3, 4
    pad_top, pad_bottom, pad_left, pad_right = 3, 3, 3, 3

    # B, IC, IH, IW = 10, 8, 51, 50
    # KH, KW = 4, 7
    # SH, SW = 3, 3
    # DH, DW = 2, 3
    # pad_top, pad_bottom, pad_left, pad_right = 3, 3, 3, 3

    for _ in range(2):
        X = np.random.rand(B, IC, IH, IW).astype(np.float32)
        # X = np.arange(16, dtype=np.float32).reshape((B, IC, IH, IW))
        args = (X, KH, KW, SH, SW, DH, DW, pad_top, pad_bottom, pad_left, pad_right)

        tic = time.time()
        o1 = im2col_cf_np(*args)
        print(f"np : {time.time() - tic} sec")

        tic = time.time()
        o2 = im2col_cf_cpp(*args)
        print(f"cpp: {time.time() - tic} sec")

        print(o1.ravel())
        print(o2.ravel())

        print(np.abs(o2 - o1).max())


if __name__ == '__main__':
    main()
