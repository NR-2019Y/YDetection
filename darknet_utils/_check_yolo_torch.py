import darknet
from basic_modules import BuildDarknetInferenceModel
import ctypes
import glob
import numpy as np
import os.path as osp
import torch
import time

set_batch_network = darknet.lib.set_batch_network
set_batch_network.argtypes = [ctypes.c_void_p, ctypes.c_int]
set_batch_network.restype = None


def ptr_to_ndarray(ptr, size):
    buffer = (ctypes.c_float * size).from_address(ctypes.addressof(ptr.contents))
    return np.frombuffer(buffer, dtype=np.float32, count=size)


# CFG_FILE = '/home/a/PROJ/AlexeyAB/darknet/cfg/yolov1/tiny-yolo.cfg'
# WEIGHT_FILE = '/home/a/PROJ/AlexeyAB/darknet/weights/tiny-yolov1.weights'
# CFG_FILE = '/home/a/PROJ/AlexeyAB/ori_darknet/darknet/cfg/yolov2-tiny.cfg'
# WEIGHT_FILE = '/home/a/PROJ/AlexeyAB/ori_darknet/darknet/weights/yolov2-tiny.weights'
# CFG_FILE = '/home/a/PROJ/AlexeyAB/ori_darknet/darknet/cfg/yolov2-voc.cfg'
# WEIGHT_FILE = '/home/a/PROJ/AlexeyAB/ori_darknet/darknet/weights/yolov2-voc.weights'
CFG_FILE = '/home/a/PROJ/AlexeyAB/ori_darknet/darknet/cfg/yolov2.cfg'
WEIGHT_FILE = '/home/a/PROJ/AlexeyAB/ori_darknet/darknet/weights/yolov2.weights'
IMAGES_PATH = glob.glob('/mnt/d/GIT_REPO/datasets/VOC2007/images/*.jpg')

net = darknet.load_net(
    CFG_FILE.encode('utf8'),
    WEIGHT_FILE.encode('utf8'), 0)
set_batch_network(net, 1)
input_width = darknet.lib.network_width(net)
input_height = darknet.lib.network_width(net)

net_torch = BuildDarknetInferenceModel(CFG_FILE, WEIGHT_FILE).model
net_torch.eval()

for imgp in IMAGES_PATH:
    im = darknet.load_image(imgp.encode('utf8'), 0, 0)
    imr = darknet.letterbox_image(im, input_width, input_height)

    tic = time.time()
    pred_darknet = darknet.network_predict(net, imr.data)
    print("darknet time:", time.time() - tic)

    tic = time.time()
    imr_arr = ptr_to_ndarray(imr.data, imr.c * imr.w * imr.h). \
        reshape(1, 3, input_width, input_height)
    imr_tensor = torch.from_numpy(imr_arr)
    with torch.no_grad():
        pred_torch = net_torch(imr_tensor).ravel().numpy()
    pred_darknet = ptr_to_ndarray(pred_darknet, pred_torch.size)
    print("torch time:", time.time() - tic)

    # print(pred_torch.max(), pred_torch.min())
    # print(pred_darknet.max(), pred_darknet.min())
    r = pred_darknet / pred_torch
    print(osp.basename(imgp), np.abs(pred_darknet - pred_torch).max(),
          r.max(), r.min(), np.allclose(pred_darknet, pred_torch))

    darknet.free_image(im)
    darknet.free_image(imr)
