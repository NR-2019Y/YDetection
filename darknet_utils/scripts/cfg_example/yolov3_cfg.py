cfg_file = "/home/a/PROJ/AlexeyAB/ori_darknet/darknet/cfg/yolov3-tiny.cfg"
weight_file = "/home/a/PROJ/AlexeyAB/ori_darknet/darknet/weights/yolov3-tiny.weights"
image_dir = {
    "train": "XXXX",
    "val": "XXXX",
}
label_dir = {
    "train": "XXXX",
    "val": "XXXX",
}
num_anchors = 3
class_names = ["XXX", "XXX"]
batch_size = 12
num_workers = 4
epochs = 160
save_freq = 20
device = "cuda"
output_dir = "_yolov3_gas_out"
lr = 1e-4
