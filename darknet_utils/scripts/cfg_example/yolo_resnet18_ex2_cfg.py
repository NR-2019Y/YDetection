image_dir = {
    "train": "XXXX",
    "val": "XXXX",
}
label_dir = {
    "train": "XXXX",
    "val": "XXXX",
}

class_names = ["XXX", "XXX"]
input_height = 416
input_width = 416
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

wh_scales = [x / 13 for x in
             (0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828)]
obj_scale = 1.0
no_obj_scale = 0.5
cls_scale = 1.0
coord_scale = 5.0
rescore = True

batch_size = 16
val_batch_size = 32
epochs = 400
base_lr = 0.001
warm_up_epochs = 10
lr_steps = [200, 300]
weight_decay = 5e-4

cache_dir = f"~/DATA_CACHE/_gas_cache_{input_height}x{input_width}_train"
val_cache_dir = f"~/DATA_CACHE/_gas_cache_{input_height}x{input_width}_val"
image_device = 'cuda'
val_image_device = 'cuda'
train_num_expand = 4

num_workers = 0
val_num_workers = 0
save_freq = 20
device = "cuda"
output_dir = "_yolo_resnet18_ex2_gas_out"
