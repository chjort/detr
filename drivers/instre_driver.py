import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DistributedSampler, DataLoader

from datasets.coco import make_coco_transforms
from util.boxes import box_xywh_to_xyxy
from util.plot_utils import plot_results
from util.misc import collate_fn, collate_fn_os

# INSTRE_PATH = "/datadrive/crr/datasets/instre"
INSTRE_PATH = "/home/ch/datasets/instre"
TRAIN_PATH = INSTRE_PATH + "/INSTRE-S-TRAIN"
TEST_PATH = INSTRE_PATH + "/INSTRE-S-TEST"


def read_instre_box(box_file):
    """
    Returns box with format [x0, y0, w, h]
    """

    with open(box_file, "r") as f:
        box = f.read().strip("\n").split(" ")
        box = [int(c) for c in box]
    return box


class OSDDataset(Dataset):
    def __init__(self, class_dirs, transforms=None):
        self.class_dirs = class_dirs
        self.transforms = transforms

    def __len__(self):
        return len(self.class_dirs)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        dir_ = self.class_dirs[item]
        dir_image_files = glob.glob(os.path.join(dir_, "*.jpg"))

        chosen_pair_images = np.random.choice(dir_image_files, 2)
        image_filenames = [os.path.basename(os.path.splitext(img)[0]) for img in chosen_pair_images]
        chosen_pair_labels = [os.path.join(dir_, fname + ".txt") for fname in image_filenames]

        chosen_pair_images = [Image.open(imgfile) for imgfile in chosen_pair_images]
        chosen_pair_labels = [read_instre_box(label_file) for label_file in chosen_pair_labels]

        chosen_pair_labels = [torch.tensor(box).reshape(-1, 4) for box in chosen_pair_labels]
        chosen_pair_labels = [box_xywh_to_xyxy(box) for box in chosen_pair_labels]
        chosen_pair_labels = [{"boxes": box, "labels": torch.tensor([item], dtype=torch.int64)} for box
                              in chosen_pair_labels]

        query_img, target_img = chosen_pair_images
        query_labels, target_labels = chosen_pair_labels

        if self.transforms is not None:
            query_img, query_labels = self.transforms(query_img, query_labels)
            target_img, target_labels = self.transforms(target_img, target_labels)

        sample = {"queries": (query_img, query_labels),
                  "targets": (target_img, target_labels)}
        return sample
        # return (query_img, query_labels)
        # return {"queries": ("qimg", "qlabels"),
        #         "targets": ("timg", "tlabels")}


# %%
BATCH_SIZE = 4
DISTRIBUTED = False
NUM_WORKERS = 2

class_dirs_train = glob.glob(os.path.join(TRAIN_PATH, "*/"))
class_dirs_val = glob.glob(os.path.join(TEST_PATH, "*/"))

dataset_train = OSDDataset(class_dirs_train,
                           make_coco_transforms("train")
                           )
dataset_val = OSDDataset(class_dirs_val,
                         make_coco_transforms("val")
                         )
# it = iter(dataset_val)

# %%
if DISTRIBUTED:
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, BATCH_SIZE, drop_last=True)

dataloader_train = DataLoader(dataset_train,
                              batch_sampler=batch_sampler_train,
                              collate_fn=collate_fn_os,
                              num_workers=NUM_WORKERS)

# %%
it = iter(dataloader_train)
b = next(it)

len(b["queries"][0].tensors)
len(b["queries"][1])

len(b["targets"][0].tensors)
len(b["targets"][1])
