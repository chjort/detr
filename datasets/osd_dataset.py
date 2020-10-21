import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from util.boxes import box_xywh_to_xyxy

# INSTRE_PATH = "/datadrive/crr/datasets/instre"
INSTRE_PATH = "/home/ch/datasets/instre"
TRAIN_PATH = INSTRE_PATH + "/INSTRE-S-TRAIN"
TEST_PATH = INSTRE_PATH + "/INSTRE-S-TEST"


def read_instre_boxes(box_file):
    """
    Returns box with format [x0, y0, w, h]
    """

    with open(box_file, "r") as f:
        boxes = f.read().strip("\n").split("\n")
        boxes = [box.split(" ") for box in boxes]
        boxes = [[int(c) for c in box] for box in boxes]
    return boxes


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
        chosen_pair_labels = [read_instre_boxes(label_file) for label_file in chosen_pair_labels]

        chosen_pair_labels = [torch.tensor(boxes).reshape(-1, 4) for boxes in chosen_pair_labels]
        chosen_pair_labels = [box_xywh_to_xyxy(boxes) for boxes in chosen_pair_labels]
        chosen_pair_labels = [{"boxes": boxes, "labels": torch.tensor([item], dtype=torch.int64).repeat(len(boxes))}
                              for boxes in chosen_pair_labels]

        query_img, target_img = chosen_pair_images
        query_labels, target_labels = chosen_pair_labels

        if self.transforms is not None:
            query_img, query_labels = self.transforms(query_img, query_labels)
            target_img, target_labels = self.transforms(target_img, target_labels)

        sample = {"queries": (query_img, query_labels),
                  "targets": (target_img, target_labels)}
        return sample
