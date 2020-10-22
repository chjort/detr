import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from util.boxes import box_xywh_to_xyxy


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
    def __init__(self, class_dirs, query_transforms=None, target_transforms=None):
        self.class_dirs = class_dirs
        self.query_transforms = query_transforms
        self.target_transforms = target_transforms

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

        # split to query and target
        query_img, target_img = chosen_pair_images
        query_labels, target_labels = chosen_pair_labels

        # prepare query
        query_labels = torch.tensor(query_labels).reshape(-1, 4)
        query_labels = box_xywh_to_xyxy(query_labels)

        # crop to query patch
        idx = torch.randint(0, len(query_labels), [1])[0]
        box_to_crop = query_labels[idx].numpy()
        query_img = query_img.crop(box_to_crop)

        query_labels = {"boxes": query_labels,
                        "labels": torch.tensor([item], dtype=torch.int64).repeat(len(query_labels))}

        # prepare target
        target_labels = torch.tensor(target_labels).reshape(-1, 4)
        target_labels = box_xywh_to_xyxy(target_labels)
        target_labels[:, 2:] += target_labels[:, :2]
        target_labels[:, 0::2].clamp_(min=0, max=target_img.width)
        target_labels[:, 1::2].clamp_(min=0, max=target_img.height)
        target_labels = {"boxes": target_labels,
                         "labels": torch.tensor([item], dtype=torch.int64).repeat(len(target_labels))}

        if self.query_transforms is not None:
            query_img, query_labels = self.query_transforms(query_img, query_labels)
            target_img, target_labels = self.target_transforms(target_img, target_labels)

        # overwrite default label dict
        query_labels = {"label": item}

        sample = {"queries": (query_img, query_labels),
                  "targets": (target_img, target_labels)}
        return sample
