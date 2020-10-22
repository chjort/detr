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
    def __init__(self, class_dirs, labels, query_transforms=None, target_transforms=None):
        self.class_dirs = class_dirs
        self.labels = labels
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
        chosen_pair_boxes = [os.path.join(dir_, fname + ".txt") for fname in image_filenames]

        chosen_pair_images = [Image.open(imgfile) for imgfile in chosen_pair_images]
        chosen_pair_boxes = [read_instre_boxes(label_file) for label_file in chosen_pair_boxes]

        # split to query and target
        query_img, target_img = chosen_pair_images
        query_boxes, target_boxes = chosen_pair_boxes

        # prepare query
        query_boxes = torch.tensor(query_boxes).reshape(-1, 4)
        query_boxes = box_xywh_to_xyxy(query_boxes)

        ## crop to query patch
        idx = torch.randint(0, len(query_boxes), [1])[0]
        box_to_crop = query_boxes[idx].numpy()
        query_img = query_img.crop(box_to_crop)

        label = torch.tensor([self.labels[item]], dtype=torch.int64)
        query_boxes = {"boxes": query_boxes,  # bounding boxes are only kept to comply with the original augmentations
                        "labels": label.repeat(len(query_boxes))}

        # prepare target
        target_boxes = torch.tensor(target_boxes).reshape(-1, 4)
        target_boxes = box_xywh_to_xyxy(target_boxes)
        target_boxes = {"boxes": target_boxes,
                         "labels": label.repeat(len(target_boxes))}


        # apply augmentations
        if self.query_transforms is not None:
            query_img, query_boxes = self.query_transforms(query_img, query_boxes)
            target_img, target_boxes = self.target_transforms(target_img, target_boxes)

        # overwrite default label dict (remove bounding boxes now that augmentation has been performed).
        query_boxes = {"label": item}

        sample = {"queries": (query_img, query_boxes),
                  "targets": (target_img, target_boxes)}
        return sample
