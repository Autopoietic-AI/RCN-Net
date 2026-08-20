"""Dataset and data-loading utilities."""
import json
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data._utils.collate import default_collate

from .config import MEAN, STD


class SurgicalTransform:
    """Training/validation transforms with brightness-aware jitter."""

    def __init__(self, input_size=512, train=True, target_brightness=1.0):
        self.train = train
        self.input_size = input_size
        self.target_brightness = target_brightness

        if train:
            low = max(0, target_brightness - 0.2)
            high = target_brightness + 0.2
            self.transform = transforms.Compose([
                transforms.Resize(600),
                transforms.RandomRotation(10),
                transforms.RandomCrop(input_size),
                transforms.ColorJitter(brightness=(low, high), contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value='random'),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(600),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ])

    def __call__(self, img):
        return self.transform(img)


class WrapperSurgicalDataset(Dataset):
    """
    Dataset wrapper that builds a 4-channel tensor (RGB + instrument ROI mask)
    from YOLO detection results stored in a JSON file.
    """

    def __init__(self, base_ds, transform, instrument_aware=True, json_path="all_instrument_bboxes3.json"):
        self.base = base_ds
        self.transform = transform
        self.instrument_aware = instrument_aware
        with open(json_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)

        if hasattr(base_ds, "dataset"):  # from random_split
            ds, idxs = base_ds.dataset, base_ds.indices
            self.paths = [ds.samples[i][0] for i in idxs]
            self.labels = [ds.samples[i][1] for i in idxs]
        else:  # plain ImageFolder
            self.paths = [s[0] for s in base_ds.samples]
            self.labels = [s[1] for s in base_ds.samples]

    @staticmethod
    def build_roi_tensor(img_pil, file_name, mapping, transform):
        """
        Given a PIL RGB image and its filename, return a [4, H, W] tensor
        (RGB + binary instrument ROI mask).
        """
        h0, w0 = img_pil.size[1], img_pil.size[0]

        inst_np = np.zeros((h0, w0), dtype=np.float32)
        dets = mapping.get(file_name, [])
        if dets:  # use the first detection box
            o = dets[0]
            x1 = int(max(0, o["x_center"] - o["w"] / 2))
            y1 = int(max(0, o["y_center"] - o["h"] / 2))
            x2 = int(min(w0, o["x_center"] + o["w"] / 2))
            y2 = int(min(h0, o["y_center"] + o["h"] / 2))
            inst_np[y1:y2, x1:x2] = 1.0

        img_t = transform(img_pil)
        inst_t = cv2.resize(
            inst_np,
            (transform.input_size, transform.input_size),
            interpolation=cv2.INTER_NEAREST
        )
        inst_t = torch.from_numpy(inst_t).unsqueeze(0).float()
        return torch.cat([img_t, inst_t], dim=0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        fn = os.path.basename(path)
        img_pil = Image.open(path).convert("RGB")

        img4_t = self.build_roi_tensor(img_pil, fn, self.mapping, self.transform)

        mask_np = np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.float32)
        if self.instrument_aware:
            arr = np.array(img_pil)
            hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
            m1 = cv2.inRange(hsv, (15, 50, 50), (40, 255, 255))
            m2 = cv2.inRange(lab, (0, 120, 120), (255, 140, 140))
            comb = cv2.bitwise_or(m1, m2)
            K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            comb = cv2.morphologyEx(comb, cv2.MORPH_OPEN, K)
            mask_np = comb.astype(np.float32) / 255.0

        mask_t = cv2.resize(
            mask_np,
            (self.transform.input_size, self.transform.input_size),
            interpolation=cv2.INTER_NEAREST
        )
        mask_t = torch.from_numpy(mask_t).unsqueeze(0).float()
        return img4_t, label, mask_t


def build_loaders(train_dir, val_dir, batch_size, num_workers, val_ratio,
                  json_path, input_size=512):
    """Build train/validation dataloaders."""
    base = datasets.ImageFolder(train_dir, transform=None)
    total = len(base)
    vnum = int(total * val_ratio)
    tnum = total - vnum
    train_base, val_base = random_split(base, [tnum, vnum])

    class_mapping = {d: int(d) for d in base.classes}
    train_base.dataset.class_to_idx = class_mapping
    val_base.dataset.class_to_idx = class_mapping

    train_ds = WrapperSurgicalDataset(
        train_base,
        transform=SurgicalTransform(input_size, train=True),
        json_path=json_path
    )
    val_ds = WrapperSurgicalDataset(
        val_base,
        transform=SurgicalTransform(input_size, train=False),
        json_path=json_path
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=default_collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=default_collate
    )
    return train_loader, val_loader, class_mapping
