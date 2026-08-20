"""Inference utilities."""
import json
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import datasets
from tqdm import tqdm

from .config import DEVICE
from .data import WrapperSurgicalDataset, SurgicalTransform
from .models import SurgicalNet


def load_model(num_classes, weights_path, device):
    """Load a trained SurgicalNet."""
    model = SurgicalNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict_and_save(val_dir, csv_in, csv_out, weights_path, train_dir,
                     json_path, input_size=512):
    """
    Run inference on images listed in `csv_in['Frame_id']` and write
    predictions to `csv_out`.
    """
    df = pd.read_csv(csv_in)
    if "Frame_id" not in df.columns:
        raise KeyError("CSV must contain 'Frame_id' column")

    class_names = datasets.ImageFolder(train_dir, transform=None).classes
    mapping = json.load(open(json_path, 'r', encoding='utf-8'))
    infer_tf = SurgicalTransform(input_size, train=False)
    model = load_model(len(class_names), weights_path, DEVICE)

    preds = []
    for fn in tqdm(df["Frame_id"], desc="Inference"):
        fp = os.path.join(val_dir, fn)
        if not os.path.isfile(fp):
            raise FileNotFoundError(fp)

        img_pil = Image.open(fp).convert("RGB")
        img4 = WrapperSurgicalDataset.build_roi_tensor(img_pil, fn, mapping, infer_tf)
        x = img4.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(x)
        preds.append(class_names[int(out.argmax(1))])

    df["Phase_predict"] = preds
    df.to_csv(csv_out, index=False)
    print(f"Inference complete, results saved to {csv_out}")
