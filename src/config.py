"""Central configuration for the surgical phase classifier."""
import os
import torch
from pathlib import Path

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths (modify these before running)
TRAIN_DIR = r"D:\白昊天\tianchi\take_data_class_16"
VAL_DIR = r"D:\白昊天\tianchi\frames"
CSV_IN = r"APTOS_val2 (1).csv"
CSV_OUT = r"APTOS_val2_with_preds(a31_yolo_den).csv"
JSON_PATH = "all_instrument_bboxes3.json"
MODEL_PATH = "best_surgical_model_a31_yolo.pth"

# Training hyperparameters
BATCH_SIZE = 16
NUM_WORKERS = 26
VAL_RATIO = 0.3
EPOCHS = 30
INPUT_SIZE = 512
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MASK_WEIGHT = 0.3

# Normalization stats
MEAN = [0.468, 0.325, 0.224]
STD = [0.198, 0.163, 0.142]
