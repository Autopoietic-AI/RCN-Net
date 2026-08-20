# Surgical Phase Classification with YOLO ROI + 4-Channel Input

A PyTorch implementation of a surgical phase classifier that combines:

- **4-channel input**: RGB image + instrument ROI mask derived from YOLO detections.
- **RegNetY-008 backbone** adapted for 4-channel inputs (via `timm`).
- **Co-attention module** with instrument/eye-region attention and a spatial gating branch.
- **Mixed-precision training** with AdamW + CosineAnnealingWarmRestarts and early stopping.

This repo is a refactored/package version of the original single-file script `cla_a31_yolo1_4channal_v(1).py`.

## Project Structure

```
surgical_yolo4ch_classifier/
├── main.py              # Entry point: train, evaluate, infer
├── requirements.txt
├── README.md
└── src/
    ├── config.py        # Paths and hyperparameters
    ├── data.py          # Dataset, transforms, and data loaders
    ├── models.py        # SurgicalCoAttention + SurgicalNet
    ├── trainer.py       # Training loop with AMP and early stopping
    ├── inference.py     # Model loading + CSV inference
    └── visualize.py     # Confusion matrix / per-class accuracy plots
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Edit `src/config.py` to point to your local data paths:
   - `TRAIN_DIR`
   - `VAL_DIR`
   - `CSV_IN`
   - `JSON_PATH` (YOLO bbox mapping)

2. Run the full pipeline:

```bash
python main.py
```

The script will:
- Train a model if `MODEL_PATH` does not exist.
- Evaluate on the validation set and show confusion matrix / per-class accuracy.
- Run inference on `CSV_IN` and write predictions to `CSV_OUT`.

## Key Classes

- `src.data.SurgicalTransform`: brightness-aware train/val augmentation pipeline.
- `src.data.WrapperSurgicalDataset`: builds 4-channel tensors from RGB + YOLO ROI masks.
- `src.models.SurgicalCoAttention`: co-attention with learnable scaling and spatial gating.
- `src.models.SurgicalNet`: RegNetY-008 backbone + co-attention + decoder + classifier.
- `src.trainer.SurgicalTrainer`: handles training, validation, AMP, scheduler, and early stopping.

## Notes

- Make sure `all_instrument_bboxes3.json` maps each image filename to a list of YOLO detections with keys `x_center`, `y_center`, `w`, `h`.
- The first detection box in the list is used to build the ROI mask.
- Adjust `NUM_WORKERS`, `BATCH_SIZE`, and `INPUT_SIZE` in `src/config.py` based on your hardware.
