"""Entry point: train (if needed), evaluate, and run inference."""
import os
import torch

from src.config import (
    TRAIN_DIR, VAL_DIR, CSV_IN, CSV_OUT, JSON_PATH, MODEL_PATH,
    BATCH_SIZE, NUM_WORKERS, VAL_RATIO, EPOCHS, INPUT_SIZE,
    PATIENCE, LEARNING_RATE, WEIGHT_DECAY, DEVICE
)
from src.trainer import SurgicalTrainer
from src.inference import predict_and_save
from src.visualize import visualize_metrics


def main():
    trainer = SurgicalTrainer(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO,
        json_path=JSON_PATH,
        input_size=INPUT_SIZE,
        patience=PATIENCE,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    if os.path.exists(MODEL_PATH):
        print(f"Found existing model {MODEL_PATH}, skipping training.")
        trainer.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE), strict=False
        )
    else:
        print("No trained model found, starting training...")
        trainer.run(epochs=EPOCHS, save_path=MODEL_PATH)

    class_names = list(trainer.class_mapping.keys())
    print("Class names:", class_names)

    trainer.model.to(DEVICE).eval()
    visualize_metrics(trainer.model, trainer.val_loader, class_names, DEVICE)

    predict_and_save(
        val_dir=VAL_DIR,
        csv_in=CSV_IN,
        csv_out=CSV_OUT,
        weights_path=MODEL_PATH,
        train_dir=TRAIN_DIR,
        json_path=JSON_PATH,
        input_size=INPUT_SIZE
    )


if __name__ == "__main__":
    main()
