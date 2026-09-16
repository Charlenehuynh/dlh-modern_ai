#!/usr/bin/env python3
"""
This module performs two-phase hyperparameter tuning and training for a
YOLO object detection model: a lightweight hyperparameter search followed
by full training from the best discovered checkpoint.
"""
from ultralytics import YOLO


def tune_hyperparameters():
    """
    Performs two-phase hyperparameter tuning and training:
    Phase 1: Lightweight hyperparameter search
    Phase 2: Continue training from best checkpoint

    Returns:
        None
    """
    data = "datasets/detection/data.yaml"

    model = YOLO("yolov8n.pt")

    model.tune(
        data=data,
        epochs=10,
        iterations=20,
        optimizer="AdamW",
        plots=False,
        save=False,
        val=True,
        space={
            "lr0": (1e-5, 1e-1),
            "lrf": (0.01, 1.0),
            "momentum": (0.6, 0.98),
            "weight_decay": (0.0, 0.001),
            "box": (0.02, 0.2),
            "cls": (0.2, 4.0),
            "dfl": (0.4, 6.0),
            "hsv_h": (0.0, 0.1),
            "hsv_s": (0.0, 0.9),
            "hsv_v": (0.0, 0.9),
            "degrees": (0.0, 45.0),
            "translate": (0.0, 0.9),
            "scale": (0.0, 0.9),
            "mosaic": (0.0, 1.0),
            "mixup": (0.0, 1.0),
        },
    )

    best_weights = "runs/detect/tune/weights/best.pt"
    best_hyp_path = "runs/detect/tune/best_hyperparameters.yaml"

    model = YOLO(best_weights)

    model.train(
        data=data,
        epochs=150,
        cfg=best_hyp_path,
        patience=20,
        plots=True,
        save=True,
        verbose=True,
    )

    model.save("best_model.pt")

    return None
