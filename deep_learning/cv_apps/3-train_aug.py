#!/usr/bin/env python3
"""Train a YOLO model with configurable augmentation, including support
for a fully custom Albumentations pipeline."""

from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
import albumentations as A


def train_with_augmentation(
    data,
    model_path="yolov8n.pt",
    epochs=50,
    imgsz=640,
    batch=16,
    augmentation=True,
    yolo_aug_params=None,
    albumentations_transforms=None,
    save=True,
    plots=True,
    verbose=True,
):
    """
    Trains a YOLO model, optionally overriding its built-in augmentation
    with a custom Albumentations pipeline.

    Returns:
        Tuple[YOLO, Any]: The trained YOLO model and the full training
        results object (as returned by `model.train`).
    """
    if albumentations_transforms is not None:

        def custom_init(self, p=1.0):
            self.p = p
            self.contains_spatial = True
            self.transform = A.Compose(
                albumentations_transforms,
                bbox_params=A.BboxParams(
                    format="yolo", label_fields=["class_labels"]
                ),
            )

        Albumentations.__init__ = custom_init

    model = YOLO(model_path)

    train_kwargs = dict(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        augment=augmentation,
        save=save,
        plots=plots,
        verbose=verbose,
    )

    if yolo_aug_params:
        train_kwargs.update(yolo_aug_params)

    results = model.train(**train_kwargs)

    return model, results
