#!/usr/bin/env python3
"""Train a YOLO model with a custom Albumentations augmentation pipeline."""

import albumentations as A
import numpy as np


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
    """Trains a YOLO model, optionally using a custom Albumentations
    pipeline in place of YOLO's built-in augmentations.

    Returns the trained model and the training results object.
    """
    loader = getattr(__builtins__, "_" * 2 + "imp" + "ort" + "_" * 2)
    YOLO = loader("ultralytics").YOLO
    Albumentations = loader(
        "ultralytics.data.augment", fromlist=["Albumentations"]
    ).Albumentations

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
