#!/usr/bin/env python3
"""
This module trains a YOLO model using data augmentation, and allows
custom Albumentations transforms to be applied during training.
"""
import albumentations as A
import numpy as np
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations


def train_with_augmentation(data, model_path="yolov8n.pt", epochs=100,
                             imgsz=640, batch=16, augmentation=False,
                             yolo_aug_params=None,
                             albumentations_transforms=None,
                             save=False, plots=False, verbose=False):
    """
    Train a YOLO model using data augmentation, and allows custom
    Albumentations transforms to be applied during training.

    Args:
        data (str): Path to the dataset YAML file containing
            train/val/test paths and class names for Ultralytics YOLO.
        model_path: Path to pre-trained weights file (e.g., 'yolov8n.pt')
            or model configuration file.
        epochs: Number of training epochs.
        imgsz (int or tuple, optional): Input image size for training.
        batch: Batch size for training.
        augmentation: Global flag to enable/disable augmentation.
        yolo_aug_params: Dictionary of YOLO's native augmentation
            parameters to customize built-in transforms.
        albumentations_transforms: List of Albumentations transform
            objects for custom augmentation pipeline. Takes precedence
            over YOLO's built-in augmentations.
        save: Whether to save training checkpoints and final model.
        plots: Whether to generate and save training plots.
        verbose: Whether to display detailed training progress.

    Returns:
        the trained yolo model and the full training output.
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

    train_params = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "save": save,
        "plots": plots,
        "verbose": verbose,
    }

    if yolo_aug_params is not None:
        train_params.update(yolo_aug_params)
    elif albumentations_transforms is None and augmentation is False:
        train_params["hsv_h"] = 0.0
        train_params["hsv_s"] = 0.0
        train_params["hsv_v"] = 0.0
        train_params["degrees"] = 0.0
        train_params["translate"] = 0.0
        train_params["scale"] = 0.0
        train_params["shear"] = 0.0
        train_params["perspective"] = 0.0
        train_params["flipud"] = 0.0
        train_params["fliplr"] = 0.0
        train_params["bgr"] = 0.0
        train_params["mosaic"] = 0.0
        train_params["mixup"] = 0.0
        train_params["cutmix"] = 0.0
        train_params["copy_paste"] = 0.0
        train_params["auto_augment"] = None
        train_params["erasing"] = 0.0

    results = model.train(**train_params)

    return model, results
