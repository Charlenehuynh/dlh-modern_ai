#!/usr/bin/env python3
"""Basic YOLO-compatible data augmentation using Albumentations."""

import albumentations as A
import numpy as np


def basic_aug(image, bboxes, labels):
    """
    Applies YOLO-compatible data augmentation to an image and its
    bounding boxes.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[int]]:
            - Augmented image.
            - Augmented bounding boxes.
            - Augmented labels.
    """
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Affine(
                translate_percent=0.1,
                scale=0.1,
                rotate=[-30, 0],
                p=0.5,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
        ),
        seed=42,
    )

    augmented = transform(image=image, bboxes=bboxes, labels=labels)

    aug_image = augmented["image"]
    aug_bboxes = np.array(augmented["bboxes"])
    aug_labels = augmented["labels"]

    return aug_image, aug_bboxes, aug_labels
