#!/usr/bin/env python3
"""Custom YOLO-compatible data augmentation using Albumentations-exclusive
transforms."""

import albumentations as A
import numpy as np


def custom_aug(image, bboxes, labels):
    """
    Applies Albumentations-exclusive data augmentation to an image and its
    bounding boxes.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[int]]:
            - Augmented image.
            - Augmented bounding boxes.
            - Augmented labels.
    """
    transform = A.Compose(
        [
            A.MotionBlur(blur_limit=5, p=0.9),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
                    A.OpticalDistortion(distort_limit=0.05, p=0.2),
                ],
                p=0.9,
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
