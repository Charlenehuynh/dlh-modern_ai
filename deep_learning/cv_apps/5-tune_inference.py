#!/usr/bin/env python3
"""
This module performs inference-time hyperparameter tuning to find the
optimal confidence and IoU thresholds for a trained YOLO model.
"""
import numpy as np
from ultralytics import YOLO


def tune_inference(model, val_images_path,
                    conf_thresholds=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                    iou_thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
                    imgsz=640):
    """
    Grid-searches confidence/IoU thresholds to find the best-performing
    combination on the validation set.

    Returns:
        dict with best_conf, best_iou, best_metrics, and all_results.
    """
    if isinstance(model, str):
        model = YOLO(model)

    all_results = []
    best_f1 = -1.0
    best_conf = None
    best_iou = None
    best_metrics = None

    for conf in conf_thresholds:
        for iou in iou_thresholds:
            metrics = model.val(
                data=val_images_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
                plots=False,
            )

            results_dict = metrics.results_dict
            map50 = results_dict["metrics/mAP50(B)"]
            map50_95 = results_dict["metrics/mAP50-95(B)"]
            precision = results_dict["metrics/precision(B)"]
            recall = results_dict["metrics/recall(B)"]

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            entry = {
                "conf": conf,
                "iou": iou,
                "map50": map50,
                "map50_95": map50_95,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            all_results.append(entry)

            if f1 > best_f1:
                best_f1 = f1
                best_conf = conf
                best_iou = iou
                best_metrics = {
                    "map50": map50,
                    "map50_95": map50_95,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }

    return {
        "best_conf": best_conf,
        "best_iou": best_iou,
        "best_metrics": best_metrics,
        "all_results": all_results,
    }
