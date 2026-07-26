#!/usr/bin/env python3
"""Initialize an untrained boosting classifier by name"""

from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """Return an untrained boosting classifier based on the given name"""
    if name == "adaboost":
        model = ensemble.AdaBoostClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == "gradientboosting":
        model = ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
    elif name == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, random_state=random_state)
    elif name == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, random_state=random_state, verbose=-1
        )
    else:
        raise ValueError(f"Unknown model name '{name}'")
    return model
