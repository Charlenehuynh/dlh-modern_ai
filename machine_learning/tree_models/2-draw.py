#!/usr/bin/env python3

"""displays the textual structure of tree"""

from sklearn import tree


def draw(clf, feature_names, class_names):
    """
    clf: A trained DecisionTreeClassifier instance from Scikit-learn
    feature_names: A list of the input feature names
    class_names: A list of the target class names
    """
    tree_text = tree.export_text(
        clf,
        feature_names=list(feature_names),
        class_names=list(class_names),
    )
    print(tree_text)
