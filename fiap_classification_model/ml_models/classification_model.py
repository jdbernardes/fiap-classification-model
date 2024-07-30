from typing import List
import numpy as np
import csv
import os

# Metrics processing imports
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    mean_absolute_percentage_error,
)

# Utils
from fiap_classification_model.utils.ml_utils import (
    normalize_data,
    encode_labels,
    splitter,
)


class ClassificationModel:
    def __init__(self, X, y) -> None:
        self._X = X
        self._y = y
        self._SEED = 42

    def train_model(self, model: object, test_split: float = 0.30) -> List[any]:
        """
        This is the simple run method so it exptects the model
        object already with all relevant hyperarameters
        """
        X = normalize_data(self._X)
        y = encode_labels(self._y)
        X_train, X_test, y_train, y_test = splitter(
            features=X, classes=y, test_size=test_split
        )
        model.fit(X_train, y_train)
        return [model, X_test, y_test]

    def run_predict(
        self, model: object, X_test: np.ndarray, y_test: np.ndarray, result_path: str
    ) -> None:
        y_pred = model.predict(X_test)
        self.calculate_results(y_pred, y_test, model, result_path)

    def calculate_results(
        self, y_pred: np.ndarray, y_test: np.ndarray, model: object, result_path: str
    ) -> None:
        field_names = ["Model", "Accuracy", "Precision", "Recall", "MAPE"]
        row = {
            "Model": str(model.__class__()).replace("()", ""),
            "Accuracy": round(accuracy_score(y_pred, y_test), 4),
            "Precision": round(precision_score(y_pred, y_test), 4),
            "Recall": round(recall_score(y_pred, y_test), 4),
            "MAPE": round(mean_absolute_percentage_error(y_pred, y_test), 4),
        }
        if os.path.isfile(result_path):
            with open(result_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writerow(row)
        else:
            with open(result_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writeheader()
                writer.writerow(row)
