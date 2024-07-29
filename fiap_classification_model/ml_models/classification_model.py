import numpy as np

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

    def run_model(self, model: object, test_split: float = 0.30) -> list:
        """
        This is the simple run method so it exptects the model
        object already with all relevant hyperarameters
        """
        X = normalize_data(self._X)
        y = encode_labels(self._y)
        X_train, X_test, y_train, y_test = splitter(
            features=X, classes=y, test_size=0.3
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result_list = self.calculate_results(y_pred, y_test, model)
        return result_list

    def calculate_results(
        self, y_pred: np.ndarray, y_test: np.ndarray, model: object
    ) -> list:
        result: list = []
        result.append(model.__str__())
        result.append(str(model.__class__()).replace("()", ""))
        result.append(round(accuracy_score(y_pred, y_test), 4))
        result.append(round(precision_score(y_pred, y_test), 4))
        result.append(round(recall_score(y_pred, y_test), 4))
        result.append(round(mean_absolute_percentage_error(y_pred, y_test), 4))
        return result
