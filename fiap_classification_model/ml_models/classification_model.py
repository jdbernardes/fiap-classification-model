import numpy as nd
from typing import List

# pre processing imports
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    mean_absolute_percentage_error,
)


class ClassificationModel:
    def __init__(self, X, y) -> None:
        self._X = X
        self._y = y
        self._SEED = 42

    def normalize(self) -> nd.ndarray:
        X = normalize(self._X)
        return X

    def categorize(self) -> nd.ndarray:
        l_encoder = LabelEncoder()
        y = l_encoder.fit_transform(self._y)
        return y

    def regular_spliter(self, percent: float = 0.30) -> List[nd.ndarray]:
        X = self.normalize()
        y = self.categorize()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=percent, random_state=self._SEED
        )
        return X_train, X_test, y_train, y_test

    def calculate_results(self, y_pred: nd.ndarray, y_test: nd.ndarray) -> list:
        result: list = []
        result.append(round(accuracy_score(y_pred, y_test), 4))
        result.append(round(precision_score(y_pred, y_test), 4))
        result.append(round(recall_score(y_pred, y_test), 4))
        result.append(round(mean_absolute_percentage_error(y_pred, y_test), 4))
        return result

    def run_model(self, model: object, test_split: float = 0.30) -> list:
        """
        This is the simple run method so it exptects the model
        object already with all relevant hyperarameters
        """
        X_train, X_test, y_train, y_test = self.regular_spliter(test_split)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        result_list = self.calculate_results(y_pred, y_test)
        return result_list
