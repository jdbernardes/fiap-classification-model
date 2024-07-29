import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def normalize_data(norm_data: pd.DataFrame) -> np.ndarray:
    return normalize(norm_data)


def encode_labels(classes: np.ndarray) -> np.ndarray:
    l_encoder = LabelEncoder()
    y = l_encoder.fit_transform(classes)
    return y


def splitter(
    features: np.ndarray, classes: np.ndarray, test_size: float = 0.3, seed: int = 42
) -> List[np.ndarray]:
    X = features
    y = classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_train, X_test, y_train, y_test
