import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold


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


def cross_val_k_folders(mode: str = "str", k: int = 5, seed: int = 42):
    """
    mode: accepts "str" for strafified, "full" for notmal k fold
    k: Number of folds, standard is 5
    seed: random see, standard is 42
    """
    if mode.lower() == "str":
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return skf
    elif mode.lower() == "full":
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        return kf
