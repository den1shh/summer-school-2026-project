from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


N_SPLITS = 5
RANDOM_STATE = 42


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """5-fold stratified split. Returns list of (idx_train, None, idx_test)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    splits = []
    for idx_train, idx_test in skf.split(np.zeros(len(y)), y):
        splits.append((np.asarray(idx_train), None, np.asarray(idx_test)))
    return splits
