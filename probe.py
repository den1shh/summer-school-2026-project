"""probe.py — student-implemented (SMILES-2026 submission).

ULTIMATE v10 — 2-base Brier soft-vote.

Rank analysis of OOF correlation matrix showed effective_n = 1.91 / 8 for the
prior 8-base ULTIMATE v8: all bases span only 2 independent directions of
hallucination signal. v10 keeps the 2 orthogonal representatives:
  - lookback   (attention pattern)
  - multilayer (hidden state, PCA-128)

Pipeline:
  1. 25-fold OOF per base (5x5) with isotonic-calibrated LR.
  2. Brier-optimal weights via scipy Nelder-Mead.
  3. Acc-optimal threshold on the OOF blend.

Self-contained; relies only on numpy, scipy, sklearn, torch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from aggregation import SLICE_LOOKBACK, SLICE_MULTILAYER, FEATURE_DIM

# ---------------------------------------------------------------------------

_BASES = {
    "lookback":   {"slice": SLICE_LOOKBACK,   "C": 0.003, "pca": None},
    "multilayer": {"slice": SLICE_MULTILAYER, "C": 0.01,  "pca": 128},
}

_RANDOM_STATE = 42
_N_FOLDS = 5
_N_REPEATS = 5


def _tune_threshold(y_true, y_score):
    cand = np.unique(np.concatenate([y_score, np.linspace(0, 1, 201)]))
    best_t, best_acc = 0.5, -1.0
    for t in cand:
        a = accuracy_score(y_true, (y_score >= t).astype(int))
        if a > best_acc:
            best_acc, best_t = a, float(t)
    return best_t


def _brier_optimal_weights(P_oof, y, anchor=0.0):
    n_bases = P_oof.shape[1]
    uniform = np.ones(n_bases) / n_bases
    def neg_brier(w_raw):
        w = np.abs(w_raw); w = w / (w.sum() + 1e-12)
        pred = P_oof @ w
        return ((pred - y) ** 2).mean() + anchor * ((w - uniform) ** 2).sum()
    res = minimize(neg_brier, x0=uniform, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    w = np.abs(res.x); w = w / w.sum()
    return w


class HallucinationProbe(nn.Module):
    """ULTIMATE v10: lookback + multilayer with Brier soft-vote."""

    def __init__(self) -> None:
        super().__init__()
        self._base_names = list(_BASES.keys())
        self._pcas: dict = {}
        self._base_classifiers: dict = {name: [] for name in self._base_names}
        self._weights: np.ndarray | None = None
        self._threshold: float = 0.5

    def _get_base_X(self, X, name, fit_pca_idx=None):
        cfg = _BASES[name]
        Xb = X[:, cfg["slice"]]
        if cfg.get("pca"):
            if fit_pca_idx is not None:
                pca = PCA(n_components=cfg["pca"], random_state=_RANDOM_STATE).fit(Xb[fit_pca_idx])
                self._pcas[name] = pca
            Xb = self._pcas[name].transform(Xb)
        return Xb.astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        assert X.shape[1] == FEATURE_DIM, f"unexpected feature dim {X.shape[1]} (expected {FEATURE_DIM})"
        y = y.astype(int)
        n = len(X)

        # PCA fitted once on full training fold
        for name in self._base_names:
            if _BASES[name].get("pca"):
                self._get_base_X(X, name, fit_pca_idx=np.arange(n))

        # 25-fold OOF per base with isotonic-calibrated LR
        oof = {name: np.zeros(n, dtype=np.float64) for name in self._base_names}
        oof_count = np.zeros(n, dtype=np.int32)
        for name in self._base_names:
            self._base_classifiers[name] = []

        for rep in range(_N_REPEATS):
            skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True,
                                   random_state=_RANDOM_STATE + rep)
            for tr_idx, va_idx in skf.split(np.zeros(n), y):
                for name in self._base_names:
                    cfg = _BASES[name]
                    Xb = self._get_base_X(X, name)
                    sc = StandardScaler().fit(Xb[tr_idx])
                    base = LogisticRegression(
                        C=cfg["C"], penalty="l2", solver="liblinear",
                        class_weight="balanced", max_iter=2000,
                        random_state=_RANDOM_STATE * 13 + rep,
                    )
                    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
                    clf.fit(sc.transform(Xb[tr_idx]), y[tr_idx])
                    oof[name][va_idx] += clf.predict_proba(sc.transform(Xb[va_idx]))[:, 1]
                    self._base_classifiers[name].append((sc, clf))
                oof_count[va_idx] += 1

        for name in self._base_names:
            oof[name] = oof[name] / np.maximum(oof_count, 1)

        # Brier-optimal weights
        P_oof = np.column_stack([oof[name] for name in self._base_names])
        self._weights = _brier_optimal_weights(P_oof, y, anchor=0.0)
        oof_blend = P_oof @ self._weights

        # Acc-optimal threshold
        self._threshold = _tune_threshold(y, oof_blend)
        return self

    def fit_hyperparameters(self, X_val, y_val):
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = _tune_threshold(y_val.astype(int), probs)
        return self

    def predict_proba(self, X):
        n = len(X)
        per_base_probs = []
        for name in self._base_names:
            Xb = self._get_base_X(X, name)
            preds = np.zeros(n, dtype=np.float64)
            for sc, clf in self._base_classifiers[name]:
                preds += clf.predict_proba(sc.transform(Xb))[:, 1]
            per_base_probs.append(preds / max(len(self._base_classifiers[name]), 1))
        P = np.column_stack(per_base_probs)
        prob_pos = np.clip(P @ self._weights, 0.0, 1.0)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def forward(self, x):
        return torch.zeros(x.shape[0])
