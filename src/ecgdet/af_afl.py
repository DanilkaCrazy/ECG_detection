"""Inference wrapper for the specialised AF-vs-AFL binary head.

Supports two backends:
- ``dl``     : CNN1D/ResNet1D sigmoid checkpoint from ``train_af_afl_full_dl``
- ``classic``: joblib model with advanced features from ``train_af_afl_imbalance``

Example
-------
>>> head = AFAFLClassifier()                 # auto-detect best available model
>>> label, prob_af = head.predict(sig_12x2500)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ecgdet import config
from ecgdet.features_advanced import extract_advanced_features

__all__ = ["AFAFLClassifier"]

ROOT = Path(__file__).resolve().parents[2]
FULL = config.PROCESSED_ROOT / "af_afl_full"
SMALL = config.PROCESSED_ROOT / "af_afl"


class AFAFLClassifier:
    """AF (1) vs AFL (0) predictor on raw 12-lead ECG signals."""

    def __init__(self, backend: str = "auto", path: str | Path | None = None):
        import joblib
        import torch

        self.backend = backend
        if backend == "auto":
            classic = ROOT / "reports" / "models" / "af_afl_imbalance" / "best_model.joblib"
            dl = ROOT / "reports" / "models" / "af_afl_full_dl"
            if classic.exists():
                backend = "classic"
            elif any(dl.glob("*.pt")):
                backend = "dl"
            else:
                small = ROOT / "reports" / "models" / "af_afl"
                if any(small.glob("*.pt")):
                    backend = "dl"
                    path = sorted(small.glob("CNN1D_aug0.pt"))[0]
                else:
                    raise FileNotFoundError("no AF/AFL model found — train one first")
        self.backend = backend

        if backend == "classic":
            bundle = joblib.load(path or ROOT / "reports" / "models" / "af_afl_imbalance" / "best_model.joblib")
            self.model = bundle["model"]
            self.threshold = float(bundle["threshold"])
            prep_path = ROOT / "reports" / "models" / "af_afl_imbalance" / "preprocessing.joblib"
            self.prep = joblib.load(prep_path) if prep_path.exists() else None
            names = FULL / "feature_names.csv"
            self.feature_names = (
                __import__("pandas").read_csv(names)["name"].tolist() if names.exists() else None
            )
        elif backend == "dl":
            import torch
            from ecgdet import normalize
            from ecgdet.models import build_model

            if path is None:
                cands = sorted((ROOT / "reports" / "models" / "af_afl_full_dl").glob("*.pt"))
                if not cands:
                    raise FileNotFoundError("no DL checkpoint found")
                path = cands[0]
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.model = build_model(ck.get("model_class", "CNN1D"), num_classes=2,
                                     in_channels=12, binary=True)
            self.model.load_state_dict(ck["model_state"])
            self.model.eval()
            self.threshold = float(ck.get("threshold", 0.5))
            norm_path = FULL / "norm_fitted.joblib"
            if norm_path.exists():
                self.norm = joblib.load(norm_path)
            else:
                Xtr = np.load(SMALL / "X_train.npy").astype(np.float32)
                cfg = normalize.RobustNormConfig(method=ck.get("norm_method", "median_mad"))
                self.norm = normalize.fit_robust_norm(np.transpose(Xtr, (0, 2, 1)), cfg)
            self.norm_method = ck.get("norm_method", "median_mad")
            self._torch = torch
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def predict_proba(self, sig: np.ndarray) -> float:
        """Return P(AF) for a single signal of shape (12, n_samples)."""
        sig = np.asarray(sig, dtype=np.float32)
        if sig.ndim != 2:
            raise ValueError(f"expected (12, samples), got {sig.shape}")
        if sig.shape[0] < 12:
            sig = np.pad(sig, ((0, 12 - sig.shape[0]), (0, 0)))
        sig = sig[:12]
        if sig.shape[1] < 2500:
            sig = np.pad(sig, ((0, 0), (0, 2500 - sig.shape[1])))
        sig = sig[:, :2500]
        sig = np.nan_to_num(sig)

        if self.backend == "classic":
            feats = extract_advanced_features(sig.T, fs=500.0)
            keys = self.feature_names if self.feature_names is not None else sorted(feats.keys())
            row = np.array([[float(feats.get(k, np.nan)) if np.isfinite(feats.get(k, np.nan)) else np.nan
                             for k in keys]])
            if self.prep is not None:
                prep = self.prep
                allnan = prep.get("allnan_cols", np.array([], dtype=int))
                keep = np.array([j for j in range(len(keys)) if j not in set(allnan.tolist())])
                row = row[:, keep]
                fill = prep["median_fill"]
                row = np.where(np.isnan(row), fill, row)
                row = np.clip((row - prep["center"]) / prep["scale"], -20, 20)
            return float(self.model.predict_proba(row)[0, 1])

        from ecgdet import normalize
        x3 = np.transpose(sig, (1, 0))[None, ...]  # (1, S, L)
        xn = normalize.apply_robust_norm(x3, self.norm)
        xn = np.transpose(xn, (0, 2, 1)).astype(np.float32)
        t = self._torch.tensor(xn, dtype=self._torch.float32)
        with self._torch.no_grad():
            logit = self.model(t).view(-1).item()
        return float(1.0 / (1.0 + np.exp(-logit)))

    def predict(self, sig: np.ndarray) -> tuple[str, float]:
        """Return (label, P(AF)); label in {'AF', 'AFL'}."""
        p = self.predict_proba(sig)
        return ("AF" if p >= self.threshold else "AFL"), p
