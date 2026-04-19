"""
Feature 1 – Liquid Neural Network (LNN) Adaptive Classifier
Replaces DBSCAN heading consensus with a Liquid Time-Constant (LTC) RNN
that adapts its internal weights continuously with incoming GPS pings.
No retraining needed per-segment — the ODE dynamics self-adjust.
"""

import numpy as np
import random
import torch
import torch.nn as nn
from pathlib import Path

try:
    from ncps.torch import LTC
    from ncps import wirings
    HAS_NCPS = True
except Exception:
    LTC = None
    wirings = None
    HAS_NCPS = False

WEIGHTS_PATH = Path(__file__).parent.parent / "data" / "lnn_weights.pt"
_LNN_CACHE = {}


class LNNClassifier(nn.Module):
    def __init__(self, input_size=6, units=19):
        super().__init__()
        self.uses_liquid_core = HAS_NCPS
        if HAS_NCPS:
            wiring = wirings.AutoNCP(units=units, output_size=1)
            self.temporal = LTC(input_size, wiring, batch_first=True)
        else:
            # Graceful demo fallback when ncps is unavailable.
            self.temporal = nn.GRU(input_size, units, batch_first=True)
            self.proj = nn.Linear(units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hx=None):
        # x: [batch, seq_len, input_size]
        out, hx = self.temporal(x, hx)
        if self.uses_liquid_core:
            logits = out[:, -1, :]
        else:
            logits = self.proj(out[:, -1, :])
        return self.sigmoid(logits), hx  # last step output


def _heading_delta(h1, h2):
    diff = abs(float(h1) - float(h2)) % 360
    return min(diff, 360 - diff)


def _build_feature_array(df, road_direction):
    arr = df[["lat", "lon", "speed", "time"]].values.astype(np.float32)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = np.where(maxs - mins < 1e-8, 1.0, maxs - mins)
    arr_norm = (arr - mins) / ranges

    headings = df["heading"].values.astype(np.float32)
    road_delta = np.array([_heading_delta(h, road_direction) / 180.0 for h in headings], dtype=np.float32)
    turn_rate = np.abs(np.diff(headings, prepend=headings[0]))
    turn_rate = np.minimum(turn_rate, 360.0 - turn_rate) / 180.0
    return np.column_stack([arr_norm, road_delta, turn_rate]).astype(np.float32)


def _build_training_data(df, road_direction, window=30, stride=5):
    """Sliding window sequences from a GPS trace dataframe."""
    arr_norm = _build_feature_array(df, road_direction)

    # Label: wrong-way = heading delta > 90° from road direction
    def hd(h):
        d = abs(float(h) - road_direction) % 360
        return min(d, 360 - d)

    X, y = [], []
    for i in range(0, len(arr_norm) - window, stride):
        seq = arr_norm[i: i + window]
        raw_h = df["heading"].values[i + window - 1]
        label = 1.0 if hd(raw_h) > 90 else 0.0
        X.append(seq)
        y.append(label)

    if not X:
        return None, None
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1),
    )


def _set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_lnn(traces_dict, road_directions, epochs=25, lr=3e-3, seed=None, save_weights=True):
    """
    traces_dict: {loc_key: {"normal": df, "wrongway": df, ...}}
    road_directions: {loc_key: float}
    Returns trained model.
    """
    _set_seed(seed)
    model = LNNClassifier(input_size=6, units=19)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    all_X, all_y = [], []
    for loc_key, tdict in traces_dict.items():
        rd = road_directions.get(loc_key, 180.0)
        for ttype, df in tdict.items():
            X, y = _build_training_data(df, rd)
            if X is not None:
                all_X.append(X)
                all_y.append(y)

    if not all_X:
        return model

    X_all = torch.cat(all_X)
    y_all = torch.cat(all_y)

    # Shuffle
    idx = torch.randperm(len(X_all))
    X_all, y_all = X_all[idx], y_all[idx]

    model.train()
    losses = []
    for ep in range(epochs):
        # mini-batch
        batch_sz = min(32, len(X_all))
        bi = torch.randint(0, len(X_all), (batch_sz,))
        Xb, yb = X_all[bi], y_all[bi]
        optimizer.zero_grad()
        pred, _ = model(Xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if save_weights:
        try:
            torch.save(model.state_dict(), WEIGHTS_PATH)
        except Exception:
            pass
    return model, losses


def load_or_train_lnn(traces_dict, road_directions, force_retrain=False, seed=None, save_weights=True):
    if not force_retrain and WEIGHTS_PATH.exists():
        model = LNNClassifier(input_size=6, units=19)
        try:
            model.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
            model.eval()
            return model, []
        except Exception:
            force_retrain = True

    model, losses = train_lnn(traces_dict, road_directions, seed=seed, save_weights=save_weights)
    model.eval()
    return model, losses


def lnn_predict_trace(model, df, window=30, threshold=0.30, road_direction=None):
    """
    Run LNN on a trace, returning per-step P(wrong-way) scores.
    """
    if road_direction is None:
        road_direction = float(np.median(df["heading"].values))
    arr_norm = _build_feature_array(df, road_direction)

    scores = []
    model.eval()
    with torch.no_grad():
        hx = None
        for i in range(len(arr_norm)):
            start = max(0, i - window + 1)
            seq = arr_norm[start: i + 1]
            # Pad if needed
            if len(seq) < window:
                pad = np.zeros((window - len(seq), seq.shape[1]), dtype=np.float32)
                seq = np.vstack([pad, seq])
            x = torch.tensor(seq[np.newaxis], dtype=torch.float32)
            prob, hx = model(x, hx)
            scores.append(float(prob.squeeze()))

    return {
        "scores": scores,
        "max_score": max(scores),
        "mean_score": float(np.mean(scores)),
        "wrong_way_steps": sum(1 for s in scores if s > threshold),
        "threshold": threshold,
        "backend": "LTC" if getattr(model, "uses_liquid_core", False) else "GRU fallback",
    }
