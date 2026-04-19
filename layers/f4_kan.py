"""
Feature 4 – Kolmogorov-Arnold Network (KAN) Interpretable Decision Layer
Learns B-spline activation functions per edge — readable as curves.
Input features: [heading_dev, speed_norm, fft_peak, topology_score,
                 time_of_day_norm, surprise_score]
Output: P(wrong-way) with human-readable decision graph.
"""

import numpy as np
import torch
from pathlib import Path

KAN_WEIGHTS_PATH = Path(__file__).parent.parent / "data" / "kan_weights.pt"

# ── Minimal pure-PyTorch KAN ─────────────────────────────────────────────────
# Using a lightweight BSpline-based KAN layer that doesn't need the full pykan
# GUI (which requires matplotlib interaction). We implement the core math.

class BSplineActivation(torch.nn.Module):
    """Trainable B-spline activation on a 1-D input."""
    def __init__(self, n_knots=5, degree=3):
        super().__init__()
        self.n_knots = n_knots
        self.degree = degree
        self.knots = torch.linspace(-1, 1, n_knots + degree + 1)
        self.coeffs = torch.nn.Parameter(torch.randn(n_knots) * 0.1)

    def forward(self, x):
        # Cox-de Boor recursion (simplified: use RBF basis as proxy)
        centers = torch.linspace(-1, 1, self.n_knots).to(x.device)
        sigma = 0.4
        basis = torch.exp(-((x.unsqueeze(-1) - centers) ** 2) / (2 * sigma ** 2))
        return (basis * self.coeffs.to(x.device)).sum(-1)

    def get_curve(self, n=50):
        """Return (x_vals, y_vals) for plotting the learned activation."""
        xs = torch.linspace(-1, 1, n)
        with torch.no_grad():
            ys = self.forward(xs)
        return xs.numpy(), ys.numpy()


class KANLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, n_knots=5):
        super().__init__()
        self.activations = torch.nn.ModuleList([
            BSplineActivation(n_knots) for _ in range(in_features * out_features)
        ])
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # x: [batch, in_features]
        outputs = []
        for j in range(self.out_features):
            out_j = torch.zeros(x.shape[0])
            for i in range(self.in_features):
                act = self.activations[i * self.out_features + j]
                xi = x[:, i].clamp(-1, 1)
                out_j = out_j + act(xi)
            outputs.append(out_j)
        return torch.stack(outputs, dim=1)  # [batch, out_features]


class KANClassifier(torch.nn.Module):
    """[6 features] → [4 hidden] → [1 output]"""
    def __init__(self, in_dim=6, hidden_dim=4):
        super().__init__()
        self.layer1 = KANLayer(in_dim, hidden_dim)
        self.layer2 = KANLayer(hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        h = self.layer1(x)
        out = self.layer2(h)
        return self.sigmoid(out)

    def get_edge_curves(self, feature_names):
        """Extract all learned B-spline curves for visualization."""
        curves = {}
        for j in range(self.layer1.out_features):
            for i, fname in enumerate(feature_names):
                act = self.layer1.activations[i * self.layer1.out_features + j]
                xs, ys = act.get_curve()
                curves[f"{fname}→h{j}"] = (xs, ys)
        return curves


FEATURE_NAMES = [
    "heading_dev", "speed_norm", "fft_peak",
    "topology_score", "time_of_day", "surprise_score",
]


def _extract_features(l1_res, l3_res, topo_res, surprise_res, t):
    """Build a [6] feature vector from pipeline outputs at time t."""
    results = l1_res.get("results", [])
    if t < len(results):
        row = results[t]
        heading_dev = min(row["road_delta"] / 180.0, 1.0)
        speed_norm = min(row["speed"] / 150.0, 1.0)
    else:
        heading_dev, speed_norm = 0.0, 0.5

    fft_peak = min(l3_res.get("dominant_freq", 0.0) * 4, 1.0)
    topology_score = topo_res.get("topology_score", 0.0) if topo_res else 0.0
    time_of_day = (t % 86400) / 86400.0
    surprise = min(surprise_res.get("current_surprise", 0.0) / 40.0, 1.0) \
        if surprise_res else 0.0

    return np.array([heading_dev, speed_norm, fft_peak,
                     topology_score, time_of_day, surprise], dtype=np.float32)


def build_training_dataset(all_l1, all_l3, all_topo, labels):
    """Build feature matrix from a list of pipeline results."""
    X, y = [], []
    for l1, l3, topo, label in zip(all_l1, all_l3, all_topo, labels):
        results = l1.get("results", [])
        n = len(results)
        for t in range(0, n, 10):
            feat = _extract_features(l1, l3, topo, None, t)
            X.append(feat)
            y.append(1.0 if label == "wrongway" else 0.0)
    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1),
    )


def train_kan(X, y, epochs=60, lr=2e-3):
    model = KANClassifier(in_dim=6, hidden_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    losses = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        Xs, ys = X[perm], y[perm]
        for i in range(0, len(Xs), 32):
            Xb, yb = Xs[i:i+32], ys[i:i+32]
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        losses.append(loss.item())
    torch.save(model.state_dict(), KAN_WEIGHTS_PATH)
    return model, losses


def load_or_train_kan(X=None, y=None, force_retrain=False):
    model = KANClassifier(in_dim=6, hidden_dim=4)
    if not force_retrain and KAN_WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(KAN_WEIGHTS_PATH, weights_only=True))
    elif X is not None and y is not None:
        model, _ = train_kan(X, y)
    model.eval()
    return model


def kan_predict(model, l1, l3, topo, surprise_res, t):
    feat = _extract_features(l1, l3, topo, surprise_res, t)
    x = torch.tensor(feat[np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        prob = float(model(x).squeeze())
    curves = model.get_edge_curves(FEATURE_NAMES)
    return {
        "probability": round(prob, 4),
        "features": dict(zip(FEATURE_NAMES, feat.tolist())),
        "edge_curves": curves,
        "decision": "WRONG-WAY" if prob > 0.5 else "NORMAL",
    }
