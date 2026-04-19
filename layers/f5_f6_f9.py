"""
Feature 5 – Spectral Vulnerability (Fiedler Vector Criticality)
Feature 6 – Neural ODE Trajectory Interpolation
Feature 9 – Information-Theoretic Surprise Score (KDE)
"""

# ═══════════════════════════════════════════════════════════════
# FEATURE 5 – SPECTRAL VULNERABILITY
# ═══════════════════════════════════════════════════════════════

import numpy as np
import networkx as nx
from scipy.stats import gaussian_kde
import torch
import torch.nn as nn
from pathlib import Path
import json

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except Exception:
    HAS_TORCHDIFFEQ = False

    def odeint(model, y0, t_span, method="euler"):
        states = [y0]
        for i in range(1, len(t_span)):
            dt = t_span[i] - t_span[i - 1]
            next_state = states[-1] + dt * model(t_span[i - 1], states[-1])
            states.append(next_state)
        return torch.stack(states, dim=0)

ODE_WEIGHTS_PATH = Path(__file__).parent.parent / "data" / "ode_weights.pt"
KDE_PATH = Path(__file__).parent.parent / "data" / "kde_models.npz"


def build_road_graph_from_geojson(geo_features):
    """Build a NetworkX directed graph from GeoJSON road features."""
    G = nx.DiGraph()
    for feat in geo_features:
        if feat["properties"].get("cycleway"):
            continue
        coords = feat["geometry"]["coordinates"]
        for i in range(len(coords) - 1):
            n1 = tuple(coords[i])
            n2 = tuple(coords[i + 1])
            G.add_node(n1, lon=coords[i][0], lat=coords[i][1])
            G.add_node(n2, lon=coords[i + 1][0], lat=coords[i + 1][1])
            dist = ((n2[0] - n1[0]) ** 2 + (n2[1] - n1[1]) ** 2) ** 0.5
            construction = feat["properties"].get("construction", False)
            G.add_edge(n1, n2, weight=dist,
                       lanes=feat["properties"].get("lanes", 1),
                       construction=construction)
    return G


def compute_spectral_vulnerability(geo_features):
    """
    Compute Fiedler vector for road graph.
    High |fiedler_component| = critical node — wrong-way here is catastrophic.
    Returns node criticality scores and global algebraic connectivity.
    """
    G = build_road_graph_from_geojson(geo_features)
    if len(G) < 4:
        return {"ac": 0.0, "node_criticality": {}, "critical_segments": [],
                "max_criticality": 0.0}

    Gu = G.to_undirected()
    # Ensure connected
    if not nx.is_connected(Gu):
        Gu = Gu.subgraph(max(nx.connected_components(Gu), key=len)).copy()

    try:
        ac = nx.algebraic_connectivity(Gu, method="tracemin_lu")
        fv = nx.fiedler_vector(Gu, method="tracemin_lu")
    except Exception:
        nodes = list(Gu.nodes())
        ac = 0.1
        fv = np.random.randn(len(nodes))

    nodes = list(Gu.nodes())
    node_crit = {nodes[i]: float(abs(fv[i])) for i in range(len(nodes))}

    # Identify top 20% critical segments
    crit_vals = np.array(list(node_crit.values()))
    threshold = np.percentile(crit_vals, 80) if len(crit_vals) > 0 else 0.5
    critical = [
        {"node": str(n), "lat": n[1], "lon": n[0],
         "criticality": round(v, 5)}
        for n, v in node_crit.items()
        if v >= threshold
    ]

    return {
        "ac": round(float(ac), 6),
        "node_criticality": {str(k): v for k, v in node_crit.items()},
        "critical_segments": critical[:30],
        "max_criticality": round(float(crit_vals.max()), 5) if len(crit_vals) > 0 else 0.0,
        "fiedler_vector": fv.tolist(),
    }


def get_segment_criticality(lat, lon, spectral_result, default=1.0):
    """Find criticality score for a given (lat, lon) position."""
    if not spectral_result["node_criticality"]:
        return default
    best_d, best_v = float("inf"), default
    for node_str, v in spectral_result["node_criticality"].items():
        try:
            n = eval(node_str)  # (lon, lat) tuple
            d = abs(lat - n[1]) + abs(lon - n[0])
            if d < best_d:
                best_d, best_v = d, v
        except Exception:
            continue
    return best_v


def adjusted_severity(base_confidence, criticality):
    """Scale alert severity by road criticality."""
    return min(1.0, base_confidence * (1 + criticality * 1.5))


# ═══════════════════════════════════════════════════════════════
# FEATURE 6 – NEURAL ODE TRAJECTORY INTERPOLATION
# ═══════════════════════════════════════════════════════════════

class VehicleDynamicsODE(nn.Module):
    """Learns the vector field of normal vehicle motion in state space."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),   # [lat_norm, lon_norm, heading_norm, speed_norm]
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4),
        )

    def forward(self, t, state):
        return self.net(state)


def _normalize_state(df):
    arr = df[["lat", "lon", "heading", "speed"]].values.astype(np.float32)
    mins = arr.min(axis=0)
    ranges = arr.max(axis=0) - mins
    ranges = np.where(ranges < 1e-8, 1.0, ranges)
    return arr, mins, ranges


def train_ode(normal_df, epochs=40, lr=1e-3):
    """Train ODE on normal traffic to learn the motion manifold."""
    arr, mins, ranges = _normalize_state(normal_df)
    arr_norm = (arr - mins) / ranges

    model = VehicleDynamicsODE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    N = len(arr_norm) - 1
    for ep in range(epochs):
        # Pick random 10-step windows
        starts = np.random.randint(0, max(1, N - 12), size=8)
        ep_loss = 0.0
        for s in starts:
            window = arr_norm[s: s + 11]
            y0 = torch.tensor(window[0], dtype=torch.float32)
            t_span = torch.linspace(0, 1, 11)
            target = torch.tensor(window, dtype=torch.float32)
            opt.zero_grad()
            pred = odeint(model, y0, t_span, method="euler")
            loss = nn.MSELoss()(pred, target)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / 8)

    torch.save({
        "weights": model.state_dict(),
        "mins": mins.tolist(),
        "ranges": ranges.tolist(),
    }, ODE_WEIGHTS_PATH)
    return model, mins, ranges, losses


def load_or_train_ode(normal_df, force_retrain=False):
    model = VehicleDynamicsODE()
    if not force_retrain and ODE_WEIGHTS_PATH.exists():
        ckpt = torch.load(ODE_WEIGHTS_PATH, weights_only=True)
        model.load_state_dict(ckpt["weights"])
        mins = np.array(ckpt["mins"])
        ranges = np.array(ckpt["ranges"])
        model.eval()
        return model, mins, ranges

    model, mins, ranges, _ = train_ode(normal_df)
    model.eval()
    return model, mins, ranges


def ode_reconstruction_error(model, mins, ranges, df, window=20):
    """
    For each GPS window, reconstruct trajectory via ODE.
    High MSE = off-manifold = anomalous motion.
    """
    arr = df[["lat", "lon", "heading", "speed"]].values.astype(np.float32)
    arr_norm = (arr - mins) / ranges
    errors = []
    with torch.no_grad():
        for i in range(0, len(arr_norm) - window, window // 2):
            seg = arr_norm[i: i + window]
            y0 = torch.tensor(seg[0], dtype=torch.float32)
            t_sp = torch.linspace(0, 1, window)
            pred = odeint(model, y0, t_sp, method="euler")
            target = torch.tensor(seg, dtype=torch.float32)
            mse = float(nn.MSELoss()(pred, target))
            errors.append({
                "step": i,
                "mse": round(mse, 6),
                "anomalous": mse > 0.05,
                "pred": pred.numpy().tolist(),
                "target": target.numpy().tolist(),
            })

    max_err = max(e["mse"] for e in errors) if errors else 0.0
    return {
        "errors": errors,
        "max_mse": round(max_err, 6),
        "anomalous_windows": sum(1 for e in errors if e["anomalous"]),
        "total_windows": len(errors),
        "manifold_score": round(1.0 - min(1.0, max_err / 0.15), 4),
    }


# ═══════════════════════════════════════════════════════════════
# FEATURE 9 – INFORMATION-THEORETIC SURPRISE SCORE
# ═══════════════════════════════════════════════════════════════

class SurpriseScoreModel:
    """
    KDE-based surprise score per road segment.
    Fits density on normal headings & speeds.
    Wrong-way: extremely low density → high bits of surprise.
    """
    def __init__(self):
        self.kde = None
        self.fitted = False

    def fit(self, normal_df):
        headings = normal_df["heading"].values.astype(float)
        speeds = normal_df["speed"].values.astype(float)
        data = np.vstack([headings / 360.0, speeds / 150.0])
        self.kde = gaussian_kde(data, bw_method=0.25)
        self.fitted = True

    def surprise_bits(self, heading, speed):
        """Returns bits of surprise: normal ≈ 0-3, wrong-way ≈ 15-40."""
        if not self.fitted:
            return 0.0
        pt = np.array([[heading / 360.0], [speed / 150.0]])
        p = float(self.kde.evaluate(pt)[0])
        p = max(p, 1e-12)
        return -np.log2(p)

    def score_trace(self, df, window=10):
        """Score full trace, accumulating surprise in rolling window."""
        headings = df["heading"].values
        speeds = df["speed"].values
        scores, accumulated = [], 0.0
        for i, (h, s) in enumerate(zip(headings, speeds)):
            bits = self.surprise_bits(h, s)
            accumulated = accumulated * 0.9 + bits  # exponential decay accumulator
            wrong_way = accumulated > 35.0
            scores.append({
                "t": int(df["time"].values[i]),
                "surprise_bits": round(bits, 3),
                "accumulated": round(accumulated, 3),
                "anomalous": wrong_way,
                "current_surprise": round(bits, 3),
            })
        total_anomalous = sum(1 for s in scores if s["anomalous"])
        return {
            "scores": scores,
            "max_accumulated": round(max(s["accumulated"] for s in scores), 2),
            "total_anomalous": total_anomalous,
            "anomaly_rate": round(total_anomalous / max(1, len(scores)), 4),
            "current_surprise": scores[-1]["surprise_bits"] if scores else 0.0,
        }
