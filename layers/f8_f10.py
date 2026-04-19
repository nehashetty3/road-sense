"""
Feature 8 – Hyperbolic GNN (Poincaré Ball Road Embeddings)
Feature 10 – World Model / RSSM (100-futures prediction)
"""

# ═══════════════════════════════════════════════════════════════
# FEATURE 8 – HYPERBOLIC ROAD EMBEDDINGS (POINCARÉ BALL)
# ═══════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json

try:
    import geoopt
    from geoopt import PoincareBall
    HAS_GEOOPT = True
except Exception:
    geoopt = None
    PoincareBall = None
    HAS_GEOOPT = False

HYPER_WEIGHTS_PATH = Path(__file__).parent.parent / "data" / "hyper_weights.pt"
RSSM_WEIGHTS_PATH = Path(__file__).parent.parent / "data" / "rssm_weights.pt"


class EuclideanBallFallback:
    def expmap0(self, x):
        return torch.tanh(x)

    def dist(self, z1, z2):
        return torch.norm(z1 - z2)


class HyperbolicRoadEncoder(nn.Module):
    """
    Encodes road segment features into Poincaré disk (2-D).
    Motorways → near centre (large hierarchy subtree).
    Local roads → near boundary.
    Wrong-way events → off-cluster anomaly.
    """
    def __init__(self, in_dim=5, hidden=16, out_dim=2):
        super().__init__()
        self.ball = PoincareBall() if HAS_GEOOPT else EuclideanBallFallback()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        euclidean = self.fc2(h)
        # Map to hyperbolic space via exponential map at origin
        norm = euclidean.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # Constrain to Poincaré disk (radius < 1)
        safe = torch.tanh(norm) * euclidean / norm
        return self.ball.expmap0(safe * 0.85)  # keep inside disk

    def poincare_distance(self, z1, z2):
        return self.ball.dist(z1, z2)


def _segment_features(geo_features):
    """Extract feature vectors from GeoJSON segments."""
    feats, meta = [], []
    for feat in geo_features:
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        if not coords:
            continue
        mid = coords[len(coords) // 2]
        highway = props.get("highway", "secondary")
        hw_enc = {"motorway": 1.0, "primary": 0.7, "secondary": 0.4,
                  "cycleway": 0.1, "footway": 0.05}.get(highway, 0.3)
        lanes = props.get("lanes", 1) / 4.0
        oneway = 1.0 if props.get("oneway", False) else 0.0
        construction = 1.0 if props.get("construction", False) else 0.0
        direction_norm = props.get("direction", 180.0) / 360.0
        feats.append([hw_enc, lanes, oneway, construction, direction_norm])
        meta.append({
            "lat": mid[1], "lon": mid[0],
            "highway": highway,
            "oneway": props.get("oneway", False),
            "construction": props.get("construction", False),
        })
    return np.array(feats, dtype=np.float32), meta


def train_hyperbolic_encoder(geo_features, epochs=80, lr=2e-3):
    """Self-supervised: similar road types should be close in hyperbolic space."""
    feats, meta = _segment_features(geo_features)
    if len(feats) < 4:
        model = HyperbolicRoadEncoder()
        return model, []

    model = HyperbolicRoadEncoder(in_dim=5)
    opt = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr) if HAS_GEOOPT else torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    X = torch.tensor(feats, dtype=torch.float32)
    # Labels: highway class (motorway=0, primary=1, secondary=2, etc.)
    hw_labels = np.array([m["highway"] for m in meta])
    unique_hw = list(set(hw_labels))
    label_idx = np.array([unique_hw.index(h) for h in hw_labels])

    for ep in range(epochs):
        opt.zero_grad()
        z = model(X)
        # Contrastive loss: same type → close, different → far
        loss = torch.tensor(0.0)
        n = len(z)
        if n > 1:
            for i in range(min(n, 8)):
                for j in range(min(n, 8)):
                    if i == j:
                        continue
                    d = model.poincare_distance(z[i], z[j])
                    same = 1.0 if label_idx[i] == label_idx[j] else 0.0
                    # Pull same-type together, push different apart
                    loss = loss + same * d + (1 - same) * torch.clamp(1.0 - d, min=0)
        loss = loss / max(1, min(n, 8) ** 2)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    torch.save(model.state_dict(), HYPER_WEIGHTS_PATH)
    return model, losses


def load_or_train_hyperbolic(geo_features, force_retrain=False):
    model = HyperbolicRoadEncoder(in_dim=5)
    if not force_retrain and HYPER_WEIGHTS_PATH.exists():
        model.load_state_dict(torch.load(HYPER_WEIGHTS_PATH, weights_only=True))
    else:
        model, _ = train_hyperbolic_encoder(geo_features)
    model.eval()
    return model


def get_hyperbolic_embeddings(model, geo_features):
    """Return 2-D Poincaré disk coordinates for all road segments."""
    feats, meta = _segment_features(geo_features)
    if len(feats) == 0:
        return {"points": [], "disk_points": []}

    X = torch.tensor(feats, dtype=torch.float32)
    with torch.no_grad():
        z = model(X).numpy()

    points = []
    for i, m in enumerate(meta):
        points.append({
            "lat": m["lat"], "lon": m["lon"],
            "disk_x": round(float(z[i, 0]), 5),
            "disk_y": round(float(z[i, 1]), 5),
            "highway": m["highway"],
            "construction": m["construction"],
            "radius": round(float(np.linalg.norm(z[i])), 5),
        })

    return {"points": points, "disk_raw": z.tolist()}


def embed_vehicle_event(model, heading, speed, road_direction):
    """Embed a single vehicle observation in hyperbolic space."""
    heading_dev = abs(heading - road_direction) % 360
    heading_dev = min(heading_dev, 360 - heading_dev) / 180.0
    feat = np.array([[0.5, 0.5, 1.0, 0.0, heading_dev]], dtype=np.float32)
    X = torch.tensor(feat, dtype=torch.float32)
    with torch.no_grad():
        z = model(X).numpy()[0]
    return {"disk_x": float(z[0]), "disk_y": float(z[1]),
            "radius": float(np.linalg.norm(z))}


# ═══════════════════════════════════════════════════════════════
# FEATURE 10 – WORLD MODEL / RSSM FUTURE SAMPLING
# ═══════════════════════════════════════════════════════════════

class SimpleRSSM(nn.Module):
    """
    Recurrent State Space Model.
    Learns latent vehicle dynamics; samples N plausible futures.
    """
    def __init__(self, state_dim=32, obs_dim=4):
        super().__init__()
        self.state_dim = state_dim

        # Recurrent prior: state → next state
        self.transition = nn.GRUCell(obs_dim, state_dim)

        # Decoder: state → predicted GPS observation mean + log_std
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, obs_dim * 2),
        )

        # Encoder: observation → state mean + log_std
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim * 2),
        )

    def encode(self, obs):
        params = self.encoder(obs)
        mean, log_std = params.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-4, 2))
        z = mean + std * torch.randn_like(std)
        return z, mean, std

    def imagine_rollout(self, initial_obs, steps=12, n_samples=20):
        """Sample N plausible futures from current observation."""
        obs = torch.tensor(initial_obs, dtype=torch.float32).unsqueeze(0)
        state, _, _ = self.encode(obs)

        futures = []
        for _ in range(n_samples):
            s = state + torch.randn_like(state) * 0.15
            traj = []
            for t in range(steps):
                params = self.decoder(s)
                mean, log_std = params.chunk(2, dim=-1)
                std = torch.exp(log_std.clamp(-4, 2))
                sample = mean + std * torch.randn_like(std)
                traj.append(sample.squeeze().detach().numpy())
                s = self.transition(mean.detach(), s.detach())
            futures.append(np.array(traj))
        return np.array(futures)  # [n_samples, steps, obs_dim]


def train_rssm(normal_df, epochs=50, lr=1e-3):
    arr = normal_df[["lat", "lon", "heading", "speed"]].values.astype(np.float32)
    mins = arr.min(axis=0)
    ranges = np.where(arr.max(axis=0) - mins < 1e-8, 1.0, arr.max(axis=0) - mins)
    arr_norm = (arr - mins) / ranges

    model = SimpleRSSM(state_dim=32, obs_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    N = len(arr_norm)
    for ep in range(epochs):
        # Pick random 15-step sequence
        start = np.random.randint(0, max(1, N - 16))
        seq = torch.tensor(arr_norm[start: start + 15], dtype=torch.float32)
        opt.zero_grad()
        state = torch.zeros(1, 32)
        ep_loss = 0.0
        for t in range(len(seq) - 1):
            obs = seq[t].unsqueeze(0)
            state, mean_z, _ = model.encode(obs)
            params = model.decoder(state)
            mean_obs, log_std_obs = params.chunk(2, dim=-1)
            std_obs = torch.exp(log_std_obs.clamp(-4, 2))
            # Gaussian log-likelihood
            nll = 0.5 * ((seq[t + 1] - mean_obs.squeeze()) ** 2 /
                         std_obs.squeeze() ** 2 + torch.log(std_obs.squeeze() ** 2)).mean()
            ep_loss += nll
            state = model.transition(obs, state)
        ep_loss /= max(1, len(seq) - 1)
        ep_loss.backward()
        opt.step()
        losses.append(float(ep_loss))

    torch.save({
        "weights": model.state_dict(),
        "mins": mins.tolist(),
        "ranges": ranges.tolist(),
    }, RSSM_WEIGHTS_PATH)
    return model, mins, ranges, losses


def load_or_train_rssm(normal_df, force_retrain=False):
    model = SimpleRSSM(state_dim=32, obs_dim=4)
    if not force_retrain and RSSM_WEIGHTS_PATH.exists():
        ckpt = torch.load(RSSM_WEIGHTS_PATH, weights_only=True)
        model.load_state_dict(ckpt["weights"])
        mins = np.array(ckpt["mins"])
        ranges = np.array(ckpt["ranges"])
        model.eval()
        return model, mins, ranges

    model, mins, ranges, _ = train_rssm(normal_df)
    model.eval()
    return model, mins, ranges


def sample_futures(model, mins, ranges, df, current_t, n_samples=20, steps=12):
    """
    Sample N futures from the RSSM at time current_t.
    Returns lat/lon trajectories in geo-space.
    """
    arr = df[["lat", "lon", "heading", "speed"]].values.astype(np.float32)
    arr_norm = (arr - mins) / ranges

    t = min(current_t, len(arr_norm) - 1)
    initial_obs = arr_norm[t]

    futures_norm = model.imagine_rollout(initial_obs, steps=steps, n_samples=n_samples)
    # Denormalise lat/lon only
    futures_geo = futures_norm[:, :, :2] * ranges[:2] + mins[:2]

    # Check collision: do any futures approach the wrong-way corridor?
    # Heuristic: heading component of future should diverge from origin
    future_headings = futures_norm[:, :, 2] * ranges[2] + mins[2]
    road_dir = float(df["heading"].values[:30].mean())

    collision_futures = 0
    for fi in range(n_samples):
        final_h = float(future_headings[fi, -1])
        delta = abs(final_h - road_dir) % 360
        delta = min(delta, 360 - delta)
        # Also flag futures whose lat/lon diverge significantly
        start_pos = futures_geo[fi, 0]
        end_pos   = futures_geo[fi, -1]
        pos_div   = abs(end_pos[0] - start_pos[0]) + abs(end_pos[1] - start_pos[1])
        if delta > 70 or pos_div > 0.002:
            collision_futures += 1

    collision_fraction = collision_futures / n_samples

    return {
        "futures_geo": futures_geo.tolist(),  # [n_samples, steps, 2]
        "futures_heading": future_headings.tolist(),
        "n_samples": n_samples,
        "steps": steps,
        "collision_fraction": round(collision_fraction, 3),
        "collision_count": collision_futures,
        "current_pos": arr[t, :2].tolist(),
        "origin_lat": float(arr[t, 0]),
        "origin_lon": float(arr[t, 1]),
    }
