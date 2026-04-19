from __future__ import annotations

import json
import os
from typing import Any

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium


API_URL = os.environ.get("SWARMMIND_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="SwarmMind Toolbox3 Panel",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
body, .stApp { background: #080b10; color: #d7dde8; }
section[data-testid="stSidebar"] { background: #0d1118; border-right: 1px solid #1c2635; }
.toolbox-title { font-size: 1.55rem; font-weight: 800; letter-spacing: .04em; color: #f4f7fb; }
.subtle { color: #78869a; font-size: .82rem; }
.status-normal { background: linear-gradient(90deg,#06391f,#0b5a35); color:#8fffc1; }
.status-caution { background: linear-gradient(90deg,#3f2f03,#7c5c06); color:#ffe08a; }
.status-wrong { background: linear-gradient(90deg,#4a1018,#8e1729); color:#ff9aaa; animation:pulse 1s infinite alternate; }
.statusbar { border:1px solid #263348; border-radius:14px; padding:16px 18px; font-size:1.1rem; font-weight:800; text-align:center; }
.metric-card { background:#0f1622; border:1px solid #263348; border-radius:14px; padding:14px; }
.metric-card strong { font-size:1.4rem; color:#f4f7fb; }
.layer-table table { font-size: .86rem; }
.timeline-item { border-left:2px solid #ff395d; padding: 0 0 12px 12px; margin-left:8px; }
.timeline-time { color:#ff91a4; font-weight:700; font-size:.8rem; }
@keyframes pulse { from { box-shadow: 0 0 0 rgba(255,57,93,.2);} to { box-shadow:0 0 24px rgba(255,57,93,.35);} }
</style>
""",
    unsafe_allow_html=True,
)


def api_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{API_URL}{path}", timeout=5)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any], stream: bool = False):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=None if stream else 10, stream=stream)
    response.raise_for_status()
    return response


def status_class(status: str) -> str:
    if "WRONG" in status:
        return "status-wrong"
    if "CAUTION" in status:
        return "status-caution"
    return "status-normal"


def render_map(state: dict[str, Any], event: dict[str, Any] | None = None):
    vehicles = event.get("vehicles", []) if event else state.get("vehicles", [])
    alerts = state.get("alerts", [])
    if vehicles:
        center = [vehicles[0]["lat"], vehicles[0]["lon"]]
    elif alerts:
        center = [alerts[-1]["lat"], alerts[-1]["lon"]]
    else:
        center = [33.9206, -118.3462]

    fmap = folium.Map(location=center, zoom_start=14, tiles="CartoDB dark_matter", prefer_canvas=True)
    for vehicle in vehicles:
        color = "red" if "wrong" in vehicle.get("behavior", "") else "blue"
        folium.Marker(
            location=[vehicle["lat"], vehicle["lon"]],
            icon=folium.Icon(color=color, icon="car", prefix="glyphicon"),
            tooltip=f"{vehicle['id']} · {vehicle['behavior']} · {vehicle['speed']} km/h",
        ).add_to(fmap)

    for alert in alerts[-10:]:
        folium.Circle(
            location=[alert["lat"], alert["lon"]],
            radius=float(alert.get("v2x_radius_km", 3.0)) * 1000,
            color="#ff395d",
            fill=True,
            fill_opacity=0.08,
            weight=2,
            tooltip=f"{alert['type']} · conf={alert['confidence']}",
        ).add_to(fmap)
        folium.CircleMarker(
            location=[alert["lat"], alert["lon"]],
            radius=7,
            color="#ff395d",
            fill=True,
            fill_opacity=0.9,
        ).add_to(fmap)
    return fmap


def render_layers(layers: list[dict[str, Any]]):
    rows = []
    for layer in layers:
        status = layer["status"]
        color = {
            "active": "🟢",
            "idle": "⚪",
            "suppressed": "🟡",
            "failed": "🔴",
        }.get(status, "⚪")
        rows.append({
            "Layer": layer["layer"],
            "Confidence": layer["confidence"],
            "Status": f"{color} {status}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_alert_timeline(alerts: list[dict[str, Any]]):
    if not alerts:
        st.caption("No alerts yet. Run a wrong-way trace to populate the timeline.")
        return
    for alert in reversed(alerts[-25:]):
        st.markdown(
            f"""
<div class="timeline-item">
  <div class="timeline-time">t={alert['time']}s · confidence={alert['confidence']}</div>
  <div><strong>{alert['type']}</strong></div>
  <div class="subtle">{alert['location']} · V2X radius {alert.get('v2x_radius_km', 3.0)} km</div>
</div>
""",
            unsafe_allow_html=True,
        )


try:
    state = api_get("/state")
    config = api_get("/config")
    csvs = api_get("/csvs")["csvs"]
    api_online = True
except Exception as exc:
    api_online = False
    state = {"status": "API OFFLINE", "vehicles": [], "alerts": [], "layers": [], "time": 0.0}
    config = {}
    csvs = []
    st.error(f"Simulation API is not reachable at `{API_URL}`. Start it with `uvicorn simulation_engine:app --reload`. Details: {exc}")


with st.sidebar:
    st.markdown('<div class="toolbox-title">SwarmMind Toolbox3</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Diagnostic simulation panel · API controlled</div>', unsafe_allow_html=True)
    st.divider()
    selected_csv = st.selectbox("CSV Trace Player", csvs or ["wrong_way.csv"])
    delay_ms = st.slider("Playback delay per frame", 0, 250, 35, 5)
    max_seconds = st.slider("Demo length", 30, 300, 90, 10)
    play = st.button("▶ Play Trace", type="primary", use_container_width=True, disabled=not api_online)
    refresh = st.button("↻ Refresh State", use_container_width=True, disabled=not api_online)
    st.divider()
    st.markdown("### Config Editor")
    lnn_threshold = st.slider("LNN threshold", 0.0, 1.0, float(config.get("lnn_threshold", 0.30)), 0.01)
    ode_threshold = st.slider("ODE threshold", 0.0, 1.0, float(config.get("ode_threshold", 0.05)), 0.01)
    tti_threshold = st.slider("TTI threshold", 0.0, 1.0, float(config.get("tti_threshold", 0.55)), 0.01)
    wrong_way_delta = st.slider("Wrong-way heading delta", 30.0, 150.0, float(config.get("wrong_way_delta", 90.0)), 1.0)
    min_wrong_steps = st.slider("Min wrong-way steps", 1, 60, int(config.get("min_wrong_steps", 8)), 1)
    kan_weights = config.get("kan_weights", {})
    st.markdown("#### KAN Weights")
    edited_weights = {}
    for key, value in kan_weights.items():
        edited_weights[key] = st.slider(key, -1.0, 1.0, float(value), 0.01)
    if st.button("Save Config", use_container_width=True, disabled=not api_online):
        config = api_post(
            "/config",
            {
                "lnn_threshold": lnn_threshold,
                "ode_threshold": ode_threshold,
                "tti_threshold": tti_threshold,
                "wrong_way_delta": wrong_way_delta,
                "min_wrong_steps": min_wrong_steps,
                "kan_weights": edited_weights,
            },
        ).json()
        st.success("Config saved.")


st.markdown('<div class="toolbox-title">SwarmMind Safety Net · Diagnostic Panel</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Tesla-Toolbox3-inspired simulation cockpit for wrong-way detection, 9-layer inspection, and V2X alert replay.</div>', unsafe_allow_html=True)

status = state.get("status", "NORMAL")
st.markdown(f'<div class="statusbar {status_class(status)}">{status}</div>', unsafe_allow_html=True)

top1, top2, top3, top4 = st.columns(4)
top1.markdown(f'<div class="metric-card"><div class="subtle">Simulation Time</div><strong>{state.get("time", 0):.1f}s</strong></div>', unsafe_allow_html=True)
top2.markdown(f'<div class="metric-card"><div class="subtle">Vehicles</div><strong>{len(state.get("vehicles", []))}</strong></div>', unsafe_allow_html=True)
top3.markdown(f'<div class="metric-card"><div class="subtle">Alerts</div><strong>{len(state.get("alerts", []))}</strong></div>', unsafe_allow_html=True)
top4.markdown(f'<div class="metric-card"><div class="subtle">Active CSV</div><strong>{state.get("active_csv") or "none"}</strong></div>', unsafe_allow_html=True)

map_slot, inspector_slot = st.columns([1.55, 1.0])
event_placeholder = None

if play and api_online:
    map_box = map_slot.empty()
    layers_box = inspector_slot.empty()
    alerts_box = inspector_slot.empty()
    response = api_post(
        "/playback",
        {"csv_filename": selected_csv, "delay_ms": delay_ms, "max_seconds": max_seconds},
        stream=True,
    )
    current_state = state
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        payload = json.loads(raw_line.removeprefix("data: "))
        if payload.get("event") == "playback-complete":
            break
        current_state = api_get("/state")
        with map_box.container():
            st.markdown("### Central Map")
            st_folium(render_map(current_state, payload), height=560, use_container_width=True)
        with layers_box.container():
            st.markdown("### 9-Layer Inspector")
            render_layers(payload.get("layers", []))
        with alerts_box.container():
            st.markdown("### Alert Log Timeline")
            render_alert_timeline(current_state.get("alerts", []))
    st.success("Playback complete.")
    state = api_get("/state")
else:
    with map_slot:
        st.markdown("### Central Map")
        st_folium(render_map(state), height=560, use_container_width=True)
    with inspector_slot:
        st.markdown("### 9-Layer Inspector")
        render_layers(state.get("layers", []))
        st.markdown("### Alert Log Timeline")
        render_alert_timeline(state.get("alerts", []))

st.divider()
with st.expander("Raw /state JSON"):
    st.json(state)
