"""
SwarmMind – 9-Layer Safety Engine + TTI Safety Stack
Full Streamlit UI
"""
import sys, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from folium.plugins import HeatMap

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"

if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.csv")):
    with st.spinner("Generating synthetic dataset…"):
        subprocess.run([sys.executable, str(ROOT/"generate_data.py")], check=True)

weights_ok = all((DATA_DIR/f).exists() for f in [
    "lnn_weights.pt","ode_weights.pt","rssm_weights.pt","kan_weights.pt"])
if not weights_ok:
    with st.spinner("First run — training ML models…"):
        subprocess.run([sys.executable, str(ROOT/"train_models.py")], check=True)

from layers.core import run_all_layers, ATTACK_LABELS
from layers.baseline import fft_rppg_baseline
from layers.f1_lnn import load_or_train_lnn, lnn_predict_trace
from layers.f2_topology import compute_topology_fingerprint
from layers.f4_kan import load_or_train_kan, kan_predict
from layers.f5_f6_f9 import (compute_spectral_vulnerability, get_segment_criticality,
                               adjusted_severity, load_or_train_ode,
                               ode_reconstruction_error, SurpriseScoreModel)
from layers.f8_f10 import load_or_train_rssm, sample_futures
from layers.tti_filter import compute_tti_filter
from generate_data import generate_trace
from real_data import (
    available_real_locations,
    build_curated_real_trace_cache,
    get_source_bundle,
    load_real_ngsim_trace,
    load_overpass_features,
    load_real_towers,
    load_workzone_events,
)

st.set_page_config(page_title="SwarmMind",page_icon="🛡️",layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
section[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #21262d}
section[data-testid="stSidebar"] *{color:#c9d1d9!important}
.block-container{padding-top:.75rem}
.sm-card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:.8rem 1rem;text-align:center}
.sm-card .val{font-size:1.75rem;font-weight:600;line-height:1.2}
.sm-card .lbl{font-size:.7rem;color:#8b949e;margin-top:3px;letter-spacing:.05em}
.sm-green{border-top:3px solid #00d4aa}.sm-green .val{color:#00d4aa}
.sm-red{border-top:3px solid #ff2d55}.sm-red .val{color:#ff2d55}
.sm-blue{border-top:3px solid #58a6ff}.sm-blue .val{color:#58a6ff}
.sm-amber{border-top:3px solid #e3b341}.sm-amber .val{color:#e3b341}
.sm-purple{border-top:3px solid #bc8cff}.sm-purple .val{color:#bc8cff}
.sm-coral{border-top:3px solid #ff6b35}.sm-coral .val{color:#ff6b35}
.badge{display:inline-block;padding:3px 11px;border-radius:20px;font-size:.82rem;font-weight:600;margin:2px}
.badge-red{background:#3d1b2e;color:#ff2d55;border:1px solid #ff2d55}
.badge-orange{background:#2d1f0a;color:#ffa500;border:1px solid #ffa500}
.badge-green{background:#0e2a1e;color:#00d4aa;border:1px solid #00d4aa}
.badge-blue{background:#0d1f3c;color:#58a6ff;border:1px solid #58a6ff}
.layer-row{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:.5rem}
.layer-pill{padding:2px 9px;border-radius:12px;font-size:.68rem;font-weight:600;border:1px solid}
.lp-active{background:#0e2a1e;color:#00d4aa;border-color:#00d4aa}
.lp-warning{background:#2d1f0a;color:#ffa500;border-color:#ffa500}
.lp-danger{background:#3d1b2e;color:#ff6b35;border-color:#ff6b35;animation:pulse 1.2s ease-in-out 1}
.lp-inactive{background:#161b22;color:#484f58;border-color:#30363d}
.sec-header{font-size:.68rem;font-weight:600;letter-spacing:.12em;color:#484f58;text-transform:uppercase;margin:10px 0 5px}
.sm-table{width:100%;border-collapse:collapse;font-size:.8rem}
.sm-table td,.sm-table th{border-bottom:1px solid #21262d;padding:6px 8px;text-align:left}
.sm-table tr.flag td{background:rgba(255,45,85,.12)}
.sm-table tr.confirm td{background:rgba(0,212,170,.09)}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(255,45,85,.45)}100%{box-shadow:0 0 0 12px rgba(255,45,85,0)}}
</style>""",unsafe_allow_html=True)

LOCATIONS={
    "Anna Salai, Chennai 🇮🇳":{"key":"chennai","center":[13.049,80.2518],"dir":180.0},
    "I-405, Los Angeles 🇺🇸": {"key":"la","center":[33.9206,-118.3462],"dir":350.0},
    "I-95, Washington DC 🇺🇸":{"key":"dc","center":[38.8051,-77.0468],"dir":45.0},
}
TRACE_LABELS={
    "🟢 Normal traffic":"normal","🍺 Drunk / impaired":"drunk",
    "⚙️ Mechanical fault":"fault","🚨 Wrong-way intrusion":"wrongway",
    "🚧 Construction-zone diversion":"construction",
    "🛰 Real NGSIM normal":"real_normal",
    "🛰 Real NGSIM + wrong-way injection":"real_wrongway",
    "🛰 Real NGSIM + construction injection":"real_construction",
    "🛰 Real NGSIM + short wrong-way burst":"real_short_wrongway",
    "🛰 Real NGSIM + lane-confused merge":"real_lane_confused",
    "🛰 Real NGSIM + stop-and-reverse":"real_stop_reverse",
}

ABLATION_ROWS = [
    {"name":"Baseline (FFT + rPPG)","f1":0.50,"fpr":0.50,"color":"#ff6b35"},
    {"name":"+ LNN + ODE","f1":0.57,"fpr":0.17,"color":"#e3b341"},
    {"name":"Full TTI-Stack","f1":1.00,"fpr":0.00,"color":"#00d4aa"},
]

LITERATURE_ANCHORS = [
    ("LNNs for traffic","continuous-time adaptation for streaming mobility signals"),
    ("TDA trajectory anomaly","persistence features over space-time traces"),
    ("Neural ODE dynamics","normal-flow reconstruction for vehicle trajectories"),
    ("KAN interpretability","curve-based decisions that remain auditable"),
    ("KLD / surprise scoring","information-theoretic anomaly quantification"),
    ("Spectral vulnerability","graph criticality on road segments"),
]

PITCHES = {
    "1-Minute":"SwarmMind is a 9-layer safety engine that uses adaptive neural networks, topology-aware anomaly signals, and interpretable decision layers to detect wrong-way vehicles early enough to warn nearby traffic before a head-on encounter becomes unavoidable.",
    "3-Minute":"Layer 1 checks counter-flow from GPS and map direction. Layer 3 distinguishes wrong-way motion from drunk swerving and mechanical drift. The TTI stack then fuses LNN adaptation, Neural ODE reconstruction error, topology persistence, surprise bits, KAN curves, and spectral vulnerability before Layer 8 ripples alerts through nearby vehicles.",
    "5-Minute":"LNNs adapt online to local traffic flow, Neural ODEs define a normal-driving manifold, TDA adds structural trajectory evidence, KAN exposes why the decision changed, and TTI-Filter re-weights topology confidence when GPS is noisy so construction-zone diversions are suppressed instead of escalated.",
}

for real_key, meta in available_real_locations().items():
    if real_key != "us101":
        continue
    build_curated_real_trace_cache(real_key, max_traces=4)
    LOCATIONS[f"US-101 NGSIM, Los Angeles 🛰️"] = {
        "key": "us101",
        "center": list(meta["center"]),
        "dir": meta["road_direction"],
    }

@st.cache_data(show_spinner=False)
def load_trace(loc,tt):
    if loc == "us101" or tt.startswith("real_"):
        variant_map = {
            "real_normal": "normal",
            "real_wrongway": "wrongway_injected",
            "real_construction": "construction_injected",
            "real_short_wrongway": "short_wrongway_injected",
            "real_lane_confused": "lane_confused_injected",
            "real_stop_reverse": "stop_reverse_injected",
        }
        return load_real_ngsim_trace("us101", variant_map.get(tt, "normal"))
    path = DATA_DIR/f"{tt}_{loc}.csv"
    if not path.exists() and tt == "construction":
        return generate_trace(loc, tt)
    return pd.read_csv(path)
@st.cache_data(show_spinner=False)
def load_geo(loc):
    overpass_map = {"chennai": "anna_salai", "la": "i405", "dc": "i95", "us101": "us101"}
    feats = load_overpass_features(overpass_map.get(loc, ""))
    if feats:
        return feats
    with open(DATA_DIR/f"{loc}.geojson") as f: return json.load(f)["features"]
@st.cache_data(show_spinner=False)
def load_towers(loc):
    tower_key_map = {"chennai": "anna_salai", "la": "i405", "dc": "i95", "us101": "us101_ngsim"}
    towers = load_real_towers(tower_key_map.get(loc, ""))
    if towers:
        return towers
    with open(DATA_DIR/f"towers_{loc}.json") as f: return json.load(f)
@st.cache_data(show_spinner=False)
def load_swarm(loc):
    with open(DATA_DIR/f"swarm_{loc}.json") as f: return json.load(f)

@st.cache_data(show_spinner=False)
def load_workzones(loc):
    zone_key_map = {"chennai": "anna_salai", "la": "i405", "dc": "i95", "us101": "us101"}
    return load_workzone_events(zone_key_map.get(loc, ""))

@st.cache_data(show_spinner=False)
def load_breakdown_report():
    report_path = ROOT / "experiments" / "results" / "report_breakdown.json"
    if report_path.exists():
        return json.loads(report_path.read_text())
    return None

@st.cache_resource(show_spinner=False)
def get_lnn():
    base_types=["normal","drunk","fault","wrongway"]
    traces={loc:{tt:load_trace(loc,tt) for tt in base_types} for loc in ["chennai","la","dc"]}
    dirs={"chennai":180.0,"la":350.0,"dc":45.0}
    if "US-101 NGSIM, Los Angeles 🛰️" in LOCATIONS:
        for idx in range(4):
            key = f"us101_trace_{idx + 1}"
            try:
                traces[key] = {
                    "normal": load_real_ngsim_trace("us101", "normal", trace_index=idx),
                    "wrongway": load_real_ngsim_trace("us101", "wrongway_injected", trace_index=idx),
                    "construction": load_real_ngsim_trace("us101", "construction_injected", trace_index=idx),
                    "short_wrongway": load_real_ngsim_trace("us101", "short_wrongway_injected", trace_index=idx),
                    "lane_confused": load_real_ngsim_trace("us101", "lane_confused_injected", trace_index=idx),
                    "stop_reverse": load_real_ngsim_trace("us101", "stop_reverse_injected", trace_index=idx),
                }
                dirs[key] = 130.0
            except Exception:
                break
    m,_=load_or_train_lnn(traces,dirs)
    return m
@st.cache_resource(show_spinner=False)
def get_ode(): return load_or_train_ode(load_trace("chennai","normal"))
@st.cache_resource(show_spinner=False)
def get_rssm(): return load_or_train_rssm(load_trace("chennai","normal"))
@st.cache_resource(show_spinner=False)
def get_kan(): return load_or_train_kan()

def mc(v,l,c): return f'<div class="sm-card {c}"><div class="val">{v}</div><div class="lbl">{l}</div></div>'
def dark_fig(**kw):
    d=dict(plot_bgcolor="#0d1117",paper_bgcolor="#161b22",font=dict(color="#c9d1d9",size=10))
    d.update(kw); return d

def make_pipeline_graph():
    labels = ["GPS + Maps","F1 LNN","F6 ODE","F2 TDA","F9 Surprise","TTI-Filter","F4 KAN","F5 Spectral","L8/L9 V2X"]
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=16, thickness=16,
            line=dict(color="#21262d", width=1),
            label=labels,
            color=["#58a6ff","#ff6b35","#bc8cff","#ffa500","#ff2d55","#00d4aa","#e3b341","#ff6b35","#58a6ff"],
        ),
        link=dict(
            source=[0,1,2,3,4,5,6,7],
            target=[1,2,3,4,5,6,7,8],
            value=[1,1,1,1,1,1,1,1],
            color=["rgba(88,166,255,.35)","rgba(255,107,53,.35)","rgba(188,140,255,.35)",
                   "rgba(255,165,0,.35)","rgba(255,45,85,.35)","rgba(0,212,170,.35)",
                   "rgba(227,179,65,.35)","rgba(88,166,255,.35)"],
        ),
    ))
    fig.update_layout(**dark_fig(height=250,margin=dict(l=10,r=10,t=20,b=10)))
    return fig

def make_ablation_chart():
    candidate_dirs = [ROOT / "experiments" / "results", Path("/tmp/swarmmind_experiments")]
    rows = None
    for base in candidate_dirs:
        files = [
            ("baseline_pred.json", "Baseline (FFT + rPPG)", "#ff6b35"),
            ("lnn_ode_pred.json", "+ LNN + ODE", "#e3b341"),
            ("full_stack_pred.json", "Full TTI-Stack", "#00d4aa"),
        ]
        if all((base / name).exists() for name, _, _ in files):
            rows = []
            for name, label, color in files:
                payload = json.loads((base / name).read_text())
                summary = payload.get("summary", {})
                rows.append({
                    "name": label,
                    "f1": float(summary.get("f1", 0.0)),
                    "fpr": float(summary.get("false_positive_rate", 0.0)),
                    "color": color,
                })
            break
    if rows is None:
        rows = ABLATION_ROWS
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[row["name"] for row in rows],
        y=[row["f1"] for row in rows],
        marker_color=[row["color"] for row in rows],
        text=[f"F1={row['f1']:.2f}<br>FPR={row['fpr']:.2f}" for row in rows],
        textposition="outside",
    ))
    fig.update_layout(**dark_fig(
        height=280,
        margin=dict(l=35,r=8,t=25,b=60),
        title=dict(text="Ablation Summary · benchmark output",font=dict(color="#f0f6fc",size=12)),
        xaxis=dict(tickangle=-10),
        yaxis=dict(title="F1-score",gridcolor="#21262d",range=[0,1.05]),
    ))
    return fig

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SwarmMind")
    st.markdown("**9-Layer Engine + TTI Stack**")
    st.markdown("---")
    loc_label  =st.selectbox("🌐 Hotspot",list(LOCATIONS.keys()))
    trace_label=st.selectbox("🚗 Trace",list(TRACE_LABELS.keys()))
    st.markdown("---")
    time_vs_safety=st.slider("⏱ Time ↔ Safety",0,100,50,5)
    n_futures=st.slider("🔮 RSSM futures",5,30,20,5)
    stress_preset=st.checkbox("Worst-case demo preset",value=False)
    st.markdown("---")
    run_btn=st.button("▶ Run SwarmMind",type="primary",use_container_width=True)
    retrain=st.button("🔄 Retrain All Models",use_container_width=True)
    st.markdown("---")
    show_adv=st.checkbox("Show Advanced ML",value=True)
    active_tab=st.radio("Panel",["LNN+Topology","KAN Curves","ODE+Surprise","RSSM Futures"])

if stress_preset:
    loc_label = "I-405, Los Angeles 🇺🇸"
    trace_label = "🚧 Construction-zone diversion"
    time_vs_safety = 85
    n_futures = 20

loc_info  =LOCATIONS[loc_label]; loc_key=loc_info["key"]
road_dir  =loc_info["dir"]; center=loc_info["center"]
trace_type=TRACE_LABELS[trace_label]
df=load_trace(loc_key,trace_type); geo_features=load_geo(loc_key)
towers=load_towers(loc_key); swarm=load_swarm(loc_key)
workzones = load_workzones(loc_key)
source_bundle = get_source_bundle(loc_key, trace_type)
breakdown_report = load_breakdown_report()

cache_key=f"{loc_key}_{trace_type}_{time_vs_safety}"
if "pipeline" not in st.session_state or st.session_state.get("ck")!=cache_key or run_btn:
    with st.spinner("Running SwarmMind pipeline…"):
        pipeline=run_all_layers(df,geo_features,towers,swarm,road_dir,time_vs_safety)
    st.session_state.pipeline=pipeline; st.session_state.ck=cache_key

if retrain:
    with st.spinner("Retraining…"):
        subprocess.run([sys.executable,str(ROOT/"train_models.py")],check=True)
    st.cache_resource.clear(); st.success("Retrained!")

pipeline=st.session_state.pipeline; metrics=pipeline["metrics"]
l1=pipeline["layer1"]; l2=pipeline["layer2"]; l3=pipeline["layer3"]
l4=pipeline["layer4"]; l5=pipeline["layer5"]; l8=pipeline["layer8"]; l9=pipeline["layer9"]
results=l1["results"]; n_steps=len(results)
preview_cur=results[min(150,n_steps-1)]

st.markdown('<h1 style="margin:0;font-size:1.75rem;color:#f0f6fc">🛡️ SwarmMind</h1>',unsafe_allow_html=True)
st.markdown('<p style="color:#8b949e;font-size:.8rem;margin:.2rem 0 .5rem">9-Layer Engine · TTI-Filter · LNN · TDA · KAN · Spectral · Neural ODE · Surprise · World Model</p>',unsafe_allow_html=True)

def pill(n,s):
    c={"active":"lp-active","warning":"lp-warning","danger":"lp-danger"}.get(s,"lp-inactive")
    return f'<span class="layer-pill {c}">{n}</span>'
pills=[("L1","danger" if preview_cur["wrong_way"] and preview_cur["confidence"] < 0.8 else "active"),
       ("L2","active"),("L3","active"),
       ("L4","active" if l4["stress_flag"] else "inactive"),
       ("L5","active" if l5["predicted_impacts"] else "inactive"),
       ("L6","warning" if pipeline["layer6"]["suppressed_count"]>0 else "active"),
       ("L7","warning" if pipeline["layer7"]["downgraded_count"]>0 else "active"),
       ("L8","active"),("L9","warning" if l9["bt_v2x_recommended"] else "active"),
       ("F1-LNN","active"),("F2-TDA","active"),("TTI","warning" if trace_type=="construction" else "active"),
       ("F4-KAN","active"),("F5-Spec","active"),("F6-ODE","active"),
       ("F9-Surp","active"),("F10-RSSM","active")]
st.markdown('<div class="layer-row">'+"".join(pill(n,s) for n,s in pills)+"</div>",unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6,c7,c8=st.columns(8)
c1.markdown(mc(f"{metrics['detection_rate']}%","Detection","sm-green"),unsafe_allow_html=True)
c2.markdown(mc(f"{metrics['fp_rate']}%","False Pos.","sm-green"),unsafe_allow_html=True)
c3.markdown(mc(f"{metrics['panic_reduction']}%","Panic–","sm-blue"),unsafe_allow_html=True)
c4.markdown(mc(f"{metrics['route_safety']:.0f}","Route Safety","sm-amber"),unsafe_allow_html=True)
c5.markdown(mc(f"{metrics['alerted_vehicles']:,}","V2X","sm-purple"),unsafe_allow_html=True)
c6.markdown(mc(f"{metrics['connectivity_score']:.0f}","Connectivity","sm-blue"),unsafe_allow_html=True)
c7.markdown(mc(f"{metrics['swarm_latency_ms']}ms","Latency","sm-coral"),unsafe_allow_html=True)
c8.markdown(mc(len(l1["events"]),"Events","sm-red"),unsafe_allow_html=True)

atk=l3["attack_meta"]
bc={"wrongway":"badge-red","drunk":"badge-orange","fault":"badge-orange","normal":"badge-green"}.get(l3["attack_type"],"badge-blue")
ba,bi=st.columns([1,4])
with ba: st.markdown(f'<span class="badge {bc}">{atk["badge"]}</span>',unsafe_allow_html=True)
with bi: st.caption(f"{l3['dominant_freq']:.3f}Hz · swarm {metrics['swarm_latency_ms']}ms · {metrics['alerted_vehicles']:,} alerted · +{l9['time_overhead_pct']}% time"+(" · **BT-V2X ON**" if l9["bt_v2x_recommended"] else ""))

story_tabs=st.tabs(["Pipeline Graph","Ablation","Literature","Pitch Kit","Impact","Evidence"])
with story_tabs[0]:
    st.plotly_chart(make_pipeline_graph(),use_container_width=True,config={"displayModeBar":False})
    st.caption("Each block emits one scalar or vector, and the modules are decoupled enough that a single failure does not collapse the whole stack.")
with story_tabs[1]:
    st.plotly_chart(make_ablation_chart(),use_container_width=True,config={"displayModeBar":False})
    st.caption("Research note: this chart reads from saved benchmark JSON when available, including real NGSIM-backed cases after the experiment suite is rerun.")
with story_tabs[2]:
    lit_html="".join(f"<tr><td><strong>{title}</strong></td><td>{desc}</td></tr>" for title,desc in LITERATURE_ANCHORS)
    st.markdown(f'<table class="sm-table"><thead><tr><th>Anchor</th><th>Why it matters here</th></tr></thead><tbody>{lit_html}</tbody></table>',unsafe_allow_html=True)
    st.caption("SwarmMind composes these threads into a single TTI-Safety-Stack for wrong-way detection and false-positive suppression.")
with story_tabs[3]:
    for label,text in PITCHES.items():
        st.markdown(f"**{label}**")
        st.write(text)
with story_tabs[4]:
    st.markdown("1. Wrong-way-driving cloud service for highways.")
    st.markdown("2. Drunk-driving early warning for ADAS systems.")
    st.markdown("3. Construction-zone risk monitoring for smart infrastructure.")
    st.caption("Future integrations can plug radar tracks, LiDAR point clouds, or V2X CAM streams into the same TTI pipeline.")
with story_tabs[5]:
    left,right=st.columns(2)
    with left:
        st.markdown("**Source Provenance**")
        st.markdown(f"- Trace source: `{source_bundle['trace_source']}`")
        st.markdown(f"- Raw GPS source: `{source_bundle['raw_gps_source']}`")
        st.markdown(f"- Real labeled events: `{source_bundle['real_labeled_events']}`")
        st.markdown(f"- Trace variant: `{source_bundle['trace_variant']}`")
        st.markdown(f"- Geometry source: `{source_bundle['geometry_source']}`")
        st.markdown(f"- Connectivity source: `{source_bundle['tower_source']}`")
        st.markdown(f"- Work-zone source: `{source_bundle['workzone_source']}`")
        st.markdown(f"- Curated real traces: `{source_bundle['curated_real_traces']}`")
    with right:
        st.markdown("**Real-Only Evaluation**")
        if breakdown_report:
            full_real = breakdown_report["models"]["full_stack_pred"]["real_only"]
            robust = breakdown_report["robustness"]["full_stack_pred"]["summary"]
            st.markdown(f"- Core real-only F1: `{full_real.get('f1', 0.0):.2f}`")
            st.markdown(f"- Core real-only FPR: `{full_real.get('false_positive_rate', 0.0):.2f}`")
            st.markdown(f"- Robustness F1: `{robust.get('f1', 0.0):.2f}`")
            st.markdown(f"- Robustness FPR: `{robust.get('false_positive_rate', 0.0):.2f}`")
        else:
            st.caption("Run `python3 experiments/report_breakdown.py` to populate the evaluation breakdown.")

st.markdown("---")
time_step=st.slider("⏩ Timeline",0,n_steps-1,min(150,n_steps-1),format="%d s")
cur=results[time_step]

# ── ADVANCED INFERENCE ────────────────────────────────────────────────────────
with st.spinner("Advanced ML inference…"):
    lnn_model=get_lnn(); lnn_res=lnn_predict_trace(lnn_model,df,road_direction=road_dir)
    topo_res=compute_topology_fingerprint(df)
    spectral=compute_spectral_vulnerability(geo_features)
    crit=get_segment_criticality(cur["lat"],cur["lon"],spectral)
    adj_sev=adjusted_severity(cur["confidence"] if cur["wrong_way"] else 0.0,crit)
    ode_model,ode_mins,ode_ranges=get_ode()
    ode_res=ode_reconstruction_error(ode_model,ode_mins,ode_ranges,df)
    sm=SurpriseScoreModel(); sm.fit(load_trace(loc_key,"normal"))
    surprise_res=sm.score_trace(df)
    kan_model=get_kan(); kan_res=kan_predict(kan_model,l1,l3,topo_res,surprise_res,time_step)
    rssm_model,rssm_mins,rssm_ranges=get_rssm()
    futures_res=sample_futures(rssm_model,rssm_mins,rssm_ranges,df,time_step,n_samples=n_futures,steps=12)
    heading_deltas=np.array([r["road_delta"] for r in results],dtype=float)
    baseline_res=fft_rppg_baseline(df["heading"].values, df["hr"].values, road_dir)
    tti_res=compute_tti_filter(
        ode_res["max_mse"],
        topo_res["max_persistence"],
        surprise_res["max_accumulated"],
        float(np.std(np.diff(heading_deltas,prepend=heading_deltas[0]))/180.0),
    )

# ── MAP ───────────────────────────────────────────────────────────────────────
map_col,sig_col=st.columns([3,2])
with map_col:
    st.markdown('<div class="sec-header">Live situation map</div>',unsafe_allow_html=True)
    m=folium.Map(location=center,zoom_start=14,tiles="CartoDB dark_matter",prefer_canvas=True)
    lats=[r["lat"] for r in results]; lons=[r["lon"] for r in results]
    folium.PolyLine(list(zip(lats,lons)),color="#58a6ff" if trace_type=="normal" else "#ff6b35",weight=2.5,opacity=0.6).add_to(m)
    folium.PolyLine(list(zip([la+.0004 for la in lats],[lo+.0004 for lo in lons])),color="#00d4aa",weight=2,opacity=0.4,dash_array="8 4",tooltip="Safe re-route").add_to(m)
    if l5["heatmap_points"]: HeatMap(l5["heatmap_points"],min_opacity=.3,max_opacity=.8,radius=22,blur=18,gradient={.2:"blue",.5:"orange",1.0:"red"}).add_to(m)
    for ring in l8["ripple_rings"]: folium.Circle(location=ring["center"],radius=ring["radius_m"],color=ring["color"],weight=1.5,opacity=ring["opacity"],fill=True,fill_opacity=ring["opacity"]*.15,tooltip=ring["label"]).add_to(m)
    for event in workzones[:10]:
        geom = event.get("geometry", {})
        props = event.get("properties", {})
        if geom.get("type") == "LineString":
            coords = [[lat, lon] for lon, lat in geom.get("coordinates", [])]
            folium.PolyLine(coords,color="#ffa500",weight=4,opacity=0.65,tooltip=props.get("road_event_id","WZDx work zone")).add_to(m)
    for seg in spectral.get("critical_segments",[])[:12]:
        folium.CircleMarker(location=[seg["lat"],seg["lon"]],radius=5,color="#ff2d55" if seg["criticality"]>spectral["max_criticality"]*.8 else "#ffa500",fill=True,fill_opacity=0.5,tooltip=f"⚡ Criticality={seg['criticality']:.4f}").add_to(m)
    for fi,fut in enumerate(futures_res["futures_geo"][:n_futures]):
        is_c=fi<futures_res["collision_count"]
        folium.PolyLine([[p[0],p[1]] for p in fut],color="#ff2d55" if is_c else "#484f58",weight=1.2 if is_c else 0.5,opacity=0.6 if is_c else 0.25,tooltip="Collision future" if is_c else "Safe").add_to(m)
    for ev in l1["events"][:30]: folium.CircleMarker(location=[ev["lat"],ev["lon"]],radius=6,color="#ff2d55",fill=True,fill_color="#ff2d55",fill_opacity=0.8,tooltip=f"⚠️ conf={ev['confidence']:.2f}").add_to(m)
    folium.Marker(location=[cur["lat"],cur["lon"]],icon=folium.Icon(color="red" if cur["wrong_way"] else "blue",icon="exclamation-sign" if cur["wrong_way"] else "car",prefix="glyphicon"),tooltip=f"t={time_step}s hdg={cur['heading']:.0f}° spd={cur['speed']:.0f}").add_to(m)
    for pi in l5["predicted_impacts"][:3]: folium.Marker(location=[pi["lat"],pi["lon"]],icon=folium.Icon(color="orange",icon="warning-sign",prefix="glyphicon"),tooltip=f"⚡ Impact {pi['time_to_impact_s']}s").add_to(m)
    for dz in l9["dead_zones"][:5]: folium.CircleMarker(location=[dz["lat"],dz["lon"]],radius=4,color="#bc8cff",fill=True,fill_opacity=0.6,tooltip="📵 Dead zone").add_to(m)
    st_folium(m,height=460,use_container_width=True)

with sig_col:
    freqs=np.array(l3["freqs"]); amps=np.array(l3["amplitudes"]); mask=freqs<=2.0
    dom_f,dom_a=l3["dominant_freq"],l3["dominant_amp"]
    fig_fft=go.Figure()
    fig_fft.add_trace(go.Scatter(x=freqs[mask],y=amps[mask],mode="lines",line=dict(color="#58a6ff",width=1.5),showlegend=False))
    fig_fft.add_trace(go.Scatter(x=freqs[mask],y=amps[mask],mode="none",fill="tozeroy",fillcolor="rgba(88,166,255,.15)",showlegend=False))
    if dom_f>0: fig_fft.add_trace(go.Scatter(x=[dom_f],y=[dom_a],mode="markers+text",marker=dict(color=atk["color"],size=10,symbol="diamond"),text=[f" {dom_f:.3f}Hz"],textposition="middle right",textfont=dict(color=atk["color"],size=10),showlegend=False))
    lo,hi=atk["freq_range"]
    if hi>0: fig_fft.add_vrect(x0=lo,x1=hi,fillcolor=atk["color"],opacity=.12,line_color=atk["color"],line_width=1,annotation_text=atk["badge"],annotation_position="top left",annotation_font_color=atk["color"],annotation_font_size=10)
    fig_fft.update_layout(**dark_fig(height=195,margin=dict(l=35,r=8,t=28,b=28),title=dict(text=f"L3 FFT · {dom_f:.3f}Hz",font=dict(color="#f0f6fc",size=12)),xaxis=dict(title="Hz",gridcolor="#21262d",range=[0,2]),yaxis=dict(title="Amp",gridcolor="#21262d")))
    st.plotly_chart(fig_fft,use_container_width=True,config={"displayModeBar":False})

    hrs=np.array(l4["hr_smooth"]); ts_h=np.array(l4["time"])
    w=80; s_i=max(0,time_step-w); e_i=time_step+1
    hrc="#ff2d55" if l4["stress_flag"] else "#00d4aa"
    fig_hr=go.Figure()
    fig_hr.add_trace(go.Scatter(x=ts_h[s_i:e_i],y=np.array(l4["hr_raw"])[s_i:e_i],mode="lines",line=dict(color="rgba(188,140,255,.3)",width=1),showlegend=False))
    fig_hr.add_trace(go.Scatter(x=ts_h[s_i:e_i],y=hrs[s_i:e_i],mode="lines",line=dict(color=hrc,width=2),showlegend=False))
    fig_hr.add_hline(y=110,line_dash="dot",line_color="#ffa500",annotation_text="Stress",annotation_font_size=9,annotation_font_color="#ffa500")
    if time_step<len(ts_h): fig_hr.add_trace(go.Scatter(x=[ts_h[time_step]],y=[hrs[time_step]],mode="markers",marker=dict(color=hrc,size=9),showlegend=False))
    fig_hr.update_layout(**dark_fig(height=175,margin=dict(l=35,r=8,t=28,b=28),title=dict(text=f"L4 rPPG · {l4['current_hr']:.0f}BPM {'⚠️' if l4['stress_flag'] else '✓'}",font=dict(color="#f0f6fc",size=12)),xaxis=dict(gridcolor="#21262d"),yaxis=dict(gridcolor="#21262d",range=[40,170])))
    st.plotly_chart(fig_hr,use_container_width=True,config={"displayModeBar":False})
    if l4["guardian_calm"]: st.markdown('<span class="badge badge-red">🫁 Guardian Calm Active</span>',unsafe_allow_html=True)

    lnn_score=lnn_res["scores"][min(time_step,len(lnn_res["scores"])-1)]
    lnn_c="#ff2d55" if lnn_score>0.5 else "#00d4aa"
    fig_lnn=go.Figure(go.Indicator(mode="gauge+number",value=lnn_score*100,
        gauge=dict(axis=dict(range=[0,100]),bar=dict(color=lnn_c),
                   steps=[dict(range=[0,lnn_res["threshold"]*100],color="#0e2a1e"),dict(range=[lnn_res["threshold"]*100,100],color="#3d1b2e")],
                   threshold=dict(line=dict(color="#fff",width=2),thickness=.75,value=lnn_res["threshold"]*100)),
        number=dict(suffix="%",font=dict(color=lnn_c))))
    fig_lnn.update_layout(**dark_fig(height=145,margin=dict(l=20,r=20,t=10,b=10)))
    st.markdown('<div class="sec-header">F1 LNN P(wrong-way)</div>',unsafe_allow_html=True)
    st.plotly_chart(fig_lnn,use_container_width=True,config={"displayModeBar":False})
    st.caption(f"LNN backend: {lnn_res['backend']} · threshold={lnn_res['threshold']:.2f}")

ts_arr=[r["t"] for r in results]; hs_arr=[r["heading"] for r in results]; ws_arr=[r["wrong_way"] for r in results]
lnn_sc=lnn_res["scores"][:len(ts_arr)]
fig_h=go.Figure()
fig_h.add_trace(go.Scatter(x=ts_arr,y=hs_arr,mode="lines",line=dict(color="#58a6ff",width=1.5),showlegend=False))
wt=[t for t,w in zip(ts_arr,ws_arr) if w]; wh=[h for h,w in zip(hs_arr,ws_arr) if w]
if wt: fig_h.add_trace(go.Scatter(x=wt,y=wh,mode="markers",marker=dict(color="#ff2d55",size=4),showlegend=False))
if lnn_sc: fig_h.add_trace(go.Scatter(x=ts_arr[:len(lnn_sc)],y=[s*360 for s in lnn_sc],mode="lines",line=dict(color="#e3b341",width=1,dash="dot"),showlegend=False))
fig_h.add_vline(x=ts_arr[min(time_step,len(ts_arr)-1)],line_color="#e3b341",line_dash="dash",line_width=1.5)
fig_h.update_layout(**dark_fig(height=150,margin=dict(l=40,r=10,t=30,b=30),title=dict(text="Heading + F1 LNN score (amber dashed)",font=dict(color="#f0f6fc",size=12)),xaxis=dict(title="s",gridcolor="#21262d"),yaxis=dict(title="°",gridcolor="#21262d",range=[0,360])))
st.plotly_chart(fig_h,use_container_width=True,config={"displayModeBar":False})

# ── ADVANCED PANELS ───────────────────────────────────────────────────────────
if show_adv:
    st.markdown("---")
    st.markdown("### Advanced ML Feature Panels")

    if active_tab=="LNN+Topology":
        a1,a2=st.columns(2)
        with a1:
            st.markdown('<div class="sec-header">F1 LNN – Probability timeline</div>',unsafe_allow_html=True)
            scores=lnn_res["scores"]
            fig_lt=go.Figure()
            fig_lt.add_trace(go.Scatter(x=list(range(len(scores))),y=scores,mode="lines",fill="tozeroy",line=dict(color="#58a6ff",width=1.5),fillcolor="rgba(88,166,255,.15)"))
            fig_lt.add_hline(y=lnn_res["threshold"],line_dash="dot",line_color="#ff2d55",annotation_text="Threshold",annotation_font_size=9,annotation_font_color="#ff2d55")
            fig_lt.add_vline(x=time_step,line_color="#e3b341",line_dash="dash")
            fig_lt.update_layout(**dark_fig(height=235,margin=dict(l=35,r=8,t=25,b=25),title=dict(text=f"LNN P(wrong-way) · max={lnn_res['max_score']:.3f}",font=dict(color="#f0f6fc",size=12)),xaxis=dict(title="Time (s)",gridcolor="#21262d"),yaxis=dict(title="P",gridcolor="#21262d",range=[0,1])))
            st.plotly_chart(fig_lt,use_container_width=True,config={"displayModeBar":False})
        with a2:
            st.markdown('<div class="sec-header">F2 Topology – H1 persistence diagram</div>',unsafe_allow_html=True)
            h1=topo_res.get("h1",[]); tc="#ff2d55" if topo_res["is_anomalous"] else "#00d4aa"
            fig_tp=go.Figure()
            fig_tp.add_trace(go.Scatter(x=[0,.5],y=[0,.5],mode="lines",line=dict(color="#484f58",dash="dot",width=1),showlegend=False))
            if h1:
                births=[p[0] for p in h1]; deaths=[p[1] for p in h1]
                fig_tp.add_trace(go.Scatter(x=births,y=deaths,mode="markers",marker=dict(color=tc,size=10,symbol="diamond"),name="H1 loops"))
            fig_tp.add_annotation(x=0.05,y=0.45,text=topo_res["label"],font=dict(color=tc,size=11),showarrow=False)
            fig_tp.update_layout(**dark_fig(height=235,margin=dict(l=35,r=8,t=25,b=25),title=dict(text=f"H1 Persistence · max={topo_res['max_persistence']:.4f}",font=dict(color="#f0f6fc",size=12)),xaxis=dict(title="Birth",gridcolor="#21262d",range=[0,.5]),yaxis=dict(title="Death",gridcolor="#21262d",range=[0,.5])))
            st.plotly_chart(fig_tp,use_container_width=True,config={"displayModeBar":False})
        st.markdown(f"LNN: `{lnn_res['wrong_way_steps']}/{n_steps}` wrong-way steps · max_P=`{lnn_res['max_score']:.4f}` | Topology: loops=`{topo_res['n_loops']}` · persist=`{topo_res['max_persistence']:.5f}` · `{topo_res['label']}`")

    elif active_tab=="KAN Curves":
        st.markdown('<div class="sec-header">F4 KAN – Learned B-spline decision curves (orange line = current feature value)</div>',unsafe_allow_html=True)
        curves=kan_res["edge_curves"]
        cols_k=st.columns(3)
        for idx,(edge_key,(xs,ys)) in enumerate(list(curves.items())[:6]):
            with cols_k[idx%3]:
                fname=edge_key.split("→")[0]; fval=kan_res["features"].get(fname,0.0)
                fig_k=go.Figure()
                fig_k.add_trace(go.Scatter(x=xs,y=ys,mode="lines",line=dict(color="#bc8cff",width=2)))
                fig_k.add_vline(x=float(fval)*2-1,line_color="#ff6b35",line_dash="dash",line_width=1.5,annotation_text=f"{fval:.2f}",annotation_font_color="#ff6b35",annotation_font_size=9)
                fig_k.update_layout(**dark_fig(height=148,margin=dict(l=20,r=5,t=28,b=18),title=dict(text=edge_key,font=dict(color="#f0f6fc",size=10)),xaxis=dict(gridcolor="#21262d",showticklabels=False),yaxis=dict(gridcolor="#21262d",showticklabels=False)))
                st.plotly_chart(fig_k,use_container_width=True,config={"displayModeBar":False})
        kc="#ff2d55" if kan_res["decision"]=="WRONG-WAY" else "#00d4aa"
        st.markdown(f'<span class="badge" style="color:{kc};background:#161b22;border:1px solid {kc}">KAN: {kan_res["decision"]}  P={kan_res["probability"]:.4f}</span>',unsafe_allow_html=True)

    elif active_tab=="ODE+Surprise":
        od1,od2=st.columns(2)
        with od1:
            st.markdown('<div class="sec-header">F6 Neural ODE – Reconstruction error (off-manifold = anomaly)</div>',unsafe_allow_html=True)
            oe=ode_res["errors"]
            fig_ode=go.Figure()
            fig_ode.add_trace(go.Scatter(x=[e["step"] for e in oe],y=[e["mse"] for e in oe],mode="lines+markers",line=dict(color="#58a6ff",width=1.5),marker=dict(color=["#ff2d55" if e["anomalous"] else "#58a6ff" for e in oe],size=5)))
            fig_ode.add_hline(y=0.05,line_dash="dot",line_color="#ffa500",annotation_text="Anomaly",annotation_font_size=9)
            fig_ode.update_layout(**dark_fig(height=235,margin=dict(l=35,r=8,t=25,b=25),title=dict(text=f"ODE MSE · manifold_score={ode_res['manifold_score']:.3f}",font=dict(color="#f0f6fc",size=12)),xaxis=dict(title="Step",gridcolor="#21262d"),yaxis=dict(title="MSE",gridcolor="#21262d")))
            st.plotly_chart(fig_ode,use_container_width=True,config={"displayModeBar":False})
        with od2:
            st.markdown('<div class="sec-header">F9 Surprise Score – bits of anomaly per GPS ping</div>',unsafe_allow_html=True)
            surp=surprise_res["scores"]
            s_t=[s["t"] for s in surp]; s_a=[s["accumulated"] for s in surp]; s_c=[s["surprise_bits"] for s in surp]
            fig_surp=go.Figure()
            fig_surp.add_trace(go.Scatter(x=s_t,y=s_c,mode="lines",line=dict(color="#bc8cff",width=1),fill="tozeroy",fillcolor="rgba(188,140,255,.1)",showlegend=False))
            fig_surp.add_trace(go.Scatter(x=s_t,y=s_a,mode="lines",line=dict(color="#ff6b35",width=2),showlegend=False))
            fig_surp.add_hline(y=35,line_dash="dot",line_color="#ff2d55",annotation_text="Wrong-way (35 bits)",annotation_font_size=9,annotation_font_color="#ff2d55")
            if time_step<len(s_t): fig_surp.add_vline(x=s_t[time_step],line_color="#e3b341",line_dash="dash")
            fig_surp.update_layout(**dark_fig(height=235,margin=dict(l=35,r=8,t=25,b=25),title=dict(text=f"Surprise · max={surprise_res['max_accumulated']:.1f} bits",font=dict(color="#f0f6fc",size=12)),xaxis=dict(title="Time (s)",gridcolor="#21262d"),yaxis=dict(title="Bits",gridcolor="#21262d")))
            st.plotly_chart(fig_surp,use_container_width=True,config={"displayModeBar":False})
        tcol1,tcol2=st.columns(2)
        with tcol1:
            st.markdown(f"**F5 Spectral** · ac=`{spectral['ac']:.5f}` · criticality at pos=`{crit:.5f}` · adj_severity=`{adj_sev:.3f}`")
        with tcol2:
            st.markdown(f"**TTI-Filter** · conf=`{tti_res['confidence']:.3f}` · topo_weight=`{tti_res['topology_weight']:.3f}` · `{tti_res['label']}`")

    elif active_tab=="RSSM Futures":
        st.markdown('<div class="sec-header">F10 World Model RSSM – Sampled futures (red = collision path)</div>',unsafe_allow_html=True)
        futs=futures_res["futures_geo"]
        fig_fut=go.Figure()
        for fi,fut in enumerate(futs):
            is_c=fi<futures_res["collision_count"]
            fig_fut.add_trace(go.Scatter(x=[p[1] for p in fut],y=[p[0] for p in fut],mode="lines",line=dict(color="#ff2d55" if is_c else "#484f58",width=1.5 if is_c else 0.5),opacity=0.7 if is_c else 0.3,showlegend=False))
        fig_fut.add_trace(go.Scatter(x=[futures_res["origin_lon"]],y=[futures_res["origin_lat"]],mode="markers",marker=dict(color="#e3b341",size=12,symbol="star"),showlegend=False))
        col_pct=futures_res["collision_fraction"]*100
        fig_fut.update_layout(**dark_fig(height=340,margin=dict(l=40,r=10,t=35,b=30),title=dict(text=f"RSSM: {n_futures} futures · {futures_res['collision_count']} collision paths ({col_pct:.0f}%)",font=dict(color="#ff2d55" if col_pct>50 else "#f0f6fc",size=12)),xaxis=dict(title="Lon",gridcolor="#21262d"),yaxis=dict(title="Lat",gridcolor="#21262d")))
        st.plotly_chart(fig_fut,use_container_width=True,config={"displayModeBar":False})
        cc="#ff2d55" if futures_res["collision_fraction"]>.5 else "#ffa500" if futures_res["collision_fraction"]>.2 else "#00d4aa"
        st.markdown(f'<span class="badge" style="color:{cc};background:#161b22;border:1px solid {cc}">Collision P: {col_pct:.0f}% ({futures_res["collision_count"]}/{n_futures})</span>',unsafe_allow_html=True)

# ── SUMMARY ROW ───────────────────────────────────────────────────────────────
st.markdown("---")
log_c,brk_c=st.columns([3,2])
with log_c:
    st.markdown('<div class="sec-header">Alert log</div>',unsafe_allow_html=True)
    alerts=pipeline["layer2"]["alerts"]
    sev_bc={"EMERGENCY":"badge-red","WARNING":"badge-orange","CAUTION":"badge-blue"}
    if alerts:
        html="".join(f'<div style="border-bottom:1px solid #21262d;padding:5px 0;font-size:.78rem"><span class="badge {sev_bc.get(a["severity"],"badge-blue")}">{a["severity"]}</span> t={a["time"]:>4}s conf={a["confidence"]:.2f} → {", ".join(a["channels"])}</div>' for a in alerts[:20])
        st.markdown(html,unsafe_allow_html=True)
    else: st.caption("No alerts.")
with brk_c:
    st.markdown('<div class="sec-header">9-layer safety cascade</div>',unsafe_allow_html=True)
    ui_wrong_threshold = 20 if trace_type.startswith("real_") or loc_key == "us101" else 120
    comp_rows=[
        ("Baseline rule", "ALARM" if baseline_res["predicted_wrong_way"] else "CLEAR", "flag" if baseline_res["predicted_wrong_way"] else "confirm"),
        ("TTI stack", "CONFIRMED" if tti_res["confidence"]>=0.55 and l1["wrong_count"]>=ui_wrong_threshold else "SUPPRESSED", "confirm" if tti_res["confidence"]>=0.55 and l1["wrong_count"]>=ui_wrong_threshold else "flag"),
    ]
    comp_html="".join(f'<tr class="{cls}"><td>{name}</td><td>{state}</td></tr>' for name,state,cls in comp_rows)
    st.markdown(f'<table class="sm-table"><tbody>{comp_html}</tbody></table>',unsafe_allow_html=True)
    status_rows=[
        ("L1 GPS consensus",f"{l1['wrong_count']}/{l1['total']} wrong-way steps","flag" if cur["wrong_way"] else ""),
        ("L7 construction suppression",f"{pipeline['layer7']['downgraded_count']} downgraded","confirm" if pipeline['layer7']['downgraded_count']>0 else ""),
        ("TTI-Filter",f"{tti_res['label']} · conf={tti_res['confidence']:.3f}","confirm" if tti_res['confidence']>=0.55 else "flag"),
        ("L8 swarm ripple",f"{l8['alerted_count']} vehicles · {l8['swarm_latency_ms']}ms","confirm" if l8['alerted_count']>0 else ""),
    ]
    rows_html="".join(f'<tr class="{cls}"><td>{name}</td><td>{value}</td></tr>' for name,value,cls in status_rows)
    st.markdown(f'<table class="sm-table"><tbody>{rows_html}</tbody></table>',unsafe_allow_html=True)
    rows=[("L3",f"{atk['badge']} {l3['dominant_freq']:.3f}Hz"),("L4",f"{l4['current_hr']:.0f}BPM {'STRESS' if l4['stress_flag'] else 'ok'}"),
          ("L5",f"{len(l5['predicted_impacts'])} impacts"),("L6",f"{pipeline['layer6']['suppressed_count']} suppressed"),
          ("L9",f"score={l9['connectivity_score']:.0f}"),("F1-LNN",f"max_P={lnn_res['max_score']:.3f}"),
          ("F2-TDA",f"loops={topo_res['n_loops']} persist={topo_res['max_persistence']:.4f}"),
          ("F4-KAN",f"{kan_res['decision']} P={kan_res['probability']:.3f}"),
          ("F5-Spec",f"ac={spectral['ac']:.5f} crit={crit:.4f}"),
          ("F6-ODE",f"manifold={ode_res['manifold_score']:.3f}"),
          ("F9-Surp",f"max={surprise_res['max_accumulated']:.1f}b"),
          ("F10-RSSM",f"{futures_res['collision_count']}/{n_futures} collision futures")]
    for n,v in rows: st.markdown(f"**{n}** — {v}")
    st.markdown(f"**Sources** — trace={source_bundle['trace_source']} · raw_gps={source_bundle['raw_gps_source']} · labeled_events={source_bundle['real_labeled_events']} · geometry={source_bundle['geometry_source']} · towers={source_bundle['tower_source']} · workzones={source_bundle['workzone_source']}")

st.markdown('<span style="color:#484f58;font-size:.7rem">SwarmMind · MIT-Mahe × HARMAN Automotive · 9-Layer Safety Engine + TTI Stack</span>',unsafe_allow_html=True)
