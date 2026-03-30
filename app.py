"""
Aerospace TPMS Lattice — Inverse Design Explorer
=================================================
Interactive forward-prediction and inverse-design tool for
additive-manufactured lattice structures.

Deploy: Hugging Face Spaces (Streamlit SDK)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib, json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aerospace Lattice Design Explorer",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — clean, modern, professional
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #f5f7fa;
    color: #1e293b;
}

.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #0ea5e9 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    color: white;
}
.hero h1 {
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
    color: white;
}
.hero p {
    font-size: 0.92rem;
    margin: 0;
    opacity: 0.85;
    color: white;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #64748b;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 5px;
    margin-bottom: 1rem;
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #2563eb;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 0.5rem;
}
.metric-card.good   { border-top-color: #10b981; }
.metric-card.warn   { border-top-color: #f59e0b; }
.metric-card.info   { border-top-color: #6366f1; }
.metric-card.neutral{ border-top-color: #64748b; }
.mc-label { font-size: 0.68rem; color: #64748b; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.5px; }
.mc-value { font-size: 1.45rem; font-weight: 700; color: #1e293b; line-height: 1.15; }
.mc-unit  { font-size: 0.72rem; color: #94a3b8; }

/* ── Alert boxes ── */
.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.83rem;
    color: #1e3a5f;
    margin-bottom: 1rem;
}
.warn-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.83rem;
    color: #78350f;
    margin-bottom: 1rem;
}
.success-box {
    background: #f0fdf4;
    border-left: 4px solid #10b981;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.83rem;
    color: #14532d;
    margin-bottom: 1rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b;
    margin-top: 1.4rem;
    padding-top: 0.8rem;
    border-top: 1px solid #f1f5f9;
}
.stSlider label { font-size: 0.83rem !important; }
.stSelectbox label { font-size: 0.83rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    width: 100%;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.9; }

/* ── Tabs ── */
[data-testid="stTab"] { font-size: 0.86rem; font-weight: 500; }

hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MATERIALS = {
    "Ti6Al4V":         {"E_s": 114.0, "sigma_s": 880.0,  "rho_s": 4430.0, "label": "Titanium Alloy (Ti6Al4V)"},
    "AlSi10Mg":        {"E_s":  70.0, "sigma_s": 325.0,  "rho_s": 2670.0, "label": "Aluminium Alloy (AlSi10Mg)"},
    "316L_SS":         {"E_s": 193.0, "sigma_s": 480.0,  "rho_s": 7900.0, "label": "Stainless Steel 316L"},
    "PLA":             {"E_s":   3.5, "sigma_s":  65.0,  "rho_s": 1240.0, "label": "PLA Polymer"},
    "TPU":             {"E_s":  0.04, "sigma_s":   8.0,  "rho_s": 1200.0, "label": "TPU Elastomer"},
    "GPR":             {"E_s": 1.916, "sigma_s": 64.35,  "rho_s": 1180.0, "label": "GPR Resin"},
    "HEC":             {"E_s": 1.001, "sigma_s": 30.75,  "rho_s": 1150.0, "label": "HEC Resin"},
    "HTB":             {"E_s": 1.291, "sigma_s": 36.02,  "rho_s": 1200.0, "label": "HTB Resin"},
    "DUR":             {"E_s": 1.260, "sigma_s": 32.00,  "rho_s": 1100.0, "label": "DUR Resin"},
    "BMC":             {"E_s": 1.600, "sigma_s": 45.00,  "rho_s": 1170.0, "label": "BMC Composite"},
}
PROCESSES = {
    "LPBF":             {"min_wall": 0.20, "surface_penalty": 0.05, "support_risk": 0.25, "label": "Laser Powder Bed Fusion (LPBF)"},
    "EBM":              {"min_wall": 0.40, "surface_penalty": 0.03, "support_risk": 0.15, "label": "Electron Beam Melting (EBM)"},
    "FDM":              {"min_wall": 0.40, "surface_penalty": 0.10, "support_risk": 0.35, "label": "Fused Deposition Modelling (FDM)"},
    "SLA":              {"min_wall": 0.10, "surface_penalty": 0.02, "support_risk": 0.10, "label": "Stereolithography (SLA)"},
    "Material_Jetting": {"min_wall": 0.15, "surface_penalty": 0.03, "support_risk": 0.12, "label": "Material Jetting"},
}
TPMS_PARAMS = {
    "gyroid":    {"C1": 0.300, "n1": 2.10, "C2": 0.300, "n2": 1.50, "ea_factor": 1.05},
    "diamond":   {"C1": 0.350, "n1": 1.90, "C2": 0.350, "n2": 1.40, "ea_factor": 1.10},
    "primitive": {"C1": 0.200, "n1": 2.30, "C2": 0.250, "n2": 1.60, "ea_factor": 0.90},
    "iwp":       {"C1": 0.280, "n1": 2.00, "C2": 0.300, "n2": 1.50, "ea_factor": 1.00},
}
TARGET_COLS = ["E_eff_GPa", "sigma_y_MPa", "EA_vol_MJm3"]
TARGET_LABELS = {
    "E_eff_GPa":   ("Effective Stiffness", "E*",  "GPa"),
    "sigma_y_MPa": ("Yield Strength",       "σ_y", "MPa"),
    "EA_vol_MJm3": ("Energy Absorption",    "EA",  "MJ/m³"),
}

# Vibrant, accessible colour palette per TPMS family
TPMS_COLORS = {
    "gyroid":    "#2563eb",
    "diamond":   "#10b981",
    "primitive": "#f59e0b",
    "iwp":       "#ef4444",
}
PLOTLY_TEMPLATE = "plotly_white"

# Feature columns (must match training pipeline)
FEATURE_COLS_CORE = [
    "relative_density", "cell_size_mm", "wall_thickness_mm",
    "source_Synthetic", "source_FEA", "source_Experimental",
]
TPMS_DUMMIES = [f"tpms_family_{t}" for t in TPMS_PARAMS]
MAT_DUMMIES  = [f"material_{m}"    for m in MATERIALS]
PROC_DUMMIES = [f"process_{p}"     for p in PROCESSES]

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading prediction engine…")
def load_surrogate():
    model_dir = Path("ml_surrogate")
    model  = joblib.load(model_dir / "gbr_surrogate.pkl")
    scaler = joblib.load(model_dir / "feature_scaler.pkl")
    meta   = json.loads((model_dir / "surrogate_meta.json").read_text())
    return model, scaler, meta

# ─────────────────────────────────────────────────────────────────────────────
# Surrogate helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(tpms, material, process, rho, cs, source="FEA", meta=None):
    wall = cs * 0.17 * (rho / 0.30) ** 0.5
    feat_cols = (meta or {}).get("feature_cols",
                 FEATURE_COLS_CORE + TPMS_DUMMIES + MAT_DUMMIES + PROC_DUMMIES)
    row = {c: 0.0 for c in feat_cols}
    row["relative_density"]  = rho
    row["cell_size_mm"]      = cs
    row["wall_thickness_mm"] = wall
    src_key = f"source_{source}"
    if src_key in row:
        row[src_key] = 1.0
    elif "source_Synthetic" in row:
        row["source_Synthetic"] = 1.0
    for k in [f"tpms_family_{tpms}", f"material_{material}", f"process_{process}"]:
        if k in row:
            row[k] = 1.0
    return np.array([row[c] for c in feat_cols], dtype=np.float32).reshape(1, -1)


def predict_properties(model, scaler, tpms, material, process, rho, cs, meta=None):
    x    = build_feature_vector(tpms, material, process, rho, cs, meta=meta)
    x_sc = scaler.transform(x)
    pred = model.predict(x_sc)[0]
    return {k: max(0.0, float(v)) for k, v in zip(TARGET_COLS, pred)}


def compute_manufacturability(tpms, material, process, rho, cs):
    pp   = PROCESSES.get(process, {"min_wall": 0.3, "surface_penalty": 0.05, "support_risk": 0.2})
    wall = cs * 0.17 * (rho / 0.30) ** 0.5
    wall_ok = min(1.0, wall / pp["min_wall"])
    score = wall_ok * (1.0 - pp["surface_penalty"]) * (1.0 - 0.5 * pp["support_risk"] * rho)
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def compute_sea(props, material, rho):
    rho_eff = rho * MATERIALS.get(material, {"rho_s": 4430.0})["rho_s"]
    return (props["EA_vol_MJm3"] * 1e6) / rho_eff / 1000.0 if rho_eff > 0 else 0.0


def inverse_design(stiffness_priority, ea_priority, rho_max, material, process):
    scores = {}
    for tpms, tp in TPMS_PARAMS.items():
        s_stiff = tp["C1"] * (0.3 ** tp["n1"])
        s_ea    = tp["ea_factor"] * tp["C2"] * (0.3 ** tp["n2"])
        scores[tpms] = stiffness_priority * s_stiff + ea_priority * s_ea
    best_tpms = max(scores, key=scores.get)
    pp       = PROCESSES.get(process, {"min_wall": 0.3})
    rho_sel  = min(rho_max, 0.35)
    cs_sel   = 4.0
    wall     = cs_sel * 0.17 * (rho_sel / 0.30) ** 0.5
    feasible = wall >= pp["min_wall"]
    return best_tpms, rho_sel, cs_sel, scores, feasible

# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builders — all interactive & coloured
# ─────────────────────────────────────────────────────────────────────────────
_BASE = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=11),
    margin=dict(l=55, r=20, t=70, b=50),
    hovermode="x unified",
)


def chart_density_sweep(model, scaler, material, process, meta):
    rho_arr = np.linspace(0.05, 0.60, 60)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Effective Stiffness E* [GPa]",
            "Yield Strength σ_y [MPa]",
            "Volumetric Energy Absorption [MJ/m³]",
        ),
    )
    for tpms, color in TPMS_COLORS.items():
        E_v, S_v, EA_v = [], [], []
        for rho in rho_arr:
            p = predict_properties(model, scaler, tpms, material, process, rho, 4.0, meta)
            E_v.append(p["E_eff_GPa"])
            S_v.append(p["sigma_y_MPa"])
            EA_v.append(p["EA_vol_MJm3"])
        kw = dict(x=rho_arr, mode="lines", name=tpms.capitalize(),
                  line=dict(color=color, width=2.5),
                  legendgroup=tpms, showlegend=True)
        fig.add_trace(go.Scatter(y=E_v,  hovertemplate="ρ*=%{x:.2f}  E*=%{y:.4f} GPa<extra>" + tpms + "</extra>", **kw), row=1, col=1)
        kw["showlegend"] = False
        fig.add_trace(go.Scatter(y=S_v,  hovertemplate="ρ*=%{x:.2f}  σ_y=%{y:.3f} MPa<extra>" + tpms + "</extra>", **kw), row=1, col=2)
        fig.add_trace(go.Scatter(y=EA_v, hovertemplate="ρ*=%{x:.2f}  EA=%{y:.4f} MJ/m³<extra>" + tpms + "</extra>", **kw), row=1, col=3)
    fig.update_xaxes(title_text="Relative density ρ*")
    fig.update_layout(
        **_BASE,
        height=400,
        hovermode="x unified",
        title=dict(
            text=f"Property Sweep vs Relative Density — {MATERIALS[material]['label']} · {PROCESSES[process]['label']}",
            font=dict(size=13, color="#1e293b"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1,
                    font=dict(size=11)),
        margin=dict(l=55, r=20, t=90, b=50),
    )
    return fig


def chart_sea(model, scaler, material, process, meta):
    rho_arr = np.linspace(0.05, 0.60, 60)
    fig = go.Figure()
    for tpms, color in TPMS_COLORS.items():
        sea_vals = []
        for rho in rho_arr:
            p   = predict_properties(model, scaler, tpms, material, process, rho, 4.0, meta)
            sea_vals.append(compute_sea(p, material, rho))
        fig.add_trace(go.Scatter(
            x=rho_arr, y=sea_vals, mode="lines", name=tpms.capitalize(),
            line=dict(color=color, width=2.5),
            hovertemplate="ρ*=%{x:.2f}  SEA=%{y:.2f} kJ/kg<extra>" + tpms + "</extra>",
        ))
    fig.update_layout(
        **_BASE,
        height=380,
        title=dict(text=f"Specific Energy Absorption — {MATERIALS[material]['label']}",
                   font=dict(size=13, color="#1e293b")),
        xaxis_title="Relative Density ρ*",
        yaxis_title="SEA [kJ/kg]",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def chart_radar(props_by_tpms):
    keys   = TARGET_COLS
    labels = ["Eff. Stiffness (GPa)", "Yield Strength (MPa)", "Energy Absorption (MJ/m³)"]
    all_v  = np.array([[props_by_tpms[t][k] for k in keys] for t in TPMS_PARAMS])
    maxv   = all_v.max(axis=0); maxv[maxv == 0] = 1.0
    fig = go.Figure()
    for tpms, color in TPMS_COLORS.items():
        vals = [props_by_tpms[tpms][k] / maxv[i] for i, k in enumerate(keys)]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=labels + [labels[0]],
            fill="toself", fillcolor=color, opacity=0.18,
            line=dict(color=color, width=2.5),
            name=tpms.capitalize(),
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<extra>" + tpms + "</extra>",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.1], tickfont=dict(size=9),
                            gridcolor="#e2e8f0", linecolor="#e2e8f0"),
            angularaxis=dict(tickfont=dict(size=10)),
            bgcolor="white",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
                    font=dict(size=11)),
        title=dict(text="TPMS Performance Trade-off (normalised)",
                   font=dict(size=13, color="#1e293b")),
        height=430,
        template=PLOTLY_TEMPLATE,
        margin=dict(l=40, r=40, t=60, b=60),
        paper_bgcolor="white",
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    return fig


def chart_property_bars(props_by_tpms, title_suffix=""):
    tpms_list = list(TPMS_PARAMS)
    colors    = [TPMS_COLORS[t] for t in tpms_list]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Effective Stiffness [GPa]",
                                        "Yield Strength [MPa]",
                                        "Energy Absorption [MJ/m³]"))
    for ci, (key, (name, sym, unit)) in enumerate(TARGET_LABELS.items(), start=1):
        vals  = [props_by_tpms[t][key] for t in tpms_list]
        xvals = [t.capitalize() for t in tpms_list]
        fig.add_trace(go.Bar(
            x=xvals, y=vals, marker_color=colors,
            text=[f"{v:.3g}" for v in vals], textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{sym}: %{{y:.4g}} {unit}<extra></extra>",
            showlegend=False,
        ), row=1, col=ci)
    fig.update_layout(
        **_BASE,
        height=380,
        title=dict(text=f"Predicted Mechanical Properties — {title_suffix}",
                   font=dict(size=13, color="#1e293b")),
        hovermode="closest",
    )
    return fig


def chart_cell_size(model, scaler, tpms, material, process, rho, meta):
    cs_arr = np.linspace(1.5, 12.0, 50)
    E_v, S_v, EA_v = [], [], []
    for cs in cs_arr:
        p = predict_properties(model, scaler, tpms, material, process, rho, cs, meta)
        E_v.append(p["E_eff_GPa"])
        S_v.append(p["sigma_y_MPa"])
        EA_v.append(p["EA_vol_MJm3"])
    colors = ["#2563eb", "#10b981", "#f59e0b"]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("E* [GPa]", "σ_y [MPa]", "EA [MJ/m³]"))
    for ci, (vals, color, lbl) in enumerate(
        zip([E_v, S_v, EA_v], colors, ["E* [GPa]", "σ_y [MPa]", "EA [MJ/m³]"]), start=1
    ):
        fig.add_trace(go.Scatter(
            x=cs_arr, y=vals, mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=color + "18",
            hovertemplate=f"cs=%{{x:.1f}} mm  {lbl}=%{{y:.4g}}<extra></extra>",
            showlegend=False,
        ), row=1, col=ci)
        fig.add_vline(x=4.0, line=dict(dash="dash", color="#94a3b8", width=1.2),
                      row=1, col=ci)   # type: ignore
    fig.update_xaxes(title_text="Cell size [mm]")
    fig.update_layout(
        **_BASE,
        height=380,
        hovermode="x unified",
        title=dict(
            text=f"Cell Size Sensitivity — {tpms.capitalize()} · {MATERIALS[material]['label']} · ρ*={rho:.2f}",
            font=dict(size=13, color="#1e293b"),
        ),
    )
    return fig


def chart_material_comparison(model, scaler, tpms, process, rho, cs, meta):
    mat_list = list(MATERIALS.keys())
    palette  = px.colors.qualitative.Bold
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Effective Stiffness [GPa]",
                                        "Yield Strength [MPa]",
                                        "Energy Absorption [MJ/m³]"))
    for ci, (key, (name, sym, unit)) in enumerate(TARGET_LABELS.items(), start=1):
        vals, labels, bar_colors = [], [], []
        for i, mat in enumerate(mat_list):
            try:
                p = predict_properties(model, scaler, tpms, mat, process, rho, cs, meta)
                vals.append(p[key])
                labels.append(MATERIALS[mat]["label"])
                bar_colors.append(palette[i % len(palette)])
            except Exception:
                pass
        fig.add_trace(go.Bar(
            x=labels, y=vals, marker_color=bar_colors,
            text=[f"{v:.3g}" for v in vals], textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{sym}: %{{y:.4g}} {unit}<extra></extra>",
            showlegend=False,
        ), row=1, col=ci)
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=8))
    fig.update_layout(
        **_BASE,
        height=430,
        hovermode="closest",
        title=dict(
            text=f"Material Comparison — {tpms.capitalize()} · ρ*={rho:.2f} · {PROCESSES[process]['label']}",
            font=dict(size=13, color="#1e293b"),
        ),
        margin=dict(l=55, r=20, t=80, b=110),
    )
    return fig


def chart_pareto(model, scaler, material, process, meta):
    rho_arr = np.linspace(0.05, 0.60, 22)
    cs_arr  = [2.0, 4.0, 6.0]
    records = []
    for tpms in TPMS_PARAMS:
        for rho in rho_arr:
            for cs in cs_arr:
                p = predict_properties(model, scaler, tpms, material, process, rho, cs, meta)
                records.append({
                    "TPMS":    tpms.capitalize(),
                    "ρ*":      round(rho, 3),
                    "Cell (mm)": cs,
                    "E* [GPa]":  round(p["E_eff_GPa"], 4),
                    "EA [MJ/m³]": round(p["EA_vol_MJm3"], 4),
                    "σ_y [MPa]":  round(p["sigma_y_MPa"], 3),
                })
    df_p = pd.DataFrame(records)
    fig = px.scatter(
        df_p,
        x="E* [GPa]", y="EA [MJ/m³]",
        color="ρ*", symbol="TPMS",
        color_continuous_scale="Viridis",
        hover_data={"TPMS": True, "ρ*": ":.3f", "E* [GPa]": ":.4f",
                    "EA [MJ/m³]": ":.4f", "σ_y [MPa]": ":.3f", "Cell (mm)": True},
        title=(
            f"Design Space — Stiffness vs Energy Absorption<br>"
            f"<sup>{MATERIALS[material]['label']} · {PROCESSES[process]['label']}</sup>"
        ),
        template=PLOTLY_TEMPLATE,
        height=530,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.82, line=dict(width=0.3, color="white")))
    fig.update_layout(
        coloraxis_colorbar=dict(title="ρ*", len=0.6, thickness=14, tickfont=dict(size=9)),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=60, r=20, t=90, b=55),
        font=dict(family="Inter, Segoe UI, sans-serif", size=11),
    )
    return fig


def chart_inverse_scores(scores, best_tpms, sp, ep):
    tpms_list = list(scores.keys())
    vals      = [scores[t] for t in tpms_list]
    bar_colors = [TPMS_COLORS[t] if t == best_tpms else "#cbd5e1" for t in tpms_list]
    fig = go.Figure(go.Bar(
        x=vals,
        y=[t.capitalize() for t in tpms_list],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.5f}" for v in vals],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.5f}<extra></extra>",
    ))
    fig.update_layout(
        **_BASE,
        height=290,
        hovermode="closest",
        title=dict(
            text=f"Inverse Design Score  (stiffness w={sp:.2f} · energy w={ep:.2f})",
            font=dict(size=13, color="#1e293b"),
        ),
        xaxis_title="Combined Score",
        margin=dict(l=90, r=20, t=65, b=45),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Load surrogate
# ─────────────────────────────────────────────────────────────────────────────
try:
    model, scaler, meta = load_surrogate()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR   = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔩 Aerospace Lattice Design Explorer</h1>
  <p>
    Interactive inverse-design tool for additive-manufactured TPMS lattice structures &nbsp;·&nbsp;
    Predict stiffness, yield strength &amp; energy absorption &nbsp;·&nbsp;
    Compare materials, processes &amp; cell geometries instantly
  </p>
</div>
""", unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error(
        "⚠️ **Prediction engine could not be loaded.**\n\n"
        "Please ensure the `ml_surrogate/` folder containing `gbr_surrogate.pkl`, "
        "`feature_scaler.pkl`, and `surrogate_meta.json` is present in the Space root.\n\n"
        f"Technical detail: `{LOAD_ERROR}`"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [("results_ready", False), ("last_inputs", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Design Inputs")

    st.markdown("### Material & Process")
    material = st.selectbox("Material",  list(MATERIALS.keys()),
                            format_func=lambda k: MATERIALS[k]["label"], index=0)
    process  = st.selectbox("AM Process", list(PROCESSES.keys()),
                            format_func=lambda k: PROCESSES[k]["label"], index=0)

    st.markdown("### Lattice Geometry")
    tpms = st.selectbox("TPMS Family", list(TPMS_PARAMS.keys()),
                        format_func=str.capitalize, index=0)
    rho  = st.slider("Relative Density ρ*", 0.05, 0.60, 0.30, 0.01, format="%.2f")
    cs   = st.slider("Unit Cell Size [mm]", 1.5, 12.0, 4.0, 0.5, format="%.1f")

    st.markdown("### Inverse Design Targets")
    st.caption("Set your performance priorities to get an automated lattice recommendation.")
    stiffness_priority = st.slider("Stiffness Priority",          0.0, 1.0, 0.60, 0.05)
    ea_priority        = st.slider("Energy Absorption Priority",  0.0, 1.0, 0.40, 0.05)
    rho_max            = st.slider("Max Allowable Density",       0.10, 0.60, 0.40, 0.01)

    st.markdown("---")
    run_clicked = st.button("⚡ Run Analysis", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Compute
# ─────────────────────────────────────────────────────────────────────────────
current_inputs = dict(
    material=material, process=process, tpms=tpms, rho=rho, cs=cs,
    stiffness_priority=stiffness_priority, ea_priority=ea_priority, rho_max=rho_max,
)

if run_clicked or not st.session_state.results_ready:
    with st.spinner("Running prediction engine…"):
        _props     = predict_properties(model, scaler, tpms, material, process, rho, cs, meta)
        _manuf     = compute_manufacturability(tpms, material, process, rho, cs)
        _sea       = compute_sea(_props, material, rho)
        _rho_eff   = rho * MATERIALS[material]["rho_s"]
        _wall_mm   = cs * 0.17 * (rho / 0.30) ** 0.5
        _pp_min    = PROCESSES[process]["min_wall"]
        _printable = _wall_mm >= _pp_min
        _all_tpms_props = {
            t: predict_properties(model, scaler, t, material, process, rho, cs, meta)
            for t in TPMS_PARAMS
        }
        _best_tpms, _rho_sel, _cs_sel, _scores, _feasible = inverse_design(
            stiffness_priority, ea_priority, rho_max, material, process
        )
        _inv_props = predict_properties(model, scaler, _best_tpms, material, process,
                                        _rho_sel, _cs_sel, meta)
        _inv_manuf = compute_manufacturability(_best_tpms, material, process, _rho_sel, _cs_sel)
        _inv_all   = {
            t: predict_properties(model, scaler, t, material, process, _rho_sel, _cs_sel, meta)
            for t in TPMS_PARAMS
        }
        st.session_state.update(dict(
            props=_props, manuf=_manuf, sea=_sea,
            rho_eff=_rho_eff, wall_mm=_wall_mm, pp_min=_pp_min, printable=_printable,
            all_tpms_props=_all_tpms_props,
            best_tpms=_best_tpms, rho_sel=_rho_sel, cs_sel=_cs_sel,
            scores=_scores, inv_feasible=_feasible,
            inv_props=_inv_props, inv_manuf=_inv_manuf, inv_all=_inv_all,
            inv_sp=stiffness_priority, inv_ea=ea_priority,
            results_ready=True,
        ))
        st.session_state.last_inputs = current_inputs

# Bind
props          = st.session_state.props
manuf          = st.session_state.manuf
sea            = st.session_state.sea
rho_eff        = st.session_state.rho_eff
wall_mm        = st.session_state.wall_mm
pp_min         = st.session_state.pp_min
printable      = st.session_state.printable
all_tpms_props = st.session_state.all_tpms_props
best_tpms      = st.session_state.best_tpms
rho_sel        = st.session_state.rho_sel
cs_sel         = st.session_state.cs_sel
scores         = st.session_state.scores
inv_feasible   = st.session_state.inv_feasible
inv_props      = st.session_state.inv_props
inv_manuf      = st.session_state.inv_manuf
inv_all        = st.session_state.inv_all
inv_sp         = st.session_state.inv_sp
inv_ea         = st.session_state.inv_ea

# ─────────────────────────────────────────────────────────────────────────────
# KPI metric strip
# ─────────────────────────────────────────────────────────────────────────────
manuf_class = "good" if manuf >= 0.7 else ("warn" if manuf >= 0.4 else "neutral")

c1, c2, c3, c4, c5, c6 = st.columns(6)
kpi_data = [
    (c1, "Effective Stiffness",  f"{props['E_eff_GPa']:.4f}",   "GPa",    "info"),
    (c2, "Yield Strength",       f"{props['sigma_y_MPa']:.3f}",  "MPa",    "info"),
    (c3, "Energy Absorption",    f"{props['EA_vol_MJm3']:.4f}",  "MJ/m³",  "info"),
    (c4, "Specific EA",          f"{sea:.2f}",                   "kJ/kg",  "info"),
    (c5, "Manufacturability",    f"{manuf:.3f}",                 "/ 1.00", manuf_class),
    (c6, "Effective Density",    f"{rho_eff:.0f}",               "kg/m³",  "neutral"),
]
for col, lbl, val, unit, cls in kpi_data:
    with col:
        st.markdown(f"""
        <div class="metric-card {cls}">
          <div class="mc-label">{lbl}</div>
          <div class="mc-value">{val}</div>
          <div class="mc-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

# Printability banner
if not printable:
    st.markdown(
        f'<div class="warn-box">⚠️ <b>Printability warning:</b> Estimated wall thickness '
        f'<b>{wall_mm:.3f} mm</b> is below the minimum for '
        f'<b>{PROCESSES[process]["label"]}</b> ({pp_min:.2f} mm minimum). '
        f'Try reducing cell size or increasing relative density.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="success-box">✅ Wall thickness <b>{wall_mm:.3f} mm</b> is within '
        f'the printable range for <b>{PROCESSES[process]["label"]}</b> '
        f'(minimum {pp_min:.2f} mm).</div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Forward Prediction",
    "🎯 Inverse Design",
    "📈 Density Sweep",
    "🔬 Material Comparison",
    "🗺️ Design Space",
])

# ── Tab 0: Forward Prediction ─────────────────────────────────────────────
with tabs[0]:
    st.markdown('<p class="section-label">TPMS Family Comparison — Current Configuration</p>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([1.65, 1])
    with col_l:
        st.plotly_chart(
            chart_property_bars(
                all_tpms_props,
                title_suffix=f"{MATERIALS[material]['label']} · ρ*={rho:.2f} · {cs:.1f} mm",
            ),
            use_container_width=True,
        )
    with col_r:
        st.plotly_chart(chart_radar(all_tpms_props), use_container_width=True)

    st.markdown('<p class="section-label">Selected Design — Full Summary</p>',
                unsafe_allow_html=True)
    design_row = {
        "TPMS Family":         tpms.capitalize(),
        "Material":            MATERIALS[material]["label"],
        "AM Process":          PROCESSES[process]["label"],
        "Relative Density ρ*": f"{rho:.3f}",
        "Unit Cell Size":      f"{cs:.1f} mm",
        "Wall Thickness":      f"{wall_mm:.4f} mm",
        "Effective Stiffness": f"{props['E_eff_GPa']:.5f} GPa",
        "Yield Strength":      f"{props['sigma_y_MPa']:.4f} MPa",
        "Energy Absorption":   f"{props['EA_vol_MJm3']:.5f} MJ/m³",
        "Specific EA":         f"{sea:.3f} kJ/kg",
        "Manufacturability":   f"{manuf:.4f} / 1.00",
        "Printable":           "✅ Yes" if printable else f"❌ No — wall {wall_mm:.3f} mm < {pp_min:.2f} mm",
    }
    st.dataframe(
        pd.DataFrame([design_row]).T.rename(columns={0: "Value"}),
        use_container_width=True,
    )

    st.markdown('<p class="section-label">Cell Size Sensitivity</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">The dashed reference line marks the default 4 mm cell size. '
        'Hover to inspect predicted values at any cell size.</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        chart_cell_size(model, scaler, tpms, material, process, rho, meta),
        use_container_width=True,
    )

# ── Tab 1: Inverse Design ─────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<p class="section-label">Automated TPMS Selector — Target-Driven Recommendation</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Adjust <b>Stiffness Priority</b> and '
        '<b>Energy Absorption Priority</b> in the sidebar, then click <b>Run Analysis</b> '
        'to update the recommendation.</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        badge  = "success-box" if inv_feasible else "warn-box"
        status = "✅ Feasible" if inv_feasible else "⚠️ Check wall thickness"
        st.markdown(f"""
        <div class="{badge}">
        <b>Recommended Design</b><br>
        TPMS Family: <b>{best_tpms.upper()}</b> &nbsp;·&nbsp;
        Relative Density: <b>{rho_sel:.2f}</b> &nbsp;·&nbsp;
        Cell Size: <b>{cs_sel:.1f} mm</b><br>
        {status}
        </div>""", unsafe_allow_html=True)

        rec_table = {
            "Effective Stiffness":  f"{inv_props['E_eff_GPa']:.4f} GPa",
            "Yield Strength":       f"{inv_props['sigma_y_MPa']:.3f} MPa",
            "Energy Absorption":    f"{inv_props['EA_vol_MJm3']:.4f} MJ/m³",
            "Manufacturability":    f"{inv_manuf:.4f} / 1.00",
        }
        st.dataframe(
            pd.DataFrame([rec_table]).T.rename(columns={0: "Predicted"}),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(chart_inverse_scores(scores, best_tpms, inv_sp, inv_ea),
                        use_container_width=True)

    st.markdown('<p class="section-label">Performance Trade-off — Recommended Design</p>',
                unsafe_allow_html=True)
    col_r1, col_r2 = st.columns([1, 1.65])
    with col_r1:
        st.plotly_chart(chart_radar(inv_all), use_container_width=True)
    with col_r2:
        st.plotly_chart(
            chart_property_bars(
                inv_all,
                title_suffix=f"ρ*={rho_sel:.2f} · {cs_sel:.1f} mm",
            ),
            use_container_width=True,
        )

# ── Tab 2: Density Sweep ─────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<p class="section-label">Property Sweep vs Relative Density</p>',
                unsafe_allow_html=True)
    st.plotly_chart(chart_density_sweep(model, scaler, material, process, meta),
                    use_container_width=True)
    st.plotly_chart(chart_sea(model, scaler, material, process, meta),
                    use_container_width=True)

    with st.expander("📋 View sweep data table"):
        sweep_rows = []
        for rho_i in np.round(np.linspace(0.05, 0.60, 23), 3):
            p = predict_properties(model, scaler, tpms, material, process, rho_i, cs, meta)
            m = compute_manufacturability(tpms, material, process, rho_i, cs)
            s = compute_sea(p, material, rho_i)
            sweep_rows.append({
                "ρ*":                round(rho_i, 3),
                "E* [GPa]":          round(p["E_eff_GPa"], 5),
                "σ_y [MPa]":         round(p["sigma_y_MPa"], 4),
                "EA [MJ/m³]":        round(p["EA_vol_MJm3"], 5),
                "SEA [kJ/kg]":       round(s, 3),
                "Manufacturability": round(m, 4),
            })
        st.dataframe(pd.DataFrame(sweep_rows), use_container_width=True)

# ── Tab 3: Material Comparison ───────────────────────────────────────────
with tabs[3]:
    st.markdown('<p class="section-label">Material Comparison — Fixed Geometry</p>',
                unsafe_allow_html=True)
    st.markdown(
        f'Fixed: **{tpms.capitalize()}** · ρ* = **{rho:.2f}** · '
        f'cell size = **{cs:.1f} mm** · **{PROCESSES[process]["label"]}**'
    )
    st.plotly_chart(
        chart_material_comparison(model, scaler, tpms, process, rho, cs, meta),
        use_container_width=True,
    )
    st.markdown(
        '<div class="info-box">Each bar shows the predicted performance for a different '
        'material with identical lattice geometry. Use this view to select the '
        'best-performing material for your application.</div>',
        unsafe_allow_html=True,
    )

# ── Tab 4: Design Space ───────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<p class="section-label">Pareto Landscape — Stiffness vs Energy Absorption</p>',
                unsafe_allow_html=True)
    st.markdown(
        f'All TPMS families × relative densities 0.05–0.60 × cell sizes 2–6 mm &nbsp;·&nbsp; '
        f'**{MATERIALS[material]["label"]}** · **{PROCESSES[process]["label"]}**'
    )
    with st.spinner("Computing design space…"):
        st.plotly_chart(chart_pareto(model, scaler, material, process, meta),
                        use_container_width=True)
    st.markdown(
        '<div class="info-box">'
        'Each point is a unique design configuration (TPMS family × relative density × cell size). '
        'Colour encodes relative density. Use this chart to identify Pareto-optimal regions '
        'before committing to fabrication. Hover for full details on any point.'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<small style='color:#94a3b8'>"
    "Aerospace TPMS Lattice Design Explorer &nbsp;·&nbsp; "
    "For engineering concept exploration only — "
    "validate all designs with FEA and coupon testing before fabrication."
    "</small>",
    unsafe_allow_html=True,
)