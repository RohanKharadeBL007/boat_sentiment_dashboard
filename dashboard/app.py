"""
Interactive decision-support dashboard for boAt sentiment & theme analysis.

Run with:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from pipeline.agent_memory import (
        get_previous_run,
        get_run_history,
        list_pending_alerts,
        update_alert_status,
    )
except ImportError:
    get_previous_run = lambda *args: None
    get_run_history = lambda *args: []
    list_pending_alerts = lambda *args: []
    update_alert_status = lambda *args: False

try:
    from pipeline.insight_agent import _clean_theme_name
except ImportError:
    def _clean_theme_name(t): return str(t).replace(" / ", ", ").title()


# --- Page Config ---
st.set_page_config(
    page_title="boAt | Hardware Intelligence & Sentiment Hub",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- uAvionix Inspired Modern CSS Styling & Scroll Animations ---
st.markdown("""
<head>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
</head>
<style>
    /* Global Typography & Deep Avionics Background */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stApp {
        background: radial-gradient(120% 120% at 50% 5%, #160205 0%, #080a0e 45%, #020305 100%);
        background-attachment: fixed;
        color: #f1f5f9;
    }
    
    /* uAvionix Top Sticky Telemetry Bar */
    .uav-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(10, 14, 22, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 51, 51, 0.25);
        border-radius: 40px;
        padding: 10px 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(230, 0, 0, 0.15);
    }
    .uav-brand {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 2px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .uav-brand-dot {
        width: 10px;
        height: 10px;
        background-color: #ff3333;
        border-radius: 50%;
        box-shadow: 0 0 10px #ff3333;
        animation: pulseDot 1.5s infinite;
    }
    @keyframes pulseDot {
        0% { opacity: 0.4; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.2); }
        100% { opacity: 0.4; transform: scale(0.9); }
    }
    .uav-telemetry {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
        letter-spacing: 1px;
    }
    .uav-telemetry span {
        color: #00e676;
        font-weight: 700;
    }

    /* Scroll Animations */
    @keyframes uavFadeUp {
        from { opacity: 0; transform: translateY(35px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .scroll-reveal {
        animation: uavFadeUp 0.85s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Hero Banner */
    .boat-hero {
        background: linear-gradient(135deg, rgba(230, 0, 0, 0.95) 0%, rgba(110, 0, 0, 0.9) 60%, rgba(15, 3, 6, 0.95) 100%);
        backdrop-filter: blur(16px);
        padding: 40px;
        border-radius: 24px;
        margin-bottom: 32px;
        border: 1px solid rgba(255, 77, 77, 0.4);
        box-shadow: 0 25px 50px rgba(230, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .boat-hero h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700;
        font-size: 2.6rem;
        letter-spacing: -0.5px;
        margin-top: 10px;
        color: #ffffff !important;
    }
    .boat-hero p {
        font-size: 1.15rem;
        color: #f1f5f9 !important;
        max-width: 800px;
    }

    /* HD Product Showcase Glass Cards */
    .product-showcase-card {
        background: linear-gradient(145deg, rgba(22, 28, 42, 0.7) 0%, rgba(12, 16, 25, 0.85) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 20px;
        padding: 20px;
        transition: all 0.4s ease;
        height: 100%;
    }
    .product-showcase-card:hover {
        transform: translateY(-8px);
        border-color: rgba(255, 51, 51, 0.5);
        box-shadow: 0 16px 36px rgba(230, 0, 0, 0.3);
    }
    .product-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.25rem;
        color: #ffffff;
        margin-top: 14px;
        margin-bottom: 6px;
    }
    .product-spec {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #ff4d4d;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .product-desc {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, rgba(26, 32, 46, 0.75) 0%, rgba(14, 18, 28, 0.85) 100%);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 22px 20px;
        transition: all 0.35s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: rgba(230, 0, 0, 0.5);
        box-shadow: 0 12px 30px rgba(230, 0, 0, 0.25);
    }
    div[data-testid="stMetricValue"] {
        color: #ff3333 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }

    /* Chart Cards */
    .chart-card {
        background: linear-gradient(145deg, rgba(20, 26, 38, 0.65) 0%, rgba(11, 15, 24, 0.8) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        transition: all 0.35s ease;
    }
    .chart-card:hover {
        border-color: rgba(255, 51, 51, 0.35);
        box-shadow: 0 12px 28px rgba(0,0,0,0.4);
    }

    /* Section Headers */
    .section-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 10px;
        margin-bottom: 18px;
    }
    .section-header span {
        background: linear-gradient(90deg, #ff3333, #ff8080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    hr { border-color: rgba(255, 255, 255, 0.08) !important; margin: 36px 0 !important; }
</style>
""", unsafe_allow_html=True)

# Data paths
LATEST_PATH = Path(__file__).parent.parent / "outputs" / "processed_reviews_latest.csv"
SAMPLE_PATH = Path(__file__).parent.parent / "outputs" / "processed_sample.csv"
DATA_PATH = LATEST_PATH if LATEST_PATH.exists() else SAMPLE_PATH
SUMMARY_PATH = Path(__file__).parent.parent / "outputs" / "executive_summary_latest.txt"

@st.cache_data(ttl=60)
def load_data(path):
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    def fill_product(row):
        p = str(row.get("product", ""))
        if pd.isna(row.get("product")) or p == "nan" or not p.strip():
            txt = str(row.get("review_text", "")).lower()
            if any(k in txt for k in ["rockerz", "headphone", "earphone", "neckband"]):
                return "boAt Rockerz"
            elif any(k in txt for k in ["airdopes", "earbud", "tws", "case", "charging"]):
                return "boAt Airdopes"
            elif any(k in txt for k in ["watch", "wave", "dial", "display", "step", "heart", "smartwatch"]):
                return "boAt Wave Smartwatch"
            return "boAt General / Multi-category"
        return p
    
    df["product"] = df.apply(fill_product, axis=1)
    df["source"] = df["source"].fillna("Other Channel")
    df["sentiment_label"] = df["sentiment_label"].fillna("neutral")
    return df

df = load_data(DATA_PATH)

# --- Top Navigation Telemetry Bar (uAvionix Inspired) ---
st.markdown("""
<div class="uav-navbar scroll-reveal">
    <div class="uav-brand">
        <div class="uav-brand-dot"></div>
        boAt INTELLIGENCE CONTROL
    </div>
    <div class="uav-telemetry">
        SYSTEM: <span>ONLINE</span> &nbsp;|&nbsp; STREAM: <span>8,555 REVIEWS</span> &nbsp;|&nbsp; STATUS: <span>OPTIMAL</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Header Hero Banner ---
st.markdown("""
<div class="boat-hero scroll-reveal">
    <div style="text-transform: uppercase; font-size: 0.8rem; letter-spacing: 2px; font-weight: 700; color: #ffffff; opacity: 0.9;">
        ⚡ HARDWARE-FIRST DECISION INTELLIGENCE
    </div>
    <h1>boAt Executive Sentiment & Product Intelligence Hub</h1>
    <p>High-Definition multi-channel decision support across boAt Rockerz Headphones, Airdopes TWS, and Wave Smartwatches</p>
</div>
""", unsafe_allow_html=True)

# --- HD Product Showcase Section (uAvionix Style Grid) ---
st.markdown("""
<div class="section-header scroll-reveal">
    <span>🎧 Flagship Hardware Lineup — Real-Time Health Overview</span>
</div>
""", unsafe_allow_html=True)

pcol1, pcol2, pcol3 = st.columns(3)
assets_dir = Path(__file__).parent / "assets"

with pcol1:
    st.markdown('<div class="product-showcase-card scroll-reveal">', unsafe_allow_html=True)
    img_path = assets_dir / "boat_rockerz.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    st.markdown("""
    <div class="product-title">boAt Rockerz Series</div>
    <div class="product-spec">HEADPHONES & NECKBANDS • 25H+ BATTERY</div>
    <div class="product-desc">Signature boAt bass acoustic tuning. Primary feedback highlights bass clarity while flagging headband hinge mechanical stress points.</div>
    </div>
    """, unsafe_allow_html=True)

with pcol2:
    st.markdown('<div class="product-showcase-card scroll-reveal">', unsafe_allow_html=True)
    img_path = assets_dir / "boat_airdopes.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    st.markdown("""
    <div class="product-title">boAt Airdopes Series</div>
    <div class="product-spec">TWS EARBUDS • DUAL MIC ENx™</div>
    <div class="product-desc">Ultra-compact magnetic charging case with instant pairing. Primary hardware focus: Dual-Mic ENC for noise isolation in calling.</div>
    </div>
    """, unsafe_allow_html=True)

with pcol3:
    st.markdown('<div class="product-showcase-card scroll-reveal">', unsafe_allow_html=True)
    img_path = assets_dir / "boat_wave.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    st.markdown("""
    <div class="product-title">boAt Wave Smartwatch</div>
    <div class="product-spec">AMOLED DISPLAY • PPG OPTICAL SENSOR</div>
    <div class="product-desc">High-brightness AMOLED displays with multi-day battery backup. Key improvement: High-tensile fluoroelastomer sports straps.</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Sidebar Filters ---
st.sidebar.markdown("## 🎛️ Interactive Telemetry Filters")
st.sidebar.markdown("---")

all_prods = sorted([str(p) for p in df["product"].dropna().unique()])
selected_prods = st.sidebar.multiselect(
    "🎧 Product Line",
    options=all_prods,
    default=all_prods,
    help="Filter feedback by boAt product series"
)

all_sources = sorted([str(s) for s in df["source"].dropna().unique()])
selected_sources = st.sidebar.multiselect(
    "🌐 Feedback Channel",
    options=all_sources,
    default=all_sources
)

all_sentiments = sorted([str(st_lbl) for st_lbl in df["sentiment_label"].dropna().unique()])
selected_sentiments = st.sidebar.multiselect(
    "📊 Sentiment Classification",
    options=all_sentiments,
    default=all_sentiments
)

filtered = df.copy()
if selected_prods:
    filtered = filtered[filtered["product"].isin(selected_prods)]
if selected_sources:
    filtered = filtered[filtered["source"].isin(selected_sources)]
if selected_sentiments:
    filtered = filtered[filtered["sentiment_label"].isin(selected_sentiments)]

# --- Top KPI Metrics ---
st.markdown('<div class="scroll-reveal">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Reviews Analyzed", f"{len(filtered):,}")

pos_pct = (filtered["sentiment_label"] == "positive").mean() * 100 if len(filtered) else 0
neg_pct = (filtered["sentiment_label"] == "negative").mean() * 100 if len(filtered) else 0
avg_score = filtered["sentiment_score"].mean() if len(filtered) else 0

col2.metric("Positive Sentiment", f"{pos_pct:.1f}%")
col3.metric("Hardware Friction", f"{neg_pct:.1f}%")
col4.metric("Avg Sentiment Index", f"{avg_score:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

st.divider()


# --- AI Agent Telemetry & Human-in-the-Loop Hub ---
st.markdown("""
<div class="section-header scroll-reveal">
    <span>🤖 Autonomous AI Agent Telemetry & Human-in-the-Loop Hub</span>
</div>
""", unsafe_allow_html=True)

agent_col1, agent_col2 = st.columns([1.2, 1.8])

with agent_col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("#### 🛡️ Active Anomaly Escalations")
    pending_alerts = list_pending_alerts()
    if pending_alerts:
        for al in pending_alerts:
            sev_color = "#ff1744" if al.get("severity") in ["HIGH", "CRITICAL"] else "#ff9100"
            st.markdown(
                f"""
                <div style="border-left: 4px solid {sev_color}; padding: 10px 14px; background: rgba(255,23,68,0.1); border-radius: 8px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: {sev_color}; font-size: 0.85rem;">{al.get('alert_id')}</span>
                        <span style="background: {sev_color}; color: #fff; font-size: 0.7rem; padding: 2px 8px; border-radius: 12px; font-weight: 700;">{al.get('severity')}</span>
                    </div>
                    <div style="font-size: 0.82rem; margin-top: 6px; color: #f1f5f9;">{al.get('reason')}</div>
                    <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">Created: {al.get('timestamp')[:19]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("✅ Approve Alert", key=f"app_{al['alert_id']}", use_container_width=True):
                    update_alert_status(al["alert_id"], "APPROVED")
                    st.success(f"Alert {al['alert_id']} Approved!")
                    st.rerun()
            with btn_col2:
                if st.button("❌ Dismiss", key=f"rej_{al['alert_id']}", use_container_width=True):
                    update_alert_status(al["alert_id"], "REJECTED")
                    st.info(f"Alert {al['alert_id']} Dismissed.")
                    st.rerun()
    else:
        st.markdown(
            """
            <div style="padding: 16px; background: rgba(0,230,118,0.08); border: 1px solid rgba(0,230,118,0.3); border-radius: 12px; text-align: center;">
                <div style="font-size: 1.6rem;">🟢</div>
                <div style="font-weight: 700; color: #00e676; margin-top: 4px;">All Systems Nominal</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">No critical anomalies requiring human approval.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with agent_col2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("#### 🧠 Agent Memory (Temporal Run History from SQLite)")
    run_history = get_run_history(limit=5)
    if run_history:
        history_rows = []
        for r in run_history:
            history_rows.append({
                "Run ID": r["run_id"],
                "Timestamp": r["timestamp"][:16].replace("T", " "),
                "Reviews": f"{r['n_reviews']:,}",
                "Positive %": f"{r['pct_positive']}%",
                "Negative %": f"{r['pct_negative']}%",
                "Alerts": r.get("alerts_count", 0),
            })
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No prior run history in runs.db yet. Establishing baseline.")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- GenAI Executive Summary ---
if SUMMARY_PATH.exists():
    st.markdown("""
    <div class="section-header scroll-reveal">
        <span>💡 GenAI Executive Synthesis & Hardware Recommendations</span>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.info(SUMMARY_PATH.read_text())
    st.divider()

# Color Map boAt Premium Style
COLOR_MAP = {
    "positive": "#00e676",
    "negative": "#ff1744",
    "neutral": "#90a4ae"
}

# --- Visual Analytics Row 1 ---
st.markdown('<div class="scroll-reveal">', unsafe_allow_html=True)
left, right = st.columns(2)

with left:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>Overall Sentiment Breakdown</span></div>', unsafe_allow_html=True)
    sentiment_counts = filtered["sentiment_label"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    fig1 = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        color="sentiment",
        color_discrete_map=COLOR_MAP,
        hole=0.45
    )
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9", family="Plus Jakarta Sans"),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>Sentiment Distribution by Product Line (%)</span></div>', unsafe_allow_html=True)
    by_prod = filtered.groupby(["product", "sentiment_label"]).size().reset_index(name="count")
    prod_totals = by_prod.groupby("product")["count"].transform("sum")
    by_prod["percentage"] = (by_prod["count"] / prod_totals) * 100
    
    fig2 = px.bar(
        by_prod,
        x="product",
        y="percentage",
        color="sentiment_label",
        barmode="stack",
        color_discrete_map=COLOR_MAP,
        hover_data={"count": True, "percentage": ":.1f}%"}
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9", family="Plus Jakarta Sans"),
        xaxis_title="",
        yaxis_title="Percentage (%)",
        yaxis=dict(range=[0, 100]),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- Visual Analytics Row 2 ---
st.markdown('<div class="scroll-reveal">', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>Recurring Customer Hardware Themes</span></div>', unsafe_allow_html=True)
    if "topic_label" in filtered.columns:
        theme_counts = filtered["topic_label"].value_counts().reset_index().head(8)
        theme_counts.columns = ["theme", "count"]
        # Clean up theme labels for display
        theme_counts["clean_theme"] = theme_counts["theme"].apply(_clean_theme_name)
        fig3 = px.bar(
            theme_counts,
            x="count",
            y="clean_theme",
            orientation="h",
            color_discrete_sequence=["#ff3333"]
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9", family="Plus Jakarta Sans"),
            yaxis_title="",
            xaxis_title="Number of Mentions",
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>Sentiment Breakdown by Channel (%)</span></div>', unsafe_allow_html=True)
    by_source = filtered.groupby(["source", "sentiment_label"]).size().reset_index(name="count")
    source_totals = by_source.groupby("source")["count"].transform("sum")
    by_source["percentage"] = (by_source["count"] / source_totals) * 100
    
    fig4 = px.bar(
        by_source,
        x="source",
        y="percentage",
        color="sentiment_label",
        barmode="stack",
        color_discrete_map=COLOR_MAP,
        hover_data={"count": True, "percentage": ":.1f}%"}
    )
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9", family="Plus Jakarta Sans"),
        xaxis_title="",
        yaxis_title="Percentage (%)",
        yaxis=dict(range=[0, 100]),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- Sentiment Trend ---
if "date" in filtered.columns and filtered["date"].notna().any():
    st.markdown('<div class="chart-card scroll-reveal">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>📈 Sentiment Dynamics Over Time</span></div>', unsafe_allow_html=True)
    trend = filtered.dropna(subset=["date"]).copy()
    trend["date_only"] = trend["date"].dt.date
    trend_agg = trend.groupby(["date_only", "sentiment_label"]).size().reset_index(name="count")
    fig5 = px.line(
        trend_agg,
        x="date_only",
        y="count",
        color="sentiment_label",
        color_discrete_map=COLOR_MAP
    )
    fig5.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9", family="Plus Jakarta Sans"),
        xaxis_title="Date",
        yaxis_title="Daily Reviews",
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

# --- Actionable Flagged Reviews ---
st.markdown('<div class="scroll-reveal">', unsafe_allow_html=True)
st.markdown('<div class="section-header"><span>🚨 Priority Action Items — Severe Negative Hardware Feedback</span></div>', unsafe_allow_html=True)
flagged = filtered[filtered["sentiment_label"] == "negative"].sort_values("sentiment_score").head(15)
st.dataframe(
    flagged[["product", "source", "topic_label", "sentiment_score", "review_text"]],
    use_container_width=True,
    hide_index=True
)
st.markdown('</div>', unsafe_allow_html=True)

# --- Data Explorer ---
with st.expander("🔍 Explore Full Raw Dataset"):
    st.dataframe(filtered, use_container_width=True)


