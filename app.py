"""
QuantFlow Institutional HFT & Microstructure Terminal
=====================================================
State-of-the-Art Quantitative Trading Platform powered by Biomimetic Mormyrid
Swarm Consensus Intelligence, Level 2/3 Limit Order Book dynamics, Hawkes point
processes, Swarm-Skewed Avellaneda-Stoikov Market Making, and Free Real-Time Market APIs.
"""

import sys
import os
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Enforce UTF-8 encoding for Windows consoles
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# QuantFlow imports
from models import (
    LimitOrderBook,
    L2Snapshot,
    OrderSide,
    generate_synthetic_lob_stream,
    MultiLevelOFI,
    VPIN,
    StoikovMicroPrice,
    HawkesProcess,
    MormyridSwarmConsensusEngine,
    SwarmConsensusSignal,
    AgentRole,
    FishAgent,
    SwarmAvellanedaStoikov,
    MarketMakerQuotes,
    AlmgrenChrissExecution,
    AlgorithmicRouter,
    HFTSimulator,
    SimulationResult,
    HFTRiskEngine,
    BlackScholesModel,
    BinomialTreeModel,
    MonteCarloSimulation,
    GreeksCalculator,
)
from data.realtime_feed import realtime_feed, RealtimeQuote
from utils import config, format_currency, format_percentage

# Page Configuration
st.set_page_config(
    page_title="QuantFlow - Institutional HFT & Swarm Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Institutional Dark Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Outfit:wght@300;400;600;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d111e 0%, #06080f 100%);
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Typography & Hierarchy */
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .terminal-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00F5A0 0%, #00D9F5 50%, #7B61FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(0, 245, 160, 0.25);
        margin-bottom: 0.2rem;
    }
    
    .terminal-sub {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Glassmorphism Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
    }
    
    div[data-testid="stMetricLabel"] p {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    div[data-testid="stMetricValue"] div {
        color: #f8fafc !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    
    /* Regime Badges */
    .badge-momentum {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-reversion {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-stable {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-toxic {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-live {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(0, 245, 160, 0.15);
        color: #00F5A0;
        border: 1px solid rgba(0, 245, 160, 0.4);
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Code & Tables */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Cache / Session State
if 'simulation_cache' not in st.session_state:
    st.session_state.simulation_cache = None
if 'active_ticker' not in st.session_state:
    st.session_state.active_ticker = "NVDA"


# Helper function to generate or retrieve live LOB stream using free public APIs
@st.cache_data(ttl=30)
def get_market_simulation_data(ticker: str, n_ticks: int = 300, seed: int = 42, provider_mode: str = "auto"):
    if provider_mode != "synthetic":
        preferred = "marketstack" if "Marketstack" in provider_mode else "auto"
        snapshots, df, quote = realtime_feed.get_market_stream(
            ticker=ticker,
            n_ticks=n_ticks,
            seed=seed,
            preferred_source=preferred
        )
        return snapshots, df, quote
    else:
        initial_prices = {
            "NVDA": 140.0, "AAPL": 225.0, "TSLA": 210.0,
            "MSFT": 420.0, "SPY": 560.0, "QQQ": 480.0, "BTC-USD": 62500.0, "ETH-USD": 3400.0
        }
        s0 = initial_prices.get(ticker, 140.0)
        snapshots, df = generate_synthetic_lob_stream(
            n_ticks=n_ticks,
            initial_price=s0,
            annual_vol=0.35,
            tick_size=0.01,
            seed=seed,
        )
        quote = RealtimeQuote(
            ticker=ticker,
            price=s0,
            bid=round(s0 - 0.01, 2),
            ask=round(s0 + 0.01, 2),
            spread=0.02,
            volume=500000.0,
            timestamp=datetime.now(),
            source="Synthetic Generator",
            is_live=False
        )
        return snapshots, df, quote


def run_full_swarm_pipeline(snapshots: List[L2Snapshot]):
    """Execute Mormyrid Swarm and high-frequency signal pipeline over snapshots"""
    swarm_engine = MormyridSwarmConsensusEngine(n_scouts=6, n_predators=8, n_schoolers=10, n_sentinels=4)
    ofi_calc = MultiLevelOFI(depth_levels=5)
    vpin_calc = VPIN(bucket_size=300.0)
    hawkes = HawkesProcess(alpha=0.65, beta=1.8, mu=0.9)

    records = []
    event_times = []
    agent_history = []

    for i, snap in enumerate(snapshots):
        t = snap.timestamp
        mid = snap.mid_price
        
        ofi = ofi_calc.update(snap)
        if len(records) > 0:
            vpin_calc.update_trade(mid, snap.total_bid_depth * 0.08, records[-1]["mid_price"])
        vpin = vpin_calc.get_vpin()

        event_times.append(t)
        hawkes_intensity = hawkes.intensity(t, np.array(event_times))
        branching = hawkes.branching_ratio()

        micro_dev_bps = ((snap.micro_price - mid) / mid) * 10000.0
        rel_spread_bps = (snap.spread / mid) * 10000.0

        signal: SwarmConsensusSignal = swarm_engine.step_market_state(
            ofi=ofi,
            vpin=vpin,
            hawkes_intensity=hawkes_intensity,
            micro_price_dev=micro_dev_bps,
            relative_spread=rel_spread_bps,
            hawkes_branching_ratio=branching,
        )

        records.append({
            "timestamp": t,
            "mid_price": mid,
            "micro_price": snap.micro_price,
            "spread": snap.spread,
            "ofi": ofi,
            "vpin": vpin,
            "hawkes_intensity": hawkes_intensity,
            "drift_bps": signal.predicted_drift_bps,
            "jump_prob": signal.jump_probability,
            "crowding_index": signal.market_crowding_index,
            "adverse_risk": signal.adverse_selection_risk,
            "confidence": signal.swarm_confidence,
            "regime": signal.dominant_regime,
            "quote_skew": signal.optimal_quote_skew,
        })
        
        if i == len(snapshots) - 1:
            agent_history = signal.agent_states

    return pd.DataFrame(records), signal, agent_history


def main():
    # Sidebar Controls
    st.sidebar.markdown("### **QuantFlow Engine Controls**")
    
    ticker_choice = st.sidebar.selectbox(
        "Target Instrument",
        ["NVDA", "AAPL", "TSLA", "MSFT", "SPY", "QQQ", "BTC-USD", "ETH-USD"],
        index=0
    )
    st.session_state.active_ticker = ticker_choice

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### **Real-Time Data Feed**")
    provider_mode = st.sidebar.selectbox(
        "Market Data Source",
        [
            "Auto (Marketstack + Binance L2 + Yahoo Live)",
            "Marketstack API (Official Key)",
            "Yahoo Finance Live",
            "Synthetic Simulation",
        ],
        index=0
    )
    
    with st.sidebar.expander("API Key Configuration", expanded=False):
        st.caption("Active Marketstack Key:")
        st.code("24b40dae0167960b6bd3ec0ce5dfd4f9", language="text")
    
    if st.sidebar.button("Refresh Live Quote", use_container_width=True):
        st.cache_data.clear()

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### **Simulation Parameters**")
    n_ticks = st.sidebar.slider("LOB Tick Horizon", min_value=100, max_value=800, value=300, step=50)
    sim_seed = st.sidebar.number_input("Random Seed", value=42, step=1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### **Mormyrid Swarm Setup**")
    st.sidebar.caption("Biomimetic active electrolocation swarm with Jamming Avoidance Response (JAR).")
    n_scouts = st.sidebar.slider("Scout Agents (Depth Discovery)", 2, 12, 6)
    n_predators = st.sidebar.slider("Predator Agents (Hawkes Momentum)", 2, 16, 8)
    n_schoolers = st.sidebar.slider("Schooler Agents (Mean-Reversion)", 4, 20, 10)
    n_sentinels = st.sidebar.slider("Sentinel Agents (VPIN Toxicity)", 2, 8, 4)

    # Load data and run pipeline
    snapshots, raw_df, live_quote = get_market_simulation_data(
        ticker=ticker_choice,
        n_ticks=n_ticks,
        seed=sim_seed,
        provider_mode=provider_mode,
    )
    pipeline_df, latest_signal, latest_agents = run_full_swarm_pipeline(snapshots)
    latest_snap = snapshots[-1]

    # Header section
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown('<div class="terminal-title">QUANTFLOW HFT TERMINAL</div>', unsafe_allow_html=True)
        feed_label = f"[{live_quote.source}]"
        st.markdown(
            f'<div class="terminal-sub">Institutional Market Microstructure & Biomimetic Swarm Intelligence | <b>{ticker_choice}</b> @ ${latest_snap.mid_price:.2f} <span class="badge-live">{feed_label}</span></div>',
            unsafe_allow_html=True
        )
    with col_h2:
        regime = latest_signal.dominant_regime
        badge_class = (
            "badge-momentum" if regime == "HAWKES_MOMENTUM"
            else "badge-toxic" if regime == "TOXIC_DRAIN"
            else "badge-reversion" if regime == "MEAN_REVERSION"
            else "badge-stable"
        )
        st.markdown(f"""
        <div style="text-align: right; margin-top: 10px;">
            <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 4px;">ACTIVE MICRO-REGIME</div>
            <span class="{badge_class}">[{regime}]</span>
        </div>
        """, unsafe_allow_html=True)

    # Global KPI Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("LOB Mid Price", f"${latest_snap.mid_price:.2f}", f"Spread: ${latest_snap.spread:.2f}")
    with kpi2:
        st.metric("Stoikov Micro-Price", f"${latest_snap.micro_price:.2f}", f"Dev: {(latest_snap.micro_price - latest_snap.mid_price)*100:.1f}c")
    with kpi3:
        st.metric("Swarm Drift Forecast", f"{latest_signal.predicted_drift_bps:+.2f} bps", f"Conf: {latest_signal.swarm_confidence*100:.0f}%")
    with kpi4:
        st.metric("Hawkes Jump Prob", f"{latest_signal.jump_probability*100:.1f}%", f"VPIN: {pipeline_df['vpin'].iloc[-1]:.2f}")
    with kpi5:
        st.metric("JAR Crowding Index", f"{latest_signal.market_crowding_index*100:.1f}%", f"Jammed: {latest_signal.jammed_agent_ratio*100:.0f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Tabs Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Live LOB & Microstructure",
        "Mormyrid Swarm Matrix",
        "Swarm HFT Market Maker",
        "Algorithmic Backtester",
        "Options Intelligence & Kernels",
    ])

    # ==========================================
    # TAB 1: LIVE LOB & MICROSTRUCTURE DEPTH
    # ==========================================
    with tab1:
        st.markdown("### **Level 2 Limit Order Book Depth & Microstructural Pressure**")
        
        col_lob1, col_lob2 = st.columns([3, 2])
        
        with col_lob1:
            # Interactive Level 2 Depth Ladder
            bids_p = latest_snap.bid_prices[:8]
            bids_v = latest_snap.bid_volumes[:8]
            asks_p = latest_snap.ask_prices[:8]
            asks_v = latest_snap.ask_volumes[:8]

            fig_lob = go.Figure()
            # Bids (Green horizontal bars)
            fig_lob.add_trace(go.Bar(
                y=[f"${p:.2f}" for p in bids_p],
                x=bids_v,
                orientation='h',
                name='Bid Depth (Liquidity)',
                marker=dict(color='rgba(16, 185, 129, 0.75)', line=dict(color='#10b981', width=1)),
                text=[f"{int(v):,} shs" for v in bids_v],
                textposition='inside',
            ))
            # Asks (Red horizontal bars)
            fig_lob.add_trace(go.Bar(
                y=[f"${p:.2f}" for p in asks_p],
                x=asks_v,
                orientation='h',
                name='Ask Depth (Liquidity)',
                marker=dict(color='rgba(239, 68, 68, 0.75)', line=dict(color='#ef4444', width=1)),
                text=[f"{int(v):,} shs" for v in asks_v],
                textposition='inside',
            ))

            fig_lob.update_layout(
                title=f"Level 2 Order Book Depth Ladder ({ticker_choice} - {live_quote.source})",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=380,
                margin=dict(l=40, r=20, t=50, b=30),
                barmode='group',
                xaxis=dict(title="Aggregate Limit Volume", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Price Level", autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_lob, use_container_width=True)

        with col_lob2:
            st.markdown("#### **Microstructure Pressure Gauges**")
            
            # Multi-level OFI Gauge
            current_ofi = pipeline_df["ofi"].iloc[-1]
            fig_ofi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_ofi,
                title={'text': "Multi-Level OFI (Order Flow Imbalance)", 'font': {'size': 14, 'color': '#e2e8f0'}},
                gauge={
                    'axis': {'range': [-1.0, 1.0], 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#00F5A0" if current_ofi >= 0 else "#EF4444"},
                    'steps': [
                        {'range': [-1.0, -0.4], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [-0.4, 0.4], 'color': "rgba(100, 116, 139, 0.15)"},
                        {'range': [0.4, 1.0], 'color': "rgba(16, 185, 129, 0.2)"},
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 3},
                        'thickness': 0.75,
                        'value': current_ofi
                    }
                }
            ))
            fig_ofi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=180,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_ofi, use_container_width=True)

            # VPIN Toxicity Gauge
            current_vpin = pipeline_df["vpin"].iloc[-1]
            fig_vpin = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_vpin,
                title={'text': "VPIN (Probability of Toxicity)", 'font': {'size': 14, 'color': '#e2e8f0'}},
                gauge={
                    'axis': {'range': [0.0, 1.0], 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#F59E0B" if current_vpin > 0.5 else "#3B82F6"},
                    'steps': [
                        {'range': [0.0, 0.4], 'color': "rgba(59, 130, 246, 0.2)"},
                        {'range': [0.4, 0.7], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [0.7, 1.0], 'color': "rgba(239, 68, 68, 0.3)"},
                    ],
                }
            ))
            fig_vpin.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=180,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_vpin, use_container_width=True)

        # Micro-Price vs Mid-Price Stream
        st.markdown("#### **High-Frequency Stoikov Micro-Price & Hawkes Intensity Stream**")
        fig_stream = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_stream.add_trace(
            go.Scatter(x=pipeline_df["timestamp"], y=pipeline_df["mid_price"], name="Mid Price", line=dict(color="#38bdf8", width=1.5)),
            secondary_y=False,
        )
        fig_stream.add_trace(
            go.Scatter(x=pipeline_df["timestamp"], y=pipeline_df["micro_price"], name="Stoikov Micro-Price", line=dict(color="#00f5a0", width=1.5, dash="dot")),
            secondary_y=False,
        )
        fig_stream.add_trace(
            go.Scatter(x=pipeline_df["timestamp"], y=pipeline_df["hawkes_intensity"], name="Hawkes Arrival Intensity", line=dict(color="#f43f5e", width=1.2)),
            secondary_y=True,
        )
        
        fig_stream.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=300,
            margin=dict(l=40, r=40, t=20, b=30),
            xaxis=dict(title="Timestamp (Seconds)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Price ($)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis2=dict(title="Hawkes Intensity", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_stream, use_container_width=True)

    # ==========================================
    # TAB 2: MORMYRID SWARM INTELLIGENCE MATRIX
    # ==========================================
    with tab2:
        st.markdown("### **Biomimetic Mormyrid Swarm Consensus Intelligence**")
        st.caption(
            "Inspired by weakly electric fish (Mormyridae) active electrolocation and Jamming Avoidance Response (JAR). "
            "Agents emit high-frequency probe pulses to resolve murky LOB dynamics, reach decentralized consensus, and detect impending Hawkes jumps."
        )

        col_sw1, col_sw2 = st.columns([3, 2])
        
        with col_sw1:
            # 2D/3D Feature Space Scatter of Fish Agents
            if latest_agents:
                agent_df = pd.DataFrame([
                    {
                        "Agent ID": a.agent_id,
                        "Role": a.role.value.upper(),
                        "OFI Perception (X)": a.position[0],
                        "VPIN Perception (Y)": a.position[1],
                        "Hawkes Perception (Z)": a.position[2],
                        "EOD Frequency (Hz)": a.eod_frequency,
                        "Fitness": a.perceived_fitness,
                        "Confidence": a.confidence,
                        "Jammed Status": "JAMMED (JAR Active)" if a.is_jammed else "SYNCHRONIZED",
                    }
                    for a in latest_agents
                ])

                role_color_map = {
                    "SCOUT": "#38BDF8",      # Light Blue
                    "PREDATOR": "#EF4444",   # Red
                    "SCHOOLER": "#10B981",   # Green
                    "SENTINEL": "#F59E0B",   # Amber
                }

                fig_agents = px.scatter(
                    agent_df,
                    x="OFI Perception (X)",
                    y="VPIN Perception (Y)",
                    color="Role",
                    size="Confidence",
                    hover_data=["Agent ID", "EOD Frequency (Hz)", "Fitness", "Jammed Status"],
                    color_discrete_map=role_color_map,
                    title="Active Electrolocation Swarm Field (LOB Feature Space)",
                )
                
                # Add market true state marker
                fig_agents.add_trace(go.Scatter(
                    x=[pipeline_df["ofi"].iloc[-1]],
                    y=[pipeline_df["vpin"].iloc[-1]],
                    mode="markers+text",
                    name="True Market State",
                    text=["[Market State]"],
                    textposition="top center",
                    marker=dict(size=16, color="#ffffff", symbol="star", line=dict(color="#00F5A0", width=2))
                ))

                fig_agents.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,23,42,0.6)",
                    height=420,
                    margin=dict(l=40, r=20, t=50, b=30),
                    xaxis=dict(title="Order Flow Imbalance Subspace (OFI)", gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title="Toxicity Subspace (VPIN)", gridcolor="rgba(255,255,255,0.05)"),
                )
                st.plotly_chart(fig_agents, use_container_width=True)

        with col_sw2:
            st.markdown("#### **Swarm Telemetry & Consensus Voting**")
            
            # Role contribution bar chart
            role_contribs = latest_signal.role_contributions
            fig_roles = go.Figure(go.Bar(
                x=list(role_contribs.keys()),
                y=list(role_contribs.values()),
                marker_color=["#38BDF8", "#EF4444", "#10B981", "#F59E0B"],
            ))
            fig_roles.update_layout(
                title="Decentralized Role Signal Contributions",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=200,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(title="Signal Bias", gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_roles, use_container_width=True)

            # Jamming Avoidance Radar / Breakdown
            st.markdown("""
            <div style="background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px; font-size: 0.9rem;">
                <b>Biological Jamming Avoidance (JAR) Status:</b><br>
                - Frequency Band: 40 Hz - 180 Hz<br>
                - Active Desynchronization: Repulsive force applied to crowded queues.<br>
                - Adverse Selection Protection: Anti-herd slippage mitigation active.<br>
            </div>
            """, unsafe_allow_html=True)

        # Swarm Consensus Historical Drift Forecast
        fig_drift = go.Figure()
        fig_drift.add_trace(go.Scatter(
            x=pipeline_df["timestamp"],
            y=pipeline_df["drift_bps"],
            name="Swarm Drift Forecast (bps)",
            line=dict(color="#00F5A0", width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 245, 160, 0.1)'
        ))
        fig_drift.add_trace(go.Scatter(
            x=pipeline_df["timestamp"],
            y=pipeline_df["jump_prob"] * 30.0,
            name="Jump Risk Index (Scaled)",
            line=dict(color="#EF4444", width=1.5, dash="dash")
        ))
        fig_drift.update_layout(
            title="Continuous Swarm Drift Forecast & Hawkes Jump Probability",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=250,
            margin=dict(l=40, r=20, t=40, b=30),
            xaxis=dict(title="Timestamp (Seconds)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Predicted Drift (bps)", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_drift, use_container_width=True)

    # ==========================================
    # TAB 3: SWARM HFT MARKET MAKING
    # ==========================================
    with tab3:
        st.markdown("### **Swarm-Skewed Avellaneda-Stoikov High-Frequency Market Maker**")
        st.caption(
            "Optimal quotes derived by modulating classical inventory risk aversion with real-time Mormyrid Swarm consensus drift and JAR spread widening."
        )

        col_mm1, col_mm2 = st.columns([1, 2])
        
        with col_mm1:
            st.markdown("#### **Market Maker Parameters**")
            gamma = st.slider("Inventory Risk Aversion (gamma)", 0.01, 1.0, 0.15, 0.01)
            kappa = st.slider("Order Arrival Liquidity (kappa)", 0.5, 5.0, 1.8, 0.1)
            curr_inv = st.slider("Current Inventory (q)", -50, 50, 0, 5)
            swarm_mult = st.slider("Swarm Skew Multiplier", 0.0, 3.0, 1.5, 0.1)

            as_model = SwarmAvellanedaStoikov(
                gamma=gamma,
                kappa=kappa,
                sigma=0.35,
                swarm_skew_multiplier=swarm_mult,
            )

            quotes = as_model.calculate_quotes(
                mid_price=latest_snap.mid_price,
                inventory=curr_inv,
                time_remaining=0.5,
                swarm_drift_bps=latest_signal.predicted_drift_bps,
                jar_crowding_index=latest_signal.market_crowding_index,
                adverse_selection_risk=latest_signal.adverse_selection_risk,
                hawkes_intensity=pipeline_df["hawkes_intensity"].iloc[-1],
            )

        with col_mm2:
            st.markdown("#### **Live Quote Ladder & Reservation Price**")
            
            # Metrics
            q_col1, q_col2, q_col3 = st.columns(3)
            with q_col1:
                st.metric("Optimal Bid Quote", f"${quotes.bid_price:.2f}", f"P(Fill): {quotes.bid_fill_probability*100:.1f}%")
            with q_col2:
                st.metric("Reservation Price r(s,q,t)", f"${quotes.reservation_price:.2f}", f"Skew: {quotes.swarm_skew_bps:+.1f} bps")
            with q_col3:
                st.metric("Optimal Ask Quote", f"${quotes.ask_price:.2f}", f"P(Fill): {quotes.ask_fill_probability*100:.1f}%")

            # Quote Ladder Visualization
            fig_ladder = go.Figure()
            
            # Mid price reference line
            fig_ladder.add_vline(x=quotes.mid_price, line_width=2, line_dash="dash", line_color="#94a3b8", annotation_text="Mid")
            fig_ladder.add_vline(x=quotes.reservation_price, line_width=2, line_color="#7B61FF", annotation_text="Reservation")

            fig_ladder.add_trace(go.Bar(
                name="Bid Quote Offset",
                y=["Market Maker Quotes"],
                x=[quotes.mid_price - quotes.bid_price],
                base=quotes.bid_price,
                orientation='h',
                marker_color='rgba(16, 185, 129, 0.8)',
                text=[f"Bid: ${quotes.bid_price:.2f}"],
                textposition='inside',
            ))
            fig_ladder.add_trace(go.Bar(
                name="Ask Quote Offset",
                y=["Market Maker Quotes"],
                x=[quotes.ask_price - quotes.mid_price],
                base=quotes.mid_price,
                orientation='h',
                marker_color='rgba(239, 68, 68, 0.8)',
                text=[f"Ask: ${quotes.ask_price:.2f}"],
                textposition='inside',
            ))

            fig_ladder.update_layout(
                title="Dynamic Bid / Ask Spread Asymmetry",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=180,
                margin=dict(l=40, r=40, t=40, b=20),
                xaxis=dict(title="Price ($)", range=[quotes.bid_price - 0.1, quotes.ask_price + 0.1]),
                barmode='overlay',
            )
            st.plotly_chart(fig_ladder, use_container_width=True)

            # Fill Probability Surface
            offsets = np.linspace(0.01, 0.15, 30)
            p_fills = np.exp(-kappa * (offsets / quotes.mid_price) * 100)
            
            fig_fill = go.Figure()
            fig_fill.add_trace(go.Scatter(
                x=offsets * 100,
                y=p_fills * 100,
                mode='lines+markers',
                line=dict(color="#38BDF8", width=2),
                name="Fill Probability vs Spread Offset"
            ))
            fig_fill.update_layout(
                title="Arrival Fill Probability Decay vs Distance from Mid (cents)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=200,
                margin=dict(l=40, r=20, t=40, b=30),
                xaxis=dict(title="Offset from Mid (cents)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_fill, use_container_width=True)

    # ==========================================
    # TAB 4: ALGORITHMIC BACKTESTER
    # ==========================================
    with tab4:
        st.markdown("### **Event-Driven High-Frequency Strategy Backtesting & Telemetry**")
        
        sim_harness = HFTSimulator(initial_cash=100000.0)
        
        col_bt_btn, col_bt_info = st.columns([1, 3])
        with col_bt_btn:
            run_bt = st.button("Run Comparative Backtest", type="primary", use_container_width=True)

        # Run 3 strategies: Swarm AS, Classical AS, Momentum Predator
        res_swarm = sim_harness.run_simulation(snapshots, strategy_type="swarm_as", as_model=as_model)
        res_class = sim_harness.run_simulation(snapshots, strategy_type="classical_as", as_model=as_model)
        res_momen = sim_harness.run_simulation(snapshots, strategy_type="momentum_predator")

        # Comparative KPI Cards
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Swarm AS PnL", f"${res_swarm.total_pnl:,.2f}", f"Sharpe: {res_swarm.sharpe_ratio:.2f}")
        with b2:
            st.metric("Classical AS PnL", f"${res_class.total_pnl:,.2f}", f"Sharpe: {res_class.sharpe_ratio:.2f}")
        with b3:
            st.metric("Momentum Predator PnL", f"${res_momen.total_pnl:,.2f}", f"Sharpe: {res_momen.sharpe_ratio:.2f}")
        with b4:
            st.metric("Swarm Max Drawdown", f"${res_swarm.max_drawdown:,.2f}", f"Win Rate: {res_swarm.win_rate*100:.1f}%")

        # Cumulative PnL Comparison Chart
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=list(range(len(res_swarm.pnl_series))),
            y=res_swarm.pnl_series,
            name="Swarm-Skewed AS Market Maker",
            line=dict(color="#00F5A0", width=2.5)
        ))
        fig_pnl.add_trace(go.Scatter(
            x=list(range(len(res_class.pnl_series))),
            y=res_class.pnl_series,
            name="Classical Avellaneda-Stoikov",
            line=dict(color="#94A3B8", width=1.5, dash="dash")
        ))
        fig_pnl.add_trace(go.Scatter(
            x=list(range(len(res_momen.pnl_series))),
            y=res_momen.pnl_series,
            name="Hawkes Momentum Predator",
            line=dict(color="#F43F5E", width=1.5, dash="dot")
        ))

        fig_pnl.update_layout(
            title="Cumulative Mark-to-Market PnL Progression ($)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=360,
            margin=dict(l=40, r=20, t=50, b=30),
            xaxis=dict(title="Event Tick Index", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Cumulative PnL ($)", gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        # Inventory Path & Fills
        col_inv, col_fill = st.columns([1, 1])
        with col_inv:
            fig_inv = go.Figure()
            fig_inv.add_trace(go.Scatter(
                x=list(range(len(res_swarm.inventory_series))),
                y=res_swarm.inventory_series,
                name="Swarm AS Inventory",
                line=dict(color="#38BDF8", width=1.5),
                fill='tozeroy',
                fillcolor='rgba(56, 189, 248, 0.1)'
            ))
            fig_inv.update_layout(
                title="Market Maker Inventory Path (Shares)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                height=220,
                margin=dict(l=40, r=20, t=40, b=20),
                yaxis=dict(title="Inventory (q)", gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_inv, use_container_width=True)

        with col_fill:
            st.markdown("#### **Recent Simulated Fills**")
            if not res_swarm.fill_log.empty:
                st.dataframe(res_swarm.fill_log.tail(8), use_container_width=True, height=180)
            else:
                st.info("No fills recorded yet.")

    # ==========================================
    # TAB 5: OPTIONS INTELLIGENCE & NATIVE ENGINES
    # ==========================================
    with tab5:
        st.markdown("### **Options Pricing Suite & Native Kernels**")
        
        c_opt1, c_opt2, c_opt3 = st.columns(3)
        with c_opt1:
            opt_strike = st.number_input("Strike Price ($)", value=float(round(latest_snap.mid_price, 2)), step=1.0)
        with c_opt2:
            opt_days = st.slider("Days to Expiry", 5, 180, 45)
            opt_t = opt_days / 365.0
        with c_opt3:
            opt_vol = st.slider("Implied Volatility (sigma)", 0.10, 1.20, 0.35, 0.05)

        # Calculate Option Pricing Models
        bs = BlackScholesModel(S=latest_snap.mid_price, K=opt_strike, T=opt_t, r=0.05, sigma=opt_vol)
        bin_model = BinomialTreeModel(S=latest_snap.mid_price, K=opt_strike, T=opt_t, r=0.05, sigma=opt_vol, n_steps=50)
        mc_model = MonteCarloSimulation(S=latest_snap.mid_price, K=opt_strike, T=opt_t, r=0.05, sigma=opt_vol, n_simulations=10000)

        bs_call = bs.price('call')
        bs_put = bs.price('put')
        bin_call = bin_model.price('call')
        mc_call = mc_model.price('call')['price']

        ensemble_call = (bs_call + bin_call + mc_call) / 3.0

        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            st.metric("Ensemble Fair Value (Call)", f"${ensemble_call:.2f}")
        with p_col2:
            st.metric("Black-Scholes Call", f"${bs_call:.2f}")
        with p_col3:
            st.metric("Binomial Tree Call", f"${bin_call:.2f}")
        with p_col4:
            st.metric("Monte Carlo Call", f"${mc_call:.2f}")

        # Greeks Section
        st.markdown("#### **First & Second Order Greeks Sensitivity**")
        greeks = bs.all_greeks('call')
        
        g1, g2, g3, g4, g5 = st.columns(5)
        with g1:
            st.metric("Delta", f"{greeks['delta']:.3f}")
        with g2:
            st.metric("Gamma", f"{greeks['gamma']:.4f}")
        with g3:
            st.metric("Theta", f"${greeks['theta']:.3f}/day")
        with g4:
            st.metric("Vega", f"${greeks['vega']:.3f}/1% vol")
        with g5:
            st.metric("Rho", f"${greeks['rho']:.3f}/1% rate")

        # Almgren-Chriss Optimal Liquidation Trajectory
        st.markdown("#### **Almgren-Chriss Optimal Liquidation Curve**")
        ac_exec = AlmgrenChrissExecution(total_shares=10000.0, time_horizon=1.0, num_intervals=15, volatility=opt_vol, initial_price=latest_snap.mid_price)
        traj = ac_exec.calculate_trajectory()

        fig_ac = go.Figure()
        fig_ac.add_trace(go.Scatter(
            x=traj.times * 60,
            y=traj.holdings,
            mode='lines+markers',
            name="Optimal Inventory Liquidation",
            line=dict(color="#00F5A0", width=2.5)
        ))
        fig_ac.update_layout(
            title="Optimal Risk-Averse Liquidation Schedule (Shares vs Time)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=260,
            margin=dict(l=40, r=20, t=40, b=30),
            xaxis=dict(title="Execution Time (Minutes)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Remaining Holdings (Shares)", gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig_ac, use_container_width=True)


if __name__ == "__main__":
    main()
