"""
QuantFlow Interactive Dashboard
Streamlit web application for real-time options analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Import QuantFlow modules
import sys
import os

# Enforce UTF-8 encoding for Windows consoles
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append('.')

from main import QuantFlow
from models import GreeksCalculator, BlackScholesModel
from analysis import ScenarioAnalyzer, PortfolioAnalyzer
from analysis.portfolio_greeks import OptionPosition
from utils import config, format_currency, format_percentage

# Page config
st.set_page_config(
    page_title="QuantFlow - Options Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Calm Professional Theme
import base64
import os

def get_img_as_base64(file_path):
    """Convert image to base64 for embedding"""
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load background image
# Load background image (using minimalist version)
bg_img_path = os.path.join("assets", "background_minimal.png")
if not os.path.exists(bg_img_path):
    # Fallback to standard bg if minimal not found
    bg_img_path = os.path.join("assets", "background.png")

if os.path.exists(bg_img_path):
    img_b64 = get_img_as_base64(bg_img_path)
    bg_style = f"""
    .stApp {{
        background-color: #050510;
        background-image: url("data:image/png;base64,{img_b64}");
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }}
    """
else:
    bg_style = ""

st.markdown(f"""
<style>
    /* RESET and BASE */
    {bg_style}
    
    /* Overlay for readability - darker and cleaner */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at center, rgba(0,0,0,0.5) 0%, rgba(0,0,0,0.92) 100%);
        pointer-events: none;
        z-index: -1;
    }}

    /* TYPOGRAPHY - Enhanced Contrast */
    h1, h2, h3 {{
        font-family: 'Inter', sans-serif;
        color: #fff;
        text-shadow: 0 2px 4px rgba(0,0,0,0.8);
    }}
    
    .main-header {{
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-top: 1rem;
        color: #fff;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.4);
    }}
    
    p, li, .stMarkdown {{
        font-size: 1.05rem;
        color: #e0e0e0 !important; /* Higher contrast text */
    }}

    /* GLASS CARDS - HIGHER OPACITY FOR READABILITY */
    .glass-card, div[data-testid="stMetric"], div[data-testid="stExpander"], .feature-card {{
        background: rgba(15, 20, 35, 0.85) !important; /* Less transparent */
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
    }}
    
    /* FEATURE CARD STYLING */
    .feature-card {{
        text-align: center;
        height: 100%;
        transition: transform 0.3s ease;
    }}
    .feature-card:hover {{
        transform: translateY(-5px);
        border-color: rgba(0, 255, 136, 0.5) !important;
    }}
    .feature-icon {{
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }}
    .feature-title {{
        font-size: 1.4rem;
        font-weight: bold;
        color: #00FF88;
        margin-bottom: 0.5rem;
    }}

</style>
""", unsafe_allow_html=True)
# Onboarding tooltips dictionary
TOOLTIPS = {
    "ensemble": "📊 **Ensemble Pricing**: A weighted average of Black-Scholes, Binomial, and Monte Carlo models to reduce individual model errors and find the 'Mathematical Truth'.",
    "delta": "📈 **Delta**: Your 'Stock Equivalent'—how much the option gains for every $1 move in the stock. Delta of 0.62 = option moves $0.62 per $1 stock move.",
    "gamma": "⚡ **Gamma**: The 'Accelerator'—how fast your Delta changes as the stock price moves. High Gamma = rapidly changing risk exposure.",
    "theta": "⏰ **Theta**: Time decay in dollars per day. This is money you lose every day just from the passage of time.",
    "vega": "🌊 **Vega**: Volatility sensitivity. How much you gain/lose for every 1% change in implied volatility.",
    "var": "⚠️ **VaR (Value at Risk)**: The dollar amount your position could lose with 95% or 99% statistical certainty in normal markets.",
    "cvar": "🔴 **CVaR (Expected Shortfall)**: The average loss in the absolute worst 5% of market outcomes. Your 'tail risk'.",
    "regime": "🎯 **Market Regime**: Our AI identifies the current market environment (Low/High Vol × Bull/Bear) so Greeks can be adjusted for the conditions you actually face.",
    "mispricing": "🔍 **Mispricing Score**: 0-100 scale measuring statistical edge. 80+ means the math is heavily in your favor. Based on XGBoost + SHAP analysis.",
    "vanna": "📊 **Vanna**: How Delta changes as Volatility changes. Important for understanding vol crush scenarios.",
    "charm": "⏱️ **Charm**: How Delta decays as time passes (Delta-bleed). Critical for longer-dated positions."
}

def show_tooltip(key):
    """Display tooltip for a metric"""
    if key in TOOLTIPS:
        st.info(TOOLTIPS[key])

# Initialize session state
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'results' not in st.session_state:
    st.session_state.results = None


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 QuantFlow - AI-Powered Options Intelligence</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #aaa;">Where Classical Finance Meets Machine Learning</p>', unsafe_allow_html=True)
    
    # Sidebar - User Inputs
    st.sidebar.header("⚙️ Configuration")
    
    # Option parameters
    st.sidebar.subheader("Option Details")
    
    # Major tickers dropdown with free text option
    major_tickers = [
        "NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "SPY", "QQQ",
        "COIN", "INTC", "NFLX", "DIS", "JPM", "V", "WMT", "KO", "MSTR", "PLTR"
    ]
    selected_ticker = st.sidebar.selectbox("Ticker", major_tickers, index=0)
    
    # Allow custom ticker entry if needed
    ticker = selected_ticker
    
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    strike = st.sidebar.number_input("Strike Price ($)", value=140.0, min_value=1.0)
    
    # Expiry date
    days_out = st.sidebar.slider("Days to Expiry", min_value=1, max_value=365, value=93)
    expiry_date = (datetime.now() + timedelta(days=days_out)).strftime("%Y-%m-%d")
    
    # Analysis button
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    force_refresh = st.checkbox("Force Refresh Data", value=False)
    
    # Main content area
    if run_analysis or st.session_state.analysis_run:
        st.session_state.analysis_run = True
        
        with st.spinner(f"🔄 Fetching real-time market data for {ticker}..."):
            # Initialize QuantFlow
            qf = QuantFlow(ticker=ticker, option_type=option_type, 
                          strike=strike, expiry=expiry_date)
            
            # Run analysis
            try:
                qf.fetch_data(force_refresh=force_refresh)
                pricing = qf.get_ensemble_pricing()
                greeks = qf.get_greeks()

                
                ml_results = qf.run_ml_analysis()
                scenario_results = qf.run_scenario_analysis()
                
                st.session_state.results = {
                    'qf': qf,
                    'pricing': pricing,
                    'greeks': greeks,
                    'ml_results': ml_results,
                    'scenario_results': scenario_results
                }
            except Exception as e:
                st.error(f"⚠️ Error running analysis: {str(e)}")
                return
        
        results = st.session_state.results
        qf = results['qf']
        pricing = results['pricing']
        greeks = results['greeks']
        ml_results = results['ml_results']
        scenario_results = results['scenario_results']
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Summary", "💰 Pricing", "📐 Greeks", "🤖 AI Insights", "⚠️ Risk Analysis"
        ])
        
        with tab1:
            display_summary(qf, pricing, greeks, ml_results)
        
        with tab2:
            display_pricing(pricing, ml_results)
        
        with tab3:
            display_greeks(qf, greeks)
        
        with tab4:
            display_ml_insights(ml_results)
        
        with tab5:
            display_risk_analysis(scenario_results)
    else:
        # Welcome screen content
        st.markdown('')
        st.markdown('')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">📊</span>
                <div class="feature-title">Classical Pricing</div>
                <div style="text-align: left; margin-top: 1rem;">
                    • Black-Scholes Model<br>
                    • Binomial Trees<br>
                    • Monte Carlo Sims<br>
                    • Ensemble Fair Value
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">🤖</span>
                <div class="feature-title">AI Intelligence</div>
                <div style="text-align: left; margin-top: 1rem;">
                    • GARCH Volatility<br>
                    • XGBoost Mispricing<br>
                    • HMM Regime Detection<br>
                    • Explainable AI (SHAP)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">🛡️</span>
                <div class="feature-title">Risk Management</div>
                <div style="text-align: left; margin-top: 1rem;">
                    • Full Greeks Analysis<br>
                    • Delta-Neutral Hedging<br>
                    • Scenario Stress Tests<br>
                    • VaR / CVaR Metrics
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('')
        st.info("👈 **Get Started**: Select a company (e.g., NVDA, AAPL) in the sidebar and click **Run Analysis**")


def display_summary(qf, pricing, greeks, ml_results):
    """Display executive summary with onboarding"""
    st.header(f"📊 {qf.ticker} Executive Summary")
    
    # Onboarding hint
    with st.expander("💡 New to QuantFlow? Start Here", expanded=False):
        st.markdown("""
        ### How to Use This Dashboard
        
        1. **Fair Value Hook** 🎯: Check if the option is mathematically mispriced
           - Look at **Mispricing Score** below
           - 80+ = Strong buy signal
           
        2. **Regime Detection** 🌍: Understand the current market environment
           - Different strategies work in different regimes
           - We auto-adjust Greeks for current conditions
           
        3. **Greeks Sensitivity** 📈: See how your position changes
           - Use the Greeks tab to visualize risk
           
        4. **Stress Test** ⚠️: Know your worst-case scenario
           - Check VaR in Risk Analysis tab
           - Never trade more than you can afford to lose
        """)
    
    # Key metrics in columns with tooltips
    col1, col2, col3, col4 = st.columns(4)
    
    # Contract Header
    st.markdown(f"""
    <div style="padding: 1rem; background: rgba(255, 255, 255, 0.05); border-radius: 10px; margin-bottom: 2rem; border: 1px solid rgba(255,255,255,0.1);">
        <h3 style="margin:0; font-size: 1.2rem; color: #aaa;">Contract Details</h3>
        <div style="display: flex; gap: 2rem; align-items: center; margin-top: 0.5rem;">
            <div>
                <span style="font-size: 0.9rem; color: #888;">Ticker</span>
                <div style="font-size: 1.5rem; font-weight: bold; color: white;">{qf.ticker}</div>
            </div>
            <div>
                 <span style="font-size: 0.9rem; color: #888;">Type</span>
                 <div style="font-size: 1.5rem; font-weight: bold; color: {'#00FF88' if qf.option_type=='call' else '#FF5555'}; text-transform: uppercase;">
                    {qf.option_type}
                 </div>
            </div>
             <div>
                <span style="font-size: 0.9rem; color: #888;">Strike</span>
                <div style="font-size: 1.5rem; font-weight: bold; color: white;">${qf.K:.2f}</div>
            </div>
             <div>
                <span style="font-size: 0.9rem; color: #888;">Expiry</span>
                <div style="font-size: 1.5rem; font-weight: bold; color: white;">{qf.expiry}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Underlying Stock Price",
            format_currency(qf.S),
            delta=None,
            help="Current spot price of the stock"
        )
    
    with col2:
        bid = qf.market_data['option'].get('bid', 0)
        ask = qf.market_data['option'].get('ask', 0)
        
        # Use Mid Price if available, otherwise Last
        if bid > 0 and ask > 0:
            current_price = (bid + ask) / 2
            price_label = "Mark Price (Mid)"
        else:
            current_price = pricing['market_price']
            price_label = "Last Price"
            
        spread_text = f"Bid: {format_currency(bid)} | Ask: {format_currency(ask)}"
        
        st.metric(
            price_label,
            format_currency(current_price),
            delta=None,
            help=f"Mid-point of Bid/Ask Spread. {spread_text}"
        )
        st.caption(spread_text)
    
    with col3:
        # Use Forecast Fair Value if available, else Ensemble
        fair_value = ml_results.get('forecast_fair_value', pricing['ensemble_fair_value'])
        divergence = ml_results.get('divergence_pct', pricing['divergence_pct'])
        
        st.metric(
             "Forecast Fair Value",
             format_currency(fair_value),
             delta=f"{divergence:+.1f}%",
             delta_color="normal", # Green if positive (Undervalued), Red if negative (Overvalued)
             help="Fair Value based on AI-Forecasted Volatility, not just current Market IV."
        )
    
    with col4:
        st.metric(
            "Market Regime",
            ml_results['regime']['regime_label'],
            delta=f"{ml_results['regime']['confidence']*100:.1f}% confidence",
            help=TOOLTIPS["regime"]
        )
    
    # Mispricing Score Row
    score_col1, score_col2 = st.columns([1, 1])
    with score_col1:
         st.metric(
            "Mispricing Score",
            f"{ml_results['mispricing_score']:.0f}/100",
            delta=ml_results['mispricing_assessment'],
            help="Based on divergence between Market Price and AI-Forecasted Fair Value."
        )
    
    # Smart Position Sizer
    st.subheader("🎯 Smart Position Sizer")

    # Volatility Reality Check
    if "Low Vol" in ml_results['regime']['regime_label'] and qf.sigma > 0.4:
        st.warning(f"⚠️ **Anomaly Detected**: AI detects 'Low Volatility' regime, but Option Implied Volatility is high ({qf.sigma:.1%}). Market might be pricing in an event not yet in historical features.")
    

    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_size = st.number_input("Portfolio Size ($)", value=50000, step=1000)
        risk_pct = st.slider("Max Risk per Trade (%)", 1, 5, 2)
    
    with col2:
        max_risk = portfolio_size * (risk_pct / 100)
        st.metric("Max Risk Amount", format_currency(max_risk))
        
    # Calculate recommended size
    # Logic: Risk = Amount lost if stop loss hit OR complete loss of premium
    # Conservative approach: Assume 50% loss of premium is the "Risk" for sizing
    # Strict Limit: Total invested amount cannot exceed 10% of portfolio
    
    # Use Fair value if market price is stale (flagged by model_is_valid)
    effective_price = pricing['market_price']
    
    # Check for stale data warning
    is_stale = False
    if hasattr(qf.market_data.get('option'), 'get') and not qf.market_data['option'].get('model_is_valid', True):
        is_stale = True
        # If stale, we rely on Ensemble Fair Value as the "Real" price for sizing
        if pricing['ensemble_fair_value'] > 0:
            effective_price = pricing['ensemble_fair_value']
            st.warning(f"⚠ **STALE DATA DETECTED**: Market price is unrealistic (${pricing['market_price']:.2f}). Using Fair Value (${effective_price:.2f}) for sizing.")
    
    contract_cost = effective_price * 100
    
    if contract_cost > 0:
        # 1. Risk-based sizing (assuming 100% loss worst case for safety)
        # contracts_risk = max_risk / (contract_cost) 
        # But let's assume sophisticated user exits at -50%. So Risk = 0.5 * Cost
        # Therefore: Max_Risk = N * 0.5 * Cost  =>  N = Max_Risk / (0.5 * Cost)
        risk_per_contract = contract_cost * 0.5  # Assumes 50% stop loss
        contracts_by_risk = int(max_risk / risk_per_contract)
        
        # 2. Hard Capital Limit (Max 10% of portfolio in one trade)
        max_capital_allocation = portfolio_size * 0.10
        contracts_by_capital = int(max_capital_allocation / contract_cost)
        
        # Take the stricter limit
        recommended_contracts = max(0, min(contracts_by_risk, contracts_by_capital))
        
        st.success(f"### 💡 Recommendation: {recommended_contracts} contracts")
        
        # Cost breakdown
        total_cost = recommended_contracts * contract_cost
        pct_portfolio = (total_cost / portfolio_size) * 100
        
        st.markdown(f"""
        - **Total Cost**: {format_currency(total_cost)} ({pct_portfolio:.1f}% of portfolio)
        - **Max Risk (at 50% stop)**: {format_currency(total_cost * 0.5)}
        """)
        
        if recommended_contracts == 0:
            st.error("❌ Trade is too expensive for your risk rules. Increase portfolio size or choose cheaper option.")
        elif pct_portfolio > 10:
             st.warning(f"⚠️ High Allocation: This trade uses {pct_portfolio:.1f}% of your capital. Proceed with caution.")
    else:
        st.error("Invalid option pricing. Cannot calculate position size.")
    
    st.divider()
    
    # Assessment banner
    if abs(pricing['divergence_pct']) < 2:
        st.success("✅ " + pricing['assessment'])
    elif pricing['divergence_pct'] > 0:
        st.success("🟢 " + pricing['assessment'])
    else:
        st.error("🔴 " + pricing['assessment'])
    
    # Quick facts
    st.subheader("Quick Facts")
    facts_col1, facts_col2 = st.columns(2)
    
    with facts_col1:
        st.markdown(f"""
        **Market Data**
        - Spot Price: ${qf.S:.2f}
        - Strike: ${qf.K:.2f}
        - Days to Expiry: {int(qf.T * 365)} days
        - Implied Volatility: {qf.sigma*100:.2f}%
        """)
    
    with facts_col2:
        st.markdown(f"""
        **Key Greeks**
        - Delta: {greeks['delta']:.4f}
        - Gamma: {greeks['gamma']:.4f}
        - Theta: ${greeks['theta_per_day']:.4f}/day
        - Vega: ${greeks['vega_percent']:.4f}/1% vol
        """)


def display_pricing(pricing, ml_results=None):
    """Display pricing analysis"""
    # Visual Header
    if os.path.exists(os.path.join("assets", "pricing.png")):
        st.image(os.path.join("assets", "pricing.png"), use_container_width=True)
        
    st.header("💰 Ensemble Pricing Analysis")
    
    # Use Forecast Fair Value if available (Synced with Summary)
    ensemble_price = pricing['ensemble_fair_value']
    if ml_results and 'forecast_fair_value' in ml_results:
        ensemble_price = ml_results['forecast_fair_value']
    
    # Model comparison
    st.subheader("Model Comparison")
    
    models_data = {
        'Model': ['Black-Scholes', 'Binomial (European)', 'Monte Carlo', 'Ensemble'],
        'Price': [
            pricing['black_scholes'],
            pricing['binomial_european'],
            pricing['monte_carlo'],
            ensemble_price
        ]
    }
    
    df_models = pd.DataFrame(models_data)
    df_models['Difference from Market (%)'] = (
        (df_models['Price'] - pricing['market_price']) / pricing['market_price'] * 100
    )
    
    st.dataframe(df_models.style.format({
        'Price': '${:.2f}',
        'Difference from Market (%)': '{:+.2f}%'
    }), use_container_width=True)
    
    # Price comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_models['Model'],
        y=df_models['Price'],
        text=df_models['Price'].apply(lambda x: f'${x:.2f}'),
        textposition='outside',
        marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe']
    ))
    
    fig.add_hline(
        y=pricing['market_price'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Market: ${pricing['market_price']:.2f}"
    )
    
    fig.update_layout(
        title="Pricing Model Comparison",
        xaxis_title="Model",
        yaxis_title="Price ($)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_greeks(qf, greeks):
    """Display Greeks analysis with enhanced interactivity"""
    st.header("📐 Greeks Analysis")
    
    # Greeks table with better formatting
    greeks_data = {
        'Greek': ['Delta', 'Gamma', 'Theta (annual)', 'Theta (per day)', 'Vega', 'Rho'],
        'Value': [
            greeks['delta'],
            greeks['gamma'],
            greeks['theta'],
            greeks['theta_per_day'],
            greeks['vega_percent'],
            greeks['rho_percent']
        ],
        'Interpretation': [
            f"Option moves ${abs(greeks['delta']*qf.S):.2f} for every $1 stock move",
            f"Delta changes by {greeks['gamma']:.4f} per $1 stock move",
            f"Annual time decay: ${greeks['theta']:.2f}",
            f"Option loses ${abs(greeks['theta_per_day']):.2f} per day",
            f"Option gains ${greeks['vega_percent']:.2f} per 1% IV increase",
            f"Option gains ${greeks['rho_percent']:.2f} per 1% rate increase"
        ]
    }
    
    df_greeks = pd.DataFrame(greeks_data)
    
    # Add tooltips column
    st.dataframe(
        df_greeks.style.set_properties(**{'text-align': 'left'}),
        use_container_width=True,
        hide_index=True
    )
    
    # Greeks vs Spot visualization - ENHANCED
    st.subheader("📊 Interactive Greeks Sensitivity")
    
    # Chart type selector
    chart_type = st.radio(
        "Select View:",
        ["Delta & Gamma", "All Greeks"],
        horizontal=True,
        help="Choose which Greeks to visualize"
    )
    
    calc = GreeksCalculator(qf.S, qf.K, qf.T, qf.r, qf.sigma, qf.option_type, qf.q)
    greeks_vs_spot = calc.greeks_vs_spot(spot_range=(0.8, 1.2), n_points=100)
    
    if chart_type == "Delta & Gamma":
        # Create interactive figure with advanced features
        fig = go.Figure()
        
        # Delta trace with custom hover
        fig.add_trace(go.Scatter(
            x=greeks_vs_spot['spot_price'],
            y=greeks_vs_spot['delta'],
            name='Delta',
            line=dict(color='#00FF88', width=3),
            mode='lines',
            hovertemplate='<b>Delta</b><br>' +
                         'Stock: $%{x:.2f}<br>' +
                         'Delta: %{y:.4f}<br>' +
                         '<i>Option moves $%{customdata:.2f} per $1 stock move</i><br>' +
                         '<extra></extra>',
            customdata=greeks_vs_spot['delta'] * greeks_vs_spot['spot_price']
        ))
        
        # Gamma trace (scaled) with custom hover
        fig.add_trace(go.Scatter(
            x=greeks_vs_spot['spot_price'],
            y=greeks_vs_spot['gamma'] * 10,
            name='Gamma (×10)',
            line=dict(color='#FFB020', width=3, dash='dash'),
            mode='lines',
            yaxis='y2',
            hovertemplate='<b>Gamma</b><br>' +
                         'Stock: $%{x:.2f}<br>' +
                         'Gamma: %{customdata:.4f}<br>' +
                         '<i>Delta changes by this amount per $1 stock move</i><br>' +
                         '<extra></extra>',
            customdata=greeks_vs_spot['gamma']
        ))
        
        # Current spot price marker
        fig.add_vline(
            x=qf.S,
            line_dash="dot",
            line_color="#FF004D",
            line_width=2,
            annotation_text=f"Current: ${qf.S:.2f}",
            annotation_position="top"
        )
        
        # Strike price marker
        fig.add_vline(
            x=qf.K,
            line_dash="dot",
            line_color="#00C8FF",
            line_width=2,
            annotation_text=f"Strike: ${qf.K:.2f}",
            annotation_position="bottom"
        )
        
        # Update layout with advanced features
        fig.update_layout(
            title={
                'text': "Delta & Gamma vs Spot Price",
                'font': {'size': 20, 'color': '#00FF88'}
            },
            xaxis_title="Stock Price ($)",
            yaxis_title="Delta",
            yaxis2=dict(
                title="Gamma (×10)",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            height=500,
            hovermode='x unified',
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Add range slider
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.05),
                type="linear"
            )
        )
        
        # Add download button
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'greeks_delta_gamma',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }
        
        st.plotly_chart(fig, use_container_width=True, config=config)
        
        # Interactive insights
        st.info(f"""
        💡 **Interactive Features**:
        - **Hover** over lines to see exact values
        - **Click and drag** to zoom into specific regions
        - **Double-click** to reset zoom
        - **Download** chart using camera icon (top right)
        - **Range slider** below chart for quick navigation
        
        **Current Analysis**:
        - Delta = {greeks['delta']:.4f} (option moves ${abs(greeks['delta']*qf.S):.2f} per $1 stock move)
        - Gamma = {greeks['gamma']:.4f} (delta accelerates {"quickly" if greeks['gamma'] > 0.02 else "moderately"})
        """)
    
    elif chart_type == "All Greeks":
        # Multi-subplot interactive chart
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta', 'Gamma', 'Theta (per day)', 'Vega'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Delta
        fig.add_trace(
            go.Scatter(
                x=greeks_vs_spot['spot_price'],
                y=greeks_vs_spot['delta'],
                name='Delta',
                line=dict(color='#00FF88', width=2),
                hovertemplate='$%{x:.2f}: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Gamma
        fig.add_trace(
            go.Scatter(
                x=greeks_vs_spot['spot_price'],
                y=greeks_vs_spot['gamma'],
                name='Gamma',
                line=dict(color='#FFB020', width=2),
                hovertemplate='$%{x:.2f}: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Theta
        fig.add_trace(
            go.Scatter(
                x=greeks_vs_spot['spot_price'],
                y=greeks_vs_spot['theta_per_day'],
                name='Theta',
                line=dict(color='#FF6B6B', width=2),
                hovertemplate='$%{x:.2f}: $%{y:.2f}/day<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Vega
        fig.add_trace(
            go.Scatter(
                x=greeks_vs_spot['spot_price'],
                y=greeks_vs_spot['vega_percent'],
                name='Vega',
                line=dict(color='#00C8FF', width=2),
                hovertemplate='$%{x:.2f}: $%{y:.2f}/1%vol<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Add current spot markers to all
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_vline(
                    x=qf.S,
                    line_dash="dot",
                    line_color="#FF004D",
                    row=row, col=col
                )
        
        fig.update_layout(
            height=700,
            template='plotly_dark',
            showlegend=False,
            hovermode='x unified'
        )
        
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'greeks_all',
                'height': 900,
                'width': 1400,
                'scale': 2
            }
        }
        
        st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Hedging recommendation
    st.subheader("🛡️ Delta-Neutral Hedging Strategy")
    
    hedge = calc.delta_neutral_hedge()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Recommended Hedge**:
        {hedge['recommendation']}
        
        **Hedge Details**:
        - Shares to hedge: {abs(hedge['hedge_position']):.0f}
        - Direction: {"SHORT" if hedge['hedge_position'] < 0 else "LONG"}
        - Notional: ${abs(hedge['hedge_notional']):,.0f}
        """)
    
    with col2:
        st.markdown(f"""
        **Rebalancing**:
        - Trigger: Spot moves >{hedge['rehedge_threshold_pct']:.1f}%
        - Threshold price: ${hedge['rehedge_threshold_price']:.2f}
        - Transaction cost: ${hedge['transaction_cost']:.2f} per rehedge
        """)
    
def display_ml_insights(ml_results):
    """Display ML insights"""
    # Visual Header
    if os.path.exists(os.path.join("assets", "brain.png")):
        st.image(os.path.join("assets", "brain.png"), use_container_width=True)
        
    st.header("🤖 AI-Powered Insights")
    
    # Regime detection
    st.subheader("Market Regime Detection")
    regime = ml_results['regime']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Current Regime:** {regime['regime_label']}")
        st.markdown(f"**Confidence:** {regime['confidence']*100:.1f}%")
        
        # Regime probabilities
        probs_df = pd.DataFrame({
            'Regime': list(regime['state_probabilities'].keys()),
            'Probability': list(regime['state_probabilities'].values())
        })
        
        fig = px.bar(probs_df, x='Regime', y='Probability', 
                    title='Regime Probabilities',
                    color='Probability',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Adjusted Greeks:**")
        adjusted = ml_results['adjusted_greeks']
        st.markdown(f"- Delta: {adjusted['delta']:.4f}")
        st.markdown(f"- Range: [{adjusted['delta_lower']:.4f}, {adjusted['delta_upper']:.4f}]")
        st.markdown(f"- Gamma: {adjusted['gamma']:.4f}")
        
        st.info(adjusted['recommendation'])
    
    # Volatility forecast
    st.subheader("Volatility Forecast")
    vol_forecast = ml_results['volatility_forecast']
    
    vol_data = {
        'Model': ['Historical (20d)', 'GARCH', 'Ensemble'],
        'Forecast (%)': [
            vol_forecast.get('historical_20d', 0) * 100,
            vol_forecast.get('garch', 0) * 100,
            vol_forecast.get('ensemble', 0) * 100
        ]
    }
    
    df_vol = pd.DataFrame(vol_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_vol['Model'],
        y=df_vol['Forecast (%)'],
        text=df_vol['Forecast (%)'].apply(lambda x: f'{x:.2f}%'),
        textposition='outside',
        marker_color=['#4facfe', '#00f2fe', '#43e97b']
    ))
    
    fig.update_layout(
        title="Volatility Forecasts",
        xaxis_title="Model",
        yaxis_title="Volatility (%)",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_risk_analysis(scenario_results):
    """Display risk analysis with enhanced interactivity"""
    # Visual Header
    if os.path.exists(os.path.join("assets", "shield.png")):
        st.image(os.path.join("assets", "shield.png"), use_container_width=True)
        
    st.header("⚠️ Risk Analysis & Scenarios")
    
    # Initialize lock state
    if 'risk_unlocked' not in st.session_state:
        st.session_state.risk_unlocked = False
        
    # Lock Mechanism
    if not st.session_state.risk_unlocked:
        st.markdown("""
        <div style="
            background: rgba(255, 60, 60, 0.1); 
            border: 1px solid rgba(255, 60, 60, 0.3);
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
        ">
            <h3 style="color: #FF6B6B; margin-top: 0;">🔒 CLASSIFIED RISK PROTOCOLS</h3>
            <p style="color: #ccc;">Advanced risk mitigation strategies are encrypted for unauthorized personnel.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔓 AUTHENTICATE & DECRYPT", type="primary", use_container_width=True):
                st.session_state.risk_unlocked = True
                st.rerun()
        return  # Stop rendering if locked
        
    # Unlocked Content
    if st.button("🔒 Re-Lock Protocols", type="secondary"):
        st.session_state.risk_unlocked = False
        st.rerun()
        
    st.success("✅ **ACCESS GRANTED**: displaying classified risk scenarios.")
    
    # Scenario comparison
    st.subheader("📊 Scenario Analysis")
    scenarios = scenario_results['scenarios']
    
    # Interactive scenario selector
    selected_scenario = st.selectbox(
        "Click to view scenario details:",
        scenarios['scenario_name'].tolist(),
        help="Select a scenario to see detailed breakdown"
    )
    
    # Create interactive bar chart
    fig = go.Figure()
    
    colors = ['#22c55e', '#84cc16', '#9ca3af', '#fb923c', '#ef4444', '#7f1d1d']
    
    fig.add_trace(go.Bar(
        y=scenarios['scenario_name'],
        x=scenarios['total_pnl'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=2)
        ),
        text=scenarios['total_pnl'].apply(lambda x: f'${x:.0f}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' +
                     'P&L: $%{x:,.0f}<br>' +
                     '<i>Click bar for details</i><br>' +
                     '<extra></extra>'
    ))
    
    fig.add_vline(x=0, line_color='white', line_width=2)
    
    fig.update_layout(
        title={
            'text': "30-Day Scenario P&L (Click bars for details)",
            'font': {'size': 18, 'color': '#00FF88'}
        },
        xaxis_title="P&L ($)",
        yaxis_title="",
        height=400,
        template='plotly_dark',
        hovermode='closest'
    )
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'scenario_analysis',
            'height': 600,
            'width': 1000,
            'scale': 2
        }
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    # Show selected scenario details
    selected_data = scenarios[scenarios['scenario_name'] == selected_scenario].iloc[0]
    
    with st.container():
        st.markdown(f"""
        <div class="glass-card">
            <h3>📋 {selected_scenario} Scenario Details</h3>
            <p><strong>Market Conditions</strong>:<br>
            • Stock Price Change: {selected_data.get('stock_change_pct', 0):.1f}%<br>
            • Volatility Change: {selected_data.get('vol_change_pct', 0):.1f}%</p>
            <p><strong>Position Impact</strong>:<br>
            • Total P&L: <strong>${selected_data['total_pnl']:,.0f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
    
    # Monte Carlo metrics - ENHANCED
    st.subheader("🎲 Monte Carlo Risk Metrics (10,000 Simulations)")
    mc_dist = scenario_results['monte_carlo_distribution']
    

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Expected P&L", format_currency(mc_dist['mean_pnl']))
        st.metric("VaR (95%)", format_currency(mc_dist['var_95']), help=TOOLTIPS["var"])
    
    with col2:
        st.metric("VaR (99%)", format_currency(mc_dist['var_99']), help="Worst loss in 99% of scenarios")
        st.metric("CVaR (95%)", format_currency(mc_dist['cvar_95']), help=TOOLTIPS["cvar"])
    
    with col3:
        st.metric("Probability of Profit", f"{mc_dist['prob_profit']*100:.1f}%")
        st.metric("Std Deviation", format_currency(mc_dist['std_pnl']))
    
    # Enhanced P&L distribution with click interaction
    st.subheader("📈 Interactive P&L Distribution")
    
    pnl_data = mc_dist['all_pnl']
    
    # Create histogram with KDE overlay
    fig_dist = go.Figure()
    
    # Histogram
    fig_dist.add_trace(go.Histogram(
        x=pnl_data,
        nbinsx=60,
        name='P&L Distribution',
        marker=dict(
            color='#3b82f6',
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        opacity=0.8,
        hovertemplate='P&L Range: $%{x:,.0f}<br>' +
                     'Frequency: %{y}<br>' +
                     '<extra></extra>'
    ))
    
    # Add VaR lines with annotations
    fig_dist.add_vline(
        x=mc_dist['var_95'],
        line_dash='dash',
        line_color='#FFB020',
        line_width=3,
        annotation_text=f"VaR 95%: ${mc_dist['var_95']:,.0f}",
        annotation_position="top left"
    )
    
    fig_dist.add_vline(
        x=mc_dist['var_99'],
        line_dash='dash',
        line_color='#FF004D',
        line_width=3,
        annotation_text=f"VaR 99%: ${mc_dist['var_99']:,.0f}",
        annotation_position="top left"
    )
    
    fig_dist.add_vline(x=0, line_color='white', line_width=2, annotation_text="Break-even")
    
    fig_dist.update_layout(
        title={'text': "P&L Distribution", 'font': {'size': 18, 'color': '#00FF88'}},
        xaxis_title="P&L ($)",
        yaxis_title="Frequency",
        height=500,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Tail risk explorer
    with st.expander("🔍 Explore Tail Events (Click to expand)", expanded=False):
        selected_percentile = st.slider("Explore Percentile", 1, 99, 5)
        percentile_value = np.percentile(pnl_data, selected_percentile)
        st.info(f"**{selected_percentile}th Percentile**: ${percentile_value:,.0f}")



if __name__ == "__main__":
    main()
