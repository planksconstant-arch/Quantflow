# QuantFlow - Run Instructions

## Phase 1-3: Core Analysis

```bash
# Run main demo (all 3 phases)
python main.py

# Generate all deliverables (charts, slide deck, memo)
python generate_deliverables.py
```

## Phase 4: Advanced Features

### 1. Interactive Dashboard

```bash
# Install streamlit if not already installed
pip install streamlit kaleido

# Run dashboard
streamlit run app.py
```

This opens an interactive web dashboard at `http://localhost:8501` with:
- Real-time option analysis
- Interactive Greeks charts
- Scenario simulator
- AI insights visualization
- Downloadable reports

### 2. Portfolio Analysis

```python
from analysis import PortfolioAnalyzer, OptionPosition

# Create portfolio
analyzer = PortfolioAnalyzer(spot_price=145, risk_free_rate=0.045)

# Add long call
analyzer.add_position(
    OptionPosition(ticker='NVDA', option_type='call', strike=140,
                  expiry='2026-04-17', quantity=10, entry_price=10.50),
    implied_vol=0.35,
    time_to_maturity=93/365
)

# Add short put (hedge)
analyzer.add_position(
    OptionPosition(ticker='NVDA', option_type='put', strike=150,
                  expiry='2026-04-17', quantity=-5, entry_price=12.00),
    implied_vol=0.38,
    time_to_maturity=93/365
)

# Calculate aggregated Greeks
greeks = analyzer.calculate_portfolio_greeks()

# Get hedge recommendation
hedge = analyzer.calculate_portfolio_hedge(greeks)

# Scenario analysis
scenarios = analyzer.portfolio_scenario_analysis([-0.10, -0.05, 0, 0.05, 0.10])
```

### 3. Backtesting

```python
from analysis import OptionsBacktester
import pandas as pd

# Initialize backtester
backtester = OptionsBacktester(initial_capital=10000, commission_per_contract=0.65)

# Prepare historical data (your option chain data)
historical_data = pd.DataFrame({
    'date': [...],
    'spot_price': [...],
    'strike': [...],
    'option_type': [...],
    'market_price': [...],
    'fair_value': [...],
    'implied_vol': [...],
    'risk_free_rate': [...]
})

# Run backtest
results = backtester.backtest_mispricing_strategy(
    historical_data,
    mispricing_threshold=5.0,  # Buy if >5% undervalued
    hold_days=30
)

# View metrics
print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Win Rate: {results['win_rate']*100:.1f}%")
print(f"Max Drawdown: ${results['max_drawdown']:.0f}")
```

## Docker

```bash
# Build
docker build -t quantflow:latest .

# Run analysis
docker run quantflow:latest python main.py

# Run dashboard (with port mapping)
docker run -p 8501:8501 quantflow:latest streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

## Output Files

All outputs saved to `outputs/charts/`:
- **Charts**: 6 visualizations (Greeks, P&L, scenarios, volatility)
- **Documents**: slide_deck_content.md, executive_memo.txt, summary_data.json

## Quick Examples

### Single Option Analysis
```bash
python main.py
```

### Custom Option
```python
from main import QuantFlow

qf = QuantFlow(ticker="AAPL", option_type="put", strike=180, expiry="2024-06-21")
results = qf.run_phase2_demo()
```

### Generate Charts Only
```python
from visualization import GreeksVisualizer
from models import GreeksCalculator

viz = GreeksVisualizer()
calc = GreeksCalculator(S=145, K=140, T=0.25, r=0.045, sigma=0.35, option_type='call')

# Generate charts
greeks_vs_spot = calc.greeks_vs_spot()
viz.plot_greeks_vs_spot(greeks_vs_spot, current_spot=145)
```

## Troubleshooting

**Issue**: Missing kaleido for 3D charts
```bash
pip install --upgrade kaleido
```

**Issue**: Streamlit not found
```bash
pip install streamlit
```

**Issue**: Port 8501 already in use
```bash
streamlit run app.py --server.port=8502
```

---

**QuantFlow**: Production-ready options intelligence system 🚀
