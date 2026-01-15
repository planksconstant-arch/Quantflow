# Microstructure Analysis in QuantFlow

## Overview

Market microstructure studies the mechanics of price formation at the **order flow** level. QuantFlow implements:

1. **Hawkes Processes**: Self-exciting models for order arrivals
2. **LOB Analytics**: Order book imbalance and liquidity metrics
3. **Regime Detection**: "Bull Rise" identification

## Hawkes Processes

### Mathematical Model

Hawkes processes model event arrival rates that **depend on past events** (self-excitation):

$$\lambda(t) = \mu + \sum_{t_i < t} \phi(t - t_i)$$

Where:
- $\mu$: Baseline intensity (exogenous)
- $\phi(\cdot)$: Excitation kernel (typically $\alpha \beta e^{-\beta \tau}$)
- $t_i$: Past event times

### Branching Ratio

The **branching ratio** $n = \int_0^\infty \phi(\tau) d\tau$ determines stationarity:

- $n < 1$: **Stationary** (stable)
- $n \to 1$: **Critical** (trending/momentum)
- $n \geq 1$: **Explosive** (bubble/crash)

### Bivariate Hawkes (Buy/Sell)

Model buy and sell orders jointly:

$$\lambda_{\text{buy}}(t) = \mu_+ + \sum_{buy} \phi_{++}(\cdot) + \sum_{sell} \phi_{-+}(\cdot)$$
$$\lambda_{\text{sell}}(t) = \mu_- + \sum_{buy} \phi_{+-}(\cdot) + \sum_{sell} \phi_{--}(\cdot)$$

Cross-excitation captures **feedback** (e.g., sells triggering more sells).

## "Bull Rise" Detection

### Definition

A "Bull Rise" regime is characterized by:

1. **High buy self-excitation**: $n_{++} > 0.7$ (critical branching)
2. **Low sell cascades**: $n_{--} < 0.5$
3. **Asymmetric intensity**: $\mu_{buy} > \mu_{sell}$

### Detection Algorithm

```python
from models.microstructure.hawkes import detect_bull_rise_regime

# Historical order times
buy_times = [...]  # Timestamps of buy orders
sell_times = [...]  # Timestamps of sell orders
T = 10.0  # Observation window (hours)

is_bull, diagnostics = detect_bull_rise_regime(buy_times, sell_times, T)

if is_bull:
    print(f"Bull Rise detected!")
    print(f"  Buy branching: {diagnostics['n_buy']:.2f}")
    print(f"  Sell branching: {diagnostics['n_sell']:.2f}")
```

## Order Book Analytics

### Order Book Imbalance (OBI)

$$\text{OBI} = \frac{V_{\text{bid}} - V_{\text{ask}}}{V_{\text{bid}} + V_{\text{ask}}}$$

- OBI > 0: **Buy pressure**
- OBI < 0: **Sell pressure**

### Kyle's Lambda (Price Impact)

Linear regression of price change on signed volume:

$$\Delta P_t = \lambda \cdot Q_t + \epsilon_t$$

Where:
- $\lambda$: Kyle's Lambda ($/share)
- $Q_t$: Signed order flow (+ buy, - sell)

High $\lambda$ → **Low liquidity** (large impact)

## Usage

### Fit Hawkes Process

```python
from models.microstructure.hawkes import HawkesProcess

# Observed event times (e.g., trade timestamps)
events = np.array([0.1, 0.5, 0.7, 1.2, 1.5, ...])
T = 10.0

# Fit via MLE
fitted_model = HawkesProcess.fit(events, T)

print(f"Baseline intensity (μ): {fitted_model.mu:.3f}")
print(f"Excitation strength (α): {fitted_model.alpha:.3f}")
print(f"Branching ratio: {fitted_model.branching_ratio():.3f}")
```

### LOB Analysis

```python
from data.lob_analyzer import LOBAnalyzer

# Snapshot data
bid_volume = 5000
ask_volume = 3000

obi = LOBAnalyzer.order_book_imbalance(bid_volume, ask_volume)
print(f"Order Book Imbalance: {obi:.2f}")  # +0.25 (buy pressure)

# Kyle's Lambda
price_changes = [0.01, -0.02, 0.015, ...]
signed_volumes = [100, -150, 80, ...]

lambda_kyle = LOBAnalyzer.kyles_lambda(price_changes, signed_volumes)
print(f"Kyle's Lambda: ${lambda_kyle:.4f} per share")
```

## Applications

1. **Momentum Detection**: Identify trending markets via branching ratio
2. **Liquidity Assessment**: Use Kyle's Lambda for execution cost estimation
3. **Regime Forecasting**: OBI predicts short-term price direction

## Files

- `models/microstructure/hawkes.py`: Hawkes process implementation
- `data/lob_analyzer.py`: Order book metrics

## References

- Hawkes (1971): "Spectra of some self-exciting and mutually exciting point processes"
- Kyle (1985): "Continuous Auctions and Insider Trading"
- Cont et al. (2010): "The Price Impact of Order Book Events"
