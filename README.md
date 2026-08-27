# QuantFlow - Institutional HFT & Biomimetic Swarm Intelligence Platform

[![CI](https://github.com/planksconstant-arch/Quantflow/actions/workflows/ci.yml/badge.svg)](https://github.com/planksconstant-arch/Quantflow/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/planksconstant-arch/Quantflow/branch/main/graph/badge.svg)](https://codecov.io/gh/planksconstant-arch/Quantflow)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![HFT Latency](https://img.shields.io/badge/order_match-2.9us-brightgreen)

## Executive Summary

**QuantFlow** is an institutional-grade quantitative high-frequency trading (HFT), market microstructure intelligence, and options analytics platform. It introduces **Biomimetic Mormyrid Swarm Consensus Intelligence**—a multi-agent framework inspired by weakly electric fish (Mormyridae) active electrolocation, Jamming Avoidance Response (JAR), and decentralized Byzantine-resilient consensus—integrated with real-time Level 2/3 Limit Order Book (LOB) matching and Swarm-Skewed Avellaneda-Stoikov market making.

---

## High-Frequency Quantitative Architecture

```mermaid
graph TB
    subgraph Market_Microstructure_Layer ["Layer 1: High-Frequency Market Microstructure & LOB"]
        LOB[Level 2/3 Order Book Engine] --> DepthFeed[LOB Depth & Queue Array]
        TickGen[Tick & Hawkes Jump Generator] --> LOB
        DepthFeed --> Hawkes[Bivariate Hawkes Self/Cross Excitation]
        DepthFeed --> OFI[Multi-Level OFI Engine]
        DepthFeed --> VPIN[Volume-Synchronized Toxicity VPIN]
        DepthFeed --> MicroPrice[Stoikov Micro-Price Estimator]
    end

    subgraph Swarm_Consensus_Layer ["Layer 2: Biomimetic Mormyrid Swarm Intelligence"]
        Hawkes & OFI & VPIN & MicroPrice --> SensoryVector[LOB Sensory Field Matrix]
        SensoryVector --> ScoutFish[Scout Fish: LOB Depth & Spread Anomalies]
        SensoryVector --> PredatorFish[Predator Fish: Hawkes Cascades & Momentum]
        SensoryVector --> SchoolingFish[Schooling Fish: Queue Imbalance & Reversion]
        SensoryVector --> SentinelFish[Sentinel Fish: Toxicity & Adverse Selection]
        
        ScoutFish & PredatorFish & SchoolingFish & SentinelFish --> JAR[Jamming Avoidance Response Engine]
        JAR --> SwarmConsensus[Decentralized Swarm Consensus Engine]
        SwarmConsensus --> ConsensusOutput[Consensus Signal: Drift, Jump Prob, Skew, Crowding]
    end

    subgraph Execution_Strategy_Layer ["Layer 3: HFT Execution & Market Making"]
        ConsensusOutput --> AvellanedaStoikov[Swarm-Skewed Avellaneda-Stoikov MM]
        ConsensusOutput --> OptimalExec[Almgren-Chriss Liquidation & TWAP/VWAP]
        AvellanedaStoikov --> HFTSim[Event-Driven Execution & Latency Simulator]
        OptimalExec --> HFTSim
    end

    subgraph Institutional_UI_Layer ["Layer 4: Institutional Trading Terminal"]
        HFTSim --> UI_LOB[Live LOB Depth Ladder & Walls]
        SwarmConsensus --> UI_Swarm[2D/3D Mormyrid Swarm Vector Visualizer]
        AvellanedaStoikov --> UI_MM[MM Quote Ladder & Inventory Risk]
        HFTSim --> UI_Telemetry[Institutional Risk & Telemetry Engine]
    end
```

---

## Biomimetic Mormyrid Swarm Intelligence

Mormyrid weakly electric fish navigate murky waters by emitting **Electric Organ Discharge (EOD)** pulses, perceiving environmental distortions, and coordinating through **Jamming Avoidance Responses (JAR)**:

| Specialized Agent Class | Biological Role | HFT Sensory Dimension | Primary Signal Function |
| :--- | :--- | :--- | :--- |
| **`ScoutFishAgent`** | Exploratory Active Electrolocation | Multi-level depth & spread dispersion | Detects hidden liquidity & spread anomalies |
| **`PredatorFishAgent`** | Aggressive Chasing & Burst Detection | Bivariate Hawkes jump cascades ($n \ge 1$) | Exploits momentum bursts & aggressive order flow |
| **`SchoolingFishAgent`** | Cohesive Swarming & Alignment | Queue imbalance & micro-price deviation | Captures mean-reversion & spread replenishment |
| **`SentinelFishAgent`** | Hazard & Jamming Perception | VPIN toxicity & crowding index | Mitigates adverse selection & inventory loss |

### Mathematical Specifications:
1. **Active Electrolocation Field Return**:
   $$f_i(X_t) = \frac{A_i}{1.0 + \sum_{k} w_{i,k} (x_{i,k} - s_k)^2}$$
2. **Jamming Avoidance Response (JAR)**:
   $$\Delta f_i = \text{sign}(f_i - \bar{f}_j) \cdot \exp\left(-\frac{\|X_i - X_j\|^2}{2\sigma_{\text{JAR}}^2}\right)$$
3. **Swarm-Skewed Avellaneda-Stoikov Reservation Price**:
   $$r_{\text{swarm}}(s, q, t) = s - q \gamma \sigma^2 (T - t) + \kappa_{\text{swarm}} \cdot \hat{\mu}_{\text{consensus}}$$

---

## Quick Start

### 1. Launch Institutional Trading Terminal
```bash
# Run interactive Streamlit terminal
streamlit run app.py
```

### 2. Run Command-Line HFT Simulation
```bash
# Run HFT simulation and swarm pipeline
python main.py
```

### 3. Run Microsecond Latency Benchmark
```bash
python tests/benchmark_latency.py
```

### 4. Run Unit Test Suite
```bash
pytest tests/ -v
```

---

## Latency & Execution Benchmarks

Benchmarked on standard x86-64 hardware:

| Component | Operation | Mean Latency | Throughput |
| :--- | :--- | :--- | :--- |
| **Limit Order Book** | Order Insertion (`add_limit_order`) | **1.21 us** | ~825,000 ops/sec |
| **Limit Order Book** | Market Order Matching (`execute_market_order`) | **2.95 us** | ~339,000 ops/sec |
| **Microstructure Pipeline** | Multi-Level OFI + VPIN + Stoikov | **21.81 us** | ~45,800 ops/sec |
| **Avellaneda-Stoikov MM** | Optimal Bid/Ask Quote Generation | **15.09 us** | ~66,200 ops/sec |
| **Mormyrid Swarm Cycle** | 28 Agents EOD + JAR + Consensus | **714.66 us** | ~1,400 cycles/sec |

---

## Project Structure

```
quantflow/
├── app.py                     # Institutional Streamlit Trading Terminal (5 Tabs)
├── main.py                    # Main Entry Point & Simulation Driver
├── models/
│   ├── swarm/                 # Biomimetic Mormyrid Swarm Intelligence
│   │   ├── mormyrid_agent.py  # Scout, Predator, Schooler, Sentinel Fish Agents
│   │   ├── jamming_avoidance.py # Biological Jamming Avoidance Response (JAR)
│   │   ├── consensus_engine.py  # Byzantine Swarm Consensus & Kalman Smoothing
│   │   └── afsa.py            # Artificial Fish Swarm Algorithm (AFSA) Optimizer
│   ├── microstructure/        # High-Frequency Market Microstructure
│   │   ├── order_book.py      # L2/L3 Limit Order Book & Synthetic Tick Engine
│   │   ├── signals.py         # Multi-Level OFI, VPIN, Stoikov Micro-Price
│   │   └── hawkes.py          # Univariate & Bivariate Hawkes Processes
│   ├── hft_execution/         # High-Frequency Trading Execution
│   │   ├── avellaneda_stoikov.py # Swarm-Skewed Avellaneda-Stoikov MM
│   │   ├── optimal_execution.py  # Almgren-Chriss Liquidation & TWAP/VWAP
│   │   └── hft_simulator.py      # Event-Driven HFT Backtester & Ledger
│   ├── risk/
│   │   └── hft_risk.py        # Real-time VaR, CVaR, Sharpe, Drawdown, Limits
│   ├── pricing/               # Classical Black-Scholes, Binomial, Monte Carlo
│   ├── greeks/                # Greeks Sensitivity Calculators
│   └── acceleration.py        # Numba JIT & Vectorized Routines
├── tests/
│   ├── test_swarm.py          # Swarm Intelligence Unit Tests
│   ├── test_microstructure.py # LOB & Microstructure Signal Tests
│   ├── test_hft_execution.py  # HFT Execution & Strategy Tests
│   ├── test_models.py         # Classical Options Models Tests
│   └── benchmark_latency.py   # Microsecond Latency Benchmark Harness
└── requirements.txt
```

---

## License
MIT License. Open-source for quantitative research and algorithmic trading.
