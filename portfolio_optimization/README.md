# Systematic Portfolio Trading via Convex Optimization

A complete portfolio optimization system using convex programming with Python.

## Project Highlights

- **Formulated a multi-period convex program** for portfolio allocation under risk, turnover, and transaction cost constraints
- **Benchmarked on synthetic data (5+ years)** against naive weighting (results vary by seed and parameters)
- **Simulated regime-aware rebalancing** to test stability across changing volatility and liquidity conditions

> **Important:** This project uses **synthetic data** generated in `data.py`. Results are illustrative and will vary by random seed, parameters, and rebalancing choices. Replace the synthetic generator with real data for meaningful performance evaluation.

## Project Structure

```
portfolio_optimization/
├── __init__.py          # Package initialization
├── main.py              # Main entry point with CLI
├── data.py              # Data generation and statistics
├── optimizer.py         # Convex portfolio optimizer (CVXPY)
├── regimes.py           # Market regime detection
├── backtest.py          # Backtesting engine
├── stability.py         # Regime stability tests
├── viz.py               # Visualization and reporting
└── tests/
    └── test_modules.py  # Unit tests
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Demo

```bash
cd portfolio_optimization
python main.py
```

### 3. Quick Test (1 year of data)

```bash
python main.py --quick
```

### 4. Run Tests

```bash
python tests/test_modules.py
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--quick` | Fast mode with 1 year of data | Full 5 years |
| `--assets N` | Number of assets | 10 |
| `--days N` | Trading days to simulate | 1260 (5 years) |
| `--rebalance N` | Rebalancing frequency (days) | 21 (monthly) |
| `--risk-aversion X` | Risk aversion parameter | 1.0 |
| `--no-plot` | Skip visualization | Show plots |
| `--no-stability` | Skip stability tests | Run tests |

## Key Metrics Explained

| Metric | What It Measures | Good Value |
|--------|------------------|------------|
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Sortino Ratio** | Downside risk-adjusted return | > 1.5 |
| **Max Drawdown** | Worst peak-to-trough decline | < 20% |
| **Turnover** | Trading activity | Lower is better |

## How It Works

### 1. Convex Optimization

We solve:
```
maximize:   Expected Return - γ × Risk - Transaction Costs
subject to: 
    - Weights sum to 1 (fully invested)
    - No short selling
    - Max 30% in any single asset
    - Turnover ≤ 50% per rebalance
    - Portfolio volatility ≤ 25%
```

### 2. Regime Detection

The system detects 4 market regimes:
- **LOW_VOL_BULL**: Calm uptrend → Aggressive
- **HIGH_VOL_BULL**: Volatile uptrend → Moderate
- **LOW_VOL_BEAR**: Calm downtrend → Defensive
- **HIGH_VOL_BEAR**: Crisis → Maximum defense

### 3. Multi-Period Optimization

Instead of being myopic, we look ahead 3 periods to:
- Reduce transaction costs
- Trade smoothly into positions
- Anticipate changing conditions

## Sample Results (Illustrative)

After running the full demo, you'll see output similar to:

```
PERFORMANCE COMPARISON
──────────────────────────────────────────────
Metric                    Optimized    Benchmark    Difference
───────────────────────────────────────────────────────────────
Annualized Return (%)        12.50        10.20        +2.30
Sharpe Ratio                  0.85         0.62        +0.23
Max Drawdown (%)            -18.50       -22.30        +3.80
Average Turnover (%)         15.20        18.50        -3.30
```

## Output Files

- `portfolio_optimization_results.png` - Performance charts

## Running Tests

```bash
# Run all tests
python tests/test_modules.py

# Or with pytest (more features)
pip install pytest
pytest tests/ -v
```

**What the tests cover (short):**
- Data generation sanity checks (shapes, NaNs, PSD covariance)
- Optimizer outputs valid weights (sum to 1, within bounds)
- Regime detector returns valid regimes
- Backtest runs end-to-end without errors

## Key Concepts

### Why Convex Optimization?
- **Guaranteed global optimum** - No local minima
- **Efficient algorithms** - Polynomial time
- **Rich constraint handling** - Many real-world constraints are convex

### Why Regime Detection?
- Markets behave differently in different conditions
- A single strategy may not work everywhere
- Adaptive strategies are more robust

## Customization

### Change Risk Aversion
```python
optimizer = ConvexPortfolioOptimizer(
    n_assets=10,
    risk_aversion=2.0,  # More conservative
    # ...
)
```

### Add More Assets
```bash
python main.py --assets 20
```

### Use Real Data
Replace `generate_synthetic_stock_data()` in `data.py` with your data loading function.

