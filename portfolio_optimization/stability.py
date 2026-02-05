"""
================================================================================
REGIME STABILITY TESTING MODULE
================================================================================

This module tests strategy robustness across different market conditions.

Why Stability Testing?
----------------------
A strategy that only works in one market condition is fragile.
We want strategies that:
1. Outperform across multiple regimes
2. Don't catastrophically fail in any regime
3. Adapt appropriately to changing conditions

Test Scenarios:
---------------
1. Normal Market    - Baseline conditions
2. Low Volatility   - Unusually calm markets (easy mode)
3. High Volatility  - Turbulent markets (hard mode)
4. Bull Market      - Upward trending with higher returns
5. Bear Market      - Downward trending with higher volatility

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from .optimizer import ConvexPortfolioOptimizer
from .backtest import BacktestEngine


def run_regime_stability_tests(
    returns: pd.DataFrame,
    n_simulations: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Test strategy stability across different market regimes.
    
    We modify the historical returns to simulate different market conditions,
    then run backtests in each scenario to see how the strategy performs.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns data
    n_simulations : int
        Number of regime scenarios to test (max 5)
    verbose : bool
        Whether to print detailed progress
    
    Returns:
    --------
    stability_results : dict
        Contains:
        - detailed_results: DataFrame with all scenario results
        - avg_outperformance: Average Sharpe improvement vs benchmark
        - consistency_ratio: Outperformance / Std of outperformance
        - win_rate: % of scenarios where we beat benchmark
    
    Example:
    --------
    >>> results = run_regime_stability_tests(returns_df, n_simulations=5)
    >>> print(f"Win rate: {results['win_rate']:.0f}%")
    """
    if verbose:
        print("\n" + "=" * 80)
        print("REGIME STABILITY TESTING")
        print("=" * 80)
    
    # Define test scenarios
    # Each scenario modifies returns to simulate different market conditions
    regime_scenarios = [
        {
            'name': 'Normal Market',
            'vol_multiplier': 1.0,      # No change to volatility
            'return_shift': 0.0,        # No change to average return
            'description': 'Baseline - unchanged market conditions'
        },
        {
            'name': 'Low Volatility',
            'vol_multiplier': 0.5,      # Half the volatility
            'return_shift': 0.0,
            'description': 'Unusually calm markets - easy trading'
        },
        {
            'name': 'High Volatility',
            'vol_multiplier': 2.0,      # Double the volatility
            'return_shift': 0.0,
            'description': 'Crisis-like volatility - challenging'
        },
        {
            'name': 'Bull Market',
            'vol_multiplier': 1.0,
            'return_shift': 0.05 / 252, # Add 5% annual return
            'description': 'Strong uptrend - rising tide lifts all boats'
        },
        {
            'name': 'Bear Market',
            'vol_multiplier': 1.5,      # 50% more volatility
            'return_shift': -0.10 / 252, # Subtract 10% annual return
            'description': 'Downturn with elevated volatility'
        }
    ]
    
    results = []
    
    for scenario in regime_scenarios[:n_simulations]:
        if verbose:
            print(f"\nTesting scenario: {scenario['name']}")
            print(f"  {scenario['description']}")
        
        # =====================================================================
        # Modify returns to simulate this scenario
        # =====================================================================
        modified_returns = returns.copy()
        
        # Apply volatility scaling
        # Keep means the same, scale deviations from mean
        means = modified_returns.mean()
        modified_returns = means + (modified_returns - means) * scenario['vol_multiplier']
        
        # Apply return shift (adds/subtracts constant from all returns)
        modified_returns = modified_returns + scenario['return_shift']
        
        # =====================================================================
        # Run backtest with modified returns
        # =====================================================================
        optimizer = ConvexPortfolioOptimizer(
            n_assets=len(returns.columns),
            risk_aversion=1.0,
            max_weight=0.30,
            max_turnover=0.50,
            transaction_cost=0.001,
            max_risk=0.25
        )
        
        engine = BacktestEngine(
            returns=modified_returns,
            initial_capital=1_000_000,
            rebalance_frequency=21,
            transaction_cost=0.001,
            lookback_window=60
        )
        
        # Run optimized strategy
        opt_results = engine.run_backtest(
            optimizer,
            regime_aware=True,
            use_multi_period=False  # Single-period for speed
        )
        
        # Run benchmark
        bench_results = engine.run_benchmark()
        
        # Store results
        scenario_result = {
            'scenario': scenario['name'],
            'optimized_sharpe': opt_results['sharpe_ratio'],
            'benchmark_sharpe': bench_results['sharpe_ratio'],
            'sharpe_difference': opt_results['sharpe_ratio'] - bench_results['sharpe_ratio'],
            'optimized_return': opt_results['annualized_return'],
            'benchmark_return': bench_results['annualized_return'],
            'optimized_volatility': opt_results['annualized_volatility'],
            'optimized_max_dd': opt_results['max_drawdown'],
            'benchmark_max_dd': bench_results['max_drawdown'],
            'avg_turnover': opt_results['avg_turnover']
        }
        
        results.append(scenario_result)
        
        if verbose:
            outperformance = opt_results['sharpe_ratio'] - bench_results['sharpe_ratio']
            print(f"  Optimized Sharpe: {opt_results['sharpe_ratio']:.2f}")
            print(f"  Benchmark Sharpe: {bench_results['sharpe_ratio']:.2f}")
            print(f"  Outperformance: {outperformance:+.2f}")
    
    
    # Calculate summary statistics
    
    df_results = pd.DataFrame(results)
    
    sharpe_diff = df_results['optimized_sharpe'] - df_results['benchmark_sharpe']
    avg_outperformance = sharpe_diff.mean()
    std_outperformance = sharpe_diff.std()
    
    # Consistency ratio: how reliable is our outperformance?
    # Higher = more consistent
    consistency_ratio = avg_outperformance / std_outperformance if std_outperformance > 0 else np.inf
    
    # Win rate: what percentage of scenarios did we beat the benchmark?
    win_rate = (df_results['optimized_sharpe'] > df_results['benchmark_sharpe']).mean() * 100
    
    if verbose:
        print("\n" + "-" * 40)
        print("STABILITY SUMMARY")
        print("-" * 40)
        print(f"\nAverage Sharpe Outperformance: {avg_outperformance:+.2f}")
        print(f"Std Dev of Outperformance: {std_outperformance:.2f}")
        print(f"Consistency Ratio: {consistency_ratio:.2f}")
        print(f"Win Rate across regimes: {win_rate:.0f}%")
    
    return {
        'detailed_results': df_results,
        'avg_outperformance': avg_outperformance,
        'std_outperformance': std_outperformance,
        'consistency_ratio': consistency_ratio,
        'win_rate': win_rate
    }


def run_stress_test(
    returns: pd.DataFrame,
    crash_magnitude: float = -0.20,
    recovery_days: int = 60
) -> Dict:
    """
    Run a stress test simulating a market crash and recovery.
    
    This tests how the portfolio behaves during extreme events like:
    - 2008 Financial Crisis
    - 2020 COVID Crash
    - Flash crashes
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns data
    crash_magnitude : float
        Size of the simulated crash (e.g., -0.20 = -20%)
    recovery_days : int
        Days for market to recover
    
    Returns:
    --------
    stress_results : dict
        Performance during and after the stress event
    """
    print("\n" + "=" * 80)
    print("STRESS TEST: SIMULATED MARKET CRASH")
    print("=" * 80)
    
    # Create modified returns with a crash event
    modified_returns = returns.copy()
    n_days = len(modified_returns)
    
    # Insert crash at the midpoint of the data
    crash_start = n_days // 2
    crash_end = crash_start + recovery_days
    
    # Simulate sudden crash
    crash_day_return = crash_magnitude / 5  # Spread over 5 days
    for i in range(5):
        if crash_start + i < n_days:
            modified_returns.iloc[crash_start + i] = crash_day_return
    
    # Simulate gradual recovery
    for i in range(5, min(recovery_days, n_days - crash_start)):
        recovery_factor = (i - 5) / (recovery_days - 5)
        original = returns.iloc[crash_start + i].values
        modified_returns.iloc[crash_start + i] = original * (0.5 + 0.5 * recovery_factor)
    
    print(f"Simulated {crash_magnitude*100:.0f}% crash at day {crash_start}")
    print(f"Recovery period: {recovery_days} days")
    
    # Run backtest
    optimizer = ConvexPortfolioOptimizer(
        n_assets=len(returns.columns),
        risk_aversion=1.0,
        max_weight=0.30,
        max_turnover=0.50,
        transaction_cost=0.001
    )
    
    engine = BacktestEngine(
        returns=modified_returns,
        initial_capital=1_000_000,
        rebalance_frequency=21,
        transaction_cost=0.001
    )
    
    opt_results = engine.run_backtest(optimizer, regime_aware=True)
    bench_results = engine.run_benchmark()
    
    print(f"\nStress Test Results:")
    print(f"  Optimized Max Drawdown: {opt_results['max_drawdown']*100:.1f}%")
    print(f"  Benchmark Max Drawdown: {bench_results['max_drawdown']*100:.1f}%")
    print(f"  Optimized Final Return: {opt_results['total_return']*100:.1f}%")
    print(f"  Benchmark Final Return: {bench_results['total_return']*100:.1f}%")
    
    return {
        'optimized': opt_results,
        'benchmark': bench_results,
        'crash_magnitude': crash_magnitude,
        'recovery_days': recovery_days
    }
