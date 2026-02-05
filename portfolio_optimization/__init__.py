"""
Portfolio Optimization Package

This package provides tools for systematic portfolio trading using convex optimization.

Modules:
--------
- data: Data generation and statistical calculations
- optimizer: Convex portfolio optimizer with CVXPY
- regimes: Market regime detection
- backtest: Backtesting engine
- stability: Regime stability testing
- viz: Visualization and reporting
"""

from .data import generate_synthetic_stock_data, calculate_statistics
from .optimizer import ConvexPortfolioOptimizer
from .regimes import RegimeDetector
from .backtest import BacktestEngine
from .stability import run_regime_stability_tests
from .viz import plot_results, print_results_summary

__all__ = [
    'generate_synthetic_stock_data',
    'calculate_statistics',
    'ConvexPortfolioOptimizer',
    'RegimeDetector',
    'BacktestEngine',
    'run_regime_stability_tests',
    'plot_results',
    'print_results_summary'
]

__version__ = '1.0.0'
