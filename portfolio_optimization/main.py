"""
================================================================================
SYSTEMATIC PORTFOLIO TRADING VIA CONVEX OPTIMIZATION
================================================================================

Main entry point for running the complete portfolio optimization system.

USAGE:
------
Full run (5 years of data, all tests):
    python main.py

Quick test (1 year, faster):
    python main.py --quick

Custom parameters:
    python main.py --assets 15 --days 500 --rebalance 10

Without plots (for servers):
    python main.py --no-plot

WHAT THIS DOES:
---------------
1. Generates synthetic market data (or loads real data)
2. Initializes a convex portfolio optimizer with constraints
3. Runs backtest with regime-aware optimization
4. Compares against equal-weight benchmark
5. Tests stability across different market regimes
6. Creates visualizations and reports

KEY RESULTS TO LOOK FOR:
------------------------
- Sharpe Ratio > 1 is good, > 2 is excellent
- Lower turnover = lower transaction costs
- Smaller max drawdown = better risk management
- Outperformance in most regime scenarios = robust strategy

================================================================================
"""

import argparse
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our modules
from .data import generate_synthetic_stock_data
from .optimizer import ConvexPortfolioOptimizer
from .backtest import BacktestEngine
from .stability import run_regime_stability_tests
from .viz import plot_results, print_results_summary


def parse_arguments():
    """
    Parse command line arguments for customizing the run.
    """
    parser = argparse.ArgumentParser(
        description='Systematic Portfolio Trading via Convex Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Full run with defaults
  python main.py --quick          # Quick test (1 year)
  python main.py --assets 15      # Use 15 assets
  python main.py --no-plot        # Skip visualization (for servers)
        """
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick mode: 1 year of data, fewer tests (for testing)'
    )
    
    parser.add_argument(
        '--assets', 
        type=int, 
        default=10,
        help='Number of assets in portfolio (default: 10)'
    )
    
    parser.add_argument(
        '--days', 
        type=int, 
        default=None,
        help='Number of trading days (default: 1260 for 5 years, 252 for --quick)'
    )
    
    parser.add_argument(
        '--rebalance', 
        type=int, 
        default=21,
        help='Rebalancing frequency in days (default: 21 = monthly)'
    )
    
    parser.add_argument(
        '--risk-aversion', 
        type=float, 
        default=1.0,
        help='Risk aversion parameter (default: 1.0, higher = more conservative)'
    )
    
    parser.add_argument(
        '--no-plot', 
        action='store_true',
        help='Skip plotting (useful for headless servers)'
    )
    
    parser.add_argument(
        '--no-stability', 
        action='store_true',
        help='Skip stability tests (faster)'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run the complete portfolio optimization system.
    
    This demonstrates:
    1. Data generation (or loading)
    2. Portfolio optimization with convex programming
    3. Backtesting against benchmark
    4. Regime stability testing
    5. Results visualization and reporting
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set defaults based on mode
    if args.quick:
        n_days = args.days or 252  # 1 year
        n_simulations = 2
        print("\n🚀 QUICK MODE: Using 1 year of data")
    else:
        n_days = args.days or 1260  # 5 years
        n_simulations = 5
    
    print("\n" + "=" * 80)
    print("SYSTEMATIC PORTFOLIO TRADING VIA CONVEX OPTIMIZATION")
    print("=" * 80)
    
    # ========================================================================
    # Step 1: Generate or Load Data
    # ========================================================================
    print("\n[Step 1] Generating synthetic market data...")
    
    prices, returns = generate_synthetic_stock_data(
        n_assets=args.assets,
        n_days=n_days,
        seed=args.seed
    )
    
    print(f"\nData shape: {returns.shape}")
    print(f"Assets: {list(returns.columns)}")
    
    # ========================================================================
    # Step 2: Initialize Portfolio Optimizer
    # ========================================================================
    print("\n[Step 2] Initializing convex portfolio optimizer...")
    
    optimizer = ConvexPortfolioOptimizer(
        n_assets=args.assets,
        risk_aversion=args.risk_aversion,
        max_weight=0.30,        # Max 30% in any single asset
        min_weight=0.0,         # No short selling
        max_turnover=0.50,      # Max 50% turnover per rebalance
        transaction_cost=0.001, # 10 basis points per trade
        max_risk=0.25           # Max 25% annualized volatility
    )
    
    print("Optimizer initialized with:")
    print(f"  - Risk aversion: {optimizer.risk_aversion}")
    print(f"  - Max weight per asset: {optimizer.max_weight * 100}%")
    print(f"  - Max turnover: {optimizer.max_turnover * 100}%")
    print(f"  - Transaction cost: {optimizer.transaction_cost * 100}%")
    print(f"  - Max portfolio risk: {optimizer.max_risk * 100}%")
    
    # ========================================================================
    # Step 3: Run Backtest
    # ========================================================================
    print("\n[Step 3] Running backtest...")
    
    engine = BacktestEngine(
        returns=returns,
        initial_capital=1_000_000,
        rebalance_frequency=args.rebalance,
        transaction_cost=0.001,
        lookback_window=60
    )
    
    # Run optimized strategy with regime awareness
    print("\nRunning optimized strategy (regime-aware)...")
    optimized_results = engine.run_backtest(
        optimizer=optimizer,
        regime_aware=True,
        use_multi_period=not args.quick  # Skip multi-period in quick mode
    )
    
    # Run benchmark strategy
    print("\nRunning benchmark strategy (equal-weight)...")
    benchmark_results = engine.run_benchmark(strategy='equal_weight')
    
    # ========================================================================
    # Step 4: Print Results Summary
    # ========================================================================
    print_results_summary(optimized_results, benchmark_results)
    
    # ========================================================================
    # Step 5: Run Regime Stability Tests (optional)
    # ========================================================================
    stability_results = None
    if not args.no_stability:
        print("\n[Step 4] Running regime stability tests...")
        stability_results = run_regime_stability_tests(
            returns, 
            n_simulations=n_simulations
        )
    
    # ========================================================================
    # Step 6: Visualize Results (optional)
    # ========================================================================
    if not args.no_plot:
        print("\n[Step 5] Generating visualizations...")
        try:
            plot_results(
                optimized_results,
                benchmark_results,
                returns.index,
                save_path='portfolio_optimization_results.png',
                show_plot=True
            )
        except Exception as e:
            print(f"Warning: Could not display plot ({e})")
            print("Plot saved to portfolio_optimization_results.png")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    
    print("\n Key Achievements:")
    
    # Calculate turnover reduction
    if benchmark_results['avg_turnover'] > 0:
        turnover_reduction = (1 - optimized_results['avg_turnover'] / benchmark_results['avg_turnover']) * 100
        if turnover_reduction > 0:
            print(f"✓ Achieved {turnover_reduction:.1f}% lower turnover vs benchmark")
        else:
            print(f"• Turnover increased by {-turnover_reduction:.1f}% (more active trading)")
    
    # Sharpe improvement
    if benchmark_results['sharpe_ratio'] > 0:
        sharpe_improvement = ((optimized_results['sharpe_ratio'] / benchmark_results['sharpe_ratio']) - 1) * 100
        if sharpe_improvement > 0:
            print(f"✓ Improved risk-adjusted returns by {sharpe_improvement:.1f}%")
        else:
            print(f"• Risk-adjusted returns decreased by {-sharpe_improvement:.1f}%")
    
    # Stability
    if stability_results:
        print(f"✓ Strategy outperformed in {stability_results['win_rate']:.0f}% of regime scenarios")
    
    print(" See result in portfolio_optimization_results.png (visualization)")
    
    return {
        'optimized_results': optimized_results,
        'benchmark_results': benchmark_results,
        'stability_results': stability_results,
        'prices': prices,
        'returns': returns
    }


if __name__ == "__main__":
    results = main()
