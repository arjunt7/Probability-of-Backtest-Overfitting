"""
================================================================================
VISUALIZATION AND REPORTING MODULE
================================================================================

This module creates charts and reports for communicating results.

Charts Included:
1. Portfolio Value Over Time - Shows wealth growth
2. Cumulative Returns - Highlights outperformance periods
3. Drawdowns - Shows risk and recovery periods
4. Rolling Sharpe Ratio - Performance consistency over time
5. Turnover Comparison - Trading activity
6. Performance Metrics Bar Chart - Summary comparison

Tips for Presentations:
-----------------------
- Lead with the portfolio value chart (most intuitive)
- Use Sharpe ratio for risk-adjusted comparisons
- Drawdowns show risk management effectiveness
- Turnover reduction shows practical benefits (lower costs)

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_results(
    optimized_results: Dict,
    benchmark_results: Dict,
    dates: pd.DatetimeIndex,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Create comprehensive visualization of backtest results.
    
    Parameters:
    -----------
    optimized_results : dict
        Results from optimized strategy (from BacktestEngine)
    benchmark_results : dict
        Results from benchmark strategy
    dates : pd.DatetimeIndex
        Date index for x-axis
    save_path : str, optional
        Path to save the figure (e.g., 'results.png')
    show_plot : bool
        Whether to display the plot interactively
    
    Example:
    --------
    >>> plot_results(opt_results, bench_results, returns.index, 'results.png')
    """
    # Create figure with 6 subplots (3 rows × 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Adjust dates to match portfolio values length
    # Portfolio values has one more entry than trading days
    plot_dates = dates[-len(optimized_results['portfolio_values']):]
    
    
    # Plot 1: Portfolio Value Over Time
    
    # This is the most intuitive chart - shows actual wealth growth
    ax1 = axes[0, 0]
    ax1.plot(
        plot_dates, 
        optimized_results['portfolio_values'], 
        label='Convex Optimization', 
        linewidth=2, 
        color='blue'
    )
    ax1.plot(
        plot_dates, 
        benchmark_results['portfolio_values'], 
        label='Equal Weight Benchmark', 
        linewidth=2, 
        color='orange', 
        alpha=0.7
    )
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    
    # Plot 2: Cumulative Returns Comparison
    
    # Shows periods of outperformance (green) and underperformance (red)
    ax2 = axes[0, 1]
    
    opt_cumret = optimized_results['portfolio_values'] / optimized_results['portfolio_values'][0] - 1
    bench_cumret = benchmark_results['portfolio_values'] / benchmark_results['portfolio_values'][0] - 1
    
    # Fill between to highlight out/under-performance
    ax2.fill_between(
        plot_dates, opt_cumret, bench_cumret, 
        where=opt_cumret >= bench_cumret, 
        color='green', alpha=0.3, 
        label='Outperformance'
    )
    ax2.fill_between(
        plot_dates, opt_cumret, bench_cumret, 
        where=opt_cumret < bench_cumret, 
        color='red', alpha=0.3, 
        label='Underperformance'
    )
    ax2.plot(plot_dates, opt_cumret, color='blue', linewidth=2, label='Optimized')
    ax2.plot(plot_dates, bench_cumret, color='orange', linewidth=2, label='Benchmark')
    
    ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    
    # Plot 3: Drawdowns
    
    # Shows risk - how far did we fall from peak?
    ax3 = axes[1, 0]
    
    # Calculate drawdowns
    opt_cummax = np.maximum.accumulate(optimized_results['portfolio_values'])
    opt_drawdown = (optimized_results['portfolio_values'] - opt_cummax) / opt_cummax
    
    bench_cummax = np.maximum.accumulate(benchmark_results['portfolio_values'])
    bench_drawdown = (benchmark_results['portfolio_values'] - bench_cummax) / bench_cummax
    
    ax3.fill_between(
        plot_dates, opt_drawdown, 0, 
        color='blue', alpha=0.3, 
        label='Optimized'
    )
    ax3.fill_between(
        plot_dates, bench_drawdown, 0, 
        color='orange', alpha=0.3, 
        label='Benchmark'
    )
    
    ax3.set_title('Drawdowns (Peak-to-Trough Decline)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    
    # Plot 4: Rolling Sharpe Ratio (1-year)
    
    # Shows consistency of risk-adjusted performance
    ax4 = axes[1, 1]
    window = 252  # 1 year of trading days
    
    if len(optimized_results['daily_returns']) > window:
        # Calculate rolling Sharpe: (mean * 252 - rf) / (std * sqrt(252))
        opt_rolling_sharpe = pd.Series(optimized_results['daily_returns']).rolling(window).apply(
            lambda x: (x.mean() * 252 - 0.02) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            raw=True
        )
        bench_rolling_sharpe = pd.Series(benchmark_results['daily_returns']).rolling(window).apply(
            lambda x: (x.mean() * 252 - 0.02) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            raw=True
        )
        
        rolling_dates = plot_dates[1:len(opt_rolling_sharpe) + 1]
        ax4.plot(
            rolling_dates, opt_rolling_sharpe.values, 
            label='Optimized', linewidth=2, color='blue'
        )
        ax4.plot(
            rolling_dates, bench_rolling_sharpe.values, 
            label='Benchmark', linewidth=2, color='orange', alpha=0.7
        )
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Good (>1)')
        
        ax4.set_title('Rolling 1-Year Sharpe Ratio', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for rolling Sharpe', 
                ha='center', va='center', transform=ax4.transAxes)
    
    
    # Plot 5: Turnover Comparison
    
    # Shows trading activity - lower is better (less costs)
    ax5 = axes[2, 0]
    
    opt_turnover = optimized_results['turnover_history']
    bench_turnover = benchmark_results['turnover_history']
    
    if opt_turnover and bench_turnover:
        # Ensure same length for comparison
        min_len = min(len(opt_turnover), len(bench_turnover))
        x = range(min_len)
        width = 0.35
        
        ax5.bar(
            [i - width/2 for i in x], 
            opt_turnover[:min_len], 
            width, 
            label='Optimized', 
            color='blue', 
            alpha=0.7
        )
        ax5.bar(
            [i + width/2 for i in x], 
            bench_turnover[:min_len], 
            width, 
            label='Benchmark', 
            color='orange', 
            alpha=0.7
        )
        
        ax5.set_title('Turnover per Rebalancing Period', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Rebalancing Period')
        ax5.set_ylabel('Turnover (fraction of portfolio traded)')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'No turnover data available', 
                ha='center', va='center', transform=ax5.transAxes)
    
    
    # Plot 6: Performance Metrics Comparison
    
    # Bar chart summarizing key metrics
    ax6 = axes[2, 1]
    
    metrics = ['Annual\nReturn (%)', 'Sharpe\nRatio', 'Max DD\n(%)', 'Avg Turn-\nover (%)']
    
    opt_values = [
        optimized_results['annualized_return'] * 100,
        optimized_results['sharpe_ratio'],
        abs(optimized_results['max_drawdown']) * 100,
        optimized_results['avg_turnover'] * 100
    ]
    bench_values = [
        benchmark_results['annualized_return'] * 100,
        benchmark_results['sharpe_ratio'],
        abs(benchmark_results['max_drawdown']) * 100,
        benchmark_results['avg_turnover'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, opt_values, width, label='Optimized', color='blue', alpha=0.7)
    bars2 = ax6.bar(x + width/2, bench_values, width, label='Benchmark', color='orange', alpha=0.7)
    
    ax6.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, opt_values):
        height = bar.get_height()
        ax6.annotate(
            f'{val:.1f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9
        )
    
    for bar, val in zip(bars2, bench_values):
        height = bar.get_height()
        ax6.annotate(
            f'{val:.1f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom', fontsize=9
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def print_results_summary(optimized_results: Dict, benchmark_results: Dict):
    """
    Print a comprehensive summary of backtest results to console.
    
    This is useful for quick analysis and for copying into reports.
    
    Parameters:
    -----------
    optimized_results : dict
        Results from optimized strategy
    benchmark_results : dict
        Results from benchmark strategy
    """
    print("\n" + "=" * 80)
    print("SYSTEMATIC PORTFOLIO OPTIMIZATION - BACKTEST RESULTS")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Performance Comparison Table
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("PERFORMANCE COMPARISON")
    print("-" * 40)
    
    print(f"\n{'Metric':<30} {'Optimized':>15} {'Benchmark':>15} {'Difference':>15}")
    print("-" * 75)
    
    # Total Return
    opt_ret = optimized_results['total_return'] * 100
    bench_ret = benchmark_results['total_return'] * 100
    diff_ret = opt_ret - bench_ret
    print(f"{'Total Return (%)':<30} {opt_ret:>15.2f} {bench_ret:>15.2f} {diff_ret:>+15.2f}")
    
    # Annualized Return
    opt_ann = optimized_results['annualized_return'] * 100
    bench_ann = benchmark_results['annualized_return'] * 100
    diff_ann = opt_ann - bench_ann
    print(f"{'Annualized Return (%)':<30} {opt_ann:>15.2f} {bench_ann:>15.2f} {diff_ann:>+15.2f}")
    
    # Volatility
    opt_vol = optimized_results['annualized_volatility'] * 100
    bench_vol = benchmark_results['annualized_volatility'] * 100
    diff_vol = opt_vol - bench_vol
    print(f"{'Annualized Volatility (%)':<30} {opt_vol:>15.2f} {bench_vol:>15.2f} {diff_vol:>+15.2f}")
    
    # Sharpe Ratio
    opt_sharpe = optimized_results['sharpe_ratio']
    bench_sharpe = benchmark_results['sharpe_ratio']
    diff_sharpe = opt_sharpe - bench_sharpe
    print(f"{'Sharpe Ratio':<30} {opt_sharpe:>15.2f} {bench_sharpe:>15.2f} {diff_sharpe:>+15.2f}")
    
    # Sortino Ratio
    opt_sortino = optimized_results['sortino_ratio']
    bench_sortino = benchmark_results['sortino_ratio']
    diff_sortino = opt_sortino - bench_sortino
    print(f"{'Sortino Ratio':<30} {opt_sortino:>15.2f} {bench_sortino:>15.2f} {diff_sortino:>+15.2f}")
    
    # Max Drawdown
    opt_dd = optimized_results['max_drawdown'] * 100
    bench_dd = benchmark_results['max_drawdown'] * 100
    diff_dd = opt_dd - bench_dd
    print(f"{'Max Drawdown (%)':<30} {opt_dd:>15.2f} {bench_dd:>15.2f} {diff_dd:>+15.2f}")
    
    # Calmar Ratio
    opt_calmar = optimized_results['calmar_ratio']
    bench_calmar = benchmark_results['calmar_ratio']
    diff_calmar = opt_calmar - bench_calmar
    print(f"{'Calmar Ratio':<30} {opt_calmar:>15.2f} {bench_calmar:>15.2f} {diff_calmar:>+15.2f}")
    
    # -------------------------------------------------------------------------
    # Trading Activity
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("TRADING ACTIVITY")
    print("-" * 40)
    
    opt_turn = optimized_results['avg_turnover'] * 100
    bench_turn = benchmark_results['avg_turnover'] * 100
    diff_turn = opt_turn - bench_turn
    print(f"\n{'Average Turnover (%)':<30} {opt_turn:>15.2f} {bench_turn:>15.2f} {diff_turn:>+15.2f}")
    
    opt_total_turn = optimized_results['total_turnover'] * 100
    bench_total_turn = benchmark_results['total_turnover'] * 100
    diff_total_turn = opt_total_turn - bench_total_turn
    print(f"{'Total Turnover (%)':<30} {opt_total_turn:>15.2f} {bench_total_turn:>15.2f} {diff_total_turn:>+15.2f}")
    
    # Turnover reduction percentage
    if bench_turn > 0:
        turnover_reduction = (1 - opt_turn / bench_turn) * 100
        print(f"\n{'Turnover Reduction':<30} {turnover_reduction:>15.2f}%")
    
    # -------------------------------------------------------------------------
    # Key Insights
    # -------------------------------------------------------------------------
    print("\n" + "-" * 40)
    print("KEY INSIGHTS")
    print("-" * 40)
    
    # Risk-adjusted outperformance
    if bench_sharpe > 0:
        sharpe_improvement = ((opt_sharpe / bench_sharpe) - 1) * 100
        print(f"\n• Sharpe Ratio improved by {sharpe_improvement:.1f}%")
    
    # Drawdown comparison
    if bench_dd != 0:
        dd_improvement = ((abs(bench_dd) - abs(opt_dd)) / abs(bench_dd)) * 100
        if dd_improvement > 0:
            print(f"• Maximum Drawdown reduced by {dd_improvement:.1f}%")
        else:
            print(f"• Maximum Drawdown increased by {-dd_improvement:.1f}%")
    
    # Turnover efficiency
    if bench_turn > 0 and turnover_reduction > 0:
        print(f"• Turnover reduced by {turnover_reduction:.1f}% while improving returns")
    
    print("\n" + "=" * 80)


def create_regime_summary_plot(
    regime_history: list,
    dates: pd.DatetimeIndex,
    save_path: Optional[str] = None
):
    """
    Create a visualization of detected market regimes over time.
    
    Parameters:
    -----------
    regime_history : list
        List of regime dictionaries from backtest
    dates : pd.DatetimeIndex
        Date index
    save_path : str, optional
        Path to save the figure
    """
    if not regime_history:
        print("No regime history available")
        return
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Color map for regimes
    regime_colors = {
        'LOW_VOL_BULL': 'green',
        'HIGH_VOL_BULL': 'lightgreen',
        'LOW_VOL_BEAR': 'lightsalmon',
        'HIGH_VOL_BEAR': 'red'
    }
    
    regimes = [r['regime'] for r in regime_history]
    
    # Create colored bars for each regime period
    for i, regime in enumerate(regimes):
        color = regime_colors.get(regime, 'gray')
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.7)
    
    # Add legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.7) 
               for c in regime_colors.values()]
    labels = list(regime_colors.keys())
    ax.legend(handles, labels, loc='upper right')
    
    ax.set_title('Detected Market Regimes Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rebalancing Period')
    ax.set_ylabel('Regime')
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Regime plot saved to: {save_path}")
    
    plt.show()
