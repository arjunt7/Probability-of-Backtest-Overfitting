"""
================================================================================
BACKTESTING ENGINE MODULE
================================================================================

This module simulates portfolio performance over historical data.

Backtesting answers: "How would this strategy have performed in the past?"

CRITICAL CONSIDERATIONS:
------------------------
1. Look-ahead bias: We CANNOT use future data to make past decisions
   - Only use data available at each point in time
   - Rolling windows, not full-sample estimates

2. Transaction costs: Real trading costs money
   - Brokerage fees, bid-ask spread, market impact
   - High turnover = high costs = lower returns

3. Survivorship bias: (Not addressed in synthetic data)
   - Real data should include delisted stocks
   - Otherwise we overestimate returns

4. Rebalancing frequency: How often to adjust
   - Too frequent = high costs
   - Too infrequent = drift from optimal

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .optimizer import ConvexPortfolioOptimizer
from .regimes import RegimeDetector
from .data import calculate_statistics


class BacktestEngine:
    """
    Backtesting engine for portfolio strategies.
    
    This class simulates trading over historical data, carefully avoiding
    look-ahead bias by only using past information at each decision point.
    
    Workflow:
    ---------
    1. Start with initial capital and equal weights
    2. Each day: Update portfolio value based on asset returns
    3. On rebalancing days: Optimize weights using past data only
    4. Track all metrics: returns, drawdowns, turnover, etc.
    
    Example:
    --------
    >>> engine = BacktestEngine(returns_df, initial_capital=1_000_000)
    >>> optimizer = ConvexPortfolioOptimizer(n_assets=10)
    >>> results = engine.run_backtest(optimizer, regime_aware=True)
    >>> print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        initial_capital: float = 1_000_000,
        rebalance_frequency: int = 21,
        transaction_cost: float = 0.001,
        lookback_window: int = 60
    ):
        """
        Initialize backtest engine.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns data (rows = dates, columns = assets)
            These are simple returns: (P_t - P_{t-1}) / P_{t-1}
        
        initial_capital : float
            Starting portfolio value in dollars
            Default: $1,000,000
        
        rebalance_frequency : int
            Days between rebalancing (default: 21 ≈ monthly)
            - 5 = weekly
            - 21 = monthly
            - 63 = quarterly
        
        transaction_cost : float
            Cost per unit traded (default: 0.001 = 10 basis points)
            Total cost = turnover × transaction_cost × portfolio_value
        
        lookback_window : int
            Days of data to use for optimization (default: 60)
            Longer = more stable estimates, less responsive
            Shorter = more responsive, noisier estimates
        """
        self.returns = returns
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.n_assets = len(returns.columns)
    
    def run_backtest(
        self,
        optimizer: ConvexPortfolioOptimizer,
        regime_aware: bool = True,
        use_multi_period: bool = False
    ) -> Dict:
        """
        Run a full backtest of the portfolio strategy.
        
        Parameters:
        -----------
        optimizer : ConvexPortfolioOptimizer
            The optimizer to use for portfolio construction
        
        regime_aware : bool
            Whether to adapt strategy parameters to market regimes
            If True: More aggressive in calm markets, defensive in crises
            If False: Use same parameters throughout
        
        use_multi_period : bool
            Whether to use multi-period optimization
            If True: Consider future periods when making today's decision
            If False: Myopic single-period optimization
        
        Returns:
        --------
        results : dict
            Comprehensive backtest results including:
            - portfolio_values: Daily portfolio value
            - daily_returns: Daily portfolio returns
            - total_return: Overall return
            - annualized_return: Annualized return
            - sharpe_ratio: Risk-adjusted return measure
            - max_drawdown: Worst peak-to-trough decline
            - avg_turnover: Average trading activity
            - weights_history: Portfolio weights over time
            - regime_history: Detected regimes over time
        """
        n_days = len(self.returns)
        
        # Initialize tracking variables
        portfolio_values = [self.initial_capital]
        weights_history = []
        turnover_history = []
        regime_history = []
        
        # Start with equal weights (1/N allocation)
        current_weights = np.ones(self.n_assets) / self.n_assets
        weights_history.append(current_weights.copy())
        
        # Initialize regime detector
        regime_detector = RegimeDetector()
        
        # Store original optimizer parameters (to restore after regime adjustments)
        original_risk_aversion = optimizer.risk_aversion
        original_max_risk = optimizer.max_risk
        
        print("Running backtest...")
        print(f"Start date: {self.returns.index[self.lookback_window]}")
        print(f"End date: {self.returns.index[-1]}")
        print(f"Rebalancing: Every {self.rebalance_frequency} days")
        
        
        # Main backtest loop
        
        for t in range(self.lookback_window, n_days):
            # Get today's returns for all assets
            today_returns = self.returns.iloc[t].values
            
            
            # Step 1: Update portfolio value based on today's returns
            
            # Portfolio return = weighted average of individual asset returns
            portfolio_return = np.sum(current_weights * today_returns)
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
            
            
            # Step 2: Update weights due to price movements (drift)
            
            # Weights change as assets perform differently
            # If Asset A goes up 10% and Asset B stays flat:
            # - Asset A's weight increases
            # - Asset B's weight decreases
            drifted_weights = current_weights * (1 + today_returns)
            drifted_weights = drifted_weights / drifted_weights.sum()
            current_weights = drifted_weights
            
            
            # Step 3: Check if it's a rebalancing day
            
            days_since_start = t - self.lookback_window
            is_rebalance_day = (
                days_since_start % self.rebalance_frequency == 0 
                and days_since_start > 0
            )
            
            if is_rebalance_day:
                # Get historical data for optimization (NO LOOK-AHEAD!)
                # Only use data from BEFORE today
                historical_returns = self.returns.iloc[t - self.lookback_window:t]
                
                # Detect regime if using regime-aware optimization
                if regime_aware:
                    regime_info = regime_detector.detect_regime(historical_returns)
                    regime_history.append(regime_info)
                    
                    # Adapt optimizer parameters based on regime
                    optimizer.risk_aversion = regime_info['recommended_risk_aversion']
                    optimizer.max_risk = regime_info['recommended_max_risk']
                
                # Calculate statistics for optimization
                stats = calculate_statistics(
                    historical_returns, 
                    self.lookback_window,
                    ensure_psd=True  # Critical for numerical stability
                )
                
                # -------------------------------------------------------------
                # Get optimal weights
                # -------------------------------------------------------------
                if use_multi_period and t + 3 < n_days:
                    # Multi-period: look ahead using current estimates
                    # (In practice, you'd use forecasts)
                    future_returns = [stats['expected_returns']] * 3
                    future_covs = [stats['covariance_matrix']] * 3
                    
                    optimal_weights = optimizer.multi_period_optimize(
                        future_returns,
                        future_covs,
                        current_weights,
                        horizon=3
                    )
                else:
                    optimal_weights = optimizer.optimize(
                        stats['expected_returns'],
                        stats['covariance_matrix'],
                        current_weights
                    )
                
                # Calculate turnover (trading activity)
                turnover = np.sum(np.abs(optimal_weights - current_weights))
                turnover_history.append(turnover)
                
                # Apply transaction costs
                trading_cost = turnover * self.transaction_cost * portfolio_values[-1]
                portfolio_values[-1] -= trading_cost
                
                # Update weights to optimal
                current_weights = optimal_weights
            
            weights_history.append(current_weights.copy())
        
        # Restore original optimizer parameters
        optimizer.risk_aversion = original_risk_aversion
        optimizer.max_risk = original_max_risk
        
        
        # Calculate performance metrics
        
        portfolio_values = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        results = self._calculate_metrics(
            portfolio_values,
            daily_returns,
            turnover_history,
            weights_history,
            regime_history
        )
        
        return results
    
    def run_benchmark(self, strategy: str = 'equal_weight') -> Dict:
        """
        Run benchmark strategy for comparison.
        
        The benchmark helps answer: "Is our optimization actually adding value?"
        If we can't beat a simple equal-weight strategy, our optimization
        might not be worth the complexity.
        
        Parameters:
        -----------
        strategy : str
            'equal_weight': 1/N allocation (no optimization)
                - Simple, hard to beat in many studies
                - No estimation error
        
        Returns:
        --------
        results : dict
            Benchmark performance results
        """
        n_days = len(self.returns)
        portfolio_values = [self.initial_capital]
        weights_history = []
        turnover_history = []
        
        # Equal weight allocation: 1/N in each asset
        target_weights = np.ones(self.n_assets) / self.n_assets
        weights = target_weights.copy()
        weights_history.append(weights.copy())
        
        print("\nRunning benchmark strategy (equal-weight)...")
        
        for t in range(self.lookback_window, n_days):
            today_returns = self.returns.iloc[t].values
            
            # Portfolio return
            portfolio_return = np.sum(weights * today_returns)
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
            
            # Weights drift with price movements
            drifted_weights = weights * (1 + today_returns)
            drifted_weights = drifted_weights / drifted_weights.sum()
            
            # Rebalance back to equal weight periodically
            days_since_start = t - self.lookback_window
            if days_since_start % self.rebalance_frequency == 0 and days_since_start > 0:
                # Calculate turnover
                turnover = np.sum(np.abs(target_weights - drifted_weights))
                turnover_history.append(turnover)
                
                # Transaction costs
                trading_cost = turnover * self.transaction_cost * portfolio_values[-1]
                portfolio_values[-1] -= trading_cost
                
                # Reset to equal weight
                weights = target_weights.copy()
            else:
                weights = drifted_weights
            
            weights_history.append(weights.copy())
        
        portfolio_values = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        results = self._calculate_metrics(
            portfolio_values,
            daily_returns,
            turnover_history,
            weights_history,
            []  # No regime history for benchmark
        )
        
        return results
    
    def _calculate_metrics(
        self,
        portfolio_values: np.ndarray,
        daily_returns: np.ndarray,
        turnover_history: List,
        weights_history: List,
        regime_history: List
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Key Metrics Explained:
        ----------------------
        - Total Return: (Final Value / Initial Value) - 1
        
        - Sharpe Ratio: (Return - Risk Free Rate) / Volatility
          Measures excess return per unit of risk
          > 1 is good, > 2 is excellent
        
        - Sortino Ratio: Like Sharpe but only penalizes downside volatility
          Better for asymmetric return distributions
        
        - Max Drawdown: Worst peak-to-trough decline
          If you invested at the worst time, how much would you lose?
        
        - Calmar Ratio: Return / |Max Drawdown|
          How much return per unit of worst-case pain
        """
        # Total return: how much did we make overall?
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        
        # Annualized return: what's the equivalent yearly return?
        n_years = len(daily_returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        
        # Annualized volatility: how bumpy was the ride?
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        
        # Sharpe ratio: risk-adjusted return (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        # Maximum drawdown: worst peak-to-trough decline
        cumulative_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio: return per unit of max drawdown
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Average turnover: how much did we trade on average?
        avg_turnover = np.mean(turnover_history) if turnover_history else 0
        total_turnover = np.sum(turnover_history) if turnover_history else 0
        
        # Sortino ratio: only penalizes downside volatility
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.001
        sortino_ratio = (annualized_return - risk_free_rate) / downside_vol
        
        return {
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'avg_turnover': avg_turnover,
            'total_turnover': total_turnover,
            'weights_history': weights_history,
            'turnover_history': turnover_history,
            'regime_history': regime_history
        }
