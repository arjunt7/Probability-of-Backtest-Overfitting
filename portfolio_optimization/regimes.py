"""
================================================================================
REGIME DETECTION MODULE
================================================================================

This module detects market regimes (conditions) and provides recommended
parameter adjustments for the portfolio optimizer.

Market Regimes:
---------------
1. LOW_VOL_BULL  - Calm uptrend: Best for risk-taking, markets are stable and rising
2. HIGH_VOL_BULL - Volatile uptrend: Markets rising but with big swings
3. LOW_VOL_BEAR  - Calm downtrend: Markets declining steadily
4. HIGH_VOL_BEAR - Crisis mode: Markets crashing with high uncertainty

Why Regime Detection Matters:
-----------------------------
- A strategy that works in calm markets may fail during a crisis
- Risk tolerance should adapt to market conditions
- Transaction costs matter more when markets are choppy
- Different assets perform better in different regimes

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict


class RegimeDetector:
    """
    Detect market regimes and provide regime-appropriate parameters.
    
    This class uses simple but effective indicators:
    1. Volatility: Is the market calm or turbulent?
    2. Trend: Is the market going up or down?
    
    Combining these gives us 4 regime classifications:
    
    +------------------+------------------+
    |  LOW_VOL_BEAR    |  HIGH_VOL_BEAR   |
    |  (Calm decline)  |  (Crisis!)       |
    +------------------+------------------+
    |  LOW_VOL_BULL    |  HIGH_VOL_BULL   |
    |  (Goldilocks)    |  (Risky rally)   |
    +------------------+------------------+
         Low Vol            High Vol
    
    Example:
    --------
    >>> detector = RegimeDetector(volatility_threshold=0.20)
    >>> regime_info = detector.detect_regime(returns_df)
    >>> print(regime_info['regime'])  # 'LOW_VOL_BULL'
    >>> print(regime_info['recommended_risk_aversion'])  # 0.5
    """
    
    def __init__(
        self,
        volatility_lookback: int = 20,
        trend_lookback: int = 60,
        volatility_threshold: float = 0.20
    ):
        """
        Initialize regime detector.
        
        Parameters:
        -----------
        volatility_lookback : int
            Days to look back for volatility calculation (default: 20 = 1 month)
            Shorter = more responsive but noisier
            Longer = smoother but slower to react
        
        trend_lookback : int
            Days to look back for trend detection (default: 60 = 3 months)
            We use longer period for trends to avoid false signals
        
        volatility_threshold : float
            Annualized volatility level separating "low" from "high"
            Default: 0.20 = 20% annual volatility
            - S&P 500 average is ~15-20%
            - During crises can spike to 40-80%
        """
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        self.volatility_threshold = volatility_threshold
    
    def detect_regime(self, returns: pd.DataFrame) -> Dict:
        """
        Detect the current market regime based on recent returns.
        
        Process:
        1. Calculate portfolio returns (equal-weighted average)
        2. Measure recent volatility
        3. Determine trend direction
        4. Classify regime and recommend parameters
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Historical returns data (rows = dates, columns = assets)
            Needs at least max(volatility_lookback, trend_lookback) rows
        
        Returns:
        --------
        regime_info : dict
            Dictionary containing:
            - regime: String identifier (e.g., 'LOW_VOL_BULL')
            - volatility: Current annualized volatility
            - trend_return: Cumulative return over trend period
            - is_high_volatility: Boolean
            - is_uptrend: Boolean
            - recommended_risk_aversion: Suggested γ parameter
            - recommended_max_risk: Suggested max volatility cap
            - description: Human-readable description
        """
        # Calculate average portfolio return (equal-weighted)
        # This gives us a simple "market" indicator
        portfolio_returns = returns.mean(axis=1)
        
        # =====================================================================
        # Step 1: Calculate current volatility
        # =====================================================================
        # Volatility = standard deviation of returns
        # Annualize by multiplying by √252 (trading days per year)
        recent_returns = portfolio_returns.iloc[-self.volatility_lookback:]
        recent_vol = recent_returns.std() * np.sqrt(252)
        
        # =====================================================================
        # Step 2: Detect trend direction
        # =====================================================================
        # Calculate cumulative return over lookback period
        # Positive = uptrend, Negative = downtrend
        trend_returns = portfolio_returns.iloc[-self.trend_lookback:]
        cumulative_return = (1 + trend_returns).prod() - 1
        is_uptrend = cumulative_return > 0
        
        # =====================================================================
        # Step 3: Classify regime
        # =====================================================================
        is_high_vol = recent_vol > self.volatility_threshold
        
        if is_uptrend and not is_high_vol:
            # BEST CASE: Markets rising smoothly
            regime = "LOW_VOL_BULL"
            recommended_risk_aversion = 0.5    # Be aggressive
            recommended_max_risk = 0.30        # Allow more risk
            description = "Calm uptrend - favorable for risk-taking"
            
        elif is_uptrend and is_high_vol:
            # Markets rising but with big swings - stay alert
            regime = "HIGH_VOL_BULL"
            recommended_risk_aversion = 1.0    # Balanced
            recommended_max_risk = 0.25        # Moderate risk
            description = "Volatile uptrend - moderate caution advised"
            
        elif not is_uptrend and not is_high_vol:
            # Markets declining steadily - defensive
            regime = "LOW_VOL_BEAR"
            recommended_risk_aversion = 2.0    # Conservative
            recommended_max_risk = 0.20        # Reduce risk
            description = "Calm downtrend - defensive positioning recommended"
            
        else:  # not is_uptrend and is_high_vol
            # WORST CASE: Crisis - markets crashing with high volatility
            regime = "HIGH_VOL_BEAR"
            recommended_risk_aversion = 5.0    # Very defensive
            recommended_max_risk = 0.15        # Minimize risk
            description = "Crisis mode - maximum risk reduction"
        
        return {
            'regime': regime,
            'volatility': recent_vol,
            'trend_return': cumulative_return,
            'is_high_volatility': is_high_vol,
            'is_uptrend': is_uptrend,
            'recommended_risk_aversion': recommended_risk_aversion,
            'recommended_max_risk': recommended_max_risk,
            'description': description
        }
    
    def get_regime_summary(self, regime_history: list) -> Dict:
        """
        Summarize regime statistics from backtest history.
        
        Parameters:
        -----------
        regime_history : list
            List of regime_info dictionaries from backtest
        
        Returns:
        --------
        summary : dict
            Statistics about time spent in each regime
        """
        if not regime_history:
            return {}
        
        regimes = [r['regime'] for r in regime_history]
        unique_regimes = set(regimes)
        
        summary = {}
        for regime in unique_regimes:
            count = regimes.count(regime)
            pct = count / len(regimes) * 100
            summary[regime] = {
                'count': count,
                'percentage': pct
            }
        
        return summary
