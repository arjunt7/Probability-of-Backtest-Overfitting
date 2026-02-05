"""
================================================================================
CONVEX PORTFOLIO OPTIMIZER MODULE
================================================================================

This module contains the core portfolio optimization logic using CVXPY.

Key Features:
1. Single-period mean-variance optimization
2. Multi-period optimization with look-ahead planning
3. Robust numerical handling (PSD projection, solver fallbacks)
4. Soft constraint handling to avoid infeasibility

Convex Optimization Benefits:
- Guaranteed global optimum (no local minima)
- Efficient polynomial-time algorithms
- Rich theory for constraint handling
================================================================================
"""

import numpy as np
import cvxpy as cp
from typing import List, Optional
import warnings


def ensure_positive_definite(matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
    """
    Project a matrix to the nearest positive semi-definite (PSD) matrix.
    
    This is critical for numerical stability in optimization:
    - Sample covariance matrices can be slightly non-PSD due to numerical precision
    - CVXPY's quad_form requires PSD matrices
    - Adding small diagonal "jitter" also helps solver convergence
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input covariance matrix
    min_eigenvalue : float
        Minimum eigenvalue to enforce (adds robustness)
    
    Returns:
    --------
    psd_matrix : np.ndarray
        Positive semi-definite version of input
    """
    # Make symmetric first
    symmetric = (matrix + matrix.T) / 2
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    
    # Clip negative eigenvalues to small positive value
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct matrix
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Ensure perfect symmetry
    return (psd_matrix + psd_matrix.T) / 2


class ConvexPortfolioOptimizer:
    """
    A portfolio optimizer using convex optimization with CVXPY.
    
    Why Convex Optimization for Portfolios?
    ----------------------------------------
    1. Guaranteed global optimum - no local minima traps
    2. Efficient algorithms exist (interior point, ADMM)
    3. Many realistic constraints are naturally convex
    4. Theory is well-developed with convergence guarantees
    
    The Optimization Problem:
    -------------------------
    maximize:   E[Return] - γ × Risk - τ × Transaction Costs
    subject to: 
        - Sum of weights = 1 (fully invested)
        - Weights in [min_weight, max_weight]
        - Turnover ≤ max_turnover
        - Portfolio volatility ≤ max_risk
    
    Where:
        γ (gamma) = risk aversion parameter
        τ (tau) = transaction cost rate
    
    Example:
    --------
    >>> optimizer = ConvexPortfolioOptimizer(n_assets=10, risk_aversion=1.0)
    >>> weights = optimizer.optimize(expected_returns, covariance_matrix)
    >>> print(weights.sum())  # Should be 1.0
    """
    
    # List of solvers to try in order of preference
    # OSQP is fast for QPs, ECOS is robust, SCS is general-purpose
    SOLVER_PRIORITY = [cp.OSQP, cp.ECOS, cp.SCS, cp.CLARABEL]
    
    def __init__(
        self,
        n_assets: int,
        risk_aversion: float = 1.0,
        max_weight: float = 0.30,
        min_weight: float = 0.0,
        max_turnover: float = 0.50,
        transaction_cost: float = 0.001,
        max_risk: float = 0.25,
        soft_risk_penalty: float = 10.0
    ):
        """
        Initialize the portfolio optimizer.
        
        Parameters:
        -----------
        n_assets : int
            Number of assets to optimize over
        risk_aversion : float
            How much we penalize risk (higher = more conservative)
            - γ = 0: Only care about returns (very risky)
            - γ = 1: Balanced approach
            - γ = 5+: Very risk-averse
        max_weight : float
            Maximum weight for any single asset (diversification constraint)
            Example: 0.30 means no more than 30% in one stock
        min_weight : float
            Minimum weight for any asset
            - 0.0: Can exclude assets entirely
            - 0.02: Must hold at least 2% in each
        max_turnover : float
            Maximum allowed turnover per rebalancing period
            Turnover = sum of |new_weight - old_weight| for all assets
            Example: 0.50 means at most 50% of portfolio changes
        transaction_cost : float
            Cost per unit traded (e.g., 0.001 = 0.1% = 10 basis points)
            Includes brokerage fees, bid-ask spread, market impact
        max_risk : float
            Maximum allowed portfolio volatility (annualized)
            Example: 0.25 = 25% annual volatility cap
        soft_risk_penalty : float
            Penalty for exceeding risk limit (soft constraint fallback)
            If hard constraint fails, we use this to penalize excess risk
        """
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_turnover = max_turnover
        self.transaction_cost = transaction_cost
        self.max_risk = max_risk
        self.soft_risk_penalty = soft_risk_penalty
    
    def _try_solve(self, problem: cp.Problem, verbose: bool = False) -> bool:
        """
        Try to solve the optimization problem with multiple solvers.
        
        We try solvers in order of preference because:
        - OSQP: Very fast for quadratic programs
        - ECOS: Robust for second-order cone programs
        - SCS: General-purpose, handles more problem types
        - CLARABEL: Good for QP and SDP
        
        Returns True if problem was solved optimally.
        """
        for solver in self.SOLVER_PRIORITY:
            try:
                problem.solve(solver=solver, verbose=verbose)
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    return True
            except (cp.error.SolverError, Exception):
                continue
        return False
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        current_weights: np.ndarray = None,
        use_soft_constraints: bool = True
    ) -> np.ndarray:
        """
        Solve the convex optimization problem to find optimal portfolio weights.
        
        The Math Behind It:
        -------------------
        We solve:
            maximize:   μᵀw - γ·wᵀΣw - τ·‖w - w_current‖₁
            subject to: 1ᵀw = 1
                        w_min ≤ w ≤ w_max
                        ‖w - w_current‖₁ ≤ turnover_max
                        √(wᵀΣw) ≤ risk_max
        
        Where:
            w = portfolio weights (decision variable)
            μ = expected returns vector
            Σ = covariance matrix
            γ = risk aversion
            τ = transaction cost
        
        Parameters:
        -----------
        expected_returns : np.ndarray
            Expected return for each asset (shape: n_assets)
        covariance_matrix : np.ndarray
            Covariance matrix of returns (shape: n_assets × n_assets)
        current_weights : np.ndarray
            Current portfolio weights (for transaction costs)
        use_soft_constraints : bool
            If True, use soft constraints as fallback when hard constraints fail
            This prevents the optimizer from giving up
        
        Returns:
        --------
        optimal_weights : np.ndarray
            Optimal portfolio allocation (sums to 1)
        """
        # If no current weights provided, assume equal weight starting point
        if current_weights is None:
            current_weights = np.ones(self.n_assets) / self.n_assets
        
        # Ensure covariance matrix is positive definite
        # This is CRITICAL for numerical stability
        cov_psd = ensure_positive_definite(covariance_matrix)
        
        # =====================================================================
        # Define the optimization variable
        # =====================================================================
        # w is what we're solving for - the optimal weight for each asset
        w = cp.Variable(self.n_assets)
        
        # =====================================================================
        # Define the objective function components
        # =====================================================================
        
        # Term 1: Expected portfolio return (MAXIMIZE this)
        # Portfolio return = weighted average of individual returns
        expected_portfolio_return = expected_returns @ w
        
        # Term 2: Portfolio risk/variance (MINIMIZE this)
        # Portfolio variance = wᵀΣw (quadratic form)
        # This captures how portfolio moves as assets fluctuate together
        portfolio_variance = cp.quad_form(w, cov_psd)
        
        # Term 3: Transaction costs (MINIMIZE this)
        # Turnover = sum of absolute weight changes
        # We use L1 norm: ‖w - w_current‖₁
        turnover = cp.norm(w - current_weights, 1)
        transaction_cost_penalty = self.transaction_cost * turnover
        
        # Combine into objective: Return - Risk - Costs
        objective = (
            expected_portfolio_return 
            - self.risk_aversion * portfolio_variance 
            - transaction_cost_penalty
        )
        
        # =====================================================================
        # Define hard constraints
        # =====================================================================
        constraints = []
        
        # Constraint 1: Fully invested (weights sum to 1)
        # No cash position - all money is in the market
        constraints.append(cp.sum(w) == 1)
        
        # Constraint 2: Long-only (no short selling)
        # Can be relaxed for long-short strategies
        constraints.append(w >= self.min_weight)
        
        # Constraint 3: Diversification limit
        # Prevents putting too much in any single asset
        constraints.append(w <= self.max_weight)
        
        # Constraint 4: Turnover limit
        # Controls trading activity to manage transaction costs
        constraints.append(turnover <= self.max_turnover)
        
        # Constraint 5: Risk limit
        # Caps portfolio volatility at acceptable level
        # We use variance ≤ max_risk² (equivalent to std ≤ max_risk)
        constraints.append(cp.quad_form(w, cov_psd) <= self.max_risk ** 2)
        
        # =====================================================================
        # Solve with hard constraints first
        # =====================================================================
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        if self._try_solve(problem):
            # Success! Clean up and return
            return self._clean_weights(w.value, current_weights)
        
        # =====================================================================
        # Fallback: Try with relaxed turnover constraint
        # =====================================================================
        if use_soft_constraints:
            # Remove turnover constraint, let objective handle it
            relaxed_constraints = [c for c in constraints if c is not constraints[3]]
            relaxed_constraints.append(turnover <= self.max_turnover * 1.5)  # 50% slack
            
            problem = cp.Problem(cp.Maximize(objective), relaxed_constraints)
            if self._try_solve(problem):
                return self._clean_weights(w.value, current_weights)
            
            # =====================================================================
            # Last resort: Remove risk constraint entirely, rely on objective
            # =====================================================================
            # Add heavy penalty for exceeding risk budget
            excess_risk = cp.pos(portfolio_variance - self.max_risk ** 2)
            soft_objective = objective - self.soft_risk_penalty * excess_risk
            
            basic_constraints = [
                cp.sum(w) == 1,
                w >= self.min_weight,
                w <= self.max_weight
            ]
            
            problem = cp.Problem(cp.Maximize(soft_objective), basic_constraints)
            if self._try_solve(problem):
                return self._clean_weights(w.value, current_weights)
        
        # All attempts failed - return current weights (no trade)
        warnings.warn("Optimization failed - keeping current weights")
        return current_weights
    
    def _clean_weights(
        self,
        weights: np.ndarray,
        fallback: np.ndarray
    ) -> np.ndarray:
        """
        Clean up optimized weights for numerical stability.
        
        - Replace NaN with fallback
        - Remove negative weights (numerical noise)
        - Renormalize to sum to exactly 1
        """
        if weights is None or np.any(np.isnan(weights)):
            return fallback
        
        # Floor at zero (remove tiny negative values from numerical error)
        cleaned = np.maximum(weights, 0)
        
        # Renormalize to ensure sum = 1
        if cleaned.sum() > 0:
            cleaned = cleaned / cleaned.sum()
        else:
            return fallback
        
        return cleaned
    
    def multi_period_optimize(
        self,
        expected_returns_path: List[np.ndarray],
        covariance_matrices: List[np.ndarray],
        current_weights: np.ndarray,
        horizon: int = 3
    ) -> np.ndarray:
        """
        Multi-period optimization with look-ahead planning.
        
        Why Multi-Period?
        -----------------
        Single-period optimization is myopic - it only considers today.
        Multi-period optimization considers future trading costs and conditions.
        
        Example: If we expect volatility to spike next month, we might trade
        into safer assets NOW rather than waiting and paying higher costs later.
        
        The Approach:
        -------------
        1. Create weight variables for each future period
        2. Sum discounted objectives across all periods
        3. Chain constraints (period t starts where t-1 ends)
        4. Solve jointly and return only period-1 weights
        
        This is called "receding horizon control" or "model predictive control"
        
        Parameters:
        -----------
        expected_returns_path : List[np.ndarray]
            Expected returns for each future period
        covariance_matrices : List[np.ndarray]
            Covariance matrices for each future period
        current_weights : np.ndarray
            Starting portfolio weights
        horizon : int
            Number of periods to look ahead
        
        Returns:
        --------
        optimal_weights : np.ndarray
            Optimal weights for the FIRST period only
            (We'll re-optimize next period with new information)
        """
        # Limit horizon to available data
        horizon = min(horizon, len(expected_returns_path))
        
        # If only one period, fall back to single-period optimization
        if horizon <= 1:
            return self.optimize(
                expected_returns_path[0],
                covariance_matrices[0],
                current_weights
            )
        
        # =====================================================================
        # Setup: Create weight variable for each period
        # =====================================================================
        weights = [cp.Variable(self.n_assets) for _ in range(horizon)]
        
        # Discount factor: we care more about near-term than far-term
        # 0.95 per period means 3 periods ahead is discounted to 0.95³ ≈ 0.86
        discount_factor = 0.95
        
        total_objective = 0
        constraints = []
        
        # =====================================================================
        # Build multi-period objective and constraints
        # =====================================================================
        for t in range(horizon):
            # Reference point: where did we start this period?
            prev_weights = current_weights if t == 0 else weights[t - 1]
            
            # Ensure covariance is PSD
            cov_psd = ensure_positive_definite(covariance_matrices[t])
            
            # Period t's objective components
            period_return = expected_returns_path[t] @ weights[t]
            period_risk = cp.quad_form(weights[t], cov_psd)
            turnover = cp.norm(weights[t] - prev_weights, 1)
            transaction_cost = self.transaction_cost * turnover
            
            # Discount and add to total objective
            discount = discount_factor ** t
            period_objective = discount * (
                period_return 
                - self.risk_aversion * period_risk 
                - transaction_cost
            )
            total_objective += period_objective
            
            # Add constraints for this period
            constraints.extend([
                cp.sum(weights[t]) == 1,           # Fully invested
                weights[t] >= self.min_weight,     # No shorts
                weights[t] <= self.max_weight,     # Diversification
                turnover <= self.max_turnover,     # Turnover limit
                period_risk <= self.max_risk ** 2  # Risk limit
            ])
        
        # =====================================================================
        # Solve the multi-period problem
        # =====================================================================
        problem = cp.Problem(cp.Maximize(total_objective), constraints)
        
        if self._try_solve(problem):
            return self._clean_weights(weights[0].value, current_weights)
        
        # Fallback to single-period if multi-period fails
        warnings.warn("Multi-period optimization failed, falling back to single-period")
        return self.optimize(
            expected_returns_path[0],
            covariance_matrices[0],
            current_weights
        )
