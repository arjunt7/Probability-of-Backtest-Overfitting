"""
================================================================================
DATA GENERATION AND PREPARATION MODULE
================================================================================

This module handles:
1. Synthetic stock data generation for backtesting
2. Statistical calculations for portfolio optimization
3. Helper functions for data preprocessing

In production, you would replace generate_synthetic_stock_data with
real data loading from APIs (Yahoo Finance, Bloomberg, etc.)
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


def generate_synthetic_stock_data(
    n_assets: int = 10,
    n_days: int = 1260,  # ~5 years of trading days
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic stock price data for backtesting.
    
    In real-world applications, you would use actual historical data.
    Here we simulate realistic stock behavior including:
    - Different expected returns for each asset
    - Correlations between assets (stocks move together)
    - Varying volatilities
    - Fat tails (more extreme moves than normal distribution)
    
    Parameters:
    -----------
    n_assets : int
        Number of assets in our investment universe
    n_days : int
        Number of trading days to simulate (252 = 1 year)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    prices : pd.DataFrame
        Daily price data for each asset
    returns : pd.DataFrame
        Daily return data for each asset
    
    Example:
    --------
    >>> prices, returns = generate_synthetic_stock_data(n_assets=5, n_days=252)
    >>> print(prices.shape)  # (252, 5)
    """
    np.random.seed(seed)
    
    # Create realistic asset names (like ticker symbols)
    asset_names = [f'ASSET_{i+1}' for i in range(n_assets)]
    
    # Generate date range (trading days only, skip weekends)
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    
    
    # Step 1: Create a correlation matrix for our assets
    
    # In real markets, stocks are correlated (they tend to move together)
    # For example, all tech stocks might rise or fall together
    
    # Generate random matrix and make it symmetric positive definite
    # This mathematical trick ensures we get a valid correlation matrix
    random_matrix = np.random.randn(n_assets, n_assets)
    correlation_matrix = np.dot(random_matrix, random_matrix.T)
    
    # Normalize to get correlation matrix (diagonal elements = 1)
    d = np.sqrt(np.diag(correlation_matrix))
    correlation_matrix = correlation_matrix / np.outer(d, d)
    
    
    # Step 2: Set expected returns and volatilities for each asset
    
    # Annual expected returns: range from 5% to 20%
    # These represent how much we expect each asset to grow per year
    # Higher return usually means higher risk
    annual_returns = np.random.uniform(0.05, 0.20, n_assets)
    
    # Annual volatilities: range from 15% to 40%
    # These represent how much the asset price fluctuates
    # Volatility = standard deviation of returns
    annual_volatilities = np.random.uniform(0.15, 0.40, n_assets)
    
    # Convert to daily values
    # Daily return = Annual return / 252 trading days
    # Daily volatility = Annual volatility / sqrt(252) [volatility scales with sqrt of time]
    daily_returns_mean = annual_returns / 252
    daily_volatilities = annual_volatilities / np.sqrt(252)
    
    
    # Step 3: Create covariance matrix from correlations and volatilities
    
    # Covariance measures how two assets move together
    # Covariance(i,j) = Correlation(i,j) * Volatility(i) * Volatility(j)
    covariance_matrix = correlation_matrix * np.outer(daily_volatilities, daily_volatilities)
    
    
    # Step 4: Generate correlated returns using Cholesky decomposition
    
    # Cholesky decomposition: breaks a matrix into L * L^T
    # This allows us to transform independent random numbers into correlated ones
    cholesky = np.linalg.cholesky(covariance_matrix)
    
    # Generate independent random returns (each column is independent)
    independent_returns = np.random.randn(n_days, n_assets)
    
    # Transform to correlated returns by multiplying with Cholesky matrix
    correlated_returns = independent_returns @ cholesky.T
    
    # Add the expected return (drift) to each day
    # This shifts returns up/down based on expected growth
    correlated_returns += daily_returns_mean
    
    
    # Step 5: Add fat tails (realistic extreme events)
    
    # Real markets have more extreme moves than normal distribution predicts
    # Think of flash crashes or massive rallies - they happen more often than
    # a normal distribution would suggest
    for i in range(n_assets):
        # 2% chance of extreme move on any day
        extreme_days = np.random.random(n_days) < 0.02
        # Extreme moves are 3-5x normal volatility
        extreme_magnitude = np.random.uniform(3, 5) * daily_volatilities[i]
        # Random direction (up or down)
        extreme_direction = np.random.choice([-1, 1], n_days)
        correlated_returns[extreme_days, i] += extreme_magnitude * extreme_direction[extreme_days]
    
    
    # Step 6: Convert returns to prices
    
    # Start all assets at price = 100 (arbitrary starting point)
    initial_prices = np.ones(n_assets) * 100
    
    # Cumulative returns give us price levels
    # Price_t = Price_0 * (1 + r_1) * (1 + r_2) * ... * (1 + r_t)
    # This is compound growth
    cumulative_returns = np.cumprod(1 + correlated_returns, axis=0)
    prices = initial_prices * cumulative_returns
    
    # Create DataFrames with proper date index
    prices_df = pd.DataFrame(prices, index=dates, columns=asset_names)
    returns_df = pd.DataFrame(correlated_returns, index=dates, columns=asset_names)
    
    print(f"Generated {n_days} days of data for {n_assets} assets")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    
    return prices_df, returns_df


def ensure_positive_definite(matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
    """
    Ensure a covariance matrix is positive semi-definite (PSD).
    
    Why is this important?
    - Covariance matrices must be PSD for mathematical validity
    - Sample covariance from limited data can be slightly non-PSD due to numerical issues
    - CVXPY's quad_form will fail if the matrix is not PSD
    
    Method: Project to nearest PSD matrix using eigenvalue decomposition
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input covariance matrix (may be slightly non-PSD)
    min_eigenvalue : float
        Minimum allowed eigenvalue (adds small "jitter" to diagonal)
    
    Returns:
    --------
    psd_matrix : np.ndarray
        Positive semi-definite version of input matrix
    """
    # Make symmetric (average of matrix and its transpose)
    # This handles any small asymmetries from numerical precision
    symmetric = (matrix + matrix.T) / 2
    
    # Eigenvalue decomposition: matrix = V * D * V^T
    # D is diagonal with eigenvalues, V is matrix of eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    
    # Replace any negative eigenvalues with small positive value
    # This is what makes the matrix PSD
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct the matrix with fixed eigenvalues
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Ensure exact symmetry
    psd_matrix = (psd_matrix + psd_matrix.T) / 2
    
    return psd_matrix


def calculate_statistics(
    returns: pd.DataFrame,
    window: int = 60,
    ensure_psd: bool = True
) -> Dict:
    """
    Calculate rolling statistics for portfolio optimization.
    
    We use rolling windows because:
    1. Markets change over time (non-stationary)
    2. Recent data is more relevant than old data
    3. It allows our model to adapt to changing conditions
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Daily returns for each asset
    window : int
        Number of days to look back for calculations (60 days ≈ 3 months)
    ensure_psd : bool
        Whether to project covariance matrix to be positive definite
    
    Returns:
    --------
    stats : dict
        Dictionary containing:
        - expected_returns: Expected return for each asset (annualized)
        - covariance_matrix: Covariance between assets (annualized)
        - asset_names: List of asset names
    
    Example:
    --------
    >>> stats = calculate_statistics(returns, window=60)
    >>> print(stats['expected_returns'])  # Array of expected returns
    """
    # Calculate expected returns (mean of past returns)
    # We annualize by multiplying by 252 trading days
    # This converts daily returns to yearly returns
    expected_returns = returns.iloc[-window:].mean() * 252
    
    # Calculate covariance matrix
    # We annualize by multiplying by 252
    # Covariance tells us how assets move together
    covariance_matrix = returns.iloc[-window:].cov() * 252
    
    # Convert to numpy array
    cov_array = covariance_matrix.values
    
    # Ensure the covariance matrix is positive semi-definite
    # This prevents numerical issues in optimization
    if ensure_psd:
        cov_array = ensure_positive_definite(cov_array)
    
    return {
        'expected_returns': expected_returns.values,
        'covariance_matrix': cov_array,
        'asset_names': returns.columns.tolist()
    }
