"""
================================================================================
UNIT TESTS FOR PORTFOLIO OPTIMIZATION MODULES
================================================================================

These tests verify that each component works correctly.

Run tests with:
    python -m pytest tests/test_modules.py -v

Or run directly:
    python tests/test_modules.py

================================================================================
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import unittest

from portfolio_optimization.data import (
    generate_synthetic_stock_data,
    calculate_statistics,
    ensure_positive_definite
)
from portfolio_optimization.optimizer import ConvexPortfolioOptimizer
from portfolio_optimization.regimes import RegimeDetector
from portfolio_optimization.backtest import BacktestEngine


class TestDataModule(unittest.TestCase):
    """Tests for data generation and statistics."""
    
    def test_generate_data_shape(self):
        """Test that generated data has correct shape."""
        prices, returns = generate_synthetic_stock_data(
            n_assets=5, 
            n_days=100, 
            seed=42
        )
        
        self.assertEqual(prices.shape, (100, 5))
        self.assertEqual(returns.shape, (100, 5))
    
    def test_generate_data_no_nan(self):
        """Test that generated data has no NaN values."""
        prices, returns = generate_synthetic_stock_data(
            n_assets=5, 
            n_days=100, 
            seed=42
        )
        
        self.assertFalse(prices.isna().any().any())
        self.assertFalse(returns.isna().any().any())
    
    def test_prices_positive(self):
        """Test that all prices are positive."""
        prices, _ = generate_synthetic_stock_data(
            n_assets=5, 
            n_days=100, 
            seed=42
        )
        
        self.assertTrue((prices > 0).all().all())
    
    def test_ensure_positive_definite(self):
        """Test PSD projection works correctly."""
        # Create a matrix that's not quite PSD
        matrix = np.array([
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.9],
            [0.8, 0.9, 1.0]
        ])
        # Make it slightly non-PSD by adding noise
        matrix += np.random.randn(3, 3) * 0.01
        matrix = (matrix + matrix.T) / 2  # Symmetrize
        
        psd_matrix = ensure_positive_definite(matrix)
        
        # Check eigenvalues are all positive
        eigenvalues = np.linalg.eigvalsh(psd_matrix)
        self.assertTrue(np.all(eigenvalues > 0))
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        _, returns = generate_synthetic_stock_data(
            n_assets=5, 
            n_days=100, 
            seed=42
        )
        
        stats = calculate_statistics(returns, window=60)
        
        self.assertIn('expected_returns', stats)
        self.assertIn('covariance_matrix', stats)
        self.assertEqual(len(stats['expected_returns']), 5)
        self.assertEqual(stats['covariance_matrix'].shape, (5, 5))


class TestOptimizer(unittest.TestCase):
    """Tests for the convex portfolio optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_assets = 5
        self.optimizer = ConvexPortfolioOptimizer(
            n_assets=self.n_assets,
            risk_aversion=1.0,
            max_weight=0.40,
            min_weight=0.0,
            max_turnover=0.50,
            transaction_cost=0.001,
            max_risk=0.30
        )
        
        # Create simple test inputs
        self.expected_returns = np.array([0.10, 0.12, 0.08, 0.15, 0.09])
        
        # Create valid covariance matrix
        volatilities = np.array([0.20, 0.25, 0.15, 0.30, 0.18])
        corr = np.array([
            [1.0, 0.3, 0.2, 0.4, 0.1],
            [0.3, 1.0, 0.3, 0.2, 0.3],
            [0.2, 0.3, 1.0, 0.1, 0.2],
            [0.4, 0.2, 0.1, 1.0, 0.3],
            [0.1, 0.3, 0.2, 0.3, 1.0]
        ])
        self.covariance = corr * np.outer(volatilities, volatilities)
    
    def test_weights_sum_to_one(self):
        """Test that optimized weights sum to 1."""
        weights = self.optimizer.optimize(
            self.expected_returns,
            self.covariance
        )
        
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
    
    def test_weights_non_negative(self):
        """Test that weights are non-negative (no short selling)."""
        weights = self.optimizer.optimize(
            self.expected_returns,
            self.covariance
        )
        
        self.assertTrue(np.all(weights >= -1e-6))  # Allow tiny numerical error
    
    def test_weights_below_max(self):
        """Test that no weight exceeds maximum."""
        weights = self.optimizer.optimize(
            self.expected_returns,
            self.covariance
        )
        
        self.assertTrue(np.all(weights <= self.optimizer.max_weight + 1e-6))
    
    def test_with_current_weights(self):
        """Test optimization with starting weights."""
        current_weights = np.ones(self.n_assets) / self.n_assets
        
        weights = self.optimizer.optimize(
            self.expected_returns,
            self.covariance,
            current_weights=current_weights
        )
        
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
    
    def test_multi_period_returns_valid_weights(self):
        """Test multi-period optimization returns valid weights."""
        current_weights = np.ones(self.n_assets) / self.n_assets
        
        weights = self.optimizer.multi_period_optimize(
            [self.expected_returns] * 3,
            [self.covariance] * 3,
            current_weights,
            horizon=3
        )
        
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        self.assertTrue(np.all(weights >= -1e-6))


class TestRegimeDetector(unittest.TestCase):
    """Tests for regime detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = RegimeDetector(
            volatility_lookback=20,
            trend_lookback=60,
            volatility_threshold=0.20
        )
    
    def test_detect_regime_returns_dict(self):
        """Test that detect_regime returns a dictionary with expected keys."""
        _, returns = generate_synthetic_stock_data(
            n_assets=5, 
            n_days=100, 
            seed=42
        )
        
        regime = self.detector.detect_regime(returns)
        
        self.assertIn('regime', regime)
        self.assertIn('volatility', regime)
        self.assertIn('trend_return', regime)
        self.assertIn('recommended_risk_aversion', regime)
    
    def test_regime_is_valid(self):
        """Test that detected regime is one of the valid types."""
        _, returns = generate_synthetic_stock_data(
            n_assets=5, 
            n_days=100, 
            seed=42
        )
        
        regime = self.detector.detect_regime(returns)
        
        valid_regimes = ['LOW_VOL_BULL', 'HIGH_VOL_BULL', 'LOW_VOL_BEAR', 'HIGH_VOL_BEAR']
        self.assertIn(regime['regime'], valid_regimes)


class TestBacktestEngine(unittest.TestCase):
    """Tests for the backtesting engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        _, self.returns = generate_synthetic_stock_data(
            n_assets=5, 
            n_days=200,  # Enough for lookback + some trading
            seed=42
        )
        
        self.engine = BacktestEngine(
            returns=self.returns,
            initial_capital=1_000_000,
            rebalance_frequency=21,
            transaction_cost=0.001,
            lookback_window=60
        )
        
        self.optimizer = ConvexPortfolioOptimizer(
            n_assets=5,
            risk_aversion=1.0,
            max_weight=0.40,
            max_turnover=0.50,
            transaction_cost=0.001
        )
    
    def test_backtest_returns_dict(self):
        """Test that backtest returns a dictionary with expected keys."""
        results = self.engine.run_backtest(
            self.optimizer,
            regime_aware=True,
            use_multi_period=False
        )
        
        expected_keys = [
            'portfolio_values', 'daily_returns', 'total_return',
            'annualized_return', 'sharpe_ratio', 'max_drawdown'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
    
    def test_benchmark_runs(self):
        """Test that benchmark runs without errors."""
        results = self.engine.run_benchmark(strategy='equal_weight')
        
        self.assertIn('portfolio_values', results)
        self.assertIn('sharpe_ratio', results)
    
    def test_portfolio_values_positive(self):
        """Test that portfolio values are always positive."""
        results = self.engine.run_backtest(
            self.optimizer,
            regime_aware=False,
            use_multi_period=False
        )
        
        self.assertTrue(np.all(results['portfolio_values'] > 0))
    
    def test_turnover_reasonable(self):
        """Test that turnover is within reasonable bounds."""
        results = self.engine.run_backtest(
            self.optimizer,
            regime_aware=False,
            use_multi_period=False
        )
        
        if results['turnover_history']:
            max_turnover = max(results['turnover_history'])
            self.assertLessEqual(max_turnover, 2.0)  # Can't trade more than 200%


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline(self):
        """Test the complete optimization pipeline."""
        # Generate data
        prices, returns = generate_synthetic_stock_data(
            n_assets=5, 
            n_days=150, 
            seed=42
        )
        
        # Create optimizer
        optimizer = ConvexPortfolioOptimizer(
            n_assets=5,
            risk_aversion=1.0,
            max_weight=0.40,
            max_turnover=0.50
        )
        
        # Run backtest
        engine = BacktestEngine(
            returns=returns,
            initial_capital=1_000_000,
            rebalance_frequency=21,
            lookback_window=60
        )
        
        opt_results = engine.run_backtest(optimizer)
        bench_results = engine.run_benchmark()
        
        # Verify both ran successfully
        self.assertIsNotNone(opt_results['sharpe_ratio'])
        self.assertIsNotNone(bench_results['sharpe_ratio'])
        
        # Verify we have reasonable metrics
        self.assertFalse(np.isnan(opt_results['sharpe_ratio']))
        self.assertFalse(np.isnan(bench_results['sharpe_ratio']))


def run_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 60)
    print("RUNNING PORTFOLIO OPTIMIZATION TESTS")
    print("=" * 60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataModule))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestRegimeDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
    else:
        print(f" {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
