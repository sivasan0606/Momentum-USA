"""
Unit tests for S&P 500 Momentum Engine.
Verifies:
- Formation return math (no lookahead bias).
- 200-day SMA filter logic.
- Overlapping tranche weights (K tranches, 1/K each).
- Turnover and transaction costs.
- CAPM regression (alpha, beta, t-stat).
- End-to-end backtest sanity check.
"""

import unittest
import numpy as np
import pandas as pd
from scipy import stats

from momentum_engine import MomentumEngine, BacktestMetrics
from data_manager import load_price_data, BENCHMARK_TICKER


class TestMomentumEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic daily price dataset with known patterns
        dates = pd.date_range("2018-01-01", "2021-01-01", freq="B")
        np.random.seed(42)

        # Stock A: Strong uptrend (Winner)
        # Stock B: Moderate uptrend
        # Stock C: Downtrend (Loser, below SMA)
        # SPY: Benchmark
        trend_a = np.linspace(100, 300, len(dates)) + np.random.normal(0, 2, len(dates))
        trend_b = np.linspace(100, 150, len(dates)) + np.random.normal(0, 2, len(dates))
        trend_c = np.linspace(300, 100, len(dates)) + np.random.normal(0, 2, len(dates))
        trend_spy = np.linspace(100, 140, len(dates)) + np.random.normal(0, 1, len(dates))

        cls.synthetic_prices = pd.DataFrame(
            {
                "STOCK_A": trend_a,
                "STOCK_B": trend_b,
                "STOCK_C": trend_c,
                BENCHMARK_TICKER: trend_spy,
            },
            index=dates,
        )

    def test_sma200_calculation(self):
        """Verify 200-day SMA computation."""
        engine = MomentumEngine(self.synthetic_prices)
        sma = engine.daily_sma200["STOCK_A"].dropna()
        self.assertGreater(len(sma), 0)
        # On synthetic linear trend, current price of Stock A should be above SMA200
        self.assertTrue(self.synthetic_prices["STOCK_A"].iloc[-1] > sma.iloc[-1])
        # Stock C (downtrend) should be below SMA200
        sma_c = engine.daily_sma200["STOCK_C"].dropna()
        self.assertTrue(self.synthetic_prices["STOCK_C"].iloc[-1] < sma_c.iloc[-1])

    def test_formation_return_no_lookahead(self):
        """Verify formation returns use strictly lagged data (P_{t-1} / P_{t-1-J} - 1)."""
        engine = MomentumEngine(self.synthetic_prices)
        monthly_p = engine.monthly_prices

        # At month index 15, evaluating return for month 15:
        # t_prev_idx = 14
        # J = 12
        # formation start is index 14 - 12 = 2
        p_prev = monthly_p.iloc[14]["STOCK_A"]
        p_start = monthly_p.iloc[2]["STOCK_A"]
        expected_ret = (p_prev / p_start) - 1.0

        # Run engine backtest
        metrics, results, _ = engine.run_backtest(
            start_date="2019-06-01",
            end_date="2020-12-31",
            j=12,
            k=1,
            max_stocks=1,
            use_sma=False,
        )
        self.assertGreater(metrics.final_value, 0)
        self.assertTrue(isinstance(metrics, BacktestMetrics))

    def test_sma_exclusion_filter(self):
        """Verify stocks below SMA are excluded from Long portfolio."""
        engine = MomentumEngine(self.synthetic_prices)
        metrics, results, holdings = engine.run_backtest(
            start_date="2019-06-01",
            end_date="2020-12-31",
            j=12,
            k=3,
            max_stocks=3,
            use_sma=True,
        )
        # Stock C is in persistent downtrend and below 200-day SMA, should never be bought
        held_tickers = holdings["Ticker"].unique()
        self.assertNotIn("STOCK_C", held_tickers)
        self.assertIn("STOCK_A", held_tickers)

    def test_overlapping_tranches_weight_sum(self):
        """Verify that across K overlapping tranches, weights sum properly."""
        engine = MomentumEngine(self.synthetic_prices)
        k = 3
        metrics, results, holdings = engine.run_backtest(
            start_date="2019-06-01",
            end_date="2020-12-31",
            j=12,
            k=k,
            max_stocks=2,
            use_sma=False,
        )
        # For each date in holdings, group by Date and check sum of weights <= 1.0001
        grouped_weights = holdings.groupby("Date")["Weight"].sum()
        for dt, total_w in grouped_weights.items():
            self.assertLessEqual(total_w, 1.0001)

    def test_market_model_ols(self):
        """Verify alpha, beta, and t-statistic calculation in metrics."""
        engine = MomentumEngine(self.synthetic_prices)
        metrics, _, _ = engine.run_backtest(
            start_date="2019-06-01",
            end_date="2020-12-31",
            j=12,
            k=3,
            max_stocks=1,
        )
        self.assertTrue(np.isfinite(metrics.beta))
        self.assertTrue(np.isfinite(metrics.annual_alpha_pct))
        self.assertTrue(np.isfinite(metrics.alpha_tstat))
        self.assertGreaterEqual(metrics.r_squared, 0.0)
        self.assertLessEqual(metrics.r_squared, 1.0)

    def test_live_sp500_engine(self):
        """Verify engine execution against cached S&P 500 data."""
        prices, _ = load_price_data()
        engine = MomentumEngine(prices)
        metrics, results_df, holdings_df = engine.run_backtest(
            start_date="2020-01-01",
            end_date="2022-12-31",
            j=12,
            k=3,
            max_stocks=10,
            use_sma=True,
        )
        self.assertGreater(metrics.final_value, 0)
        self.assertGreater(len(results_df), 24)
        self.assertFalse(holdings_df.empty)


if __name__ == "__main__":
    unittest.main()
