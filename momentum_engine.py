"""
Jegadeesh & Titman (1993) Momentum Engine for S&P 500.

Implements:
- Model A: J-month formation / K-month holding decile strategy (with overlapping monthly tranches).
- Model B: Weighted Relative Strength Strategy (WRSS).
- 200-day SMA trend filter (no lookahead bias).
- Concentration caps (default top 10 stocks).
- Transaction cost deduction on turnover.
- Market-model regression against S&P 500 (SPY) for Jensen's Alpha, Beta, t-stat, R^2.
- Full performance analytics: CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar.
- Year-by-Year returns breakdown.
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from data_manager import BENCHMARK_TICKER, load_price_data

OUTPUT_DIR = Path(__file__).parent / "output"


@dataclass
class BacktestMetrics:
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    cagr_pct: float
    annual_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_pct: float
    avg_turnover_pct: float
    avg_holdings: float
    # Benchmark stats
    benchmark_cagr_pct: float
    benchmark_volatility_pct: float
    benchmark_sharpe: float
    benchmark_max_drawdown_pct: float
    # Market Model OLS stats
    annual_alpha_pct: float
    beta: float
    alpha_tstat: float
    r_squared: float


class MomentumEngine:
    def __init__(
        self,
        prices_df: pd.DataFrame,
        benchmark_ticker: str = BENCHMARK_TICKER,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
    ):
        """
        prices_df: Daily adjusted close prices with tickers as columns, DatetimeIndex.
        """
        self.raw_prices = prices_df.copy()
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate

        if benchmark_ticker not in self.raw_prices.columns:
            raise ValueError(f"Benchmark ticker '{benchmark_ticker}' not found in prices DataFrame.")

        # Compute daily 200-day SMA
        self.daily_sma200 = self.raw_prices.rolling(window=200, min_periods=150).mean()

        # Resample to calendar month-ends (using ME or M for pandas compatibility)
        try:
            self.monthly_prices = self.raw_prices.resample("ME").last()
            self.monthly_sma200 = self.daily_sma200.resample("ME").last()
        except ValueError:
            self.monthly_prices = self.raw_prices.resample("M").last()
            self.monthly_sma200 = self.daily_sma200.resample("M").last()

        # Monthly returns: R_t = P_t / P_{t-1} - 1
        self.monthly_returns = self.monthly_prices.pct_change(fill_method=None)

    def run_backtest(
        self,
        start_date: str = "2016-01-01",
        end_date: str = "2026-09-01",
        j: int = 12,  # Formation period (months)
        k: int = 3,  # Holding period (months)
        max_stocks: int = 10,  # Max portfolio concentration
        use_sma: bool = True,  # 200-day SMA trend filter
        model: str = "A",  # 'A' = Decile/Rank, 'B' = WRSS
        long_short: bool = False,  # True = Long Winners / Short Losers
        trans_cost: float = 0.0010,  # 0.10% per turnover
        initial_capital: float = 100_000.0,
    ) -> Tuple[BacktestMetrics, pd.DataFrame, pd.DataFrame]:
        """
        Execute the backtest.
        Returns:
            metrics: BacktestMetrics
            monthly_summary: pd.DataFrame with columns ['Date', 'Strategy_Return', 'Benchmark_Return', 'Equity', 'Drawdown', ...]
            holdings_history: pd.DataFrame recording monthly holdings and weights
        """
        # Align monthly dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Ensure we have enough history for formation (J + 1 months before start_date)
        all_month_ends = self.monthly_prices.index
        valid_eval_months = all_month_ends[all_month_ends >= start_dt]
        valid_eval_months = valid_eval_months[valid_eval_months <= end_dt]

        if len(valid_eval_months) < 3:
            raise ValueError(f"Insufficient monthly data between {start_date} and {end_date}.")

        # Exclude benchmark from stock universe
        stock_universe = [c for c in self.monthly_prices.columns if c != self.benchmark_ticker]

        # Structure to track tranche allocations over time:
        # tranche_history[month_idx] = dict of {ticker: weight} for the tranche formed at month_idx
        tranche_history: Dict[int, Dict[str, float]] = {}
        monthly_records = []
        holdings_records = []

        capital = initial_capital
        prev_portfolio_weights: Dict[str, float] = {}

        # Monthly risk-free rate
        rf_monthly = (1.0 + self.risk_free_rate) ** (1.0 / 12.0) - 1.0

        for m_idx, current_month in enumerate(all_month_ends):
            # Check if this month is within the backtest window
            if current_month < start_dt or current_month > end_dt:
                continue

            # In month t (evaluating for returns from end of t-1 to end of t),
            # we form tranche based on information available at end of t-1!
            # Let t_prev_idx be index of t-1 in all_month_ends
            t_prev_idx = m_idx - 1
            if t_prev_idx < j + 1:
                # Not enough history for J-minus-1 formation
                continue

            # Formation return: 12-minus-1 momentum -> P[t-1] / P[t-1-J] - 1
            # To skip month t-1 -> t reversal effect:
            p_prev = self.monthly_prices.iloc[t_prev_idx][stock_universe]
            p_form_start = self.monthly_prices.iloc[t_prev_idx - j][stock_universe]

            formation_returns = (p_prev / p_form_start) - 1.0
            formation_returns = formation_returns.dropna()

            # Filter candidates: require price at t-1 and history
            if use_sma:
                sma_prev = self.monthly_sma200.iloc[t_prev_idx][stock_universe]
                # Condition: P[t-1] > SMA200[t-1]
                above_sma = p_prev > sma_prev
                eligible_tickers = formation_returns.index[
                    formation_returns.index.isin(above_sma.index[above_sma.fillna(False)])
                ]
            else:
                eligible_tickers = formation_returns.index

            # Determine weights for the NEW tranche formed at t_prev
            new_tranche_weights: Dict[str, float] = {}

            if model.upper() == "A":
                # Model A: Top Decile or top max_stocks
                if len(eligible_tickers) > 0:
                    ranked_winners = formation_returns.loc[eligible_tickers].sort_values(ascending=False)
                    n_select = min(max_stocks, len(ranked_winners))
                    selected_longs = ranked_winners.index[:n_select].tolist()

                    if long_short:
                        # Losers: bottom from entire universe or below SMA
                        losers_pool = formation_returns.sort_values(ascending=True)
                        selected_shorts = losers_pool.index[:n_select].tolist()

                        for ticker in selected_longs:
                            new_tranche_weights[ticker] = 0.5 / n_select
                        for ticker in selected_shorts:
                            new_tranche_weights[ticker] = -0.5 / n_select
                    else:
                        # Long-only
                        w_per_stock = 1.0 / n_select
                        for ticker in selected_longs:
                            new_tranche_weights[ticker] = w_per_stock

            elif model.upper() == "B":
                # Model B: WRSS (Weighted Relative Strength Strategy)
                # w_i = r_i - mean(r)
                if len(eligible_tickers) > 0:
                    eligible_returns = formation_returns.loc[eligible_tickers]
                    r_bar = eligible_returns.mean()
                    deviations = eligible_returns - r_bar

                    if long_short:
                        pos_dev = deviations[deviations > 0]
                        neg_dev = deviations[deviations < 0]
                        if not pos_dev.empty and not neg_dev.empty:
                            w_long = 0.5 * (pos_dev / pos_dev.sum())
                            w_short = -0.5 * (neg_dev.abs() / neg_dev.abs().sum())
                            for t, w in w_long.items():
                                new_tranche_weights[t] = w
                            for t, w in w_short.items():
                                new_tranche_weights[t] = w
                    else:
                        # Long-only WRSS
                        pos_dev = deviations[deviations > 0].sort_values(ascending=False)
                        if not pos_dev.empty:
                            top_pos = pos_dev.head(max_stocks)
                            w_normalized = top_pos / top_pos.sum()
                            for t, w in w_normalized.items():
                                new_tranche_weights[t] = w

            # Save this month's formed tranche
            tranche_history[t_prev_idx] = new_tranche_weights

            # Overlapping K-month aggregation:
            # Active portfolio at month m_idx is average of tranches formed at:
            # t_prev_idx, t_prev_idx - 1, ..., t_prev_idx - (K - 1)
            active_weights: Dict[str, float] = {}
            active_tranches_count = 0

            for lag in range(k):
                eval_lag_idx = t_prev_idx - lag
                if eval_lag_idx in tranche_history:
                    t_w = tranche_history[eval_lag_idx]
                    active_tranches_count += 1
                    for ticker, w in t_w.items():
                        active_weights[ticker] = active_weights.get(ticker, 0.0) + (w / k)

            # Monthly stock returns realized over month m_idx: from t-1 to t
            m_returns = self.monthly_returns.iloc[m_idx]
            bench_ret = m_returns.get(self.benchmark_ticker, 0.0)

            # Calculate gross return
            gross_return = 0.0
            invested_weight = 0.0
            for ticker, w in active_weights.items():
                r_stock = m_returns.get(ticker, 0.0)
                if pd.isna(r_stock):
                    r_stock = 0.0
                gross_return += w * r_stock
                invested_weight += abs(w)

            # Cash portion earns monthly risk-free rate (for long-only)
            cash_weight = max(0.0, 1.0 - invested_weight)
            gross_return += cash_weight * rf_monthly

            # Compute turnover and transaction cost:
            # Turnover = sum(|w_{i, t} - w_{i, t-1 end}|)
            turnover = 0.0
            all_tickers = set(active_weights.keys()).union(set(prev_portfolio_weights.keys()))
            for ticker in all_tickers:
                w_current = active_weights.get(ticker, 0.0)
                w_prev_end = prev_portfolio_weights.get(ticker, 0.0)
                turnover += abs(w_current - w_prev_end)

            cost = 0.5 * turnover * trans_cost
            net_return = gross_return - cost

            # Update capital
            capital *= (1.0 + net_return)

            # Update prev_portfolio_weights adjusted for stock drift over the month
            prev_portfolio_weights = {}
            total_end_equity_ratio = (1.0 + gross_return)
            if total_end_equity_ratio > 1e-6:
                for ticker, w in active_weights.items():
                    r_stock = m_returns.get(ticker, 0.0)
                    if pd.isna(r_stock):
                        r_stock = 0.0
                    w_end = w * (1.0 + r_stock) / total_end_equity_ratio
                    prev_portfolio_weights[ticker] = w_end

            # Record
            monthly_records.append({
                "Date": current_month,
                "Strategy_Return": net_return,
                "Gross_Return": gross_return,
                "Benchmark_Return": bench_ret,
                "Turnover": turnover,
                "Cost": cost,
                "Equity": capital,
                "Num_Holdings": len(active_weights),
                "Invested_Weight": invested_weight,
                "Cash_Weight": cash_weight,
            })

            # Record top holdings
            for ticker, w in active_weights.items():
                if abs(w) > 0.001:
                    holdings_records.append({
                        "Date": current_month,
                        "Ticker": ticker,
                        "Weight": w,
                    })

        if not monthly_records:
            raise RuntimeError("No backtest periods generated. Please check start/end date range.")

        results_df = pd.DataFrame(monthly_records).set_index("Date")
        holdings_df = pd.DataFrame(holdings_records)

        # Calculate equity curve for benchmark
        bench_ret_series = results_df["Benchmark_Return"].fillna(0.0)
        results_df["Benchmark_Equity"] = initial_capital * (1.0 + bench_ret_series).cumprod()

        # Calculate drawdowns
        strat_peak = results_df["Equity"].cummax()
        results_df["Strategy_Drawdown"] = (results_df["Equity"] - strat_peak) / strat_peak

        bench_peak = results_df["Benchmark_Equity"].cummax()
        results_df["Benchmark_Drawdown"] = (results_df["Benchmark_Equity"] - bench_peak) / bench_peak

        # Compute summary performance metrics
        metrics = self._compute_metrics(
            results_df=results_df,
            initial_capital=initial_capital,
            rf_annual=self.risk_free_rate,
        )

        return metrics, results_df, holdings_df

    def _compute_metrics(
        self,
        results_df: pd.DataFrame,
        initial_capital: float,
        rf_annual: float,
    ) -> BacktestMetrics:
        """Compute performance analytics and OLS market-model regression."""
        strat_returns = results_df["Strategy_Return"]
        bench_returns = results_df["Benchmark_Return"]

        n_months = len(results_df)
        years = n_months / 12.0

        final_val = results_df["Equity"].iloc[-1]
        total_ret = (final_val / initial_capital) - 1.0
        cagr = (final_val / initial_capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0

        ann_vol = strat_returns.std() * np.sqrt(12.0)
        sharpe = (cagr - rf_annual) / ann_vol if ann_vol > 1e-6 else 0.0

        # Sortino
        downside_returns = strat_returns[strat_returns < 0.0]
        downside_vol = downside_returns.std() * np.sqrt(12.0) if len(downside_returns) > 1 else 1e-6
        sortino = (cagr - rf_annual) / downside_vol if downside_vol > 1e-6 else 0.0

        max_dd = results_df["Strategy_Drawdown"].min()
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0
        win_rate = (strat_returns > 0).mean()
        avg_turnover = results_df["Turnover"].mean()
        avg_holdings = results_df["Num_Holdings"].mean()

        # Benchmark stats
        bench_final = results_df["Benchmark_Equity"].iloc[-1]
        bench_cagr = (bench_final / initial_capital) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        bench_vol = bench_returns.std() * np.sqrt(12.0)
        bench_sharpe = (bench_cagr - rf_annual) / bench_vol if bench_vol > 1e-6 else 0.0
        bench_max_dd = results_df["Benchmark_Drawdown"].min()

        # Market-Model OLS Regression: (R_p - R_f) = alpha + beta * (R_m - R_f)
        rf_monthly = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
        excess_strat = strat_returns - rf_monthly
        excess_bench = bench_returns - rf_monthly

        res = stats.linregress(excess_bench, excess_strat)
        beta = res.slope
        alpha_monthly = res.intercept
        annual_alpha = (1.0 + alpha_monthly) ** 12 - 1.0
        
        # Calculate t-stat of alpha using intercept standard error
        alpha_se = getattr(res, "intercept_stderr", None)
        if alpha_se is None or np.isnan(alpha_se) or alpha_se < 1e-8:
            # Fallback formula for intercept standard error: se = s * sqrt(1/N + x_bar^2 / sum((x - x_bar)^2))
            N = len(excess_bench)
            residuals = excess_strat - (alpha_monthly + beta * excess_bench)
            s_err = np.sqrt(np.sum(residuals**2) / max(1, N - 2))
            x_bar = np.mean(excess_bench)
            ss_x = np.sum((excess_bench - x_bar)**2)
            alpha_se = s_err * np.sqrt((1.0 / N) + (x_bar**2 / ss_x)) if ss_x > 0 else 1e-6

        alpha_tstat = (alpha_monthly / alpha_se) if alpha_se > 1e-8 else 0.0
        r_squared = res.rvalue ** 2

        return BacktestMetrics(
            start_date=results_df.index[0].strftime("%Y-%m-%d"),
            end_date=results_df.index[-1].strftime("%Y-%m-%d"),
            initial_capital=initial_capital,
            final_value=final_val,
            total_return_pct=total_ret * 100.0,
            cagr_pct=cagr * 100.0,
            annual_volatility_pct=ann_vol * 100.0,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd * 100.0,
            calmar_ratio=calmar,
            win_rate_pct=win_rate * 100.0,
            avg_turnover_pct=avg_turnover * 100.0,
            avg_holdings=avg_holdings,
            benchmark_cagr_pct=bench_cagr * 100.0,
            benchmark_volatility_pct=bench_vol * 100.0,
            benchmark_sharpe=bench_sharpe,
            benchmark_max_drawdown_pct=bench_max_dd * 100.0,
            annual_alpha_pct=annual_alpha * 100.0,
            beta=beta,
            alpha_tstat=alpha_tstat,
            r_squared=r_squared,
        )

    def get_annual_returns(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute year-by-year returns for Strategy vs Benchmark."""
        df = results_df[["Strategy_Return", "Benchmark_Return"]].copy()
        df["Year"] = df.index.year

        annual_records = []
        for year, group in df.groupby("Year"):
            strat_ann = (1.0 + group["Strategy_Return"]).prod() - 1.0
            bench_ann = (1.0 + group["Benchmark_Return"]).prod() - 1.0
            excess = strat_ann - bench_ann
            annual_records.append({
                "Year": str(year),
                "Strategy Return (%)": round(strat_ann * 100.0, 2),
                "S&P 500 Return (%)": round(bench_ann * 100.0, 2),
                "Excess Return (%)": round(excess * 100.0, 2),
                "Win": "WIN" if excess > 0 else "LOSS",
            })

        return pd.DataFrame(annual_records).set_index("Year")

    def plot_equity_curve(
        self,
        results_df: pd.DataFrame,
        metrics: BacktestMetrics,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot and save performance comparison chart."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

        # Equity Curve
        ax1.plot(results_df.index, results_df["Equity"], label=f"JT Momentum (CAGR {metrics.cagr_pct:.1f}%)", color="#1f77b4", lw=2)
        ax1.plot(results_df.index, results_df["Benchmark_Equity"], label=f"S&P 500 SPY (CAGR {metrics.benchmark_cagr_pct:.1f}%)", color="#7f7f7f", ls="--", lw=1.5)
        ax1.set_title(f"Jegadeesh-Titman S&P 500 Momentum (10-Year: {metrics.start_date} to {metrics.end_date})", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Portfolio Value ($ USD)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", frameon=True)

        # Drawdown Curve
        ax2.plot(results_df.index, results_df["Strategy_Drawdown"] * 100, label="Strategy Drawdown", color="#d62728", lw=1.5)
        ax2.plot(results_df.index, results_df["Benchmark_Drawdown"] * 100, label="SPY Drawdown", color="#7f7f7f", ls=":", lw=1.2)
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower left", frameon=True)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            print(f"Equity chart saved to {save_path}")
        plt.close()


def print_backtest_report(metrics: BacktestMetrics, annual_df: pd.DataFrame) -> None:
    """Print clean summary performance report to stdout."""
    print("\n" + "=" * 75)
    print("      JEGADEESH & TITMAN (1993) S&P 500 MOMENTUM 10-YEAR BACKTEST")
    print("=" * 75)
    print(f"Period:              {metrics.start_date} to {metrics.end_date}")
    print(f"Initial Capital:     ${metrics.initial_capital:,.2f}")
    print(f"Final Value:         ${metrics.final_value:,.2f}  (Total Return: {metrics.total_return_pct:+.2f}%)")
    print("-" * 75)
    print(f"{'Metric':<30} {'JT Strategy':<20} {'S&P 500 (SPY)':<20}")
    print("-" * 75)
    print(f"{'CAGR':<30} {metrics.cagr_pct:>10.2f}%       {metrics.benchmark_cagr_pct:>10.2f}%")
    print(f"{'Annualized Volatility':<30} {metrics.annual_volatility_pct:>10.2f}%       {metrics.benchmark_volatility_pct:>10.2f}%")
    print(f"{'Sharpe Ratio (Rf=2%)':<30} {metrics.sharpe_ratio:>10.2f}        {metrics.benchmark_sharpe:>10.2f}")
    print(f"{'Sortino Ratio':<30} {metrics.sortino_ratio:>10.2f}        {'N/A':>10}")
    print(f"{'Maximum Drawdown':<30} {metrics.max_drawdown_pct:>10.2f}%       {metrics.benchmark_max_drawdown_pct:>10.2f}%")
    print(f"{'Calmar Ratio':<30} {metrics.calmar_ratio:>10.2f}        {'N/A':>10}")
    print(f"{'Monthly Win Rate':<30} {metrics.win_rate_pct:>10.2f}%       {'N/A':>10}")
    print(f"{'Avg Monthly Turnover':<30} {metrics.avg_turnover_pct:>10.2f}%       {'N/A':>10}")
    print(f"{'Avg Unique Holdings':<30} {metrics.avg_holdings:>10.1f} stocks   {'500 stocks':>10}")
    print("-" * 75)
    print("CAPM / MARKET-MODEL REGRESSION (vs S&P 500):")
    print(f"Annualized Jensen's Alpha: {metrics.annual_alpha_pct:+.2f}%")
    print(f"Portfolio Beta:            {metrics.beta:.2f}")
    print(f"Alpha t-statistic:         {metrics.alpha_tstat:.2f} {'(Statistically Significant)' if abs(metrics.alpha_tstat) >= 1.96 else '(Not Significant)'}")
    print(f"R-squared:                 {metrics.r_squared:.3f}")
    print("-" * 75)
    print("\nYEAR-BY-YEAR PERFORMANCE:")
    print(annual_df.to_string())
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run S&P 500 Jegadeesh & Titman Momentum Backtest.")
    parser.add_argument("--start", type=str, default="2016-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-09-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--j", type=int, default=12, help="Formation period in months (default: 12)")
    parser.add_argument("--k", type=int, default=3, help="Holding period in months (default: 3)")
    parser.add_argument("--max-stocks", type=int, default=10, help="Max holdings cap (default: 10)")
    parser.add_argument("--model", type=str, default="A", choices=["A", "B"], help="Model A (Decile) or Model B (WRSS)")
    parser.add_argument("--no-sma", action="store_true", help="Disable 200-day SMA trend filter")
    parser.add_argument("--long-short", action="store_true", help="Enable long-short decile strategy")
    parser.add_argument("--cost", type=float, default=0.0010, help="Transaction cost per turnover (default: 0.0010 = 10bps)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital in USD (default: 100000)")
    parser.add_argument("--plot", type=str, default="output/sp500_momentum_equity.png", help="Path to save equity plot")

    args = parser.parse_args()

    prices, _ = load_price_data()
    engine = MomentumEngine(prices)

    metrics, results_df, holdings_df = engine.run_backtest(
        start_date=args.start,
        end_date=args.end,
        j=args.j,
        k=args.k,
        max_stocks=args.max_stocks,
        use_sma=not args.no_sma,
        model=args.model,
        long_short=args.long_short,
        trans_cost=args.cost,
        initial_capital=args.capital,
    )

    annual_df = engine.get_annual_returns(results_df)
    print_backtest_report(metrics, annual_df)

    if args.plot:
        engine.plot_equity_curve(results_df, metrics, save_path=args.plot)


if __name__ == "__main__":
    main()
