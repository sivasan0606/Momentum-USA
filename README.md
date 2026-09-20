# JT 1993 Momentum Backtester (S&P 500)

Production-ready implementation of **Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"** for the **S&P 500 universe** (US Equities), delivered as a Python backtesting engine (`momentum_engine.py`), an interactive **Streamlit** dashboard (`app.py`), and a **monthly/quarterly momentum advisor** (`advisor.py`) that emits self-contained interactive `advisor.html` and `portfolio.html` reports.

---

## Strategies (per the paper)

### Model A — J-month / K-month Decile Strategy
- **Formation return** ($12$-minus-$1$ when $J=12$):
  $$f_{i, t} = \frac{P_{i, t-1}}{P_{i, t-1-J}} - 1$$
  Skipping the immediate past month ($t-1 \to t$) avoids microstructure frictions, bid-ask bounce, and 1-month short-term reversals.
- **Rank & Sort**: Sort all ~503 S&P 500 stocks each month by formation return: **long the top $N$ (e.g. 10 stocks)**, optionally short the bottom $N$ (Losers).
- **Overlapping Tranches**: Hold each formation slice for **$K$ months** ($K=3$ default) using overlapping monthly rebalancing. In month $t$, the portfolio averages the $1/K$ slices formed over the past $K$ months, reducing monthly turnover and matching the true JT portfolio construction.

### Model B — Weighted Relative Strength Strategy (WRSS)
- Continuous weights proportional to each stock's deviation from the cross-sectional universe mean:
  $$w_{it} = r_{it} - \bar{r}_t$$
- Long top positive deviations, normalized to 100% long (or long-short).

---

## Risk Overlays & Execution Constraints

- **200-Day SMA Trend Filter**: Long positions are only initiated/maintained in confirmed uptrends ($P[t-1] > \text{SMA}_{200}[t-1]$), evaluated using end-of-previous-month data (zero look-ahead bias).
- **Max-Stock Concentration Cap**: Active long holdings per tranche capped (default: 10 stocks; also 6, 15, or decile).
- **Transaction Costs**: Charged on monthly turnover (default: 0.10% / 10 bps for liquid US equities).
- **Rebalance Schedule**: Pure 3-month quarterly rebalance cycle (aligns with $K=3$ holding period; no intra-quarter stop loss).
- **Quarterly Exit Discipline**: Holdings are reviewed at 3-month quarter ends and exited only if they drop out of top rankings or fall below their 200-day SMA.

---

## Performance Analytics & OLS Regression

- **Metrics**: Total return, annualized CAGR, annualized volatility, Sharpe ratio ($R_f=2.0\%$), Sortino ratio, max drawdown, Calmar ratio, monthly win rate, and turnover.
- **Market-Model Regression** against `SPY` (S&P 500 ETF) via OLS:
  $$R_{p, t} - R_{f, t} = \alpha + \beta (R_{m, t} - R_{f, t}) + \epsilon_t$$
  Emits annualized **Jensen's Alpha**, **Beta**, Alpha **t-statistic**, and $R^2$.

---

## Reference Results (2016-01 to 2026-YTD, S&P 500 Universe, $100k, 10 bps cost)

S&P 500 (`SPY`) Buy & Hold over the window: **+385.2% Total Return, 15.7% CAGR, 0.91 Sharpe, -23.9% Max Drawdown**.

| Strategy Model | Formation $J$ / Holding $K$ | Max Stocks | 200-SMA Filter | CAGR | Sharpe | Max Drawdown | Alpha (ann.) | Beta | Alpha t-stat | Avg Turnover |
|---|---|---|---|---|---|---|---|---|---|---|
| **Model A (Top 6)** | $J=12$, $K=3$ | 6 | Yes | **68.0%** | **1.65** | -32.3% | **+39.9%** | 1.75 | 3.55 (p < 0.001) | 35.5% |
| **Model A (Top 10)** | $J=12$, $K=3$ | 10 | Yes | **50.7%** | **1.39** | -31.1% | **+25.3%** | 1.64 | 2.88 (p < 0.01) | 36.0% |
| **Model A (Top 15)** | $J=12$, $K=3$ | 15 | Yes | **41.6%** | **1.29** | -29.7% | **+18.5%** | 1.50 | 2.56 (p < 0.05) | 35.7% |
| **Model B (WRSS 10)** | $J=12$, $K=3$ | 10 | Yes | **63.6%** | **1.51** | -32.8% | **+36.2%** | 1.79 | 3.19 (p < 0.01) | 33.1% |
| **Model A (No SMA)** | $J=12$, $K=3$ | 10 | No | 51.4% | 1.40 | -31.1% | +25.7% | 1.66 | 2.90 (p < 0.01) | 35.6% |
| **Model A Long-Short** | $J=12$, $K=3$ | 10 L / 10 S | Yes | 6.8% | 0.25 | -35.5% | +8.3% | -0.10 | 1.29 | 42.5% |
| **S&P 500 (SPY)** | Buy & Hold | 500 | — | 15.7% | 0.91 | -23.9% | 0.0% | 1.00 | — | 0.0% |

### Year-by-Year Performance (Model A, 10 Stocks):
```
      Strategy Return (%)  S&P 500 Return (%)  Excess Return (%)  Outperform
Year                                                                        
2016                32.20               17.87             +14.33         WIN
2017                36.94               21.71             +15.23         WIN
2018                 2.44               -4.57              +7.01         WIN
2019                36.64               31.22              +5.42         WIN
2020                86.33               18.33             +67.99         WIN
2021                31.53               28.73              +2.80         WIN
2022                 7.33              -18.18             +25.50         WIN
2023                42.31               26.18             +16.14         WIN
2024               157.36               24.89            +132.47         WIN
2025                76.63               17.72             +58.91         WIN
2026 (YTD)          77.91               13.08             +64.83         WIN
```

---

## File Structure

```
Momemtum USA/
├── data_manager.py           # Scrapes S&P 500 constituents, downloads & caches prices in Parquet
├── momentum_engine.py        # Core backtest engine & CLI (Models A/B, SMA filter, OLS regression)
├── app.py                    # Interactive Streamlit dashboard (visualizations, heatmaps, live sliders)
├── advisor.py                # Momentum Advisor generating interactive advisor.html & portfolio.html
├── holdings.csv              # Initial seed holdings for portfolio tracker
├── tests/
│   └── test_momentum.py      # Unit test suite (reversals, SMA exclusion, weights, CAPM)
├── output/
│   └── sp500_momentum_equity.png # Exported 10-year equity & drawdown comparison chart
├── advisor.html              # Standalone interactive recommendation report
└── portfolio.html            # Standalone interactive portfolio tracker (with localStorage)
```

---

## Installation & Setup

```bash
pip install numpy pandas matplotlib yfinance pyarrow streamlit scipy requests beautifulsoup4
```

---

## How to Run

### 1. CLI Backtest
Run standard 10-year backtest (Model A, $J=12$, $K=3$, 10 stocks, SMA filter):
```bash
python3 momentum_engine.py
```

Custom CLI arguments:
```bash
python3 momentum_engine.py \
  --start 2016-01-01 \
  --end 2026-09-01 \
  --j 12 \
  --k 3 \
  --max-stocks 10 \
  --model A \
  --capital 100000 \
  --plot output/sp500_momentum_equity.png
```

### 2. Run Unit Tests
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Interactive Streamlit Dashboard
Launch the web interface:
```bash
streamlit run app.py
```

### 4. Monthly/Quarterly Advisor & Interactive Web Suite
Generate fresh recommendations and portfolio tracking HTML files:
```bash
python3 advisor.py --cash 50000 --max-stocks 6
```
To enable **1-click browser live scans and backtests**:
```bash
python3 advisor.py --serve --cash 50000
```
Open in any browser:
- **`http://localhost:8770/advisor.html`**: Momentum Advisor (Target allocations, order tickets, 1-click live scan).
- **`http://localhost:8770/portfolio.html`**: Portfolio Tracker (Live holdings, P&L, quarterly rebalance monitor).
- **`http://localhost:8770/backtest.html`**: Backtest Lab (Run custom backtests anytime with live charts & KPIs).

---

## Caveats & Academic Notes
- **Survivorship Bias**: As is standard when using current S&P 500 constituents without a subscription to point-in-time CRSP/Compustat survivorship-free databases, stocks delisted or acquired over the past decade are absent from the universe.
- **Adjusted Prices**: Dividend-adjusted total return prices from Yahoo Finance are utilized.
- **Regime Shifts**: Momentum generates outsized alpha during trending and dispersion markets, but can experience elevated volatility or drawdown during sudden sharp macro reversals.