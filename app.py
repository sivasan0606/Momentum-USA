"""
Interactive Streamlit Web Dashboard for Jegadeesh & Titman (1993) S&P 500 Momentum Strategy.
Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_manager import BENCHMARK_TICKER, load_price_data
from momentum_engine import MomentumEngine

st.set_page_config(
    page_title="JT Momentum 1993 - S&P 500",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #262c3a 100%);
        border: 1px solid #363d4e;
        border-radius: 10px;
        padding: 16px 20px;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .text-green { color: #34d399; }
    .text-red { color: #f87171; }
    .text-muted { color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_cached_market_data():
    prices, constituents = load_price_data()
    return prices, constituents


def main():
    st.title("📈 Jegadeesh & Titman (1993) Momentum Engine")
    st.caption("10-Year Backtest & Analytics for the S&P 500 Universe (2016–2026 YTD)")

    with st.spinner("Loading S&P 500 price matrix..."):
        prices, constituents = get_cached_market_data()

    engine = MomentumEngine(prices)

    # ---------------- Sidebar Controls ----------------
    st.sidebar.header("⚙️ Strategy Parameters")

    model = st.sidebar.selectbox(
        "Strategy Model",
        options=["A", "B"],
        format_func=lambda x: "Model A: J/K Decile Rank Strategy" if x == "A" else "Model B: Weighted Relative Strength (WRSS)",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        j = st.number_input("Formation J (mo)", min_value=3, max_value=24, value=12, step=1, help="Momentum lookback (months)")
    with col2:
        k = st.number_input("Holding K (mo)", min_value=1, max_value=12, value=3, step=1, help="Overlapping holding duration")

    max_stocks = st.sidebar.slider(
        "Max Holdings per Tranche",
        min_value=3,
        max_value=30,
        value=10,
        step=1,
        help="Target number of stocks picked per formation slice",
    )

    use_sma = st.sidebar.checkbox(
        "200-Day SMA Trend Filter",
        value=True,
        help="Only long stocks trading above their 200-day simple moving average at formation",
    )

    long_short = st.sidebar.checkbox(
        "Long-Short Strategy",
        value=False,
        help="Long winners and short losers (default is Long-Only)",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("💵 Portfolio & Capital")

    initial_capital = st.sidebar.number_input(
        "Initial Capital ($ USD)",
        min_value=10_000.0,
        max_value=10_000_000.0,
        value=100_000.0,
        step=10_000.0,
    )

    trans_cost_bps = st.sidebar.slider(
        "Transaction Fee (bps)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        help="10 bps = 0.10% per turnover",
    )
    trans_cost = trans_cost_bps / 10_000.0

    min_date = pd.to_datetime("2016-01-01")
    max_date = prices.index[-1]
    date_range = st.sidebar.date_input(
        "Backtest Period",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date = str(date_range[0])
        end_date = str(date_range[1])
    else:
        start_date = "2016-01-01"
        end_date = str(max_date.date())

    # Run Backtest
    with st.spinner("Executing backtest across S&P 500..."):
        metrics, results_df, holdings_df = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
            j=j,
            k=k,
            max_stocks=max_stocks,
            use_sma=use_sma,
            model=model,
            long_short=long_short,
            trans_cost=trans_cost,
            initial_capital=initial_capital,
        )
        annual_df = engine.get_annual_returns(results_df)

    # ---------------- KPI Cards ----------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Portfolio Value</div>
                <div class="metric-value">${metrics.final_value:,.0f}</div>
                <div class="metric-sub text-green">+{metrics.total_return_pct:,.1f}% Total Return</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        cagr_diff = metrics.cagr_pct - metrics.benchmark_cagr_pct
        diff_color = "text-green" if cagr_diff > 0 else "text-red"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">CAGR (Annualized)</div>
                <div class="metric-value">{metrics.cagr_pct:.1f}%</div>
                <div class="metric-sub {diff_color}">{cagr_diff:+.1f}% vs SPY ({metrics.benchmark_cagr_pct:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Sharpe Ratio</div>
                <div class="metric-value">{metrics.sharpe_ratio:.2f}</div>
                <div class="metric-sub text-muted">SPY: {metrics.benchmark_sharpe:.2f} | Sortino: {metrics.sortino_ratio:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Max Drawdown</div>
                <div class="metric-value" style="color: #f87171;">{metrics.max_drawdown_pct:.1f}%</div>
                <div class="metric-sub text-muted">SPY: {metrics.benchmark_max_drawdown_pct:.1f}% | Calmar: {metrics.calmar_ratio:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Jensen's Alpha</div>
                <div class="metric-value" style="color: #a855f7;">{metrics.annual_alpha_pct:+.1f}%</div>
                <div class="metric-sub text-muted">Beta: {metrics.beta:.2f} | t-stat: {metrics.alpha_tstat:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- Tabs ----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance & Equity",
        "📅 Year-by-Year Analysis",
        "🎯 Current Live Portfolio",
        "🗓️ Monthly Returns Heatmap",
        "📖 Strategy Methodology",
    ])

    with tab1:
        st.subheader("Equity Curve vs S&P 500 (SPY)")

        chart_data = results_df[["Equity", "Benchmark_Equity"]].reset_index()
        chart_data.columns = ["Date", "JT Strategy", "S&P 500 (SPY)"]
        melted_data = chart_data.melt("Date", var_name="Portfolio", value_name="Value")

        line_chart = (
            alt.Chart(melted_data)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Value:Q", title="Portfolio Value ($ USD)"),
                color=alt.Color(
                    "Portfolio:N",
                    scale=alt.Scale(domain=["JT Strategy", "S&P 500 (SPY)"], range=["#38bdf8", "#94a3b8"]),
                ),
                tooltip=["Date:T", "Portfolio:N", alt.Tooltip("Value:Q", format="$,.0f")],
            )
            .properties(height=420)
            .interactive()
        )
        st.altair_chart(line_chart, use_container_width=True)

        st.subheader("Underwater Drawdown")
        dd_data = (results_df[["Strategy_Drawdown", "Benchmark_Drawdown"]] * 100).reset_index()
        dd_data.columns = ["Date", "JT Strategy", "S&P 500 (SPY)"]
        melted_dd = dd_data.melt("Date", var_name="Portfolio", value_name="Drawdown")

        dd_chart = (
            alt.Chart(melted_dd)
            .mark_area(opacity=0.35)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Drawdown:Q", title="Drawdown (%)"),
                color=alt.Color(
                    "Portfolio:N",
                    scale=alt.Scale(domain=["JT Strategy", "S&P 500 (SPY)"], range=["#ef4444", "#64748b"]),
                ),
                tooltip=["Date:T", "Portfolio:N", alt.Tooltip("Drawdown:Q", format=".2f")],
            )
            .properties(height=240)
            .interactive()
        )
        st.altair_chart(dd_chart, use_container_width=True)

    with tab2:
        st.subheader("Annual Returns: Strategy vs S&P 500")

        # Annual bar chart
        bar_df = annual_df.reset_index()
        bar_melted = bar_df.melt(
            id_vars=["Year"],
            value_vars=["Strategy Return (%)", "S&P 500 Return (%)"],
            var_name="Asset",
            value_name="Return",
        )

        bar_chart = (
            alt.Chart(bar_melted)
            .mark_bar()
            .encode(
                x=alt.X("Year:O", title="Calendar Year"),
                y=alt.Y("Return:Q", title="Annual Return (%)"),
                color=alt.Color(
                    "Asset:N",
                    scale=alt.Scale(
                        domain=["Strategy Return (%)", "S&P 500 Return (%)"],
                        range=["#38bdf8", "#64748b"],
                    ),
                ),
                xOffset="Asset:N",
                tooltip=["Year:O", "Asset:N", alt.Tooltip("Return:Q", format="+.2f")],
            )
            .properties(height=380)
        )
        st.altair_chart(bar_chart, use_container_width=True)

        st.subheader("Performance Breakdown Table")
        st.dataframe(
            annual_df.style.map(
                lambda v: "color: #34d399; font-weight: bold;" if v == "WIN" else ("color: #f87171;" if v == "LOSS" else ""),
                subset=["Win"],
            ),
            use_container_width=True,
        )

    with tab3:
        st.subheader("Latest Formed Portfolio Holdings")
        if not holdings_df.empty:
            latest_date = holdings_df["Date"].max()
            current_holdings = holdings_df[holdings_df["Date"] == latest_date].copy()
            current_holdings["Weight (%)"] = (current_holdings["Weight"] * 100).round(2)

            # Merge sector metadata
            merged_holdings = current_holdings.merge(
                constituents[["Yahoo_Symbol", "Security", "GICS Sector"]],
                left_on="Ticker",
                right_on="Yahoo_Symbol",
                how="left",
            )
            cols = ["Ticker", "Security", "GICS Sector", "Weight (%)"]
            st.write(f"**As of {latest_date.strftime('%B %Y')}** (Total Active Holdings: {len(merged_holdings)} stocks)")
            st.dataframe(merged_holdings[cols].sort_values("Weight (%)", ascending=False), use_container_width=True)

            # Sector distribution
            if "GICS Sector" in merged_holdings.columns:
                st.subheader("Sector Allocation")
                sector_dist = merged_holdings.groupby("GICS Sector")["Weight (%)"].sum().reset_index()
                pie = (
                    alt.Chart(sector_dist)
                    .mark_arc(innerRadius=50)
                    .encode(
                        theta=alt.Theta("Weight (%):Q"),
                        color=alt.Color("GICS Sector:N"),
                        tooltip=["GICS Sector:N", alt.Tooltip("Weight (%):Q", format=".1f")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(pie, use_container_width=True)

    with tab4:
        st.subheader("Monthly Returns Matrix (%)")
        matrix_df = results_df[["Strategy_Return"]].copy()
        matrix_df["Year"] = matrix_df.index.year
        matrix_df["Month"] = matrix_df.index.strftime("%b")

        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivoted = matrix_df.pivot_table(index="Year", columns="Month", values="Strategy_Return", aggfunc="mean")
        pivoted = (pivoted * 100).round(2)
        pivoted = pivoted.reindex(columns=[m for m in month_order if m in pivoted.columns])

        # Add Year total
        pivoted["Total Year"] = annual_df["Strategy Return (%)"]

        def color_returns(val):
            if pd.isna(val):
                return ""
            color = "#34d399" if val > 0 else "#f87171"
            return f"color: {color}; font-weight: 500;"

        st.dataframe(pivoted.style.map(color_returns), use_container_width=True)

    with tab5:
        st.subheader("Academic Specification: Jegadeesh & Titman (1993)")
        st.markdown(
            """
            ### Background & Methodology
            This engine implements **Jegadeesh & Titman (1993)**, *"Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency"* (*Journal of Finance*), adapted for the US S&P 500 universe.

            #### 1. Momentum Ranking (12-minus-1)
            Stocks are ranked based on their past $J=12$ months return, **skipping the most recent month ($t-1$)**:
            $$f_{i, t} = \\frac{P_{i, t-1}}{P_{i, t-1-J}} - 1$$
            Skipping the immediate past month avoids microstructure bid-ask bounce and the well-documented 1-month short-term reversal effect.

            #### 2. Overlapping Tranches ($K=3$ Holding Period)
            To avoid 100% turnover every $K$ months, the portfolio holds $K$ overlapping slices. At any month $t$:
            $$W_t = \\frac{1}{K} \\sum_{\\tau=0}^{K-1} W_{t-\\tau}^{\\text{tranche}}$$
            Each month, only $\\frac{1}{K}$ of the portfolio rolls off and is replaced by the newest top momentum slice.

            #### 3. 200-Day SMA Trend Filter
            A position is only taken if the stock's price at formation is strictly above its 200-day simple moving average:
            $$P_{i, t-1} > \\text{SMA}_{200, i}(t-1)$$
            This prevents buying high-momentum names that have begun to break down or rollover into secular bear markets.
            """
        )


if __name__ == "__main__":
    main()
