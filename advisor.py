"""
Quarterly & Monthly Momentum Advisor for S&P 500.

Generates two linked, self-contained interactive HTML dashboards:
1. advisor.html: BUY / TOP-UP / HOLD / SELL recommendations with target allocations.
2. portfolio.html: Live portfolio tracking, localStorage persistence, stop-loss monitors.

Can run as CLI or with --serve for one-click browser re-scans.
"""

import argparse
import http.server
import json
import os
import socketserver
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd

from data_manager import BENCHMARK_TICKER, get_sp500_constituents, load_price_data
from momentum_engine import MomentumEngine

DEFAULT_PORT = 8770


def get_current_momentum_targets(
    max_stocks: int = 10,
    j: int = 12,
    use_sma: bool = True,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, float]:
    """
    Computes current top momentum targets from latest available S&P 500 prices.
    Returns:
        targets_df: DataFrame with ['Rank', 'Ticker', 'Security', 'GICS Sector', 'Momentum_12M_Pct', 'Current_Price', 'SMA_200', 'Above_SMA']
        latest_prices: Series of latest prices for all stocks
        benchmark_price: Latest SPY price
    """
    prices, constituents = load_price_data(force_refresh=force_refresh)

    # 200-day daily SMA
    daily_sma = prices.rolling(window=200, min_periods=150).mean()

    # Resample to month ends
    try:
        monthly_prices = prices.resample("ME").last()
    except ValueError:
        monthly_prices = prices.resample("M").last()

    # Most recent month-end formation return (skipping last 1 month to avoid reversal)
    # If currently midway through month, we use P[t-1] / P[t-1-J] - 1
    # or latest available price vs price 12 months ago
    n_months = len(monthly_prices)
    if n_months < j + 2:
        raise ValueError("Insufficient history for formation.")

    p_eval = monthly_prices.iloc[-2]  # end of previous month
    p_start = monthly_prices.iloc[-2 - j]
    mom_returns = (p_eval / p_start) - 1.0

    latest_close = prices.iloc[-1]
    latest_sma = daily_sma.iloc[-1]

    stock_tickers = [c for c in prices.columns if c != BENCHMARK_TICKER]

    records = []
    for ticker in stock_tickers:
        cur_p = latest_close.get(ticker)
        cur_sma = latest_sma.get(ticker)
        ret_12m = mom_returns.get(ticker)

        if pd.isna(cur_p) or pd.isna(ret_12m):
            continue

        above_sma = (cur_p > cur_sma) if pd.notna(cur_sma) else False

        if use_sma and not above_sma:
            continue

        records.append({
            "Ticker": ticker,
            "Momentum_12M_Pct": round(ret_12m * 100.0, 2),
            "Current_Price": round(float(cur_p), 2),
            "SMA_200": round(float(cur_sma), 2) if pd.notna(cur_sma) else None,
            "Above_SMA": above_sma,
        })

    df = pd.DataFrame(records)
    df.sort_values("Momentum_12M_Pct", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Rank"] = df.index + 1

    # Merge metadata
    const_meta = constituents.set_index("Yahoo_Symbol")[["Security", "GICS Sector"]].to_dict("index")
    df["Security"] = df["Ticker"].map(lambda t: const_meta.get(t, {}).get("Security", t))
    df["GICS Sector"] = df["Ticker"].map(lambda t: const_meta.get(t, {}).get("GICS Sector", "N/A"))

    top_targets = df.head(max_stocks).copy()
    bench_price = float(latest_close.get(BENCHMARK_TICKER, 0.0))

    return top_targets, latest_close, bench_price


def build_advisor_html(
    targets_df: pd.DataFrame,
    cash: float,
    stoploss_pct: float = 0.0,
    seed_holdings_file: str = "holdings.csv",
) -> str:
    """Build the standalone interactive advisor.html string."""
    targets_json = targets_df.to_json(orient="records")

    # Load initial seed holdings if available
    seed_holdings = []
    if os.path.exists(seed_holdings_file):
        try:
            sdf = pd.read_csv(seed_holdings_file)
            seed_holdings = sdf.to_dict(orient="records")
        except Exception:
            pass
    seed_holdings_json = json.dumps(seed_holdings)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>S&P 500 Momentum Advisor</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #151b2b;
      --border: #242f48;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --green: #10b981;
      --red: #ef4444;
      --yellow: #f59e0b;
      --purple: #8b5cf6;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    header {{ display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }}
    h1 {{ font-size: 1.6rem; color: #fff; font-weight: 700; }}
    .nav-links a {{ color: var(--accent); text-decoration: none; margin-left: 18px; font-weight: 500; font-size: 0.95rem; }}
    .nav-links a:hover {{ text-decoration: underline; }}
    .btn {{ background: #2563eb; color: #fff; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600; transition: background 0.2s; }}
    .btn:hover {{ background: #1d4ed8; }}
    .btn-scan {{ background: #059669; }}
    .btn-scan:hover {{ background: #047857; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }}
    .card-title {{ font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
    .card-val {{ font-size: 1.6rem; font-weight: 700; color: #fff; }}
    .card-sub {{ font-size: 0.85rem; color: var(--text-muted); margin-top: 4px; }}
    .input-group {{ display: flex; align-items: center; gap: 10px; margin-top: 8px; }}
    .input-group input {{ background: #0b0f19; border: 1px solid var(--border); border-radius: 6px; color: #fff; padding: 6px 10px; font-size: 1rem; width: 140px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: var(--card-bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }}
    th, td {{ padding: 12px 16px; text-align: left; font-size: 0.9rem; }}
    th {{ background: #1c2438; color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
    tr:not(:last-child) td {{ border-bottom: 1px solid var(--border); }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }}
    .badge-buy {{ background: rgba(16, 185, 129, 0.2); color: var(--green); }}
    .badge-topup {{ background: rgba(56, 189, 248, 0.2); color: var(--accent); }}
    .badge-hold {{ background: rgba(245, 158, 11, 0.2); color: var(--yellow); }}
    .badge-sell {{ background: rgba(239, 68, 68, 0.2); color: var(--red); }}
    .badge-sma {{ background: rgba(139, 92, 246, 0.2); color: var(--purple); }}
    .playbook {{ background: #111827; border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-top: 30px; font-size: 0.9rem; }}
    .playbook h3 {{ color: var(--accent); margin-bottom: 10px; }}
    .playbook ul {{ padding-left: 20px; color: var(--text-muted); }}
    .playbook li {{ margin-bottom: 6px; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>🎯 S&P 500 Momentum Advisor</h1>
        <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 4px;">Jegadeesh & Titman (1993) Systematic Execution & Rebalance Plan</p>
      </div>
      <div class="nav-links">
        <button class="btn btn-scan" id="runScanBtn" onclick="triggerScan()">🔄 Run Live Scan</button>
        <a href="portfolio.html">💼 Portfolio Tracker</a>
        <a href="backtest.html">📊 Backtest Lab →</a>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-title">Available New Cash</div>
        <div class="input-group">
          <span style="font-size: 1.2rem; color: var(--accent);">$</span>
          <input type="number" id="cashInput" value="{cash:.0f}" step="1000" onchange="renderAdvisor()" />
        </div>
        <div class="card-sub">Adjust cash to update share orders</div>
      </div>
      <div class="card">
        <div class="card-title">Target Universe</div>
        <div class="card-val">{len(targets_df)} Stocks</div>
        <div class="card-sub">Top 12M Momentum &gt; 200 SMA</div>
      </div>
      <div class="card">
        <div class="card-title">Rebalance Horizon</div>
        <div class="card-val">Quarterly (3 Months)</div>
        <div class="card-sub">No intra-quarter stop loss</div>
      </div>
      <div class="card">
        <div class="card-title">Rebalance Cycle</div>
        <div class="card-val">Jan / Apr / Jul / Oct</div>
        <div class="card-sub">Top-Up only / No premature trims</div>
      </div>
    </div>

    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px; margin-bottom: 8px;">
      <h2 style="font-size: 1.2rem; color: #fff;">📋 Target Portfolio & Action Recommendations</h2>
      <label style="font-size: 0.9rem; color: var(--accent); cursor: pointer; display: flex; align-items: center; gap: 8px;">
        <input type="checkbox" id="fractionalToggle" checked onchange="renderAdvisor()" style="transform: scale(1.2); cursor: pointer;" />
        <strong>Enable Fractional Shares (Dollar-Based Sizing)</strong>
      </label>
    </div>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Ticker</th>
          <th>Security</th>
          <th>Sector</th>
          <th>Current Price</th>
          <th>12M Mom (%)</th>
          <th>Recommendation</th>
          <th>Target Shares</th>
          <th>Target Value</th>
          <th>200-Day SMA</th>
        </tr>
      </thead>
      <tbody id="recommendationsTable">
        <!-- Injected via JavaScript -->
      </tbody>
    </table>

    <div class="playbook">
      <h3>📜 Momentum Playbook Rules</h3>
      <ul>
        <li><strong>Buy</strong>: Top-decile 12-month momentum stocks trading above their 200-day simple moving average. Allocate capital equally across positions.</li>
        <li><strong>Rebalance Cycle</strong>: Strictly every 3 months (Quarterly: Jan, Apr, Jul, Oct). Hold positions through intermediate fluctuations with no intra-quarter stop loss.</li>
        <li><strong>Quarterly Review</strong>: At the 3-month review, exit only holdings that have fallen out of the top momentum rankings or dropped below their 200-day SMA.</li>
        <li><strong>Golden Rule for Winners</strong>: Never trim an existing winner during rebalance while it remains in the top rankings—let compounders run.</li>
      </ul>
    </div>
  </div>

  <script>
    const targets = {targets_json};
    const defaultSeed = {seed_holdings_json};

    function getSavedHoldings() {{
      const stored = localStorage.getItem("sp500_momentum_holdings");
      if (stored) {{
        try {{ return JSON.parse(stored); }} catch(e) {{}}
      }}
      return defaultSeed;
    }}

    function renderAdvisor() {{
      const cash = parseFloat(document.getElementById("cashInput").value) || 0;
      const isFractional = document.getElementById("fractionalToggle").checked;
      const holdings = getSavedHoldings();
      const tbody = document.getElementById("recommendationsTable");
      tbody.innerHTML = "";

      const heldTickers = new Set(holdings.map(h => h.ticker));
      const cashPerStock = targets.length > 0 ? cash / targets.length : 0;

      targets.forEach(t => {{
        const isHeld = heldTickers.has(t.Ticker);
        let action = isHeld ? "TOP-UP" : "BUY";
        let badgeClass = isHeld ? "badge-topup" : "badge-buy";

        const curPrice = t.Current_Price;
        let sharesToBuy = "0";
        let estCost = "0.00";

        if (curPrice > 0) {{
          if (isFractional) {{
            sharesToBuy = (cashPerStock / curPrice).toFixed(4) + " shs";
            estCost = cashPerStock.toFixed(2);
          }} else {{
            const wholeShs = Math.floor(cashPerStock / curPrice);
            sharesToBuy = wholeShs + " shs";
            estCost = (wholeShs * curPrice).toFixed(2);
          }}
        }}
        const smaVal = t.SMA_200 ? "$" + t.SMA_200.toFixed(2) : "N/A";

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><strong>#${{t.Rank}}</strong></td>
          <td style="color: var(--accent); font-weight: 700;">${{t.Ticker}}</td>
          <td>${{t.Security}}</td>
          <td><span class="badge badge-sma">${{t["GICS Sector"]}}</span></td>
          <td>$${{curPrice.toFixed(2)}}</td>
          <td style="color: #34d399; font-weight: 600;">+${{t.Momentum_12M_Pct}}%</td>
          <td><span class="badge ${{badgeClass}}">${{action}}</span></td>
          <td><strong>${{sharesToBuy}}</strong></td>
          <td>$${{Number(estCost).toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}}</td>
          <td style="color: #38bdf8; font-weight: 600;">${{smaVal}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function triggerScan() {{
      fetch('/api/scan')
        .then(res => res.json())
        .then(data => {{
          alert("Scan completed! Reloading targets...");
          location.reload();
        }})
        .catch(err => {{
          alert("Note: To enable 1-click live scanning, start advisor with:\\npython3 advisor.py --serve");
        }});
    }}

    window.addEventListener("DOMContentLoaded", renderAdvisor);
  </script>
</body>
</html>
"""
    return html


def build_portfolio_html(
    latest_prices: pd.Series,
    stoploss_pct: float = 0.0,
    seed_holdings_file: str = "holdings.csv",
) -> str:
    """Build the standalone interactive portfolio.html string."""
    # Convert latest prices to a clean dict
    prices_dict = {str(k): round(float(v), 2) for k, v in latest_prices.items() if pd.notna(v)}
    prices_json = json.dumps(prices_dict)

    seed_holdings = []
    if os.path.exists(seed_holdings_file):
        try:
            sdf = pd.read_csv(seed_holdings_file)
            seed_holdings = sdf.to_dict(orient="records")
        except Exception:
            pass
    seed_holdings_json = json.dumps(seed_holdings)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>S&P 500 Portfolio Tracker</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #151b2b;
      --border: #242f48;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --green: #10b981;
      --red: #ef4444;
      --yellow: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    header {{ display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }}
    h1 {{ font-size: 1.6rem; color: #fff; font-weight: 700; }}
    .nav-links a {{ color: var(--accent); text-decoration: none; margin-left: 18px; font-weight: 500; }}
    .btn {{ background: #2563eb; color: #fff; border: none; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; }}
    .btn:hover {{ background: #1d4ed8; }}
    .btn-danger {{ background: #dc2626; }}
    .btn-danger:hover {{ background: #b91c1c; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }}
    .card-title {{ font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .card-val {{ font-size: 1.6rem; font-weight: 700; color: #fff; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: var(--card-bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }}
    th, td {{ padding: 12px 14px; text-align: left; font-size: 0.9rem; }}
    th {{ background: #1c2438; color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
    tr:not(:last-child) td {{ border-bottom: 1px solid var(--border); }}
    input.editable {{ background: #0b0f19; border: 1px solid var(--border); border-radius: 4px; color: #fff; padding: 4px 8px; width: 90px; font-size: 0.9rem; }}
    .status-ok {{ color: var(--green); font-weight: 600; }}
    .status-alert {{ color: var(--red); font-weight: 700; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>💼 S&P 500 Portfolio Monitor</h1>
        <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 4px;">Live Positions, P&amp;L Tracking &amp; Quarterly (3-Month) Rebalance</p>
      </div>
      <div class="nav-links">
        <button class="btn" onclick="addHoldingRow()">+ Add Holding</button>
        <button class="btn btn-danger" onclick="resetToDefault()">Reset Defaults</button>
        <a href="advisor.html">🎯 Momentum Advisor</a>
        <a href="backtest.html">📊 Backtest Lab →</a>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="card-title">Total Portfolio Value</div>
        <div class="card-val" id="totalValue">$0.00</div>
      </div>
      <div class="card">
        <div class="card-title">Total Invested</div>
        <div class="card-val" id="totalInvested">$0.00</div>
      </div>
      <div class="card">
        <div class="card-title">Open P&amp;L ($)</div>
        <div class="card-val" id="totalProfitLoss">$0.00</div>
      </div>
      <div class="card">
        <div class="card-title">Open Return (%)</div>
        <div class="card-val" id="totalReturnPct">0.00%</div>
      </div>
    </div>

    <h2 style="font-size: 1.2rem; margin-bottom: 8px; color: #fff;">📊 Active Holdings</h2>
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Quantity</th>
          <th>Avg Buy Price ($)</th>
          <th>Current Price ($)</th>
          <th>Invested ($)</th>
          <th>Current Value ($)</th>
          <th>P&amp;L ($)</th>
          <th>Return (%)</th>
          <th>Holding Horizon</th>
          <th>Status</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody id="holdingsBody">
        <!-- Injected via JS -->
      </tbody>
    </table>
  </div>

  <script>
    const prices = {prices_json};
    const defaultSeed = {seed_holdings_json};
    const STORAGE_KEY = "sp500_momentum_holdings";

    function loadHoldings() {{
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {{
        try {{ return JSON.parse(raw); }} catch(e) {{}}
      }}
      return defaultSeed;
    }}

    function saveHoldings(data) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    }}

    function renderTable() {{
      const holdings = loadHoldings();
      const tbody = document.getElementById("holdingsBody");
      tbody.innerHTML = "";

      let totalInv = 0;
      let totalVal = 0;

      holdings.forEach((h, idx) => {{
        const ticker = (h.ticker || "").toUpperCase();
        const qty = parseFloat(h.quantity) || 0;
        const buyPrice = parseFloat(h.buy_price) || 0;
        const curPrice = prices[ticker] || buyPrice;

        const inv = qty * buyPrice;
        const val = qty * curPrice;
        const pl = val - inv;
        const plPct = inv > 0 ? (pl / inv) * 100 : 0;

        totalInv += inv;
        totalVal += val;

        const plColor = pl >= 0 ? "var(--green)" : "var(--red)";

        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><input class="editable" value="${{ticker}}" onchange="updateHolding(${{idx}}, 'ticker', this.value)" style="width: 75px; font-weight: bold; color: var(--accent);" /></td>
          <td><input class="editable" type="number" step="0.0001" value="${{qty}}" onchange="updateHolding(${{idx}}, 'quantity', this.value)" style="width: 85px;" /></td>
          <td><input class="editable" type="number" step="0.01" value="${{buyPrice}}" onchange="updateHolding(${{idx}}, 'buy_price', this.value)" /></td>
          <td>$${{curPrice.toFixed(2)}}</td>
          <td>$${{inv.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}}</td>
          <td>$${{val.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}}</td>
          <td style="color: ${{plColor}}; font-weight: 600;">${{pl >= 0 ? '+' : ''}}$${{pl.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}}</td>
          <td style="color: ${{plColor}}; font-weight: 600;">${{plPct >= 0 ? '+' : ''}}$${{plPct.toFixed(2)}}%</td>
          <td style="color: var(--accent); font-weight: 500;">3M Quarterly</td>
          <td><span class="status-ok">HOLDING</span></td>
          <td><button class="btn btn-danger" style="padding: 4px 8px;" onclick="deleteHolding(${{idx}})">✕</button></td>
        `;
        tbody.appendChild(tr);
      }});
        tbody.appendChild(tr);
      }});

      const totalPL = totalVal - totalInv;
      const totalRet = totalInv > 0 ? (totalPL / totalInv) * 100 : 0;

      document.getElementById("totalValue").innerText = "$" + totalVal.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
      document.getElementById("totalInvested").innerText = "$" + totalInv.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});

      const plEl = document.getElementById("totalProfitLoss");
      plEl.innerText = (totalPL >= 0 ? "+" : "") + "$" + totalPL.toLocaleString(undefined, {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
      plEl.style.color = totalPL >= 0 ? "var(--green)" : "var(--red)";

      const retEl = document.getElementById("totalReturnPct");
      retEl.innerText = (totalRet >= 0 ? "+" : "") + totalRet.toFixed(2) + "%";
      retEl.style.color = totalRet >= 0 ? "var(--green)" : "var(--red)";
    }}

    function updateHolding(idx, field, val) {{
      const holdings = loadHoldings();
      if (field === 'quantity' || field === 'buy_price') {{
        holdings[idx][field] = parseFloat(val) || 0;
      }} else {{
        holdings[idx][field] = val.trim().toUpperCase();
      }}
      saveHoldings(holdings);
      renderTable();
    }}

    function addHoldingRow() {{
      const holdings = loadHoldings();
      holdings.push({{ ticker: "NVDA", quantity: 10, buy_price: 120.0, buy_date: new Date().toISOString().slice(0,10) }});
      saveHoldings(holdings);
      renderTable();
    }}

    function deleteHolding(idx) {{
      const holdings = loadHoldings();
      holdings.splice(idx, 1);
      saveHoldings(holdings);
      renderTable();
    }}

    function resetToDefault() {{
      if (confirm("Reset to default seed holdings?")) {{
        saveHoldings(defaultSeed);
        renderTable();
      }}
    }}

    window.addEventListener("DOMContentLoaded", renderTable);
  </script>
</body>
</html>
"""
    return html


_cached_engine = None


def get_cached_engine():
    global _cached_engine
    if _cached_engine is None:
        prices, _ = load_price_data()
        _cached_engine = MomentumEngine(prices)
    return _cached_engine


def run_server(port: int = DEFAULT_PORT, max_stocks: int = 6, no_browser: bool = False):
    """Start local threaded HTTP server with live re-scan and backtest endpoints."""
    class AdvisorRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            parsed_url = urlparse(self.path)

            if parsed_url.path == "/api/backtest":
                query = parse_qs(parsed_url.query)
                model = query.get("model", ["A"])[0]
                max_s = int(query.get("max_stocks", [6])[0])
                j = int(query.get("j", [12])[0])
                k = int(query.get("k", [3])[0])
                start = query.get("start", ["2016-01-01"])[0]
                end = query.get("end", ["2026-09-01"])[0]
                capital = float(query.get("capital", [100000.0])[0])
                use_sma = query.get("use_sma", ["true"])[0].lower() in ["true", "1", "yes"]
                long_short = query.get("long_short", ["false"])[0].lower() in ["true", "1", "yes"]

                print(f"Running simulation: Model {model}, Top {max_s}, J={j}, K={k}, {start} to {end}...")
                try:
                    engine = get_cached_engine()
                    metrics, results_df, _ = engine.run_backtest(
                        start_date=start,
                        end_date=end,
                        j=j,
                        k=k,
                        max_stocks=max_s,
                        use_sma=use_sma,
                        model=model,
                        long_short=long_short,
                        initial_capital=capital,
                    )
                    annual_df = engine.get_annual_returns(results_df)

                    curve_data = []
                    for dt, row in results_df.iterrows():
                        curve_data.append({
                            "date": dt.strftime("%Y-%m"),
                            "equity": round(float(row["Equity"]), 2),
                            "bench_equity": round(float(row["Benchmark_Equity"]), 2),
                            "drawdown": round(float(row["Strategy_Drawdown"]), 4),
                            "bench_drawdown": round(float(row["Benchmark_Drawdown"]), 4),
                        })

                    response_payload = {
                        "status": "ok",
                        "metrics": {
                            "final_value": metrics.final_value,
                            "total_return_pct": metrics.total_return_pct,
                            "cagr_pct": metrics.cagr_pct,
                            "annual_volatility_pct": metrics.annual_volatility_pct,
                            "sharpe_ratio": metrics.sharpe_ratio,
                            "sortino_ratio": metrics.sortino_ratio,
                            "max_drawdown_pct": metrics.max_drawdown_pct,
                            "calmar_ratio": metrics.calmar_ratio,
                            "benchmark_cagr_pct": metrics.benchmark_cagr_pct,
                            "benchmark_volatility_pct": metrics.benchmark_volatility_pct,
                            "benchmark_sharpe": metrics.benchmark_sharpe,
                            "benchmark_max_drawdown_pct": metrics.benchmark_max_drawdown_pct,
                            "annual_alpha_pct": metrics.annual_alpha_pct,
                            "beta": metrics.beta,
                            "alpha_tstat": metrics.alpha_tstat,
                            "r_squared": metrics.r_squared,
                        },
                        "annual_returns": annual_df.reset_index().to_dict(orient="records"),
                        "equity_curve": curve_data,
                    }

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response_payload).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                return

            if parsed_url.path == "/api/scan":
                print("Triggered live scan...")
                try:
                    targets, latest_prices, _ = get_current_momentum_targets(max_stocks=max_stocks, force_refresh=True)
                    adv_html = build_advisor_html(targets, cash=60000.0)
                    port_html = build_portfolio_html(latest_prices)
                    base_dir = Path(__file__).parent
                    (base_dir / "advisor.html").write_text(adv_html, encoding="utf-8")
                    (base_dir / "portfolio.html").write_text(port_html, encoding="utf-8")

                    global _cached_engine
                    _cached_engine = None

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "targets_count": len(targets)}).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
                return
            super().do_GET()

    # Enable address reuse to avoid TIME_WAIT conflicts
    socketserver.ThreadingTCPServer.allow_reuse_address = True

    current_port = port
    max_attempts = 10
    httpd = None

    for attempt in range(max_attempts):
        try:
            httpd = socketserver.ThreadingTCPServer(("", current_port), AdvisorRequestHandler)
            break
        except OSError as e:
            if attempt < max_attempts - 1:
                current_port += 1
            else:
                raise RuntimeError(f"Could not bind to any port in range {port} - {current_port}: {e}")

    advisor_url = f"http://localhost:{current_port}/advisor.html"
    portfolio_url = f"http://localhost:{current_port}/portfolio.html"

    print(f"\n" + "=" * 65)
    print(f"  S&P 500 Momentum Advisor Server running on port {current_port}")
    print(f"  Open in browser: {advisor_url}")
    print(f"  Portfolio monitor: {portfolio_url}")
    print(f"=" * 65 + "\n")
    print("Press Ctrl+C to stop the server.\n")

    # Automatically launch web browser after server starts
    if not no_browser:
        def open_in_browser():
            time.sleep(0.8)
            print(f"Opening {advisor_url} in your default browser...")
            webbrowser.open(advisor_url)

        threading.Thread(target=open_in_browser, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


def print_advisory_guide(targets_df: pd.DataFrame, cash: float, stoploss_pct: float = 0.0):
    """Print clean terminal execution instructions for momentum portfolio."""
    n_stocks = len(targets_df)
    cash_per_stock = cash / n_stocks if n_stocks > 0 else 0

    print("\n" + "=" * 82)
    print(f"        S&P 500 MOMENTUM ADVISOR: TOP {n_stocks} STOCKS (MODEL A - JT 1993)")
    print("=" * 82)
    print(f"Strategy:            Jegadeesh & Titman 12-minus-1 Momentum + 200-Day SMA Filter")
    print(f"Target Holdings:     Top {n_stocks} Momentum Winners (Target: 1/{n_stocks} = {100/n_stocks:.1f}% per stock)")
    print(f"Total New Capital:   ${cash:,.2f}  (${cash_per_stock:,.2f} per stock)")
    print(f"Rebalance Horizon:   Quarterly (3 Months) — Pure holding period, no intra-quarter stop loss")
    print("-" * 82)
    print("METHOD 1: FRACTIONAL SHARES (RECOMMENDED FOR US RETAIL & $5,000 BUDGET)")
    print("Most US brokers (IBKR, Robinhood, Schwab, Fidelity, Webull) support dollar-based trades.")
    print("-" * 82)
    print(f"{'Rank':<5} {'Ticker':<7} {'Security':<20} {'Price ($)':<10} {'Mom 12M':<10} {'Exact Shares':<14} {'Target Value':<13} {'200-Day SMA'}")
    print("-" * 82)

    for _, row in targets_df.iterrows():
        p = row["Current_Price"]
        sma = row.get("SMA_200")
        frac_shs = (cash_per_stock / p) if p > 0 else 0.0
        sma_str = f"${sma:.2f}" if pd.notna(sma) and sma else "N/A"
        print(f"#{row['Rank']:<4} {row['Ticker']:<7} {row['Security'][:18]:<20} ${p:<9.2f} {row['Momentum_12M_Pct']:>+7.1f}%   {frac_shs:>8.4f} shs    ${cash_per_stock:<12.2f} {sma_str}")

    print("-" * 82)
    print(f"100% Fully Invested: ${cash:,.2f} across all {n_stocks} stocks ($0 unallocated cash)")
    print("=" * 82)

    # Check if whole share math leaves substantial cash
    total_whole_cost = 0.0
    whole_alloc = []
    for _, row in targets_df.iterrows():
        p = row["Current_Price"]
        shs = int(cash_per_stock // p) if p > 0 else 0
        cost = shs * p
        total_whole_cost += cost
        whole_alloc.append((row, shs, cost))

    leftover = cash - total_whole_cost
    if leftover > 0.15 * cash:
        print("\n" + "·" * 82)
        print("METHOD 2: WHOLE SHARES ALTERNATIVE (IF YOUR BROKER DOES NOT ALLOW FRACTIONAL)")
        print(f"Note: Some high-priced winners trade above ${cash_per_stock:.0f}; whole shares leave")
        print(f"${leftover:,.2f} unallocated unless using budget reallocation or fractional shares.")
        print("·" * 82)

    print("\n" + "═" * 80)
    print("                    📅 WHEN TO REBALANCE (SCHEDULE)")
    print("═" * 80)
    print("QUARTERLY REBALANCE (3-Month Pure Holding Period):")
    print("  • Frequency: Once every 3 months (strictly aligns with K=3 holding period).")
    print("  • Rebalance Dates: First trading day of January, April, July, and October.")
    print("  • Holding Policy: No intra-quarter stop-loss exits. Positions are held through")
    print("    short-term noise to let momentum risk premiums develop.")
    print("  • Action on Rebalance Day: Re-run this advisor (`./run_advisor.sh` or web app).")
    print("═" * 80)

    print("\n" + "═" * 80)
    print("                    🛠️ HOW TO REBALANCE (STEP-BY-STEP)")
    print("═" * 80)
    print("STEP 1: Check Exits at 3-Month Quarter End (SELL)")
    print("  • Drop-Out Exit: If a stock you own is no longer in the Top rankings at quarter end, SELL.")
    print("  • Trend Breakdown Exit: If a stock has fallen below its 200-day SMA at quarter end, SELL.")
    print("\nSTEP 2: Check Existing Winners (HOLD / TOP-UP ONLY)")
    print("  • Golden Rule: NEVER trim a winner! If a stock remains in the Top ranks, DO NOT sell")
    print("    shares even if it has grown to be 20% or 30% of your portfolio.")
    print("  • Let your top performers compound.")
    print("\nSTEP 3: Deploy Capital into New Winners (BUY)")
    print("  • Sum up proceeds from exited positions + any new cash added.")
    print("  • Divide cash equally among the open target slots.")
    print("  • Buy shares as shown in the table above.")
    print("  • Any leftover cash carries forward into the cash reserve.")
    print("\nSTEP 4: Hold for Full 3 Months")
    print("  • Hold positions until the next quarterly rebalance date.")
    print("═" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate S&P 500 Momentum Advisor HTML reports.")
    parser.add_argument("--cash", type=float, default=60000.0, help="Available cash for new allocations ($ USD, default: $60,000)")
    parser.add_argument("--max-stocks", type=int, default=10, help="Target number of stocks (default: 10)")
    parser.add_argument("--stoploss", type=float, default=0.0, help="Optional stop loss threshold (default: 0.0 = disabled)")
    parser.add_argument("--serve", action="store_true", help="Start local web server on port 8770")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser on start")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for local server")

    args = parser.parse_args()

    print(f"\nEvaluating latest S&P 500 momentum data for Top {args.max_stocks} stocks...")
    targets_df, latest_prices, bench_price = get_current_momentum_targets(max_stocks=args.max_stocks)

    advisor_html = build_advisor_html(targets_df, cash=args.cash, stoploss_pct=args.stoploss)
    portfolio_html = build_portfolio_html(latest_prices, stoploss_pct=args.stoploss)

    adv_path = Path(__file__).parent / "advisor.html"
    port_path = Path(__file__).parent / "portfolio.html"

    adv_path.write_text(advisor_html, encoding="utf-8")
    port_path.write_text(portfolio_html, encoding="utf-8")

    # Print rich advisory guidance
    print_advisory_guide(targets_df, cash=args.cash, stoploss_pct=args.stoploss)

    print(f"Generated Interactive Dashboards:")
    print(f"  • Advisor HTML:   file://{adv_path.resolve()}")
    print(f"  • Portfolio HTML: file://{port_path.resolve()}")

    if args.serve:
        run_server(args.port, max_stocks=args.max_stocks, no_browser=args.no_browser)


if __name__ == "__main__":
    main()

