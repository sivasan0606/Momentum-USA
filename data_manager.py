"""
Data Manager for S&P 500 Momentum Engine.
Fetches S&P 500 constituents, downloads historical daily prices,
and caches everything locally in Parquet format.
"""

import io
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"
CONSTITUENTS_CSV = DATA_DIR / "sp500_constituents.csv"
PRICES_PARQUET = DATA_DIR / "sp500_daily_prices.parquet"
BENCHMARK_TICKER = "SPY"


def get_sp500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch S&P 500 constituents from Wikipedia or local cache.
    Returns DataFrame with ['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry', 'Yahoo_Symbol'].
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not force_refresh and CONSTITUENTS_CSV.exists():
        return pd.read_csv(CONSTITUENTS_CSV)

    print("Fetching S&P 500 constituent list from Wikipedia...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0].copy()

    # Standardize column names
    col_map = {
        "Symbol": "Symbol",
        "Security": "Security",
        "GICS Sector": "GICS Sector",
        "GICS Sub-Industry": "GICS Sub-Industry",
    }
    for col in df.columns:
        for k in col_map:
            if k.lower() in str(col).lower():
                df.rename(columns={col: k}, inplace=True)
                break

    # Fix ticker symbols for Yahoo Finance (e.g., BRK.B -> BRK-B)
    df["Yahoo_Symbol"] = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
    cols_to_keep = [c for c in ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "Yahoo_Symbol"] if c in df.columns]
    df = df[cols_to_keep]

    df.to_csv(CONSTITUENTS_CSV, index=False)
    print(f"Saved {len(df)} S&P 500 constituents to {CONSTITUENTS_CSV}")
    return df


def download_sp500_daily_prices(
    start_date: str = "2014-12-01",
    end_date: Optional[str] = None,
    force_refresh: bool = False,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices for all S&P 500 stocks + benchmark (SPY).
    Stores and loads from data/sp500_daily_prices.parquet.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if end_date is None:
        end_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    constituents = get_sp500_constituents(force_refresh=force_refresh)
    tickers = constituents["Yahoo_Symbol"].unique().tolist()
    if BENCHMARK_TICKER not in tickers:
        tickers.append(BENCHMARK_TICKER)

    existing_df = None
    if PRICES_PARQUET.exists():
        try:
            existing_df = pd.read_parquet(PRICES_PARQUET)
        except Exception:
            existing_df = None

    if not force_refresh and existing_df is not None and BENCHMARK_TICKER in existing_df.columns:
        missing_tickers = [t for t in tickers if t not in existing_df.columns]
        # Only reuse cache if it has essentially the complete universe (missing <= 15 tickers)
        if len(missing_tickers) <= 15:
            print(f"Loading cached prices from {PRICES_PARQUET}...")
            print(f"Loaded price matrix with shape {existing_df.shape} ({existing_df.index[0].date()} to {existing_df.index[-1].date()})")
            return existing_df
        print(f"Cached price matrix only has {existing_df.shape[1]} tickers ({len(missing_tickers)} missing). Fetching all {len(tickers)} symbols...")

    print(f"Downloading historical daily data for {len(tickers)} symbols from {start_date} to {end_date}...")
    all_series = {}

    # Download in batches to avoid network timeouts and large payload errors
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    t0 = time.time()

    for idx in range(0, len(tickers), batch_size):
        batch = tickers[idx : idx + batch_size]
        batch_num = (idx // batch_size) + 1
        print(f"[{batch_num}/{total_batches}] Downloading {len(batch)} tickers ({batch[0]}..{batch[-1]})...")
        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,  # Dividend & split adjusted
                threads=True,
            )
            if data.empty:
                print(f"Warning: batch {batch_num} returned empty data.")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                # MultiIndex: (Price, Ticker)
                close_data = data["Close"]
            else:
                close_data = data[["Close"]]
                close_data.columns = batch

            for col in close_data.columns:
                series = close_data[col].dropna()
                if len(series) > 50:  # Must have some data
                    all_series[col] = series
        except Exception as e:
            print(f"Error downloading batch {batch_num}: {e}")

    if not all_series and existing_df is None:
        raise RuntimeError("No price data could be downloaded.")

    if all_series:
        new_df = pd.DataFrame(all_series)
        new_df.index = pd.to_datetime(new_df.index)
        new_df.sort_index(inplace=True)
        if existing_df is not None:
            # Combine: preserve all historical tickers and update with newly downloaded prices
            price_df = existing_df.combine_first(new_df)
            price_df.update(new_df)
        else:
            price_df = new_df
    else:
        price_df = existing_df

    # Forward fill small gaps (up to 5 days, e.g. holidays or delayed reports)
    price_df = price_df.ffill(limit=5)

    # Ensure SPY benchmark is always present
    if BENCHMARK_TICKER not in price_df.columns:
        print(f"Downloading benchmark {BENCHMARK_TICKER}...")
        try:
            spy_data = yf.download(BENCHMARK_TICKER, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(spy_data.columns, pd.MultiIndex):
                price_df[BENCHMARK_TICKER] = spy_data["Close"][BENCHMARK_TICKER]
            elif "Close" in spy_data:
                price_df[BENCHMARK_TICKER] = spy_data["Close"]
            price_df[BENCHMARK_TICKER] = price_df[BENCHMARK_TICKER].ffill(limit=5)
        except Exception as e:
            print(f"Warning: Failed to fetch {BENCHMARK_TICKER}: {e}")

    print(f"Consolidated price matrix: {price_df.shape[0]} trading days x {price_df.shape[1]} tickers.")
    price_df.to_parquet(PRICES_PARQUET)
    print(f"Saved cache to {PRICES_PARQUET} (elapsed: {round(time.time() - t0, 1)}s)")
    return price_df


def load_price_data(
    start_date: str = "2014-12-01",
    end_date: Optional[str] = None,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (prices_df, constituents_df).
    prices_df contains all S&P 500 daily prices + 'SPY'.
    """
    constituents = get_sp500_constituents(force_refresh=force_refresh)
    prices = download_sp500_daily_prices(
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
    )
    return prices, constituents


if __name__ == "__main__":
    prices, constituents = load_price_data()
    print("Sample tickers:", list(prices.columns[:10]))
    print(f"Benchmark {BENCHMARK_TICKER} present: {BENCHMARK_TICKER in prices.columns}")
    print("Latest available date:", prices.index[-1].strftime("%Y-%m-%d"))
