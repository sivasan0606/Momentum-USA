#!/usr/bin/env bash
# ==============================================================================
# S&P 500 Momentum Advisor Runner (Model A - Top 6 Stocks)
# Usage:
#   ./run_advisor.sh                  # Terminal advice + HTML reports ($60,000 cash)
#   ./run_advisor.sh --cash 100000    # Run with custom cash amount
#   ./run_advisor.sh --serve          # Start local web app on http://localhost:8770
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

python3 advisor.py "$@"
