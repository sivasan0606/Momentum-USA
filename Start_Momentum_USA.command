#!/usr/bin/env bash
# ==============================================================================
# Double-clickable macOS launcher for S&P 500 Momentum Advisor.
# Starts the server in the background (if not already running)
# and opens the advisor in your default web browser automatically.
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

mkdir -p logs

PORT=8770
URL="http://localhost:${PORT}/advisor.html"

# Check if server is already running on port 8770
if ! lsof -i :${PORT} >/dev/null 2>&1; then
    echo "Starting S&P 500 Momentum Server in the background on port ${PORT}..."
    nohup python3 advisor.py --serve --port ${PORT} > logs/server.log 2>&1 &
    sleep 1.5
else
    echo "Server is already running on port ${PORT}."
fi

echo "Opening ${URL} in your browser..."
open "${URL}"

# If opened by double-click in Terminal, close terminal window cleanly
sleep 0.5
osascript -e 'tell application "Terminal" to close (every window whose name contains "Start_Momentum_USA")' &>/dev/null || true
