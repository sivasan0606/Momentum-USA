#!/usr/bin/env bash
# ==============================================================================
# Double-clickable script to stop the background S&P 500 Momentum Server.
# ==============================================================================

PORT=8770
PIDS=$(lsof -ti :${PORT} 2>/dev/null || true)

if [ -n "$PIDS" ]; then
    echo "Stopping S&P 500 Momentum Server on port ${PORT} (PID: $PIDS)..."
    kill -9 $PIDS
    echo "Server stopped successfully."
else
    echo "No server running on port ${PORT}."
fi

sleep 1
osascript -e 'tell application "Terminal" to close (every window whose name contains "Stop_Momentum_USA")' &>/dev/null || true
