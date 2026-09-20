#!/usr/bin/env bash
# ==============================================================================
# Install S&P 500 Momentum Advisor as a persistent macOS LaunchAgent.
# The server will automatically start at login and run 24/7 in the background.
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PLIST_NAME="com.momentum.usa.advisor"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
PYTHON_BIN="$(which python3)"

LOG_DIR="$HOME/Library/Logs/MomentumUSA"
mkdir -p "$HOME/Library/LaunchAgents"
mkdir -p "$DIR/logs"
mkdir -p "$LOG_DIR"

if [ "$1" == "--uninstall" ]; then
    echo "Uninstalling ${PLIST_NAME}..."
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    rm -f "$PLIST_PATH"
    echo "LaunchAgent uninstalled successfully."
    exit 0
fi

echo "Creating ${PLIST_PATH}..."
cat <<EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_BIN}</string>
        <string>${DIR}/advisor.py</string>
        <string>--serve</string>
        <string>--port</string>
        <string>8770</string>
        <string>--no-browser</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/autostart.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/autostart.error.log</string>
</dict>
</plist>
EOF

chmod 644 "$PLIST_PATH"
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "====================================================================="
echo "  LaunchAgent ${PLIST_NAME} successfully installed!"
echo "  The S&P 500 Momentum Advisor is now running 24/7 in the background."
echo "  It will automatically start whenever you turn on or log in to your Mac."
echo "  URL: http://localhost:8770/advisor.html"
echo "====================================================================="
