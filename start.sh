#!/usr/bin/env bash
# SPIOMetrologyViewer on Linux: get the latest version from GitHub, install what it needs, start
# the GUI. Same steps as start.bat on Windows. ./start.sh --console keeps the GUI in this terminal
# (to see error messages).
cd "$(dirname "$(readlink -f "$0")")" || exit 1

fail() {
    echo "*** $*"
    read -r -p "Press Enter to close." _ </dev/tty 2>/dev/null
    exit 1
}

# 1. Update
if [ ! -d .git ]; then
    echo "Not a git clone: can't check for updates. See README, Getting started."
elif ! command -v git >/dev/null; then
    echo "Git is not installed: can't check for updates (sudo apt install git)."
else
    echo "Checking for updates..."
    git pull --ff-only || { echo "*** Could not update (offline, or files changed locally). Starting the current version."; sleep 5; }
fi

# uv (https://docs.astral.sh/uv) brings its own Python 3.14 with tkinter, no sudo needed
UV=$(command -v uv || { [ -x "$HOME/.local/bin/uv" ] && echo "$HOME/.local/bin/uv"; })

# 2. First start: a private Python environment in .venv
if [ ! -x .venv/bin/python ]; then
    echo "First start: setting up Python packages, this takes a few minutes..."
    if [ -n "$UV" ]; then
        "$UV" venv --python 3.14 .venv
    else
        python3 -m venv .venv
    fi || {
        rm -rf .venv
        fail "Could not set up Python. Install uv and run this again:
    curl -LsSf https://astral.sh/uv/install.sh | sh
(or: sudo apt install python3-venv python3-tk)"
    }
fi

# 3. Packages, only when requirements.txt changed since the last install
if ! cmp -s requirements.txt .venv/requirements.installed; then
    if [ -n "$UV" ]; then
        "$UV" pip install --python .venv/bin/python -r requirements.txt
    else
        .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
    fi || fail "Installing the packages failed, see above."
    cp requirements.txt .venv/requirements.installed
fi

.venv/bin/python -c "import tkinter" 2>/dev/null ||
    fail "Python has no tkinter (the GUI toolkit): sudo apt install python3-tk, or install uv, delete .venv and run this again."

# 4. Start
if [ "$1" = "--console" ]; then
    exec .venv/bin/python -m spiometrologyviewer.gui
fi
nohup .venv/bin/python -m spiometrologyviewer.gui >/dev/null 2>&1 &
