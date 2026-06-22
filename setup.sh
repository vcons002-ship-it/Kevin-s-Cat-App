#!/usr/bin/env bash
#
# Kevin's Cat App — one-shot setup.
#
#   git clone / download this repo, then:   ./setup.sh
#   then:                                   ./venv/bin/python run.py
#
# Creates a Python virtualenv, installs dependencies, generates the default
# treat chime, and creates config.yaml. No Docker, no other services.
set -euo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
echo "==> Using $($PY --version)"

# This app's dependencies (pychromecast 14, zeroconf) require Python >= 3.11.
# Fail early with a clear message instead of a confusing pip resolver error.
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "ERROR: Python 3.11 or newer is required (found $($PY --version 2>&1))." >&2
  echo "       OpenMediaVault 7 (Debian 12) ships Python 3.11. On older systems," >&2
  echo "       install a newer Python and re-run with:  PYTHON=python3.11 ./setup.sh" >&2
  exit 1
fi

# Debian/OMV often ship venv + pip as separate apt packages. Check up front.
if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
  echo "ERROR: Python's venv/pip support is missing." >&2
  echo "       On OpenMediaVault/Debian install it with:  sudo apt install python3-venv python3-pip" >&2
  exit 1
fi

if [ ! -d venv ]; then
  echo "==> Creating virtualenv in ./venv"
  "$PY" -m venv venv
fi

echo "==> Installing dependencies (this can take a few minutes the first time)"
./venv/bin/python -m pip install --upgrade pip >/dev/null
./venv/bin/python -m pip install -r requirements.txt

echo "==> Generating the default treat chime"
./venv/bin/python d20app/sounds/generate_chime.py

if [ ! -f config.yaml ]; then
  echo "==> Creating config.yaml from config.example.yaml"
  cp config.example.yaml config.yaml
else
  echo "==> Keeping existing config.yaml"
fi

# Detect a LAN IP just for the friendly hint below.
IP="$(./venv/bin/python - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("8.8.8.8", 80)); print(s.getsockname()[0])
except OSError:
    print("localhost")
finally:
    s.close()
PY
)"

cat <<EOF

============================================================
  Setup complete! ✅

  Start the app:
      ./venv/bin/python run.py

  Then open the GUI in a browser on the same WiFi:
      http://${IP}:8080

  In the GUI: pick your camera & speaker (auto-detected),
  choose a sound, set the dice/DC/interval, then Start.
============================================================
EOF
