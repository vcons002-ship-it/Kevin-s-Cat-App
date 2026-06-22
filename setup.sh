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
