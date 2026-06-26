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

# Ask a yes/no question; default Yes. Returns 0 for yes, 1 for no.
# If there's no interactive terminal, don't hang — treat it as "no".
confirm() {
  local prompt="$1" answer
  if [ ! -t 0 ]; then
    return 1
  fi
  read -r -p "$prompt [Y/n] " answer || return 1
  case "${answer:-Y}" in
    [Yy]*|"") return 0 ;;
    *) return 1 ;;
  esac
}

# Install Debian/Ubuntu packages, using sudo only if we're not already root.
apt_install() {
  if [ "$(id -u)" -eq 0 ]; then
    apt-get update && apt-get install -y "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y "$@"
  else
    echo "ERROR: need root (or sudo) to install: $*" >&2
    return 1
  fi
}

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

# Debian/OMV often ship venv + pip as separate apt packages. If they're
# missing, offer to install them rather than just failing.
if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
  echo
  echo "Python's virtual-environment support (the 'python3-venv' and 'python3-pip'"
  echo "packages) isn't installed — this app needs it."
  if command -v apt-get >/dev/null 2>&1 && confirm "Install them now with apt?"; then
    apt_install python3-venv python3-pip || true   # fall through to the re-check
  fi
  # Re-check, whether the user declined or the install didn't take.
  if ! "$PY" -c 'import venv, ensurepip' 2>/dev/null; then
    echo "ERROR: venv/pip is still unavailable." >&2
    echo "       Install it manually, then re-run ./setup.sh :" >&2
    echo "           sudo apt install python3-venv python3-pip" >&2
    exit 1
  fi
fi

if [ ! -d venv ]; then
  echo "==> Creating virtualenv in ./venv"
  "$PY" -m venv venv
fi

echo "==> Installing dependencies (this can take a few minutes the first time)"
./venv/bin/python -m pip install --upgrade pip >/dev/null
./venv/bin/python -m pip install -r requirements.txt

# Optional: play the chime on this machine's own speakers (vs. only a Google Home).
if confirm "Install local PC audio output (play the chime on this machine's speakers)?"; then
  ./venv/bin/python -m pip install playsound3 \
    || echo "    (playsound3 install failed — the 'This PC (local audio)' option will be unavailable)"
fi

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
