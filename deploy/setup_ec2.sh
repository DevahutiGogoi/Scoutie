#!/usr/bin/env bash
# Run this ON the EC2 instance, after the code has been copied over.
# Assumes Amazon Linux 2023 (the default free-tier AMI). Installs Python deps into a venv and
# registers scoutie as a systemd service.

set -euo pipefail

# Named scoutie-app, not scoutie, deliberately -- the scoutie/ Python package also needs to live
# inside this directory (as scoutie-app/scoutie/), and reusing the same name for the repo root
# and the package caused a real rsync mix-up during the first deploy.
APP_DIR="$HOME/scoutie-app"
cd "$APP_DIR"

# Measured directly (see the deploy conversation): the catalog + its three retrieval indices sit
# at ~693MB RSS steady-state on a 1GB instance -- real but tight (roughly 150-300MB left for the
# OS/sshd/systemd/etc). A swap file is the standard, free mitigation: it won't make
# a genuinely oversized process fast, but it turns "the OOM killer kills Scoutie mid-demo" into
# "briefly slower," which is the right tradeoff for a low-traffic demo box. Skipped if swap
# already exists (e.g. re-running this script).
if ! swapon --show | grep -q .; then
  echo "== Adding a 1GB swap file (safety net for the ~693MB steady-state footprint) =="
  sudo dd if=/dev/zero of=/swapfile bs=1M count=1024 status=none
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo "/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab >/dev/null
fi

echo "== Installing system Python deps =="
sudo dnf install -y python3.11 python3.11-pip >/dev/null

echo "== Creating venv =="
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "== Installing systemd service =="
sudo cp deploy/scoutie.service /etc/systemd/system/scoutie.service
sudo systemctl daemon-reload
sudo systemctl enable scoutie
sudo systemctl restart scoutie

echo "== Done. Check status with: sudo systemctl status scoutie =="
echo "== Logs: sudo journalctl -u scoutie -f =="
