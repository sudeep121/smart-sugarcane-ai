#!/usr/bin/env bash
#
# Prepare a fresh Oracle Cloud Always Free instance to run Smart Sugarcane AI.
#
#   curl -fsSL https://raw.githubusercontent.com/<you>/smart-sugarcane-ai/main/deploy/oracle/setup.sh | bash
# or, after cloning:
#   bash deploy/oracle/setup.sh
#
# Installs Docker, opens the firewall and builds the stack. Two things here
# catch nearly everyone deploying to Oracle for the first time:
#
#  1. Oracle's Ubuntu images ship a restrictive iptables ruleset that drops
#     inbound 80 and 443 even after the port is opened in the cloud console.
#     Opening the Security List alone is not enough; both layers must allow it.
#
#  2. Ampere A1 capacity is frequently exhausted in popular regions, so the
#     instance may fail to create with "Out of host capacity" before any of
#     this runs. That is a console problem, not a script problem.
#
# Safe to re-run: every step checks its own state first.

set -euo pipefail

GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'; RED=$'\033[0;31m'; NC=$'\033[0m'
say()  { printf '%s==>%s %s\n' "$GREEN" "$NC" "$1"; }
warn() { printf '%s!! %s %s\n' "$YELLOW" "$NC" "$1"; }
die()  { printf '%sXX %s %s\n' "$RED" "$NC" "$1" >&2; exit 1; }

[[ $EUID -eq 0 ]] && die "Run as the ubuntu user, not root. The script uses sudo where it needs to."

REPO_DIR="${REPO_DIR:-$HOME/smart-sugarcane-ai}"
COMPOSE_DIR="$REPO_DIR/deploy/oracle"

# --------------------------------------------------------------- 1. packages
say "Updating packages"
sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl git jq

# ----------------------------------------------------------------- 2. Docker
if command -v docker >/dev/null 2>&1; then
  say "Docker already installed ($(docker --version))"
else
  say "Installing Docker"
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
  warn "Added $USER to the docker group. Log out and back in, or the rest of"
  warn "this script will keep using sudo for docker."
fi

DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# --------------------------------------------------------------- 3. firewall
# The cloud Security List is configured in the OCI console; this handles the
# instance-local rules, which are what silently drop traffic afterwards.
say "Opening ports 80 and 443 on the instance firewall"
if command -v netfilter-persistent >/dev/null 2>&1 || [[ -f /etc/iptables/rules.v4 ]]; then
  for port in 80 443; do
    if ! sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
      # Insert above Oracle's catch-all REJECT rather than appending after it.
      sudo iptables -I INPUT 6 -p tcp --dport "$port" -m state --state NEW,ESTABLISHED -j ACCEPT
      say "  opened $port"
    else
      say "  $port already open"
    fi
  done
  sudo netfilter-persistent save >/dev/null 2>&1 || sudo sh -c 'iptables-save > /etc/iptables/rules.v4'
else
  warn "iptables-persistent not found; installing so the rules survive a reboot"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq iptables-persistent
fi

if command -v firewall-cmd >/dev/null 2>&1; then
  sudo firewall-cmd --permanent --add-service=http  >/dev/null 2>&1 || true
  sudo firewall-cmd --permanent --add-service=https >/dev/null 2>&1 || true
  sudo firewall-cmd --reload >/dev/null 2>&1 || true
fi

# ------------------------------------------------------------------ 4. swap
# Building TensorFlow's dependency tree peaks well above steady-state usage.
# 12 GB is comfortable, but swap costs nothing and prevents an OOM kill during
# the build on a smaller shape.
if ! sudo swapon --show | grep -q .; then
  say "Adding 2 GB swap"
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
  say "Swap already configured"
fi

# --------------------------------------------------------------- 5. the app
[[ -d "$COMPOSE_DIR" ]] || die "Expected the repository at $REPO_DIR. Clone it there first, or set REPO_DIR."
cd "$COMPOSE_DIR"

if [[ ! -f .env ]]; then
  cp .env.example .env
  SECRET=$(openssl rand -hex 32)
  sed -i "s|^SECRET_KEY=.*|SECRET_KEY=$SECRET|" .env
  warn "Created .env with a generated SECRET_KEY."
  warn "Edit it now and set DOMAIN to your DuckDNS hostname, then re-run this script."
  exit 0
fi

DOMAIN=$(grep -E '^DOMAIN=' .env | cut -d= -f2-)
[[ -z "$DOMAIN" || "$DOMAIN" == "yourname.duckdns.org" ]] && die "Set DOMAIN in $COMPOSE_DIR/.env first."

say "Checking that $DOMAIN points at this machine"
PUBLIC_IP=$(curl -fsS --max-time 10 https://api.ipify.org || echo "")
RESOLVED=$(getent hosts "$DOMAIN" | awk '{print $1}' | head -1 || echo "")
if [[ -n "$PUBLIC_IP" && -n "$RESOLVED" && "$PUBLIC_IP" != "$RESOLVED" ]]; then
  warn "$DOMAIN resolves to $RESOLVED but this VM is $PUBLIC_IP."
  warn "Let's Encrypt will refuse the certificate until DNS is correct."
fi

say "Building and starting (first build downloads TensorFlow - expect 15-25 min on 2 ARM cores)"
$DOCKER compose up -d --build

say "Waiting for the application to report healthy"
for _ in $(seq 1 60); do
  if $DOCKER compose exec -T app curl -fsS http://127.0.0.1:7860/api/system/health >/dev/null 2>&1; then
    say "Application is up"
    break
  fi
  sleep 10
done

echo
say "Done. Your app should be live at: https://$DOMAIN"
echo
echo "  logs      : $DOCKER compose -f $COMPOSE_DIR/docker-compose.yml logs -f"
echo "  restart   : $DOCKER compose -f $COMPOSE_DIR/docker-compose.yml restart"
echo "  update    : git -C $REPO_DIR pull && $DOCKER compose -f $COMPOSE_DIR/docker-compose.yml up -d --build"
echo
warn "Oracle reclaims Always Free instances that stay idle. Keeping the app"
warn "reachable and occasionally used is usually enough; a weekly visit is not."
