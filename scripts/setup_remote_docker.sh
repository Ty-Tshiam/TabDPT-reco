#!/usr/bin/env bash
# ==============================================================================
# setup_remote_docker.sh: Install Docker, Compose & NVIDIA Container Toolkit
# ==============================================================================

set -e

echo "=== [Setup] Checking & Configuring Docker Environment ==="

# 1. Install prerequisites
apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates gnupg lsb-release >/dev/null 2>&1 || true

# 2. Install Docker if not present
if ! command -v docker >/dev/null 2>&1; then
    echo "[Setup] Docker not detected. Installing official Docker..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sh /tmp/get-docker.sh >/dev/null 2>&1
    rm -f /tmp/get-docker.sh
    echo "[OK] Docker successfully installed."
else
    echo "[OK] Docker is already installed: $(docker --version)"
fi

# 3. Ensure Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "[Setup] Starting Docker daemon..."
    service docker start 2>/dev/null || systemctl start docker 2>/dev/null || (dockerd >/tmp/dockerd.log 2>&1 & sleep 3)
fi

# 4. Ensure Docker Compose plugin is present
if ! docker compose version >/dev/null 2>&1; then
    echo "[Setup] Installing Docker Compose plugin..."
    apt-get update -qq && apt-get install -y -qq --no-install-recommends docker-compose-plugin >/dev/null 2>&1 || true
    echo "[OK] Docker Compose plugin installed."
else
    echo "[OK] Docker Compose is ready: $(docker compose version)"
fi

# 5. Install & configure NVIDIA Container Toolkit for GPU passthrough
if ! command -v nvidia-ctk >/dev/null 2>&1; then
    echo "[Setup] Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || true
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null || true
    apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit >/dev/null 2>&1 || true
fi

if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "[Setup] Registering NVIDIA runtime with Docker..."
    nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
    service docker restart 2>/dev/null || systemctl restart docker 2>/dev/null || true
    echo "[OK] NVIDIA Container Runtime configured."
fi

echo "=== [Setup] Docker & GPU Runtime Setup Complete ==="
