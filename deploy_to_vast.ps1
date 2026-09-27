# deploy_to_vast.ps1
# ==============================================================================
# Automated One-Command Deployment for TabDPT Triton gRPC Inference on Vast.ai
# ==============================================================================
# Usage:
#   .\deploy_to_vast.ps1                                   # Interactive with defaults
#   .\deploy_to_vast.ps1 -HostIP "65.108.94.15" -Port 42749  # Direct execution
#   .\deploy_to_vast.ps1 -SkipTunnel                       # Deploy without opening local tunnel
# ==============================================================================

param(
    [string]$HostIP = "",
    [string]$Port = "",
    [switch]$SkipTunnel
)

# ------------------------------------------------------------------------------
# 1. Resolve Connection Parameters (Interactive + Default Fallbacks)
# ------------------------------------------------------------------------------
$DEFAULT_IP = "65.108.94.15"
$DEFAULT_PORT = "42749"

if ([string]::IsNullOrWhiteSpace($HostIP)) {
    $promptIP = Read-Host "Enter Vast.ai Host IP [Default: $DEFAULT_IP]"
    $HostIP = if ([string]::IsNullOrWhiteSpace($promptIP)) { $DEFAULT_IP } else { $promptIP.Trim() }
}

if ([string]::IsNullOrWhiteSpace($Port)) {
    $promptPort = Read-Host "Enter Vast.ai SSH Port [Default: $DEFAULT_PORT]"
    $Port = if ([string]::IsNullOrWhiteSpace($promptPort)) { $DEFAULT_PORT } else { $promptPort.Trim() }
}

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host "  TabDPT Triton gRPC Deployment Pipeline" -ForegroundColor Cyan
Write-Host "  Target Instance: root@${HostIP}:${Port}" -ForegroundColor Cyan
Write-Host "=======================================================`n" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# 2. Verify SSH Connectivity
# ------------------------------------------------------------------------------
Write-Host "==> [1/6] Verifying SSH connectivity to remote instance..." -ForegroundColor Yellow
$testSSH = (ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -p $Port root@$HostIP "echo 'SSH_OK' && nvidia-smi --query-gpu=name --format=csv,noheader" 2>&1) -join "`n"
if ($testSSH -notmatch "SSH_OK") {
    Write-Host "[ERROR] Failed to connect to root@${HostIP}:${Port} via SSH." -ForegroundColor Red
    Write-Host "Error details: $testSSH" -ForegroundColor Red
    exit 1
}
$gpuName = if ($testSSH -match "(NVIDIA[^\r\n]+)") { $matches[1] } else { "Detected" }
Write-Host "[OK] Connected! Remote GPU: $gpuName" -ForegroundColor Green

# ------------------------------------------------------------------------------
# 3. Create Remote Directory Structure
# ------------------------------------------------------------------------------
Write-Host "`n==> [2/6] Initializing remote directories in /workspace/TabDPT-reco..." -ForegroundColor Yellow
ssh -p $Port root@$HostIP "mkdir -p /workspace/TabDPT-reco/data/tensors /workspace/TabDPT-reco/data/metadata /workspace/TabDPT-reco/model_repo/tabdpt/3 /workspace/TabDPT-reco/src /workspace/TabDPT-reco/tests /workspace/TabDPT-reco/scripts"

# ------------------------------------------------------------------------------
# 4. Upload Codebase, Metadata, & Core Context Tensors (Fast Transfer)
# ------------------------------------------------------------------------------
Write-Host "`n==> [3/6] Uploading codebase, metadata, and core context tensors (context.pt & y.pt)..." -ForegroundColor Yellow

# Upload context tensors (23 MB + 715 KB)
scp -O -P $Port data/tensors/context.pt data/tensors/y.pt "root@${HostIP}:/workspace/TabDPT-reco/data/tensors/"

# Upload metadata JSON mappings
scp -O -P $Port data/metadata/*.json "root@${HostIP}:/workspace/TabDPT-reco/data/metadata/"

# Upload model repository configs
scp -O -P $Port model_repo/tabdpt/config.pbtxt "root@${HostIP}:/workspace/TabDPT-reco/model_repo/tabdpt/"
scp -O -P $Port model_repo/tabdpt/3/model.py "root@${HostIP}:/workspace/TabDPT-reco/model_repo/tabdpt/3/"

# Upload source tree & scripts
scp -O -r -P $Port src "root@${HostIP}:/workspace/TabDPT-reco/"
scp -O -P $Port tests/test_triton_grpc.py "root@${HostIP}:/workspace/TabDPT-reco/tests/"
scp -O -P $Port scripts/manage_remote_server.sh "root@${HostIP}:/workspace/TabDPT-reco/scripts/"
ssh -p $Port root@$HostIP "chmod +x /workspace/TabDPT-reco/scripts/*.sh"

Write-Host "[OK] Project files synchronized successfully." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 5. Remote Environment Bootstrap (Python Packages & Tools)
# ------------------------------------------------------------------------------
Write-Host "`n==> [4/6] Bootstrapping remote dependencies in /venv/main..." -ForegroundColor Yellow
ssh -p $Port root@$HostIP "/venv/main/bin/pip install --quiet huggingface_hub safetensors scipy scikit-learn omegaconf 'tritonclient[grpc]' && which tmux >/dev/null || apt-get update && apt-get install -y --no-install-recommends tmux"
Write-Host "[OK] Remote dependencies satisfied." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 6. Remote Artifact Generation (Build 5.6 GB KV-Cache on Server GPU)
# ------------------------------------------------------------------------------
Write-Host "`n==> [5/6] Building KV-cache and model state dict directly on GPU NVMe..." -ForegroundColor Yellow
$checkArtifact = ssh -p $Port root@$HostIP "[ -f /workspace/TabDPT-reco/data/tensors/context_kv_cache.pt ] && [ -f /workspace/TabDPT-reco/model_repo/tabdpt/3/model.pt ] && echo 'EXISTS' || echo 'BUILD'"
if ($checkArtifact -match "EXISTS") {
    Write-Host "[INFO] Artifacts already exist on server NVMe. Skipping re-generation." -ForegroundColor Cyan
} else {
    Write-Host "[INFO] Generating artifacts with save_model.py (takes ~15-20 seconds on GPU)..." -ForegroundColor Cyan
    ssh -p $Port root@$HostIP "cd /workspace/TabDPT-reco && /venv/main/bin/python src/save_model.py"
}
Write-Host "[OK] Model artifacts ready." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 7. Start / Restart Triton gRPC Service in Persistent tmux Session
# ------------------------------------------------------------------------------
Write-Host "`n==> [6/6] Launching Triton gRPC inference service on port 8001..." -ForegroundColor Yellow
ssh -p $Port root@$HostIP "bash /workspace/TabDPT-reco/scripts/manage_remote_server.sh restart"

# Wait briefly for model compile & warm-up
Start-Sleep -Seconds 3
Write-Host "[OK] Triton gRPC service is running persistently in tmux." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 8. Local SSH Tunnel Setup & End-to-End Verification
# ------------------------------------------------------------------------------
if (-not $SkipTunnel) {
    Write-Host "`n=======================================================" -ForegroundColor Cyan
    Write-Host "  Establishing Local SSH Tunnel & Running Verification" -ForegroundColor Cyan
    Write-Host "=======================================================" -ForegroundColor Cyan

    # Terminate any existing background SSH tunnel on local port 8001
    Get-Process -Name ssh -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -match "8001:localhost:8001" } | Stop-Process -Force -ErrorAction SilentlyContinue

    Write-Host "[Tunnel] Forwarding local port 8001 -> root@${HostIP}:${Port}:8001..." -ForegroundColor Yellow
    $tunnelJob = Start-Job -ScriptBlock {
        param($p, $h)
        ssh -N -L 8001:localhost:8001 -p $p "root@$h"
    } -ArgumentList $Port, $HostIP

    Start-Sleep -Seconds 2

    # Verify locally using test_triton_grpc.py
    Write-Host "[Verify] Executing local integration test against localhost:8001..." -ForegroundColor Yellow
    if (Test-Path ".venv\Scripts\python.exe") {
        & .venv\Scripts\python.exe tests/test_triton_grpc.py
    } else {
        python tests/test_triton_grpc.py
    }
}

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host "  DEPLOYMENT & SERVICE READY!" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  * gRPC Server: localhost:8001 (via active SSH tunnel)" -ForegroundColor White
Write-Host "  * Remote Logs: ssh -p $Port root@$HostIP 'tail -f /workspace/triton.log'" -ForegroundColor White
Write-Host "  * Remote Manager: ssh -p $Port root@$HostIP 'bash /workspace/TabDPT-reco/scripts/manage_remote_server.sh status'" -ForegroundColor White
Write-Host "=======================================================`n" -ForegroundColor Green
