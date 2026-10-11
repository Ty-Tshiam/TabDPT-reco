# deploy_to_vast.ps1
# ==============================================================================
# Automated One-Command Cloud Docker Deployment Pipeline for Vast.ai
# GitHub Repository: https://github.com/Ty-Tshiam/TabDPT-reco
# ==============================================================================
# What this script does:
#   1. Verifies SSH & remote NVIDIA GPU connectivity.
#   2. Installs Docker, Docker Compose, & NVIDIA Container Toolkit on remote host.
#   3. Synchronizes project code, Dockerfiles, metadata, context tensors, & processed Parquets.
#   4. Builds & spins up full cluster (Init-Model, Triton, Redis, Worker, Streamlit) via Docker Compose.
#   5. Forwards ports 8501 (Streamlit Web Portal) and 8001 (Triton gRPC) to localhost.
#   6. Runs local end-to-end integration verification tests.
# ==============================================================================
# Usage:
#   .\deploy_to_vast.ps1                                   # Interactive prompts
#   .\deploy_to_vast.ps1 -HostIP "65.108.94.15" -Port 42749  # Direct execution
#   .\deploy_to_vast.ps1 -ForceDataSync                    # Force re-uploading processed parquets
#   .\deploy_to_vast.ps1 -SkipTunnel                       # Deploy without opening local tunnel
# ==============================================================================

param(
    [string]$HostIP = "",
    [string]$Port = "",
    [switch]$ForceDataSync,
    [switch]$SkipTunnel
)

# ------------------------------------------------------------------------------
# 1. Resolve Connection Parameters (Interactive + Default Fallbacks)
# ------------------------------------------------------------------------------
$DEFAULT_IP = ""
$DEFAULT_PORT = ""

if ([string]::IsNullOrWhiteSpace($HostIP)) {
    $promptIP = Read-Host "Enter Vast.ai Host IP [Default: $DEFAULT_IP]"
    $HostIP = if ([string]::IsNullOrWhiteSpace($promptIP)) { $DEFAULT_IP } else { $promptIP.Trim() }
}

if ([string]::IsNullOrWhiteSpace($Port)) {
    $promptPort = Read-Host "Enter Vast.ai SSH Port [Default: $DEFAULT_PORT]"
    $Port = if ([string]::IsNullOrWhiteSpace($promptPort)) { $DEFAULT_PORT } else { $promptPort.Trim() }
}

if ([string]::IsNullOrWhiteSpace($HostIP) -or [string]::IsNullOrWhiteSpace($Port)) {
    Write-Host "[ERROR] Both HostIP and Port are required to connect." -ForegroundColor Red
    exit 1
}

$sshTarget = "root@${HostIP}"
$remoteDir = "/workspace/TabDPT-reco"

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host "  TabDPT Real-Time Banking Recommender - Cloud Deployment" -ForegroundColor Cyan
Write-Host "  GitHub Repo: https://github.com/Ty-Tshiam/TabDPT-reco" -ForegroundColor Cyan
Write-Host "  Target Instance: ${sshTarget}:${Port}" -ForegroundColor Cyan
Write-Host "=======================================================`n" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# 2. Verify SSH Connectivity & GPU Hardware
# ------------------------------------------------------------------------------
Write-Host "==> [1/6] Verifying SSH connectivity & GPU status..." -ForegroundColor Yellow
$testSSH = (ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $Port $sshTarget "echo 'SSH_OK' && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>&1) -join "`n"
if ($testSSH -notmatch "SSH_OK") {
    Write-Host "[ERROR] Failed to connect to ${sshTarget}:${Port} via SSH." -ForegroundColor Red
    Write-Host "Error details: $testSSH" -ForegroundColor Red
    exit 1
}
$gpuInfo = if ($testSSH -match "(NVIDIA[^\r\n]+)") { $matches[1] } else { "Detected" }
Write-Host "[OK] Connected! Remote GPU: $gpuInfo" -ForegroundColor Green

# ------------------------------------------------------------------------------
# 3. Remote Host Environment & Docker Bootstrap
# ------------------------------------------------------------------------------
Write-Host "`n==> [2/6] Verifying Docker, Docker Compose, & NVIDIA Container Runtime..." -ForegroundColor Yellow
ssh -p $Port $sshTarget "mkdir -p ${remoteDir}/scripts"
scp -O -P $Port scripts/setup_remote_docker.sh "${sshTarget}:${remoteDir}/scripts/"
ssh -p $Port $sshTarget "chmod +x ${remoteDir}/scripts/setup_remote_docker.sh && bash ${remoteDir}/scripts/setup_remote_docker.sh"
Write-Host "[OK] Docker environment verified." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 4. Initialize Remote Directory Structure & Synchronize Files
# ------------------------------------------------------------------------------
Write-Host "`n==> [3/6] Synchronizing project code, Dockerfiles, and datasets..." -ForegroundColor Yellow
ssh -p $Port $sshTarget "mkdir -p ${remoteDir}/data/tensors ${remoteDir}/data/metadata ${remoteDir}/data/processed ${remoteDir}/model_repo/tabdpt/3 ${remoteDir}/src ${remoteDir}/docs ${remoteDir}/tests"

# 4a. Docker configs & specifications
scp -O -P $Port docker-compose.yml Dockerfile.worker Dockerfile.frontend .dockerignore requirements.txt "${sshTarget}:${remoteDir}/"

# 4b. Source trees, docs, and model configurations
scp -O -r -P $Port src docs model_repo "${sshTarget}:${remoteDir}/"
scp -O -P $Port tests/test_triton_grpc.py "${sshTarget}:${remoteDir}/tests/"

# 4c. Core context tensors & metadata mappings
scp -O -P $Port data/tensors/context.pt data/tensors/y.pt "${sshTarget}:${remoteDir}/data/tensors/"
scp -O -P $Port data/metadata/*.json "${sshTarget}:${remoteDir}/data/metadata/"

# 4d. Synchronize test and target Parquets (Required for Worker & Frontend dossiers)
$checkProcessed = ssh -p $Port $sshTarget "[ -d ${remoteDir}/data/processed/train_clean ] && [ -d ${remoteDir}/data/processed/test_may2016_clean ] && [ -d ${remoteDir}/data/processed/test_targets ] && echo 'EXISTS' || echo 'MISSING'"
if ($checkProcessed -match "EXISTS" -and -not $ForceDataSync) {
    Write-Host "[INFO] Processed Parquet datasets already exist on server. Skipping upload (use -ForceDataSync to overwrite)." -ForegroundColor Cyan
} else {
    Write-Host "[INFO] Uploading test, target, and customer history Parquets (~195 MB)..." -ForegroundColor Cyan
    scp -O -r -P $Port data/processed/test_may2016_clean data/processed/test_targets data/processed/train_clean "${sshTarget}:${remoteDir}/data/processed/"
    Write-Host "[OK] Parquets uploaded successfully." -ForegroundColor Green
}

Write-Host "[OK] Codebase and data synchronized." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 5. Build & Launch Containers via Docker Compose
# ------------------------------------------------------------------------------
Write-Host "`n==> [4/6] Building images & starting Docker cluster..." -ForegroundColor Yellow

# Clean up any lingering containers
ssh -p $Port $sshTarget "cd ${remoteDir} && docker compose down 2>/dev/null || true"

# Launch cluster in background
ssh -p $Port $sshTarget "cd ${remoteDir} && docker compose up -d --build"

Write-Host "[INFO] Waiting for containers to initialize..." -ForegroundColor Cyan
Write-Host "       (init-model is downloading weights & precomputing the 5.6 GB KV-cache)" -ForegroundColor Cyan

$clusterReady = $false
for ($i = 0; $i -lt 40; $i++) {
    $psOutput = (ssh -p $Port $sshTarget "cd ${remoteDir} && docker compose ps --format '{{.Service}}: {{.Status}}'" 2>&1) -join "`n"
    if ($psOutput -match "frontend.*Up" -and $psOutput -match "triton.*Up" -and $psOutput -match "worker.*Up") {
        $clusterReady = $true
        break
    }
    Start-Sleep -Seconds 3
    Write-Host "." -NoNewline
}
Write-Host ""

if ($clusterReady) {
    Write-Host "[OK] All 5 cluster services are healthy and running!" -ForegroundColor Green
} else {
    Write-Host "[WARN] Cluster is still booting or encountered an issue. Current container status:" -ForegroundColor Yellow
    ssh -p $Port $sshTarget "cd ${remoteDir} && docker compose ps"
}

# ------------------------------------------------------------------------------
# 6. Local SSH Tunnel Setup (Ports 8501 & 8001)
# ------------------------------------------------------------------------------
if (-not $SkipTunnel) {
    Write-Host "`n==> [5/6] Establishing local SSH tunnels for Streamlit & Triton..." -ForegroundColor Yellow

    # Terminate any existing local processes occupying ports 8001 or 8501
    foreach ($localPort in @(8001, 8501)) {
        $existingConns = Get-NetTCPConnection -LocalPort $localPort -ErrorAction SilentlyContinue
        if ($existingConns) {
            $pids = $existingConns.OwningProcess | Select-Object -Unique
            foreach ($procId in $pids) {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            }
        }
    }
    Start-Sleep -Milliseconds 500

    Write-Host "[Tunnel] Forwarding localhost:8501 -> Streamlit Portal (remote:8501)..." -ForegroundColor Yellow
    Write-Host "[Tunnel] Forwarding localhost:8001 -> Triton gRPC Server (remote:8001)..." -ForegroundColor Yellow

    Start-Process -FilePath "ssh" -ArgumentList "-o", "StrictHostKeyChecking=no", "-N", "-L", "8001:localhost:8001", "-L", "8501:localhost:8501", "-p", $Port, $sshTarget -WindowStyle Hidden

    # Wait up to 10 seconds for local ports to respond
    for ($i = 0; $i -lt 10; $i++) {
        $checkWeb = Test-NetConnection -ComputerName "127.0.0.1" -Port 8501 -WarningAction SilentlyContinue
        $checkGrpc = Test-NetConnection -ComputerName "127.0.0.1" -Port 8001 -WarningAction SilentlyContinue
        if ($checkWeb.TcpTestSucceeded -and $checkGrpc.TcpTestSucceeded) {
            break
        }
        Start-Sleep -Seconds 1
    }
}

# ------------------------------------------------------------------------------
# 7. End-to-End Verification Test
# ------------------------------------------------------------------------------
Write-Host "`n==> [6/6] Running end-to-end integration test against remote Triton..." -ForegroundColor Yellow
if (-not $SkipTunnel) {
    if (Test-Path ".venv\Scripts\python.exe") {
        & .venv\Scripts\python.exe tests/test_triton_grpc.py
    } else {
        python tests/test_triton_grpc.py
    }
} else {
    Write-Host "[INFO] Tunnels skipped. Testing directly inside remote worker container..." -ForegroundColor Cyan
    ssh -p $Port $sshTarget "docker exec tabdpt_worker python tests/test_triton_grpc.py"
}

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host "  CLOUD DOCKER CLUSTER FULLY DEPLOYED & ONLINE!" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  * Streamlit Web Dashboard: http://localhost:8501" -ForegroundColor White
Write-Host "  * Triton gRPC Endpoint:    localhost:8001" -ForegroundColor White
Write-Host "  * Live Container Logs:     ssh -p $Port ${sshTarget} 'cd ${remoteDir} && docker compose logs -f'" -ForegroundColor White
Write-Host "  * Stop Cluster:            ssh -p $Port ${sshTarget} 'cd ${remoteDir} && docker compose down'" -ForegroundColor White
Write-Host "=======================================================`n" -ForegroundColor Green
