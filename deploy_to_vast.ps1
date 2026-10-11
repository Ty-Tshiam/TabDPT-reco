# deploy_to_vast.ps1
# ==============================================================================
# Automated One-Command Cloud Deployment Pipeline for Vast.ai / Cloud GPUs
# GitHub Repository: https://github.com/Ty-Tshiam/TabDPT-reco
# ==============================================================================
# Architecture:
#   - Auto-detects environment:
#     * VM Instance (Docker daemon accessible) -> Uses Docker Compose
#     * Vast.ai Container Instance (Standard)  -> Uses Native High-Performance Mode
#       (Direct GPU execution via tmux + Python + Redis, avoiding nested Docker errors)
# ==============================================================================
# Usage:
#   .\deploy_to_vast.ps1                                   # Interactive prompts
#   .\deploy_to_vast.ps1 -HostIP "65.108.94.15" -Port 42749  # Direct execution
#   .\deploy_to_vast.ps1 -ForceDataSync                    # Force re-uploading processed Parquets
#   .\deploy_to_vast.ps1 -FullStack                        # Launch Redis + Worker + Streamlit on server too
#   .\deploy_to_vast.ps1 -SkipTunnel                       # Deploy without opening local tunnel
# ==============================================================================

param(
    [string]$HostIP = "",
    [string]$Port = "",
    [switch]$ForceDataSync,
    [switch]$FullStack,
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
# 3. Detect Host Environment (Docker VM vs Vast.ai Container)
# ------------------------------------------------------------------------------
Write-Host "`n==> [2/6] Detecting runtime capabilities..." -ForegroundColor Yellow
$dockerCheck = (ssh -p $Port $sshTarget "docker info 2>&1" ) -join "`n"
$useDocker = ($dockerCheck -match "Server Version" -or $dockerCheck -match "Containers:")

if ($useDocker) {
    Write-Host "[MODE] Full VM detected with active Docker daemon -> Deploying via Docker Compose." -ForegroundColor Green
} else {
    Write-Host "[MODE] Standard Vast.ai Container detected (host docker.sock not mounted)." -ForegroundColor Cyan
    Write-Host "       Deploying natively with GPU acceleration via tmux (Faster & 10x lower RAM overhead)." -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# 4. Synchronize Codebase, Models, & Data Tensors
# ------------------------------------------------------------------------------
Write-Host "`n==> [3/6] Synchronizing project files and tensors..." -ForegroundColor Yellow
ssh -p $Port $sshTarget "mkdir -p ${remoteDir}/data/tensors ${remoteDir}/data/metadata ${remoteDir}/data/processed ${remoteDir}/model_repo/tabdpt/3 ${remoteDir}/src ${remoteDir}/docs ${remoteDir}/tests ${remoteDir}/scripts"

# Copy configurations & scripts
scp -O -P $Port docker-compose.yml Dockerfile.worker Dockerfile.frontend .dockerignore requirements.txt "${sshTarget}:${remoteDir}/"
scp -O -r -P $Port src docs model_repo scripts "${sshTarget}:${remoteDir}/"
scp -O -P $Port tests/test_triton_grpc.py "${sshTarget}:${remoteDir}/tests/"
ssh -p $Port $sshTarget "chmod +x ${remoteDir}/scripts/*.sh"

# Copy core context prompt tensors (23 MB + 715 KB) & metadata JSON mappings
scp -O -P $Port data/tensors/context.pt data/tensors/y.pt "${sshTarget}:${remoteDir}/data/tensors/"
scp -O -P $Port data/metadata/*.json "${sshTarget}:${remoteDir}/data/metadata/"

# Optional / FullStack data sync for test customer dossiers
if ($FullStack -or $ForceDataSync) {
    $checkProcessed = ssh -p $Port $sshTarget "[ -d ${remoteDir}/data/processed/train_clean ] && [ -d ${remoteDir}/data/processed/test_may2016_clean ] && echo 'EXISTS' || echo 'MISSING'"
    if ($checkProcessed -match "EXISTS" -and -not $ForceDataSync) {
        Write-Host "[INFO] Processed Parquet datasets already exist on server. Skipping upload." -ForegroundColor Cyan
    } else {
        Write-Host "[INFO] Uploading test, target, and customer history Parquets (~195 MB)..." -ForegroundColor Cyan
        scp -O -r -P $Port data/processed/test_may2016_clean data/processed/test_targets data/processed/train_clean "${sshTarget}:${remoteDir}/data/processed/"
        Write-Host "[OK] Parquets uploaded successfully." -ForegroundColor Green
    }
}

Write-Host "[OK] Files synchronized successfully." -ForegroundColor Green

# ------------------------------------------------------------------------------
# 5. Service Launch (Docker or Native)
# ------------------------------------------------------------------------------
Write-Host "`n==> [4/6] Bootstrapping environment & starting services..." -ForegroundColor Yellow

if ($useDocker) {
    # ------------------ DOCKER COMPOSE PATH ------------------
    ssh -p $Port $sshTarget "cd ${remoteDir} && docker compose down 2>/dev/null || true && docker compose up -d --build"

    Write-Host "[INFO] Waiting for Docker cluster to become healthy..." -ForegroundColor Cyan
    $clusterReady = $false
    for ($i = 1; $i -le 45; $i++) {
        $rawPs = (ssh -p $Port $sshTarget "cd ${remoteDir} && docker compose ps 2>&1") -join "`n"
        if ($rawPs -match "triton.*Up" -or $rawPs -match "tabdpt_triton.*Up") {
            $clusterReady = $true
            Write-Host "`n[OK] Triton container is up and running!" -ForegroundColor Green
            break
        }
        Start-Sleep -Seconds 4
        Write-Host "." -NoNewline
    }
} else {
    # ------------------ NATIVE VAST.AI PATH ------------------
    Write-Host "[1/3] Bootstrapping Python dependencies and tmux..." -ForegroundColor Cyan
    ssh -p $Port $sshTarget "bash ${remoteDir}/scripts/manage_remote_server.sh bootstrap"

    Write-Host "[2/3] Checking precomputed KV-cache & model weights..." -ForegroundColor Cyan
    $checkCache = ssh -p $Port $sshTarget "[ -f ${remoteDir}/data/tensors/context_kv_cache.pt ] && [ -f ${remoteDir}/model_repo/tabdpt/3/model.pt ] && echo 'EXISTS' || echo 'BUILD'"
    if ($checkCache -match "EXISTS") {
        Write-Host "[INFO] Precomputed 5.6 GB KV-cache already exists on server NVMe. Skipping re-computation." -ForegroundColor Cyan
    } else {
        Write-Host "[INFO] Encoding 91k context prompt with save_model.py (~15-20s on GPU)..." -ForegroundColor Cyan
        ssh -p $Port $sshTarget "bash ${remoteDir}/scripts/manage_remote_server.sh build_model"
    }

    Write-Host "[3/3] Launching Triton gRPC inference service on port 8001..." -ForegroundColor Cyan
    if ($FullStack) {
        ssh -p $Port $sshTarget "bash ${remoteDir}/scripts/manage_remote_server.sh start_all"
    } else {
        ssh -p $Port $sshTarget "bash ${remoteDir}/scripts/manage_remote_server.sh restart"
    }
}

# ------------------------------------------------------------------------------
# 6. Local SSH Tunnel Setup
# ------------------------------------------------------------------------------
if (-not $SkipTunnel) {
    Write-Host "`n==> [5/6] Establishing local SSH tunnels for Triton (8001) & Streamlit (8501)..." -ForegroundColor Yellow

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

    Write-Host "[Tunnel] Forwarding localhost:8001 -> Remote Triton gRPC (port 8001)..." -ForegroundColor Yellow
    if ($FullStack) {
        Write-Host "[Tunnel] Forwarding localhost:8501 -> Remote Streamlit UI (port 8501)..." -ForegroundColor Yellow
        Start-Process -FilePath "ssh" -ArgumentList "-o", "StrictHostKeyChecking=no", "-N", "-L", "8001:localhost:8001", "-L", "8501:localhost:8501", "-p", $Port, $sshTarget -WindowStyle Hidden
    } else {
        Start-Process -FilePath "ssh" -ArgumentList "-o", "StrictHostKeyChecking=no", "-N", "-L", "8001:localhost:8001", "-p", $Port, $sshTarget -WindowStyle Hidden
    }

    # Wait up to 10 seconds for local port 8001 to respond
    for ($i = 0; $i -lt 10; $i++) {
        $checkGrpc = Test-NetConnection -ComputerName "127.0.0.1" -Port 8001 -WarningAction SilentlyContinue
        if ($checkGrpc.TcpTestSucceeded) {
            break
        }
        Start-Sleep -Seconds 1
    }
}

# ------------------------------------------------------------------------------
# 7. End-to-End Verification Test
# ------------------------------------------------------------------------------
Write-Host "`n==> [6/6] Verifying live inference through the tunnel..." -ForegroundColor Yellow
if (-not $SkipTunnel) {
    if (Test-Path ".venv\Scripts\python.exe") {
        & .venv\Scripts\python.exe tests/test_triton_grpc.py
    } else {
        python tests/test_triton_grpc.py
    }
}

Write-Host "`n=======================================================" -ForegroundColor Green
Write-Host "  CLOUD DEPLOYMENT READY & SERVING!" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "  * Triton gRPC Endpoint:    localhost:8001" -ForegroundColor White
if ($FullStack) {
    Write-Host "  * Streamlit Web Dashboard: http://localhost:8501" -ForegroundColor White
} else {
    Write-Host "  * Run Local UI Dashboard: streamlit run src/frontend/app.py" -ForegroundColor White
}
Write-Host "  * Remote Manager Status:   ssh -p $Port ${sshTarget} 'bash ${remoteDir}/scripts/manage_remote_server.sh status'" -ForegroundColor White
Write-Host "  * Remote Server Logs:      ssh -p $Port ${sshTarget} 'tail -f ${remoteDir}/triton.log 2>/dev/null || tail -f /workspace/triton.log'" -ForegroundColor White
Write-Host "=======================================================`n" -ForegroundColor Green
