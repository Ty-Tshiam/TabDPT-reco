# deploy_to_vast.ps1
# Automates cloning TabDPT-reco, uploading data, and setting up dependencies on any Vast.ai GPU instance.
param(
    [string]$HostIP = "64.90.9.69",
    [string]$Port = "10196"
)

Write-Host "==> [1/5] Cloning GitHub repo on remote instance ($HostIP:$Port)..." -ForegroundColor Cyan
ssh -n -p $Port root@$HostIP "git clone https://github.com/Ty-Tshiam/TabDPT-reco.git /workspace/TabDPT-reco && ln -s /workspace/TabDPT-reco /root/TabDPT-reco && mkdir -p /workspace/TabDPT-reco/data/output && ln -s /workspace/TabDPT-reco/data/output /workspace/TabDPT-reco/output"

Write-Host "==> [2/5] Uploading compressed dataset (226 MB via SCP)..." -ForegroundColor Cyan
scp -P $Port data/santander-product-recommendation.zip "root@${HostIP}:/workspace/TabDPT-reco/data/"

Write-Host "==> [3/5] Extracting dataset on remote NVMe drive..." -ForegroundColor Cyan
ssh -n -p $Port root@$HostIP "cd /workspace/TabDPT-reco/data && unzip -q -o santander-product-recommendation.zip"

Write-Host "==> [4/5] Uploading pre-cleaned Parquet dataset (189 MB via SCP)..." -ForegroundColor Cyan
scp -P $Port -r data/output "root@${HostIP}:/workspace/TabDPT-reco/data/output/"

Write-Host "==> [5/5] Installing Java 21 JRE and Python dependencies into /venv/main..." -ForegroundColor Cyan
ssh -n -p $Port root@$HostIP "apt-get update && apt-get install -y default-jre-headless && /venv/main/bin/pip install -r /workspace/TabDPT-reco/requirements.txt"

Write-Host "`nAll done! You can connect with:" -ForegroundColor Green
Write-Host "ssh -p $Port root@$HostIP -L 8080:localhost:8080" -ForegroundColor Yellow
