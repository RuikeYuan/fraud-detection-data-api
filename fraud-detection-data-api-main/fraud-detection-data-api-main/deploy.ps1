#
# deploy.ps1 — 一键 Azure 部署脚本（全自动，含工具安装 + 密钥提示）
#
# 用法（PowerShell 管理员终端）：
#   cd fraud-detection-data-api-main\fraud-detection-data-api-main
#   .\deploy.ps
#
# 可选参数：
#   -ResourceGroup  资源组名称（默认 fraud-detection-rg）
#   -Location       Azure 区域（默认 swedencentral）
#   -ImageTag       镜像标签（默认 latest）
#   -Subscription   Azure 订阅 ID（有多个订阅时指定）
#
param(
    [string]$ResourceGroup = "fraud-detection-rg",
    [string]$Location      = "swedencentral",
    [string]$ImageTag      = "latest",
    [string]$Subscription  = ""
)

$ErrorActionPreference = "Stop"

# PSScriptRoot = .../fraud-detection-data-api-main/fraud-detection-data-api-main/
# ProjectRoot  = .../fraud-detection-data-api-main/
# RepoRoot     = .../fraud-detection-data-api/  （含 demo/ 文件夹）
$ProjectRoot = (Get-Item $PSScriptRoot).Parent.FullName
$RepoRoot    = (Get-Item $ProjectRoot).Parent.FullName

# ─── 0. 自动安装缺失工具 ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== 0/6  Checking & installing prerequisites ===" -ForegroundColor Cyan

function Ensure-Tool {
    param([string]$Cmd, [string]$WingetId, [string]$Label)
    if (-not (Get-Command $Cmd -ErrorAction SilentlyContinue)) {
        Write-Host "  $Label not found — installing via winget..." -ForegroundColor Yellow
        winget install --id $WingetId -e --accept-package-agreements --accept-source-agreements | Out-Null
        # 刷新 PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH","User")
        if (-not (Get-Command $Cmd -ErrorAction SilentlyContinue)) {
            Write-Host "  ERROR: $Label still not found after install. Please restart the terminal." -ForegroundColor Red
            exit 1
        }
    }
    Write-Host "  $Label : OK" -ForegroundColor Green
}

Ensure-Tool "az"      "Microsoft.AzureCLI"     "Azure CLI"
Ensure-Tool "docker"  "Docker.DockerDesktop"   "Docker"
Ensure-Tool "kubectl" "Kubernetes.kubectl"     "kubectl"

# 检查 Docker 守护进程是否运行（忽略插件 WARNING，只看是否能连到 daemon）
$dockerInfo = docker info 2>&1
if (-not ($dockerInfo | Select-String -Pattern "Server Version" -Quiet)) {
    Write-Host "  ERROR: Docker daemon is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}
Write-Host "  Docker daemon: running" -ForegroundColor Green

# ─── ANTHROPIC_API_KEY 处理 ────────────────────────────────────────────────
# 若环境变量未设置，交互式提示输入（不会回显到终端）
if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host ""
    Write-Host "  ANTHROPIC_API_KEY is not set." -ForegroundColor Yellow
    Write-Host "  You can get your key from: https://console.anthropic.com/settings/keys"
    $secureKey = Read-Host "  Paste your Anthropic API key (sk-ant-...)" -AsSecureString
    $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $env:ANTHROPIC_API_KEY = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}
if ($env:ANTHROPIC_API_KEY -notlike "sk-ant-*") {
    Write-Host "  WARNING: Key doesn't start with 'sk-ant-', double-check it." -ForegroundColor Yellow
}
Write-Host "  ANTHROPIC_API_KEY: set ($($env:ANTHROPIC_API_KEY.Length) chars)" -ForegroundColor Green

# ─── Azure 登录 ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Checking Azure login..."
az account show 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Not logged in — opening browser for az login..." -ForegroundColor Yellow
    az login
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: az login failed." -ForegroundColor Red; exit 1 }
}

if ($Subscription) {
    az account set --subscription $Subscription
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

$SubName = az account show --query "name" -o tsv 2>$null
$SubId   = az account show --query "id"   -o tsv 2>$null
if (-not $SubId -or $SubName -like "*tenant level*") {
    Write-Host "  ERROR: No valid subscription. Run: az account list" -ForegroundColor Red; exit 1
}

Write-Host "  Azure subscription: $SubName ($SubId)" -ForegroundColor Green
Write-Host ""
Write-Host "  Resource Group : $ResourceGroup"
Write-Host "  Location       : $Location"
Write-Host "  Image Tag      : $ImageTag"
Write-Host ""

# ─── 资源提供程序 ──────────────────────────────────────────────────────────
function Register-AzProvider {
    param([string]$Namespace)
    $state = az provider show --namespace $Namespace --query "registrationState" -o tsv 2>$null
    if ($state -ne "Registered") {
        Write-Host "  Registering $Namespace ..."
        az provider register --namespace $Namespace 2>$null
        for ($i = 0; $i -lt 30; $i++) {
            $state = az provider show --namespace $Namespace --query "registrationState" -o tsv 2>$null
            if ($state -eq "Registered") { break }
            Start-Sleep -Seconds 10
        }
        if ($state -ne "Registered") {
            Write-Host "ERROR: $Namespace registration timed out" -ForegroundColor Red; exit 1
        }
    }
    Write-Host "  $Namespace : OK" -ForegroundColor Green
}

Write-Host "=== 0b/6  Registering resource providers ===" -ForegroundColor Cyan
Register-AzProvider "Microsoft.ContainerRegistry"
Register-AzProvider "Microsoft.ContainerService"
Register-AzProvider "Microsoft.OperationsManagement"

# ─── 1. 创建资源组 ─────────────────────────────────────────────────────────
Write-Host "=== 1/6  Creating resource group ===" -ForegroundColor Cyan
az group create --name $ResourceGroup --location $Location -o none
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "  Resource group '$ResourceGroup': ready" -ForegroundColor Green

# ─── 2. Bicep：ACR + AKS ──────────────────────────────────────────────────
Write-Host "=== 2/6  Deploying ACR + AKS via Bicep (~5 min) ===" -ForegroundColor Cyan
$DeployOut = az deployment group create `
    --resource-group $ResourceGroup `
    --template-file "$PSScriptRoot\azure\main.bicep" `
    --query "properties.outputs" -o json
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Bicep failed." -ForegroundColor Red; exit 1 }

$outputs        = $DeployOut | ConvertFrom-Json
$AcrName        = $outputs.acrName.value
$AcrLoginServer = $outputs.acrLoginServer.value
$AksName        = $outputs.aksName.value
Write-Host "  ACR : $AcrLoginServer" -ForegroundColor Green
Write-Host "  AKS : $AksName"        -ForegroundColor Green

# ─── 3. 构建 + 推送三个 Docker 镜像 ───────────────────────────────────────
Write-Host "=== 3/6  Building & pushing 3 Docker images ===" -ForegroundColor Cyan
az acr login --name $AcrName
if ($LASTEXITCODE -ne 0) { exit 1 }

$images = @(
    @{
        Name    = "fraud-data-api"
        Tag     = "${AcrLoginServer}/fraud-data-api:${ImageTag}"
        Context = $PSScriptRoot
        File    = "$PSScriptRoot\Dockerfile"
    },
    @{
        Name    = "fraud-gnn-api"
        Tag     = "${AcrLoginServer}/fraud-gnn-api:${ImageTag}"
        Context = "$ProjectRoot\fraud-gnn-model"
        File    = "$ProjectRoot\fraud-gnn-model\Dockerfile"
    },
    @{
        Name    = "fraud-demo"
        Tag     = "${AcrLoginServer}/fraud-demo:${ImageTag}"
        Context = $RepoRoot
        File    = "$RepoRoot\demo\Dockerfile"
    }
)

foreach ($img in $images) {
    Write-Host "  [$($img.Name)] Building..." -ForegroundColor Yellow
    docker build -t $img.Tag -f $img.File $img.Context
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: build failed for $($img.Name)" -ForegroundColor Red; exit 1 }
    Write-Host "  [$($img.Name)] Pushing..." -ForegroundColor Yellow
    docker push $img.Tag
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: push failed for $($img.Name)" -ForegroundColor Red; exit 1 }
    Write-Host "  [$($img.Name)] Done" -ForegroundColor Green
}

# ─── 4. 连接 AKS ──────────────────────────────────────────────────────────
Write-Host "=== 4/6  Connecting kubectl to AKS ===" -ForegroundColor Cyan
az aks get-credentials --resource-group $ResourceGroup --name $AksName --overwrite-existing
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "  kubectl context: $AksName" -ForegroundColor Green

# ─── 5. 创建 K8s Secret ────────────────────────────────────────────────────
Write-Host "=== 5/6  Creating Kubernetes secret ===" -ForegroundColor Cyan
kubectl create namespace fraud-detection --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic anthropic-secret `
    --from-literal=api-key=$env:ANTHROPIC_API_KEY `
    --namespace fraud-detection `
    --dry-run=client -o yaml | kubectl apply -f -
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host "  anthropic-secret: created" -ForegroundColor Green

# ─── 6. 部署到 Kubernetes ──────────────────────────────────────────────────
Write-Host "=== 6/6  Applying Kubernetes manifests ===" -ForegroundColor Cyan
$manifest = (Get-Content "$PSScriptRoot\azure\k8s-deployment.yaml" -Raw) `
    -replace '__ACR__', $AcrLoginServer `
    -replace '__TAG__', $ImageTag

$tmp = [System.IO.Path]::GetTempFileName() + ".yaml"
$manifest | Set-Content -Path $tmp -Encoding UTF8
kubectl apply -f $tmp
Remove-Item $tmp -ErrorAction SilentlyContinue
if ($LASTEXITCODE -ne 0) { exit 1 }

foreach ($dep in @("data-api", "gnn-api", "demo")) {
    Write-Host "  Waiting for $dep rollout..." -ForegroundColor Yellow
    kubectl rollout status deployment/$dep -n fraud-detection --timeout=300s
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $dep timed out. Logs:" -ForegroundColor Red
        kubectl logs -n fraud-detection -l app=$dep --tail=30
        exit 1
    }
    Write-Host "  $dep : ready" -ForegroundColor Green
}

# ─── 完成 ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Deployment complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

$DemoIp = kubectl get svc demo-service -n fraud-detection `
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>$null
if (-not $DemoIp) { $DemoIp = "<pending>" }

Write-Host ""
Write-Host "  Demo UI  : http://${DemoIp}/"         -ForegroundColor Cyan
Write-Host "  GNN API  : http://${DemoIp}:8001/health  (内网)"
Write-Host "  Data API : http://${DemoIp}:8000/stream/status  (内网)"
Write-Host ""
Write-Host "  常用诊断命令："
Write-Host "    kubectl get pods -n fraud-detection"
Write-Host "    kubectl logs -n fraud-detection -l app=gnn-api"
Write-Host "    kubectl logs -n fraud-detection -l app=demo"
