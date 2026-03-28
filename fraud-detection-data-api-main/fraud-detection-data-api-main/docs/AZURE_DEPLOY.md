# Azure 云部署指南

## 架构概览

```
GitHub Push → GitHub Actions CI/CD
  ├── Build Docker Image
  ├── Push to Azure Container Registry (ACR)
  └── Deploy to Azure Kubernetes Service (AKS)
        ├── fraud-detection-api (x2 replicas)
        └── redis (x1 replica)
```

## 前置条件

- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) 已安装
- [Docker](https://docs.docker.com/get-docker/) 已安装
- [kubectl](https://kubernetes.io/docs/tasks/tools/) 已安装
- Azure 订阅并已登录 (`az login`)

## 文件说明

| 文件 | 用途 |
|------|------|
| `azure/main.bicep` | Bicep IaC 模板 — 创建 ACR + AKS |
| `azure/k8s-deployment.yaml` | Kubernetes 部署清单 (API + Redis + Service) |
| `.github/workflows/azure-deploy.yml` | GitHub Actions CI/CD 流水线 |
| `deploy.sh` | 一键部署脚本 |

---

## 方式一：一键脚本部署

```bash
# 登录 Azure
az login

# 执行部署 (默认 eastasia 区域)
chmod +x deploy.sh
./deploy.sh

# 自定义参数
./deploy.sh --resource-group my-rg --location westus2 --image-tag v1.0
```

脚本会自动完成：资源组创建 → ACR + AKS 部署 → 镜像构建推送 → K8s 应用部署。

---

## 方式二：手动分步部署

### 1. 创建资源组

```bash
az group create --name fraud-detection-rg --location eastasia
```

### 2. 部署基础设施

```bash
az deployment group create \
  --resource-group fraud-detection-rg \
  --template-file azure/main.bicep
```

记下输出中的 `acrLoginServer` 和 `aksName`。

### 3. 构建并推送镜像

```bash
# 登录 ACR (替换为实际名称)
az acr login --name <acrName>

# 构建 & 推送
docker build -t <acrLoginServer>/fraud-detection:latest .
docker push <acrLoginServer>/fraud-detection:latest
```

### 4. 部署到 AKS

```bash
# 获取 AKS 凭据
az aks get-credentials --resource-group fraud-detection-rg --name frauddet-aks

# 替换镜像占位符并部署
sed "s|__ACR_LOGIN_SERVER__|<acrLoginServer>|g; s|__IMAGE_TAG__|latest|g" \
  azure/k8s-deployment.yaml | kubectl apply -f -

# 查看部署状态
kubectl rollout status deployment/fraud-detection-api -n fraud-detection
```

### 5. 获取外部访问地址

```bash
kubectl get svc fraud-detection-api -n fraud-detection
```

`EXTERNAL-IP` 列即为 API 访问地址：`http://<EXTERNAL-IP>/heterodata`

---

## 方式三：GitHub Actions 自动部署

### 配置 Secrets

在 GitHub 仓库 → Settings → Secrets and variables → Actions 中添加：

**`AZURE_CREDENTIALS`** — 服务主体凭据 JSON：

```bash
az ad sp create-for-rbac \
  --name "fraud-detection-cicd" \
  --role contributor \
  --scopes /subscriptions/<subscription-id>/resourceGroups/fraud-detection-rg \
  --json-auth
```

将输出的 JSON 整体粘贴为 Secret 值。

### 触发部署

推送到 `main` 分支即自动触发构建和部署，也可在 Actions 页面手动触发。

---

## 常用运维命令

```bash
# 查看 Pod 状态
kubectl get pods -n fraud-detection

# 查看 API 日志
kubectl logs -l component=api -n fraud-detection -f

# 手动扩缩容
kubectl scale deployment fraud-detection-api -n fraud-detection --replicas=3

# 删除全部资源
az group delete --name fraud-detection-rg --yes --no-wait
```
