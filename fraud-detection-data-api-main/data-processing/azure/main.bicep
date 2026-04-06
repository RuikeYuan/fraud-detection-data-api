// azure/main.bicep — ACR + AKS 基础设施
// 部署：Azure Container Registry + Azure Kubernetes Service

@description('Azure region')
param location string = resourceGroup().location

@description('资源名称前缀')
param prefix string = 'frauddet'

@description('AKS 节点数量')
param aksNodeCount int = 1   // 1 个节点（4 vCPU / 16 GB），适配 Azure for Students 6 vCPU 配额

@description('AKS 节点 VM 规格')
// Standard_D4s_v3: 4 vCPU / 16 GB RAM
// GNN 推理加载 PyTorch + PyG + 图结构，至少需要 4–6 GB 内存
// 原来的 Standard_B2s_v2（2核4GB）会因 OOM 导致 gnn-api pod 被杀掉
param aksNodeVmSize string = 'Standard_D4s_v3'

// ─── Azure Container Registry ────────────────────────────────────────────────
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${prefix}acr${uniqueString(resourceGroup().id)}'
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// ─── AKS Cluster ─────────────────────────────────────────────────────────────
resource aks 'Microsoft.ContainerService/managedClusters@2024-01-01' = {
  name: '${prefix}-aks'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    dnsPrefix: '${prefix}-aks'
    agentPoolProfiles: [
      {
        name: 'default'
        count: aksNodeCount
        vmSize: aksNodeVmSize
        osType: 'Linux'
        mode: 'System'
      }
    ]
    networkProfile: {
      networkPlugin: 'azure'
      loadBalancerSku: 'standard'
    }
  }
}

// ─── 授予 AKS 拉取 ACR 镜像的权限 ────────────────────────────────────────────
resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, aks.id, 'acrpull')
  scope: acr
  properties: {
    principalId: aks.properties.identityProfile.kubeletidentity.objectId
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7f951dda-4ed3-4680-a7ca-43fe172d538d'  // AcrPull 内置角色
    )
    principalType: 'ServicePrincipal'
  }
}

// ─── Outputs（deploy.ps1 从此处读取资源名称）────────────────────────────────
output acrLoginServer string = acr.properties.loginServer
output acrName        string = acr.name
output aksName        string = aks.name
output aksResourceGroup string = resourceGroup().name
