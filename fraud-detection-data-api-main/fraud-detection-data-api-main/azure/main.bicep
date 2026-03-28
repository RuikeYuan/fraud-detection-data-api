// Azure Bicep template for Fraud Detection system
// Deploys: Azure Container Registry (ACR) + Azure Kubernetes Service (AKS)

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Unique prefix for resource names')
param prefix string = 'frauddet'

@description('AKS node count')
param aksNodeCount int = 1

@description('AKS node VM size')
param aksNodeVmSize string = 'Standard_B2s_v2'

// ─── Azure Container Registry ───
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

// ─── AKS Cluster ───
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

// ─── ACR Pull Role for AKS ───
resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, aks.id, 'acrpull')
  scope: acr
  properties: {
    principalId: aks.properties.identityProfile.kubeletidentity.objectId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d') // AcrPull
    principalType: 'ServicePrincipal'
  }
}

// ─── Outputs ───
output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
output aksName string = aks.name
output aksResourceGroup string = resourceGroup().name
