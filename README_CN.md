# 实时金融欺诈检测系统

> **在线演示：** [http://135.116.191.128/](http://135.116.191.128/)

一个生产级端到端欺诈检测系统，将 **GraphSAGE 图神经网络**与 **Claude AI Agent** 相结合，实现实时交易风险分析。基于 Anthropic Hackathon 构建。

---

## 系统架构

```
PaySim CSV（630万条交易）
        │ 分块读取
        ▼
stream_producer.py ──XADD──▶ Redis Stream（transactions:stream）
                                      │
              ┌───────────────────────┤
              │                       │ XREAD
              ▼                       ▼
   ┌─────────────────┐     ┌──────────────────────────┐
   │  数据层          │     │  评分与演示层              │
   │  :8000          │     │  :8081                    │
   │                 │     │                           │
   │ NetworkX 图构建  │     │ /batch  每批100条/60秒    │
   │ HeteroData      │────▶│ GNN概率 + 行为特征打分     │
   │ /heterodata     │     │                           │
   └─────────────────┘     │ [Investigate Batch] ──▶ Claude Sonnet 4.6
              │             │   ReAct tool-use 循环     Agent
   ┌──────────┘             │   → 调查报告卡片          │
   ▼                        │                          │
   ┌─────────────────┐      │ [AI Analysis Report] ──▶ Claude Opus 4.6
   │  模型层          │      │   SSE 流式 HTML 报告      Agent
   │  :8001          │      │                          │
   │                 │      └──────────────────────────┘
   │ GraphSAGE 2层   │
   │ /predict/tx     │
   └─────────────────┘
```

**5层架构：**
| 层级 | 端口 | 职责 |
|------|------|------|
| 数据管道层 | :8000 | Redis Stream → NetworkX 图 → PyG HeteroData |
| GNN 推理层 | :8001 | GraphSAGE 2层 → 每账户欺诈概率 |
| 评分与演示层 | :8081 | 批量打分 + Claude Agent + SSE 报告 + Dashboard |
| Redis | :6379 | 流式消息代理（持久化，支持断点续读） |
| Azure AKS | — | Kubernetes 编排，公网 LoadBalancer |

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 流式消息代理 | Redis Stream（`XADD` / `XREAD`） |
| 图构建 | NetworkX（异构账户关系图） |
| GNN 模型 | GraphSAGE · PyTorch Geometric（2层，hidden=64） |
| API 框架 | FastAPI（异步） |
| AI Agent | Claude Sonnet 4.6 — ReAct tool-use 循环 |
| AI 报告 | Claude Opus 4.6 — SSE 流式 HTML |
| 容器化 | Docker（3个镜像） |
| 云端部署 | Azure Kubernetes Service（AKS）+ ACR |
| 基础设施即代码 | Azure Bicep |

---

## 核心功能

- **实时流处理** — PaySim 630万条交易通过 Redis Stream 按时间步回放；Dashboard 每60秒刷新一次
- **图感知欺诈检测** — GraphSAGE 聚合2跳邻居特征；坏人的邻居往往也是坏人，图结构天然捕捉欺诈团伙
- **按需 Claude Agent 调查** — 点击"Investigate This Batch"触发 ReAct 循环，调用 `score_transaction` 和 `get_account_profile` 工具，生成带风险结论和处置建议的英文调查报告
- **SSE 流式 AI 报告** — 点击"AI Analysis Report"，Claude Opus 4.6 实时流式生成完整 HTML 欺诈分析报告直接推送到浏览器
- **三种欺诈图模式检测**（详见 `fraud_patterns.html`）：
  - 账户清空 + 转移（线性链）
  - 资金循环洗钱（强连通分量环）
  - 扇形打款（星型拓扑）

---

## 项目结构

```
fraud-detection-data-api/
├── demo/                          # 演示前端（FastAPI SSE + Dashboard）
│   ├── app.py                     # FastAPI: /batch /investigate /report SSE
│   ├── index.html                 # 主 Dashboard 界面
│   ├── dashboard.html             # 静态 Dashboard 参考
│   └── Dockerfile
│
├── fraud-detection-data-api-main/
│   ├── fraud-detection-data-api-main/   # 数据管道服务
│   │   ├── pipeline/
│   │   │   ├── data_pipeline.py   # NetworkX 图构建器
│   │   │   ├── stream_consumer.py # Redis XREAD 消费者
│   │   │   └── stream_producer.py # PaySim CSV → Redis XADD
│   │   ├── api/main.py            # FastAPI 数据层（:8000）
│   │   ├── azure/
│   │   │   ├── k8s-deployment.yaml
│   │   │   └── main.bicep
│   │   └── deploy.ps1             # 一键 Azure 部署脚本
│   │
│   └── fraud-gnn-model/           # GNN 模型服务
│       ├── model/
│       │   ├── graphsage.py       # GraphSAGE 2层模型
│       │   ├── graph_builder.py   # HeteroData 构建
│       │   └── trainer.py         # 训练循环
│       ├── api/main.py            # FastAPI 推理层（:8001）
│       ├── batch_server.py        # 批量评分服务器（:8091）
│       ├── demo.py                # Claude Agent 调查脚本
│       ├── train.py               # 模型训练入口
│       └── checkpoints/           # 模型权重文件
│
├── agent/                         # Claude Agent 集成模块
├── orchestrator/                  # MCP 服务器与调度器
├── fraud_patterns.html            # 三种欺诈图模式可视化
├── docker-compose.yml
└── README_CN.md
```

---

## 快速开始

### 本地开发

```bash
# 1. 启动 Redis
docker run -d -p 6379:6379 redis:7-alpine

# 2. 安装依赖
pip install -r fraud-detection-data-api-main/fraud-detection-data-api-main/requirements.txt

# 3. 启动数据管道（端口 8000）
cd fraud-detection-data-api-main/fraud-detection-data-api-main
uvicorn api.main:app --port 8000

# 4. 启动 GNN 推理服务（端口 8001）
cd ../fraud-gnn-model
uvicorn api.main:app --port 8001

# 5. 启动演示前端（端口 8081）
cd ../../demo
ANTHROPIC_API_KEY=sk-ant-... uvicorn app:app --port 8081

# 6. 开始流式推送交易数据
cd ../fraud-detection-data-api-main/fraud-detection-data-api-main
python run_stream.py
```

访问 [http://localhost:8081](http://localhost:8081)

### Azure 一键部署

```powershell
cd fraud-detection-data-api-main\fraud-detection-data-api-main
$env:ANTHROPIC_API_KEY = "sk-ant-..."
.\deploy.ps1
```

脚本自动完成：通过 Bicep 创建 ACR + AKS、构建并推送3个 Docker 镜像、将所有服务部署到 Kubernetes、输出公网 IP。

---

## 模型训练

```bash
cd fraud-detection-data-api-main/fraud-gnn-model
python train.py --steps 100 --fraud-only --epochs 50
```

训练完成后生成 `checkpoints/best_model.pt` 和 `checkpoints/fraud_probs.json`（含80,486个账户的预计算欺诈概率）。

---

## Claude Agent 工作原理

Agent 采用 **ReAct（推理+行动）循环**：

```
用户点击"Investigate This Batch"
        │
        ▼
Claude Sonnet 4.6
  ├── 调用工具: score_transaction(src, dst, amount, type)
  │     └── 返回: 发款方概率、收款方概率、风险等级、原因
  ├── 调用工具: get_account_profile(account_id)
  │     └── 返回: GNN欺诈概率、风险等级、数据来源
  └── end_turn → 结构化英文调查报告
        └── 渲染为 Dashboard 上的可折叠卡片
```

---

## 在线演示

**[http://135.116.191.128/](http://135.116.191.128/)**

- Live Transaction Feed 每60秒刷新（每批100条交易）
- 点击 **Investigate This Batch** 对最可疑交易运行 Claude Agent 调查
- 点击 **AI Analysis Report** 由 Claude Opus 4.6 流式生成完整 HTML 报告
- KPI 看板实时显示所有已处理批次的欺诈统计

---

## 数据集

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — 合成移动支付交易数据集，630万条记录，欺诈率约10%。仅用于流式回放模拟，不含真实金融数据。
