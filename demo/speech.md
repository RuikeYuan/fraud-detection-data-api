# Speech Script — ~1.5 minutes

---

Hi everyone.

Today I want to show you a multi-agent AI system
that catches financial fraud in real time.

---

Here's the core insight.

Fraudsters don't work alone.
They move money through a ring of accounts — each transfer looks completely normal.
Traditional systems check one transaction at a time, and they miss the whole picture.

Our system takes a different approach.
We treat every bank account like a node in a social network,
and every transfer as a connection between them.
Then we use AI to spot the bad patterns hiding in those connections.

---

We built four specialized agents working together.

A Data Agent that listens to a live transaction stream.
A Graph Agent that maps out the entire network in real time.
An Inference Agent running a graph neural network to score each account.
And an Alert Agent that generates the final report and takes action.

Sitting above all of them is Claude — acting as the brain.

Claude doesn't just run one check. It reasons through the problem step by step.
Score the transaction first. If something looks off, check the account history.
Then look at the graph — is this account stuck in a loop with other suspicious accounts?

This is called a ReAct loop — Reason, then Act, based on what you find.

---

Once Claude has enough evidence, the system responds automatically.

High risk — the transaction gets blocked immediately.
Medium risk — it goes to a human reviewer.
Low risk — it passes through, and we update the graph in the background.

---

Let me show you a quick demo.

A large transfer comes in. Looks normal.
Claude scores it, finds the sender is flagged at 85% fraud probability.
Digs into the account — 14 transactions in the last 24 hours.
Checks the graph — finds a 4-account cycle. Classic money laundering.
Transaction blocked. Full report generated. In seconds.

---

The key thing I want to leave you with:
because Claude is the orchestrator, we can swap out the fraud model,
add new data sources, or change the rules —
without rewriting any of the agent logic.

That's what makes this system flexible enough for the real world.

Thank you.

---

> **Timing guide:**
> - Total words: ~270
> - Pace: calm, ~150 words/min
> - Target time: ~1 min 45 sec (trim the demo paragraph if needed)
> - Pause at each `---` break
> - Slow down on "ReAct loop" and "4-account cycle" — these are the key moments

---
---

# 演讲稿（中文版）—— 约 1.5 分钟

---

大家好。

今天我想向大家展示一个多智能体 AI 系统，
能够实时识别金融欺诈。

---

先说一个核心洞察。

欺诈分子从不单独行动。
他们把资金在一圈账户之间辗转转移——每一笔看起来都完全正常。
传统系统一次只检查一笔交易，结果漏掉了整体规律。

我们的系统换了一种思路。
我们把每个银行账户当作社交网络中的一个节点，
每一笔转账就是节点之间的连接。
然后用 AI 找出藏在这些连接里的异常模式。

---

我们搭建了四个专职智能体协同工作。

数据智能体，负责监听实时交易流。
图智能体，负责实时绘制整个账户网络。
推理智能体，运行图神经网络，给每个账户打欺诈风险分。
告警智能体，负责生成最终报告并触发处置动作。

在这四个智能体之上，是 Claude——充当整个系统的大脑。

Claude 不只做一次检查，而是一步步推理整个问题。
先给交易打分，如果发现异常，再查账户历史。
然后看图谱——这个账户有没有和其他可疑账户形成环路？

这叫做 ReAct 循环——先推理，再行动，根据发现的证据决定下一步。

---

一旦 Claude 收集到足够的证据，系统就会自动响应。

高风险——交易立即被拦截。
中风险——进入人工审核队列。
低风险——正常放行，图谱在后台异步更新。

---

让我给大家演示一下。

一笔大额转账进来了，表面上看起来很正常。
Claude 给它打分，发现发款方的欺诈概率高达 85%。
深挖账户历史——24 小时内有 14 笔交易，典型的分散转账特征。
再看图谱——发现了一个四账户环路，经典的洗钱手法。
交易被拦截，完整调查报告生成完毕——整个过程只需几秒钟。

---

最后我想留给大家一个重点：
因为 Claude 是整个系统的编排者，
我们可以随时换掉底层的欺诈检测模型、接入新的数据源、或者调整风控规则——
而完全不需要改动任何智能体逻辑。

这正是这套系统能够应对真实业务场景的关键所在。

谢谢大家。

---

> **时间参考：**
> - 总字数：约 350 字
> - 语速：平稳，约 230 字/分钟
> - 目标时长：约 1 分 30 秒
> - 每个 `---` 处自然停顿
> - "ReAct 循环"和"四账户环路"处放慢语速，这是两个关键点
