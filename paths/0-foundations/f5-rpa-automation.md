# F5. RPA 与低代码自动化实战 | RPA & No-Code Automation

> **路径**: Path 0: AI 基础先行 · **模块**: F5
> **最后更新**: 2026-03-14
> **难度**: ⭐⭐ 中级
> **预计时间**: 2-3 小时
> **前置模块**: [F4 自动化与 Agent](f4-agent-automation.md)

🏠 [Hub 首页](../../README.md) · 📋 [Path 0 总览](README.md)

---

## 📖 章节导航

1. [RPA vs 工作流自动化 vs AI Agent](#1-rpa-vs-工作流自动化-vs-ai-agent)
2. [低代码自动化工具全景](#2-低代码自动化工具全景)
3. [n8n 深度实战](#3-n8n-深度实战)
4. [Zapier / Make 实战](#4-zapier--make-实战)
5. [跨境电商 10 大自动化工作流](#5-跨境电商-10-大自动化工作流)
6. [RPA 工具与浏览器自动化](#6-rpa-工具与浏览器自动化)
7. [AI + 自动化的融合](#7-ai--自动化的融合)
8. [工具选择决策框架](#8-工具选择决策框架)
9. [完成标志](#9-完成标志)

---

## 本模块你将学会

F4 讲了自动化的概念和 Agent 的理论。本模块聚焦实操——用具体的工具搭建真实的自动化工作流。

完成后你将能够：
- 区分 RPA、工作流自动化、AI Agent 的适用场景
- 用 n8n 搭建跨境电商自动化工作流（免费、自托管）
- 用 Zapier/Make 快速搭建简单自动化（付费、零代码）
- 了解 Defy、Bardeen、Browse AI 等浏览器 RPA 工具
- 搭建 10 个跨境电商核心自动化场景
- 把 AI（ChatGPT/Claude API）集成到自动化工作流中

> 💡 **与 F4 的区别**：F4 讲的是"AI Agent 能做什么"（概念层），本模块讲的是"用什么工具、怎么搭"（实操层）。F4 偏理论，F5 偏动手。

---

## 1. RPA vs 工作流自动化 vs AI Agent

### 1.1 三种自动化的本质区别

| 维度 | RPA（机器人流程自动化） | 工作流自动化 | AI Agent |
|------|----------------------|------------|---------|
| 核心逻辑 | 模拟人类操作（点击、输入、复制） | 通过 API 连接系统 | AI 自主决策+执行 |
| 典型工具 | UiPath、Automation Anywhere、Defy、Bardeen | n8n、Zapier、Make | LangGraph、CrewAI、OpenClaw |
| 需要代码？ | 不需要（录制操作） | 不需要（拖拽连线） | 需要（Python） |
| 灵活性 | 低（固定流程） | 中（条件分支） | 高（自主决策） |
| 稳定性 | 低（UI 变化就崩） | 高（API 稳定） | 中（AI 可能出错） |
| 成本 | 低-中 | 低-高（按量计费） | 高（API 调用费） |
| 适合场景 | 没有 API 的系统（Seller Central 后台操作） | 有 API 的系统间连接 | 需要判断和决策的复杂任务 |

### 1.2 跨境电商卖家怎么选

```
你的自动化需求是什么？

需要操作没有 API 的网页后台？（Seller Central、QuickSight）
└── → RPA（Defy、Bardeen、Browse AI）

需要连接多个有 API 的系统？（Shopify→Google Sheets→Slack）
└── → 工作流自动化（n8n、Zapier、Make）

需要 AI 判断和决策？（分析数据后自动调整策略）
└── → AI Agent（LangGraph + 工作流工具）

预算有限，想免费？
└── → n8n（自托管免费）+ Defy（免费版）

不想折腾，愿意付费？
└── → Zapier（最简单）或 Make（性价比高）
```

---

## 2. 低代码自动化工具全景

### 2.1 工具对比

| 工具 | 类型 | 价格 | 集成数 | 自托管 | AI 集成 | 适合 |
|------|------|------|--------|--------|---------|------|
| **n8n** | 工作流 | 免费（自托管）/ $20/月（云） | 400+ | ✅ | ✅（AI Agent 节点） | 技术型卖家，需要完全控制 |
| **Zapier** | 工作流 | 免费（100 任务/月）/ $20/月起 | 7000+ | ❌ | ✅（AI 步骤） | 非技术卖家，快速上手 |
| **Make** | 工作流 | 免费（1000 操作/月）/ $9/月起 | 1500+ | ❌ | ✅ | 性价比最高，复杂工作流 |
| **Defy** | 浏览器 RPA | 免费版可用 | 浏览器操作 | ❌ | ✅ | 网页后台自动化 |
| **Bardeen** | 浏览器 RPA | 免费版可用 / $10/月 | 浏览器+API | ❌ | ✅ | 数据抓取+自动化 |
| **Browse AI** | 网页抓取 | 免费（50 次/月）/ $49/月 | 网页抓取 | ❌ | ✅ | 竞品监控、价格抓取 |
| **Power Automate** | 工作流+RPA | $15/月起 | Microsoft 生态 | ❌ | ✅（Copilot） | 已用 Microsoft 365 的团队 |

### 2.2 跨境电商场景适配

| 场景 | 最佳工具 | 理由 |
|------|---------|------|
| Seller Central 报告下载 | Defy / Bardeen | 没有 API，需要模拟浏览器操作 |
| 多平台库存同步 | n8n / Make | 需要连接多个 API |
| 新差评通知 | Zapier | 最简单，5 分钟搞定 |
| 竞品价格监控 | Browse AI + n8n | 抓取+处理+通知 |
| 广告报告自动分析 | n8n + OpenAI API | 下载→AI 分析→生成报告 |
| 社交媒体内容排期 | Zapier / Make | 连接 Meta/YouTube API |
| 订单→发货→通知 | n8n / Zapier | 标准工作流 |
| 多语言 Listing 批量生成 | n8n + OpenAI API | 批量调用 AI 翻译 |
| Review 监控+情感分析 | n8n + OpenAI API | 抓取→AI 分析→分类→通知 |
| 月度运营报告自动生成 | n8n + Google Sheets | 汇总数据→生成图表→发送邮件 |

---

## 3. n8n 深度实战

### 3.1 为什么推荐 n8n

n8n 是跨境电商卖家最值得学的自动化工具：

- **免费自托管**：Docker 一键部署，数据完全在自己手里
- **AI 原生集成**：内置 AI Agent 节点，可以直接调用 OpenAI/Claude API
- **400+ 集成**：Shopify、Google Sheets、Slack、Telegram、HTTP Request 等
- **可视化编辑**：拖拽连线，不需要写代码
- **社区活跃**：大量现成的工作流模板可以直接导入

### 3.2 n8n 安装（5 分钟）

```bash
# Docker 一键安装（推荐）
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n

# 打开浏览器访问 http://localhost:5678
```

或者使用 n8n Cloud（免费 14 天试用）：https://n8n.io

### 3.3 电商工作流实战：Review 监控 + AI 分析

```
工作流结构：

[Schedule Trigger] 每小时执行一次
    ↓
[HTTP Request] 抓取 Amazon 产品页面的最新 Review
    ↓
[IF] Review 评分 ≤ 3 星？
    ├── 是 →
    │   [OpenAI] 分析差评内容，提取痛点和情感
    │       ↓
    │   [Google Sheets] 记录到差评追踪表
    │       ↓
    │   [Slack/Telegram] 通知运营团队
    │       ↓
    │   [OpenAI] 生成回复草稿
    │
    └── 否 →
        [Google Sheets] 记录到好评统计表
```

### 3.4 电商工作流实战：多平台库存同步

```
工作流结构（基于 n8n）：

[Webhook] Shopify 订单创建触发
    ↓
[Shopify] 获取订单详情（SKU、数量）
    ↓
[Code] 计算新库存数量
    ↓
[并行执行]
├── [Amazon SP-API] 更新 Amazon 库存
├── [Walmart API] 更新 Walmart 库存
├── [Google Sheets] 更新库存追踪表
└── [Slack] 通知团队库存变化
```

### 3.5 电商工作流实战：广告报告 AI 分析

```
工作流结构：

[Schedule Trigger] 每周一早上 9 点
    ↓
[Amazon SP-API] 下载过去 7 天的搜索词报告
    ↓
[Code] 数据清洗和格式化
    ↓
[OpenAI] 分析报告，生成优化建议
    ↓
[Google Docs] 生成周报文档
    ↓
[Gmail] 发送给团队
```

> 📎 **相关阅读**: [A3 广告优化](../a-operators/a3-advertising.md) — 搜索词报告分析的方法论，可以作为 AI 分析的 Prompt 模板。

---

## 4. Zapier / Make 实战

### 4.1 Zapier：最简单的自动化

Zapier 适合不想折腾的卖家——5 分钟搭建一个自动化：

**示例：新差评 Slack 通知**

```
触发器（Trigger）：Amazon Seller Central → New Review（需要第三方集成）
    ↓
过滤器（Filter）：评分 ≤ 3 星
    ↓
动作（Action）：Slack → 发送消息到 #reviews 频道
    ↓
动作（Action）：Google Sheets → 添加一行到差评追踪表
```

**Zapier 电商常用 Zap：**

| Zap | 触发器 | 动作 | 用途 |
|-----|--------|------|------|
| 新订单通知 | Shopify 新订单 | Slack 消息 | 实时订单监控 |
| 库存预警 | Google Sheets 库存 < 阈值 | Email 通知 | 避免断货 |
| 新 Review 记录 | 第三方 Review 工具 | Google Sheets 记录 | Review 追踪 |
| 社交媒体排期 | Google Sheets 内容日历 | Buffer/Later 发布 | 内容自动发布 |
| 客户反馈收集 | Typeform 提交 | Notion 数据库 | 客户洞察 |

### 4.2 Make（原 Integromat）：性价比之王

Make 比 Zapier 便宜，而且支持更复杂的工作流（分支、循环、错误处理）：

**Make vs Zapier 对比：**

| 维度 | Zapier | Make |
|------|--------|------|
| 免费额度 | 100 任务/月 | 1000 操作/月 |
| 付费起步 | $20/月 | $9/月 |
| 复杂工作流 | 线性为主 | 支持分支/循环/并行 |
| 可视化 | 简单列表 | 画布式拖拽（更直观） |
| 学习曲线 | 极低 | 低 |
| 集成数量 | 7000+ | 1500+ |
| 适合 | 简单自动化 | 复杂工作流 |

---

## 5. 跨境电商 10 大自动化工作流

### 按 ROI 排序的自动化优先级

| 优先级 | 工作流 | 节省时间 | 推荐工具 | 难度 |
|--------|--------|---------|---------|------|
| 1 | 新差评实时通知 | 2 小时/周 | Zapier | ⭐ |
| 2 | 库存低预警 | 3 小时/周 | Zapier / n8n | ⭐ |
| 3 | 竞品价格监控 | 5 小时/周 | Browse AI + n8n | ⭐⭐ |
| 4 | 广告报告自动下载+分析 | 4 小时/周 | n8n + OpenAI | ⭐⭐ |
| 5 | 多平台库存同步 | 3 小时/周 | n8n | ⭐⭐⭐ |
| 6 | 社交媒体内容自动排期 | 5 小时/周 | Zapier / Make | ⭐ |
| 7 | 客服自动回复（常见问题） | 10 小时/周 | n8n + OpenAI | ⭐⭐ |
| 8 | 月度运营报告自动生成 | 8 小时/月 | n8n + Google Sheets | ⭐⭐ |
| 9 | 多语言 Listing 批量生成 | 10 小时/批 | n8n + OpenAI | ⭐⭐ |
| 10 | Review 情感分析+趋势追踪 | 5 小时/周 | n8n + OpenAI | ⭐⭐⭐ |

> **总计**：如果全部实现，每周可节省 40+ 小时。从优先级 1-3 开始，投入最小、回报最快。

---

## 6. RPA 工具与浏览器自动化

### 6.1 为什么需要 RPA

很多电商后台没有 API（或 API 功能有限）：
- Amazon Seller Central 的很多功能没有 SP-API 对应
- QuickSight 报告只能手动下载
- 各平台的后台操作（批量修改价格、上传图片等）

这时候需要 RPA——模拟人类在浏览器中的操作。

### 6.2 Defy

Defy 是一个浏览器 RPA 工具，可以录制和回放浏览器操作：

| 功能 | 说明 |
|------|------|
| 录制操作 | 像录屏一样录制你的浏览器操作 |
| 回放执行 | 自动重复执行录制的操作 |
| 数据提取 | 从网页中提取数据到表格 |
| 定时执行 | 设置定时任务自动运行 |
| AI 辅助 | 用 AI 理解页面结构，更稳定 |

**电商应用场景：**
- 批量下载 Seller Central 报告
- 批量修改产品价格
- 批量上传产品图片
- 竞品页面数据抓取

### 6.3 Bardeen

Bardeen 是另一个浏览器自动化工具，更偏向数据抓取和工作流：

| 功能 | 说明 |
|------|------|
| 网页抓取 | 从任何网页提取结构化数据 |
| 工作流 | 连接浏览器操作和 API |
| AI 集成 | 内置 AI 处理抓取的数据 |
| 模板库 | 大量现成的自动化模板 |

**电商应用场景：**
- 抓取竞品 Review 数据
- 抓取竞品价格和库存状态
- 自动填写各平台的产品信息
- LinkedIn 达人信息抓取（用于达人合作）

### 6.4 Browse AI

Browse AI 专注于网页数据抓取和监控：

| 功能 | 说明 |
|------|------|
| 无代码抓取 | 点击选择要抓取的数据 |
| 定时监控 | 定期抓取并对比变化 |
| 变化通知 | 数据变化时自动通知 |
| API 输出 | 抓取结果可以通过 API 获取 |

**电商应用场景：**
- 竞品价格监控（每天抓取，价格变化时通知）
- 竞品新品监控（发现新上架产品）
- BSR 排名追踪
- Review 数量和评分追踪

---

## 7. AI + 自动化的融合

### 7.1 AI 在自动化工作流中的角色

```
传统自动化：触发 → 固定流程 → 输出
AI 增强自动化：触发 → AI 分析/判断 → 动态流程 → 输出

示例：Review 监控工作流

传统版：
新 Review → 评分 ≤ 3？→ 通知团队

AI 增强版：
新 Review → AI 分析情感和主题 → 
├── 产品质量问题 → 通知产品团队 + 生成改进建议
├── 物流问题 → 通知物流团队 + 检查 FBA 库存
├── 使用方法问题 → 生成 FAQ 更新建议
└── 恶意差评 → 标记 + 生成申诉草稿
```

### 7.2 n8n + OpenAI API 集成

n8n 内置了 OpenAI 节点，可以直接在工作流中调用 AI：

```
n8n AI 节点类型：

1. OpenAI Chat Model — 调用 GPT-4/GPT-4o
2. AI Agent — 让 AI 自主决策下一步操作
3. AI Chain — 多步 AI 处理链
4. AI Memory — 给 AI 添加记忆
5. AI Tool — 让 AI 调用外部工具

电商工作流中的 AI 节点用法：
├── 文本分析：Review 情感分析、关键词提取
├── 内容生成：Listing 文案、广告文案、客服回复
├── 数据分析：报告摘要、趋势识别、异常检测
├── 翻译：多语言 Listing 生成
└── 决策：基于数据自动调整策略
```

### 7.3 AI Prompt 模板（用于自动化工作流）

```
你是一个跨境电商运营 AI 助手，正在自动化工作流中被调用。

输入数据：
{{$json.review_text}}

请分析这条 Review：
1. 情感：正面/中性/负面
2. 主题分类：产品质量/物流/使用方法/价格/其他
3. 关键痛点（如果是负面）
4. 建议的回复草稿（如果是负面）
5. 是否需要人工介入：是/否

输出格式：JSON
```

---

## 8. 工具选择决策框架

```
你是一个跨境电商自动化顾问。

我的情况：
- 团队规模：[X] 人
- 技术能力：[无代码/会用 Excel/会写 Python]
- 月预算（自动化工具）：$[X]
- 主要平台：[Amazon/Shopify/Walmart/...]
- 最想自动化的 3 个任务：[列出]

请推荐：
1. 最适合我的自动化工具组合
2. 每个工具的具体用途
3. 实施优先级（先做什么）
4. 预估每周节省的时间
5. 第一个月的行动计划
```

---

## 9. 完成标志

- [ ] 理解 RPA、工作流自动化、AI Agent 的区别和适用场景
- [ ] 安装并运行 n8n（Docker 或 Cloud）
- [ ] 搭建至少 1 个自动化工作流（推荐：新差评通知）
- [ ] 尝试在工作流中集成 AI（OpenAI API）
- [ ] 制定你的自动化优先级清单

> **下一步**：如果你想深入构建 AI Agent 系统，进入 [Path B: B4 AI Agent 与自动化](../b-developers/b4-agent-workflow.md)。如果你想先用好现有工具，回到 [Path A](../a-operators/) 把 AI 应用到具体运营场景。

---
> 🏠 [Hub 首页](../../README.md) · 📋 [Path 0 总览](README.md) · 📊 [AI 全景评估](ai-landscape.md)
> 
> **Path 0**: [F1 AI 演进](f1-ai-evolution.md) · [F2 Prompt 工程](f2-prompt-engineering.md) · [F3 RAG 知识库](f3-rag-knowledge.md) · [F4 Agent 自动化](f4-agent-automation.md) · [F5 RPA 自动化](f5-rpa-automation.md) · [AI 全景](ai-landscape.md)
> 
> **快速跳转**: [Path A 运营](../a-operators/) · [Path B 技术](../b-developers/) · [Path C 管理](../c-managers/) · [Path D 多平台](../d-platforms/) · [Path E 社交媒体](../e-social-media/)