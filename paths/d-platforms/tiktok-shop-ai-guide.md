# D2. TikTok Shop AI 实战指南 | TikTok Shop AI Playbook

> **路径**: Path D: 多平台 · **模块**: D2  
> **最后更新**: 2026-03-13  
> **难度**: ⭐⭐ 中级  
> **预计时间**: 2-3 小时  
> **前置模块**: [Path 0 基础](../0-foundations/) · [AI 全景评估](../0-foundations/ai-landscape.md)

🏠 [Hub 首页](../../README.md) · 📋 [Path D 总览](README.md)

---

## 📖 本模块章节导航

1. [TikTok Shop vs Amazon vs Shopify](#1-tiktok-shop-vs-amazon-vs-shopify) · 2. [短视频内容创作](#2-ai-短视频内容创作) · 3. [达人合作与匹配](#3-达人合作与-ai-匹配) · 4. [直播电商](#4-直播电商与-ai) · 5. [商品优化](#5-商品页面与-seo-优化) · 6. [广告投放](#6-tiktok-ads-ai-优化) · 7. [数据分析](#7-数据分析与运营优化) · 8. [Prompt 模板](#8-prompt-模板tiktok-shop-专用) · 9. [AI 工具全景](#9-ai-工具全景) · 10. [常见陷阱](#10-常见陷阱) · 11. [案例分析](#11-案例分析) · 12. [🦞 OpenClaw 自动化](#12-用-openclaw-自动化-tiktok-shop-运营) · 13. [完成标志](#13-完成标志)

---

## 本模块你将产出

一套完整的 TikTok Shop AI 运营工作流。完成后你将拥有：

- 一套 AI 驱动的短视频批量生产流程（从脚本到成片）
- 一套达人筛选和匹配的 AI 方法论
- 一套直播脚本和话术的 AI 生成方案
- 一套 TikTok Ads 的 AI 优化策略
- 一套 TikTok Shop 专用的 Prompt 模板库

> 💡 **核心理念**：TikTok Shop 是"内容驱动"的电商，和 Amazon（搜索驱动）、Shopify（品牌驱动）完全不同。AI 在 TikTok 的核心价值是内容生产效率 — 谁能用 AI 更快地生产更多优质短视频，谁就赢。


---

## 1. TikTok Shop vs Amazon vs Shopify

### 1.1 三大平台核心差异

| 维度 | Amazon | Shopify | TikTok Shop |
|------|--------|---------|-------------|
| **流量逻辑** | 搜索意图（用户主动找产品） | 站外引流（SEO/广告/邮件） | 算法推荐（内容触发兴趣） |
| **购买决策** | 理性对比（Review/价格/参数） | 品牌信任（故事/设计/口碑） | 冲动消费（视频种草/直播氛围） |
| **内容形态** | 图文 Listing（格式固定） | 产品页（自由设计） | 短视频 + 直播（15-60秒决胜） |
| **竞争核心** | 关键词排名 + Review 数量 | 品牌差异化 + CAC 控制 | 内容质量 + 发布频率 + 达人矩阵 |
| **AI 核心价值** | Listing SEO + Review 分析 | 广告创意 + 邮件个性化 | 视频批量生产 + 达人匹配 + 直播脚本 |
| **数据获取** | Seller Central 报告 | GA4 + Shopify Analytics | TikTok Seller Center + Creator Marketplace |
| **复购机制** | Subscribe & Save | 邮件 + 会员 | 粉丝关注 + 直播间复购 |
| **利润结构** | 佣金 15% + FBA | 支付 2.9% + 月租 | 佣金 2-8% + 运费补贴（新卖家） |

### 1.2 TikTok Shop 的 AI 独特优势

**优势一：内容生产可以完全 AI 化**

TikTok 的核心是短视频。AI 可以：
- 自动生成视频脚本（痛点→产品→CTA 的 15 秒结构）
- 自动剪辑产品展示视频（CapCut AI 一键成片）
- 自动生成多语言字幕和配音
- 批量生产变体（同一产品 20+ 不同角度的视频）

**优势二：达人匹配可以数据驱动**

TikTok Creator Marketplace 提供达人数据。AI 可以：
- 基于产品属性自动筛选匹配达人
- 预测达人合作的 ROI（基于历史数据）
- 自动生成达人邀约话术
- 批量管理 100+ 达人合作

**优势三：算法友好 = 内容量友好**

TikTok 算法不看你有多少粉丝，看你的内容质量。AI 帮你：
- 每天发布 3-5 条视频（人工做不到，AI 可以）
- 快速测试不同内容角度（哪个 hook 最有效）
- 追踪趋势并快速跟进（热门音乐/话题/格式）

Content rephrased for compliance with licensing restrictions. Sources: [TikTok Shop Automation 2026](https://iterathon.tech/blog/tiktok-shop-instagram-shopping-automation-2026), [Influencer Marketing Hub](https://influencermarketinghub.com/tiktok-influencer-marketing-platforms/)


---

## 2. AI 短视频内容创作

### 2.1 TikTok 爆款视频的结构公式

```
前 3 秒：Hook（抓住注意力，决定用户是否继续看）
├── 痛点型："你是不是也遇到过 [问题]？"
├── 反差型："我花了 $200 买了这个，结果..."
├── 数据型："90% 的人不知道 [事实]"
└── 悬念型："看到最后你会感谢我"

3-15 秒：产品展示（展示产品如何解决问题）
├── 使用场景演示
├── Before/After 对比
├── 开箱/拆包
└── 功能特写

15-25 秒：社会证明 + 卖点强化
├── 用户评价/UGC
├── 销量数据
├── 专业背书
└── 限时优惠

最后 3 秒：CTA（引导行动）
├── "点击下方小黄车"
├── "评论区告诉我你想要什么颜色"
├── "关注我看更多好物推荐"
└── "限时 XX 折，手慢无"
```

### 2.2 AI 视频脚本生成 Prompt

**为什么这个 Prompt 有效：** 它要求 AI 按照 TikTok 的 Hook→展示→CTA 结构生成脚本，并且指定了时长和风格，确保输出可以直接用于拍摄。

```
你是一个 TikTok 短视频创意专家，专注于电商带货视频。

产品信息：
- 产品名：[名称]
- 核心卖点：[3 个]
- 价格：$[X]（原价 $[X]）
- 目标受众：[年龄、性别、兴趣]
- 视频风格：[真人出镜/产品特写/开箱/对比/UGC风格]

请生成 5 个 15-30 秒的视频脚本：

每个脚本包含：
1. Hook（前 3 秒的台词/画面，必须在 3 秒内抓住注意力）
2. 正文（产品展示方式 + 台词/旁白）
3. CTA（引导点击购买的话术）
4. 屏幕文字叠加（每个画面上显示的关键文字）
5. 推荐背景音乐类型（节奏感强/温馨/搞笑/紧迫感）
6. 拍摄建议（镜头角度、场景、道具）

5 个脚本分别用不同角度：
- 脚本 A：痛点共鸣型
- 脚本 B：Before/After 对比型
- 脚本 C：开箱惊喜型
- 脚本 D：用户证言/UGC 型
- 脚本 E：限时紧迫型
```

### 2.3 AI 视频制作工具链

| 环节 | 推荐工具 | AI 功能 | 月费 |
|------|---------|---------|------|
| 脚本生成 | ChatGPT/Claude | 批量生成视频脚本和文案 | $20 |
| 视频剪辑 | CapCut（AI 功能） | 自动剪辑、字幕、特效、模板 | 免费-$8 |
| AI 配音 | ElevenLabs / CapCut TTS | 多语言 AI 配音、克隆声音 | 免费-$22 |
| 产品视频 | Synthesia / HeyGen | AI 数字人出镜讲解产品 | $24-$59 |
| 图片转视频 | Runway ML / Pika | 产品图片生成动态视频 | $12-$28 |
| 字幕翻译 | CapCut 自动字幕 | 多语言字幕自动生成 | 免费 |
| 趋势追踪 | TrendTok / Exolyt | AI 分析热门话题和音乐 | $10-$30 |

### 2.4 批量视频生产工作流

```
Step 1：内容规划（AI 辅助，每周 30 分钟）
├── 用 AI 分析本周 TikTok 热门趋势（话题/音乐/格式）
├── 用 AI 生成 15-20 个视频脚本（5 个产品 × 4 个角度）
├── 筛选 Top 10 脚本进入制作
└── 输出：本周内容日历

Step 2：素材准备（1-2 小时）
├── 产品实拍素材（可复用）
├── AI 生成的产品场景图
├── 用户 UGC 素材（如有）
└── 输出：素材库

Step 3：视频制作（AI 辅助，每个视频 10-15 分钟）
├── CapCut AI 自动剪辑（选模板 → 导入素材 → 一键成片）
├── AI 配音 + 自动字幕
├── 添加文字叠加和特效
└── 输出：10+ 成品视频

Step 4：发布与优化（每天 15 分钟）
├── 按最佳时间发布（AI 建议）
├── 监控前 2 小时数据（播放量/完播率/互动率）
├── 表现好的视频 → 投 Spark Ads 放量
├── 表现差的视频 → 分析原因，调整下一批
└── 输出：每天 3-5 条视频稳定发布
```

> 💡 **关键指标**：TikTok 算法最看重的是完播率（>40% 算好）和互动率（>5% 算好）。AI 帮你快速测试不同 Hook，找到完播率最高的开头。

Content rephrased for compliance with licensing restrictions. Sources: [EComposer AI TikTok Generators](https://ecomposer.io/blogs/tool-software/ai-tiktok-video-generators), [Benly TikTok Ads Tools](https://benly.ai/learn/ai-marketing/best-tiktok-ads-tools-2026)


---

## 3. 达人合作与 AI 匹配

### 3.1 达人合作模式

| 模式 | 说明 | 适合谁 | AI 辅助 |
|------|------|--------|---------|
| Affiliate（联盟） | 达人带货赚佣金，0 前期成本 | 所有卖家 | AI 批量筛选和邀约 |
| Paid Collaboration | 付费合作，固定费用 + 佣金 | 有预算的品牌 | AI 预测 ROI |
| Seeding（寄样） | 免费寄产品，达人自愿发布 | 新品推广 | AI 筛选高回复率达人 |
| Brand Ambassador | 长期合作，深度绑定 | 成熟品牌 | AI 分析达人粉丝画像匹配度 |

### 3.2 AI 达人筛选 Prompt

```
你是一个 TikTok 达人合作专家。请帮我筛选适合推广以下产品的达人。

产品信息：
- 产品：[名称和简述]
- 价格：$[X]
- 目标市场：[US/UK/全球]
- 目标受众：[年龄、性别、兴趣]
- 合作预算：$[X]/月
- 合作模式：[Affiliate/Paid/Seeding]

请输出达人筛选标准：
1. 粉丝量范围（建议 Nano/Micro/Mid/Macro 哪个层级）
2. 内容类型匹配（哪些内容标签/话题最相关）
3. 数据指标阈值（最低互动率、完播率、带货转化率）
4. 红旗信号（哪些达人应该避免）
5. 邀约话术模板（3 个变体：正式/轻松/利益驱动）
6. 合作 Brief 模板（给达人的拍摄指南）
```

### 3.3 达人层级策略

| 层级 | 粉丝量 | 合作成本 | 优势 | AI 辅助重点 |
|------|--------|---------|------|-----------|
| Nano（1K-10K） | $0-50/条 | 性价比高、真实感强 | AI 批量筛选 + 自动邀约 |
| Micro（10K-100K） | $50-500/条 | 垂直精准、互动率高 | AI 分析内容匹配度 |
| Mid（100K-500K） | $500-5K/条 | 覆盖面广、有影响力 | AI 预测 ROI + 谈判建议 |
| Macro（500K+） | $5K+/条 | 品牌背书、大曝光 | AI 分析粉丝画像重合度 |

> 💡 **实战建议**：跨境电商卖家最佳策略是"100 个 Nano + 20 个 Micro"而不是"1 个 Macro"。AI 让你能同时管理 100+ 达人合作。

---

## 4. 直播电商与 AI

### 4.1 TikTok 直播的 AI 应用场景

| 场景 | AI 能做什么 | 工具 |
|------|-----------|------|
| 直播脚本 | 生成完整的直播话术（开场→产品介绍→互动→逼单→收尾） | ChatGPT/Claude |
| 实时字幕 | 多语言实时字幕翻译 | TikTok 内置 / CapCut Live |
| 弹幕分析 | 实时分析观众情绪和问题，提示主播回应 | 定制工具 |
| 数据复盘 | 分析直播数据（观看人数曲线、转化节点、流失点） | TikTok Seller Center + AI |
| 虚拟主播 | AI 数字人 24 小时直播（适合标品） | HeyGen / D-ID |

### 4.2 直播脚本 AI 生成 Prompt

```
你是一个 TikTok 直播带货脚本专家。请为以下产品生成一场 30 分钟的直播脚本。

产品信息：
- 产品：[名称]（共 [X] 个 SKU）
- 价格：$[X]-$[X]
- 核心卖点：[3 个]
- 直播优惠：[描述]
- 目标 GMV：$[X]

请输出完整直播脚本：

**开场（0-3 分钟）**
- 欢迎话术 + 今日福利预告
- 引导关注 + 互动（"扣 1 想要的打 1"）

**产品介绍（3-20 分钟）**
- 每个 SKU 的介绍话术（痛点→演示→价格→限时优惠）
- 互动节点设计（每 5 分钟一个互动环节）
- 逼单话术（"库存只剩 XX 件"、"这个价格只有今天"）

**高潮（20-25 分钟）**
- 秒杀/抽奖环节
- 最大力度优惠释放

**收尾（25-30 分钟）**
- 总结今日福利
- 预告下次直播
- 引导关注 + 加粉丝群
```

---

## 5. 商品页面与 SEO 优化

### 5.1 TikTok Shop 商品页 vs Amazon Listing

| 元素 | Amazon | TikTok Shop |
|------|--------|-------------|
| 标题 | 关键词密集（COSMO 语义匹配） | 简短吸引（<80 字符，像短视频标题） |
| 图片 | 白底主图 + 场景图 | 生活场景为主（像社交媒体帖子） |
| 视频 | 可选（A+ Video） | 必须（视频是核心转化元素） |
| 描述 | 详细参数 + 卖点 | 简短 + 情感化（像朋友推荐） |
| SEO | COSMO/Rufus 语义优化 | TikTok 站内搜索 + 话题标签 |

### 5.2 TikTok Shop 商品优化 Prompt

```
你是一个 TikTok Shop 商品优化专家。请优化以下产品的 TikTok Shop 页面。

产品：[名称]
品类：[类型]
目标受众：[年龄、兴趣]
当前转化率：[X]%

请输出：
1. 产品标题（<80 字符，吸引点击，含热门搜索词）
2. 产品描述（200 字内，口语化，像朋友推荐）
3. 5 个产品标签（热门话题标签）
4. 主图建议（什么样的图片在 TikTok 点击率最高）
5. 视频封面建议（什么样的封面让人想点进去看）
6. 价格策略建议（TikTok 用户对价格的敏感度 vs Amazon）
```

---

## 6. TikTok Ads AI 优化

### 6.1 TikTok 广告类型

| 广告类型 | 适合阶段 | AI 辅助 | 预算建议 |
|----------|---------|---------|---------|
| In-Feed Ads | 品牌认知 + 转化 | AI 生成视频素材 + 文案 | $50+/天 |
| Spark Ads | 放大优质内容 | AI 识别高潜力自然内容 | $30+/天 |
| Shopping Ads | 直接转化 | AI 优化产品 Feed | $30+/天 |
| GMV Max | 全自动化 | TikTok AI 自动优化全链路 | $100+/天 |
| Live Shopping Ads | 直播引流 | AI 优化直播间投放时机 | $50+/天 |

### 6.2 GMV Max 策略（TikTok 的 AI 自动化广告）

GMV Max 是 TikTok 2025 年推出的全自动化广告产品，类似 Google 的 Performance Max。卖家只需要提供产品和预算，TikTok AI 自动优化：

```
GMV Max 工作原理：
1. 卖家设置：产品目录 + 日预算 + 目标 ROAS
2. TikTok AI 自动：
   ├── 选择最佳广告格式（In-Feed/Shopping/Live）
   ├── 选择最佳受众（基于产品属性和历史数据）
   ├── 选择最佳素材（从你的视频库中选）
   ├── 实时调整出价和预算分配
   └── 跨渠道优化（For You/搜索/商城/直播）
3. 卖家需要做的：持续提供新的视频素材（AI 帮你选最好的）
```

> 💡 **关键洞察**：GMV Max 的效果取决于你提供的视频素材数量和质量。AI 帮你批量生产视频 → GMV Max 帮你选最好的投放 → 形成正循环。

Content rephrased for compliance with licensing restrictions. Source: [Benly TikTok Ads Tools 2026](https://benly.ai/learn/ai-marketing/best-tiktok-ads-tools-2026)

---

## 7. 数据分析与运营优化

### 7.1 TikTok Shop 关键指标

| 指标类别 | 核心指标 | 健康基准 | AI 监控 |
|----------|---------|---------|---------|
| 内容 | 视频完播率 | >40% | AI 分析哪个 Hook 最有效 |
| 内容 | 视频互动率 | >5% | AI 识别高互动内容模式 |
| 转化 | 商品点击率 | >3% | AI 优化商品页面 |
| 转化 | 下单转化率 | >2% | AI 分析转化漏斗 |
| 达人 | 达人带货 ROI | >3x | AI 筛选高 ROI 达人 |
| 直播 | 直播间停留时长 | >3 分钟 | AI 分析流失节点 |
| 广告 | 广告 ROAS | >2x | AI 优化投放策略 |

### 7.2 数据分析 Prompt

```
你是一个 TikTok Shop 数据分析师。请分析以下店铺数据并给出优化建议。

店铺数据（过去 30 天）：
- 总 GMV：$[X]
- 订单数：[X]
- 视频发布数：[X]
- 平均视频播放量：[X]
- 平均完播率：[X]%
- 达人合作数：[X]
- 达人带货 GMV 占比：[X]%
- 直播场次：[X]
- 直播 GMV 占比：[X]%
- 广告花费：$[X]，ROAS：[X]

请输出：
1. 各渠道 GMV 贡献分析（自然流量/达人/直播/广告）
2. 内容效率分析（哪类视频表现最好/最差）
3. 达人合作 ROI 排名（哪些达人值得加大合作）
4. 广告效率分析（哪个广告类型 ROAS 最高）
5. Top 3 增长机会
6. Top 2 风险预警
7. 下月运营计划建议
```


---

## 8. Prompt 模板（TikTok Shop 专用）

### 8.1 爆款视频脚本批量生成

```
产品：[名称]，卖点：[3个]，价格：$[X]
请生成 10 个 15 秒 TikTok 视频的 Hook（前 3 秒台词），分别用以下角度：
痛点×2、反差×2、数据×2、悬念×2、挑战×1、教程×1
每个 Hook 标注预期完播率（高/中/低）和适合的拍摄方式。
```

### 8.2 达人邀约话术

```
我是 [品牌名] 的合作经理。我们的产品是 [简述]，在 TikTok Shop 上售价 $[X]。
请生成 3 个达人邀约 DM 话术：
- 版本 A：正式专业（适合 Mid-Macro 达人）
- 版本 B：轻松友好（适合 Nano-Micro 达人）
- 版本 C：利益驱动（强调佣金和免费样品）
每个版本 <100 字，包含合作方式和下一步行动。
```

### 8.3 直播间互动话术

```
产品：[名称]，直播时长：[X] 分钟
请生成以下直播互动话术：
1. 开场破冰（让观众停留的前 30 秒）
2. 产品介绍过渡语（自然引入产品）
3. 互动引导（5 个让观众评论/点赞的话术）
4. 逼单话术（3 个制造紧迫感的方式）
5. 冷场救场（观众互动低时的 3 个应急话术）
```

### 8.4 竞品 TikTok 内容分析

```
请分析以下 TikTok Shop 竞品的内容策略：
竞品账号：[@账号名]
品类：[类型]

请从以下维度分析：
1. 发布频率和时间规律
2. 视频类型分布（产品展示/教程/UGC/直播切片）
3. 最高播放量视频的共同特征（Hook 类型、时长、音乐）
4. 达人合作策略（合作达人数量、层级、频率）
5. 直播策略（频率、时长、GMV 估算）
6. 我们可以学习的 3 个点
7. 我们可以差异化的 3 个点
```

---

## 9. AI 工具全景

| 类别 | 工具 | 功能 | 月费 |
|------|------|------|------|
| 视频脚本 | ChatGPT/Claude | 批量生成脚本和文案 | $20 |
| 视频剪辑 | CapCut AI | 自动剪辑、字幕、模板 | 免费-$8 |
| AI 配音 | ElevenLabs | 多语言 AI 配音 | 免费-$22 |
| 数字人 | HeyGen / Synthesia | AI 虚拟主播 | $24-$59 |
| 达人管理 | KOL Sprite | AI 达人筛选和管理 | $49+ |
| 趋势分析 | Exolyt / TrendTok | TikTok 趋势追踪 | $10-$30 |
| 广告优化 | TikTok Ads Manager | GMV Max 自动化 | 按广告花费 |
| 数据分析 | Kalodata / FastMoss | TikTok Shop 数据分析 | $30-$100 |

Content rephrased for compliance with licensing restrictions. Sources: [KOL Sprite](https://kolsprite.com/blog/tiktok-creator-collaboration-ai-automation-data-2025), [EComposer](https://ecomposer.io/blogs/tool-software/ai-tiktok-video-generators)

---

## 10. 常见陷阱

| 陷阱 | 表现 | 正确做法 |
|------|------|----------|
| 用 Amazon 思维做 TikTok | 发产品参数图、白底图 | TikTok 要的是生活场景、真人使用、有趣内容 |
| 视频质量过高 | 花大钱拍专业广告片 | TikTok 用户更信任"真实感"，手机拍的 UGC 风格反而转化更高 |
| 发布频率太低 | 一周发 1-2 条 | 至少每天 1 条，理想 3-5 条，AI 帮你批量生产 |
| 只做自然流量 | 不投广告等自然爆发 | 自然流量不可控，Spark Ads 放大优质内容是标配 |
| 达人合作一次性 | 合作一次就结束 | 建立长期达人矩阵，持续产出内容 |
| 忽略直播 | 只做短视频不做直播 | 直播是 TikTok Shop GMV 的主要来源（占 60%+） |

---

## 11. 案例分析

### 11.1 案例：从 0 到月 GMV $100K 的 TikTok Shop 打法

**背景：** 美妆品牌，从 Amazon 扩展到 TikTok Shop US

| 阶段 | 时间 | 策略 | AI 辅助 | GMV |
|------|------|------|---------|-----|
| 冷启动 | 第 1 月 | 每天 3 条短视频 + 50 个 Nano 达人寄样 | AI 生成全部脚本 + 批量邀约 | $5K |
| 放量 | 第 2 月 | Spark Ads 放大爆款 + 20 个 Micro 达人付费合作 | AI 识别高潜力视频 + 达人 ROI 预测 | $25K |
| 直播 | 第 3 月 | 每周 3 场直播 + GMV Max 广告 | AI 生成直播脚本 + 自动化广告 | $60K |
| 稳定 | 第 4 月 | 达人矩阵 100+ + 日常直播 + 自然流量占比提升 | AI 全链路管理 | $100K |

**关键数据：**
- 视频总发布量：300+（AI 生成脚本，人工拍摄 + CapCut 剪辑）
- 达人合作总数：120+（AI 批量筛选和管理）
- 广告 ROAS：2.8（GMV Max）
- 自然流量占比：从 10% 提升到 35%

---

## 12. 用 OpenClaw 自动化 TikTok Shop 运营

### 12.1 场景：AI Agent 自动化达人管理和内容分析

```
你对 OpenClaw 说：
"每天自动分析 TikTok Shop 的视频数据和达人带货数据，
识别高潜力内容和高 ROI 达人，生成运营建议"

OpenClaw 自动执行：
1. [Heartbeat] 每天 9:00 触发
2. [Skill: web-search] 抓取 TikTok 热门趋势
3. [Skill: google-sheets] 读取视频发布和达人合作数据
4. [LLM] 分析内容表现 + 达人 ROI + 趋势匹配
5. [Skill: slack] 发送日报到 #tiktok-ops
6. [Heartbeat] 每周一生成周度内容规划
```

### 12.2 需要的 Skills

| 组件 | 用途 | 链接 |
|------|------|------|
| **google-sheets** Skill | 读写运营数据 | [ClawHub](https://clawhub.ai/) |
| **slack** Skill | 发送报告和预警 | [ClawHub](https://clawhub.ai/) |
| **web-search** Skill | 趋势追踪 | [ClawHub](https://clawhub.ai/) |
| **memory** Skill | 存储历史数据 | [OpenClaw](https://openclaw.com/) |

---

## 13. 完成标志

- [ ] 理解 TikTok Shop 和 Amazon/Shopify 的核心差异
- [ ] 用 AI 生成至少 10 个短视频脚本（不同角度）
- [ ] 用 AI 生成达人邀约话术并完成至少 5 个达人邀约
- [ ] 用 AI 生成一场完整的直播脚本
- [ ] 设置至少一个 TikTok 广告（Spark Ads 或 Shopping Ads）
- [ ] 用 AI 分析一次 TikTok Shop 数据并生成优化建议

---

## 附录：快速参考

### TikTok vs Amazon vs Shopify AI 应用速查

| AI 场景 | Amazon | Shopify | TikTok Shop |
|---------|--------|---------|-------------|
| 内容生成 | Listing 文案 | 产品页 + 博客 | 短视频脚本 + 直播话术 |
| 广告 | PPC 关键词优化 | Facebook/Google Ads | Spark Ads + GMV Max |
| 客户触达 | 站内消息（受限） | 邮件 + SMS | 短视频 + 直播 + 粉丝群 |
| 达人合作 | 几乎没有 | 有限 | 核心策略（占 GMV 40%+） |
| 数据分析 | Seller Central | GA4 + Shopify | TikTok Seller Center |

---

⬅️ [返回 Path D 总览](README.md) | 🏠 [返回 Hub 首页](../../README.md) | ⬅️ [D1 Shopify 指南](shopify-ai-guide.md)
