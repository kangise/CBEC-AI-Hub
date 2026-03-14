# CBEC-AI-Hub 内容扩展规划

> 创建日期: 2026-03-14
> 状态: 待 Ken 确认后执行

## 设计原则

1. **差异化优先**：每个平台/渠道只写与 Amazon 不同的部分
2. **通用内容回填**：如果某个 AI 场景是通用的但 Path A 没覆盖，补充到 Path A 对应模块
3. **新建 Path E**：社交媒体 AI 运营独立成一个新路径
4. **Path D 扩展**：电商平台差异化内容继续放在 Path D

---

## 一、新建 Path E: 社交媒体 AI 运营

路径: `paths/e-social-media/`

### 文件结构

```
paths/e-social-media/
├── README.md                           # Path E 总览 + 平台对比速览
├── e1-instagram-facebook-ai-guide.md   # Meta 生态 AI 运营（合并）
├── e2-youtube-ai-guide.md              # YouTube AI 运营
├── e3-xiaohongshu-ai-guide.md          # 小红书 AI 运营
├── e4-pinterest-ai-guide.md            # Pinterest AI 运营
├── e5-whatsapp-business-ai-guide.md    # WhatsApp Business AI 客服与营销
├── e6-reddit-ai-guide.md              # Reddit AI 营销
└── e7-social-media-cross-channel.md    # 社交媒体跨渠道协同策略
```

### E1. Instagram + Facebook AI 运营指南（Meta 生态）

**为什么合并**: 共用 Meta Ads Manager、Meta Business Suite、相同的广告系统

**核心差异化内容（不与 Amazon/Shopify 重复）**:

1. **Reels/Stories AI 内容创作方法论**
   - Hook 设计（与 TikTok 的区别：Instagram 更偏生活方式/美学，TikTok 偏娱乐/信息缺口）
   - AI 生成 Reels 脚本的 Prompt 模板（产品展示、使用场景、Before/After）
   - Carousel 内容策略：AI 批量生成教育型/对比型轮播图文案
   - Stories 互动设计：投票、问答、倒计时的 AI 文案模板

2. **Instagram Shopping 深度实操**
   - Shoppable Reels/Stories 标签策略（带货标签互动率+30% 的数据支撑）
   - Product Catalog 优化：AI 生成产品描述适配 Instagram 风格（短、视觉化、emoji）
   - Instagram Shop 页面布局优化
   - AR 试穿/试用功能的品类适用性分析

3. **Meta Advantage+ AI 广告深度指南**
   - Advantage+ Shopping Campaigns (ASC) 设置与优化
   - AI 动态创意优化（DCO）：素材组合测试方法论
   - Advantage+ Audience：AI 受众扩展 vs 手动定向的决策框架
   - 与 Amazon Attribution 的归因联动
   - 广告素材 AI 批量生成工作流（Midjourney/DALL-E → Canva → Meta Ads）

4. **Facebook Groups/Community 运营**
   - AI 辅助社群内容规划（每周内容日历生成）
   - 用户反馈自动分析（情感分析 + 产品改进建议提取）
   - Facebook Marketplace 本地化策略（特定品类）

5. **Meta 生态数据分析**
   - Meta Pixel + Conversions API 数据解读
   - AI 辅助广告报告分析 Prompt 模板
   - Instagram Insights 数据导出与趋势分析

**预计篇幅**: 8000-10000 字
**预计时间**: 2-3 小时阅读

---

### E2. YouTube AI 运营指南

**核心差异化内容**:

1. **YouTube 搜索 SEO 方法论（与 Amazon SEO 的本质区别）**
   - YouTube 算法机制：推荐系统 vs 搜索排名的双引擎
   - AI 关键词研究：vidIQ/TubeBuddy + ChatGPT 组合工作流
   - 标题/描述/标签的 AI 优化 Prompt 模板
   - 缩略图文案 A/B 测试策略

2. **长视频 AI 内容创作（YouTube 独有）**
   - 产品评测视频脚本结构（与 TikTok 短视频的区别：深度 vs 速度）
   - AI 生成 10-15 分钟评测脚本的 Prompt 模板
   - 开箱视频、对比评测、使用教程的差异化脚本框架
   - 章节标记（Chapters）AI 自动生成

3. **YouTube Shorts 电商化**
   - Shorts 与 TikTok/Reels 的算法差异
   - AI 批量生成 Shorts 脚本（从长视频切片 + 原创）
   - Shorts 购物标签和产品链接策略

4. **YouTube Shopping 与 Affiliate**
   - YouTube Shopping 功能设置（2026 与 Rakuten 合作）
   - 产品标记和购物卡片优化
   - YouTube Affiliate Program：达人合作的 AI 筛选模型
   - 与 Amazon Associates / Shopify Collabs 的联动

5. **YouTube 广告 AI 优化**
   - Video Action Campaigns (VAC) → Demand Gen campaigns 迁移
   - AI 生成广告脚本（6秒 Bumper / 15秒 Non-skip / 可跳过广告）
   - 受众信号 vs 自动定向的决策框架
   - YouTube 广告与 Google Shopping 的协同

6. **数据分析与优化**
   - YouTube Analytics 关键指标解读（CTR、AVD、RPM）
   - AI 辅助频道诊断 Prompt 模板
   - 竞品频道分析方法论

**预计篇幅**: 10000-12000 字
**预计时间**: 3-4 小时阅读

---

### E3. 小红书 AI 运营指南

**核心差异化内容（完全独特的平台）**:

1. **小红书平台机制与算法**
   - CES 评分机制（点赞1分+收藏1分+评论4分+转发4分+关注8分）
   - 流量分发逻辑：发现页推荐 vs 搜索 vs 关注页
   - 与 Instagram/TikTok 的本质区别：种草决策平台 vs 娱乐平台
   - 用户画像：79% 女性、高消费力、搜索渗透率 70%

2. **AI 种草笔记创作方法论**
   - 笔记类型矩阵：好物分享/教程/测评/合集/避雷
   - AI 生成种草文案的 Prompt 模板（适配小红书语气：真实、口语化、emoji 密集）
   - 封面图设计策略：AI 生成封面文案 + Canva/美图秀秀模板
   - 标题公式：数字+痛点+解决方案（与 Amazon Listing 标题的区别）
   - 标签策略：热门标签 + 长尾标签的 AI 组合

3. **小红书 SEO（站内搜索优化）**
   - 关键词布局：标题+正文前 200 字+标签
   - AI 关键词研究：小红书搜索联想 + 笔记热词分析
   - 与 Google SEO / Amazon SEO 的方法论对比

4. **KOL/KOC 合作 AI 方法论**
   - 达人筛选模型（与 TikTok 达人筛选的区别：小红书更看重笔记质量而非粉丝量）
   - AI 分析达人数据：互动率、粉丝画像、内容调性匹配
   - 蒲公英平台使用指南
   - 素人铺量 vs 腰部达人 vs 头部 KOL 的预算分配

5. **小红书电商闭环**
   - 小红书店铺 vs 引流到其他平台的策略选择
   - 直播带货（与 TikTok 直播的区别：小红书更偏"慢直播"、生活方式）
   - 笔记挂链接的转化优化

6. **跨境品牌入驻小红书**
   - 品牌号认证流程
   - 跨境品牌内容本地化策略（不是翻译，是重新创作）
   - 中国消费者对跨境品牌的认知差异
   - 合规注意事项（广告法、化妆品备案等）

**预计篇幅**: 8000-10000 字
**预计时间**: 2-3 小时阅读

---

### E4. Pinterest AI 运营指南

**核心差异化内容**:

1. **Pinterest 作为视觉搜索引擎（独特定位）**
   - Pinterest 不是社交媒体，是视觉搜索+发现引擎
   - 用户意图：高购买意图（主动搜索产品/灵感）
   - 月搜索量 800 亿次，619M MAU
   - 最强品类：家居装饰、时尚、美妆、DIY、婚礼、食品

2. **Pinterest SEO 方法论**
   - Pin 标题/描述的关键词优化（与 Google SEO 类似但更视觉化）
   - Board 策略：AI 生成 Board 名称和描述
   - Rich Pins（Product/Article/Recipe）设置与优化
   - AI 关键词研究：Pinterest Trends + 搜索联想

3. **AI 视觉内容创作**
   - Pin 设计最佳实践（2:3 竖版、文字叠加、品牌一致性）
   - AI 批量生成 Pin 图片（Midjourney/Canva AI → Pinterest 风格适配）
   - Idea Pins（类似 Stories）的 AI 脚本生成
   - 季节性内容日历 AI 规划（Pinterest 用户提前 3-6 个月搜索）

4. **Pinterest Shopping Ads**
   - Shopping Campaigns 设置（Product Catalog 对接）
   - AI 优化产品 Feed（标题/描述适配 Pinterest 搜索习惯）
   - 与 Shopify 的原生集成
   - Pinterest Ads 与 Meta Ads 的预算分配决策

5. **Pinterest 数据分析**
   - Pinterest Analytics 关键指标（Impressions、Saves、Outbound Clicks）
   - AI 辅助趋势分析 Prompt 模板
   - 竞品 Pin 分析方法论

**预计篇幅**: 6000-8000 字
**预计时间**: 1.5-2 小时阅读

---

### E5. WhatsApp Business AI 客服与营销

**核心差异化内容**:

1. **WhatsApp 作为电商客服渠道（特定市场必备）**
   - 核心市场：拉美、东南亚、中东、南欧
   - 对话式商务 2025 消费 $2900 亿
   - AI Chatbot 转化率 12.3% vs 普通浏览 3.1%

2. **AI Chatbot 搭建方法论**
   - WhatsApp Business API vs Business App 的选择
   - AI 聊天机器人工作流设计（产品推荐→下单→支付→物流跟踪）
   - 多语言自动回复模板（西语/葡语/阿拉伯语/印尼语）
   - 与 Shopify/Amazon 订单系统的集成

3. **WhatsApp 营销自动化**
   - Broadcast 消息策略（新品通知、促销、复购提醒）
   - AI 个性化消息生成（基于购买历史和浏览行为）
   - WhatsApp Catalog 优化
   - 2026.1 新政策：禁止通用 AI Bot，合规注意事项

4. **售后自动化**
   - 退换货流程自动化
   - 物流状态主动推送
   - 差评预防：AI 检测不满情绪并升级人工

**预计篇幅**: 5000-6000 字
**预计时间**: 1-1.5 小时阅读

---

### E6. Reddit AI 营销指南

**核心差异化内容**:

1. **Reddit 作为产品发现引擎**
   - "Reddit before buying" 趋势：用户在购买前搜索 Reddit 评价
   - Google 搜索结果中 Reddit 权重大幅提升
   - AI 购物搜索功能测试中（2026）

2. **Reddit 社区营销方法论（反广告平台）**
   - Reddit 的反营销文化：如何不被 downvote
   - AI 辅助内容创作：真实、有价值、非推销的帖子风格
   - Subreddit 选择策略：AI 分析目标品类相关社区
   - AMA (Ask Me Anything) 策略

3. **Reddit Ads AI 优化**
   - Reddit Ads 定向策略（Interest/Community/Conversation targeting）
   - AI 生成 Reddit 风格广告文案（与 Meta/Google Ads 风格完全不同）
   - Conversation Ads 的 AI 优化

4. **品牌口碑监控**
   - AI 监控品牌/产品在 Reddit 的讨论
   - 竞品口碑分析
   - 负面讨论应对策略

**预计篇幅**: 4000-5000 字
**预计时间**: 1 小时阅读

---

### E7. 社交媒体跨渠道协同策略

**核心内容**:

1. **一个内容，多平台适配**
   - 核心内容 → Instagram Reels / YouTube Shorts / TikTok / Pinterest Pin 的 AI 自动适配
   - 各平台最佳尺寸/时长/风格对照表
   - AI 批量改写工作流

2. **社交媒体 → 电商平台的归因**
   - Instagram/YouTube/Pinterest → Amazon/Shopify 的流量追踪
   - UTM 参数策略
   - Amazon Attribution 与 Meta/Google 的联动

3. **社交媒体内容日历 AI 规划**
   - 跨平台发布节奏
   - 季节性内容规划（各平台提前期不同）
   - AI 自动生成月度内容计划

4. **预算分配框架**
   - 各渠道 CAC 对比
   - 品牌建设 vs 直接转化的预算比例
   - AI 辅助预算优化模型

---

## 二、Path D 扩展：电商平台差异化内容

### 新增文件

```
paths/d-platforms/
├── (existing) shopify-ai-guide.md          # D1
├── (existing) tiktok-shop-ai-guide.md      # D2
├── (existing) cross-platform-strategy.md   # D3
├── (new) d4-walmart-ai-guide.md            # D4 Walmart Marketplace
├── (new) d5-temu-seller-guide.md           # D5 Temu 卖家策略
├── (new) d6-southeast-asia-ai-guide.md     # D6 东南亚电商（Shopee + Lazada）
├── (new) d7-mercado-libre-ai-guide.md      # D7 Mercado Libre 拉美
├── (new) d8-rakuten-japan-ai-guide.md      # D8 Rakuten 日本
├── (new) d9-ebay-ai-guide.md              # D9 eBay
├── (new) d10-aliexpress-ai-guide.md        # D10 AliExpress
├── (new) d11-coupang-korea-ai-guide.md     # D11 Coupang 韩国
├── (new) d12-faire-wholesale-ai-guide.md   # D12 Faire 批发
├── (new) d13-europe-marketplaces-guide.md  # D13 欧洲平台（Otto + Zalando）
└── (updated) README.md                     # 更新导航
```

### D4. Walmart Marketplace AI 指南（重点）

**只写差异化部分，与 Amazon 相似的引用 Path A**:

1. **Walmart 算法差异**
   - Walmart SEO vs Amazon A9/COSMO 的区别
   - Listing Quality Score 机制
   - Buy Box 算法差异（价格权重更高）

2. **Walmart Connect 广告（与 Amazon PPC 的区别）**
   - Sponsored Products / Sponsored Brands / Display 的差异
   - 广告报告分析（与 Amazon 搜索词报告的对比）
   - AI 优化 Prompt 模板（适配 Walmart 数据格式）
   - 广告卖家收入 +46% YoY 的机会分析

3. **WFS vs FBA 物流决策**
   - Walmart Fulfillment Services 的差异化优势
   - Multichannel Solutions (MCS) 跨平台物流
   - AI 辅助库存分配决策

4. **Walmart 特有功能**
   - Walmart+ 会员体系
   - 全渠道（线上+线下 4700+ 门店）策略
   - Walmart Marketplace 审核与合规差异

5. **Amazon → Walmart 迁移方法论**
   - Listing 迁移与优化（不是复制粘贴）
   - 定价策略差异（Walmart 用户更价格敏感）
   - 品类机会分析（Walmart 强势品类 vs Amazon）

**预计篇幅**: 6000-8000 字

---

### D5. Temu 卖家策略指南

**定位：竞争分析 + 入驻策略（不是运营优化，因为卖家自主空间有限）**:

1. **Temu 商业模式深度解析**
   - 全托管 vs 半托管模式的区别
   - 平台控价机制：卖家定价权有限
   - GMV $90-95B (2025)，跨境份额 24% 与 Amazon 持平
   - 对现有 Amazon/AliExpress 卖家的影响

2. **Temu 入驻决策框架**
   - 适合 Temu 的品类分析（低价、标品、供应链优势）
   - 不适合 Temu 的品类（品牌溢价、差异化产品）
   - AI 辅助品类机会评估 Prompt

3. **Temu 运营的有限 AI 应用**
   - 选品数据分析（热销品类、价格带）
   - 供应链成本优化
   - 产品图片 AI 优化（Temu 对主图要求严格）

4. **Temu 对跨境电商格局的影响**
   - de minimis 规则变化的影响
   - 与 Amazon/AliExpress/Shein 的竞争格局
   - 卖家多平台策略中 Temu 的定位

**预计篇幅**: 4000-5000 字

---

### D6. 东南亚电商 AI 指南（Shopee + Lazada）

**合并两个平台，重点写差异**:

1. **东南亚电商市场概览**
   - Shopee: GMV $127B, 4亿买家, 45% 市场份额
   - Lazada: 1.5亿+ 买家, Alibaba/Cainiao 物流集成
   - TikTok Shop 在东南亚的竞争（正在缩小与 Shopee 的差距）

2. **Shopee vs Lazada 差异化运营**
   - 平台选择决策框架（流量 vs 物流 vs 费率）
   - Shopee: 低费率、大流量、Flash Sale 驱动
   - Lazada: 跨境物流强、品牌旗舰店、LazMall

3. **东南亚特有的 AI 应用场景**
   - 多语言 Listing（泰语/越南语/印尼语/马来语/菲律宾语）
   - AI 翻译 + 本地化（不是直译，是文化适配）
   - 直播带货（东南亚直播渗透率极高）
   - Shopee Ads / Lazada Sponsored Solutions 优化

4. **跨境入驻实操**
   - Shopee 跨境卖家入驻流程
   - Lazada Global Selling 入驻流程
   - 物流方案选择（Shopee Logistics / Cainiao / 第三方）

**预计篇幅**: 6000-8000 字

---

### D7. Mercado Libre 拉美电商 AI 指南

1. **拉美市场概览**
   - GMV $65B (2025), 1.2亿年度买家, 收入 +39% YoY
   - 核心市场：巴西（最大）、墨西哥、阿根廷、哥伦比亚
   - Mercado Pago 支付生态 + Mercado Envios 物流

2. **西语/葡语 Listing AI 优化**
   - 巴西葡语 vs 葡萄牙葡语的区别
   - 拉美西语 vs 西班牙西语的区别
   - AI 本地化 Prompt 模板

3. **Mercado Libre 特有运营差异**
   - 算法和排名机制
   - 广告系统（Product Ads）
   - 物流等级对排名的影响
   - 退货政策差异

**预计篇幅**: 4000-5000 字

---

### D8. Rakuten 日本 AI 指南

1. **日本电商市场概览**
   - 市场规模 $258B (2025), 预计 2029 达 $338B
   - Rakuten GMV ~$31B, 日本第二大电商平台
   - 2026 与 YouTube Shopping 合作

2. **Rakuten 特有运营差异（vs Amazon JP）**
   - 店铺页面自定义（Rakuten 允许高度自定义，Amazon 不允许）
   - 积分系统（Rakuten Points 生态）
   - 邮件营销（Rakuten 鼓励卖家直接联系买家，Amazon 禁止）
   - 活动机制（Super Sale、Marathon）

3. **日语 Listing AI 优化**
   - 日语 SEO 关键词研究
   - 日本消费者文案偏好（详细、礼貌、信任感）
   - AI 生成日语产品描述 Prompt 模板

**预计篇幅**: 4000-5000 字

---

### D9-D13 简要规划

| 编号 | 平台 | 核心差异化内容 | 篇幅 |
|------|------|---------------|------|
| D9 | eBay | 拍卖 vs 固定价、二手/翻新品 AI 描述、eBay Promoted Listings | 3000-4000 字 |
| D10 | AliExpress | 全托管模式、与 Temu 的竞争策略、南欧市场机会 | 3000-4000 字 |
| D11 | Coupang | 韩国市场入门、Rocket Delivery、韩语 Listing | 3000-4000 字 |
| D12 | Faire | B2B 批发定价策略、品牌故事 AI 生成、零售商关系管理 | 2000-3000 字 |
| D13 | 欧洲平台 | Otto/Zalando 入驻、德国电商合规、欧洲 VAT/EPR | 3000-4000 字 |

---

## 三、Path A 回填内容（通用但未覆盖的 AI 场景）

在分析各平台时发现以下通用 AI 场景在 Path A 中可能未充分覆盖：

| 补充到 | 内容 | 来源 |
|--------|------|------|
| A2 Listing | AI 视频脚本生成方法论（通用框架，不限平台） | YouTube/TikTok/Instagram 共性 |
| A3 广告 | AI 广告素材批量生成工作流（图片+视频+文案） | Meta/Google/Pinterest 共性 |
| A3 广告 | 跨渠道广告归因方法论（Amazon Attribution 等） | 多平台协同 |
| A4 客服 | AI Chatbot 搭建通用方法论 | WhatsApp/Shopify Chat 共性 |
| A4 客服 | 社交媒体评论/DM 自动回复策略 | Instagram/Facebook/TikTok 共性 |
| A6 合规 | 各社交平台广告合规要求对比 | Meta/Google/TikTok/Pinterest |

---

## 四、执行优先级

### Phase 1 ✅ 已完成（2026-03-14）
- [x] E1 Instagram + Facebook AI 运营指南
- [x] E2 YouTube AI 运营指南
- [x] D4 Walmart Marketplace AI 指南
- [x] Path E README.md（总览页）
- [x] 更新 Path D README.md（新增平台导航）
- [x] 更新主 README.md（新增 Path E 入口 + Path D 扩展）

### Phase 2 ✅ 已完成（2026-03-14）
- [x] E3 小红书 AI 运营指南
- [x] E4 Pinterest AI 运营指南
- [x] D5 Temu 卖家策略指南
- [x] D6 东南亚电商 AI 指南
- [x] E7 社交媒体跨渠道协同策略

### Phase 3 ✅ 已完成（2026-03-14）
- [x] E5 WhatsApp Business AI 指南
- [x] E6 Reddit AI 营销指南
- [x] D7 Mercado Libre 拉美指南
- [x] D8 Rakuten 日本指南
- [x] D9 eBay AI 指南
- [x] D10 AliExpress AI 指南
- [x] D11 Coupang 韩国指南
- [x] D12 Faire 批发指南
- [x] D13 欧洲平台指南（Otto + Zalando）
- [x] Path A 回填内容（2026-03-14 完成）

---

## 五、README.md 更新预览

主 README 的"选择你的路径"表格新增：

```markdown
| **[Path E: 社交媒体](paths/e-social-media/)** 🆕 | 品牌营销/内容运营 | 不需要 | 每天30分钟，2-3周 | 社交媒体 AI 引流体系 |
```

Path D 的模块导航表格扩展为完整的平台矩阵。
