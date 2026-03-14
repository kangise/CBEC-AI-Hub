# A9. AI SEO 与 GEO 优化 | AI SEO & Generative Engine Optimization

> **路径**: Path A: 运营人 · **模块**: A9
> **最后更新**: 2026-03-14
> **难度**: ⭐⭐⭐ 高级
> **预计时间**: 每天 30 分钟，2-3 周

🏠 [Hub 首页](../../README.md) · 📋 [Path A 总览](README.md)

---

## 📖 章节导航

1. [从 SEO 到 GEO](#1-从-seo-到-geo) · 2. [Amazon SEO](#2-amazon-seo) · 3. [Google SEO for Shopify](#3-google-seo-for-shopify) · 4. [GEO 优化实操](#4-geo-优化实操) · 5. [社交平台 SEO](#5-社交平台-seo) · 6. [工具对比](#6-ai-seo-工具对比) · 7. [Prompt 模板](#7-prompt-模板) · 8. [完成标志](#8-完成标志)

---

## 本模块你将学会

- 理解 SEO → GEO 的范式转变（从 Google 排名到 AI 推荐）
- 掌握 Amazon SEO 最新算法（COSMO + Rufus）
- 掌握 Shopify Google SEO 方法论
- 学会 GEO 优化——让 ChatGPT/Perplexity/Gemini 推荐你的产品
- 了解各社交平台站内 SEO

> 💡 2026 年，1/3 的消费者已使用 AI Agent 进行产品发现。GEO 是 2026 年最重要的新技能。

---

## 1. 从 SEO 到 GEO

### 1.1 搜索行为三次变革

| 变革 | 时间 | 核心逻辑 | 电商影响 |
|------|------|---------|---------|
| Google 搜索 | 2000s-现在 | 关键词+链接+内容 | Shopify Google SEO |
| 平台内搜索 | 2010s-现在 | 平台规则+销量+转化率 | Amazon A9/COSMO |
| AI 搜索/GEO | 2024-现在 🆕 | 结构化数据+品牌权威+评价 | 被 ChatGPT/Perplexity 推荐 |

### 1.2 GEO vs 传统 SEO

| 维度 | 传统 SEO | GEO |
|------|---------|-----|
| 目标 | Google 排名 | AI 推荐/引用 |
| 用户行为 | 浏览搜索结果页 | 直接获得 AI 答案 |
| 排名因素 | 关键词+链接+内容 | 结构化数据+品牌权威+评价+被引用频率 |
| 内容格式 | 长文章、博客 | FAQ+Schema+结构化数据 |
| 衡量指标 | 排名/流量/CTR | AI 推荐频率/品牌提及率 |

### 1.3 为什么跨境卖家必须关注 GEO

- Shopify 推出 Agentic Storefronts（UCP 协议），AI Agent 可直接在 ChatGPT 内购买
- Perplexity Comet 浏览器可代替用户在 Amazon 购物
- Google AI Overviews 在搜索结果顶部显示 AI 答案
- 不被 AI 推荐 = 失去越来越多的流量

> 📎 **相关阅读**: [D1 Shopify](../d-platforms/shopify-ai-guide.md) — GEO 优化和 Agentic Storefronts 详见 D1

---

## 2. Amazon SEO

> 📎 **相关阅读**: [A2 Listing 优化](a2-listing-optimization.md) — A9→COSMO→Rufus 完整演进详见 A2

### 2.1 2026 Amazon SEO 核心清单

```
标题优化：
├── 核心关键词在前 80 字符
├── 自然语言（不堆砌）
├── COSMO 友好：回答"谁需要"和"为什么需要"
└── 包含品牌名

Bullet Points：
├── 利益点开头（不是功能）
├── 长尾关键词自然融入
├── Rufus 友好：回答用户可能问的问题
└── 前 3 条最重要

Backend Search Terms：
├── 不重复标题/Bullet 中已有的词
├── 包含拼写变体、同义词
├── 250 字节限制
└── 空格分隔，不用逗号

Q&A 预埋（Rufus 核心）：
├── Rufus 读取 Q&A 回答用户问题
├── 预埋 20+ 高频问题
├── 答案中自然包含关键词
└── 定期更新

A+ Content：
├── COSMO 读取理解产品
├── 包含使用场景描述
├── 图片 Alt Text 含关键词
└── 对比信息
```

### 2.2 Amazon SEO 审计 Prompt

```
你是 Amazon SEO 专家，精通 COSMO 和 Rufus 算法。

我的 Listing：
- 标题: [粘贴]
- Bullet Points: [粘贴]
- Backend Search Terms: [粘贴]
- 竞品 ASIN: [3 个]

请做 SEO 审计：
1. COSMO 友好度评分（1-10）
2. Rufus 友好度评分（1-10）
3. Backend 优化建议
4. Q&A 预埋建议（10 个问题）
5. 关键词覆盖差距
6. 优先级行动清单
```

---

## 3. Google SEO for Shopify

### 3.1 技术 SEO 检查清单

| 项目 | 要求 | 工具 |
|------|------|------|
| SSL | HTTPS（Shopify 自动） | — |
| Sitemap | 提交到 GSC | Google Search Console |
| Core Web Vitals | LCP<2.5s, FID<100ms, CLS<0.1 | PageSpeed Insights |
| Schema | Product/FAQ/Breadcrumb/Review | JSON-LD |
| 图片 | WebP，Alt Text 含关键词 | Shopify 图片优化 App |
| URL | 简洁，含关键词 | Shopify 后台 |

### 3.2 内容 SEO 策略

| 内容类型 | 示例 | 购买意图 | 频率 |
|----------|------|---------|------|
| 产品指南 | "How to Choose Best [品类]" | 高 | 每月 2 篇 |
| 对比文章 | "[A] vs [B]: Which Better?" | 高 | 每月 2 篇 |
| 教程 | "How to Use [产品]" | 中 | 每月 2 篇 |
| 清单 | "Top 10 [品类] 2026" | 高 | 每季度 |

### 3.3 Schema 结构化数据（GEO 基础）

```json
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "产品名称",
  "brand": {"@type": "Brand", "name": "品牌名"},
  "description": "产品描述",
  "offers": {
    "@type": "Offer",
    "price": "29.99",
    "priceCurrency": "USD",
    "availability": "https://schema.org/InStock"
  },
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "4.7",
    "reviewCount": "1250"
  }
}
```

---

## 4. GEO 优化实操

### 4.1 让 AI 推荐你的产品的 5 个策略

| 策略 | 说明 | 难度 | 影响 |
|------|------|------|------|
| 结构化数据 | Product/FAQ Schema | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| FAQ 优化 | 自然语言问答+Schema | ⭐⭐ | ⭐⭐⭐⭐ |
| 品牌提及 | 第三方网站被提及 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 评价覆盖 | Amazon/Trustpilot 高评分 | ⭐⭐ | ⭐⭐⭐⭐ |
| Agentic Storefronts | Shopify UCP 协议 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 4.2 GEO 效果检测

```
每月执行：
1. ChatGPT 搜索 "best [品类] 2026" → 记录是否被提及
2. Perplexity 搜索 "recommend [品类] for [场景]" → 记录
3. Gemini 搜索 "[品类] buying guide" → 记录
4. 与竞品对比：谁被推荐更多？
5. 追踪趋势：推荐频率是否在增长？
```

---

## 5. 社交平台 SEO

| 平台 | 搜索机制 | 关键词位置 | 详细指南 |
|------|---------|-----------|---------|
| TikTok | 站内搜索+推荐 | 标题+描述+字幕+Hashtag | [D2](../d-platforms/tiktok-shop-ai-guide.md) |
| YouTube | 搜索+推荐 | 标题+描述+标签+字幕 | [E2](../e-social-media/e2-youtube-ai-guide.md) |
| Pinterest | 视觉搜索 | Pin 标题+描述+Board | [E4](../e-social-media/e4-pinterest-ai-guide.md) |
| 小红书 | 站内搜索（70%渗透率） | 标题+正文前200字+标签 | [E3](../e-social-media/e3-xiaohongshu-ai-guide.md) |

---

## 6. AI SEO 工具对比

| 工具 | 功能 | 价格 | 适合 |
|------|------|------|------|
| Ahrefs | 关键词+竞品+链接 | $99/月起 | 全面 SEO |
| Semrush | 关键词+广告+内容 | $130/月起 | 企业级 |
| Surfer SEO | AI 内容优化 | $89/月起 | 内容 SEO |
| Helium 10 | Amazon 关键词+Listing | $79/月起 | Amazon SEO |
| vidIQ | YouTube SEO | 免费/$4.5/月 | YouTube |
| ChatGPT/Claude | 通用 AI 辅助 | $20/月 | 所有场景 |

---

## 7. Prompt 模板

### 7.1 GEO 审计

```
你是 GEO 专家。品牌 [X]，产品 [X]，网站 [URL]。
评估：结构化数据完整度、FAQ 优化建议（10个）、品牌提及分析、评价覆盖、竞品差距、优先行动清单。
```

### 7.2 多平台关键词研究

```
产品 [X]，品类 [X]，市场 [US]。
为 Amazon/Google/TikTok/YouTube/Pinterest 各提供 10 个关键词，标注搜索量级、竞争度、推荐内容类型。
```

---

## 8. 完成标志

- [ ] 完成 Amazon Listing SEO 审计
- [ ] 为 Shopify 添加 Schema 结构化数据
- [ ] 添加 FAQ Schema（10+ 问题）
- [ ] 在 ChatGPT/Perplexity 测试产品推荐
- [ ] 建立跨平台 SEO 关键词库

---
> 🏠 [Hub 首页](../../README.md) · 📋 [Path A 总览](README.md)
> 
> **Path A**: [A1 选品](a1-product-research.md) · [A2 Listing](a2-listing-optimization.md) · [A3 广告](a3-advertising.md) · [A4 客服](a4-customer-service.md) · [A5 库存](a5-inventory.md) · [A6 合规](a6-compliance.md) · [A7 视觉](a7-visual-content.md) · [A8 定价](a8-pricing-strategy.md) · [A9 SEO/GEO](a9-seo-geo.md)
> 
> **快速跳转**: [Path 0 基础](../0-foundations/) · [Path B 技术](../b-developers/) · [Path C 管理](../c-managers/) · [Path D 多平台](../d-platforms/) · [Path E 社交媒体](../e-social-media/)
