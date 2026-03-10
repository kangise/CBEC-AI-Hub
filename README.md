# CBEC-AI-Hub: 跨境电商 AI 实战指南

用 AI 重新定义跨境电商运营 -- 从选品到客服，从广告到供应链，每个环节的 AI 工具、方法和实战案例。

这不是一份电商入门教程，而是一份以 AI 为核心的电商运营升级指南。它假设你已经有基本的电商认知，帮你系统性地把 AI 融入运营的每一个环节。

---

## 目录

- [谁适合读这份指南](#谁适合读这份指南)
- [AI 基础能力](#ai-基础能力)
- [选品与市场分析中的 AI](#选品与市场分析中的-ai)
- [Listing 与内容创作中的 AI](#listing-与内容创作中的-ai)
- [广告优化中的 AI](#广告优化中的-ai)
- [客服与售后中的 AI](#客服与售后中的-ai)
- [库存与供应链中的 AI](#库存与供应链中的-ai)
- [合规与风控中的 AI](#合规与风控中的-ai)
- [数据分析与 BI 中的 AI](#数据分析与-bi-中的-ai)
- [运营自动化与 AI Agent](#运营自动化与-ai-agent)
- [AI 编程能力建设](#ai-编程能力建设)
- [AI 工具速查表](#ai-工具速查表)
- [贡献](#贡献)

---

## 谁适合读这份指南

你已经在做跨境电商（或者对电商运营有基本了解），想知道：

- AI 到底能帮我做什么？哪些环节能用 AI 提效？
- 具体用什么工具？怎么用？有没有免费的？
- 别人是怎么用 AI 做选品、写 Listing、优化广告的？
- 我不会写代码，能用 AI 吗？会写代码的话能做什么更高级的事？

这份指南按电商运营的核心环节组织，每个环节都回答三个问题：

1. AI 能做什么（场景和价值）
2. 用什么工具（优先免费和开源）
3. 怎么学（免费课程、教程、实战案例）

[回到目录](#目录)

---

## AI 基础能力

在深入各个电商环节之前，你需要先建立 AI 的基础认知和使用能力。这是所有后续内容的前提。

### Prompt Engineering -- 和 AI 高效沟通

这是最重要的基础技能。你和 AI 沟通的质量直接决定了输出的质量。

核心要点：
- 给 AI 明确的角色、上下文和输出格式要求
- 用 few-shot（给几个例子）比 zero-shot（不给例子）效果好得多
- 复杂任务拆成多步，不要一次让 AI 做太多
- 学会迭代：第一次输出不满意，追问和修正比重新开始更高效

免费学习资源：

| 资源 | 说明 |
|------|------|
| [DeepLearning.AI: ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) | Andrew Ng 的免费短课，1.5小时，开发者视角的 Prompt 工程 |
| [Coursera: Generative AI for Everyone (Andrew Ng)](https://www.coursera.org/learn/generative-ai-for-everyone) | 免费旁听，AI 入门最佳起点 |
| [Google: Introduction to Generative AI](https://www.cloudskillsboost.google/course_templates/536) | Google Cloud 免费课程，AI 基础概念 |
| [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) | 官方指南，最权威的 Prompt 技巧 |
| [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) | Claude 官方指南 |
| [Learn Prompting](https://learnprompting.org/) | 免费开源教程，从入门到高级 |

### 主流 AI 工具认知

你需要了解这些工具各自擅长什么，才能在不同场景选对工具。

| 工具 | 免费额度 | 擅长场景 |
|------|----------|----------|
| [ChatGPT](https://chat.openai.com/) | 免费版可用 GPT-4o mini | 通用对话、文案、分析、代码 |
| [Claude](https://claude.ai/) | 免费版可用 | 长文本分析、结构化输出、代码 |
| [Google Gemini](https://gemini.google.com/) | 免费 | 多模态（图片+文本）、Google 生态集成 |
| [Perplexity](https://www.perplexity.ai/) | 免费 | 带引用的搜索式问答，适合市场调研 |
| [Midjourney](https://www.midjourney.com/) | 付费 | 高质量产品图和营销素材生成 |
| [Microsoft Copilot](https://copilot.microsoft.com/) | 免费 | 集成 Office，适合报表和文档 |
| [Kimi](https://kimi.moonshot.cn/) | 免费 | 中文长文本处理，适合中文电商场景 |

免费学习资源：

| 资源 | 说明 |
|------|------|
| [YouTube: AI Tools for Business (Matt Wolfe)](https://www.youtube.com/@maboroshi) | 每周更新 AI 工具评测 |
| [YouTube: All About AI](https://www.youtube.com/@AllAboutAI) | AI 工具教程和对比 |
| [There's An AI For That](https://theresanaiforthat.com/) | AI 工具搜索引擎，按场景分类 |

[回到目录](#目录)

---

## 选品与市场分析中的 AI

选品是电商运营的起点，也是 AI 能产生最大杠杆效应的环节之一。传统选品依赖经验和手动调研，AI 可以把这个过程从"拍脑袋"升级到"数据驱动"。

### AI 能做什么

| 场景 | 传统做法 | AI 做法 | 效率提升 |
|------|----------|---------|----------|
| 市场趋势发现 | 手动浏览 Best Seller、New Release | AI 分析搜索趋势数据，自动发现上升品类 | 10x |
| 竞品 Review 分析 | 逐条阅读竞品评论 | AI 批量分析数千条 Review，提取痛点和需求 | 50x |
| 关键词研究 | 工具导出 + 手动筛选 | AI 做语义聚类，发现关键词背后的真实需求 | 5x |
| 需求预测 | 看历史销量趋势 | 时间序列模型预测未来需求，量化不确定性 | 3x |
| 供应商沟通 | 手动写邮件、翻译 | AI 生成多语言询盘邮件、规格书翻译 | 5x |

### 工具和方法

不需要写代码的方案：

| 工具 | 免费 | 用法 |
|------|------|------|
| ChatGPT / Claude | 是（有限额） | 粘贴竞品 Review 让 AI 分析痛点；输入品类让 AI 做市场分析框架 |
| Perplexity | 是 | 搜索式调研：直接问"2025年 Amazon US 宠物用品趋势"，带引用来源 |
| Google Gemini | 是 | 上传竞品截图，让 AI 分析 Listing 策略和定价模式 |
| Google Trends | 是 | 验证品类搜索趋势，对比不同市场的需求差异 |

需要写代码的进阶方案：

| 工具 | 开源/免费 | 用途 |
|------|-----------|------|
| [BERTopic](https://github.com/MaartenGr/BERTopic) | 开源 | 对大量 Review 做主题建模，自动发现用户关注的核心话题 |
| [KeyBERT](https://github.com/MaartenGr/KeyBERT) | 开源 | 从产品描述和评论中提取关键特征词 |
| [Facebook Prophet](https://github.com/facebook/prophet) | 开源 | 时间序列预测，预测品类销量趋势和季节性 |
| [Darts](https://github.com/unit8co/darts) | 开源 | 更强大的时间序列预测库，支持深度学习模型 |
| [GluonTS](https://github.com/awslabs/gluonts) | 开源 | AWS 出品的概率时间序列预测，适合多 SKU 预测 |

### 实战案例

案例 1：用 ChatGPT 做竞品 Review 分析

```
Prompt 示例：

你是一个资深的 Amazon 产品经理。我会给你一组竞品的 1-3 星差评。
请分析这些差评，输出：
1. 排名前5的用户痛点（按提及频率排序）
2. 每个痛点的具体描述和代表性评论
3. 针对每个痛点的产品改进建议
4. 这些痛点中，哪些是你认为最容易通过产品设计解决的

[粘贴差评内容]
```

案例 2：用 BERTopic 做大规模 Review 主题分析

当你有几千条甚至上万条 Review 时，手动分析不现实。BERTopic 可以自动聚类发现主题。

```python
from bertopic import BERTopic
import pandas as pd

reviews = pd.read_csv("competitor_reviews.csv")
topic_model = BERTopic(language="english", min_topic_size=10)
topics, probs = topic_model.fit_transform(reviews["text"].tolist())
topic_model.get_topic_info()  # 查看发现的主题
```

### 免费学习资源

| 资源 | 说明 |
|------|------|
| [YouTube: How to Use ChatGPT for Amazon Product Research](https://www.youtube.com/results?search_query=chatgpt+amazon+product+research) | 实操教程合集 |
| [Kaggle: BERTopic Tutorial](https://www.kaggle.com/learn/intro-to-machine-learning) | 免费机器学习入门（为 BERTopic 打基础） |
| [BERTopic 官方文档](https://maartengr.github.io/BERTopic/) | 完整教程和示例 |
| [Facebook Prophet 官方教程](https://facebook.github.io/prophet/docs/quick_start.html) | 快速上手时间序列预测 |
| [Google Trends](https://trends.google.com/) | 免费趋势分析工具 |

[回到目录](#目录)

---

## Listing 与内容创作中的 AI

Listing 优化和内容创作是 AI 在电商领域最成熟、最容易上手的应用场景。AI 不是替代你写 Listing，而是把你从"从零开始写"升级到"编辑和优化 AI 初稿"。

### AI 能做什么

| 场景 | AI 做法 | 注意事项 |
|------|---------|----------|
| 标题生成 | 输入关键词和卖点，AI 生成多个标题变体 | 需要人工检查关键词覆盖和平台规范 |
| 五点描述 | AI 根据产品特性和竞品分析生成卖点 | 确保差异化，不要和竞品雷同 |
| A+ Content | AI 生成文案框架和模块内容 | 视觉设计仍需人工或设计工具 |
| 多语言翻译 | AI 做本地化翻译（不只是直译） | 需要母语者审核关键市场 |
| SEO 关键词 | AI 分析竞品 Listing 提取关键词策略 | 结合工具数据验证搜索量 |
| 产品图片 | AI 生成场景图、生活方式图 | 主图仍建议用实拍白底图 |
| 视频脚本 | AI 生成产品视频脚本和分镜 | 需要人工调整节奏和品牌调性 |

### 工具和方法

文案生成（不需要写代码）：

| 工具 | 免费 | 用法 |
|------|------|------|
| ChatGPT / Claude | 是（有限额） | 生成 Listing 初稿、优化现有文案、分析竞品 Listing |
| Google Gemini | 是 | 上传竞品截图让 AI 分析，多模态能力强 |
| [Helium 10 Listing Builder](https://www.helium10.com/tools/listing-builder/) | 部分免费 | 内置 AI 生成，结合关键词数据 |
| Microsoft Copilot | 是 | 在 Word 中直接用 AI 辅助写作 |

多语言翻译：

| 工具 | 免费 | 说明 |
|------|------|------|
| [DeepL](https://www.deepl.com/) | 是（有限额） | 翻译质量优于 Google Translate，尤其是欧洲语言 |
| ChatGPT / Claude | 是（有限额） | 可以做"本地化翻译"而不是直译，告诉它目标市场的搜索习惯 |
| [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) | 开源 | Meta 的 200 语言翻译模型，可本地部署 |
| [MarianMT](https://huggingface.co/Helsinki-NLP) | 开源 | HuggingFace 上的翻译模型，支持多语言对 |

图片生成：

| 工具 | 免费 | 说明 |
|------|------|------|
| [Midjourney](https://www.midjourney.com/) | 付费 | 最高质量的 AI 图片生成，适合产品场景图 |
| [DALL-E (ChatGPT)](https://chat.openai.com/) | Plus 用户 | 集成在 ChatGPT 中，方便迭代 |
| [Stable Diffusion](https://github.com/Stability-AI/generative-models) | 开源免费 | 可本地运行，无限生成，需要 GPU |
| [Leonardo.ai](https://leonardo.ai/) | 有免费额度 | 在线使用，适合产品图生成 |
| [Canva AI](https://www.canva.com/) | 部分免费 | 集成 AI 图片生成和编辑 |
| [ControlNet](https://github.com/lllyasviel/ControlNet) | 开源 | 精确控制图片生成（保持产品轮廓） |
| [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) | 开源 | 基于现有产品图生成不同风格变体 |

### 实战案例

案例 1：用 AI 生成多语言 Listing

```
Prompt 示例：

你是一个专业的 Amazon Listing 优化专家，精通德语市场。

我的产品是一个便携式蓝牙音箱，核心卖点：
- IPX7 防水
- 24小时续航
- 360度环绕音效
- 重量仅 300g

请为 Amazon.de 生成：
1. 德语标题（不超过200字符，包含核心关键词）
2. 5个 Bullet Points（德语，每条突出一个卖点，融入德国消费者关心的点）
3. 产品描述（德语，200字以内）

要求：
- 不是直译，要符合德国消费者的搜索习惯和表达方式
- 融入德语市场常用的搜索关键词
- 语气专业但不生硬
```

案例 2：用 AI 分析竞品 Listing 策略

```
Prompt 示例：

分析以下3个竞品的 Amazon Listing，对比它们的策略差异：

[竞品A标题和五点]
[竞品B标题和五点]
[竞品C标题和五点]

请输出：
1. 三个竞品各自的核心定位和差异化策略
2. 它们共同强调的卖点（说明这是品类的"必备项"）
3. 它们各自独有的卖点（说明这是差异化机会）
4. 关键词覆盖对比（哪些关键词被多个竞品使用）
5. 基于以上分析，我的 Listing 应该如何差异化定位
```

### 免费学习资源

| 资源 | 说明 |
|------|------|
| [YouTube: AI Amazon Listing Optimization](https://www.youtube.com/results?search_query=ai+amazon+listing+optimization+2025) | 实操教程合集 |
| [YouTube: Midjourney Product Photography](https://www.youtube.com/results?search_query=midjourney+product+photography+ecommerce) | AI 产品图生成教程 |
| [Stable Diffusion WebUI 教程](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) | 免费本地 AI 图片生成 |
| [DeepL 官方文档](https://www.deepl.com/docs-api) | API 文档，可批量翻译 |
| [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course) | 免费 NLP 课程，理解翻译模型原理 |

[回到目录](#目录)

---

## 广告优化中的 AI

Amazon PPC 广告是数据密集型工作，天然适合 AI 介入。从关键词扩展到竞价优化，从文案测试到效果预测，AI 可以在广告的每个环节提供杠杆。

### AI 能做什么

| 场景 | AI 做法 |
|------|---------|
| 关键词扩展 | AI 语义分析发现相关长尾词，不只是字面匹配 |
| 否定关键词 | AI 分析搜索词报告，自动识别浪费性关键词 |
| 广告文案 | AI 批量生成 Headline 变体，加速 A/B 测试 |
| 竞价优化 | 预测模型估算不同竞价下的转化概率 |
| 预算分配 | AI 分析各 Campaign 的边际 ROAS，优化预算分配 |
| 效果预测 | 基于历史数据预测广告效果，辅助决策 |

### 工具和方法

不需要写代码：

| 工具 | 免费 | 用法 |
|------|------|------|
| ChatGPT / Claude | 是（有限额） | 分析搜索词报告、生成否定关键词列表、写广告文案变体 |
| [Amazon Advertising Learning Console](https://learningconsole.amazonadvertising.com/) | 是 | 官方免费课程，理解广告体系 |
| Google Gemini | 是 | 上传广告报告截图，让 AI 分析优化方向 |

需要写代码的进阶方案：

| 工具 | 开源/免费 | 用途 |
|------|-----------|------|
| [SentenceTransformers](https://github.com/UKPLab/sentence-transformers) | 开源 | 关键词语义相似度分析，发现语义相关但字面不同的关键词 |
| [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding) | 开源 | 中英文双语嵌入模型，分析关键词竞争度 |
| [DeepCTR](https://github.com/shenweichen/DeepCTR) | 开源 | 点击率预测框架，预测广告转化概率 |
| [AutoGluon](https://github.com/autogluon/autogluon) | 开源 | 自动化机器学习，无需深度 ML 知识即可建模 |
| [LightGBM](https://github.com/microsoft/LightGBM) | 开源 | 高效梯度提升框架，适合广告效果预测 |

### 实战案例

案例：用 ChatGPT 分析搜索词报告

```
Prompt 示例：

你是一个 Amazon PPC 广告优化专家。

以下是我的搜索词报告数据（过去30天）：
[粘贴搜索词、展示量、点击量、花费、订单数、销售额]

请分析并输出：
1. 高转化关键词 TOP 10（按 ROAS 排序），建议提高竞价
2. 高花费低转化关键词 TOP 10，建议降低竞价或否定
3. 高展示低点击关键词（CTR < 0.3%），分析可能原因
4. 建议添加的否定关键词列表
5. 整体优化建议（预算重新分配方向）
```

### 免费学习资源

| 资源 | 说明 |
|------|------|
| [Amazon Advertising Learning Console](https://learningconsole.amazonadvertising.com/) | 官方免费广告课程（必学） |
| [YouTube: Amazon PPC with AI](https://www.youtube.com/results?search_query=amazon+ppc+ai+optimization+2025) | AI 广告优化教程 |
| [Google: Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) | 免费 ML 入门，理解预测模型原理 |
| [Kaggle: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) | 免费 ML 微课程 |
| [AutoGluon 官方教程](https://auto.gluon.ai/stable/tutorials/) | 自动化 ML 快速上手 |

[回到目录](#目录)

---

## 客服与售后中的 AI

客服是重复性最高的运营环节之一，也是 AI 最容易产生立竿见影效果的场景。多语言回复、差评分析、申诉信撰写，AI 都能大幅提效。

### AI 能做什么

| 场景 | AI 做法 |
|------|---------|
| 买家消息回复 | AI 生成多语言回复模板，保持专业且符合平台规范 |
| 差评分析 | 批量分析差评内容，提取产品问题的共性模式 |
| 差评回复 | AI 生成得体的公开回复，展示品牌态度 |
| 申诉信撰写 | AI 辅助写 Plan of Action，分析违规原因并生成改善方案 |
| FAQ 知识库 | 基于产品手册和历史问答构建智能知识库 |
| 多语言支持 | 实时翻译买家消息，生成对应语言的回复 |

### 工具和方法

不需要写代码：

| 工具 | 免费 | 用法 |
|------|------|------|
| ChatGPT / Claude | 是（有限额） | 生成客服回复模板、分析差评、写申诉信 |
| [DeepL](https://www.deepl.com/) | 是（有限额） | 翻译买家消息和回复 |
| Google Gemini | 是 | 上传差评截图批量分析 |

需要写代码的进阶方案：

| 工具 | 开源/免费 | 用途 |
|------|-----------|------|
| [LlamaIndex](https://github.com/run-llama/llama_index) | 开源 | 构建产品 FAQ 知识库，AI 自动检索回答 |
| [Haystack](https://github.com/deepset-ai/haystack) | 开源 | 构建问答系统，连接产品文档和政策 |
| [NLLB-200](https://github.com/facebookresearch/fairseq/tree/nllb) | 开源 | Meta 的 200 语言翻译模型，可本地部署 |
| [FastChat](https://github.com/lm-sys/FastChat) | 开源 | 部署本地客服聊天机器人 |
| [Ollama](https://github.com/ollama/ollama) | 开源 | 本地运行 LLM，保护客户数据隐私 |

### 实战案例

案例 1：用 AI 批量分析差评并生成改善方案

```
Prompt 示例：

你是一个电商产品质量分析师。以下是我的产品最近60天的所有1-3星评论。

请完成以下分析：
1. 将所有差评按问题类型分类（如：质量问题、功能不符、物流损坏、使用困难等）
2. 每个类型的出现频率和占比
3. 每个类型中最有代表性的3条评论原文
4. 针对每个问题类型，给出：
   - 短期应对措施（Listing 调整、客服话术）
   - 长期改善方案（产品改进、供应商沟通要点）
5. 优先级排序：哪个问题最值得先解决（考虑频率和严重程度）

[粘贴差评内容]
```

案例 2：用 AI 写账号申诉信

```
Prompt 示例：

你是一个 Amazon 账号申诉专家。我的账号因为以下原因被暂停：
[具体违规通知内容]

请帮我撰写一份 Plan of Action (POA)，包含：
1. Root Cause（根本原因分析）-- 承认问题，不推卸责任
2. Immediate Actions（已采取的紧急措施）
3. Preventive Measures（防止再次发生的长期措施）

要求：
- 语气诚恳、专业
- 每个部分用具体的、可执行的行动项，不要空话
- 符合 Amazon 申诉信的格式规范
```

### 免费学习资源

| 资源 | 说明 |
|------|------|
| [LlamaIndex 官方教程](https://docs.llamaindex.ai/en/stable/getting_started/) | 免费，构建 RAG 知识库 |
| [Haystack 官方教程](https://haystack.deepset.ai/tutorials) | 免费，构建问答系统 |
| [YouTube: AI Customer Service for E-Commerce](https://www.youtube.com/results?search_query=ai+customer+service+ecommerce+amazon) | 实操教程 |
| [YouTube: Amazon Appeal Letter with AI](https://www.youtube.com/results?search_query=amazon+appeal+letter+chatgpt) | AI 写申诉信教程 |
| [Ollama 官方文档](https://ollama.com/) | 免费本地 LLM 部署 |

[回到目录](#目录)

---

## 库存与供应链中的 AI

库存管理和供应链是电商运营中最"数学密集"的环节。传统做法靠经验和 Excel，AI 可以把预测精度和决策质量提升一个量级。

### AI 能做什么

| 场景 | AI 做法 |
|------|---------|
| 需求预测 | 时间序列模型预测各 SKU 未来销量，考虑季节性和促销影响 |
| 补货决策 | 基于预测结果自动计算最优补货量和补货时间 |
| 库存优化 | 多仓库库存分配优化，平衡成本和时效 |
| 供应商评估 | AI 分析历史交货数据，评估供应商可靠性 |
| 物流路径 | 运筹优化选择最优物流方案 |
| 单据处理 | OCR 自动识别发票、装箱单等文档 |

### 工具和方法

不需要写代码：

| 工具 | 免费 | 用法 |
|------|------|------|
| ChatGPT / Claude | 是（有限额） | 分析库存数据、生成补货建议、计算安全库存 |
| Google Sheets + AI | 是 | 用 Gemini 在 Sheets 中做简单预测 |

需要写代码的进阶方案：

| 工具 | 开源/免费 | 用途 |
|------|-----------|------|
| [Facebook Prophet](https://github.com/facebook/prophet) | 开源 | 最易上手的时间序列预测，支持季节性和节假日效应 |
| [Darts](https://github.com/unit8co/darts) | 开源 | 更强大的时间序列库，支持深度学习模型 |
| [GluonTS](https://github.com/awslabs/gluonts) | 开源 | AWS 出品，概率预测，适合多 SKU 场景 |
| [NeuralProphet](https://github.com/ourownstory/neural_prophet) | 开源 | Prophet 的神经网络升级版 |
| [OR-Tools](https://github.com/google/or-tools) | 开源 | Google 运筹优化工具，补货策略和仓库分配 |
| [NetworkX](https://github.com/networkx/networkx) | 开源 | 供应链网络建模和分析 |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | 开源 | 多语言 OCR，自动识别发票和单据 |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | 开源 | 经典 OCR 引擎 |

### 实战案例

案例：用 Prophet 预测 SKU 销量

```python
from prophet import Prophet
import pandas as pd

# 准备数据：日期列 ds，销量列 y
df = pd.read_csv("daily_sales.csv")
df.columns = ["ds", "y"]

# 添加促销日（如 Prime Day、BFCM）
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.add_country_holidays(country_name="US")
model.fit(df)

# 预测未来90天
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# 查看预测结果和不确定性区间
model.plot(forecast)
model.plot_components(forecast)
```

基于预测结果计算补货量：

```
Prompt 示例（给 ChatGPT）：

我的产品预测未来90天日均销量为 50 件（置信区间 40-65）。
当前库存 2000 件，FBA 在途 500 件。
从下单到入仓的 Lead Time 是 45 天。
安全库存天数我想设为 14 天。

请帮我计算：
1. 当前库存可以支撑多少天
2. 最晚什么时候需要下新的采购订单
3. 建议采购量（考虑最优和最坏情况）
4. 如果 Lead Time 延长到 60 天，以上数字如何变化
```

### 免费学习资源

| 资源 | 说明 |
|------|------|
| [Prophet 官方教程](https://facebook.github.io/prophet/docs/quick_start.html) | 免费，快速上手时间序列预测 |
| [Darts 官方教程](https://unit8co.github.io/darts/) | 免费，更多模型选择 |
| [Coursera: Supply Chain Analytics (Rutgers)](https://www.coursera.org/learn/supply-chain-analytics) | 免费旁听，供应链分析 |
| [Google OR-Tools 教程](https://developers.google.com/optimization) | 免费，运筹优化入门 |
| [YouTube: Demand Forecasting with Python](https://www.youtube.com/results?search_query=demand+forecasting+python+prophet+tutorial) | 实操教程 |

[回到目录](#目录)

---

## 合规与风控中的 AI

跨境电商的合规要求复杂且多变（不同国家、不同品类、不同平台），AI 可以帮你快速查询、分类和监控合规风险。

### AI 能做什么

| 场景 | AI 做法 |
|------|---------|
| 合规查询 | AI 快速查询不同国家/品类的认证要求和法规变化 |
| HS Code 分类 | 机器学习模型自动分类产品海关编码 |
| Listing 合规检查 | AI 扫描 Listing 文案，识别可能违反平台政策的内容 |
| 图片合规检查 | 视觉模型检测产品图片中的潜在违规内容 |
| 政策监控 | AI 追踪平台政策更新，评估对现有产品的影响 |
| 知识产权排查 | AI 辅助检索商标和专利数据库 |

### 工具和方法

不需要写代码：

| 工具 | 免费 | 用法 |
|------|------|------|
| ChatGPT / Claude | 是（有限额） | 查询合规要求、分析政策变化、生成合规清单 |
| Perplexity | 是 | 带引用的合规信息搜索，可验证来源 |
| Google Gemini | 是 | 上传产品图片检查是否有潜在合规问题 |

需要写代码的进阶方案：

| 工具 | 开源/免费 | 用途 |
|------|-----------|------|
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | 开源 | HS Code 自动分类模型 |
| [YOLOv8](https://github.com/ultralytics/ultralytics) | 开源 | 产品图片违规内容检测 |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | 开源 | 基于文本描述的图片内容检测 |
| [DeBERTa](https://github.com/microsoft/DeBERTa) | 开源 | 文本分类，识别 Listing 中的敏感内容 |
| [ElasticSearch](https://github.com/elastic/elasticsearch) | 开源 | 构建法规知识库，快速检索 |

### 实战案例

案例：用 AI 生成多市场合规对比表

```
Prompt 示例：

我要在 Amazon US、DE、JP 三个站点销售一款儿童电动牙刷。

请帮我生成一份合规要求对比表，包含：
1. 每个市场需要的产品认证（如 FCC、CE、PSE 等）
2. 包装和标签的特殊要求
3. 电池/充电相关的运输限制
4. 儿童产品的额外安全要求（如 CPSIA、EN 71 等）
5. 预估认证费用范围和周期
6. 需要注意的常见合规陷阱

注意：请标注信息的时效性，法规可能已更新，建议我向认证机构确认最新要求。
```

### 免费学习资源

| 资源 | 说明 |
|------|------|
| [YouTube: Amazon Product Compliance with AI](https://www.youtube.com/results?search_query=amazon+product+compliance+ai+tools) | AI 合规工具教程 |
| [scikit-learn 官方教程](https://scikit-learn.org/stable/tutorial/) | 免费，机器学习分类入门 |
| [YOLOv8 官方文档](https://docs.ultralytics.com/) | 免费，目标检测入门 |
| [EU Product Safety Regulations](https://ec.europa.eu/safety-gate/) | 官方免费，欧盟产品安全法规 |

[回到目录](#目录)

---

## 数据分析与 BI 中的 AI

数据分析是电商运营从"凭感觉"到"看数据"的关键跃迁。AI 让这个过程更快、更深、门槛更低 -- 你甚至可以用自然语言直接查询数据。

### AI 能做什么

| 场景 | AI 做法 |
|------|---------|
| 数据探索 | 上传 CSV/Excel，用自然语言提问，AI 自动分析和可视化 |
| 报表生成 | AI 自动从数据中提取关键洞察，生成分析报告 |
| 异常检测 | AI 自动发现数据中的异常波动（销量骤降、退货率飙升） |
| SQL 生成 | 用自然语言描述需求，AI 生成 SQL 查询 |
| 仪表板搭建 | AI 辅助设计仪表板布局和指标选择 |
| 财务建模 | AI 辅助构建 P&L 模型和场景分析 |

### 工具和方法

不需要写代码：

| 工具 | 免费 | 用法 |
|------|------|------|
| ChatGPT (Advanced Data Analysis) | Plus 用户 | 上传 CSV/Excel，自然语言分析，自动生成图表 |
| Claude | 是（有限额） | 粘贴数据表格，让 AI 分析趋势和异常 |
| Google Gemini + Sheets | 是 | 在 Google Sheets 中用 AI 辅助分析 |
| [Julius AI](https://julius.ai/) | 有免费额度 | 上传数据文件，自然语言分析和可视化 |
| Microsoft Copilot + Excel | 是 | 在 Excel 中用 AI 辅助分析 |

需要写代码的进阶方案：

| 工具 | 开源/免费 | 用途 |
|------|-----------|------|
| [pandas-ai](https://github.com/gventuri/pandas-ai) | 开源 | 用自然语言查询 pandas DataFrame |
| [Streamlit](https://streamlit.io/) | 开源 | 快速搭建数据仪表板和分析工具 |
| [Metabase](https://github.com/metabase/metabase) | 开源 | 自托管 BI 工具，支持 SQL 和可视化 |
| [Apache Superset](https://github.com/apache/superset) | 开源 | 企业级 BI 平台 |
| [DuckDB](https://github.com/duckdb/duckdb) | 开源 | 高性能分析数据库，直接查询 CSV/Parquet |
| [Polars](https://github.com/pola-rs/polars) | 开源 | 比 pandas 快 10-100x 的数据处理框架 |

### 实战案例

案例 1：用 ChatGPT 做销售数据分析

```
Prompt 示例：

我上传了过去12个月的 Amazon 销售数据（包含日期、ASIN、销量、销售额、广告花费、退货数）。

请帮我分析：
1. 整体销售趋势（月度同比、环比）
2. TOP 10 ASIN 的表现对比（销售额、利润率、退货率）
3. 广告效率分析（TACoS 趋势、各 ASIN 的 ROAS）
4. 异常检测：哪些 ASIN 在哪些月份出现了异常波动
5. 季节性分析：哪些产品有明显的季节性模式
6. 基于以上分析的3个关键行动建议

请用图表展示关键发现。
```

案例 2：用 Streamlit 搭建运营仪表板

```python
import streamlit as st
import pandas as pd

st.title("Amazon 运营仪表板")

uploaded_file = st.file_uploader("上传销售数据", type=["csv", "xlsx"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # 核心指标
    col1, col2, col3 = st.columns(3)
    col1.metric("总销售额", f"${df['revenue'].sum():,.0f}")
    col2.metric("总订单数", f"{df['orders'].sum():,}")
    col3.metric("平均客单价", f"${df['revenue'].sum()/df['orders'].sum():.2f}")
    
    # 趋势图
    st.line_chart(df.groupby("date")["revenue"].sum())
```

### 免费学习资源

| 资源 | 说明 |
|------|------|
| [Coursera: Google Data Analytics Certificate](https://www.coursera.org/professional-certificates/google-data-analytics) | 免费旁听，数据分析入门最佳路径 |
| [Kaggle Learn](https://www.kaggle.com/learn) | 免费微课程：pandas、数据可视化、SQL |
| [Mode SQL Tutorial](https://mode.com/sql-tutorial/) | 免费交互式 SQL 教程 |
| [Streamlit 官方教程](https://docs.streamlit.io/get-started) | 免费，30分钟搭建第一个数据应用 |
| [YouTube: Data Analysis with AI Tools](https://www.youtube.com/results?search_query=data+analysis+ai+tools+chatgpt+2025) | AI 数据分析教程 |
| [DuckDB 官方教程](https://duckdb.org/docs/) | 免费，高性能数据查询 |

[回到目录](#目录)

---

## 运营自动化与 AI Agent

当你把前面各环节的 AI 应用串联起来，就进入了"运营自动化"的阶段。AI Agent 可以自主完成多步骤任务，把你从重复性工作中彻底解放出来。

### AI 能做什么

| 场景 | AI 做法 |
|------|---------|
| 日报/周报自动生成 | AI 自动拉取数据、分析趋势、生成报告 |
| 库存预警 | 自动监控库存水平，触发补货提醒 |
| 价格监控 | 自动追踪竞品价格变化，触发告警 |
| 评论监控 | 自动分析新差评，生成回复建议 |
| 广告自动优化 | AI Agent 根据规则自动调整竞价和预算 |
| 多步骤工作流 | 多个 AI Agent 协作完成复杂任务 |

### 工具和方法

无代码/低代码自动化：

| 工具 | 免费 | 用法 |
|------|------|------|
| [Zapier](https://zapier.com/) | 有免费额度 | 连接不同应用，触发自动化工作流 |
| [Make (Integromat)](https://www.make.com/) | 有免费额度 | 可视化工作流编排，比 Zapier 更灵活 |
| [n8n](https://n8n.io/) | 开源免费 | 可自部署的自动化平台，无限工作流 |
| [Airtable](https://www.airtable.com/) | 有免费额度 | 数据库 + 自动化，适合运营数据管理 |

AI Agent 框架（需要写代码）：

| 工具 | 开源/免费 | 用途 |
|------|-----------|------|
| [LangChain](https://github.com/langchain-ai/langchain) | 开源 | 最流行的 LLM 应用开发框架 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 开源 | 基于图的 AI Agent 工作流编排 |
| [CrewAI](https://github.com/joaomdmoura/crewAI) | 开源 | 多 Agent 协作框架，定义角色和任务 |
| [AutoGen](https://github.com/microsoft/autogen) | 开源 | 微软出品的多 Agent 对话框架 |
| [Prefect](https://github.com/PrefectHQ/prefect) | 开源 | 数据工作流编排和调度 |
| [Airflow](https://github.com/apache/airflow) | 开源 | 企业级工作流管理平台 |

### 实战案例

案例 1：用 n8n 搭建自动化运营监控

场景：每天早上自动检查关键指标，异常时发送告警。

工作流设计：
1. 定时触发（每天早上 9:00）
2. 从 Google Sheets / 数据库拉取昨日销售数据
3. 调用 AI（ChatGPT API）分析数据，判断是否有异常
4. 如果有异常，发送 Slack / 邮件告警，附带 AI 的分析和建议
5. 无论是否异常，生成日报存入指定文件夹

案例 2：用 CrewAI 构建竞品监控 Agent

```python
from crewai import Agent, Task, Crew

# 定义 Agent
researcher = Agent(
    role="竞品研究员",
    goal="监控竞品的价格、评论和 Listing 变化",
    backstory="你是一个资深的电商竞品分析师"
)

analyst = Agent(
    role="数据分析师",
    goal="分析竞品变化对我们的影响并给出建议",
    backstory="你是一个数据驱动的电商运营专家"
)

# 定义任务
monitor_task = Task(
    description="检查以下竞品 ASIN 的最新价格和评论变化：[ASIN列表]",
    agent=researcher
)

analysis_task = Task(
    description="基于竞品变化数据，分析对我们产品的影响并给出行动建议",
    agent=analyst
)

# 组建团队执行
crew = Crew(agents=[researcher, analyst], tasks=[monitor_task, analysis_task])
result = crew.kickoff()
```

### 免费学习资源

| 资源 | 说明 |
|------|------|
| [n8n 官方教程](https://docs.n8n.io/) | 免费，自动化工作流入门 |
| [LangChain 官方教程](https://python.langchain.com/docs/get_started/introduction) | 免费，LLM 应用开发 |
| [DeepLearning.AI: LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) | 免费短课 |
| [DeepLearning.AI: AI Agents in LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) | 免费短课 |
| [DeepLearning.AI: Multi AI Agent Systems with crewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) | 免费短课 |
| [CrewAI 官方文档](https://docs.crewai.com/) | 免费，多 Agent 系统 |
| [YouTube: n8n Automation for E-Commerce](https://www.youtube.com/results?search_query=n8n+automation+ecommerce) | 实操教程 |

[回到目录](#目录)

---

## AI 编程能力建设

你不需要成为程序员，但掌握基础编程能力会让你能用 AI 做更多事。好消息是：AI 本身就是最好的编程老师和助手。

### 学习路径

```
Level 0: 零代码
  用 ChatGPT / Claude 做分析、写文案、翻译
  用 Zapier / Make / n8n 做自动化
  用 Google Sheets + AI 做数据处理

Level 1: Python 基础（2-4周）
  变量、循环、函数、文件读写
  pandas 数据处理
  用 AI 辅助写代码（不需要记语法）

Level 2: 数据处理（2-4周）
  pandas 进阶（分组、合并、透视）
  openpyxl 自动化 Excel
  批量处理文件
  定时任务

Level 3: AI 应用开发（4-8周）
  调用 AI API（OpenAI / Anthropic）
  构建 RAG 知识库（LlamaIndex）
  搭建数据应用（Streamlit）
  部署和分享工具

Level 4: AI Agent 和系统（持续）
  LangChain / LangGraph
  多 Agent 系统
  本地模型部署
  微调模型
```

### 核心原则：AI 辅助编程

你不需要记住语法。现代 AI 编程助手可以：
- 根据你的自然语言描述生成代码
- 解释你看不懂的代码
- 调试错误并给出修复方案
- 把你的想法转化为可运行的程序

| 工具 | 免费 | 说明 |
|------|------|------|
| [GitHub Copilot](https://github.com/features/copilot) | 学生免费 | 最流行的 AI 编程助手 |
| [Kiro](https://kiro.dev/) | 免费 | AI IDE，spec 驱动开发 |
| [Cursor](https://cursor.sh/) | 有免费额度 | AI-first 代码编辑器 |
| ChatGPT / Claude | 是（有限额） | 粘贴代码让 AI 解释、调试、改进 |
| [Google Colab](https://colab.research.google.com/) | 是 | 免费在线 Python 环境，无需安装 |
| [Replit](https://replit.com/) | 有免费额度 | 在线 IDE，内置 AI 助手 |

### 免费学习资源

Python 入门（选一个即可）：

| 资源 | 说明 |
|------|------|
| [Coursera: Python for Everybody (Univ. of Michigan)](https://www.coursera.org/specializations/python) | 免费旁听，最经典的 Python 入门 |
| [freeCodeCamp: Python](https://www.freecodecamp.org/learn/scientific-computing-with-python/) | 完全免费 |
| [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/) | 免费在线书，实用自动化导向 |
| [Kaggle: Python Course](https://www.kaggle.com/learn/python) | 免费微课程，交互式 |
| [YouTube: Python for Beginners (Corey Schafer)](https://www.youtube.com/@coreyms) | 免费，高质量教程 |
| [YouTube: Programming with Mosh - Python](https://www.youtube.com/@programmingwithmosh) | 免费，6小时完整教程 |

数据处理：

| 资源 | 说明 |
|------|------|
| [Kaggle: Pandas Course](https://www.kaggle.com/learn/pandas) | 免费，pandas 入门最快路径 |
| [Coursera: Data Analysis with Python (IBM)](https://www.coursera.org/learn/data-analysis-with-python) | 免费旁听 |

AI 应用开发：

| 资源 | 说明 |
|------|------|
| [DeepLearning.AI: Building Systems with ChatGPT API](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/) | 免费短课 |
| [DeepLearning.AI: LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) | 免费短课 |
| [DeepLearning.AI: Building RAG Agents with LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/) | 免费短课 |
| [LlamaIndex 官方教程](https://docs.llamaindex.ai/en/stable/getting_started/) | 免费 |
| [Streamlit 官方教程](https://docs.streamlit.io/get-started) | 免费 |
| [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course) | 免费，NLP 和模型使用 |

本地模型部署和微调：

| 资源 | 说明 |
|------|------|
| [Ollama 官方文档](https://ollama.com/) | 免费，一行命令运行本地 LLM |
| [HuggingFace PEFT 教程](https://huggingface.co/docs/peft) | 免费，LoRA 微调入门 |
| [DeepLearning.AI: Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/) | 免费短课 |

[回到目录](#目录)

---

## AI 工具速查表

按场景快速查找适合的 AI 工具。标注 [免费] 的工具完全免费或有实用的免费额度。

### 通用 AI 助手

| 工具 | 免费 | 最适合场景 |
|------|------|------------|
| [ChatGPT](https://chat.openai.com/) | [免费] | 文案、分析、代码、通用任务 |
| [Claude](https://claude.ai/) | [免费] | 长文本分析、结构化输出、代码 |
| [Google Gemini](https://gemini.google.com/) | [免费] | 多模态分析（图片+文本） |
| [Perplexity](https://www.perplexity.ai/) | [免费] | 带引用的搜索式调研 |
| [Kimi](https://kimi.moonshot.cn/) | [免费] | 中文长文本处理 |

### 内容创作

| 工具 | 免费 | 最适合场景 |
|------|------|------------|
| [Midjourney](https://www.midjourney.com/) | 付费 | 高质量产品图和营销素材 |
| [Leonardo.ai](https://leonardo.ai/) | [免费] | 在线 AI 图片生成 |
| [Stable Diffusion](https://github.com/Stability-AI/generative-models) | [免费] 开源 | 本地无限图片生成 |
| [Canva AI](https://www.canva.com/) | [免费] | 图片编辑和设计 |
| [DeepL](https://www.deepl.com/) | [免费] | 高质量翻译 |
| [ControlNet](https://github.com/lllyasviel/ControlNet) | [免费] 开源 | 精确控制图片生成 |

### 数据分析

| 工具 | 免费 | 最适合场景 |
|------|------|------------|
| [Julius AI](https://julius.ai/) | [免费] | 上传数据自然语言分析 |
| [pandas-ai](https://github.com/gventuri/pandas-ai) | [免费] 开源 | 自然语言查询数据 |
| [Streamlit](https://streamlit.io/) | [免费] 开源 | 快速搭建数据应用 |
| [Metabase](https://github.com/metabase/metabase) | [免费] 开源 | 自托管 BI 工具 |
| [DuckDB](https://github.com/duckdb/duckdb) | [免费] 开源 | 高性能数据查询 |

### 预测和建模

| 工具 | 免费 | 最适合场景 |
|------|------|------------|
| [Facebook Prophet](https://github.com/facebook/prophet) | [免费] 开源 | 时间序列预测（最易上手） |
| [Darts](https://github.com/unit8co/darts) | [免费] 开源 | 高级时间序列预测 |
| [AutoGluon](https://github.com/autogluon/autogluon) | [免费] 开源 | 自动化机器学习 |
| [BERTopic](https://github.com/MaartenGr/BERTopic) | [免费] 开源 | 文本主题建模 |
| [OR-Tools](https://github.com/google/or-tools) | [免费] 开源 | 运筹优化 |

### 自动化

| 工具 | 免费 | 最适合场景 |
|------|------|------------|
| [n8n](https://n8n.io/) | [免费] 开源 | 自部署工作流自动化 |
| [Zapier](https://zapier.com/) | [免费] 有限额 | 无代码应用连接 |
| [Make](https://www.make.com/) | [免费] 有限额 | 可视化工作流 |

### AI 开发框架

| 工具 | 免费 | 最适合场景 |
|------|------|------------|
| [LangChain](https://github.com/langchain-ai/langchain) | [免费] 开源 | LLM 应用开发 |
| [LlamaIndex](https://github.com/run-llama/llama_index) | [免费] 开源 | RAG 知识库构建 |
| [CrewAI](https://github.com/joaomdmoura/crewAI) | [免费] 开源 | 多 Agent 协作 |
| [Ollama](https://github.com/ollama/ollama) | [免费] 开源 | 本地运行 LLM |
| [vLLM](https://github.com/vllm-project/vllm) | [免费] 开源 | 高性能 LLM 推理 |

### 编程助手

| 工具 | 免费 | 最适合场景 |
|------|------|------------|
| [GitHub Copilot](https://github.com/features/copilot) | 学生免费 | IDE 内 AI 编程 |
| [Kiro](https://kiro.dev/) | [免费] | AI IDE |
| [Cursor](https://cursor.sh/) | [免费] 有限额 | AI-first 编辑器 |
| [Google Colab](https://colab.research.google.com/) | [免费] | 在线 Python 环境 |

[回到目录](#目录)

---

## 贡献

欢迎提交 PR 补充工具、案例、学习资源或修正内容。

添加资源时请注意：
- 优先推荐免费或开源的工具和课程
- 注明是否免费
- 简要说明为什么推荐
- 如果是付费资源，说明为什么值得付费

[GitHub 仓库](https://github.com/kangise/CBEC-AI-Hub) | [提交 Issue](https://github.com/kangise/CBEC-AI-Hub/issues)

## 许可证

[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) -- 自由使用，无需署名。
