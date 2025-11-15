# CBEC-AI-Hub: 跨境电商AI知识中心

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![GitHub Stars](https://img.shields.io/github/stars/kangise/CBEC-AI-Hub?style=social)](https://github.com/kangise/CBEC-AI-Hub/stargazers)
[![Contributors](https://img.shields.io/github/contributors/kangise/CBEC-AI-Hub)](https://github.com/kangise/CBEC-AI-Hub/graphs/contributors)

> 跨境电商AI解决方案的权威开源知识库，专为开发者、数据科学家和技术领袖打造

一个全面的、社区驱动的跨境电商人工智能解决方案知识中心，汇集了100+精选工具、库和资源。

**[查看案例库](docs/case-studies.md)** | **[技术指南](docs/technical-guidelines.md)** | **[贡献指南](CONTRIBUTING.md)**

## 目录

- [项目介绍](#项目介绍)
- [核心跨境电商 AI 解决方案](#核心跨境电商-ai-解决方案)
  - [选品 (Product Research)](#1-选品--product-research--intelligence)
  - [Listing & 多语言内容 (Content & Localization)](#2-listing-生成内容创作--多语言本地化)
  - [市场与竞争分析](#3-市场--竞争分析market-intelligence)
  - [广告（Amazon Ads / Meta Ads / Google Ads）](#4-广告优化ads-optimization)
  - [运营自动化](#5-店铺运营自动化operations-automation)
  - [客服自动化](#6-客服自动化customer-service-ai)
  - [财务 & 利润分析](#7-财务--利润分析finance--profit)
  - [合规与风险管理](#8-合规风险管理compliance--risk)
  - [供应链与库存预测](#9-供应链库存预测物流规划logistics--scm)
- [AI Agents / Workflow 自动化引擎](#ai-agents--workflow-自动化引擎)
- [开发者工具 & Infra（LLM / RAG / Fine-tuning）](#开发者工具--ai-infra)
- [数据工程与可视化](#数据工程--可视化)
- [AI Research 工具](#ai-research-工具)
- [技术实施指南](docs/technical-guidelines.md)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 项目介绍

### 全球电商的AI势在必行

跨境电子商务面临着物流、法规、文化差异和金融系统等多重复杂挑战。人工智能不仅仅是优化工具，更是现代全球贸易的根本推动力，特别是对于依赖AI作为生存和发展基础的新兴"微型跨国企业"。

#### 核心挑战

**规模化运营难题**
- 多平台商品信息同步与管理
- 库存预测与补货决策复杂性
- 客服响应的多语言和时区挑战

**合规与风控压力**
- HS编码分类的准确性要求
- 各国税务和法规的动态变化
- 跨境支付的欺诈识别与防范

**用户体验本地化**
- 多语言内容的质量与一致性
- 文化差异导致的转化率差异
- 个性化推荐在不同市场的适应性

**成本与效率平衡**
- 营销投放的ROI优化
- 物流路径与成本的动态平衡
- 人工智能替代人工的投入产出比

#### AI解决方案的范式转变

**运营效率**
从人工处理转向智能自动化

**决策支持**
从经验判断转向数据驱动洞察

**战略能力**
AI从支持工具转变为核心竞争优势

## 核心跨境电商 AI 解决方案

### 1 选品 / Product Research & Intelligence

与跨境电商高度相关

市场需求预测、类目竞争分析、价格波动分析。

<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="https://github.com/facebook/prophet">Facebook Prophet</a></strong></td>
<td>时间序列预测库，支持季节性和趋势分析</td>
<td>预测产品销量趋势，识别季节性需求波动，为选品决策提供数据支持</td>
</tr>
<tr>
<td><strong><a href="https://github.com/facebookresearch/Kats">Kats</a></strong></td>
<td>时间序列分析工具包，包含预测、异常检测等功能</td>
<td>分析市场趋势变化，检测异常销量波动，优化库存管理策略</td>
</tr>
<tr>
<td><strong><a href="https://github.com/unit8co/darts">Darts</a></strong></td>
<td>现代时间序列预测库，支持深度学习模型</td>
<td>构建高精度销量预测模型，支持多变量预测和不确定性量化</td>
</tr>
<tr>
<td><strong><a href="https://github.com/awslabs/gluonts">GluonTS</a></strong></td>
<td>基于深度学习的概率时间序列建模工具包</td>
<td>构建复杂的需求预测模型，处理多产品、多市场的销量预测任务</td>
</tr>
<tr>
<td><strong><a href="https://github.com/ourownstory/neural_prophet">NeuralProphet</a></strong></td>
<td>基于神经网络的时间序列预测框架</td>
<td>结合传统时间序列方法和深度学习，提供更准确的产品需求预测</td>
</tr>
<tr>
<td><strong><a href="https://github.com/MaartenGr/BERTopic">BERTopic</a></strong></td>
<td>基于BERT的主题建模工具，支持动态主题发现</td>
<td>分析产品评论和市场讨论，发现新兴产品趋势和消费者需求变化</td>
</tr>
<tr>
<td><strong><a href="https://github.com/MaartenGr/KeyBERT">KeyBERT</a></strong></td>
<td>基于BERT的关键词提取工具</td>
<td>从产品描述和评论中提取关键特征，识别高价值产品属性</td>
</tr>
</table>

**应用案例**
- 新品机会挖掘
- 类目竞争度评分
- 历史销量预测

### 2 Listing 生成、内容创作 & 多语言本地化

核心业务：标题、五点描述、A+ 模块、SEO 关键词、图片生成。

<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="https://github.com/meta-llama/llama3">LLaMA-3</a></strong></td>
<td>Meta开源的大语言模型，支持多语言文本生成</td>
<td>生成高质量的产品标题、五点描述和A+内容，支持多语言本地化</td>
</tr>
<tr>
<td><strong><a href="https://github.com/mistralai/mistral-src">Mistral</a></strong></td>
<td>高效的开源语言模型，专注于推理和生成任务</td>
<td>快速生成产品文案和营销内容，优化转化率和用户体验</td>
</tr>
<tr>
<td><strong><a href="https://github.com/google/gemma_pytorch">Gemma</a></strong></td>
<td>Google开源的轻量级语言模型</td>
<td>在资源受限环境下生成产品描述和SEO优化内容</td>
</tr>
<tr>
<td><strong><a href="https://github.com/facebookresearch/fairseq/tree/nllb">NLLB-200</a></strong></td>
<td>Meta的200种语言翻译模型</td>
<td>将产品信息翻译成全球200+语言，实现真正的全球化销售</td>
</tr>
<tr>
<td><strong><a href="https://github.com/huggingface/transformers">MarianMT</a></strong></td>
<td>基于Transformer的神经机器翻译模型</td>
<td>高质量的产品描述翻译，保持品牌调性和技术准确性</td>
</tr>
<tr>
<td><strong><a href="https://github.com/nomic-ai/gpt4all">GPT4All</a></strong></td>
<td>可本地部署的开源GPT模型</td>
<td>离线生成SEO关键词和产品标签，保护商业机密信息</td>
</tr>
<tr>
<td><strong><a href="https://github.com/Stability-AI/generative-models">Stable Diffusion XL</a></strong></td>
<td>高分辨率图像生成模型</td>
<td>生成产品主图、场景图和营销素材，降低摄影成本</td>
</tr>
<tr>
<td><strong><a href="https://github.com/lllyasviel/ControlNet">ControlNet</a></strong></td>
<td>可控的图像生成工具，支持精确控制</td>
<td>根据产品轮廓和要求生成标准化的电商产品图片</td>
</tr>
<tr>
<td><strong><a href="https://github.com/tencent-ailab/IP-Adapter">IP-Adapter</a></strong></td>
<td>图像提示适配器，支持图像到图像的生成</td>
<td>基于现有产品图片生成不同风格和场景的营销图片</td>
</tr>
<tr>
<td><strong><a href="https://github.com/Gourieff/sd-webui-reactor">ReActor</a></strong></td>
<td>AI换脸工具，支持人脸替换和编辑</td>
<td>生成多样化的模特展示图，适应不同地区的审美偏好</td>
</tr>
</table>

### 3 市场 & 竞争分析（Market Intelligence）

<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="https://github.com/meta-llama/llama3">LLaMA-3</a></strong></td>
<td>Meta开源的大语言模型，支持多语言文本理解和情感分析</td>
<td>分析产品评论情感倾向，识别用户痛点和产品改进机会，优化产品策略</td>
</tr>
<tr>
<td><strong><a href="https://github.com/mistralai/mistral-src">Mixtral</a></strong></td>
<td>高性能混合专家模型，擅长多语言文本分析和推理</td>
<td>深度分析竞争对手产品评论，提取市场洞察和用户需求变化趋势</td>
</tr>
<tr>
<td><strong><a href="https://github.com/MaartenGr/BERTopic">BERTopic</a></strong></td>
<td>基于BERT的动态主题建模工具，支持主题演化追踪</td>
<td>聚类分析大量产品评论，发现用户关注的核心话题和新兴需求点</td>
</tr>
<tr>
<td><strong><a href="https://github.com/RaRe-Technologies/gensim">Gensim LDA</a></strong></td>
<td>经典的潜在狄利克雷分配主题建模库</td>
<td>从用户反馈中提取产品特征主题，指导产品功能优化和营销重点</td>
</tr>
<tr>
<td><strong><a href="https://github.com/facebook/prophet">Prophet</a></strong></td>
<td>Facebook开源的时间序列预测工具，处理季节性和趋势</td>
<td>预测竞争对手价格变化趋势，制定动态定价策略和促销时机</td>
</tr>
<tr>
<td><strong><a href="https://github.com/unit8co/darts">Darts</a></strong></td>
<td>现代时间序列分析库，支持多变量预测和深度学习</td>
<td>监控市场价格波动，预测最佳进入时机和库存调整策略</td>
</tr>
<tr>
<td><strong><a href="https://github.com/FlagOpen/FlagEmbedding">BGE embedding models</a></strong></td>
<td>中英文双语嵌入模型，支持语义相似度计算</td>
<td>分析搜索关键词竞争度，发现长尾关键词机会和SEO优化方向</td>
</tr>
</table>

### 4 广告优化（Ads Optimization）

包括：Amazon Ads / Google Ads / Facebook Ads。

<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="https://github.com/ray-project/ray">RLlib</a></strong></td>
<td>分布式强化学习框架，支持多智能体和大规模训练</td>
<td>构建智能出价系统，根据实时竞价环境自动调整广告出价策略，最大化ROI</td>
</tr>
<tr>
<td><strong><a href="https://github.com/stanford-futuredata/ColBERT">ColBERT</a></strong></td>
<td>高效的BERT检索模型，支持快速语义搜索</td>
<td>扩展相关关键词和ASIN，发现高转化潜力的长尾关键词组合</td>
</tr>
<tr>
<td><strong><a href="https://github.com/FlagOpen/FlagEmbedding">BGE-Large-EN</a></strong></td>
<td>大规模英文文本嵌入模型，支持语义理解</td>
<td>分析产品描述语义相似性，找到竞争产品的广告关键词机会</td>
</tr>
<tr>
<td><strong><a href="https://github.com/UKPLab/sentence-transformers">SentenceTransformers</a></strong></td>
<td>句子级别的语义嵌入工具，支持多语言</td>
<td>优化广告文案语义匹配度，提高广告相关性得分和点击率</td>
</tr>
<tr>
<td><strong><a href="https://github.com/shenweichen/DeepCTR">DeepCTR</a></strong></td>
<td>深度学习点击率预测框架，集成多种CTR模型</td>
<td>预测广告点击率和转化率，优化广告投放策略和预算分配</td>
</tr>
<tr>
<td><strong><a href="https://github.com/ChenglongChen/tensorflow-DeepFM">DeepFM</a></strong></td>
<td>结合因子分解机和深度神经网络的推荐模型</td>
<td>分析用户行为特征，预测广告转化概率，提升广告精准度</td>
</tr>
<tr>
<td><strong><a href="https://github.com/autogluon/autogluon">AutoGluon</a></strong></td>
<td>自动化机器学习框架，支持表格数据预测</td>
<td>自动构建广告效果预测模型，无需深度机器学习知识即可优化广告</td>
</tr>
</table>

### 5 店铺运营自动化（Operations Automation）

包括：KPI 监控、异常警告、自动报表、任务调度。

<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="https://github.com/langchain-ai/langgraph">LangGraph</a></strong></td>
<td>基于图的工作流编排框架，支持复杂业务逻辑自动化</td>
<td>构建智能店铺运营工作流，自动处理订单异常、库存预警、价格调整等日常任务</td>
</tr>
<tr>
<td><strong><a href="https://github.com/PrefectHQ/prefect">Prefect</a></strong></td>
<td>现代数据工作流编排平台，支持任务调度和监控</td>
<td>自动化数据同步、报表生成、KPI监控，确保多平台运营数据一致性</td>
</tr>
<tr>
<td><strong><a href="https://github.com/apache/airflow">Airflow</a></strong></td>
<td>开源工作流管理平台，支持复杂任务依赖和调度</td>
<td>编排跨境电商复杂业务流程，如库存同步、价格更新、广告优化等定时任务</td>
</tr>
<tr>
<td><strong><a href="https://github.com/gventuri/pandas-ai">pandas AI</a></strong></td>
<td>AI增强的数据分析库，支持自然语言查询数据</td>
<td>通过自然语言生成销售报表和业务洞察，简化数据分析工作流程</td>
</tr>
</table>

### 6 客服自动化（Customer Service AI）

如 Buyer Messages、差评回复、售后自动化。

<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="https://github.com/lm-sys/FastChat">FastChat</a></strong></td>
<td>开源聊天机器人训练和部署平台，支持多种LLM</td>
<td>构建智能客服系统，自动回复买家消息，处理常见售前售后问题</td>
</tr>
<tr>
<td><strong><a href="https://github.com/open-webui/open-webui">OpenWebUI</a></strong></td>
<td>开源的Web界面聊天平台，支持本地LLM部署</td>
<td>为客服团队提供AI辅助界面，快速生成专业回复，提高客服效率</td>
</tr>
<tr>
<td><strong><a href="https://github.com/facebookresearch/fairseq/tree/nllb">NLLB-200</a></strong></td>
<td>Meta的200种语言翻译模型，支持高质量多语言翻译</td>
<td>实现真正的全球客服支持，自动翻译买家消息和客服回复，打破语言障碍</td>
</tr>
<tr>
<td><strong><a href="https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100">M2M-100</a></strong></td>
<td>多对多语言翻译模型，支持100种语言互译</td>
<td>为多语言客服场景提供实时翻译，支持客服与全球买家无障碍沟通</td>
</tr>
<tr>
<td><strong><a href="https://github.com/deepset-ai/haystack">Haystack</a></strong></td>
<td>开源NLP框架，专注于构建搜索和问答系统</td>
<td>构建智能售后知识库，自动回答产品使用、退换货等常见问题</td>
</tr>
<tr>
<td><strong><a href="https://github.com/run-llama/llama_index">LlamaIndex</a></strong></td>
<td>数据框架，用于连接LLM与外部数据源</td>
<td>整合产品手册、FAQ、政策文档，为客服提供准确的信息检索和回答</td>
</tr>
</table>

### 7 财务 / 利润分析（Finance & Profit）

<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="https://github.com/facebook/prophet">Prophet</a></strong></td>
<td>Facebook开源的时间序列预测工具，支持季节性和趋势分析</td>
<td>预测月度和季度利润趋势，制定财务预算和投资决策，优化现金流管理</td>
</tr>
<tr>
<td><strong><a href="https://github.com/unit8co/darts">Darts</a></strong></td>
<td>现代时间序列预测库，支持多变量预测和深度学习</td>
<td>综合分析销售、成本、汇率等多因素，构建精准的利润预测模型</td>
</tr>
<tr>
<td><strong><a href="https://github.com/explosion/spaCy">spaCy</a></strong></td>
<td>工业级自然语言处理库，支持多语言文本分析</td>
<td>自动识别和分类发票、收据中的费用类型，简化财务记账流程</td>
</tr>
<tr>
<td><strong><a href="https://github.com/tesseract-ocr/tesseract">Tesseract</a></strong></td>
<td>开源光学字符识别引擎，支持100+种语言</td>
<td>自动识别各国发票和单据文字，提取金额、日期、供应商等关键财务信息</td>
</tr>
<tr>
<td><strong><a href="https://github.com/PaddlePaddle/PaddleOCR">PaddleOCR</a></strong></td>
<td>百度开源的OCR工具包，支持80+种语言的文字识别</td>
<td>处理多语言财务单据，自动录入费用数据，支持中文、英文、日文等主要市场</td>
</tr>
<tr>
<td><strong><a href="https://github.com/pandas-dev/pandas">pandas</a></strong></td>
<td>Python数据分析库，提供强大的数据处理和分析功能</td>
<td>构建自动化财务报表系统，生成利润表、现金流量表等关键财务报告</td>
</tr>
<tr>
<td><strong><a href="https://github.com/duckdb/duckdb">DuckDB</a></strong></td>
<td>高性能分析型数据库，支持复杂查询和数据分析</td>
<td>快速处理大量交易数据，生成实时财务仪表板和盈利分析报告</td>
</tr>
<tr>
<td><strong><a href="https://github.com/pola-rs/polars">Polars</a></strong></td>
<td>高性能数据处理框架，专为大数据分析优化</td>
<td>处理海量订单和财务数据，快速计算各产品线、各市场的盈利能力</td>
</tr>
</table>

### 8 合规/风险管理（Compliance & Risk）

包括 HS Code、各国法规、产品风险。

<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="https://github.com/scikit-learn/scikit-learn">scikit-learn</a></strong></td>
<td>机器学习库，提供分类、回归、聚类等算法</td>
<td>基于产品特征自动分类HS编码，提高海关申报准确性，降低清关风险</td>
</tr>
<tr>
<td><strong><a href="https://github.com/elastic/elasticsearch">ElasticSearch</a></strong></td>
<td>分布式搜索和分析引擎，支持全文检索和复杂查询</td>
<td>构建法规知识库，快速检索各国进出口法规，确保产品合规性</td>
</tr>
<tr>
<td><strong><a href="https://github.com/stanford-futuredata/ColBERT">ColBERT</a></strong></td>
<td>高效的BERT检索模型，支持语义搜索和文档匹配</td>
<td>智能匹配产品与相关法规条款，自动识别潜在合规风险点</td>
</tr>
<tr>
<td><strong><a href="https://github.com/ultralytics/ultralytics">YOLOv8</a></strong></td>
<td>最新的目标检测模型，支持实时图像识别和分析</td>
<td>自动检测产品图片中的违规内容，如品牌侵权、禁售物品等</td>
</tr>
<tr>
<td><strong><a href="https://github.com/IDEA-Research/GroundingDINO">GroundingDINO</a></strong></td>
<td>开放词汇目标检测模型，支持自然语言描述的物体检测</td>
<td>根据平台规则描述自动检测产品图片合规性，预防listing被下架</td>
</tr>
<tr>
<td><strong><a href="https://github.com/microsoft/DeBERTa">DeBERTa</a></strong></td>
<td>微软开源的预训练语言模型，在文本理解任务上表现优异</td>
<td>分析产品描述文本，识别可能违反平台政策的敏感词汇和表述</td>
</tr>
<tr>
<td><strong><a href="https://github.com/pytorch/fairseq">RoBERTa</a></strong></td>
<td>Facebook优化的BERT模型，在文本分类任务上性能卓越</td>
<td>自动分类产品描述风险等级，标记需要人工审核的高风险内容</td>
</tr>
</table>

### 9 供应链、库存预测、物流规划（Logistics & SCM）

<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="https://github.com/facebook/prophet">Prophet</a></strong></td>
<td>Facebook开源的时间序列预测工具，支持季节性和趋势分析</td>
<td>预测各SKU的需求变化，制定精准的补货计划，避免缺货和积压风险</td>
</tr>
<tr>
<td><strong><a href="https://github.com/unit8co/darts">Darts</a></strong></td>
<td>现代时间序列预测库，支持多变量预测和深度学习</td>
<td>综合考虑促销、季节性、市场趋势等因素，构建高精度库存需求预测模型</td>
</tr>
<tr>
<td><strong><a href="https://github.com/awslabs/gluonts">GluonTS</a></strong></td>
<td>基于深度学习的概率时间序列建模工具包</td>
<td>处理多产品、多仓库的复杂库存预测任务，提供不确定性量化</td>
</tr>
<tr>
<td><strong><a href="https://github.com/google/or-tools">OR-Tools</a></strong></td>
<td>Google开源的运筹优化工具包，支持线性规划和约束求解</td>
<td>优化补货策略和仓库分配，在成本、时效、库存水平间找到最优平衡点</td>
</tr>
<tr>
<td><strong><a href="https://github.com/networkx/networkx">NetworkX</a></strong></td>
<td>Python图论和网络分析库，支持复杂网络建模</td>
<td>建模多仓库供应链网络，优化货物流转路径，降低物流成本和配送时间</td>
</tr>
</table>

**应用案例**
- 需求预测
- 补货策略优化
- 物流路线优化
- 多仓网络建模

## AI Agents & Workflow 自动化引擎

用于构建 自动化运营系统、广告自动驾驶、客服自动处理。

<table width="100%">
<tr>
<th width="25%">类型</th>
<th width="75%">工具</th>
</tr>
<tr>
<td>LLM Agents</td>
<td><a href="https://github.com/langchain-ai/langchain">LangChain</a>, <a href="https://github.com/langchain-ai/langgraph">LangGraph</a>, <a href="https://github.com/Significant-Gravitas/AutoGPT">AutoGPT</a>, <a href="https://github.com/microsoft/AdaAgent">AdaAgent</a></td>
</tr>
<tr>
<td>Workflow Orchestration</td>
<td><a href="https://github.com/apache/airflow">Airflow</a>, <a href="https://github.com/PrefectHQ/prefect">Prefect</a>, <a href="https://github.com/dagster-io/dagster">Dagster</a></td>
</tr>
<tr>
<td>RPA 自动化</td>
<td><a href="https://github.com/kelaberetiv/TagUI">TagUI</a>, <a href="https://github.com/robotframework/robotframework">Robot Framework</a></td>
</tr>
<tr>
<td>多 Agent 系统</td>
<td><a href="https://github.com/microsoft/autogen">AutoGen</a>, <a href="https://github.com/joaomdmoura/crewAI">CrewAI</a></td>
</tr>
</table>

## 开发者工具 & AI Infra

LLM / RAG / Fine-tuning / Serving

### 模型
- [LLaMA-3](https://github.com/meta-llama/llama3), [Mixtral](https://github.com/mistralai/mistral-src), [Gemma](https://github.com/google/gemma_pytorch), [Qwen2](https://github.com/QwenLM/Qwen2)
- [Mistral-7B-Instruct](https://github.com/mistralai/mistral-src)
- DeepSeek-V2 / R1 style

### 微调工具
- [LoRA (PEFT)](https://github.com/huggingface/peft)
- [QLoRA](https://github.com/artidoro/qlora)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- OpenInstruct dataset

### RAG 框架
- [LlamaIndex](https://github.com/run-llama/llama_index)
- [Haystack](https://github.com/deepset-ai/haystack)
- [RAGFlow](https://github.com/infiniflow/ragflow)

### 推理与部署
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Ollama](https://github.com/ollama/ollama)
- [SGLang](https://github.com/sgl-project/sglang)

## 数据工程 & 可视化

<table width="100%">
<tr>
<th width="25%">类型</th>
<th width="75%">工具</th>
</tr>
<tr>
<td>数据处理</td>
<td><a href="https://github.com/pola-rs/polars">Polars</a>, <a href="https://github.com/duckdb/duckdb">DuckDB</a>, <a href="https://github.com/apache/spark">Spark</a>, <a href="https://github.com/pandas-dev/pandas">pandas</a></td>
</tr>
<tr>
<td>ETL</td>
<td><a href="https://github.com/airbytehq/airbyte">Airbyte</a>, <a href="https://github.com/mage-ai/mage-ai">Mage</a>, <a href="https://github.com/dlt-hub/dlt">dlt</a></td>
</tr>
<tr>
<td>可视化</td>
<td><a href="https://github.com/metabase/metabase">Metabase</a>, <a href="https://github.com/apache/superset">Superset</a>, <a href="https://github.com/grafana/grafana">Grafana</a>, <a href="https://github.com/getredash/redash">Redash</a></td>
</tr>
</table>

## AI Research 工具

<table width="100%">
<tr>
<th width="25%">类别</th>
<th width="75%">工具</th>
</tr>
<tr>
<td>评估 Benchmark</td>
<td><a href="https://github.com/EleutherAI/lm-evaluation-harness">lm-eval-harness</a>, <a href="https://github.com/tatsu-lab/alpaca_eval">alpaca-eval</a></td>
</tr>
<tr>
<td>数据标注</td>
<td><a href="https://github.com/heartexlabs/label-studio">Label Studio</a></td>
</tr>
<tr>
<td>数据增强</td>
<td><a href="https://github.com/facebookresearch/AugLy">AugLy</a>, <a href="https://github.com/makcedward/nlpaug">nlpaug</a></td>
</tr>
</table>

## Roadmap

- [ ] 完善各应用场景的最佳实践指南
- [ ] 建立性能基准测试框架
- [ ] 增加更多实际案例研究
- [ ] 构建社区贡献流程
- [ ] 开发配套的教程和工作坊

## 贡献指南

我们欢迎社区贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细的参与指南。

### 如何贡献

1. **Fork** 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 **Pull Request**

## 许可证

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

本知识库采用 [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) 许可证发布。

---

**[返回顶部](#cbec-ai-hub-跨境电商ai知识中心)**
