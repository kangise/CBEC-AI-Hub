# B9. AI 产品图片与视频生成 Pipeline

> **路径**: Path B: 技术人 · **模块**: B9
> **最后更新**: 2026-03-15
> **难度**: ⭐⭐⭐ 高级
> **预计时间**: 每天 1 小时，2-3 周
> **前置模块**: 无（独立模块，但建议了解 [A7 视觉内容](../a-operators/a7-visual-content.md)）

🏠 [Hub 首页](../../README.md) · 📋 [Path B 总览](README.md)

---

## 📖 章节导航

1. [为什么需要 AI 图片 Pipeline](#1-为什么需要-ai-图片-pipeline) · 2. [技术栈选择](#2-技术栈选择) · 3. [ComfyUI 产品图工作流](#3-comfyui-产品图工作流) · 4. [API 方案](#4-api-方案midjourneydall-eflux) · 5. [批量生成 Pipeline](#5-批量生成-pipeline) · 6. [视频生成](#6-ai-视频生成) · 7. [质量控制与合规](#7-质量控制与合规) · 8. [完成标志](#8-完成标志)

---

## 本模块你将构建

- 一个 ComfyUI 产品图生成工作流（白底主图 + 场景图 + 信息图）
- 一个 API 驱动的批量图片生成 Pipeline（Midjourney/DALL-E/Flux）
- 一个产品视频自动生成系统
- 品牌视觉一致性保障机制

> 💡 **核心理念**：电商产品图是转化率的第一要素。传统方式是请摄影师拍摄（$500-2000/产品），AI 方式是用 ComfyUI/Midjourney 生成（$0-50/产品）。但 AI 生成不是"一键出图"，需要构建可重复、可控制、品牌一致的 Pipeline。

> 📎 **相关阅读**: [A7 视觉内容](../a-operators/a7-visual-content.md) — 运营视角的 AI 视觉内容方法论

---

## 1. 为什么需要 AI 图片 Pipeline

### 1.1 电商图片需求矩阵

| 图片类型 | 用途 | 数量/产品 | 传统成本 | AI 成本 |
|----------|------|----------|---------|---------|
| 白底主图 | Amazon/Shopify 主图 | 1 | $100-300 | $0-5 |
| 场景图 | 使用场景展示 | 3-5 | $200-500 | $5-20 |
| 信息图 | 尺寸/对比/功能说明 | 2-3 | $100-200 | $5-10 |
| A+ Content | 品牌故事图文 | 5-7 | $300-500 | $10-30 |
| 社交媒体 | Instagram/TikTok 素材 | 10-20/月 | $500-1000/月 | $20-50/月 |
| 广告素材 | PPC/Meta/Google Ads | 5-10 变体 | $200-500 | $10-30 |

### 1.2 AI 图片生成的挑战

| 挑战 | 说明 | 解决方案 |
|------|------|---------|
| 产品一致性 | AI 生成的产品外观可能与实物不同 | 使用产品实拍图作为参考（ControlNet/IP-Adapter） |
| 品牌一致性 | 不同图片风格不统一 | 固定 Prompt 前缀 + Style Reference |
| 平台合规 | Amazon 主图要求纯白底 | 后处理去背景 + 白底合成 |
| 文字渲染 | AI 生成的文字经常出错 | 后处理用 Pillow/Canva 叠加文字 |
| 版权风险 | AI 可能生成与已有作品相似的内容 | 使用商业许可工具 + 人工审核 |

---

## 2. 技术栈选择

### 2.1 方案对比

| 方案 | 优点 | 缺点 | 成本 | 适合 |
|------|------|------|------|------|
| ComfyUI（本地） | 完全控制、可自动化、免费 | 需要 GPU、学习曲线陡 | 硬件成本 | 大量图片、技术团队 |
| Midjourney | 质量最高、风格多样 | 无 API（需要 Discord）、不可控 | $10-30/月 | 少量高质量图片 |
| DALL-E 3（API） | 有 API、可编程 | 质量中等、风格有限 | 按量付费 | 批量生成、自动化 |
| Flux（本地/API） | 开源、质量高、可微调 | 需要 GPU | 免费/按量 | 技术团队、定制化 |
| Adobe Firefly | 商业安全、有赔偿保障 | 功能有限 | $10/月起 | 商业使用、合规优先 |
| Canva AI | 简单易用、模板丰富 | 灵活性低 | $13/月 | 非技术人员 |

### 2.2 推荐组合

```
推荐的 AI 图片技术栈：

主图/场景图生成：
├── ComfyUI + Flux（本地，完全控制）
├── 或 Midjourney（云端，质量最高）
└── 或 DALL-E 3 API（可编程，批量生成）

后处理：
├── rembg（Python 去背景）
├── Pillow（图片处理、文字叠加）
└── OpenCV（高级图片处理）

批量管理：
├── Python 脚本（自动化工作流）
└── Canva Brand Kit（模板管理）
```

---

## 3. ComfyUI 产品图工作流

### 3.1 安装 ComfyUI

```bash
# 克隆 ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 安装依赖
pip3 install -r requirements.txt

# 下载模型（Flux 推荐）
# 将模型文件放到 models/checkpoints/ 目录

# 启动
python3 main.py
# 浏览器打开 http://127.0.0.1:8188
```

### 3.2 产品图生成工作流

```
ComfyUI 电商产品图工作流：

输入节点：
├── 产品实拍图（Load Image）
├── Prompt（产品描述+场景描述）
└── 负面 Prompt（排除不想要的元素）

处理节点：
├── IP-Adapter（保持产品外观一致）
├── ControlNet（控制构图和姿态）
├── KSampler（生成图片）
└── VAE Decode（解码为图片）

输出节点：
├── Save Image（保存原图）
├── 去背景（rembg 节点）
└── 白底合成（Composite 节点）
```

> **真实案例**：ComfyUI 的产品图工作流可以实现自定义产品放置和细节保留，从一张产品图和简单的 Prompt 生成专业级产品照片（[MyAIForce](https://myaiforce.com/comfyui-product-photography/)）。运行本地 AI Pipeline 确保了成本效率、数据隐私和可扩展的生产级性能（[KeyValue Systems](https://www.keyvalue.systems/blog/webui-forge-evolution-automatic1111-to-comfyui-stable-diffusion/)）。

Content rephrased for compliance with licensing restrictions.

### 3.3 电商场景 Prompt 模板

```python
# 电商产品图 Prompt 模板库
PROMPT_TEMPLATES = {
    "white_background": {
        "positive": "professional product photography, {product}, centered, pure white background, studio lighting, high resolution, 8k, sharp focus, commercial photography",
        "negative": "blurry, low quality, text, watermark, logo, human, hand, shadow on background"
    },
    "lifestyle": {
        "positive": "lifestyle product photography, {product} in use, {scene}, natural lighting, warm tones, bokeh background, professional, editorial style",
        "negative": "blurry, low quality, text, watermark, artificial looking, oversaturated"
    },
    "flat_lay": {
        "positive": "flat lay photography, {product} with complementary items, top-down view, clean arrangement, soft shadows, minimalist, {color_scheme}",
        "negative": "cluttered, messy, blurry, low quality, text"
    },
    "infographic": {
        "positive": "clean infographic background, {product}, {color_scheme}, modern design, space for text overlay, professional",
        "negative": "text, numbers, charts, cluttered, busy"
    }
}

def generate_prompt(template_name: str, product: str, **kwargs) -> dict:
    """生成产品图 Prompt"""
    template = PROMPT_TEMPLATES[template_name]
    return {
        "positive": template["positive"].format(product=product, **kwargs),
        "negative": template["negative"]
    }
```

---

## 4. API 方案（Midjourney/DALL-E/Flux）

### 4.1 DALL-E 3 批量生成

```python
from openai import OpenAI
import requests
from pathlib import Path

client = OpenAI()

def generate_product_image(
    product_description: str,
    style: str = "white_background",
    size: str = "1024x1024",
    output_dir: str = "output"
) -> str:
    """用 DALL-E 3 生成产品图"""
    
    prompts = {
        "white_background": f"Professional product photography of {product_description}, centered on pure white background, studio lighting, high resolution, commercial quality",
        "lifestyle": f"Lifestyle product photography of {product_description} being used in a modern home setting, natural lighting, warm tones, editorial quality",
        "amazon_main": f"Amazon product listing main image: {product_description}, pure white background (#FFFFFF), product fills 85% of frame, no text or logos, professional studio photography"
    }
    
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompts[style],
        size=size,
        quality="hd",
        n=1
    )
    
    # 下载图片
    image_url = response.data[0].url
    Path(output_dir).mkdir(exist_ok=True)
    
    img_data = requests.get(image_url).content
    filepath = f"{output_dir}/{product_description[:30]}_{style}.png"
    with open(filepath, "wb") as f:
        f.write(img_data)
    
    return filepath

# 批量生成
products = [
    "wireless bluetooth earbuds with charging case",
    "stainless steel water bottle 32oz",
    "portable neck fan with LED display"
]

for product in products:
    for style in ["white_background", "lifestyle"]:
        path = generate_product_image(product, style)
        print(f"Generated: {path}")
```

### 4.2 去背景 + 白底合成

```python
from rembg import remove
from PIL import Image
import io

def create_amazon_main_image(input_path: str, output_path: str):
    """创建 Amazon 合规的白底主图"""
    # 读取图片
    with open(input_path, "rb") as f:
        input_data = f.read()
    
    # 去背景
    output_data = remove(input_data)
    
    # 创建白底画布
    fg = Image.open(io.BytesIO(output_data)).convert("RGBA")
    
    # 计算产品占比（Amazon 要求 85%+）
    bbox = fg.getbbox()
    product_w = bbox[2] - bbox[0]
    product_h = bbox[3] - bbox[1]
    
    # 创建正方形白底（产品占 85%）
    canvas_size = int(max(product_w, product_h) / 0.85)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 255))
    
    # 居中放置产品
    offset_x = (canvas_size - product_w) // 2 - bbox[0]
    offset_y = (canvas_size - product_h) // 2 - bbox[1]
    canvas.paste(fg, (offset_x, offset_y), fg)
    
    # 保存为 RGB（Amazon 不接受透明背景）
    canvas.convert("RGB").save(output_path, "JPEG", quality=95)
    print(f"Amazon main image saved: {output_path}")
```

---

## 5. 批量生成 Pipeline

### 5.1 完整的产品图生成 Pipeline

```python
class ProductImagePipeline:
    """电商产品图批量生成 Pipeline"""
    
    def __init__(self, method="dalle"):
        self.method = method  # dalle / comfyui / midjourney
    
    def generate_product_set(self, product: dict) -> dict:
        """为一个产品生成完整的图片集"""
        results = {}
        
        # 1. 白底主图
        results["main"] = self.generate(product, "white_background")
        results["main_amazon"] = self.post_process_amazon(results["main"])
        
        # 2. 场景图 x3
        scenes = ["modern living room", "outdoor setting", "office desk"]
        results["lifestyle"] = [
            self.generate(product, "lifestyle", scene=s) for s in scenes
        ]
        
        # 3. 信息图背景 x2
        results["infographic_bg"] = [
            self.generate(product, "infographic", color_scheme=c)
            for c in ["blue and white", "warm earth tones"]
        ]
        
        # 4. 社交媒体素材
        results["social"] = self.generate(product, "lifestyle", 
                                          scene="aesthetic flat lay")
        
        return results
    
    def batch_generate(self, products: list) -> list:
        """批量生成多个产品的图片集"""
        all_results = []
        for i, product in enumerate(products):
            print(f"Processing {i+1}/{len(products)}: {product['name']}")
            results = self.generate_product_set(product)
            all_results.append(results)
        return all_results
```

---

## 6. AI 视频生成

### 6.1 产品视频类型

| 类型 | 时长 | 用途 | AI 工具 |
|------|------|------|---------|
| 产品展示 | 15-30s | Amazon 视频、Shopify | Runway Gen-3 / Pika |
| 使用教程 | 30-60s | A+ Content、YouTube | Synthesia / HeyGen |
| 社交短视频 | 15-60s | TikTok/Reels/Shorts | CapCut AI / Runway |
| 广告视频 | 6-15s | PPC 视频广告 | Runway / Sora |

### 6.2 产品展示视频生成

```python
# 概念代码：用 Runway API 生成产品展示视频
import runway

def generate_product_video(
    product_image: str,
    motion_prompt: str = "slow 360 degree rotation, studio lighting",
    duration: int = 4  # 秒
) -> str:
    """从产品图生成展示视频"""
    
    task = runway.image_to_video.create(
        model="gen3a_turbo",
        prompt_image=product_image,
        prompt_text=motion_prompt,
        duration=duration
    )
    
    # 等待生成完成
    task = runway.tasks.retrieve(task.id)
    while task.status != "SUCCEEDED":
        import time
        time.sleep(5)
        task = runway.tasks.retrieve(task.id)
    
    return task.output[0]  # 视频 URL
```

---

## 7. 质量控制与合规

### 7.1 Amazon 图片合规检查

```python
def check_amazon_compliance(image_path: str) -> dict:
    """检查图片是否符合 Amazon 要求"""
    img = Image.open(image_path)
    issues = []
    
    # 尺寸检查（最小 1000px）
    if min(img.size) < 1000:
        issues.append(f"尺寸不足: {img.size}，最小需要 1000x1000")
    
    # 白底检查（主图）
    pixels = list(img.getdata())
    corners = [pixels[0], pixels[img.width-1], 
               pixels[-img.width], pixels[-1]]
    for i, corner in enumerate(corners):
        if not all(c > 240 for c in corner[:3]):
            issues.append(f"角落 {i} 不是纯白: {corner}")
    
    # 产品占比检查
    # ... (检查产品是否占画面 85%+)
    
    return {
        "compliant": len(issues) == 0,
        "issues": issues,
        "size": img.size,
        "format": img.format
    }
```

### 7.2 品牌一致性检查

| 检查项 | 方法 | 工具 |
|--------|------|------|
| 配色一致 | 提取主色调对比品牌色 | Pillow + ColorThief |
| 风格一致 | CLIP 嵌入相似度 | sentence-transformers |
| Logo 位置 | 模板检查 | Pillow |
| 文字字体 | OCR + 字体匹配 | Tesseract |

---

## 8. 完成标志

- [ ] 搭建 ComfyUI 或选择 API 方案
- [ ] 为一个产品生成完整图片集（主图+场景图+信息图）
- [ ] 实现去背景+白底合成的自动化流程
- [ ] 构建批量生成 Pipeline（一次处理 5+ 产品）
- [ ] 通过 Amazon 图片合规检查
- [ ] 生成至少 1 个产品展示视频

---
> 🏠 [Hub 首页](../../README.md) · 📋 [Path B 总览](README.md)
> 
> **Path B**: [B1 数据管道](b1-data-pipeline.md) · [B2 预测模型](b2-prediction-models.md) · [B3 RAG 知识库](b3-rag-knowledge-base.md) · [B4 AI Agent](b4-agent-workflow.md) · [B5 本地模型](b5-local-model-deploy.md) · [B6 MCP 集成](b6-mcp-agentic-workflow.md) · [B7 Review NLP](b7-review-nlp-system.md) · [B8 Dashboard](b8-ecommerce-dashboard.md) · [B9 AI 图片生成](b9-ai-image-pipeline.md)
> 
> **快速跳转**: [Path 0 基础](../0-foundations/) · [Path A 运营](../a-operators/) · [Path C 管理](../c-managers/) · [Path D 多平台](../d-platforms/) · [Path E 社交媒体](../e-social-media/)
