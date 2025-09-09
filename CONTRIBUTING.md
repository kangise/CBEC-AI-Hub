---
layout: page
title: 贡献指南
permalink: /contributing/
---

# CBEC-AI-Hub 贡献指南

感谢您对 CBEC-AI-Hub 的贡献兴趣！本指南将帮助您了解如何为这个跨境电商AI知识中心做出贡献。

<div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 1rem; margin: 2rem 0;">
  <strong>🌟 每一份贡献都很重要！</strong> 无论您是添加新工具、修复错误，还是分享案例研究，您的贡献都会帮助整个社区。
</div>

## 🤝 贡献方式

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin: 2rem 0;">

  <div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
    <h3>🔧 添加新资源</h3>
    <p>推荐与跨境电商相关的AI工具、库、数据集或学习资源。</p>
    <a href="https://github.com/kangise/CBEC-AI-Hub/issues/new?template=resource_addition.md" style="color: #0366d6;">提交资源 →</a>
  </div>

  <div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
    <h3>📝 分享案例研究</h3>
    <p>贡献真实的项目实施经验和技术解决方案。</p>
    <a href="https://github.com/kangise/CBEC-AI-Hub/issues/new?template=case_study_submission.md" style="color: #0366d6;">提交案例研究 →</a>
  </div>

  <div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
    <h3>🐛 修复问题</h3>
    <p>通过修复失效链接、更新信息或纠正错误来提升质量。</p>
    <a href="https://github.com/kangise/CBEC-AI-Hub/issues" style="color: #0366d6;">查看问题 →</a>
  </div>

  <div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 1.5rem;">
    <h3>🌐 翻译内容</h3>
    <p>通过贡献翻译帮助更多开发者访问内容。</p>
    <a href="https://github.com/kangise/CBEC-AI-Hub/discussions" style="color: #0366d6;">讨论翻译 →</a>
  </div>

</div>

## 📋 贡献标准

### 新资源要求

**必须满足以下条件：**
- ✅ 开源或提供有意义的免费层级
- ✅ 与跨境电商AI直接相关
- ✅ 活跃维护（6个月内有更新）
- ✅ 良好的文档和示例
- ✅ 社区认可（100+ GitHub星标或广泛使用）

**优选特征：**
- 🌟 多语言支持
- 🌟 云原生或容器化
- 🌟 生产级性能
- 🌟 良好的API设计
- 🌟 活跃的社区生态系统

### 案例研究要求

**必需条件：**
- **真实性**：基于真实项目经验，不是理论或概念验证
- **完整性**：包含背景、解决方案、实施过程、结果和经验总结
- **技术深度**：提供充分的技术细节和实现方案
- **可复现性**：其他人应该能够参考和学习实施方法
- **业务价值**：明确的业务影响和投资回报率
- **跨境电商相关性**：与跨境电商AI应用直接相关

**案例研究提交流程：**
1. **填写案例研究模板** - 使用我们的[案例研究提交模板](https://github.com/kangise/CBEC-AI-Hub/issues/new?template=case_study_submission.md)
2. **社区讨论** - 在GitHub Issues中与维护者和社区讨论案例细节
3. **完善内容** - 根据反馈完善案例研究内容
4. **提交PR** - 将完整的案例研究文档提交到`case-studies/`目录
5. **审核发布** - 经过审核后正式发布

**案例研究格式：**
- 使用Markdown格式编写
- 包含代码示例和架构图
- 提供性能数据和业务指标
- 遵循我们的[案例研究模板](case-studies.md)结构

## 🚀 快速开始指南

### 1. Fork 和克隆

```bash
# 在 GitHub 上 Fork 仓库，然后：
git clone https://github.com/YOUR_USERNAME/CBEC-AI-Hub.git
cd CBEC-AI-Hub
```

### 2. 创建分支

```bash
git checkout -b add-new-resource
# 或者
git checkout -b fix-broken-links
```

### 3. 进行更改

遵循我们的格式指南：

```html
<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="URL">工具名称</a></strong></td>
<td>功能描述</td>
<td>跨境电商应用场景描述</td>
</tr>
</table>
```

**示例：**
```html
<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="https://github.com/streamlit/streamlit">Streamlit</a></strong></td>
<td>快速数据应用开发框架，Python原生、丰富组件、易于部署</td>
<td>构建跨境电商数据分析仪表板，实时监控销售指标、库存状态和市场趋势</td>
</tr>
</table>
```

### 4. 测试您的更改

```bash
# 安装本地测试依赖
npm install -g awesome-lint markdown-link-check

# 检查 awesome 列表格式
awesome-lint README.md

# 检查链接有效性
markdown-link-check README.md
```

### 5. 提交 Pull Request

```bash
git add .
git commit -m "feat: 添加 Streamlit 用于数据应用开发"
git push origin add-new-resource
```

然后在 GitHub 上使用我们的模板创建 Pull Request。

## 📝 格式指南

### 资源条目格式

使用HTML表格格式，确保列宽一致：

```html
<table width="100%">
<tr>
<th width="15%">工具</th>
<th width="35%">技术描述</th>
<th width="50%">跨境电商应用场景</th>
</tr>
<tr>
<td><strong><a href="GitHub链接">工具名称</a></strong></td>
<td>主要功能的简要描述</td>
<td>具体的跨境电商应用场景</td>
</tr>
</table>
```

### 描述指南

- **简洁**：用1-2句话准确描述核心功能
- **突出独特性**：强调工具的独特优势
- **避免营销语言**：使用客观的技术描述
- **跨境电商相关性**：明确说明在跨境电商中的具体应用

### 表格格式要求

- 使用 `<table width="100%">` 确保表格占满容器
- 列宽比例：工具(15%)、技术描述(35%)、跨境电商应用场景(50%)
- 工具名称使用 `<strong>` 标签加粗
- 链接使用 `<a href="">` 标签

### 链接要求

- 优先使用 GitHub 仓库链接
- 如无 GitHub，链接到官方网站
- 确保链接有效且指向正确资源

## 🏷️ 问题标签

| 标签 | 描述 |
|-------|-------------|
| `good-first-issue` | 适合新手的问题 |
| `help-wanted` | 需要社区帮助 |
| `enhancement` | 功能改进 |
| `bug` | 错误报告 |
| `resource-addition` | 新资源建议 |
| `documentation` | 文档相关 |

## 🎯 认可体系

### 贡献者级别

- **贡献者**：任何提交被接受PR的人
- **常规贡献者**：5+ 个被接受的贡献
- **核心贡献者**：持续重要贡献
- **维护者**：拥有写入权限的受信任社区成员

### 认可福利

- **README 致谢**：所有贡献者都会被列出
- **社交媒体推广**：重要贡献会被突出显示
- **推荐信**：为重要贡献者提供
- **早期访问**：预览新功能和内容

## 📞 获取帮助

需要帮助或有疑问？

- **[GitHub Issues](https://github.com/kangise/CBEC-AI-Hub/issues)** - 错误报告和功能请求
- **[GitHub Discussions](https://github.com/kangise/CBEC-AI-Hub/discussions)** - 一般讨论和问答
- **[邮件](mailto:maintainer@example.com)** - 私人咨询

## 📜 行为准则

### 我们的承诺

我们致力于为所有贡献者提供友好和包容的环境，无论背景、经验水平或身份如何。

### 期望行为

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 专注于对社区最有利的事情
- 对其他社区成员表现出同理心

### 不可接受的行为

- 骚扰、歧视或冒犯性评论
- 人身攻击或恶意行为
- 未经许可发布私人信息
- 在专业环境中不当的任何行为

### 执行

项目维护者有权利和责任删除、编辑或拒绝不符合本行为准则的贡献。

## 🎉 成功案例

<div style="background: #f6f8fa; border-radius: 8px; padding: 2rem; margin: 2rem 0;">
  <h3>最近的贡献</h3>
  <ul>
    <li><strong>v1.0.0 发布</strong> - 初始收集了100+精选资源</li>
    <li><strong>案例研究框架</strong> - 全面的技术解决方案示例</li>
    <li><strong>社区模板</strong> - GitHub issue 和 PR 模板以改善协作</li>
    <li><strong>自动化质量检查</strong> - 链接验证和格式检查</li>
  </ul>
</div>

---

<div style="text-align: center; margin: 3rem 0; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px;">
  <h3>准备好做出您的第一个贡献了吗？</h3>
  <p>加入我们的开发者、研究人员和从业者社区，共同构建跨境电商AI的未来！</p>
  <a href="https://github.com/kangise/CBEC-AI-Hub/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22" style="background: white; color: #667eea; padding: 12px 24px; border-radius: 6px; text-decoration: none; font-weight: bold; margin: 0 0.5rem;">查找新手友好问题</a>
  <a href="https://github.com/kangise/CBEC-AI-Hub/fork" style="background: rgba(255,255,255,0.2); color: white; padding: 12px 24px; border-radius: 6px; text-decoration: none; font-weight: bold; margin: 0 0.5rem; border: 1px solid white;">Fork 仓库</a>
</div>
