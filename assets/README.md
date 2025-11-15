# 资源文件目录

本目录包含项目相关的静态资源文件。

## 目录结构

- `images/` - 图片资源
  - `logos/` - 项目和合作伙伴标志
  - `screenshots/` - 功能截图
  - `diagrams/` - 流程图和架构图
- `diagrams/` - 技术架构图表
  - `architecture/` - 系统架构图
  - `workflows/` - 工作流程图
  - `data-flow/` - 数据流图

## 文件规范

### 图片文件
- 格式：PNG, JPG, SVG
- 命名：使用小写字母和连字符
- 大小：尽量控制在合理范围内

### 图表文件
- 源文件：保留可编辑的源文件（如 .drawio, .psd）
- 导出格式：PNG 或 SVG
- 分辨率：适合文档显示

## 使用方式

在文档中引用资源：
```markdown
![架构图](../assets/diagrams/architecture/system-overview.png)
```
