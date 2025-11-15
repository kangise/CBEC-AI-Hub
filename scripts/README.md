# 脚本目录

本目录包含项目维护和自动化脚本。

## 目录结构

- `setup/` - 环境设置脚本
- `deployment/` - 部署脚本
- `maintenance/` - 维护脚本
- `testing/` - 测试脚本

## 脚本说明

### 环境设置
- `setup_environment.sh` - 快速环境配置
- `install_dependencies.sh` - 依赖安装

### 部署相关
- `deploy_model.sh` - 模型部署脚本
- `update_services.sh` - 服务更新脚本

### 维护工具
- `check_links.py` - 链接检查
- `update_readme.py` - README更新
- `validate_data.py` - 数据验证

## 使用方法

大多数脚本可以直接运行：
```bash
./scripts/setup/setup_environment.sh
```

部分脚本需要参数，请查看脚本内的帮助信息：
```bash
python scripts/maintenance/check_links.py --help
```
