# 测试目录

本目录包含项目的测试代码。

## 目录结构

- `unit/` - 单元测试
- `integration/` - 集成测试
- `fixtures/` - 测试数据和配置

## 运行测试

运行所有测试：
```bash
pytest tests/
```

运行特定测试：
```bash
pytest tests/unit/test_recommender.py
```

生成覆盖率报告：
```bash
pytest --cov=. tests/
```

## 测试规范

1. 测试文件以 `test_` 开头
2. 测试函数以 `test_` 开头
3. 使用描述性的测试名称
4. 包含正常和异常情况的测试
