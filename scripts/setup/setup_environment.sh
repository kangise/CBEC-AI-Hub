#!/bin/bash

# CBEC-AI-Hub 环境设置脚本
# 用于快速配置开发环境

set -e

echo "🚀 开始设置 CBEC-AI-Hub 开发环境..."

# 检查Python版本
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        echo "✅ 检测到 Python $PYTHON_VERSION"
        
        # 检查版本是否满足要求 (>= 3.8)
        if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l) -eq 1 ]]; then
            echo "✅ Python 版本满足要求"
        else
            echo "❌ Python 版本过低，需要 3.8 或更高版本"
            exit 1
        fi
    else
        echo "❌ 未找到 Python3，请先安装 Python"
        exit 1
    fi
}

# 创建虚拟环境
create_venv() {
    if [ ! -d "venv" ]; then
        echo "📦 创建虚拟环境..."
        python3 -m venv venv
        echo "✅ 虚拟环境创建完成"
    else
        echo "✅ 虚拟环境已存在"
    fi
}

# 激活虚拟环境并安装依赖
install_dependencies() {
    echo "📥 安装基础依赖..."
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    
    # 安装基础依赖
    pip install -r requirements.txt 2>/dev/null || {
        echo "📝 创建基础 requirements.txt..."
        cat > requirements.txt << EOF
# 基础数据科学库
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0

# 深度学习框架
torch>=1.9.0
transformers>=4.0.0

# 数据可视化
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=5.0.0

# 自然语言处理
spacy>=3.4.0
langdetect>=1.0.9

# 计算机视觉
opencv-python>=4.5.0
Pillow>=8.0.0

# 推荐系统
implicit>=0.6.0
lightfm>=1.16

# 时间序列
prophet>=1.0.0

# 工具库
requests>=2.25.0
beautifulsoup4>=4.9.0
scrapy>=2.5.0

# 开发工具
jupyter>=1.0.0
pytest>=6.0.0
black>=21.0.0
flake8>=3.9.0
EOF
        pip install -r requirements.txt
    }
    
    echo "✅ 依赖安装完成"
}

# 设置Git hooks（如果是Git仓库）
setup_git_hooks() {
    if [ -d ".git" ]; then
        echo "🔧 设置 Git hooks..."
        
        # 创建pre-commit hook
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# 运行代码格式检查
echo "运行代码格式检查..."
black --check . || {
    echo "代码格式不符合要求，请运行: black ."
    exit 1
}

# 运行基础测试
echo "运行基础测试..."
python -m pytest tests/ -x || {
    echo "测试失败，请修复后再提交"
    exit 1
}
EOF
        
        chmod +x .git/hooks/pre-commit
        echo "✅ Git hooks 设置完成"
    fi
}

# 创建基础目录结构
create_directories() {
    echo "📁 检查目录结构..."
    
    directories=(
        "docs"
        "tools/data-processing"
        "tools/model-deployment"
        "tools/automation"
        "tools/monitoring"
        "datasets/preprocessing"
        "datasets/augmentation"
        "datasets/validation"
        "examples/infrastructure"
        "examples/recommendation"
        "examples/forecasting"
        "examples/nlp"
        "examples/computer-vision"
        "scripts/setup"
        "scripts/deployment"
        "scripts/maintenance"
        "assets/images"
        "assets/diagrams"
        "tests"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "📁 创建目录: $dir"
        fi
    done
    
    echo "✅ 目录结构检查完成"
}

# 主函数
main() {
    echo "CBEC-AI-Hub 环境设置"
    echo "===================="
    
    check_python
    create_directories
    create_venv
    install_dependencies
    setup_git_hooks
    
    echo ""
    echo "🎉 环境设置完成！"
    echo ""
    echo "下一步："
    echo "1. 激活虚拟环境: source venv/bin/activate"
    echo "2. 启动 Jupyter: jupyter notebook"
    echo "3. 查看示例代码: ls examples/"
    echo ""
    echo "更多信息请查看 README.md"
}

# 运行主函数
main "$@"
