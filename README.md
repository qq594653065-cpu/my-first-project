# my-first-project
专门用于从PDF、DOCX、TXT文件中提取和切分题目内容。无需网络连接和API调用，支持多格式文档的智能解析和结构化输出
# Document Extractor - 纯本地文档处理器

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

**Document Extractor** 是一个高性能、纯本地的文档处理工具，专门用于从PDF、DOCX、TXT文件中提取和切分题目内容。无需网络连接和API调用，支持多格式文档的智能解析和结构化输出。

## ✨ 核心特性

### 📁 **多格式文档支持**
- **PDF文件**：智能调度器优先使用`pdfplumber`，失败时自动降级到OCR（PaddleOCR/EasyOCR）
- **DOCX文件**：原生支持Word文档解析，保留完整格式和样式
- **TXT文件**：自动检测编码（UTF-8, GBK等），支持大文件分批处理

### 🚀 **性能优化特性**
- **并行处理**：多线程并发处理，支持自定义工作线程数
- **OCR缓存机制**：MD5哈希缓存重复图像，减少冗余识别
- **智能OCR选择**：基于图像质量自动选择最佳OCR引擎（PaddleOCR/EasyOCR）
- **内存管理**：实时内存监控，支持大文件分批处理，防止内存溢出
- **断点续传**：进度持久化存储，支持中断后恢复处理

### 🔧 **高级功能**
- **线程安全OCR**：每个线程独立OCR实例，避免资源竞争
- **智能布局分析**：基于行间距和布局特征的高精度题目切分
- **配置管理**：支持YAML/JSON配置文件、环境变量、命令行参数多级配置
- **实时进度显示**：终端进度条和详细日志，直观展示处理状态
- **错误隔离**：单文件处理失败不影响整体流程，详细的错误报告

### 📊 **监控与统计**
- **性能监控**：记录文件处理时间、内存使用峰值、OCR命中率
- **缓存统计**：OCR缓存命中率、缓存大小、优化效果分析
- **进度跟踪**：实时显示已完成文件数、题目数、内存使用情况

## 📦 安装说明

### 基础依赖
```bash
# Python 3.8+
python --version

# 安装核心依赖
pip install pdfplumber python-docx
```

### OCR引擎（可选）
```bash
# 安装PaddleOCR（推荐）
pip install paddlepaddle paddleocr

# 安装EasyOCR（备用）
pip install easyocr

# 安装图像处理库
pip install pillow numpy opencv-python
```

### 完整安装（包含所有可选依赖）
```bash
pip install pdfplumber python-docx paddlepaddle paddleocr easyocr pillow numpy opencv-python psutil PyYAML
```

## 🚀 快速开始

### 基本用法
```bash
# 使用默认配置（输入目录：~/Desktop/aaa原始资料）
python document_extractor.py

# 指定输入目录
python document_extractor.py --input ./my_documents

# 指定输出目录
python document_extractor.py --input ./documents --output ./results
```

### 高级用法
```bash
# 启用并行处理（8个线程）
python document_extractor.py --input ./documents --parallel-workers 8

# 启用断点续传
python document_extractor.py --input ./documents --resume-enable 1

# 启用OCR缓存和智能选择
python document_extractor.py --input ./documents --ocr-cache-enable 1 --ocr-smart-selection 1

# 内存优化（限制最大内存使用率）
python document_extractor.py --input ./documents --max-memory-percent 70 --batch-size 500

# 详细日志输出
python document_extractor.py --input ./documents --log-level DEBUG

# 试运行模式（只检查不处理）
python document_extractor.py --input ./documents --dry-run
```

## ⚙️ 配置选项

### 配置优先级
```
命令行参数 > 环境变量 > 配置文件 > 默认值
```

### 配置文件示例（`config.yaml`）
```yaml
# 输入输出配置
input_dir: "~/Desktop/my_documents"
output_dir: "./extracted_results"

# 并行处理配置
parallel_enable: true
parallel_workers: 4

# 内存管理配置
batch_size: 1000
max_memory_percent: 80.0

# OCR配置
ocr_cache_enable: true
ocr_cache_max_size: 1000
ocr_smart_selection: true
ocr_preferred_engine: "paddleocr"

# 断点续传配置
resume_enable: true

# 日志配置
log_level: "INFO"
log_to_file: true
log_to_console: true
```

### 环境变量示例
```bash
# Windows PowerShell
$env:INPUT_DIR="C:\Users\Username\Documents\test"
$env:PARALLEL_WORKERS="8"
$env:OCR_CACHE_ENABLE="1"
$env:LOG_LEVEL="DEBUG"

# Linux/macOS
export INPUT_DIR="~/Documents/test"
export PARALLEL_WORKERS="8"
export OCR_CACHE_ENABLE="1"
export LOG_LEVEL="DEBUG"
```

## 📄 输出格式

### 输出文件结构
```
output_jsons/
├── extracted_questions.json    # 提取结果
├── logs/
│   └── extraction_YYYYMMDD_HHMMSS.log  # 详细日志
├── checkpoints/
│   └── progress.json           # 断点续传进度
└── performance_stats.json      # 性能统计
```

### JSON输出示例
```json
[
  {
    "source_id": "文件名_页码_题目序号",
    "source_file": "原始文件名.pdf",
    "page": 1,
    "raw_text": "完整题目文本，包含题干、选项、表格等所有内容。公式如 $E=mc^2$ 会被保留。",
    "metadata": {
      "extracted_options": ["A. 选项一文本", "B. 选项二文本"],
      "has_table": true,
      "is_scanned": false,
      "module_type": "data_analysis",
      "extraction_method": "pdf_plumber"
    }
  }
]
```

## 🛠️ 技术架构

### 处理流程
```
原始文档（PDF/DOCX/TXT）
    ↓
文件检测与分类
    ↓
智能调度器
    ├── PDF: pdfplumber优先 → 降级OCR → 混合结果
    ├── DOCX: python-docx原生解析
    └── TXT: 编码检测 → 分批处理
    ↓
布局分析与题目切分
    ↓
结果结构化与验证
    ↓
输出JSON + 统计信息
```

### OCR智能调度
```
图像质量评估 → 智能选择引擎
    ├── 高质量图像 → PaddleOCR（精度优先）
    └── 低质量图像 → EasyOCR（鲁棒性优先）
    
OCR缓存机制
    ├── MD5图像哈希 → 缓存查询
    ├── 缓存命中 → 直接返回
    └── 缓存未命中 → OCR识别 → 缓存结果
```

## 📈 性能优化

### 并行处理策略
- **动态线程池**：根据文件数量和CPU核心数自动调整
- **工作窃取**：避免线程空闲，提高CPU利用率
- **文件级并行**：每个文件独立处理，互不干扰

### 内存管理
- **分批处理**：大文件自动分块，避免一次性内存加载
- **内存监控**：实时监控内存使用，超过阈值自动清理
- **资源释放**：处理完成后立即释放OCR引擎和图像数据

### 缓存优化
- **LRU缓存策略**：自动淘汰最久未使用的缓存条目
- **缓存统计**：实时显示命中率，指导缓存大小调整
- **智能预热**：常用图像模式自动预加载

## 🔍 错误处理与调试

### 常见问题解决
1. **PDF解析失败**
   - 检查pdfplumber是否正确安装：`pip show pdfplumber`
   - 尝试启用OCR降级：`--ocr-cache-enable 1`

2. **内存不足**
   - 减少并行线程数：`--parallel-workers 2`
   - 减小批处理大小：`--batch-size 200`
   - 降低内存限制：`--max-memory-percent 60`

3. **OCR识别精度低**
   - 启用智能引擎选择：`--ocr-smart-selection 1`
   - 检查图像质量，尝试预处理图像
   - 使用特定OCR引擎：设置`OCR_PREFERRED_ENGINE=paddleocr`

### 调试模式
```bash
# 启用详细日志
python document_extractor.py --log-level DEBUG

# 检查依赖
python document_extractor.py --dry-run

# 单文件测试
python document_extractor.py --input ./test.pdf --parallel-enable 0
```

## 📚 使用场景

### 教育领域
- **题库建设**：从历年试卷中提取题目，建立结构化题库
- **教学资源整理**：整理PDF讲义、Word文档中的练习题
- **在线学习平台**：批量处理上传的文档，自动生成练习模块

### 企业应用
- **内部培训**：处理培训材料，生成在线测试题目
- **知识库建设**：从技术文档中提取问题，构建FAQ系统
- **考试系统**：支持多种格式的试题导入和解析

### 个人学习
- **错题整理**：从扫描试卷中提取错题，建立个人错题本
- **学习笔记**：整理电子书中的重点题目和知识点
- **刷题系统**：配合前端刷题系统，实现完整的离线学习方案

## 🤝 贡献指南

### 开发环境设置
```bash
# 克隆项目
git clone <repository-url>
cd document_extractor

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 安装开发依赖
pip install -r requirements-dev.txt
```

### 代码规范
- 遵循PEP 8编码规范
- 使用类型注解提高代码可读性
- 添加适当的文档字符串和单元测试

### 提交规则
- 使用语义化提交信息
- 确保所有测试通过
- 更新相关文档和示例

## 📄 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## ⭐ 致谢

感谢以下开源项目的支持：
- [pdfplumber](https://github.com/jsvine/pdfplumber) - 优秀的PDF解析库
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 高精度OCR工具
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - 多语言OCR库
- [python-docx](https://github.com/python-openxml/python-docx) - Word文档处理库

## 📞 支持与反馈

如果您在使用过程中遇到问题或有改进建议：
1. 查看 [Issues](https://github.com/yourusername/document_extractor/issues) 页面
2. 提交详细的错误报告，包括：
   - 操作系统和Python版本
   - 错误日志和堆栈跟踪
   - 重现问题的步骤
   - 相关文件和配置信息

---

**Document Extractor** - 让文档处理更智能、更高效！
