# QSAR-Streamlit: 智能定量构效关系建模平台

这是一个使用 Streamlit 构建的交互式Web应用，旨在为药物研发人员提供一个无需编码、端到端的定量构效关系（QSAR）建模工具。平台集成了数据处理、分子嵌入、模型训练、性能评估和实时预测等关键功能。

---

## ✨ 核心功能

-   **🖥️ 交互式Web界面**：基于 Streamlit 构建，用户友好，无需编程经验。
-   **📤 灵活的数据输入**：支持上传CSV文件、手动输入SMILES数据或使用服务器上已有的数据集。
-   **🔬 多种分子嵌入方法**：内置多种主流分子表示方法，包括：
    -   RDKit Morgan 指纹
    -   ChemBERTa 预训练模型嵌入
    -   SMILES Transformer 嵌入
-   **🤖 集成多种机器学习算法**：支持分类和回归任务，集成以下常用算法：
    -   LightGBM
    -   XGBoost
    -   随机森林 (Random Forest)
    -   支持向量机 (SVM)
-   **📈 全面的模型评估**：自动计算并可视化关键评估指标（如准确率、F1分数、混淆矩阵、ROC曲线、R²分数等）。
-   **🧪 无缝的实时预测**：模型训练完成后，可立即在页面下方对新的单分子或批量分子进行活性/毒性预测，结果实时呈现，无需刷新。
-   **📁 项目式数据管理**：自动为每次实验创建项目，方便用户管理和追溯数据、嵌入和模型。
-   **📚 高级辅助功能**：包含文献挖掘和AI文献摘要工具，辅助科研探索。

---

## 🚀 如何使用

1.  **启动应用**
    在终端中运行以下命令：
    ```bash
    streamlit run QSAR/app.py
    ```

2.  **进入智能建模**
    在左侧导航栏中，点击 `智能建模` 进入核心功能区。

3.  **准备数据**
    -   选择 `新建项目`。
    -   通过上传CSV文件或手动输入SMILES及对应标签的方式提供数据。
    -   在界面引导下，指定哪个是SMILES列，哪个是标签列。

4.  **提取嵌入**
    -   从下拉菜单中选择一种分子嵌入方法（如 `RDKit 指纹`）。
    -   点击 `提取嵌入向量`，等待处理完成。

5.  **训练模型**
    -   嵌入提取成功后，模型训练界面会自动出现。
    -   选择任务类型（分类/回归）、算法和测试集比例。
    -   点击 `🚀 开始训练`。

6.  **查看结果与实时预测**
    -   训练完成后，页面下方将稳定显示 **模型评估结果** 和 **实时分子预测** 两个模块。
    -   您可以查看模型的各项性能指标。
    -   在预测模块中输入新分子的SMILES，即可获得预测结果。

---

## 🛠️ 安装指南

1.  **克隆仓库**
    ```bash
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2.  **创建并激活虚拟环境** (推荐)
    ```bash
    python -m venv venv
    source venv/bin/activate  # on Windows, use `venv\Scripts\activate`
    ```

3.  **安装依赖**
    ```bash
    pip install -r QSAR/requirements.txt
    ```
    > **注意**: 安装 `rdkit` 可能需要 `conda`。如果 `pip` 安装失败，请尝试使用 `conda install -c conda-forge rdkit`。

4.  **运行应用**
    ```bash
    streamlit run QSAR/app.py
    ```

---

## 📂 项目结构

```
QSAR/
├── app.py                  # Streamlit 应用主程序
├── utils/                  # 核心功能模块
│   ├── embedding_utils.py  # 分子嵌入提取工具
│   ├── model_utils.py      # 模型训练与评估工具
│   └── user_data_utils.py  # 用户数据和项目管理工具
├── scripts/                # 用于复杂嵌入提取的外部脚本
├── model/                  # 存放预训练模型（如ChemBERTa）
├── data/                   # 存放示例数据集
├── users/                  # 存储每个用户的项目、数据和嵌入
├── database/               # 数据库文件（用于存储项目元数据）
└── requirements.txt        # Python 依赖库
```

---

## 🔧 技术栈

-   **Web框架**: [Streamlit](https://streamlit.io/)
-   **数据处理**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
-   **化学信息学**: [RDKit](https://www.rdkit.org/)
-   **机器学习**: [Scikit-learn](https://scikit-learn.org/), [LightGBM](https://lightgbm.readthedocs.io/), [XGBoost](https://xgboost.readthedocs.io/)
-   **数据可视化**: [Matplotlib](https://matplotlib.org/)

## 🔬 分子嵌入方法

### 1. RDKit分子指纹
```python
# 支持的指纹类型
- Morgan指纹 (ECFP): 基于原子环境的圆形指纹
- MACCS指纹: 167位固定结构键指纹  
- 拓扑指纹: 基于分子拓扑结构
- Atom Pair指纹: 原子对描述符
```
- **维度**: 1024-4096 (可配置)
- **速度**: 极快 (~1000分子/秒)
- **适用**: 大规模筛选、快速建模

### 2. ChemBERTa嵌入
```python
# 模型信息
- 模型: seyonec/PubChem10M_SMILES_BPE_450k
- 架构: RoBERTa-based Transformer
- 训练数据: PubChem 1000万分子
```
- **维度**: 768
- **速度**: 中等 (~100分子/秒)
- **适用**: 高质量表示学习

### 3. SMILES Transformer嵌入
```python
# 自定义模型
- 架构: 专门优化的Transformer
- 训练: 大规模SMILES数据
- 特点: 序列理解能力强
```
- **维度**: 1024
- **速度**: 较慢 (~50分子/秒)
- **适用**: 精确建模、研究用途

## 🤖 机器学习算法

| 算法 | 类型 | 优势 | 适用场景 |
|------|------|------|----------|
| **LightGBM** | 梯度提升 | 速度快、内存效率高 | 大数据集、快速建模 |
| **XGBoost** | 梯度提升 | 性能优异、鲁棒性强 | 竞赛、高精度需求 |
| **Random Forest** | 集成学习 | 稳定、可解释性好 | 特征重要性分析 |
| **SVM** | 支持向量机 | 理论基础扎实 | 小数据集、非线性问题 |

## 📊 评估指标体系

### 分类任务
- **准确率** (Accuracy): 整体预测正确率
- **精确率** (Precision): 正例预测准确性
- **召回率** (Recall): 正例识别完整性
- **F1分数** (F1 Score): 精确率和召回率的调和平均
- **ROC-AUC**: 受试者工作特征曲线下面积
- **混淆矩阵**: 详细的分类结果分析

### 回归任务
- **均方误差** (MSE): 预测误差的平方均值
- **均方根误差** (RMSE): MSE的平方根
- **平均绝对误差** (MAE): 绝对误差的平均值
- **决定系数** (R²): 模型解释方差的比例

## 📚 文献挖掘功能

### 支持的数据库
- **PubMed**: 生物医学文献数据库
- **CrossRef**: 学术出版物元数据
- **Semantic Scholar**: AI驱动的学术搜索

### 检索功能
- **关键词搜索**: 支持布尔逻辑
- **高级筛选**: 年份、期刊、作者筛选
- **智能推荐**: 基于内容的关键词推荐

### 分析功能
- **发表趋势**: 时间序列分析
- **引用分析**: 影响力评估
- **研究热点**: 主题聚类分析

## 🧠 AI智能摘要

### 支持的AI模型
```python
# 预设模型
- Qwen/QwQ-32B          # 推理能力强
- Qwen/Qwen2.5-72B-Instruct  # 大参数量
- Qwen/Qwen2.5-32B-Instruct  # 平衡性能
- Qwen/Qwen2.5-14B-Instruct  # 中等规模
- Qwen/Qwen2.5-7B-Instruct   # 轻量级
- Qwen/Qwen3-8B         # 最新版本

# 自定义模型
- 支持任意硅基流动平台模型
```

### 分析模式
- **核心要点提炼**: 全面分析，提取关键信息
- **研究方法分析**: 重点分析技术路线和方法
- **创新点挖掘**: 深度挖掘创新性和突破点

### 技术特性
- **智能重试**: 自动处理API限流和错误
- **中文输出**: 专业的中文学术表达
- **批量处理**: 支持多篇文献同时分析

## 💾 数据格式规范

### 输入数据格式
```csv
# 标准格式
smiles,activity,molecular_weight,logP
CCO,1,46.07,0.31
CC(C)O,0,60.10,0.05
CC(C)(C)O,1,74.12,0.35
```

### 必需字段
- **SMILES列**: 分子结构表示 (必需)
- **标签列**: 目标变量 (分类/回归)

### 可选字段
- **分子性质**: 分子量、logP、极性表面积等
- **实验条件**: 温度、pH、浓度等
- **文献信息**: DOI、作者、期刊等

## 🔧 技术栈

### 前端框架
- **Streamlit**: 快速Web应用开发
- **HTML/CSS**: 自定义样式和布局

### 机器学习
- **scikit-learn**: 基础机器学习算法
- **LightGBM**: 微软梯度提升框架
- **XGBoost**: 极端梯度提升
- **PyTorch**: 深度学习框架

### 分子处理
- **RDKit**: 化学信息学工具包
- **Transformers**: 预训练模型库

### 数据处理
- **pandas**: 数据分析和处理
- **NumPy**: 数值计算
- **matplotlib/seaborn**: 数据可视化

### 网络请求
- **requests**: HTTP请求库
- **lxml**: XML/HTML解析

## 🛠️ 高级配置

### 模型文件配置
```bash
# SMILES Transformer模型文件
model/smiles_transformer/
├── vocab.pkl              # 词汇表文件
└── trfm_12_23000.pkl     # 预训练权重

# ChemBERTa模型 (自动下载)
model/bert-loves-chemistry/
└── chemberta/            # Hugging Face模型
```

### API配置
```python
# 硅基流动API配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_TOKEN = "your_api_token_here"

# 请求参数
MAX_TOKENS = 2048
TEMPERATURE = 0.7
REQUEST_INTERVAL = 2  # 秒
```

## 📈 性能优化建议

### 数据集大小建议
- **小数据集** (<1000): 推荐SVM、Random Forest
- **中等数据集** (1000-10000): 推荐XGBoost、LightGBM
- **大数据集** (>10000): 推荐LightGBM、深度学习

### 嵌入方法选择
- **快速原型**: RDKit指纹
- **平衡性能**: ChemBERTa
- **最高精度**: SMILES Transformer

### 系统资源
- **内存**: 建议16GB+ (大数据集)
- **CPU**: 多核处理器 (并行计算)
- **存储**: SSD推荐 (I/O密集)

## 🔍 故障排除

### 常见问题

**Q: 模型训练失败**
```bash
# 检查数据格式
- 确保SMILES列存在且有效
- 检查标签列数据类型
- 验证数据集大小 (>20样本)
```

**Q: AI摘要功能不可用**
```bash
# 检查API配置
- 验证API Token有效性
- 检查网络连接
- 确认模型名称正确
```

**Q: 嵌入提取失败**
```bash
# 检查模型文件
- 确认模型文件完整
- 检查文件路径正确
- 验证SMILES格式
```

### 日志查看
```bash
# Streamlit日志
streamlit run app.py --logger.level debug

# 系统日志
tail -f ~/.streamlit/logs/streamlit.log
```

## 🤝 贡献指南

### 开发环境设置
```bash
# 开发模式安装
pip install -e .

# 代码格式化
black app.py utils/

# 类型检查
mypy utils/
```

### 提交规范
- **feat**: 新功能
- **fix**: 错误修复
- **docs**: 文档更新
- **style**: 代码格式
- **refactor**: 代码重构

## 📄 许可证

本项目采用学术许可证，仅供教育和研究使用。商业使用请联系开发团队。

## 📞 技术支持

- **问题反馈**: 通过GitHub Issues
- **功能建议**: 欢迎提交Pull Request
- **技术交流**: 加入开发者社区

## 🏆 致谢

感谢以下开源项目的支持：
- [RDKit](https://www.rdkit.org/) - 化学信息学工具包
- [Hugging Face](https://huggingface.co/) - 预训练模型平台
- [Streamlit](https://streamlit.io/) - Web应用框架
- [硅基流动](https://siliconflow.cn/) - AI模型API服务

---

<div align="center">

**🧬 QSAR深度学习预测平台**

*让分子设计更智能，让科研更高效*

[![GitHub stars](https://img.shields.io/github/stars/username/repo.svg?style=social&label=Star)](https://github.com/SilverCopper/QSAR)
[![GitHub forks](https://img.shields.io/github/forks/username/repo.svg?style=social&label=Fork)](https://github.com/SilverCopper/QSAR)

© 2025 2251238朱江宁|CADD 课程设计
</div> 