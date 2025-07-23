# 深度订单分析与出入场可视化工具

## 📊 功能概述

这是一个专为量化交易策略设计的综合分析工具。除了提供多维度的、富有洞察力的可视化分析外，还新增了**出入场可视化功能**，能够结合市场数据直观展示交易的买卖点位置和收益情况。

## 🗂️ 文件结构

```
.
├── main_analysis.py                      # 核心策略表现分析
├── profit_correlation_analyzer.py       # 利润来源多维相关性分析
├── symbol_analysis_visualizer.py        # 交易对深度对比可视化模块
├── position_size_detail_visualizer.py   # 仓位详细分析
├── order_analyzer.py                    # 核心数据处理与分析引擎
├── order_visualizer_charts.py           # 基础图表模块
├── simple_entry_exit_analyzer.py        # 🆕 简化版出入场分析器 (推荐)
├── test_simple_analyzer.py              # 🆕 测试脚本
├── universal_order_analyzer.py          # 通用启动脚本
└── MACD-long-crypto/                     # 示例策略文件夹
    ├── 2023-2024/HOUR/base.csv
    └── run_analysis.py
```

## 🚀 使用方法

### 🎯 出入场可视化分析 (新功能!)

**简化版分析器** (推荐使用):
```bash
# 使用anaconda环境
/opt/anaconda3/bin/python simple_entry_exit_analyzer.py

# 或者直接运行测试
/opt/anaconda3/bin/python test_simple_analyzer.py
```

**功能特点**:
- 📈 **自动K线图绘制**: 结合订单数据和市场OHLCV数据
- 🎯 **智能分辨率匹配**: 根据交易时间跨度自动选择最佳时间周期
- 💰 **买卖点标注**: 绿色△买入点，红色▽卖出点，连线显示收益
- 📊 **累计收益追踪**: 实时显示策略累计表现
- 🔄 **符号自动转换**: BTCUSD→BTCUSDT等自动匹配
- ⏰ **多时间周期**: 支持15分钟/小时/日线

### 1. 核心策略表现分析

```bash
python main_analysis.py
```

### 2. 利润来源相关性分析

```bash
python profit_correlation_analyzer.py
```

### 3. 交易对深度对比分析

```bash
python symbol_analysis_visualizer.py
```

### 4. 仓位详细分析

```bash
python position_size_detail_visualizer.py
```

## 📊 生成的图表

### 🆕 出入场可视化图表

**简化版分析器生成**:
- **K线价格图**: 展示市场价格走势
- **买卖点标注**: 精确标记每笔交易的进出场位置
- **收益连线**: 用颜色和数值显示每笔交易的盈亏
- **累计收益曲线**: 展示策略整体表现趋势

**示例输出**:
```
🎯 选择分辨率: 日线 (长期交易)
📈 找到数据文件: btc_1d_data_2018_to_2025.csv
✅ 成功加载市场数据: 2730 条K线

📊 交易摘要:
  总交易次数: 257
  总收益: +134,202.91
  平均收益: +522.19
  胜率: 21.4%
  盈利交易: 55 次
  亏损交易: 202 次
```

### 1. 核心表现仪表板

由 `main_analysis.py` 生成，包含三个核心图表：
- **仓位大小分析**: 不同仓位的交易分布、平均收益、累计收益和收益分布
- **利润来源分析**: 持仓时间vs盈亏、交易对vs盈亏
- **综合表现概览**: 整体累计收益、月度表现

### 2. 混合关联矩阵

由 `profit_correlation_analyzer.py` 生成，使用科学统计方法展示变量间关联度。

### 3. 交易对深度剖析仪表板

由 `symbol_analysis_visualizer.py` 生成，包含多维盈亏图、精细化仓位分析、3D月度表现曲面图。

## ⚙️ 核心特性

### 🆕 出入场可视化特性
- ✅ **四步自动化流程**: 
  1. 输入订单数据位置，解析订单
  2. 自动匹配最佳分辨率  
  3. 自动找到市场数据源，绘制K线图
  4. 根据订单时间戳标注买卖点
- ✅ **智能数据匹配**: 自动处理符号转换和时区问题
- ✅ **多格式兼容**: 支持多种时间戳格式和数据结构
- ✅ **专业图表**: 高质量可视化，自动保存PNG格式

### 传统分析特性
- ✅ **多维深度分析**: 提供远超基本统计的深度洞察
- ✅ **科学的关联性分析**: 自动为不同数据类型选择最合适的统计模型
- ✅ **高级可视化**: 包括3D曲面图、分面图、堆叠图等
- ✅ **动态分箱**: 根据数据分布动态划分仓位和持仓周期
- ✅ **智能坐标轴**: 自动检测长尾分布并应用对数刻度
- ✅ **模块化设计**: 每个分析功能都在独立的脚本中

## 📋 数据格式要求

### 订单数据 (CSV格式)
- `Time`: 时间戳
- `Symbol`: 交易对符号 (如BTCUSD, ETHUSD)
- `Price`: 价格
- `Quantity`: 数量 (正数=买入开仓, 负数=卖出平仓)
- `Type`: 订单类型
- `Status`: 订单状态 (Filled/Invalid)
- `Value`: 订单价值
- `Tag`: 标签 (可选)

### 市场数据 (OHLCV格式)
出入场分析器会自动从以下目录加载市场数据:
```
/Users/yukiarima/Desktop/Quant/QuantFramework/data/Crypto/
├── BTCUSDT/
│   ├── btc_1d_data_2018_to_2025.csv
│   ├── SPOT-BTCUSDT-1h-*.csv
│   └── SPOT-BTCUSDT-15m-*.csv
├── ETHUSD/
└── ...
```

## 🔧 环境要求

```bash
# Python依赖
pip install pandas numpy matplotlib scipy

# 推荐使用anaconda环境
conda activate base
```

## 📝 使用示例

```python
# 快速开始 - 简化版出入场分析
from simple_entry_exit_analyzer import SimpleEntryExitAnalyzer

# 创建分析器
analyzer = SimpleEntryExitAnalyzer("path/to/your/orders.csv")

# 分析指定交易对
analyzer.analyze_symbol("BTCUSD")

# 或分析所有交易对
analyzer.analyze_all_symbols()
```