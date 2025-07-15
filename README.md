# 深度订单分析与可视化工具 (Advanced Order Analysis & Visualization Toolkit)

## 📊 功能概述

这是一个专为量化交易策略设计的深度订单分析工具。它不止于简单的收益计算，而是提供多维度的、富有洞察力的可视化分析，帮助您深入理解策略的盈利来源、风险暴露和行为模式。

## 🗂️ 文件结构

```
.
├── main_analysis.py                 # 核心策略表现分析
├── profit_correlation_analyzer.py   # 利润来源多维相关性分析
├── symbol_analysis_visualizer.py    # 交易对深度对比可视化模块
├── order_analyzer.py                # 核心数据处理与分析引擎
├── order_visualizer_charts.py       # (旧) 基础图表模块
├── universal_order_analyzer.py      # (旧) 通用启动脚本
└── MACD-long-crypto/                # 示例策略文件夹
    ├── MACD-long-crypto-2023-2024.csv
    └── run_analysis.py
```

## 🚀 使用方法

项目提供了多个独立的分析入口，您可以根据需要运行。

### 1. 核心策略表现分析

运行 `main_analysis.py` 以获取策略的整体表现概览。

```bash
python main_analysis.py
```

### 2. 利润来源相关性分析

运行 `profit_correlation_analyzer.py` 以探索利润与仓位大小、持仓时间、交易品种等维度之间的深层关系。

```bash
python profit_correlation_analyzer.py
```

### 3. 交易对深度对比分析

运行 `symbol_analysis_visualizer.py` 以生成两个独立的、信息密度极高的图表，用于对比不同交易对的表现。

```bash
python symbol_analysis_visualizer.py
```

## 📊 生成的图表

本工具可以生成多种高级可视化图表，提供深度的策略洞察。

### 1. 核心表现仪表板

由 `main_analysis.py` 生成，包含三个核心图表：
- **仓位大小分析**: 不同仓位的交易分布、平均收益、累计收益和收益分布。
- **利润来源分析**:
    - **持仓时间 vs. 盈亏**: 在不同持仓周期下，盈利、亏损和净利润的分布情况。
    - **交易对 vs. 盈亏**: 各交易对的总收益和胜率对比。
- **综合表现概览**: 整体累计收益、月度表现等。

### 2. 混合关联矩阵

由 `profit_correlation_analyzer.py` 生成。它使用最适合的统计方法（Pearson, Cramér's V, Eta-squared）来科学地计算数值型和分类型变量之间的关联度，并用热力图展示。

### 3. 交易对深度剖析仪表板

由 `symbol_analysis_visualizer.py` 生成，包含：
- **多维盈亏图**: 一个图同时展示每个交易对的总盈亏、按仓位大小划分的贡献百分比，以及基于平均单笔交易计算的盈亏比。
- **精细化仓位分析**: 为每个交易对动态地、精细地划分仓位等级，并用线图展示仓位与平均利润的关系。
- **3D月度表现曲面图**: 一个平滑、美观的3D热力图，用于展示不同交易对在各个维度的盈利趋势。

### 4. 分面散点图

由 `symbol_analysis_visualizer.py` 生成。该图为每个交易对创建一个独立的子图，以精细化地展示仓位大小与盈利的关系，并能自动处理长尾分布。

## ⚙️ 核心特性

- ✅ **多维深度分析**: 提供远超基本统计的深度洞察。
- ✅ **科学的关联性分析**: 自动为不同数据类型选择最合适的统计模型。
- ✅ **高级可视化**: 包括3D曲面图、分面图、堆叠图等多种高级图表。
- ✅ **动态分箱**: 能够根据数据分布动态、科学地划分仓位和持仓周期。
- ✅ **智能坐标轴**: 自动检测长尾分布并应用对数刻度，优化图表可读性。
- ✅ **模块化设计**: 每个分析功能都在独立的脚本中，易于使用和扩展。
- ✅ **交互式选择**: 可在运行时选择是否保存生成的图表。

## 📋 数据格式要求(直接使用的情况下)

CSV文件必须包含以下列：
- `Time`: 时间戳
- `Symbol`: 交易对符号
- `Price`: 价格
- `Quantity`: 数量
- `Type`: 订单类型
- `Status`: 订单状态 (Filled/Invalid)
- `Value`: 订单价值
- `Tag`: 标签 (可选)