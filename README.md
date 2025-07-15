# 订单分析可视化工具

## 📊 功能概述

这个工具用于分析交易策略的订单数据，提供多维度的可视化分析。

## 🗂️ 文件结构

```
orders-analysis/
├── order_visualizer.py          # 核心分析模块
├── universal_order_analyzer.py  # 通用启动脚本  
├── MACD-long-crypto/            # MACD策略分析示例
│   ├── MACD-long-crypto-2023-2024.csv
│   └── run_analysis.py         # 策略专用脚本
└── crypto-stock-momentum/       # 其他策略文件夹
```

## 🚀 使用方法

### 方法1: 通用分析器 (推荐)

```bash
# 进入任意策略文件夹
cd orders-analysis/your-strategy-folder/

# 运行通用分析器
python ../universal_order_analyzer.py
```

### 方法2: 策略专用脚本

```bash
# 在特定策略文件夹中
cd orders-analysis/MACD-long-crypto/
python run_analysis.py
```

## 📋 数据格式要求

CSV文件必须包含以下列：
- `Time`: 时间戳
- `Symbol`: 交易对符号  
- `Price`: 价格
- `Quantity`: 数量
- `Type`: 订单类型
- `Status`: 订单状态 (Filled/Invalid)
- `Value`: 订单价值
- `Tag`: 标签 (可选)

## 📊 生成的图表

1. **仓位大小分析**
   - 仓位分布饼图
   - 平均收益柱状图
   - 累计收益趋势
   - 收益分布箱线图

2. **收益类型分析** 
   - 盈亏分布
   - 订单金额分布
   - 每日盈亏情况
   - 交易对盈亏对比

3. **综合分析**
   - 累计收益曲线
   - 交易频率分析
   - 仓位vs收益散点图
   - 收益贡献分析
   - 买卖方向分析
   - 收益分布直方图

## ⚙️ 特性

- ✅ 支持macOS中文字体显示
- ✅ 可选择保存/不保存图表
- ✅ 自动数据清洗和预处理
- ✅ 详细的统计报告
- ✅ 通用化设计，支持不同策略
- ✅ 错误处理和用户友好提示

## 📝 分析报告示例

```
📊 基本统计:
总交易次数: 2,183
总收益: 115,429.54
平均单笔收益: 52.88
胜率: 50.1%

💰 仓位大小分析:
Large 仓位: 721笔交易, 总收益62,292.82, 平均86.40
Medium 仓位: 741笔交易, 总收益60,019.88, 平均81.00
Small 仓位: 721笔交易, 总收益-6,883.16, 平均-9.55
```