#!/usr/bin/env python3
"""
简化版出入场分析器
按照用户需求简化流程：
1. 输入订单数据位置，解析订单
2. 自动匹配分辨率
3. 找到数据源，画出K线图
4. 根据订单时间戳，画出买卖点
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from order_analyzer import OrderAnalyzer

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleEntryExitAnalyzer:
    """简化版出入场分析器"""
    
    def __init__(self, csv_path, market_data_path="/Users/yukiarima/Desktop/Quant/QuantFramework/data/Crypto"):
        """
        初始化分析器
        
        Args:
            csv_path (str): 订单数据CSV文件路径
            market_data_path (str): 市场数据目录路径
        """
        self.csv_path = csv_path
        self.market_data_path = market_data_path
        self.analyzer = None
        self.trades_data = None
        self.symbols = []
        
        print(f"🔄 正在加载订单数据: {csv_path}")
        self._load_order_data()
        
    def _load_order_data(self):
        """步骤1: 加载并解析订单数据"""
        try:
            self.analyzer = OrderAnalyzer(self.csv_path)
            if self.analyzer.processed_data is None or len(self.analyzer.processed_data) == 0:
                raise ValueError("未找到有效的交易数据")
            
            # 处理时区问题
            self.trades_data = self.analyzer.processed_data.copy()
            
            # 统一时区处理
            for col in ['Time', 'CloseTime']:
                if col in self.trades_data.columns:
                    self.trades_data[col] = pd.to_datetime(self.trades_data[col])
                    if self.trades_data[col].dt.tz is not None:
                        self.trades_data[col] = self.trades_data[col].dt.tz_convert('UTC').dt.tz_localize(None)
            
            self.symbols = list(self.trades_data['Symbol'].unique())
            print(f"✅ 成功加载 {len(self.trades_data)} 个交易记录")
            print(f"📊 包含交易对: {', '.join(self.symbols)}")
            
        except Exception as e:
            print(f"❌ 加载订单数据失败: {e}")
            raise
    
    def _determine_optimal_timeframe(self, symbol):
        """步骤2: 自动匹配最佳分辨率"""
        symbol_trades = self.trades_data[self.trades_data['Symbol'] == symbol]
        if symbol_trades.empty:
            return '1d'
        
        # 分析交易时间跨度
        time_span = symbol_trades['Time'].max() - symbol_trades['Time'].min()
        total_days = time_span.days
        
        # 分析平均持仓时长
        if 'Duration' in symbol_trades.columns:
            avg_duration_hours = symbol_trades['Duration'].mean()
        else:
            # 如果没有Duration列，用相邻交易的时间差估算
            avg_duration_hours = 24  # 默认1天
        
        print(f"📅 {symbol} 交易时间跨度: {total_days} 天")
        print(f"⏱️  平均持仓时长: {avg_duration_hours:.1f} 小时")
        
        # 自动选择合适的时间分辨率
        if total_days <= 7:  # 一周内
            if avg_duration_hours <= 4:
                timeframe = '15m'
                print("🎯 选择分辨率: 15分钟线 (短期高频交易)")
            elif avg_duration_hours <= 24:
                timeframe = '1h'
                print("🎯 选择分辨率: 小时线 (日内交易)")
            else:
                timeframe = '1d'
                print("🎯 选择分辨率: 日线 (中长期交易)")
        elif total_days <= 30:  # 一个月内
            if avg_duration_hours <= 12:
                timeframe = '1h'
                print("🎯 选择分辨率: 小时线 (短期交易)")
            else:
                timeframe = '1d'
                print("🎯 选择分辨率: 日线 (中期交易)")
        else:  # 超过一个月
            timeframe = '1d'
            print("🎯 选择分辨率: 日线 (长期交易)")
        
        return timeframe
    
    def _find_market_data(self, symbol, timeframe):
        """步骤3: 寻找对应的市场数据"""
        print(f"🔍 寻找 {symbol} 的 {timeframe} 市场数据...")
        
        # 构建可能的文件路径
        symbol_dir = os.path.join(self.market_data_path, symbol)
        if not os.path.exists(symbol_dir):
            # 尝试其他可能的符号格式
            for alt_symbol in [symbol + 'T', symbol.replace('USD', 'USDT')]:
                alt_dir = os.path.join(self.market_data_path, alt_symbol)
                if os.path.exists(alt_dir):
                    symbol_dir = alt_dir
                    print(f"📂 找到替代目录: {alt_symbol}")
                    break
            else:
                print(f"❌ 未找到 {symbol} 的市场数据目录")
                return None
        
        # 查找最匹配的数据文件
        if not os.path.exists(symbol_dir):
            return None
        
        files = os.listdir(symbol_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        
        # 按时间周期优先级排序查找
        timeframe_priority = {
            '15m': ['15m', '1h', '1d'],
            '1h': ['1h', '15m', '1d'], 
            '1d': ['1d', '1h', '15m']
        }
        
        selected_file = None
        for tf in timeframe_priority.get(timeframe, ['1d']):
            for file in csv_files:
                if tf in file:
                    selected_file = file
                    break
            if selected_file:
                break
        
        if not selected_file:
            # 如果没找到匹配的，选择第一个CSV文件
            selected_file = csv_files[0] if csv_files else None
        
        if selected_file:
            file_path = os.path.join(symbol_dir, selected_file)
            print(f"📈 找到数据文件: {selected_file}")
            return file_path
        else:
            print(f"❌ 未找到 {symbol} 的数据文件")
            return None
    
    def _load_market_data(self, file_path):
        """加载市场数据"""
        try:
            df = pd.read_csv(file_path)
            
            # 标准化列名 - 处理不同的时间戳格式
            if 'Open time' in df.columns:
                # 检查是否为毫秒时间戳
                first_val = df['Open time'].iloc[0]
                try:
                    # 尝试作为毫秒时间戳解析
                    if isinstance(first_val, (int, float)) and first_val > 1e12:
                        df['timestamp'] = pd.to_datetime(df['Open time'], unit='ms')
                    else:
                        # 作为日期字符串解析
                        df['timestamp'] = pd.to_datetime(df['Open time'])
                except:
                    # 备用方案：直接解析
                    df['timestamp'] = pd.to_datetime(df['Open time'])
            else:
                # 使用第一列作为时间戳
                first_col = df.columns[0]
                try:
                    first_val = df[first_col].iloc[0]
                    if isinstance(first_val, (int, float)) and first_val > 1e12:
                        df['timestamp'] = pd.to_datetime(df[first_col], unit='ms')
                    else:
                        df['timestamp'] = pd.to_datetime(df[first_col])
                except:
                    df['timestamp'] = pd.to_datetime(df[first_col])
            
            # 重命名OHLCV列 - 适配不同的列名格式
            col_mapping = {}
            for i, col in enumerate(df.columns):
                if i == 0 or 'time' in col.lower():
                    continue
                elif i == 1 or 'open' in col.lower():
                    col_mapping[col] = 'open'
                elif i == 2 or 'high' in col.lower():
                    col_mapping[col] = 'high'
                elif i == 3 or 'low' in col.lower():
                    col_mapping[col] = 'low'
                elif i == 4 or 'close' in col.lower():
                    col_mapping[col] = 'close'
                elif i == 5 or 'volume' in col.lower():
                    col_mapping[col] = 'volume'
            
            df = df.rename(columns=col_mapping)
            
            # 确保有必需的列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️  缺少必需列: {missing_cols}")
                # 如果缺少volume列，用0填充
                if 'volume' in missing_cols:
                    df['volume'] = 0
            
            # 选择需要的列
            available_cols = ['timestamp'] + [col for col in required_cols if col in df.columns]
            df = df[available_cols]
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        print(f"⚠️  转换列 {col} 数据类型失败: {e}")
                        df[col] = 0  # 填充默认值
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # 删除包含NaN的行
            df = df.dropna()
            
            print(f"✅ 成功加载市场数据: {len(df)} 条K线")
            print(f"📅 数据时间范围: {df.index.min()} 到 {df.index.max()}")
            return df
            
        except Exception as e:
            print(f"❌ 加载市场数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _plot_kline_with_trades(self, symbol, market_data, trades, timeframe):
        """步骤4: 画出K线图和买卖点"""
        print(f"🎨 正在绘制 {symbol} 的K线图和交易点...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # 过滤市场数据到交易时间范围
        if not trades.empty:
            start_time = trades['Time'].min() - timedelta(days=1)
            end_time = trades['Time'].max() + timedelta(days=1)
            market_data = market_data[(market_data.index >= start_time) & 
                                    (market_data.index <= end_time)]
        
        # 绘制价格线（简化的K线图）
        ax1.plot(market_data.index, market_data['close'], 
                label=f'{symbol} 收盘价', linewidth=1, color='#1f77b4', alpha=0.8)
        
        # 添加高低价阴影
        ax1.fill_between(market_data.index, market_data['low'], market_data['high'], 
                        alpha=0.1, color='gray', label='高低价区间')
        
        # 标注买卖点
        for _, trade in trades.iterrows():
            entry_time = trade['Time']
            entry_price = trade['OpenPrice']
            exit_time = trade.get('CloseTime')
            exit_price = trade.get('ClosePrice')
            pnl = trade['Value']
            
            # 买入点（绿色向上三角形）
            ax1.scatter(entry_time, entry_price, color='green', s=120, marker='^', 
                       label='买入' if trades.index[0] == trade.name else "", 
                       zorder=5, edgecolors='darkgreen', linewidth=1)
            
            # 如果有卖出信息
            if pd.notna(exit_time) and pd.notna(exit_price):
                # 卖出点（红色向下三角形）
                ax1.scatter(exit_time, exit_price, color='red', s=120, marker='v', 
                           label='卖出' if trades.index[0] == trade.name else "", 
                           zorder=5, edgecolors='darkred', linewidth=1)
                
                # 连接线
                line_color = 'green' if pnl > 0 else 'red'
                line_alpha = 0.7 if pnl > 0 else 0.5
                ax1.plot([entry_time, exit_time], [entry_price, exit_price], 
                        color=line_color, alpha=line_alpha, linewidth=2, linestyle='-')
                
                # 收益标注
                mid_time = entry_time + (exit_time - entry_time) / 2
                mid_price = (entry_price + exit_price) / 2
                
                # 计算收益率
                return_rate = (pnl / abs(trade.get('AbsValue', 1))) * 100
                pnl_text = f'{pnl:+.0f}\n({return_rate:+.1f}%)'
                
                bbox_color = 'lightgreen' if pnl > 0 else 'lightcoral'
                ax1.annotate(pnl_text, (mid_time, mid_price), 
                           textcoords="offset points", xytext=(0,15), ha='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.8),
                           fontsize=9, fontweight='bold')
        
        # 设置主图
        ax1.set_title(f'{symbol} 出入场分析 - {timeframe} 时间周期', fontsize=16, fontweight='bold')
        ax1.set_ylabel('价格 (USD)', fontsize=12)
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # 设置时间轴
        if timeframe == '1d':
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
        elif timeframe == '1h':
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        else:  # 15m
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 绘制累计收益曲线
        if not trades.empty:
            trades_sorted = trades.sort_values('Time')
            cumulative_pnl = trades_sorted['Value'].cumsum()
            
            ax2.plot(trades_sorted['Time'], cumulative_pnl, 
                    color='purple', linewidth=2.5, marker='o', markersize=6)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 填充盈亏区域
            ax2.fill_between(trades_sorted['Time'], cumulative_pnl, 0, 
                           where=(cumulative_pnl >= 0), color='green', alpha=0.3, label='盈利区域')
            ax2.fill_between(trades_sorted['Time'], cumulative_pnl, 0, 
                           where=(cumulative_pnl < 0), color='red', alpha=0.3, label='亏损区域')
            
            ax2.set_ylabel('累计收益', fontsize=12)
            ax2.set_xlabel('时间', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')
            
            # 同步时间轴格式
            if timeframe == '1d':
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax2.xaxis.set_major_locator(mdates.MonthLocator())
            elif timeframe == '1h':
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            else:
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        filename = f"{symbol}_{timeframe}_entry_exit_analysis.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"💾 图表已保存: {filename}")
        
        plt.show()
        return fig
    
    def analyze_symbol(self, symbol):
        """分析指定交易对的完整流程"""
        print(f"\n{'='*60}")
        print(f"🎯 开始分析 {symbol}")
        print(f"{'='*60}")
        
        # 筛选该交易对的数据
        symbol_trades = self.trades_data[self.trades_data['Symbol'] == symbol]
        
        if symbol_trades.empty:
            print(f"❌ 没有找到 {symbol} 的交易记录")
            return
        
        print(f"📊 找到 {len(symbol_trades)} 个交易记录")
        
        # 步骤2: 自动匹配分辨率
        timeframe = self._determine_optimal_timeframe(symbol)
        
        # 步骤3: 寻找市场数据
        market_file = self._find_market_data(symbol, timeframe)
        if not market_file:
            print(f"❌ 无法找到 {symbol} 的市场数据，跳过分析")
            return
        
        # 加载市场数据
        market_data = self._load_market_data(market_file)
        if market_data is None:
            print(f"❌ 无法加载 {symbol} 的市场数据")
            return
        
        # 打印交易摘要
        self._print_trade_summary(symbol_trades)
        
        # 步骤4: 绘制图表
        self._plot_kline_with_trades(symbol, market_data, symbol_trades, timeframe)
        
        print(f"✅ {symbol} 分析完成!")
    
    def _print_trade_summary(self, trades):
        """打印交易摘要"""
        total_trades = len(trades)
        total_pnl = trades['Value'].sum()
        avg_pnl = trades['Value'].mean()
        win_rate = (trades['Value'] > 0).mean() * 100
        
        winning_trades = trades[trades['Value'] > 0]
        losing_trades = trades[trades['Value'] < 0]
        
        print(f"\n📊 交易摘要:")
        print(f"  总交易次数: {total_trades}")
        print(f"  总收益: {total_pnl:+,.2f}")
        print(f"  平均收益: {avg_pnl:+,.2f}")
        print(f"  胜率: {win_rate:.1f}%")
        print(f"  盈利交易: {len(winning_trades)} 次")
        print(f"  亏损交易: {len(losing_trades)} 次")
        
        if len(winning_trades) > 0:
            print(f"  平均盈利: {winning_trades['Value'].mean():+,.2f}")
        if len(losing_trades) > 0:
            print(f"  平均亏损: {losing_trades['Value'].mean():+,.2f}")
    
    def analyze_all_symbols(self):
        """分析所有交易对"""
        print(f"🚀 开始分析所有交易对: {', '.join(self.symbols)}")
        
        for symbol in self.symbols:
            try:
                self.analyze_symbol(symbol)
            except Exception as e:
                print(f"❌ 分析 {symbol} 时出错: {e}")
                continue
        
        print(f"\n🎉 所有交易对分析完成!")


def main():
    """主函数"""
    print("🚀 简化版出入场分析器")
    print("="*50)
    
    # 步骤1: 用户输入订单数据位置
    csv_path = input("请输入订单数据CSV文件路径: ").strip()
    
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return
    
    try:
        # 创建分析器
        analyzer = SimpleEntryExitAnalyzer(csv_path)
        
        if not analyzer.symbols:
            print("❌ 未找到可交易的符号")
            return
        
        # 让用户选择分析方式
        print(f"\n🎯 可分析的交易对: {', '.join(analyzer.symbols)}")
        print("\n选择分析方式:")
        print("1. 分析所有交易对")
        print("2. 选择特定交易对")
        
        choice = input("请选择 (1 或 2): ").strip()
        
        if choice == '1':
            analyzer.analyze_all_symbols()
        elif choice == '2':
            print("\n可选交易对:")
            for i, symbol in enumerate(analyzer.symbols, 1):
                print(f"  {i}. {symbol}")
            
            try:
                symbol_choice = int(input("请选择交易对序号: ")) - 1
                if 0 <= symbol_choice < len(analyzer.symbols):
                    selected_symbol = analyzer.symbols[symbol_choice]
                    analyzer.analyze_symbol(selected_symbol)
                else:
                    print("❌ 无效选择")
            except ValueError:
                print("❌ 无效输入")
        else:
            print("❌ 无效选择")
    
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()