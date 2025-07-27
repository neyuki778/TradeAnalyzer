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
    
    def _extract_exit_reason(self, close_tag):
        """直接使用出场时的tag作为出场原因"""
        if not close_tag or close_tag == 'nan' or str(close_tag).strip() == '':
            return None
        
        close_tag = str(close_tag).strip()
        
        # 如果是纯数字（可能是其他信息），不作为出场原因
        if close_tag.replace(',', '').replace('.', '').replace('"', '').replace(' ', '').isdigit():
            return None
        
        # 直接返回出场标签，限制长度避免过长
        return close_tag[:15]
    
    def _determine_optimal_timeframe(self, symbol):
        """步骤2: 自动匹配最佳分辨率（仅基于订单来源）"""
        symbol_trades = self.trades_data[self.trades_data['Symbol'] == symbol]
        if symbol_trades.empty:
            return '1d'
        
        # 检查数据来源文件名，判断原始策略周期
        source_hint = None
        if hasattr(self, 'csv_path'):
            csv_path_upper = self.csv_path.upper()
            if 'HOUR' in csv_path_upper or '1H' in csv_path_upper:
                source_hint = '1h'
            elif 'DAILY' in csv_path_upper or '1D' in csv_path_upper:
                source_hint = '1d' 
            elif '15M' in csv_path_upper or 'MIN' in csv_path_upper:
                source_hint = '15m'
        
        # 显示基本信息
        time_span = symbol_trades['Time'].max() - symbol_trades['Time'].min()
        total_days = time_span.days
        total_trades = len(symbol_trades)
        
        print(f"📅 {symbol} 交易时间跨度: {total_days} 天")
        print(f"📊 交易总数: {total_trades} 笔")
        print(f"📁 订单数据源: {self.csv_path}")
        
        # 完全基于数据来源提示选择分辨率
        if source_hint == '15m':
            timeframe = '15m'
            print("🎯 选择分辨率: 15分钟线 (基于订单来源)")
        elif source_hint == '1h':
            timeframe = '1h'
            print("🎯 选择分辨率: 小时线 (基于订单来源)")
        elif source_hint == '1d':
            timeframe = '1d'
            print("🎯 选择分辨率: 日线 (基于订单来源)")
        else:
            # 如果无法从文件名识别，默认使用日线
            timeframe = '1d'
            print("🎯 选择分辨率: 日线 (默认，无法识别订单来源周期)")
        
        return timeframe
    
    def _find_market_data(self, symbol, timeframe):
        """步骤3: 寻找对应的市场数据 - 支持现货和期货"""
        print(f"🔍 寻找 {symbol} 的 {timeframe} 市场数据...")
        
        # 清理符号名称：移除多余的USD后缀（如SPKUSDTUSD -> SPKUSDT）
        clean_symbol = self._clean_symbol_name(symbol)
        print(f"🧹 清理后的符号: {clean_symbol}")
        
        # 先尝试直接匹配
        symbol_dir = self._try_find_symbol_directory(clean_symbol)
        
        if not symbol_dir:
            # 如果直接匹配失败，尝试符号变换
            alt_symbols = self._generate_symbol_alternatives(clean_symbol)
            for alt_symbol in alt_symbols:
                symbol_dir = self._try_find_symbol_directory(alt_symbol)
                if symbol_dir:
                    print(f"📂 找到替代符号: {alt_symbol}")
                    break
        
        if not symbol_dir:
            print(f"❌ 未找到 {symbol} 的市场数据目录")
            return None
        
        # 查找最匹配的数据文件
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
    
    def _clean_symbol_name(self, symbol):
        """清理符号名称，处理重复后缀等问题"""
        # 移除重复的USD后缀（如SPKUSDTUSD -> SPKUSDT）
        if symbol.endswith('USDUSD'):
            return symbol[:-3]  # 移除最后的USD
        elif symbol.endswith('USDTUSD'):
            return symbol[:-3]  # 移除最后的USD
        return symbol
    
    def _try_find_symbol_directory(self, symbol):
        """尝试在SPOT和FUTURES目录中查找符号"""
        # 检查SPOT目录
        spot_dir = os.path.join(self.market_data_path, 'SPOT', symbol)
        if os.path.exists(spot_dir):
            print(f"📂 在SPOT目录找到: {symbol}")
            return spot_dir
        
        # 检查FUTURES目录
        futures_dir = os.path.join(self.market_data_path, 'FUTURES', symbol)
        if os.path.exists(futures_dir):
            print(f"📂 在FUTURES目录找到: {symbol}")
            return futures_dir
        
        # 检查根目录（兼容旧格式）
        root_dir = os.path.join(self.market_data_path, symbol)
        if os.path.exists(root_dir):
            print(f"📂 在根目录找到: {symbol}")
            return root_dir
        
        return None
    
    def _generate_symbol_alternatives(self, symbol):
        """生成可能的符号变体"""
        alternatives = []
        
        # 原始符号
        alternatives.append(symbol)
        
        # 添加USDT后缀（如果没有）
        if not symbol.endswith('USDT') and not symbol.endswith('USD'):
            alternatives.append(symbol + 'USDT')
            alternatives.append(symbol + 'USD')
        
        # 替换USD为USDT或反之
        if symbol.endswith('USD') and not symbol.endswith('USDT'):
            alternatives.append(symbol + 'T')  # USD -> USDT
        elif symbol.endswith('USDT'):
            alternatives.append(symbol[:-1])   # USDT -> USD
        
        # 移除常见后缀再重新组合
        base_symbol = symbol
        for suffix in ['USDT', 'USD', 'BTC', 'ETH']:
            if symbol.endswith(suffix):
                base_symbol = symbol[:-len(suffix)]
                break
        
        # 重新组合常见后缀
        for suffix in ['USDT', 'USD']:
            if base_symbol + suffix not in alternatives:
                alternatives.append(base_symbol + suffix)
        
        print(f"🔄 生成符号变体: {alternatives}")
        return alternatives
    
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
                        # 作为日期字符串解析，使用混合格式自动推断
                        df['timestamp'] = pd.to_datetime(df['Open time'], format='mixed', dayfirst=False)
                except:
                    # 备用方案：ISO8601格式
                    try:
                        df['timestamp'] = pd.to_datetime(df['Open time'], format='ISO8601')
                    except:
                        # 最后备用方案：自动推断
                        df['timestamp'] = pd.to_datetime(df['Open time'], infer_datetime_format=True)
            else:
                # 使用第一列作为时间戳
                first_col = df.columns[0]
                try:
                    first_val = df[first_col].iloc[0]
                    if isinstance(first_val, (int, float)) and first_val > 1e12:
                        df['timestamp'] = pd.to_datetime(df[first_col], unit='ms')
                    else:
                        df['timestamp'] = pd.to_datetime(df[first_col], format='mixed', dayfirst=False)
                except:
                    try:
                        df['timestamp'] = pd.to_datetime(df[first_col], format='ISO8601')
                    except:
                        df['timestamp'] = pd.to_datetime(df[first_col], infer_datetime_format=True)
            
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
    
    def _plot_kline_with_trades(self, symbol, market_data, trades, timeframe, start_date=None, end_date=None):
        """步骤4: 画出K线图和买卖点"""
        print(f"🎨 正在绘制 {symbol} 的K线图和交易点...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # 根据用户指定的时间段或交易时间范围过滤市场数据
        if start_date or end_date:
            if start_date:
                start_dt = pd.to_datetime(start_date)
                market_data = market_data[market_data.index >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)  
                market_data = market_data[market_data.index <= end_dt]
        else:
            # 如果没有指定时间段，基于交易数据范围并扩展一些边界
            if not trades.empty:
                trade_start = trades['Time'].min() - timedelta(days=3)
                trade_end = trades['Time'].max() + timedelta(days=3)
                market_data = market_data[(market_data.index >= trade_start) & 
                                        (market_data.index <= trade_end)]
        
        if market_data.empty:
            print("⚠️  指定时间段内没有市场数据")
            return None
        
        # 绘制K线图（蜡烛图）
        from matplotlib.patches import Rectangle
        
        # 计算K线宽度（根据时间周期动态调整）
        if timeframe == '1d':
            width = 0.6  # 日线K线宽度
        elif timeframe == '1h':
            width = 0.02  # 小时线K线宽度
        else:  # 15m
            width = 0.008  # 15分钟线K线宽度
        
        # 绘制每根K线
        for idx, row in market_data.iterrows():
            open_price = row['open']
            close_price = row['close'] 
            high_price = row['high']
            low_price = row['low']
            
            # 判断涨跌
            color = 'green' if close_price >= open_price else 'red'  # 绿涨红跌
            
            # 绘制上下影线
            ax1.plot([idx, idx], [low_price, high_price], color='black', linewidth=0.8, alpha=0.7)
            
            # 绘制实体K线
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            
            # 使用Rectangle绘制K线实体
            x_pos = mdates.date2num(idx)
            rect = Rectangle((x_pos - width/2, bottom), width, height,
                           facecolor=color, edgecolor='black', alpha=0.8, linewidth=0.5)
            ax1.add_patch(rect)
        
        # 标注买卖点
        for _, trade in trades.iterrows():
            entry_time = trade['Time']
            entry_price = trade['OpenPrice']
            exit_time = trade.get('CloseTime')
            exit_price = trade.get('ClosePrice')
            pnl = trade['Value']
            trade_type = trade.get('Type', 'Long')  # 获取交易类型
            
            # 获取出场原因
            close_tag = str(trade.get('CloseTag', ''))
            exit_reason = self._extract_exit_reason(close_tag)
            
            # 根据交易类型设置不同的标记样式
            if trade_type == 'Long':
                # 多头：绿色向上三角形为开仓，红色向下三角形为平仓
                entry_color = 'green'
                entry_marker = '^'
                entry_edge_color = 'darkgreen'
                entry_label = '多头开仓'
                
                exit_color = 'red'
                exit_marker = 'v'
                exit_edge_color = 'darkred'
                exit_label = '多头平仓'
            else:  # Short
                # 空头：红色向下三角形为开仓，绿色向上三角形为平仓
                entry_color = 'red'
                entry_marker = 'v'
                entry_edge_color = 'darkred'
                entry_label = '空头开仓'
                
                exit_color = 'green'
                exit_marker = '^'
                exit_edge_color = 'darkgreen'
                exit_label = '空头平仓'
            
            # 开仓点标记
            ax1.scatter(entry_time, entry_price, color=entry_color, s=120, marker=entry_marker, 
                       label=entry_label if trades.index[0] == trade.name else "", 
                       zorder=5, edgecolors=entry_edge_color, linewidth=1)
            
            # 如果有平仓信息
            if pd.notna(exit_time) and pd.notna(exit_price):
                # 平仓点标记
                ax1.scatter(exit_time, exit_price, color=exit_color, s=120, marker=exit_marker, 
                           label=exit_label if trades.index[0] == trade.name else "", 
                           zorder=5, edgecolors=exit_edge_color, linewidth=1)
                
                # 连接线
                line_color = 'green' if pnl > 0 else 'red'
                line_alpha = 0.7 if pnl > 0 else 0.5
                ax1.plot([entry_time, exit_time], [entry_price, exit_price], 
                        color=line_color, alpha=line_alpha, linewidth=2, linestyle='-')
                
                # 交易信息标注
                mid_time = entry_time + (exit_time - entry_time) / 2
                mid_price = (entry_price + exit_price) / 2
                
                # 获取仓位大小信息和比例
                position_size = abs(trade.get('AbsValue', 0))
                tag_str = str(trade.get('Tag', '0'))
                
                # 解析Tag中的数值（去除逗号）
                try:
                    tag_value = float(tag_str.replace(',', '').replace('"', ''))
                    position_ratio = (position_size / tag_value) * 100 if tag_value > 0 else 0
                except:
                    position_ratio = 0
                
                # 计算持仓时间
                if pd.notna(exit_time) and pd.notna(entry_time):
                    holding_duration = exit_time - entry_time
                    
                    # 格式化持仓时间
                    total_hours = holding_duration.total_seconds() / 3600
                    if total_hours < 1:
                        duration_str = f"{int(holding_duration.total_seconds() / 60)}m"
                    elif total_hours < 24:
                        duration_str = f"{total_hours:.1f}h"
                    else:
                        days = int(total_hours / 24)
                        remaining_hours = total_hours % 24
                        if remaining_hours < 1:
                            duration_str = f"{days}d"
                        else:
                            duration_str = f"{days}d{remaining_hours:.0f}h"
                else:
                    duration_str = "未知"
                
                # 构建标签文本：入场价格、出场价格、出场原因、仓位百分比、持仓时间  
                # 智能价格格式化：根据价格大小选择合适的小数位数
                def format_price(price):
                    price = float(price)
                    if price >= 100:
                        return f"{price:.0f}"
                    elif price >= 10:
                        return f"{price:.1f}"
                    elif price >= 1:
                        return f"{price:.2f}"
                    elif price >= 0.1:
                        return f"{price:.3f}"
                    else:
                        return f"{price:.4f}"
                
                # 标签内容根据交易类型调整
                direction_label = "多" if trade_type == 'Long' else "空"
                label_lines = [
                    f'{direction_label}入: {format_price(entry_price)}',
                    f'{direction_label}出: {format_price(exit_price)}'
                ]
                
                # 如果有出场原因，添加到标签中
                if exit_reason:
                    label_lines.append(f'{exit_reason}')
                
                label_lines.extend([
                    f'仓: {position_ratio:.2f}%',
                    f'时: {duration_str}'
                ])
                
                label_text = '\n'.join(label_lines)
                
                bbox_color = 'lightgreen' if pnl > 0 else 'lightcoral'
                ax1.annotate(label_text, (mid_time, mid_price), 
                           textcoords="offset points", xytext=(0,15), ha='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=bbox_color, alpha=0.8),
                           fontsize=8, fontweight='bold')
                
                # 在出场点单独标注出场原因（如果存在）
                if exit_reason:
                    ax1.annotate(exit_reason, (exit_time, exit_price), 
                               textcoords="offset points", xytext=(10, -20), ha='left',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.9),
                               fontsize=7, fontweight='bold', color='red')
        
        # 设置主图标题
        time_range_str = ""
        if start_date or end_date:
            time_range_str = f" ({start_date or '开始'} 到 {end_date or '结束'})"
        ax1.set_title(f'{symbol} 出入场分析{time_range_str} - {timeframe} 时间周期', 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('价格 (USD)', fontsize=12)
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # 设置时间轴
        if timeframe == '1d':
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
        elif timeframe == '1h':
            if len(market_data) > 168:  # 超过一周的数据
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            else:
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
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
                if len(market_data) > 168:
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                else:
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            else:
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        time_suffix = ""
        if start_date or end_date:
            time_suffix = f"_{start_date or 'start'}_to_{end_date or 'end'}"
        filename = f"{symbol}_{timeframe}_entry_exit_analysis{time_suffix}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"💾 图表已保存: {filename}")
        
        plt.show()
        return fig
    
    def analyze_symbol(self, symbol, start_date=None, end_date=None):
        """分析指定交易对的完整流程"""
        print(f"\n{'='*60}")
        print(f"🎯 开始分析 {symbol}")
        if start_date or end_date:
            print(f"📅 分析时间段: {start_date or '开始'} 到 {end_date or '结束'}")
        print(f"{'='*60}")
        
        # 筛选该交易对的数据
        symbol_trades = self.trades_data[self.trades_data['Symbol'] == symbol].copy()
        
        if symbol_trades.empty:
            print(f"❌ 没有找到 {symbol} 的交易记录")
            return
        
        # 按时间段筛选
        if start_date:
            start_dt = pd.to_datetime(start_date)
            symbol_trades = symbol_trades[symbol_trades['Time'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            symbol_trades = symbol_trades[symbol_trades['Time'] <= end_dt]
        
        if symbol_trades.empty:
            print(f"❌ 指定时间段内没有 {symbol} 的交易记录")
            return
        
        print(f"📊 找到 {len(symbol_trades)} 个交易记录")
        
        # 步骤2: 自动匹配分辨率（基于筛选后的数据）
        timeframe = self._determine_optimal_timeframe_for_period(symbol_trades, start_date, end_date)
        
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
        self._plot_kline_with_trades(symbol, market_data, symbol_trades, timeframe, start_date, end_date)
        
        print(f"✅ {symbol} 分析完成!")
    
    def _determine_optimal_timeframe_for_period(self, symbol_trades, start_date=None, end_date=None):
        """为指定时间段确定最佳时间框架（仅基于订单来源）"""
        symbol = symbol_trades['Symbol'].iloc[0]
        
        # 计算实际分析的时间跨度
        if start_date and end_date:
            time_span = pd.to_datetime(end_date) - pd.to_datetime(start_date)
            analysis_days = time_span.days
        else:
            time_span = symbol_trades['Time'].max() - symbol_trades['Time'].min()
            analysis_days = time_span.days
        
        total_trades = len(symbol_trades)
        
        # 检查数据来源文件名，判断原始策略周期
        source_hint = None
        if hasattr(self, 'csv_path'):
            csv_path_upper = self.csv_path.upper()
            if 'HOUR' in csv_path_upper or '1H' in csv_path_upper:
                source_hint = '1h'
            elif 'DAILY' in csv_path_upper or '1D' in csv_path_upper:
                source_hint = '1d' 
            elif '15M' in csv_path_upper or 'MIN' in csv_path_upper:
                source_hint = '15m'
        
        print(f"📅 {symbol} 分析时间跨度: {analysis_days} 天")
        print(f"📊 交易总数: {total_trades} 笔")
        print(f"📁 订单数据源: {self.csv_path}")
        
        # 完全基于数据来源提示选择分辨率
        if source_hint == '15m':
            timeframe = '15m'
            print("🎯 选择分辨率: 15分钟线 (基于订单来源)")
        elif source_hint == '1h':
            timeframe = '1h' 
            print("🎯 选择分辨率: 小时线 (基于订单来源)")
        elif source_hint == '1d':
            timeframe = '1d'
            print("🎯 选择分辨率: 日线 (基于订单来源)")
        else:
            # 如果无法从文件名识别，默认使用日线
            timeframe = '1d'
            print("🎯 选择分辨率: 日线 (默认，无法识别订单来源周期)")
        
        return timeframe
    
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
    
    def analyze_all_symbols(self, start_date=None, end_date=None):
        """分析所有交易对"""
        time_range_str = ""
        if start_date or end_date:
            time_range_str = f" ({start_date or '开始'} 到 {end_date or '结束'})"
        
        print(f"🚀 开始分析所有交易对{time_range_str}: {', '.join(self.symbols)}")
        
        for symbol in self.symbols:
            try:
                self.analyze_symbol(symbol, start_date, end_date)
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
        
        # 询问是否指定时间段
        print(f"\n🎯 可分析的交易对: {', '.join(analyzer.symbols)}")
        print("\n是否指定分析时间段? (可提高图表清晰度)")
        use_time_range = input("输入 y 指定时间段，回车跳过: ").strip().lower()
        
        start_date = None
        end_date = None
        
        if use_time_range in ['y', 'yes', '是']:
            start_date = input("请输入开始日期 (YYYY-MM-DD, 回车跳过): ").strip()
            end_date = input("请输入结束日期 (YYYY-MM-DD, 回车跳过): ").strip()
            
            start_date = start_date if start_date else None
            end_date = end_date if end_date else None
            
            if start_date or end_date:
                print(f"📅 将分析时间段: {start_date or '开始'} 到 {end_date or '结束'}")
        
        # 让用户选择分析方式
        print("\n选择分析方式:")
        print("1. 分析所有交易对")
        print("2. 选择特定交易对")
        
        choice = input("请选择 (1 或 2): ").strip()
        
        if choice == '1':
            analyzer.analyze_all_symbols(start_date, end_date)
        elif choice == '2':
            print("\n可选交易对:")
            for i, symbol in enumerate(analyzer.symbols, 1):
                print(f"  {i}. {symbol}")
            
            try:
                symbol_choice = int(input("请选择交易对序号: ")) - 1
                if 0 <= symbol_choice < len(analyzer.symbols):
                    selected_symbol = analyzer.symbols[symbol_choice]
                    analyzer.analyze_symbol(selected_symbol, start_date, end_date)
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