"""
订单分析器模块
专门负责订单数据的预处理、清洗和分析计算
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class OrderAnalyzer:
    """订单分析器 - 专门处理订单数据分析和计算"""
    
    def __init__(self, csv_file_path):
        """
        初始化订单分析器
        
        Args:
            csv_file_path (str): 订单数据CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.raw_data = None
        self.processed_data = None
        self.analysis_results = {}
        self.load_data()
        
    def load_data(self):
        """加载订单数据"""
        try:
            self.raw_data = pd.read_csv(self.csv_file_path)
            print(f"成功加载订单数据: {len(self.raw_data)} 条记录")
            print(f"数据时间范围: {self.raw_data['Time'].min()} 到 {self.raw_data['Time'].max()}")
            self._preprocess_data()
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise
            
    def _preprocess_data(self):
        """预处理数据 - 支持多头和空头策略的开平仓配对分析"""
        if self.raw_data is None:
            raise ValueError("原始数据未加载")
            
        # 转换时间格式
        self.raw_data['Time'] = pd.to_datetime(self.raw_data['Time'])
        
        # 1. 首先过滤有效订单，排除Invalid状态
        filled_data = self.raw_data[self.raw_data['Status'] == 'Filled'].copy()
        print(f"过滤无效订单后: {len(filled_data)} 条有效订单")
        
        # 2. 智能判断开平仓 - 支持多头和空头
        filled_data['OrderSide'] = filled_data.apply(self._determine_order_side, axis=1)
        filled_data['AbsQuantity'] = abs(filled_data['Quantity'])
        filled_data['AbsValue'] = abs(filled_data['Value'])
        
        # 3. 配对开平仓订单计算真实交易
        self.processed_data = self._pair_open_close_orders(filled_data)
        
        # 4. 添加仓位大小分类
        if len(self.processed_data) > 0:
            self.processed_data['PositionSize'] = self._categorize_position_size()
            
            # 5. 添加收益计算
            self._calculate_returns()
            
            print(f"配对分析完成: {len(self.processed_data)} 个完整交易")
        else:
            print("警告: 没有找到完整的开平仓配对")
    
    def _determine_order_side(self, row):
        """智能判断订单类型：开仓还是平仓"""
        quantity = row['Quantity']
        tag = str(row['Tag']) if pd.notna(row['Tag']) else ''
        
        # 判断tag是否为数字（开仓）还是文字（平仓）
        tag_cleaned = tag.replace(',', '').replace('"', '').replace(' ', '')
        is_numeric_tag = tag_cleaned.replace('.', '').isdigit() if tag_cleaned else False
        
        if is_numeric_tag:
            # Tag是数字，表示开仓
            return 'Open'
        else:
            # Tag是文字，表示平仓
            return 'Close'
            
    def _pair_open_close_orders(self, filled_data):
        """配对开平仓订单，计算完整交易的真实收益 - 支持多头和空头"""
        trades = []
        
        print("开始配对开平仓订单...")
        
        # 按交易对分组处理
        for symbol in filled_data['Symbol'].unique():
            symbol_data = filled_data[filled_data['Symbol'] == symbol].sort_values('Time').reset_index(drop=True)
            
            # 分离开仓和平仓订单（基于Tag判断，而非数量符号）
            open_orders = symbol_data[symbol_data['OrderSide'] == 'Open'].copy()
            close_orders = symbol_data[symbol_data['OrderSide'] == 'Close'].copy()
            
            # 进一步按多空方向分类开仓订单
            long_opens = open_orders[open_orders['Quantity'] > 0].copy()  # 多头开仓
            short_opens = open_orders[open_orders['Quantity'] < 0].copy()  # 空头开仓
            
            print(f"处理 {symbol}: {len(long_opens)} 多头开仓, {len(short_opens)} 空头开仓, {len(close_orders)} 平仓")
            
            # 处理多头交易配对
            self._pair_trades_by_direction(long_opens, close_orders, trades, 'Long')
            
            # 处理空头交易配对
            self._pair_trades_by_direction(short_opens, close_orders, trades, 'Short')
            
        if not trades:
            print("⚠️  未找到匹配的开平仓配对，可能数据格式不符合预期")
            # 如果配对失败，返回原始数据（去除无效订单）
            fallback_data = filled_data.copy()
            fallback_data['AbsValue'] = abs(fallback_data['Value'])
            fallback_data = fallback_data[fallback_data['Value'] != 0]  # 排除Value为0的订单
            return fallback_data
            
        trades_df = pd.DataFrame(trades)
        print(f"✅ 成功配对 {len(trades_df)} 个完整交易")
        
        if len(trades_df) > 0:
            print(f"📈 平均持仓时长: {trades_df['Duration'].mean():.1f} 小时")
            print(f"💰 平均交易收益: {trades_df['Value'].mean():.2f}")
            
            # 统计多空比例
            long_count = len(trades_df[trades_df['Type'] == 'Long'])
            short_count = len(trades_df[trades_df['Type'] == 'Short'])
            print(f"📊 多头交易: {long_count} 笔, 空头交易: {short_count} 笔")
        
        return trades_df
    
    def _pair_trades_by_direction(self, open_orders, close_orders, trades, trade_type):
        """按方向配对交易（多头或空头）"""
        used_close_indices = set()
        
        for _, open_order in open_orders.iterrows():
            # 找到这个开仓之后且尚未被使用的平仓订单
            valid_closes = close_orders[
                (close_orders['Time'] > open_order['Time']) & 
                (~close_orders.index.isin(used_close_indices))
            ]
            
            if not valid_closes.empty:
                # 选择最近的平仓订单
                close_order = valid_closes.iloc[0]
                
                # 计算真实的交易收益
                if trade_type == 'Long':
                    # 多头：买入成本 vs 卖出收入
                    trade_pnl = abs(close_order['Value']) - abs(open_order['Value'])
                else:  # Short
                    # 空头：卖出收入 vs 买入成本（收益计算相反）
                    trade_pnl = abs(open_order['Value']) - abs(close_order['Value'])
                
                # 创建完整交易记录
                trade = {
                    'Time': open_order['Time'],
                    'CloseTime': close_order['Time'],
                    'Symbol': open_order['Symbol'],
                    'OpenPrice': open_order['Price'],
                    'ClosePrice': close_order['Price'],
                    'Quantity': open_order['AbsQuantity'],
                    'OpenValue': abs(open_order['Value']),
                    'CloseValue': abs(close_order['Value']),
                    'Value': trade_pnl,  # 真实交易净收益
                    'AbsValue': abs(open_order['Value']),  # 仓位大小
                    'Duration': (close_order['Time'] - open_order['Time']).total_seconds() / 3600,
                    'Type': trade_type,  # 'Long' 或 'Short'
                    'Status': 'Completed',
                    'Tag': open_order['Tag'] if pd.notna(open_order['Tag']) else '',
                    'OpenTag': open_order['Tag'] if pd.notna(open_order['Tag']) else '',
                    'CloseTag': close_order['Tag'] if pd.notna(close_order['Tag']) else ''
                }
                trades.append(trade)
                
                # 标记已使用的平仓订单
                used_close_indices.add(close_order.name)
        
    def _categorize_position_size(self):
        """按仓位大小分类 - 智能分布分析"""
        if self.processed_data is None or len(self.processed_data) == 0:
            return pd.Series([])
            
        values = self.processed_data['AbsValue']
        
        # 分析数据分布
        percentiles = values.quantile([0.5, 0.8, 0.9, 0.95, 0.99]).round(2)
        
        print(f"\n📊 仓位分布分析:")
        print(f"50%分位数: {percentiles[0.5]:,.2f}")
        print(f"80%分位数: {percentiles[0.8]:,.2f}")
        print(f"90%分位数: {percentiles[0.9]:,.2f}")
        print(f"95%分位数: {percentiles[0.95]:,.2f}")
        print(f"99%分位数: {percentiles[0.99]:,.2f}")
        print(f"最大值: {values.max():,.2f}")
        
        # 检查数据分布特征
        p95_ratio = (values <= percentiles[0.95]).sum() / len(values)
        print(f"95%分位数以下占比: {p95_ratio:.1%}")
        
        # 根据分布特征选择分类策略
        if p95_ratio >= 0.9:  # 如果95%以上数据都在95分位数以下
            print("📈 检测到长尾分布，使用精细化分类")
            return self._fine_grained_categorization(values, percentiles)
        else:
            print("📊 使用标准三分类")
            return self._standard_categorization(values)
    
    def _fine_grained_categorization(self, values, percentiles):
        """精细化分类 - 适用于长尾分布"""
        def categorize(value):
            if value <= percentiles[0.8]:
                return 'Small'
            elif value <= percentiles[0.95]:
                return 'Medium'
            elif value <= percentiles[0.99]:
                return 'Large'
            else:
                return 'XLarge'
                
        return values.apply(categorize)
    
    def _standard_categorization(self, values):
        """标准三分类"""
        value_quantiles = values.quantile([0.33, 0.67])
        
        def categorize(value):
            if value <= value_quantiles[0.33]:
                return 'Small'
            elif value <= value_quantiles[0.67]:
                return 'Medium'
            else:
                return 'Large'
                
        return values.apply(categorize)
    
    def _calculate_returns(self):
        """计算收益相关指标 - 基于完整交易"""
        if self.processed_data is None or len(self.processed_data) == 0:
            return
            
        # 按符号分组计算累计收益
        self.processed_data['CumulativeValue'] = self.processed_data.groupby('Symbol')['Value'].cumsum()
        
        # 计算当日收益
        self.processed_data['Date'] = self.processed_data['Time'].dt.date
        daily_returns = self.processed_data.groupby(['Symbol', 'Date'])['Value'].sum().reset_index()
        daily_returns['DailyReturn'] = daily_returns['Value']
        
        # 合并回原数据
        self.processed_data = self.processed_data.merge(
            daily_returns[['Symbol', 'Date', 'DailyReturn']], 
            on=['Symbol', 'Date'], 
            how='left'
        )
        
        # 收益类型分类
        self.processed_data['ReturnType'] = np.where(
            self.processed_data['Value'] > 0, 'Profit', 'Loss'
        )
        
        # 添加收益率计算
        self.processed_data['ReturnRate'] = (self.processed_data['Value'] / self.processed_data['AbsValue']) * 100

    def generate_basic_statistics(self):
        """生成基本统计数据"""
        if self.processed_data is None or len(self.processed_data) == 0:
            return {}
            
        data = self.processed_data
        
        stats = {
            'total_trades': len(data),
            'total_pnl': data['Value'].sum(),
            'avg_pnl': data['Value'].mean(),
            'win_rate': (data['Value'] > 0).mean() * 100,
            'profit_trades': len(data[data['Value'] > 0]),
            'loss_trades': len(data[data['Value'] < 0]),
            'total_profit': data[data['Value'] > 0]['Value'].sum() if len(data[data['Value'] > 0]) > 0 else 0,
            'total_loss': data[data['Value'] < 0]['Value'].sum() if len(data[data['Value'] < 0]) > 0 else 0
        }
        
        # 持仓时长统计
        if 'Duration' in data.columns:
            stats.update({
                'avg_duration': data['Duration'].mean(),
                'max_duration': data['Duration'].max(),
                'min_duration': data['Duration'].min()
            })
        
        # 收益率统计
        if 'ReturnRate' in data.columns:
            stats.update({
                'avg_return_rate': data['ReturnRate'].mean(),
                'max_return_rate': data['ReturnRate'].max(),
                'min_return_rate': data['ReturnRate'].min()
            })
            
        self.analysis_results['basic_stats'] = stats
        return stats

    def generate_position_analysis(self):
        """生成仓位分析数据"""
        if self.processed_data is None or len(self.processed_data) == 0:
            return {}
            
        data = self.processed_data
        
        # 仓位统计
        position_stats = data.groupby('PositionSize')['Value'].agg(['count', 'sum', 'mean', 'std']).round(2)
        position_counts = data['PositionSize'].value_counts()
        
        # 仓位价值范围
        value_ranges = data.groupby('PositionSize')['AbsValue'].agg(['min', 'max', 'median'])
        
        position_analysis = {
            'position_stats': position_stats,
            'position_counts': position_counts,
            'value_ranges': value_ranges
        }
        
        self.analysis_results['position_analysis'] = position_analysis
        return position_analysis

    def generate_symbol_analysis(self):
        """生成交易对分析数据"""
        if self.processed_data is None or len(self.processed_data) == 0:
            return {}
            
        data = self.processed_data
        
        # 交易对统计
        symbol_stats = data.groupby('Symbol')['Value'].agg(['count', 'sum', 'mean']).sort_values('sum', ascending=False)
        
        # 交易对盈亏分析
        symbol_pnl = data.groupby(['Symbol', 'ReturnType'])['Value'].sum().unstack(fill_value=0)
        
        symbol_analysis = {
            'symbol_stats': symbol_stats,
            'symbol_pnl': symbol_pnl
        }
        
        self.analysis_results['symbol_analysis'] = symbol_analysis
        return symbol_analysis

    def generate_time_analysis(self):
        """生成时间序列分析数据"""
        if self.processed_data is None or len(self.processed_data) == 0:
            return {}
            
        data = self.processed_data.sort_values('Time')
        
        # 累计收益
        cumulative_pnl = data['Value'].cumsum()
        
        # 月度交易频率
        monthly_trades = data.groupby(data['Time'].dt.to_period('M')).size()
        
        # 每日盈亏
        daily_summary = data.groupby(['Date', 'ReturnType'])['Value'].sum().unstack(fill_value=0)
        if 'Profit' in daily_summary.columns and 'Loss' in daily_summary.columns:
            daily_summary['Net'] = daily_summary['Profit'] + daily_summary['Loss']
        
        time_analysis = {
            'cumulative_pnl': cumulative_pnl,
            'monthly_trades': monthly_trades,
            'daily_summary': daily_summary
        }
        
        self.analysis_results['time_analysis'] = time_analysis
        return time_analysis

    def get_analysis_summary(self):
        """获取完整的分析摘要"""
        if not self.analysis_results:
            # 如果还没有生成分析结果，先生成
            self.generate_basic_statistics()
            self.generate_position_analysis()
            self.generate_symbol_analysis()
            self.generate_time_analysis()
            
        return self.analysis_results

    def print_summary_report(self):
        """打印分析摘要报告"""
        print("="*60)
        print("多头策略完整交易分析报告")
        print("="*60)
        
        # 生成统计数据
        basic_stats = self.generate_basic_statistics()
        position_analysis = self.generate_position_analysis()
        symbol_analysis = self.generate_symbol_analysis()
        
        # 基本统计
        print(f"\n📊 基本统计:")
        print(f"完整交易次数: {basic_stats['total_trades']}")
        print(f"总收益: {basic_stats['total_pnl']:,.2f}")
        print(f"平均单笔收益: {basic_stats['avg_pnl']:,.2f}")
        print(f"胜率: {basic_stats['win_rate']:.1f}%")
        
        # 持仓时长分析
        if 'avg_duration' in basic_stats:
            print(f"\n⏱️  持仓时长分析:")
            print(f"平均持仓: {basic_stats['avg_duration']:.1f} 小时")
            print(f"最长持仓: {basic_stats['max_duration']:.1f} 小时")
            print(f"最短持仓: {basic_stats['min_duration']:.1f} 小时")
        
        # 收益率分析
        if 'avg_return_rate' in basic_stats:
            print(f"\n📈 收益率分析:")
            print(f"平均收益率: {basic_stats['avg_return_rate']:.2f}%")
            print(f"最高收益率: {basic_stats['max_return_rate']:.2f}%")
            print(f"最低收益率: {basic_stats['min_return_rate']:.2f}%")
        
        # 仓位分析
        print(f"\n💰 仓位大小分析:")
        position_stats = position_analysis['position_stats']
        
        # 按照逻辑顺序排序仓位类别
        position_order = ['Small', 'Medium', 'Large', 'XLarge']
        position_stats = position_stats.reindex([pos for pos in position_order if pos in position_stats.index])
        
        for pos_size in position_stats.index:
            count = position_stats.loc[pos_size, 'count']
            total = position_stats.loc[pos_size, 'sum']
            avg = position_stats.loc[pos_size, 'mean']
            percentage = (count / basic_stats['total_trades']) * 100
            print(f"{pos_size:>7} 仓位: {count:>4}笔交易 ({percentage:>5.1f}%), 总收益{total:>10,.0f}, 平均{avg:>8,.2f}")
        
        # 仓位价值范围分析
        print(f"\n💵 仓位价值范围:")
        value_ranges = position_analysis['value_ranges']
        value_ranges = value_ranges.reindex([pos for pos in position_order if pos in value_ranges.index])
        
        for pos_size in value_ranges.index:
            min_val = value_ranges.loc[pos_size, 'min']
            max_val = value_ranges.loc[pos_size, 'max']
            median_val = value_ranges.loc[pos_size, 'median']
            print(f"{pos_size:>7} 仓位: {min_val:>8,.0f} - {max_val:>10,.0f} (中位数: {median_val:>8,.0f})")
        
        # 交易对分析
        print(f"\n🪙 交易对分析:")
        symbol_stats = symbol_analysis['symbol_stats']
        for symbol in symbol_stats.index:
            count = symbol_stats.loc[symbol, 'count']
            total = symbol_stats.loc[symbol, 'sum']
            print(f"{symbol}: {count:>4}笔交易, 总收益{total:>10,.0f}")
        
        # 收益类型分析
        print(f"\n📈 盈亏分析:")
        print(f"盈利交易: {basic_stats['profit_trades']:>4}笔, 总盈利{basic_stats['total_profit']:>12,.0f}")
        print(f"亏损交易: {basic_stats['loss_trades']:>4}笔, 总亏损{basic_stats['total_loss']:>12,.0f}")
        
        print("="*60)