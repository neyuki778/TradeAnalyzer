"""
订单分析可视化模块
用于分析交易策略的订单数据，包括仓位大小、收益情况等分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持 (macOS)
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OrderAnalyzer:
    """订单分析器 - 分析交易订单数据"""
    
    def __init__(self, csv_file_path):
        """
        初始化订单分析器
        
        Args:
            csv_file_path (str): 订单数据CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.processed_data = None
        self.load_data()
        
    def load_data(self):
        """加载订单数据"""
        try:
            self.data = pd.read_csv(self.csv_file_path)
            print(f"成功加载订单数据: {len(self.data)} 条记录")
            print(f"数据时间范围: {self.data['Time'].min()} 到 {self.data['Time'].max()}")
            self._preprocess_data()
        except Exception as e:
            print(f"加载数据失败: {e}")
            
    def _preprocess_data(self):
        """预处理数据 - 针对多头策略的开平仓配对分析"""
        # 转换时间格式
        self.data['Time'] = pd.to_datetime(self.data['Time'])
        
        # 1. 首先过滤有效订单，排除Invalid状态
        filled_data = self.data[self.data['Status'] == 'Filled'].copy()
        print(f"过滤无效订单后: {len(filled_data)} 条有效订单")
        
        # 2. 分析开平仓配对 - 多头策略特征分析
        filled_data['OrderSide'] = np.where(filled_data['Quantity'] > 0, 'Open', 'Close')
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
            
    def _pair_open_close_orders(self, filled_data):
        """配对开平仓订单，计算完整交易的真实收益 - 优化版本"""
        trades = []
        
        print("开始配对开平仓订单...")
        
        # 按交易对分组处理
        for symbol in filled_data['Symbol'].unique():
            symbol_data = filled_data[filled_data['Symbol'] == symbol].sort_values('Time').reset_index(drop=True)
            
            # 分离开仓和平仓订单
            open_orders = symbol_data[symbol_data['Quantity'] > 0].copy()
            close_orders = symbol_data[symbol_data['Quantity'] < 0].copy()
            
            print(f"处理 {symbol}: {len(open_orders)} 开仓, {len(close_orders)} 平仓")
            
            # 使用更高效的配对算法
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
                    # 对于多头策略: P&L = 卖出收入 - 买入成本
                    # close_order['Value'] 是卖出收入(正值)，open_order['Value'] 是买入成本(负值)
                    trade_pnl = close_order['Value'] + open_order['Value']  # 正确：正值+负值=净收益
                    
                    # 创建完整交易记录
                    trade = {
                        'Time': open_order['Time'],
                        'CloseTime': close_order['Time'],
                        'Symbol': symbol,
                        'OpenPrice': open_order['Price'],
                        'ClosePrice': close_order['Price'],
                        'Quantity': open_order['AbsQuantity'],
                        'OpenValue': abs(open_order['Value']),  # 买入成本(正值显示)
                        'CloseValue': abs(close_order['Value']), # 卖出收入(正值显示)
                        'Value': trade_pnl,  # 真实交易净收益
                        'AbsValue': abs(open_order['Value']),  # 仓位大小(买入成本)
                        'Duration': (close_order['Time'] - open_order['Time']).total_seconds() / 3600,
                        'Type': 'Long',
                        'Status': 'Completed',
                        'Tag': close_order['Tag'] if pd.notna(close_order['Tag']) else ''
                    }
                    trades.append(trade)
                    
                    # 标记已使用的平仓订单
                    used_close_indices.add(close_order.name)
        
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
        
        return trades_df
        
    def _categorize_position_size(self):
        """按仓位大小分类 - 智能分布分析"""
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
        if len(self.processed_data) == 0:
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

class OrderVisualizer:
    """订单可视化器"""
    
    def __init__(self, analyzer):
        """
        初始化可视化器
        
        Args:
            analyzer (OrderAnalyzer): 订单分析器实例
        """
        self.analyzer = analyzer
        self.data = analyzer.processed_data
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_position_size_analysis(self, figsize=(18, 12)):
        """可视化不同仓位大小的分析 - 自适应布局"""
        # 检查仓位类别数量
        position_categories = self.data['PositionSize'].unique()
        n_categories = len(position_categories)
        
        if n_categories <= 3:
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            self._plot_standard_position_analysis(fig, axes)
        else:
            fig, axes = plt.subplots(3, 2, figsize=figsize)
            self._plot_detailed_position_analysis(fig, axes)
        
        return fig
    
    def _plot_standard_position_analysis(self, fig, axes):
        """标准仓位分析 (3类别)"""
        fig.suptitle('Position Size Analysis', fontsize=16, fontweight='bold')
        
        # 1. 仓位大小分布
        position_counts = self.data['PositionSize'].value_counts()
        colors = plt.cm.Set3(range(len(position_counts)))
        axes[0,0].pie(position_counts.values, labels=position_counts.index, autopct='%1.1f%%', 
                     colors=colors, startangle=90)
        axes[0,0].set_title('Position Size Distribution')
        
        # 2. 不同仓位大小的平均收益
        position_returns = self.data.groupby('PositionSize')['Value'].agg(['mean', 'sum', 'count'])
        position_returns['mean'].plot(kind='bar', ax=axes[0,1], color=colors)
        axes[0,1].set_title('Average Returns by Position Size')
        axes[0,1].set_ylabel('Average Return')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. 仓位大小与累计收益
        for i, position_size in enumerate(self.data['PositionSize'].unique()):
            subset = self.data[self.data['PositionSize'] == position_size].sort_values('Time')
            cumulative = subset['Value'].cumsum()
            axes[1,0].plot(subset['Time'], cumulative, label=f'{position_size}', 
                          linewidth=2, color=colors[i])
        
        axes[1,0].set_title('Cumulative Returns by Position Size')
        axes[1,0].set_ylabel('Cumulative Return')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 仓位大小收益箱线图
        sns.boxplot(data=self.data, x='PositionSize', y='Value', ax=axes[1,1], palette='Set3')
        axes[1,1].set_title('Return Distribution by Position Size')
        axes[1,1].set_ylabel('Return')
        
        plt.tight_layout()
    
    def _plot_detailed_position_analysis(self, fig, axes):
        """详细仓位分析 (4+类别)"""
        fig.suptitle('Detailed Position Size Analysis', fontsize=16, fontweight='bold')
        
        position_stats = self.data.groupby('PositionSize')['Value'].agg(['count', 'sum', 'mean', 'std']).round(2)
        position_counts = self.data['PositionSize'].value_counts()
        colors = plt.cm.Set3(range(len(position_counts)))
        
        # 1. 仓位大小分布 - 饼图
        axes[0,0].pie(position_counts.values, labels=position_counts.index, autopct='%1.1f%%', 
                     colors=colors, startangle=90)
        axes[0,0].set_title('Position Size Distribution')
        
        # 2. 仓位统计表格
        axes[0,1].axis('tight')
        axes[0,1].axis('off')
        table_data = []
        for pos_size in position_stats.index:
            row = [
                pos_size,
                f"{position_stats.loc[pos_size, 'count']:,}",
                f"{position_stats.loc[pos_size, 'sum']:,.0f}",
                f"{position_stats.loc[pos_size, 'mean']:,.2f}"
            ]
            table_data.append(row)
        
        table = axes[0,1].table(cellText=table_data,
                               colLabels=['Position', 'Count', 'Total Return', 'Avg Return'],
                               cellLoc='center',
                               loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[0,1].set_title('Position Statistics Summary')
        
        # 3. 平均收益对比
        position_stats['mean'].plot(kind='bar', ax=axes[1,0], color=colors)
        axes[1,0].set_title('Average Returns by Position Size')
        axes[1,0].set_ylabel('Average Return')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. 累计收益趋势
        for i, position_size in enumerate(self.data['PositionSize'].unique()):
            subset = self.data[self.data['PositionSize'] == position_size].sort_values('Time')
            cumulative = subset['Value'].cumsum()
            axes[1,1].plot(subset['Time'], cumulative, label=f'{position_size}', 
                          linewidth=2, color=colors[i])
        
        axes[1,1].set_title('Cumulative Returns by Position Size')
        axes[1,1].set_ylabel('Cumulative Return')
        axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 5. 收益分布箱线图
        sns.boxplot(data=self.data, x='PositionSize', y='Value', ax=axes[2,0], palette='Set3')
        axes[2,0].set_title('Return Distribution by Position Size')
        axes[2,0].set_ylabel('Return')
        axes[2,0].tick_params(axis='x', rotation=45)
        
        # 6. 仓位价值分布
        sns.boxplot(data=self.data, x='PositionSize', y='AbsValue', ax=axes[2,1], palette='Set3')
        axes[2,1].set_title('Position Value Distribution')
        axes[2,1].set_ylabel('Position Value')
        axes[2,1].tick_params(axis='x', rotation=45)
        axes[2,1].set_yscale('log')  # 使用对数坐标更好显示长尾分布
        
        plt.tight_layout()
    
    def plot_return_type_analysis(self, figsize=(15, 10)):
        """可视化不同收益类型的分析"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Return Type Analysis', fontsize=16, fontweight='bold')
        
        # 1. 盈亏分布
        return_counts = self.data['ReturnType'].value_counts()
        colors = ['lightgreen' if x == 'Profit' else 'lightcoral' for x in return_counts.index]
        axes[0,0].pie(return_counts.values, labels=return_counts.index, autopct='%1.1f%%', 
                     colors=colors, startangle=90)
        axes[0,0].set_title('Profit/Loss Distribution')
        
        # 2. 不同收益类型的金额分布
        sns.boxplot(data=self.data, x='ReturnType', y='AbsValue', ax=axes[0,1])
        axes[0,1].set_title('Order Value Distribution by P&L Type')
        axes[0,1].set_ylabel('Order Value')
        
        # 3. 持仓时间vs收益分析
        if 'Duration' in self.data.columns:
            # 创建持仓时间分组
            duration_bins = pd.cut(self.data['Duration'], bins=6, 
                                 labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Ultra Long'])
            duration_returns = self.data.groupby(duration_bins)['Value'].agg(['sum', 'mean', 'count'])
            
            # 绘制持仓时间vs总收益
            duration_returns['sum'].plot(kind='bar', ax=axes[1,0], 
                                       color=['lightblue', 'skyblue', 'orange', 'coral', 'red', 'darkred'])
            axes[1,0].set_title('Returns by Holding Duration')
            axes[1,0].set_ylabel('Total Return')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # 添加数据标签
            for i, v in enumerate(duration_returns['sum']):
                axes[1,0].text(i, v, f'{v:.0f}', ha='center', va='bottom' if v >= 0 else 'top')
        else:
            axes[1,0].text(0.5, 0.5, 'Duration data not available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Holding Duration Analysis')
        
        # 4. 交易对盈亏比分析 (金额比率)
        symbol_analysis = []
        for symbol in self.data['Symbol'].unique():
            symbol_data = self.data[self.data['Symbol'] == symbol]
            profit_trades = symbol_data[symbol_data['Value'] > 0]
            loss_trades = symbol_data[symbol_data['Value'] < 0]
            
            profit_count = len(profit_trades)
            loss_count = len(loss_trades)
            total_trades = len(symbol_data)
            
            # 计算胜率
            win_rate = profit_count / total_trades * 100
            
            # 计算真正的盈亏比：平均盈利金额 / 平均亏损金额
            avg_profit = profit_trades['Value'].mean() if profit_count > 0 else 0
            avg_loss = abs(loss_trades['Value'].mean()) if loss_count > 0 else 1  # 取绝对值
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
            
            symbol_analysis.append({
                'Symbol': symbol,
                'Win_Rate': win_rate,
                'Profit_Loss_Ratio': profit_loss_ratio,
                'Avg_Profit': avg_profit,
                'Avg_Loss': avg_loss,
                'Total_Trades': total_trades
            })
        
        symbol_df = pd.DataFrame(symbol_analysis).set_index('Symbol')
        
        # 创建双轴图表
        ax1 = axes[1,1]
        ax2 = ax1.twinx()
        
        # 绘制盈亏比（左轴）
        bars1 = symbol_df['Profit_Loss_Ratio'].plot(kind='bar', ax=ax1, color='lightseagreen', alpha=0.7, width=0.4, position=0)
        ax1.set_ylabel('Profit/Loss Ratio (Amount)', color='darkslategray')
        ax1.tick_params(axis='y', labelcolor='darkslategray')
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Break-even')
        
        # 绘制胜率（右轴）
        bars2 = symbol_df['Win_Rate'].plot(kind='bar', ax=ax2, color='orange', alpha=0.7, width=0.4, position=1)
        ax2.set_ylabel('Win Rate (%)', color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_ylim(0, 100)
        
        ax1.set_title('Profit-Loss Ratio & Win Rate by Trading Pair')
        ax1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for i, (symbol, row) in enumerate(symbol_df.iterrows()):
            # 盈亏比标签
            ax1.text(i, row['Profit_Loss_Ratio'], f'{row["Profit_Loss_Ratio"]:.2f}', 
                    ha='center', va='bottom', fontsize=8, color='darkslategray')
            # 胜率标签
            ax2.text(i, row['Win_Rate'], f'{row["Win_Rate"]:.0f}%', 
                    ha='center', va='bottom', fontsize=8, color='darkorange')
        
        # 图例
        ax1.legend(['P/L Ratio', 'Break-even'], loc='upper left')
        ax2.legend(['Win Rate'], loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def plot_comprehensive_analysis(self, figsize=(20, 12)):
        """利润来源综合分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Strategy Profit Source Analysis', fontsize=18, fontweight='bold')
        
        # 1. 时间序列累计收益 (保留)
        self.data_sorted = self.data.sort_values('Time')
        cumulative_pnl = self.data_sorted['Value'].cumsum()
        axes[0,0].plot(self.data_sorted['Time'], cumulative_pnl, linewidth=2, color='navy')
        axes[0,0].set_title('Cumulative Returns')
        axes[0,0].set_ylabel('Cumulative Return')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 开仓时间段分析 (替换交易频率)
        # 分析一天中不同小时的开仓表现
        self.data['OpenHour'] = self.data['Time'].dt.hour
        hourly_performance = self.data.groupby('OpenHour')['Value'].agg(['sum', 'mean', 'count'])
        
        # 创建双轴图
        ax2_1 = axes[0,1]
        ax2_2 = ax2_1.twinx()
        
        # 绘制每小时总收益
        ax2_1.bar(hourly_performance.index, hourly_performance['sum'], alpha=0.7, color='lightblue', label='Total Return')
        ax2_1.set_ylabel('Total Return', color='blue')
        ax2_1.tick_params(axis='y', labelcolor='blue')
        
        # 绘制交易数量
        ax2_2.plot(hourly_performance.index, hourly_performance['count'], color='red', marker='o', linewidth=2, label='Trade Count')
        ax2_2.set_ylabel('Number of Trades', color='red')
        ax2_2.tick_params(axis='y', labelcolor='red')
        
        ax2_1.set_title('Profit by Opening Hour')
        ax2_1.set_xlabel('Hour of Day')
        ax2_1.grid(True, alpha=0.3)
        
        # 3. 仓位大小vs收益散点图 (保留)
        scatter = axes[0,2].scatter(self.data['AbsValue'], self.data['Value'], 
                                  c=self.data['PositionSize'].map({'Small': 0, 'Medium': 1, 'Large': 2, 'XLarge': 3}),
                                  cmap='viridis', alpha=0.6)
        axes[0,2].set_xlabel('Position Size (Absolute Value)')
        axes[0,2].set_ylabel('Return')
        axes[0,2].set_title('Position Size vs Return')
        axes[0,2].set_xscale('log')  # 使用对数刻度更好显示
        plt.colorbar(scatter, ax=axes[0,2], label='Position Size')
        
        # 4. 交易对收益贡献 (保留)
        symbol_contribution = self.data.groupby('Symbol')['Value'].sum().sort_values(ascending=False)
        symbol_contribution.plot(kind='bar', ax=axes[1,0], color='lightcoral')
        axes[1,0].set_title('Return Contribution by Trading Pair')
        axes[1,0].set_ylabel('Total Return')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. 持仓时长vs收益分析 (保留，但优化)
        if 'Duration' in self.data.columns:
            # 创建更细致的持仓时间分组
            duration_bins = pd.cut(self.data['Duration'], bins=8, 
                                 labels=['<2h', '2-6h', '6-12h', '12-24h', '1-2d', '2-4d', '4-7d', '>7d'])
            duration_returns = self.data.groupby(duration_bins)['Value'].agg(['sum', 'mean', 'count'])
            
            duration_returns['sum'].plot(kind='bar', ax=axes[1,1], 
                                       color=['lightgreen', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray'])
            axes[1,1].set_title('Returns by Holding Duration')
            axes[1,1].set_ylabel('Total Return')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # 添加数据标签
            for i, v in enumerate(duration_returns['sum']):
                if not pd.isna(v):
                    axes[1,1].text(i, v, f'{v:.0f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
        else:
            # 备选：收益率分析
            if 'ReturnRate' in self.data.columns:
                axes[1,1].hist(self.data['ReturnRate'], bins=50, alpha=0.7, color='orange', edgecolor='black')
                axes[1,1].axvline(self.data['ReturnRate'].mean(), color='red', linestyle='--', 
                                 label=f'Avg Return Rate: {self.data["ReturnRate"].mean():.2f}%')
                axes[1,1].set_title('Return Rate Distribution')
                axes[1,1].set_xlabel('Return Rate (%)')
                axes[1,1].set_ylabel('Frequency')
                axes[1,1].legend()
        
        # 6. 开平仓时机分析 (新增)
        # 分析开仓和平仓时的市场表现
        if 'CloseTime' in self.data.columns:
            # 计算开仓到平仓的时间差对应的收益率
            self.data['OpenWeekday'] = self.data['Time'].dt.day_name()
            self.data['CloseWeekday'] = self.data['CloseTime'].dt.day_name()
            
            # 按开仓日期分组
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_performance = self.data.groupby('OpenWeekday')['Value'].agg(['sum', 'mean', 'count'])
            weekday_performance = weekday_performance.reindex([day for day in weekday_order if day in weekday_performance.index])
            
            weekday_performance['mean'].plot(kind='bar', ax=axes[1,2], color='skyblue')
            axes[1,2].set_title('Average Return by Opening Weekday')
            axes[1,2].set_ylabel('Average Return')
            axes[1,2].tick_params(axis='x', rotation=45)
            
            # 添加零线
            axes[1,2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # 添加数值标签
            for i, v in enumerate(weekday_performance['mean']):
                if not pd.isna(v):
                    axes[1,2].text(i, v, f'{v:.0f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
        else:
            # 备选：收益分布直方图
            axes[1,2].hist(self.data['Value'], bins=50, alpha=0.7, color='purple', edgecolor='black')
            axes[1,2].axvline(self.data['Value'].mean(), color='red', linestyle='--', 
                             label=f'Average Return: {self.data["Value"].mean():.2f}')
            axes[1,2].set_title('Return Distribution')
            axes[1,2].set_xlabel('Single Trade Return')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_time_series_analysis(self, figsize=(20, 12)):
        """时间序列分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Time Series Analysis', fontsize=18, fontweight='bold')
        
        # 按时间排序数据
        data_sorted = self.data.sort_values('Time')
        
        # 1. 累计收益时间序列
        cumulative_pnl = data_sorted['Value'].cumsum()
        axes[0,0].plot(data_sorted['Time'], cumulative_pnl, linewidth=2, color='navy')
        axes[0,0].set_title('Cumulative P&L Over Time')
        axes[0,0].set_ylabel('Cumulative Return')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. 月度交易频率和收益
        monthly_stats = data_sorted.groupby(data_sorted['Time'].dt.to_period('M'))['Value'].agg(['count', 'sum', 'mean'])
        
        # 双轴图
        ax2_1 = axes[0,1]
        ax2_2 = ax2_1.twinx()
        
        # 交易数量（左轴）
        monthly_stats['count'].plot(kind='bar', ax=ax2_1, color='lightblue', alpha=0.7, width=0.6)
        ax2_1.set_ylabel('Number of Trades', color='blue')
        ax2_1.tick_params(axis='y', labelcolor='blue')
        
        # 月度总收益（右轴）
        monthly_stats['sum'].plot(kind='line', ax=ax2_2, color='red', marker='o', linewidth=2)
        ax2_2.set_ylabel('Monthly Return', color='red')
        ax2_2.tick_params(axis='y', labelcolor='red')
        ax2_2.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        
        ax2_1.set_title('Monthly Trading Activity & Returns')
        ax2_1.tick_params(axis='x', rotation=45)
        
        # 3. 每日交易模式分析
        data_sorted['Date'] = data_sorted['Time'].dt.date
        daily_stats = data_sorted.groupby('Date')['Value'].agg(['count', 'sum']).reset_index()
        daily_stats['Date'] = pd.to_datetime(daily_stats['Date'])
        
        # 绘制每日交易数量
        axes[1,0].scatter(daily_stats['Date'], daily_stats['count'], alpha=0.6, color='green')
        axes[1,0].set_title('Daily Trading Frequency')
        axes[1,0].set_ylabel('Trades per Day')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 收益波动性分析
        # 计算滚动收益统计
        data_sorted['CumReturn'] = data_sorted['Value'].cumsum()
        
        # 30日滚动标准差（如果有足够数据）
        if len(data_sorted) > 30:
            rolling_std = data_sorted['Value'].rolling(window=30, min_periods=10).std()
            axes[1,1].plot(data_sorted['Time'], rolling_std, linewidth=2, color='orange')
            axes[1,1].set_title('30-Trade Rolling Return Volatility')
            axes[1,1].set_ylabel('Return Volatility')
        else:
            # 备选：收益分布直方图
            axes[1,1].hist(self.data['Value'], bins=30, alpha=0.7, color='purple', edgecolor='black')
            axes[1,1].axvline(self.data['Value'].mean(), color='red', linestyle='--', 
                             label=f'Mean: {self.data["Value"].mean():.2f}')
            axes[1,1].set_title('Return Distribution')
            axes[1,1].set_xlabel('Single Trade Return')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].legend()
        
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self):
        """生成分析摘要报告"""
        print("="*60)
        print("多头策略完整交易分析报告")
        print("="*60)
        
        # 基本统计
        total_trades = len(self.data)
        total_pnl = self.data['Value'].sum()
        avg_pnl = self.data['Value'].mean()
        win_rate = (self.data['Value'] > 0).mean() * 100
        
        print(f"\n📊 基本统计:")
        print(f"完整交易次数: {total_trades}")
        print(f"总收益: {total_pnl:,.2f}")
        print(f"平均单笔收益: {avg_pnl:,.2f}")
        print(f"胜率: {win_rate:.1f}%")
        
        # 持仓时长分析
        if 'Duration' in self.data.columns:
            avg_duration = self.data['Duration'].mean()
            max_duration = self.data['Duration'].max()
            min_duration = self.data['Duration'].min()
            print(f"\n⏱️  持仓时长分析:")
            print(f"平均持仓: {avg_duration:.1f} 小时")
            print(f"最长持仓: {max_duration:.1f} 小时")
            print(f"最短持仓: {min_duration:.1f} 小时")
        
        # 收益率分析
        if 'ReturnRate' in self.data.columns:
            avg_return_rate = self.data['ReturnRate'].mean()
            max_return_rate = self.data['ReturnRate'].max()
            min_return_rate = self.data['ReturnRate'].min()
            print(f"\n📈 收益率分析:")
            print(f"平均收益率: {avg_return_rate:.2f}%")
            print(f"最高收益率: {max_return_rate:.2f}%")
            print(f"最低收益率: {min_return_rate:.2f}%")
        
        # 仓位分析 - 适应新的分类系统
        print(f"\n💰 仓位大小分析:")
        position_stats = self.data.groupby('PositionSize')['Value'].agg(['count', 'sum', 'mean'])
        
        # 按照逻辑顺序排序仓位类别
        position_order = ['Small', 'Medium', 'Large', 'XLarge']
        position_stats = position_stats.reindex([pos for pos in position_order if pos in position_stats.index])
        
        for pos_size in position_stats.index:
            count = position_stats.loc[pos_size, 'count']
            total = position_stats.loc[pos_size, 'sum']
            avg = position_stats.loc[pos_size, 'mean']
            percentage = (count / total_trades) * 100
            print(f"{pos_size:>7} 仓位: {count:>4}笔交易 ({percentage:>5.1f}%), 总收益{total:>10,.0f}, 平均{avg:>8,.2f}")
        
        # 仓位价值范围分析
        print(f"\n💵 仓位价值范围:")
        value_ranges = self.data.groupby('PositionSize')['AbsValue'].agg(['min', 'max', 'median'])
        value_ranges = value_ranges.reindex([pos for pos in position_order if pos in value_ranges.index])
        
        for pos_size in value_ranges.index:
            min_val = value_ranges.loc[pos_size, 'min']
            max_val = value_ranges.loc[pos_size, 'max']
            median_val = value_ranges.loc[pos_size, 'median']
            print(f"{pos_size:>7} 仓位: {min_val:>8,.0f} - {max_val:>10,.0f} (中位数: {median_val:>8,.0f})")
        
        # 交易对分析
        print(f"\n🪙 交易对分析:")
        symbol_stats = self.data.groupby('Symbol')['Value'].agg(['count', 'sum']).sort_values('sum', ascending=False)
        for symbol in symbol_stats.index:
            count = symbol_stats.loc[symbol, 'count']
            total = symbol_stats.loc[symbol, 'sum']
            print(f"{symbol}: {count:>4}笔交易, 总收益{total:>10,.0f}")
        
        # 收益类型分析
        print(f"\n📈 盈亏分析:")
        profit_trades = self.data[self.data['Value'] > 0]
        loss_trades = self.data[self.data['Value'] < 0]
        
        if len(profit_trades) > 0:
            print(f"盈利交易: {len(profit_trades):>4}笔, 总盈利{profit_trades['Value'].sum():>12,.0f}")
        if len(loss_trades) > 0:
            print(f"亏损交易: {len(loss_trades):>4}笔, 总亏损{loss_trades['Value'].sum():>12,.0f}")
        
        print("="*60)

def analyze_macd_crypto_orders(csv_file_path, save_plots=False):
    """
    分析MACD Long Crypto策略订单数据的主函数
    
    Args:
        csv_file_path (str): 订单数据CSV文件路径
        save_plots (bool): 是否保存图表，默认False
    """
    print("开始分析MACD Long Crypto策略订单数据...")
    
    # 创建分析器
    analyzer = OrderAnalyzer(csv_file_path)
    
    # 创建可视化器
    visualizer = OrderVisualizer(analyzer)
    
    # 生成分析报告
    visualizer.generate_summary_report()
    
    # 生成图表
    print("\n生成可视化图表...")
    
    # 仓位大小分析
    fig1 = visualizer.plot_position_size_analysis()
    if save_plots:
        fig1.savefig('MACD_position_size_analysis.png', dpi=300, bbox_inches='tight')
        print("保存: MACD_position_size_analysis.png")
    
    # 收益类型分析
    fig2 = visualizer.plot_return_type_analysis()
    if save_plots:
        fig2.savefig('MACD_return_type_analysis.png', dpi=300, bbox_inches='tight')
        print("保存: MACD_return_type_analysis.png")
    
    # 综合分析
    fig3 = visualizer.plot_comprehensive_analysis()
    if save_plots:
        fig3.savefig('MACD_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("保存: MACD_comprehensive_analysis.png")
    
    plt.show()
    
    return analyzer, visualizer

if __name__ == "__main__":
    # 示例用法
    csv_file = "/Users/yukiarima/Desktop/Quant/QuantFramework/orders-analysis/MACD-long-crypto/MACD-long-crypto-2023-2024.csv"
    analyzer, visualizer = analyze_macd_crypto_orders(csv_file)