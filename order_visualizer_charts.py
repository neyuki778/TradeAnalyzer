"""
订单可视化器模块
专门负责生成各种图表和可视化展示
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

class OrderVisualizer:
    """订单可视化器 - 专门处理数据可视化和图表生成"""
    
    def __init__(self, analyzer):
        """
        初始化可视化器
        
        Args:
            analyzer (OrderAnalyzer): 订单分析器实例
        """
        self.analyzer = analyzer
        self.data = analyzer.processed_data
        
        # 检查数据是否有效
        if self.data is None or len(self.data) == 0:
            raise ValueError("分析器中没有有效的处理数据")
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def _detect_optimal_scale(self, data_series, scale_type='auto'):
        """
        检测数据的最佳坐标轴尺度
        
        Args:
            data_series: 数据系列
            scale_type: 尺度类型 ('auto', 'linear', 'log', 'sqrt')
        
        Returns:
            str: 推荐的坐标轴尺度
        """
        if scale_type != 'auto':
            return scale_type
        
        # 检查数据是否包含负值
        has_negative = (data_series < 0).any()
        
        # 如果有负值，不能使用对数尺度
        if has_negative:
            # 计算数据的偏态性
            try:
                skewness = abs(data_series.skew()) if hasattr(data_series, 'skew') else 0
                data_std = data_series.std()
                data_mean = abs(data_series.mean())
                cv = data_std / data_mean if data_mean > 0 else float('inf')  # 变异系数
                
                # 对于包含负值的数据，只能使用平方根尺度或线性尺度
                if skewness > 3 and cv > 2:  # 高偏态且高变异
                    return 'sqrt'
                else:
                    return 'linear'
            except:
                return 'linear'
        
        # 对于只有正值的数据，使用原来的逻辑
        positive_data = data_series[data_series > 0]
        if len(positive_data) == 0:
            return 'linear'
        
        # 计算统计指标
        data_range = positive_data.max() / positive_data.min() if positive_data.min() > 0 else float('inf')
        skewness = abs(positive_data.skew()) if hasattr(positive_data, 'skew') else 0
        
        # 动态选择尺度
        if data_range > 1000:  # 跨越3个数量级以上
            return 'log'
        elif data_range > 100 and skewness > 2:  # 中等范围但高偏态
            return 'sqrt'
        elif skewness > 3:  # 高偏态分布
            return 'sqrt'
        else:
            return 'linear'
    
    def _apply_smart_scale(self, ax, data_series, axis='y', label_suffix=''):
        """
        应用智能坐标轴尺度
        
        Args:
            ax: matplotlib轴对象
            data_series: 数据系列
            axis: 坐标轴 ('x' 或 'y')
            label_suffix: 标签后缀
        """
        scale = self._detect_optimal_scale(data_series)
        
        # 对于包含负值的数据，需要特殊处理
        has_negative = (data_series < 0).any()
        
        if axis == 'y':
            if scale == 'log' and not has_negative:
                ax.set_yscale('log')
                ax.set_ylabel(ax.get_ylabel() + f' (Log Scale){label_suffix}')
            elif scale == 'sqrt':
                # 对于平方根尺度，需要处理负值
                if has_negative:
                    # 使用 symlog (对称对数) 来处理包含负值的数据
                    ax.set_yscale('symlog', linthresh=1)
                    ax.set_ylabel(ax.get_ylabel() + f' (Symlog Scale){label_suffix}')
                else:
                    ax.set_yscale('function', functions=(np.sqrt, np.square))
                    ax.set_ylabel(ax.get_ylabel() + f' (Sqrt Scale){label_suffix}')
        else:  # x axis
            if scale == 'log' and not has_negative:
                ax.set_xscale('log')
                ax.set_xlabel(ax.get_xlabel() + f' (Log Scale){label_suffix}')
            elif scale == 'sqrt':
                # 对于平方根尺度，需要处理负值
                if has_negative:
                    # 使用 symlog (对称对数) 来处理包含负值的数据
                    ax.set_xscale('symlog', linthresh=1)
                    ax.set_xlabel(ax.get_xlabel() + f' (Symlog Scale){label_suffix}')
                else:
                    ax.set_xscale('function', functions=(np.sqrt, np.square))
                    ax.set_xlabel(ax.get_xlabel() + f' (Sqrt Scale){label_suffix}')
        
        return scale
        
    def plot_position_size_analysis(self, figsize=(15, 10)):
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
        colors = plt.cm.Set3(np.arange(len(position_counts)))
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
        colors = plt.cm.Set3(np.arange(len(position_counts)))
        
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
        
        # 应用智能坐标轴
        self._apply_smart_scale(axes[2,1], self.data['AbsValue'], axis='y')
        
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
        
        # 应用智能坐标轴
        self._apply_smart_scale(axes[0,1], self.data['AbsValue'], axis='y')
        
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
    
    def plot_comprehensive_analysis(self, figsize=(16, 10)):
        """利润来源综合分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Strategy Profit Source Analysis', fontsize=18, fontweight='bold')
        
        # 1. 时间序列累计收益 (保留)
        data_sorted = self.data.sort_values('Time')
        cumulative_pnl = data_sorted['Value'].cumsum()
        axes[0,0].plot(data_sorted['Time'], cumulative_pnl, linewidth=2, color='navy')
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
        unique_positions = sorted(self.data['PositionSize'].unique())
        color_map = {pos: i for i, pos in enumerate(unique_positions)}
        colors = [color_map[pos] for pos in self.data['PositionSize']]
        
        scatter = axes[0,2].scatter(self.data['AbsValue'], self.data['Value'], 
                                  c=colors, cmap='viridis', alpha=0.6)
        axes[0,2].set_xlabel('Position Size (Absolute Value)')
        axes[0,2].set_ylabel('Return')
        axes[0,2].set_title('Position Size vs Return')
        
        # 应用智能坐标轴
        self._apply_smart_scale(axes[0,2], self.data['AbsValue'], axis='x')
        
        cbar = plt.colorbar(scatter, ax=axes[0,2])
        cbar.set_label('Position Size Category')
        
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
            axes[1,1].text(0.5, 0.5, 'Duration data not available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Holding Duration Analysis')
        
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
    
    def plot_profit_source_analysis(self, figsize=(15, 10)):
        """专门的利润来源分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Strategy Profit Source Deep Analysis', fontsize=18, fontweight='bold')
        
        # 1. 仓位大小利润分解
        position_stats = self.data.groupby('PositionSize')['Value'].agg(['sum', 'mean', 'count', 'std']).round(2)
        
        # 双轴图：总收益 + 平均收益
        ax1_1 = axes[0,0]
        ax1_2 = ax1_1.twinx()
        
        bars1 = position_stats['sum'].plot(kind='bar', ax=ax1_1, color='lightcoral', alpha=0.7, width=0.4)
        ax1_1.set_ylabel('Total Return', color='darkred')
        ax1_1.tick_params(axis='y', labelcolor='darkred')
        
        bars2 = position_stats['mean'].plot(kind='bar', ax=ax1_2, color='lightblue', alpha=0.7, width=0.4, position=1)
        ax1_2.set_ylabel('Average Return', color='darkblue')
        ax1_2.tick_params(axis='y', labelcolor='darkblue')
        
        ax1_1.set_title('Profit Source: Position Size')
        ax1_1.tick_params(axis='x', rotation=45)
        ax1_1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1_2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 2. 持仓时间利润分解  
        if 'Duration' in self.data.columns:
            duration_bins = pd.cut(self.data['Duration'], bins=6, 
                                 labels=['<6h', '6-12h', '12-24h', '1-3d', '3-7d', '>7d'])
            duration_stats = self.data.groupby(duration_bins)['Value'].agg(['sum', 'mean', 'count']).round(2)
            
            duration_stats['sum'].plot(kind='bar', ax=axes[0,1], 
                                     color=['lightgreen', 'green', 'orange', 'red', 'purple', 'brown'])
            axes[0,1].set_title('Profit Source: Holding Duration')
            axes[0,1].set_ylabel('Total Return')
            axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加交易数量标签
            for i, (idx, row) in enumerate(duration_stats.iterrows()):
                if not pd.isna(row['sum']):
                    axes[0,1].text(i, row['sum'], f"n={row['count']:.0f}", 
                                  ha='center', va='bottom' if row['sum'] >= 0 else 'top', fontsize=8)
        
        # 3. 交易对利润分解
        symbol_stats = self.data.groupby('Symbol')['Value'].agg(['sum', 'mean', 'count']).round(2)
        symbol_stats = symbol_stats.sort_values('sum', ascending=False)
        
        symbol_stats['sum'].plot(kind='bar', ax=axes[0,2], color='skyblue')
        axes[0,2].set_title('Profit Source: Trading Pairs')
        axes[0,2].set_ylabel('Total Return')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加胜率信息
        for i, (symbol, row) in enumerate(symbol_stats.iterrows()):
            symbol_data = self.data[self.data['Symbol'] == symbol]
            win_rate = (symbol_data['Value'] > 0).mean() * 100
            axes[0,2].text(i, row['sum'], f"{win_rate:.0f}%", 
                          ha='center', va='bottom' if row['sum'] >= 0 else 'top', fontsize=8)
        
        # 4. 月度交易数量和收益分析（替换开仓时间段）
        monthly_stats = self.data.groupby(self.data['Time'].dt.to_period('M'))['Value'].agg(['sum', 'mean', 'count']).round(2)
        
        # 创建双轴图显示交易数量和收益
        ax4_1 = axes[1,0]
        ax4_2 = ax4_1.twinx()
        
        # 月度交易数量（左轴，柱状图）
        monthly_stats['count'].plot(kind='bar', ax=ax4_1, color='lightblue', alpha=0.7, width=0.6)
        ax4_1.set_ylabel('Number of Trades', color='blue')
        ax4_1.tick_params(axis='y', labelcolor='blue')
        
        # 月度总收益（右轴，折线图）
        monthly_stats['sum'].plot(kind='line', ax=ax4_2, color='red', marker='o', linewidth=2, markersize=6)
        ax4_2.set_ylabel('Monthly Return', color='red')
        ax4_2.tick_params(axis='y', labelcolor='red')
        ax4_2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax4_1.set_title('Profit Source: Monthly Activity & Returns')
        ax4_1.tick_params(axis='x', rotation=45)
        ax4_1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (period, row) in enumerate(monthly_stats.iterrows()):
            # 交易数量标签
            ax4_1.text(i, row['count'], f"{row['count']:.0f}", 
                      ha='center', va='bottom', fontsize=8, color='blue')
            # 收益标签
            ax4_2.text(i, row['sum'], f"{row['sum']:.0f}", 
                      ha='center', va='bottom' if row['sum'] >= 0 else 'top', fontsize=8, color='red')
        
        # 5. 月度平均收益趋势分析（替换开仓星期）
        # 显示月度平均收益和波动性
        monthly_avg_return = monthly_stats['mean']
        monthly_volatility = self.data.groupby(self.data['Time'].dt.to_period('M'))['Value'].std().fillna(0)
        
        # 创建双轴图
        ax5_1 = axes[1,1]
        ax5_2 = ax5_1.twinx()
        
        # 月度平均收益（左轴）
        colors_monthly = ['gold' if x > 0 else 'lightcoral' for x in monthly_avg_return]
        monthly_avg_return.plot(kind='bar', ax=ax5_1, color=colors_monthly, alpha=0.8)
        ax5_1.set_ylabel('Average Return per Trade', color='darkgreen')
        ax5_1.tick_params(axis='y', labelcolor='darkgreen')
        ax5_1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 月度收益波动性（右轴）
        monthly_volatility.plot(kind='line', ax=ax5_2, color='purple', marker='s', linewidth=2, markersize=4)
        ax5_2.set_ylabel('Return Volatility', color='purple')
        ax5_2.tick_params(axis='y', labelcolor='purple')
        
        ax5_1.set_title('Profit Source: Monthly Performance & Volatility')
        ax5_1.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for i, (period, avg_ret) in enumerate(monthly_avg_return.items()):
            ax5_1.text(i, avg_ret, f"{avg_ret:.0f}", 
                      ha='center', va='bottom' if avg_ret >= 0 else 'top', fontsize=8, color='darkgreen')
        
        # 6. 综合利润贡献分析
        # 计算各维度对总体利润的贡献
        total_profit = self.data[self.data['Value'] > 0]['Value'].sum()
        total_loss = abs(self.data[self.data['Value'] < 0]['Value'].sum())
        
        contribution_data = {
            'Positive Positions': self.data[self.data['Value'] > 0].groupby('PositionSize')['Value'].sum(),
            'Negative Positions': self.data[self.data['Value'] < 0].groupby('PositionSize')['Value'].sum()
        }
        
        contribution_df = pd.DataFrame(contribution_data).fillna(0)
        contribution_df.plot(kind='bar', ax=axes[1,2], color=['lightgreen', 'lightcoral'])
        axes[1,2].set_title('Profit vs Loss by Position Size')
        axes[1,2].set_ylabel('Return Amount')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1,2].legend()
        
        plt.tight_layout()
        return fig
    
    def save_all_charts(self, prefix="analysis", dpi=300):
        """保存所有图表"""
        charts = {}
        
        # 生成所有图表
        print("生成图表...")
        charts['position'] = self.plot_position_size_analysis()
        charts['returns'] = self.plot_return_type_analysis()
        charts['comprehensive'] = self.plot_comprehensive_analysis()
        charts['timeseries'] = self.plot_time_series_analysis()
        
        # 保存图表
        saved_files = []
        for chart_type, fig in charts.items():
            filename = f"{prefix}_{chart_type}_analysis.png"
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            saved_files.append(filename)
            print(f"保存: {filename}")
        
        return saved_files
        
    def show_all_charts(self):
        """显示所有图表"""
        print("生成图表...")
        
        # 生成所有图表
        self.plot_position_size_analysis()
        self.plot_return_type_analysis()
        self.plot_comprehensive_analysis()
        self.plot_time_series_analysis()
        
        # 显示图表
        plt.show()
        
    def create_dashboard(self, figsize=(20, 14)):
        """创建综合仪表板"""
        fig = plt.figure(figsize=figsize)
        
        # 使用GridSpec进行更灵活的布局
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Trading Strategy Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # 1. 累计收益 (左上大图)
        ax1 = fig.add_subplot(gs[0, :2])
        data_sorted = self.data.sort_values('Time')
        cumulative_pnl = data_sorted['Value'].cumsum()
        ax1.plot(data_sorted['Time'], cumulative_pnl, linewidth=2, color='navy')
        ax1.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 仓位分布 (右上)
        ax2 = fig.add_subplot(gs[0, 2])
        position_counts = self.data['PositionSize'].value_counts()
        colors = plt.cm.Set3(np.arange(len(position_counts)))
        ax2.pie(position_counts.values, labels=position_counts.index, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Position Size Distribution', fontsize=12)
        
        # 3. 盈亏分布 (右上2)
        ax3 = fig.add_subplot(gs[0, 3])
        return_counts = self.data['ReturnType'].value_counts()
        colors_pnl = ['lightgreen' if x == 'Profit' else 'lightcoral' for x in return_counts.index]
        ax3.pie(return_counts.values, labels=return_counts.index, autopct='%1.1f%%', colors=colors_pnl)
        ax3.set_title('Profit/Loss Distribution', fontsize=12)
        
        # 4. 交易对收益贡献 (中左)
        ax4 = fig.add_subplot(gs[1, :2])
        symbol_contribution = self.data.groupby('Symbol')['Value'].sum().sort_values(ascending=False)
        symbol_contribution.plot(kind='bar', ax=ax4, color='lightcoral')
        ax4.set_title('Return Contribution by Trading Pair', fontsize=14)
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. 仓位vs收益 (中右)
        ax5 = fig.add_subplot(gs[1, 2:])
        unique_positions = sorted(self.data['PositionSize'].unique())
        color_map = {pos: i for i, pos in enumerate(unique_positions)}
        colors = [color_map[pos] for pos in self.data['PositionSize']]
        scatter = ax5.scatter(self.data['AbsValue'], self.data['Value'], c=colors, cmap='viridis', alpha=0.6)
        ax5.set_xlabel('Position Size')
        ax5.set_ylabel('Return')
        ax5.set_title('Position Size vs Return', fontsize=14)
        
        # 6. 月度交易频率 (下左)
        ax6 = fig.add_subplot(gs[2, :2])
        monthly_trades = self.data.groupby(self.data['Time'].dt.to_period('M')).size()
        monthly_trades.plot(kind='bar', ax=ax6, color='skyblue')
        ax6.set_title('Monthly Trading Frequency', fontsize=14)
        ax6.tick_params(axis='x', rotation=45)
        
        # 7. 收益分布直方图 (下右)
        ax7 = fig.add_subplot(gs[2, 2:])
        ax7.hist(self.data['Value'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax7.axvline(self.data['Value'].mean(), color='red', linestyle='--', 
                   label=f'Avg: {self.data["Value"].mean():.2f}')
        ax7.set_title('Return Distribution', fontsize=14)
        ax7.legend()
        
        return fig
    
    def plot_time_series_analysis(self, figsize=(16, 10)):
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