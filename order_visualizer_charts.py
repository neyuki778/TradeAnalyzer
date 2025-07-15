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
    
    def plot_profit_source_analysis(self, figsize=(18, 12)):
        """核心利润来源分析 - 专注于关键维度"""
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Profit Source Analysis - Key Dimensions', fontsize=20, fontweight='bold')
        
        # 1. 仓位大小利润总览
        position_stats = self.data.groupby('PositionSize')['Value'].agg(['sum', 'mean', 'count']).round(2)
        position_stats = position_stats.sort_values('sum', ascending=False)
        
        bars = position_stats['sum'].plot(kind='bar', ax=axes[0,0], color='lightcoral', alpha=0.8)
        axes[0,0].set_title('Profit by Position Size', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Total Return')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 添加数值标签
        for i, (pos, row) in enumerate(position_stats.iterrows()):
            axes[0,0].text(i, row['sum'], f'{row["sum"]:.0f}\n({row["count"]})', 
                          ha='center', va='bottom' if row['sum'] >= 0 else 'top', fontsize=10)
        
        # 2. 持仓时间利润分解
        if 'Duration' in self.data.columns:
            duration_bins = pd.cut(self.data['Duration'], bins=6, 
                                 labels=['<6h', '6-12h', '12-24h', '1-3d', '3-7d', '>7d'])
            duration_stats = self.data.groupby(duration_bins)['Value'].agg(['sum', 'mean', 'count']).round(2)
            
            duration_stats['sum'].plot(kind='bar', ax=axes[0,1], 
                                     color=['lightgreen', 'green', 'orange', 'red', 'purple', 'brown'])
            axes[0,1].set_title('Profit by Holding Duration', fontsize=14, fontweight='bold')
            axes[0,1].set_ylabel('Total Return')
            axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, (dur, row) in enumerate(duration_stats.iterrows()):
                if not pd.isna(row['sum']):
                    axes[0,1].text(i, row['sum'], f'{row["sum"]:.0f}\n({row["count"]})', 
                                  ha='center', va='bottom' if row['sum'] >= 0 else 'top', fontsize=10)
        else:
            axes[0,1].text(0.5, 0.5, 'Duration data not available', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Holding Duration Analysis')
        
        # 3. 交易对利润贡献 & 胜率
        symbol_analysis = []
        for symbol in self.data['Symbol'].unique():
            symbol_data = self.data[self.data['Symbol'] == symbol]
            profit_trades = symbol_data[symbol_data['Value'] > 0]
            total_trades = len(symbol_data)
            total_return = symbol_data['Value'].sum()
            win_rate = len(profit_trades) / total_trades * 100 if total_trades > 0 else 0
            
            symbol_analysis.append({
                'Symbol': symbol,
                'Total_Return': total_return,
                'Win_Rate': win_rate,
                'Trade_Count': total_trades
            })
        
        symbol_df = pd.DataFrame(symbol_analysis).set_index('Symbol')
        symbol_df = symbol_df.sort_values('Total_Return', ascending=False)
        
        # 双轴图：总收益 + 胜率
        ax3_1 = axes[0,2]
        ax3_2 = ax3_1.twinx()
        
        symbol_df['Total_Return'].plot(kind='bar', ax=ax3_1, color='skyblue', alpha=0.8)
        ax3_1.set_ylabel('Total Return', color='blue')
        ax3_1.tick_params(axis='y', labelcolor='blue')
        ax3_1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        symbol_df['Win_Rate'].plot(kind='line', ax=ax3_2, color='red', marker='o', linewidth=2)
        ax3_2.set_ylabel('Win Rate (%)', color='red')
        ax3_2.tick_params(axis='y', labelcolor='red')
        ax3_2.set_ylim(0, 100)
        
        ax3_1.set_title('Profit by Trading Pair & Win Rate', fontsize=14, fontweight='bold')
        ax3_1.tick_params(axis='x', rotation=45)
        
        # 4. 月度交易活动与收益
        monthly_stats = self.data.groupby(self.data['Time'].dt.to_period('M'))['Value'].agg(['sum', 'mean', 'count']).round(2)
        
        ax4_1 = axes[1,0]
        ax4_2 = ax4_1.twinx()
        
        monthly_stats['count'].plot(kind='bar', ax=ax4_1, color='lightblue', alpha=0.7, width=0.6)
        ax4_1.set_ylabel('Trade Count', color='blue')
        ax4_1.tick_params(axis='y', labelcolor='blue')
        
        monthly_stats['sum'].plot(kind='line', ax=ax4_2, color='red', marker='o', linewidth=2)
        ax4_2.set_ylabel('Monthly Return', color='red')
        ax4_2.tick_params(axis='y', labelcolor='red')
        ax4_2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        ax4_1.set_title('Monthly Activity & Returns', fontsize=14, fontweight='bold')
        ax4_1.tick_params(axis='x', rotation=45)
        
        # 5. 小时分析（开仓时机）
        self.data['OpenHour'] = self.data['Time'].dt.hour
        hourly_stats = self.data.groupby('OpenHour')['Value'].agg(['sum', 'mean', 'count'])
        
        ax5_1 = axes[1,1]
        ax5_2 = ax5_1.twinx()
        
        hourly_stats['sum'].plot(kind='bar', ax=ax5_1, color='lightgreen', alpha=0.7)
        ax5_1.set_ylabel('Total Return', color='green')
        ax5_1.tick_params(axis='y', labelcolor='green')
        ax5_1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        hourly_stats['count'].plot(kind='line', ax=ax5_2, color='orange', marker='s', linewidth=2)
        ax5_2.set_ylabel('Trade Count', color='orange')
        ax5_2.tick_params(axis='y', labelcolor='orange')
        
        ax5_1.set_title('Profit by Opening Hour', fontsize=14, fontweight='bold')
        ax5_1.set_xlabel('Hour of Day')
        
        # 6. 方向分析（买卖）
        if 'Type' in self.data.columns:
            direction_stats = self.data.groupby('Type')['Value'].agg(['sum', 'mean', 'count'])
            
            # 创建饼图显示方向分布
            direction_counts = self.data['Type'].value_counts()
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'orange'][:len(direction_counts)]
            axes[1,2].pie(direction_counts.values, labels=direction_counts.index, autopct='%1.1f%%', 
                         colors=colors, startangle=90)
            axes[1,2].set_title('Trade Direction Distribution', fontsize=14, fontweight='bold')
        else:
            axes[1,2].text(0.5, 0.5, 'Direction data not available', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
        
        # 7. 累计收益曲线
        data_sorted = self.data.sort_values('Time')
        cumulative_pnl = data_sorted['Value'].cumsum()
        axes[2,0].plot(data_sorted['Time'], cumulative_pnl, linewidth=2, color='navy')
        axes[2,0].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[2,0].set_ylabel('Cumulative Return')
        axes[2,0].tick_params(axis='x', rotation=45)
        axes[2,0].grid(True, alpha=0.3)
        axes[2,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 8. 盈亏分布
        profit_data = self.data[self.data['Value'] > 0]['Value']
        loss_data = self.data[self.data['Value'] < 0]['Value']
        
        axes[2,1].hist([profit_data, loss_data], bins=30, alpha=0.7, 
                      color=['lightgreen', 'lightcoral'], label=['Profit', 'Loss'])
        axes[2,1].set_title('Return Distribution', fontsize=14, fontweight='bold')
        axes[2,1].set_xlabel('Return Amount')
        axes[2,1].set_ylabel('Frequency')
        axes[2,1].legend()
        axes[2,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 9. 风险收益散点图
        position_risk_return = self.data.groupby('PositionSize')['Value'].agg(['mean', 'std']).fillna(0)
        
        colors = ['red', 'orange', 'green', 'blue', 'purple'][:len(position_risk_return)]
        for i, (pos_size, row) in enumerate(position_risk_return.iterrows()):
            axes[2,2].scatter(row['std'], row['mean'], s=100, color=colors[i], alpha=0.7, label=pos_size)
        
        axes[2,2].set_xlabel('Risk (Return Std Dev)')
        axes[2,2].set_ylabel('Expected Return')
        axes[2,2].set_title('Risk-Return by Position Size', fontsize=14, fontweight='bold')
        axes[2,2].legend()
        axes[2,2].grid(True, alpha=0.3)
        axes[2,2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[2,2].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_performance_overview(self, figsize=(16, 8)):
        """策略表现总览 - 简化版"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Strategy Performance Overview', fontsize=18, fontweight='bold')
        
        # 1. 累计收益曲线
        data_sorted = self.data.sort_values('Time')
        cumulative_pnl = data_sorted['Value'].cumsum()
        axes[0,0].plot(data_sorted['Time'], cumulative_pnl, linewidth=2, color='navy')
        axes[0,0].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Cumulative Return')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. 月度收益与交易数量
        monthly_stats = self.data.groupby(self.data['Time'].dt.to_period('M'))['Value'].agg(['sum', 'count'])
        
        ax2_1 = axes[0,1]
        ax2_2 = ax2_1.twinx()
        
        monthly_stats['sum'].plot(kind='bar', ax=ax2_1, color='lightblue', alpha=0.7)
        ax2_1.set_ylabel('Monthly Return', color='blue')
        ax2_1.tick_params(axis='y', labelcolor='blue')
        ax2_1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        monthly_stats['count'].plot(kind='line', ax=ax2_2, color='red', marker='o', linewidth=2)
        ax2_2.set_ylabel('Trade Count', color='red')
        ax2_2.tick_params(axis='y', labelcolor='red')
        
        ax2_1.set_title('Monthly Performance', fontsize=14, fontweight='bold')
        ax2_1.tick_params(axis='x', rotation=45)
        
        # 3. 仓位大小分布
        position_counts = self.data['PositionSize'].value_counts()
        colors = plt.cm.Set3(np.arange(len(position_counts)))
        axes[1,0].pie(position_counts.values, labels=position_counts.index, autopct='%1.1f%%', 
                     colors=colors, startangle=90)
        axes[1,0].set_title('Position Size Distribution', fontsize=14, fontweight='bold')
        
        # 4. 交易对收益贡献
        symbol_contribution = self.data.groupby('Symbol')['Value'].sum().sort_values(ascending=False)
        symbol_contribution.plot(kind='bar', ax=axes[1,1], color='lightcoral')
        axes[1,1].set_title('Return by Trading Pair', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('Total Return')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_detailed_profit_analysis(self, figsize=(16, 12)):
        """详细利润分析 - 补充图表"""
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Detailed Profit Analysis', fontsize=18, fontweight='bold')
        
        # 1. 月度收益波动性分析
        monthly_stats = self.data.groupby(self.data['Time'].dt.to_period('M'))['Value'].agg(['sum', 'mean', 'std']).round(2)
        monthly_avg_return = monthly_stats['mean']
        monthly_volatility = monthly_stats['std'].fillna(0)
        
        ax1_1 = axes[0,0]
        ax1_2 = ax1_1.twinx()
        
        colors_monthly = ['gold' if x > 0 else 'lightcoral' for x in monthly_avg_return]
        monthly_avg_return.plot(kind='bar', ax=ax1_1, color=colors_monthly, alpha=0.8)
        ax1_1.set_ylabel('Average Return per Trade', color='darkgreen')
        ax1_1.tick_params(axis='y', labelcolor='darkgreen')
        ax1_1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        monthly_volatility.plot(kind='line', ax=ax1_2, color='purple', marker='s', linewidth=2, markersize=4)
        ax1_2.set_ylabel('Return Volatility', color='purple')
        ax1_2.tick_params(axis='y', labelcolor='purple')
        
        ax1_1.set_title('Monthly Performance & Volatility', fontsize=14, fontweight='bold')
        ax1_1.tick_params(axis='x', rotation=45)
        
        # 2. 盈亏对比分析
        contribution_data = {
            'Profit': self.data[self.data['Value'] > 0].groupby('PositionSize')['Value'].sum(),
            'Loss': self.data[self.data['Value'] < 0].groupby('PositionSize')['Value'].sum()
        }
        
        contribution_df = pd.DataFrame(contribution_data).fillna(0)
        contribution_df.plot(kind='bar', ax=axes[0,1], color=['lightgreen', 'lightcoral'])
        axes[0,1].set_title('Profit vs Loss by Position Size', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Return Amount')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0,1].legend()
        
        # 3. 仓位规模与收益散点图
        unique_positions = sorted(self.data['PositionSize'].unique())
        color_map = {pos: i for i, pos in enumerate(unique_positions)}
        colors = [color_map[pos] for pos in self.data['PositionSize']]
        
        scatter = axes[1,0].scatter(self.data['AbsValue'], self.data['Value'], 
                                   c=colors, cmap='viridis', alpha=0.6)
        axes[1,0].set_xlabel('Position Size (Absolute Value)')
        axes[1,0].set_ylabel('Return')
        axes[1,0].set_title('Position Size vs Return Scatter', fontsize=14, fontweight='bold')
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 应用智能坐标轴
        self._apply_smart_scale(axes[1,0], self.data['AbsValue'], axis='x')
        
        # 4. 交易对胜率与盈亏比
        symbol_analysis = []
        for symbol in self.data['Symbol'].unique():
            symbol_data = self.data[self.data['Symbol'] == symbol]
            profit_trades = symbol_data[symbol_data['Value'] > 0]
            loss_trades = symbol_data[symbol_data['Value'] < 0]
            
            profit_count = len(profit_trades)
            loss_count = len(loss_trades)
            total_trades = len(symbol_data)
            
            win_rate = profit_count / total_trades * 100 if total_trades > 0 else 0
            avg_profit = profit_trades['Value'].mean() if profit_count > 0 else 0
            avg_loss = abs(loss_trades['Value'].mean()) if loss_count > 0 else 1
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
            
            symbol_analysis.append({
                'Symbol': symbol,
                'Win_Rate': win_rate,
                'Profit_Loss_Ratio': profit_loss_ratio,
                'Total_Trades': total_trades
            })
        
        symbol_df = pd.DataFrame(symbol_analysis).set_index('Symbol')
        
        ax4_1 = axes[1,1]
        ax4_2 = ax4_1.twinx()
        
        symbol_df['Profit_Loss_Ratio'].plot(kind='bar', ax=ax4_1, color='lightseagreen', alpha=0.7)
        ax4_1.set_ylabel('Profit/Loss Ratio', color='darkslategray')
        ax4_1.tick_params(axis='y', labelcolor='darkslategray')
        ax4_1.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        
        symbol_df['Win_Rate'].plot(kind='line', ax=ax4_2, color='orange', marker='o', linewidth=2)
        ax4_2.set_ylabel('Win Rate (%)', color='darkorange')
        ax4_2.tick_params(axis='y', labelcolor='darkorange')
        ax4_2.set_ylim(0, 100)
        
        ax4_1.set_title('P/L Ratio & Win Rate by Symbol', fontsize=14, fontweight='bold')
        ax4_1.tick_params(axis='x', rotation=45)
        
        # 5. 收益分布直方图
        profit_data = self.data[self.data['Value'] > 0]['Value']
        loss_data = self.data[self.data['Value'] < 0]['Value']
        
        axes[2,0].hist([profit_data, loss_data], bins=30, alpha=0.7, 
                      color=['lightgreen', 'lightcoral'], label=['Profit', 'Loss'])
        axes[2,0].set_title('Return Distribution', fontsize=14, fontweight='bold')
        axes[2,0].set_xlabel('Return Amount')
        axes[2,0].set_ylabel('Frequency')
        axes[2,0].legend()
        axes[2,0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 6. 风险收益分析
        position_risk_return = self.data.groupby('PositionSize')['Value'].agg(['mean', 'std']).fillna(0)
        
        colors = ['red', 'orange', 'green', 'blue', 'purple'][:len(position_risk_return)]
        for i, (pos_size, row) in enumerate(position_risk_return.iterrows()):
            axes[2,1].scatter(row['std'], row['mean'], s=150, color=colors[i], alpha=0.7, label=pos_size)
        
        axes[2,1].set_xlabel('Risk (Return Std Dev)')
        axes[2,1].set_ylabel('Expected Return')
        axes[2,1].set_title('Risk-Return by Position Size', fontsize=14, fontweight='bold')
        axes[2,1].legend()
        axes[2,1].grid(True, alpha=0.3)
        axes[2,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[2,1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
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
    
    def plot_quick_overview(self, figsize=(12, 8)):
        """快速概览图表"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Quick Strategy Overview', fontsize=16, fontweight='bold')
        
        # 1. 累计收益
        data_sorted = self.data.sort_values('Time')
        cumulative_pnl = data_sorted['Value'].cumsum()
        axes[0,0].plot(data_sorted['Time'], cumulative_pnl, linewidth=2, color='navy')
        axes[0,0].set_title('Cumulative Returns')
        axes[0,0].set_ylabel('Cumulative Return')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. 仓位分布
        position_counts = self.data['PositionSize'].value_counts()
        colors = plt.cm.Set3(np.arange(len(position_counts)))
        axes[0,1].pie(position_counts.values, labels=position_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0,1].set_title('Position Size Distribution')
        
        # 3. 月度表现
        monthly_stats = data_sorted.groupby(data_sorted['Time'].dt.to_period('M'))['Value'].agg(['count', 'sum'])
        monthly_stats['sum'].plot(kind='bar', ax=axes[1,0], color='lightblue')
        axes[1,0].set_title('Monthly Returns')
        axes[1,0].set_ylabel('Monthly Return')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. 收益分布
        axes[1,1].hist(self.data['Value'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1,1].axvline(self.data['Value'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.data["Value"].mean():.2f}')
        axes[1,1].set_title('Return Distribution')
        axes[1,1].set_xlabel('Single Trade Return')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        axes[1,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig