"""
交易对深度剖析可视化模块
- 创建一个包含多个子图的仪表板，用于对比分析不同交易对的表现。
- 自动识别长尾分布并使用对数刻度优化可视化。
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from order_analyzer import OrderAnalyzer
import os
import numpy as np
from scipy.stats import skew

class SymbolAnalysisVisualizer:
    """
    为不同交易对生成深度剖析仪表板
    """
    def __init__(self, analyzer: OrderAnalyzer):
        if analyzer.processed_data is None or analyzer.processed_data.empty:
            raise ValueError("传入的分析器没有有效的已处理数据。")
        self.df = analyzer.processed_data.copy()
        # 确保 'Time' 是 datetime 类型
        self.df['Time'] = pd.to_datetime(self.df['Time'])

    def _plot_total_profit_comparison(self, ax):
        """ 子图1: 各交易对总利润对比 (横向条形图) """
        profit_by_symbol = self.df.groupby('Symbol')['Value'].sum().sort_values()
        profit_by_symbol.plot(kind='barh', ax=ax, color=np.where(profit_by_symbol > 0, 'g', 'r'))
        ax.set_title('1. Total Profit by Symbol')
        ax.set_xlabel('Total Profit')
        ax.set_ylabel('Symbol')

    def _plot_profit_distribution(self, ax):
        """ 子图2: 交易利润分布 (箱线图) """
        sns.boxplot(data=self.df, x='Value', y='Symbol', ax=ax, orient='h')
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title('2. Profit Distribution by Symbol')
        ax.set_xlabel('Profit per Trade')
        ax.set_ylabel('')

    def _plot_duration_vs_profit(self, ax):
        """ 子图3: 持仓时间 vs. 利润 (散点图) """
        sns.scatterplot(data=self.df, x='Duration', y='Value', hue='Symbol', ax=ax, alpha=0.6)
        
        # 检查并应用对数刻度
        if skew(self.df['Duration'].dropna()) > 2:
            ax.set_xscale('log')
            ax.set_title('3. Holding Time (Log Scale) vs. Profit')
        else:
            ax.set_title('3. Holding Time vs. Profit')
            
        if skew(self.df['Value'].dropna()) > 2:
            ax.set_yscale('symlog', linthresh=100) # 对数刻度，但能处理0和负值
            ax.set_title(ax.get_title() + ' & Profit (Symlog Scale)')

        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Holding Time (hours)')
        ax.set_ylabel('Profit per Trade')

    def _plot_position_size_contribution(self, ax):
        """ 子图4: 仓位大小贡献分析 (堆叠条形图) """
        profit_contrib = self.df.groupby(['Symbol', 'PositionSize'])['Value'].sum().unstack(fill_value=0)
        profit_contrib.plot(kind='bar', stacked=True, ax=ax, cmap='coolwarm')
        ax.set_title('4. Profit Contribution by Position Size')
        ax.set_ylabel('Total Profit')
        ax.tick_params(axis='x', rotation=45)

    def _plot_monthly_performance(self, ax):
        """ 子图5: 月度表现热力图 """
        self.df['Month'] = self.df['Time'].dt.month
        monthly_profit = self.df.groupby(['Symbol', 'Month'])['Value'].mean().unstack()
        sns.heatmap(monthly_profit, ax=ax, cmap='coolwarm', annot=True, fmt=".0f")
        ax.set_title('5. Mean Monthly Profit by Symbol')
        ax.set_xlabel('Month')
        ax.set_ylabel('Symbol')

    def create_dashboard(self, save_plot=False, strategy_name="Strategy"):
        """
        创建并显示/保存完整的“交易对深度剖析仪表板”
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(3, 2) # 3行2列的网格布局

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :]) # 第2行横跨整行
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])
        
        self._plot_total_profit_comparison(ax1)
        self._plot_profit_distribution(ax2)
        self._plot_duration_vs_profit(ax3)
        self._plot_position_size_contribution(ax4)
        self._plot_monthly_performance(ax5)

        fig.suptitle(f'{strategy_name} - Symbol Analysis Dashboard', fontsize=24, weight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        if save_plot:
            filename = f"{strategy_name.lower().replace(' ', '_')}_symbol_analysis_dashboard.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n💾 Dashboard saved: {filename}")
        else:
            print("\n🖥️  Displaying interactive dashboard...")
            plt.show()
            
        return fig

if __name__ == '__main__':
    # --- 配置 ---
    root_dir = os.path.abspath(os.path.join(os.getcwd()))
    csv_file = os.path.join(root_dir, "MACD-long-crypto/MACD-long-crypto-2023-2024-v1.csv")
    strategy_name = "MACD Long Crypto"
    
    try:
        order_analyzer = OrderAnalyzer(csv_file_path=csv_file)
        
        if order_analyzer.processed_data is None or order_analyzer.processed_data.empty:
            print("❌ Could not process order data. Analysis cannot be performed.")
        else:
            symbol_visualizer = SymbolAnalysisVisualizer(order_analyzer)
            
            save_choice = input("Save dashboard chart? (y/n, default n): ").lower()
            should_save_plot = save_choice in ['y', 'yes']
            
            symbol_visualizer.create_dashboard(save_plot=should_save_plot, strategy_name=strategy_name)
            
            print("\n✅ Symbol analysis dashboard generated!")

    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at '{csv_file}'")
    except Exception as e:
        print(f"❌ An unexpected error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()