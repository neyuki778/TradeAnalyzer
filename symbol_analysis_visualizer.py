"""
交易对深度剖析可视化模块 v2.0
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
        self.df['Time'] = pd.to_datetime(self.df['Time'])

    def _plot_profit_loss_by_position(self, ax):
        """ 子图1: 【V2.0】按仓位划分的盈亏堆叠条形图 & 平均盈亏比 """
        df = self.df
        
        # 1. 按仓位计算盈利和亏损
        profit_by_pos = df[df['Value'] > 0].groupby(['Symbol', 'PositionSize'])['Value'].sum().unstack(fill_value=0)
        loss_by_pos = df[df['Value'] < 0].groupby(['Symbol', 'PositionSize'])['Value'].sum().unstack(fill_value=0)
        
        # 2. 绘制堆叠条形图
        profit_by_pos.plot(kind='bar', stacked=True, ax=ax, colormap='Greens_r')
        loss_by_pos.plot(kind='bar', stacked=True, ax=ax, colormap='Reds')
        
        # 3. 计算并标注平均盈亏比
        avg_win = df[df['Value'] > 0].groupby('Symbol')['Value'].mean()
        avg_loss = abs(df[df['Value'] < 0].groupby('Symbol')['Value'].mean())
        avg_ratio = (avg_win / avg_loss).fillna(0)

        for i, symbol in enumerate(profit_by_pos.index):
            total_profit = profit_by_pos.loc[symbol].sum()
            ratio = avg_ratio.get(symbol, 0)
            ax.text(i, total_profit, f' {ratio:.1f}:1', ha='center', va='bottom', color='blue', fontsize=9, weight='bold')

        ax.set_title('1. Profit/Loss Breakdown by Position Size')
        ax.set_ylabel('Amount')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Position Size')
        ax.axhline(0, color='black', linewidth=0.8)

    def _plot_profit_duration_density(self, ax):
        """ 子图2: 【保留】利润/持仓时间密度图 + 离群散点 """
        df = self.df
        
        use_log_x = skew(df['Duration'].dropna()) > 2
        use_symlog_y = skew(df['Value'].dropna()) > 2
        
        sns.kdeplot(data=df, x='Duration', y='Value', fill=True, thresh=0, levels=10, cmap="mako", ax=ax)
        
        profit_q = df['Value'].quantile([0.05, 0.95])
        duration_q = df['Duration'].quantile(0.95)
        outliers = df[
            (df['Value'] < profit_q.iloc[0]) | (df['Value'] > profit_q.iloc[1]) |
            (df['Duration'] > duration_q)
        ]
        sns.scatterplot(data=outliers, x='Duration', y='Value', hue='Symbol', ax=ax, style='PositionSize', s=50)
        
        if use_log_x:
            ax.set_xscale('log')
        if use_symlog_y:
            ax.set_yscale('symlog', linthresh=100)
            
        ax.set_title('2. Profit/Duration Density with Outliers')
        ax.set_xlabel(f'Holding Time (hours){" - Log Scale" if use_log_x else ""}')
        ax.set_ylabel(f'Profit per Trade{" - Symlog Scale" if use_symlog_y else ""}')
        ax.axhline(0, color='red', linestyle='--')

    def _plot_monthly_performance(self, ax):
        """ 子图3: 【保留】月度表现热力图 """
        self.df['Month'] = self.df['Time'].dt.month
        monthly_profit = self.df.groupby(['Symbol', 'Month'])['Value'].mean().unstack()
        sns.heatmap(monthly_profit, ax=ax, cmap='coolwarm', annot=True, fmt=".0f")
        ax.set_title('3. Mean Monthly Profit by Symbol')
        ax.set_xlabel('Month')
        ax.set_ylabel('Symbol')

    def create_dashboard(self, save_plot=False, strategy_name="Strategy"):
        """
        创建并显示/保存 v3.0 仪表板
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        
        # 重新安排布局
        self._plot_profit_loss_by_position(axes[0, 0])
        self._plot_profit_duration_density(axes[0, 1])
        self._plot_monthly_performance(axes[1, 0])
        
        # 隐藏右下角未使用的子图
        axes[1, 1].set_visible(False)

        fig.suptitle(f'{strategy_name} - Symbol Analysis Dashboard v3.0', fontsize=24, weight='bold')
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