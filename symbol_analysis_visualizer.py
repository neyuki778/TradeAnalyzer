"""
交易对深度剖析可视化模块 v5.0
- 创建一个包含多个子图的仪表板，用于对比分析不同交易对的表现。
- 自动识别长尾分布并使用对数刻度优化可视化。
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from order_analyzer import OrderAnalyzer
import os
import numpy as np
from scipy.stats import skew
from mpl_toolkits.mplot3d import Axes3D

class SymbolAnalysisVisualizer:
    """
    为不同交易对生成深度剖析仪表板
    """
    def __init__(self, analyzer: OrderAnalyzer):
        if analyzer.processed_data is None or analyzer.processed_data.empty:
            raise ValueError("传入的分析器没有有效的已处理数据。")
        self.analyzer = analyzer # 保存分析器实例
        self.df = analyzer.processed_data.copy()
        self.df['Time'] = pd.to_datetime(self.df['Time'])

    def _plot_profit_loss_by_position(self, ax):
        """ 子图1: 【V8.0】按仓位划分的盈亏堆叠条形图 & 仓位范围百分比标注 """
        df = self.df
        baseline_value = 1_000_000.0
        
        # 从分析器获取仓位价值范围
        position_analysis = self.analyzer.generate_position_analysis()
        value_ranges = position_analysis.get('value_ranges')

        profit_by_pos = df[df['Value'] > 0].groupby(['Symbol', 'PositionSize'])['Value'].sum().unstack(fill_value=0)
        loss_by_pos = df[df['Value'] < 0].groupby(['Symbol', 'PositionSize'])['Value'].sum().unstack(fill_value=0)
        
        all_symbols = sorted(df['Symbol'].unique())
        all_sizes = sorted(df['PositionSize'].unique())
        profit_by_pos = profit_by_pos.reindex(index=all_symbols, columns=all_sizes, fill_value=0)
        loss_by_pos = loss_by_pos.reindex(index=all_symbols, columns=all_sizes, fill_value=0)

        bar_width = 0.4
        index = np.arange(len(all_symbols))

        # --- 绘制盈利条 ---
        bottom_profit = np.zeros(len(all_symbols))
        greens = plt.get_cmap('Greens')
        for i, size in enumerate(profit_by_pos.columns):
            values = profit_by_pos[size]
            # 创建带范围百分比的图例标签
            range_label = ""
            if value_ranges is not None and size in value_ranges.index:
                min_percent = (value_ranges.loc[size, 'min'] / baseline_value) * 100
                max_percent = (value_ranges.loc[size, 'max'] / baseline_value) * 100
                range_label = f" ({min_percent:.1f}%-{max_percent:.1f}%)"

            ax.bar(index - bar_width/2, values, bar_width, bottom=bottom_profit, label=f'Win-{size}{range_label}', color=greens(0.6 + i*0.1))
            for j, value in enumerate(values):
                if value > 0:
                    total_profit = profit_by_pos.iloc[j].sum()
                    percentage = value / total_profit * 100 if total_profit > 0 else 0
                    ax.text(j - bar_width/2, bottom_profit[j] + value / 2, f'{percentage:.0f}%', ha='center', va='center', color='black', fontsize=8)
            bottom_profit += values

        # --- 绘制亏损条 ---
        bottom_loss = np.zeros(len(all_symbols))
        reds = plt.get_cmap('Reds')
        for i, size in enumerate(loss_by_pos.columns):
            values = loss_by_pos[size]
            # 创建带范围百分比的图例标签
            range_label = ""
            if value_ranges is not None and size in value_ranges.index:
                min_percent = (value_ranges.loc[size, 'min'] / baseline_value) * 100
                max_percent = (value_ranges.loc[size, 'max'] / baseline_value) * 100
                range_label = f" ({min_percent:.1f}%-{max_percent:.1f}%)"
            
            ax.bar(index + bar_width/2, values, bar_width, bottom=bottom_loss, label=f'Loss-{size}{range_label}', color=reds(0.6 + i*0.1))
            for j, value in enumerate(values):
                if value < 0:
                    total_loss = loss_by_pos.iloc[j].sum()
                    percentage = value / total_loss * 100 if total_loss < 0 else 0
                    ax.text(j + bar_width/2, bottom_loss[j] + value / 2, f'{percentage:.0f}%', ha='center', va='center', color='black', fontsize=8)
            bottom_loss += values
            
        # --- 恢复并标注平均盈亏比 ---
        avg_win = df[df['Value'] > 0].groupby('Symbol')['Value'].mean()
        avg_loss = abs(df[df['Value'] < 0].groupby('Symbol')['Value'].mean())
        avg_ratio = (avg_win / avg_loss).fillna(0)

        for i, symbol in enumerate(all_symbols):
            total_profit = profit_by_pos.loc[symbol].sum()
            ratio = avg_ratio.get(symbol, 0)
            if total_profit > 0: # 只在有盈利的柱子上标注
                ax.text(i - bar_width/2, total_profit, f' {ratio:.1f}:1', ha='center', va='bottom', color='blue', fontsize=9, weight='bold')

        ax.set_title('1. Profit/Loss Breakdown by Position Size')
        ax.set_ylabel('Amount')
        ax.set_xticks(index)
        ax.set_xticklabels(all_symbols, rotation=45, ha='right')
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

    def _plot_monthly_performance_3d(self, fig):
        """ 子图3: 【V2.0】3D月度表现热力图 """
        self.df['Month'] = self.df['Time'].dt.month
        monthly_profit = self.df.groupby(['Symbol', 'Month'])['Value'].mean().unstack().fillna(0)
        
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        
        X, Y = np.meshgrid(np.arange(monthly_profit.shape[1]), np.arange(monthly_profit.shape[0]))
        Z = monthly_profit.values
        
        # 使用 coolwarm colormap, 0点为白色
        norm = Normalize(vmin=Z.min(), vmax=Z.max())
        cmap = plt.get_cmap('coolwarm')
        colors = cmap(norm(Z))

        # 绘制3D曲面图
        ax.plot_surface(X, Y, Z, facecolors=colors, shade=False) # type: ignore
        
        ax.set_title('3. 3D Monthly Profit Surface')
        ax.set_xticks(np.arange(len(monthly_profit.columns)))
        ax.set_xticklabels(monthly_profit.columns)
        ax.set_yticks(np.arange(len(monthly_profit.index)))
        ax.set_yticklabels(monthly_profit.index)
        ax.set_xlabel('Month')
        ax.set_ylabel('Symbol')
        ax.set_zlabel('Mean Profit') # type: ignore

    def create_dashboard(self, save_plot=False, strategy_name="Strategy"):
        """
        创建并显示/保存 v5.0 仪表板
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(22, 20))
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        
        self._plot_profit_loss_by_position(ax1)
        self._plot_profit_duration_density(ax2)
        
        self._plot_monthly_performance_3d(fig)
        
        axes = fig.get_axes()
        if len(axes) > 3:
            axes[3].set_visible(False)

        fig.suptitle(f'{strategy_name} - Symbol Analysis Dashboard v5.0', fontsize=24, weight='bold')
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