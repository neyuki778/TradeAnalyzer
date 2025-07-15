"""
交易对深度剖析可视化模块 v12.0 (Final)
- 主仪表板新增持仓周期分析子图，用于对比不同交易对在各周期的盈亏表现。
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
    为不同交易对生成深度剖析仪表板和专项分析图
    """
    def __init__(self, analyzer: OrderAnalyzer):
        if analyzer.processed_data is None or analyzer.processed_data.empty:
            raise ValueError("传入的分析器没有有效的已处理数据。")
        self.analyzer = analyzer
        self.df = analyzer.processed_data.copy()
        self.df['Time'] = pd.to_datetime(self.df['Time'])

    def _plot_profit_loss_by_position(self, ax):
        """ 子图1: 按仓位划分的盈亏堆叠条形图 """
        df = self.df
        profit_by_pos = df[df['Value'] > 0].groupby(['Symbol', 'PositionSize'])['Value'].sum().unstack(fill_value=0)
        loss_by_pos = df[df['Value'] < 0].groupby(['Symbol', 'PositionSize'])['Value'].sum().unstack(fill_value=0)
        
        all_symbols = sorted(df['Symbol'].unique())
        all_sizes = sorted(df['PositionSize'].unique())
        profit_by_pos = profit_by_pos.reindex(index=all_symbols, columns=all_sizes, fill_value=0)
        loss_by_pos = loss_by_pos.reindex(index=all_symbols, columns=all_sizes, fill_value=0)

        bar_width = 0.4
        index = np.arange(len(all_symbols))

        bottom_profit = np.zeros(len(all_symbols))
        greens = plt.get_cmap('Greens')
        for i, size in enumerate(profit_by_pos.columns):
            values = profit_by_pos[size]
            ax.bar(index - bar_width/2, values, bar_width, bottom=bottom_profit, label=f'Win-{size}', color=greens(0.6 + i*0.1))
            bottom_profit += values

        bottom_loss = np.zeros(len(all_symbols))
        reds = plt.get_cmap('Reds')
        for i, size in enumerate(loss_by_pos.columns):
            values = loss_by_pos[size]
            ax.bar(index + bar_width/2, values, bar_width, bottom=bottom_loss, label=f'Loss-{size}', color=reds(0.6 + i*0.1))
            bottom_loss += values

        ax.set_title('1. Profit/Loss Breakdown by Position Size')
        ax.set_ylabel('Amount')
        ax.set_xticks(index)
        ax.set_xticklabels(all_symbols, rotation=45, ha='right')
        ax.legend(title='Position Size')
        ax.axhline(0, color='black', linewidth=0.8)

    def _plot_profit_duration_density(self, ax):
        """ 子图2: 利润/持仓时间密度图 + 离群散点 """
        df = self.df
        use_log_x = skew(df['Duration'].dropna()) > 2
        use_symlog_y = skew(df['Value'].dropna()) > 2
        sns.kdeplot(data=df, x='Duration', y='Value', fill=True, thresh=0, levels=10, cmap="mako", ax=ax)
        profit_q = df['Value'].quantile([0.05, 0.95])
        duration_q = df['Duration'].quantile(0.95)
        outliers = df[(df['Value'] < profit_q.iloc[0]) | (df['Value'] > profit_q.iloc[1]) | (df['Duration'] > duration_q)]
        sns.scatterplot(data=outliers, x='Duration', y='Value', hue='Symbol', ax=ax, style='PositionSize', s=50)
        if use_log_x: ax.set_xscale('log')
        if use_symlog_y: ax.set_yscale('symlog', linthresh=100)
        ax.set_title('2. Profit/Duration Density with Outliers')
        ax.set_xlabel(f'Holding Time (hours){" - Log Scale" if use_log_x else ""}')
        ax.set_ylabel(f'Profit per Trade{" - Symlog Scale" if use_symlog_y else ""}')
        ax.axhline(0, color='red', linestyle='--')

    def _plot_duration_analysis_by_symbol(self, ax):
        """ 子图3: 按交易对和持仓周期的盈亏分析 """
        df = self.df.copy()
        bins = [0, 6, 12, 24, 72, 168, np.inf]
        labels = ['<6h', '6-12h', '12-24h', '1-3d', '3-7d', '>7d']
        df['DurationBin'] = pd.cut(df['Duration'], bins=bins, labels=labels, right=False)
        
        duration_profit = df.groupby(['DurationBin', 'Symbol'])['Value'].sum().unstack()
        duration_profit.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title('3. Profit/Loss by Holding Duration and Symbol')
        ax.set_ylabel('Total Profit')
        ax.set_xlabel('Holding Duration')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.legend(title='Symbol')

    def _plot_monthly_performance_3d(self, fig):
        """ 子图4: 3D月度表现热力图 """
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        self.df['Month'] = self.df['Time'].dt.month
        monthly_profit = self.df.groupby(['Symbol', 'Month'])['Value'].mean().unstack().fillna(0)
        
        X, Y = np.meshgrid(np.arange(monthly_profit.shape[1]), np.arange(monthly_profit.shape[0]))
        Z = monthly_profit.values
        
        norm = Normalize(vmin=Z.min(), vmax=Z.max())
        cmap = plt.get_cmap('coolwarm')
        colors = cmap(norm(Z))

        ax.plot_surface(X, Y, Z, facecolors=colors, shade=False) # type: ignore
        
        ax.set_title('4. 3D Monthly Profit Surface')
        ax.set_xticks(np.arange(len(monthly_profit.columns)))
        ax.set_xticklabels(monthly_profit.columns)
        ax.set_yticks(np.arange(len(monthly_profit.index)))
        ax.set_yticklabels(monthly_profit.index)
        ax.set_xlabel('Month')
        ax.set_ylabel('Symbol')
        ax.set_zlabel('Mean Profit') # type: ignore

    def create_dashboard(self, save_plot=False, strategy_name="Strategy"):
        """
        创建并显示/保存主仪表板
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(22, 20))
        
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        
        self._plot_profit_loss_by_position(ax1)
        self._plot_profit_duration_density(ax2)
        self._plot_duration_analysis_by_symbol(ax3)
        
        self._plot_monthly_performance_3d(fig)

        fig.suptitle(f'{strategy_name} - Main Analysis Dashboard', fontsize=24, weight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        if save_plot:
            filename = f"{strategy_name.lower().replace(' ', '_')}_main_dashboard.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n💾 Main dashboard saved: {filename}")
        else:
            print("\n🖥️  Displaying interactive main dashboard...")
            plt.show()
            
        return fig

    def create_position_profit_facet_plot(self, save_plot=False, strategy_name="Strategy"):
        """
        创建分面散点图，揭示不同标的的仓位大小和盈利的关系
        """
        df = self.df.copy()
        df['Outcome'] = np.where(df['Value'] > 0, 'Profit', 'Loss')
        
        use_log_x = skew(df['AbsValue'].dropna()) > 2
        use_symlog_y = skew(df['Value'].dropna()) > 2
        
        g = sns.relplot(
            data=df, x="AbsValue", y="Value", hue="Outcome", col="Symbol",
            col_wrap=3, palette={'Profit': 'g', 'Loss': 'r'}, alpha=0.6,
            facet_kws={'sharex': False, 'sharey': False}
        )
        
        g.fig.suptitle(f'{strategy_name} - Position Size vs. Profit Facet Plot', y=1.03)
        
        for ax in g.axes.flat:
            if use_log_x: ax.set_xscale('log')
            if use_symlog_y: ax.set_yscale('symlog', linthresh=100)
            ax.axhline(0, color='grey', linestyle='--')
            ax.set_xlabel("Position Size (AbsValue)")

        if save_plot:
            filename = f"{strategy_name.lower().replace(' ', '_')}_position_profit_facet.png"
            g.savefig(filename, dpi=300)
            print(f"\n💾 Facet plot saved: {filename}")
        else:
            print("\n🖥️  Displaying interactive facet plot...")
            plt.show()
            
        return g

if __name__ == '__main__':
    # --- 配置 ---
    root_dir = os.path.abspath(os.path.join(os.getcwd()))
    csv_file = os.path.join(root_dir, "MACD-long-crypto/MACD-long-crypto-2023-2024-v2.csv")
    strategy_name = "MACD Long Crypto"
    
    try:
        order_analyzer = OrderAnalyzer(csv_file_path=csv_file)
        
        if order_analyzer.processed_data is None or order_analyzer.processed_data.empty:
            print("❌ Could not process order data. Analysis cannot be performed.")
        else:
            symbol_visualizer = SymbolAnalysisVisualizer(order_analyzer)
            
            save_all = input("Save all generated charts? (y/n, default n): ").lower() in ['y', 'yes']
            
            symbol_visualizer.create_dashboard(save_plot=save_all, strategy_name=strategy_name)
            symbol_visualizer.create_position_profit_facet_plot(save_plot=save_all, strategy_name=strategy_name)
            
            print("\n✅ All charts generated!")

    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at '{csv_file}'")
    except Exception as e:
        print(f"❌ An unexpected error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()