"""
仓位大小细分可视化器
专门用于展示不同仓位大小区间的盈亏分布情况
x轴：仓位大小（细分成10+个区间）
y轴：盈亏（win在上，loss在下）
支持多交易对分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from order_analyzer import OrderAnalyzer
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

DEFAULT_CASH = 1_000_000.0

class PositionSizeDetailVisualizer:
    """
    仓位大小细分可视化器
    """
    
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
        
        print(f"📊 数据概览:")
        print(f"   总订单数: {len(self.data)}")
        print(f"   交易对数: {len(self.data['Symbol'].unique())}")
        print(f"   仓位大小范围: {self.data['AbsValue'].min():,.0f} - {self.data['AbsValue'].max():,.0f}")
    
    def create_position_size_bins(self, n_bins=12):
        """
        创建仓位大小区间
        
        Args:
            n_bins (int): 区间数量，默认12个
            
        Returns:
            dict: 包含区间标签和边界的字典
        """
        # 使用对数分箱，因为仓位大小通常有长尾分布
        log_values = np.log10(self.data['AbsValue'] + 1)  # +1避免log(0)
        
        # 创建等宽的对数区间
        log_bins = np.linspace(log_values.min(), log_values.max(), n_bins + 1)
        
        # 转换回原始值
        bins = 10 ** log_bins - 1       
        
        # 创建区间标签
        labels = []
        for i in range(len(bins) - 1):
            if bins[i] < 1000:
                left = f"{bins[i]:.0f}"
            elif bins[i] < 1000000:
                left = f"{bins[i]/1000:.1f}K"
            else:
                left = f"{bins[i]/1000000:.1f}M"
                
            if bins[i+1] < 1000:
                right = f"{bins[i+1]:.0f}"
            elif bins[i+1] < 1000000:
                right = f"{bins[i+1]/1000:.1f}K"
            else:
                right = f"{bins[i+1]/1000000:.1f}M"
            
            labels.append(f"{left}-{right}")
        
        return {
            'bins': bins,
            'labels': labels ,
            'n_bins': n_bins
        }
    
    def plot_position_size_detail(self, figsize=(20, 16), n_bins=12):
        """
        绘制仓位大小细分图表
        
        Args:
            figsize (tuple): 图表大小
            n_bins (int): 区间数量
        """
        # 创建仓位大小区间
        bin_info = self.create_position_size_bins(n_bins)
        
        # 为数据添加区间标签
        data_with_bins = self.data.copy()
        data_with_bins['PositionBin'] = pd.cut(
            data_with_bins['AbsValue'], 
            bins=bin_info['bins'], 
            labels=bin_info['labels'],
            include_lowest=True
        )
        
        # 分离盈利和亏损数据
        profit_data = data_with_bins[data_with_bins['Value'] > 0]
        loss_data = data_with_bins[data_with_bins['Value'] < 0]
        
        # 创建图表 - 改为3个子图布局
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 2], hspace=0.3, wspace=0.3)
        
        fig.suptitle('Position Size Detail Analysis', fontsize=20, fontweight='bold')
        
        # === 子图1: 总体盈亏分布 + 盈亏比 ===
        ax1 = fig.add_subplot(gs[0, 0])
        
        # 按区间统计盈亏
        profit_by_bin = profit_data.groupby('PositionBin')['Value'].sum()
        loss_by_bin = loss_data.groupby('PositionBin')['Value'].sum()
        
        # 确保所有区间都有数据（填充0）
        all_bins = bin_info['labels']
        profit_by_bin = profit_by_bin.reindex(all_bins, fill_value=0)
        loss_by_bin = loss_by_bin.reindex(all_bins, fill_value=0)
        
        x_pos = np.arange(len(all_bins))
        
        # 绘制盈利条形图（向上）
        ax1.bar(x_pos, profit_by_bin.values, 
                color='lightgreen', alpha=0.8, label='Profit')
        
        # 绘制亏损条形图（向下）
        ax1.bar(x_pos, loss_by_bin.values, 
                color='lightcoral', alpha=0.8, label='Loss')
        
        ax1.set_title('Total P&L by Position Size', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Position Size Range')
        ax1.set_ylabel('Total P&L')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(all_bins, rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签和盈亏比
        for i, (profit, loss) in enumerate(zip(profit_by_bin.values, loss_by_bin.values)):
            if profit > 0:
                ax1.text(i, profit, f'{profit:,.0f}', ha='center', va='bottom', fontsize=8)
            if loss < 0:
                ax1.text(i, loss, f'{loss:,.0f}', ha='center', va='top', fontsize=8)
                
            # 计算并显示盈亏比
            if profit > 0 and loss < 0:
                ratio = abs(profit / loss)
                ax1.text(i, max(profit, abs(loss)) * 1.1, f'{ratio:.1f}:1', 
                        ha='center', va='bottom', fontsize=8, color='blue', fontweight='bold')
        
        # === 子图2: 综合统计 - 订单数、胜率和盈亏比 ===
        ax2 = fig.add_subplot(gs[0, 1])
        
        # 按区间统计各种指标
        combined_stats = []
        for bin_label in all_bins:
            bin_data = data_with_bins[data_with_bins['PositionBin'] == bin_label]
            if len(bin_data) == 0:
                combined_stats.append({
                    'bin': bin_label,
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_loss_ratio': 0
                })
                continue
                
            profit_trades = bin_data[bin_data['Value'] > 0]
            loss_trades = bin_data[bin_data['Value'] < 0]
            
            total_trades = len(bin_data)
            win_rate = len(profit_trades) / total_trades * 100 if total_trades > 0 else 0
            
            # 计算盈亏比
            avg_profit = profit_trades['Value'].mean() if len(profit_trades) > 0 else 0
            avg_loss = abs(loss_trades['Value'].mean()) if len(loss_trades) > 0 else 1
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
            
            combined_stats.append({
                'bin': bin_label,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio
            })
        
        # 绘制三个指标
        x_pos = np.arange(len(all_bins))
        
        # 主轴：交易数量
        ax2_1 = ax2
        trade_counts = [stat['total_trades'] for stat in combined_stats]
        bars_trades = ax2_1.bar(x_pos, trade_counts, alpha=0.6, color='lightblue', label='Trade Count')
        ax2_1.set_ylabel('Trade Count', color='blue')
        ax2_1.tick_params(axis='y', labelcolor='blue')
        
        # 右轴1：胜率
        ax2_2 = ax2_1.twinx()
        win_rates = [stat['win_rate'] for stat in combined_stats]
        line_winrate = ax2_2.plot(x_pos, win_rates, color='green', marker='o', linewidth=2, 
                                  markersize=4, label='Win Rate')
        ax2_2.set_ylabel('Win Rate (%)', color='green')
        ax2_2.tick_params(axis='y', labelcolor='green')
        ax2_2.set_ylim(0, 100)
        
        # 右轴2：盈亏比
        ax2_3 = ax2_1.twinx()
        ax2_3.spines['right'].set_position(('outward', 60))
        profit_loss_ratios = [stat['profit_loss_ratio'] for stat in combined_stats]
        line_ratio = ax2_3.plot(x_pos, profit_loss_ratios, color='red', marker='s', linewidth=2, 
                                markersize=4, label='P/L Ratio')
        ax2_3.set_ylabel('Profit/Loss Ratio', color='red')
        ax2_3.tick_params(axis='y', labelcolor='red')
        ax2_3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        ax2_1.set_title('Trade Stats: Count, Win Rate & P/L Ratio', fontsize=14, fontweight='bold')
        ax2_1.set_xlabel('Position Size Range')
        ax2_1.set_xticks(x_pos)
        ax2_1.set_xticklabels(all_bins, rotation=45, ha='right')
        ax2_1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, stat in enumerate(combined_stats):
            if stat['total_trades'] > 0:
                ax2_1.text(i, stat['total_trades'], f"{stat['total_trades']}", 
                          ha='center', va='bottom', fontsize=8)
            if stat['win_rate'] > 0:
                ax2_2.text(i, stat['win_rate'], f"{stat['win_rate']:.0f}%", 
                          ha='center', va='bottom', fontsize=8, color='green')
            if stat['profit_loss_ratio'] > 0:
                ax2_3.text(i, stat['profit_loss_ratio'], f"{stat['profit_loss_ratio']:.1f}", 
                          ha='center', va='bottom', fontsize=8, color='red')
        
        # === 子图3: 多交易对分析 - 占据整个下半部分 ===
        ax3 = fig.add_subplot(gs[1:, :])
        
        # 按交易对和区间统计盈亏
        symbol_profit = profit_data.groupby(['Symbol', 'PositionBin'])['Value'].sum().unstack(fill_value=0)
        symbol_loss = loss_data.groupby(['Symbol', 'PositionBin'])['Value'].sum().unstack(fill_value=0)
        
        # 确保所有区间都包含在内
        symbol_profit = symbol_profit.reindex(columns=all_bins, fill_value=0)
        symbol_loss = symbol_loss.reindex(columns=all_bins, fill_value=0)
        
        # 计算净盈亏
        symbol_net = symbol_profit + symbol_loss  # loss已经是负数
        
        # 绘制热力图
        im = ax3.imshow(symbol_net.values, cmap='RdYlGn', aspect='auto')
        
        # 设置标签
        ax3.set_title('Net P&L Heatmap by Symbol & Position Size', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Position Size Range', fontsize=12)
        ax3.set_ylabel('Trading Symbol', fontsize=12)
        ax3.set_xticks(range(len(all_bins)))
        ax3.set_xticklabels(all_bins, rotation=45, ha='right')
        ax3.set_yticks(range(len(symbol_net.index)))
        ax3.set_yticklabels(symbol_net.index)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Net P&L', rotation=270, labelpad=15)
        
        # 在每个格子中添加数值 - 使用黑色字体
        for i in range(len(symbol_net.index)):
            for j in range(len(all_bins)):
                value = symbol_net.iloc[i, j]
                if abs(value) > 0:
                    ax3.text(j, i, f'{value:,.0f}', ha='center', va='center', 
                            fontsize=10, color='black', fontweight='bold')
        
        return fig
    
    def plot_simplified_view(self, figsize=(16, 8), n_bins=15):
        """
        绘制简化版仓位大小分析图
        
        Args:
            figsize (tuple): 图表大小
            n_bins (int): 区间数量
        """
        # 创建仓位大小区间
        bin_info = self.create_position_size_bins(n_bins)
        
        # 为数据添加区间标签
        data_with_bins = self.data.copy()
        data_with_bins['PositionBin'] = pd.cut(
            data_with_bins['AbsValue'], 
            bins=bin_info['bins'], 
            labels=bin_info['labels'],
            include_lowest=True
        )
        
        # 分离盈利和亏损数据
        profit_data = data_with_bins[data_with_bins['Value'] > 0]
        loss_data = data_with_bins[data_with_bins['Value'] < 0]
        
        # 创建单一图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 按区间统计盈亏
        profit_by_bin = profit_data.groupby('PositionBin')['Value'].sum()
        loss_by_bin = loss_data.groupby('PositionBin')['Value'].sum()
        
        # 确保所有区间都有数据（填充0）
        all_bins = bin_info['labels']
        profit_by_bin = profit_by_bin.reindex(all_bins, fill_value=0)
        loss_by_bin = loss_by_bin.reindex(all_bins, fill_value=0)
        
        x_pos = np.arange(len(all_bins))
        
        # 绘制盈利条形图（向上）
        bars1 = ax.bar(x_pos, profit_by_bin.values, 
                      color='lightgreen', alpha=0.8, label='Profit', width=0.8)
        
        # 绘制亏损条形图（向下）
        bars2 = ax.bar(x_pos, loss_by_bin.values, 
                      color='lightcoral', alpha=0.8, label='Loss', width=0.8)
        
        ax.set_title('Profit & Loss Distribution by Position Size Range', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Position Size Range', fontsize=12)
        ax.set_ylabel('Total P&L', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_bins, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (profit, loss) in enumerate(zip(profit_by_bin.values, loss_by_bin.values)):
            if profit > 0:
                ax.text(i, profit, f'{profit:,.0f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
            if loss < 0:
                ax.text(i, loss, f'{loss:,.0f}', ha='center', va='top', 
                       fontsize=9, fontweight='bold')
        
        # 添加统计信息
        total_profit = profit_by_bin.sum()
        total_loss = loss_by_bin.sum()
        net_profit = total_profit + total_loss
        
        stats_text = f"""总统计:
总盈利: {total_profit:,.0f}
总亏损: {total_loss:,.0f}
净盈利: {net_profit:,.0f}
盈亏比: {abs(total_profit/total_loss):.2f}:1"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def save_charts(self, prefix="position_detail", n_bins=12):
        """
        保存图表
        
        Args:
            prefix (str): 文件名前缀
            n_bins (int): 区间数量
        """
        # 保存详细分析图
        fig1 = self.plot_position_size_detail(n_bins=n_bins)
        filename1 = f"{prefix}_detail_analysis.png"
        fig1.savefig(filename1, dpi=300, bbox_inches='tight')
        
        # 保存简化版图
        fig2 = self.plot_simplified_view(n_bins=n_bins)
        filename2 = f"{prefix}_simplified.png"
        fig2.savefig(filename2, dpi=300, bbox_inches='tight')
        
        print(f"📊 已保存图表:")
        print(f"  - {filename1}")
        print(f"  - {filename2}")
        
        return [filename1, filename2]
    
    def print_position_summary(self, n_bins=12):
        """
        打印仓位大小汇总信息
        
        Args:
            n_bins (int): 区间数量
        """
        # 创建仓位大小区间
        bin_info = self.create_position_size_bins(n_bins)
        
        # 为数据添加区间标签
        data_with_bins = self.data.copy()
        data_with_bins['PositionBin'] = pd.cut(
            data_with_bins['AbsValue'], 
            bins=bin_info['bins'], 
            labels=bin_info['labels'],
            include_lowest=True
        )
        
        print(f"\n📈 仓位大小细分汇总 (共{n_bins}个区间):")
        print("=" * 80)
        
        # 按区间统计
        summary = data_with_bins.groupby('PositionBin').agg({
            'Value': ['sum', 'mean', 'count'],
            'AbsValue': ['min', 'max', 'mean']
        }).round(2)
        
        # 计算盈亏分布
        for bin_label in bin_info['labels']:
            bin_data = data_with_bins[data_with_bins['PositionBin'] == bin_label]
            if len(bin_data) == 0:
                continue
                
            profit_trades = bin_data[bin_data['Value'] > 0]
            loss_trades = bin_data[bin_data['Value'] < 0]
            
            total_pnl = bin_data['Value'].sum()
            total_trades = len(bin_data)
            win_rate = len(profit_trades) / total_trades * 100 if total_trades > 0 else 0
            
            print(f"\n📊 {bin_label}:")
            print(f"   交易数量: {total_trades}")
            print(f"   总盈亏: {total_pnl:,.0f}")
            print(f"   胜率: {win_rate:.1f}%")
            print(f"   平均盈亏: {bin_data['Value'].mean():,.0f}")
            if len(profit_trades) > 0:
                print(f"   平均盈利: {profit_trades['Value'].mean():,.0f}")
            if len(loss_trades) > 0:
                print(f"   平均亏损: {loss_trades['Value'].mean():,.0f}")

def main():
    """主函数 - 示例用法"""
    import os
    
    # 数据文件路径
    root_dir = os.path.abspath(os.path.join(os.getcwd()))
    csv_file = os.path.join(root_dir, "MACD-long-crypto/2023-2024/HOUR/base.csv")
    
    try:
        # 创建分析器
        analyzer = OrderAnalyzer(csv_file)
        
        if analyzer.processed_data is None or len(analyzer.processed_data) == 0:
            print("❌ 没有找到有效的交易数据")
            return
        
        # 创建可视化器
        visualizer = PositionSizeDetailVisualizer(analyzer)
        
        # 打印汇总信息
        visualizer.print_position_summary()
        
        # 询问用户选择
        choice = input("\n请选择操作 (1=详细分析图, 2=简化图, 3=保存图表, 4=显示所有): ")
        
        if choice == "1":
            visualizer.plot_position_size_detail()
            plt.show()
        elif choice == "2":
            visualizer.plot_simplified_view()
            plt.show()
        elif choice == "3":
            visualizer.save_charts()
        elif choice == "4":
            visualizer.plot_position_size_detail()
            visualizer.plot_simplified_view()
            plt.show()
        else:
            print("显示简化图...")
            visualizer.plot_simplified_view()
            plt.show()
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()