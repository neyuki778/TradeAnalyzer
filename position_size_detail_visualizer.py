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
        self.data = analyzer.processed_data.copy()
        
        # 检查数据是否有效
        if self.data is None or len(self.data) == 0:
            raise ValueError("分析器中没有有效的处理数据")
        
        # 处理相对仓位大小
        self._process_relative_position_size()
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        
        print(f"📊 数据概览:")
        print(f"   总订单数: {len(self.data)}")
        print(f"   交易对数: {len(self.data['Symbol'].unique())}")
        print(f"   相对仓位大小范围: {self.data['RelativePositionSize'].min():.2%} - {self.data['RelativePositionSize'].max():.2%}")
    
    
    def _process_relative_position_size(self):
        """
        处理相对仓位大小计算
        从tag中提取投资组合总价值，计算下单价值/总价值的相对仓位
        """
        def parse_portfolio_value(tag):
            """从tag中解析投资组合总价值"""
            if pd.isna(tag) or tag == 'Liquidated':
                return None
            
            # 移除逗号和引号，转换为数字
            try:
                # 处理带引号的格式，如 "1,000,000.00"
                if isinstance(tag, str):
                    cleaned_tag = tag.strip('"').replace(',', '')
                    return float(cleaned_tag)
                return float(tag)
            except (ValueError, TypeError):
                return None
        
        # 解析投资组合价值
        self.data['PortfolioValue'] = self.data['Tag'].apply(parse_portfolio_value)
        
        # 调试信息：显示tag解析情况
        total_tags = len(self.data)
        valid_tags = self.data['PortfolioValue'].notna().sum()
        liquidated_tags = (self.data['Tag'] == 'Liquidated').sum()
        empty_tags = self.data['Tag'].isna().sum()
        
        print(f"📊 Tag解析调试信息:")
        print(f"   总记录数: {total_tags}")
        print(f"   有效投资组合价值: {valid_tags}")
        print(f"   Liquidated标签: {liquidated_tags}")
        print(f"   空标签: {empty_tags}")
        
        # 过滤掉无法解析投资组合价值的数据
        valid_data = self.data.dropna(subset=['PortfolioValue'])
        
        if len(valid_data) == 0:
            raise ValueError("无法从Tag中解析出有效的投资组合价值")
        
        # 计算相对仓位大小（使用AbsValue作为下单价值）
        valid_data['RelativePositionSize'] = valid_data['AbsValue'] / valid_data['PortfolioValue']
        
        # 调试信息：显示相对仓位分布
        print(f"📊 相对仓位分布调试:")
        print(f"   最小相对仓位: {valid_data['RelativePositionSize'].min():.2%}")
        print(f"   最大相对仓位: {valid_data['RelativePositionSize'].max():.2%}")
        print(f"   平均相对仓位: {valid_data['RelativePositionSize'].mean():.2%}")
        print(f"   中位相对仓位: {valid_data['RelativePositionSize'].median():.2%}")
        
        # 显示一些大仓位的详细信息
        large_positions = valid_data[valid_data['RelativePositionSize'] > 0.5]
        if len(large_positions) > 0:
            print(f"\n⚠️  发现 {len(large_positions)} 个超过50%的大仓位:")
            for i, (_, row) in enumerate(large_positions.head(5).iterrows()):
                print(f"   {i+1}. {row['Time']}: {row['Symbol']}, 仓位: {row['AbsValue']:,.0f}, 组合: {row['PortfolioValue']:,.0f}, 比例: {row['RelativePositionSize']:.2%}, Tag: '{row['Tag']}'")
        
        # 更新数据
        self.data = valid_data
        
        print(f"📊 相对仓位处理完成:")
        print(f"   有效数据: {len(self.data)} 条")
        print(f"   投资组合价值范围: {self.data['PortfolioValue'].min():,.0f} - {self.data['PortfolioValue'].max():,.0f}")
    
    def create_quantile_bins(self, n_bins=25):
        """
        创建基于分位数的相对仓位大小区间（用于简略图，确保每个区间有相对均匀的数据分布）
        
        Args:
            n_bins (int): 区间数量，默认25个
            
        Returns:
            dict: 包含区间标签和边界的字典
        """
        # 使用分位数创建区间，确保每个区间有大致相同数量的数据点
        relative_sizes = self.data['RelativePositionSize']
        
        # 创建分位数
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = relative_sizes.quantile(quantiles).values
        
        # 确保bins是单调递增的（处理重复值）
        bins = np.unique(bins)
        if len(bins) < n_bins + 1:
            # 如果有重复值导致区间减少，回退到等宽区间
            print(f"⚠️  数据重复值较多，回退到等宽区间")
            return self.create_position_size_bins(n_bins)
        
        # 创建区间标签（百分比格式）
        labels = []
        for i in range(len(bins) - 1):
            left_pct = bins[i] * 100
            right_pct = bins[i+1] * 100
            
            # 格式化百分比标签
            if left_pct < 1:
                left_label = f"{left_pct:.2f}%"
            elif left_pct < 10:
                left_label = f"{left_pct:.1f}%"
            else:
                left_label = f"{left_pct:.0f}%"
                
            if right_pct < 1:
                right_label = f"{right_pct:.2f}%"
            elif right_pct < 10:
                right_label = f"{right_pct:.1f}%"
            else:
                right_label = f"{right_pct:.0f}%"
            
            labels.append(f"{left_label}-{right_label}")
        
        return {
            'bins': bins,
            'labels': labels,
            'n_bins': len(bins) - 1
        }
    
    def create_fixed_range_bins(self, n_bins=25):
        """
        创建固定0%-100%范围的相对仓位大小区间（用于简略图）
        
        Args:
            n_bins (int): 区间数量，默认25个
            
        Returns:
            dict: 包含区间标签和边界的字典
        """
        # 设置固定的范围：0% 到 100%
        min_size = 0.0   # 0%
        max_size = 1.0   # 100%
        
        # 创建等宽区间（基于百分比）
        bins = np.linspace(min_size, max_size, n_bins + 1)
        
        # 创建区间标签（百分比格式）
        labels = []
        for i in range(len(bins) - 1):
            left_pct = bins[i] * 100
            right_pct = bins[i+1] * 100
            
            # 格式化百分比标签（简化版本，减少标签长度）
            left_label = f"{left_pct:.0f}%"
            right_label = f"{right_pct:.0f}%"
            
            labels.append(f"{left_label}-{right_label}")
        
        return {
            'bins': bins,
            'labels': labels,
            'n_bins': n_bins
        }
    
    def create_position_size_bins(self, n_bins=12):
        """
        创建相对仓位大小区间
        
        Args:
            n_bins (int): 区间数量，默认12个
            
        Returns:
            dict: 包含区间标签和边界的字典
        """
        # 使用相对仓位大小进行分箱
        relative_sizes = self.data['RelativePositionSize']
        
        # 创建等宽区间（基于百分比）
        min_size = relative_sizes.min()
        max_size = relative_sizes.max()
        bins = np.linspace(min_size, max_size, n_bins + 1)
        
        # 创建区间标签（百分比格式）
        labels = []
        for i in range(len(bins) - 1):
            left_pct = bins[i] * 100
            right_pct = bins[i+1] * 100
            
            # 格式化百分比标签
            if left_pct < 1:
                left_label = f"{left_pct:.2f}%"
            elif left_pct < 10:
                left_label = f"{left_pct:.1f}%"
            else:
                left_label = f"{left_pct:.0f}%"
                
            if right_pct < 1:
                right_label = f"{right_pct:.2f}%"
            elif right_pct < 10:
                right_label = f"{right_pct:.1f}%"
            else:
                right_label = f"{right_pct:.0f}%"
            
            labels.append(f"{left_label}-{right_label}")
        
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
        
        # 为数据添加区间标签（使用相对仓位大小）
        data_with_bins = self.data.copy()
        data_with_bins['PositionBin'] = pd.cut(
            data_with_bins['RelativePositionSize'], 
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
        
        ax1.set_title('Total P&L by Relative Position Size', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Relative Position Size Range (%)')
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
        ax2_1.set_xlabel('Relative Position Size Range (%)')
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
        ax3.set_title('Net P&L Heatmap by Symbol & Relative Position Size', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Relative Position Size Range (%)', fontsize=12)
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
    
    def plot_simplified_view(self, figsize=(20, 10), n_bins=25):
        """
        绘制简化版仓位大小分析图
        
        Args:
            figsize (tuple): 图表大小
            n_bins (int): 区间数量
        """
        # 创建基于分位数的区间，确保数据分布更均匀
        bin_info = self.create_quantile_bins(n_bins)
        
        # 为数据添加区间标签（使用相对仓位大小）
        data_with_bins = self.data.copy()
        data_with_bins['PositionBin'] = pd.cut(
            data_with_bins['RelativePositionSize'], 
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
        
        ax.set_title('Profit & Loss Distribution by Relative Position Size Range', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Relative Position Size Range (%)', fontsize=12)
        ax.set_ylabel('Total P&L', fontsize=12)
        ax.set_xticks(x_pos)
        # 只显示部分标签以避免过度拥挤
        step = max(1, len(all_bins) // 10)  # 最多显示10个标签
        xtick_labels = [all_bins[i] if i % step == 0 else '' for i in range(len(all_bins))]
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax.grid(True, alpha=0.3)
        
        # 预先调整y轴范围，为标签预留空间
        current_ylim = ax.get_ylim()
        y_range = current_ylim[1] - current_ylim[0]
        ax.set_ylim(current_ylim[0] - y_range * 0.15, current_ylim[1] + y_range * 0.2)
        
        # 创建右侧y轴用于盈亏比折线图
        ax2 = ax.twinx()
        
        # 添加盈亏金额标签和订单数，收集盈亏比数据
        profit_loss_ratios = []
        for i, (profit, loss) in enumerate(zip(profit_by_bin.values, loss_by_bin.values)):
            # 计算该区间的统计数据
            bin_label = all_bins[i]
            bin_data = data_with_bins[data_with_bins['PositionBin'] == bin_label]
            
            # 显示盈亏金额
            if profit > 0:
                ax.text(i, profit, f'{profit:,.0f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
            if loss < 0:
                ax.text(i, loss, f'{loss:,.0f}', ha='center', va='top', 
                       fontsize=9, fontweight='bold')
            
            # 计算盈亏比用于折线图
            if profit > 0 and loss < 0:
                ratio = abs(profit / loss)
                profit_loss_ratios.append(ratio)
            else:
                profit_loss_ratios.append(0)  # 没有完整盈亏对的设为0
            
            # 订单数（在x轴下方）
            if len(bin_data) > 0:
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                orders_y = y_min + y_range * 0.06
                ax.text(i, orders_y, f'{len(bin_data)}', 
                       ha='center', va='bottom', fontsize=8, color='gray', fontweight='bold')
        
        # 绘制盈亏比折线图
        x_pos_line = np.arange(len(all_bins))
        ax2.plot(x_pos_line, profit_loss_ratios, color='blue', marker='o', linewidth=2, 
                markersize=4, label='P/L Ratio', alpha=0.8)
        
        # 设置右侧y轴
        ax2.set_ylabel('Profit/Loss Ratio', color='blue', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.axhline(y=1, color='blue', linestyle='--', alpha=0.5)  # 盈亏平衡线
        
        # 在折线图数据点上显示具体的盈亏比值
        for i, ratio in enumerate(profit_loss_ratios):
            if ratio > 0:  # 只显示有效的盈亏比
                ax2.text(i, ratio, f'{ratio:.1f}', ha='center', va='bottom', 
                        fontsize=8, color='blue', fontweight='bold')
        
        # 创建组合图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
        
        # 添加统计信息（英文）
        total_profit = profit_by_bin.sum()
        total_loss = loss_by_bin.sum()
        net_profit = total_profit + total_loss
        
        stats_text = f"""Summary Statistics:
Total Profit: {total_profit:,.0f}
Total Loss: {total_loss:,.0f}
Net Profit: {net_profit:,.0f}
P/L Ratio: {abs(total_profit/total_loss):.2f}:1"""
        
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
        
        # 为数据添加区间标签（使用相对仓位大小）
        data_with_bins = self.data.copy()
        data_with_bins['PositionBin'] = pd.cut(
            data_with_bins['RelativePositionSize'], 
            bins=bin_info['bins'], 
            labels=bin_info['labels'],
            include_lowest=True
        )
        
        print(f"\n📈 相对仓位大小细分汇总 (共{n_bins}个区间):")
        print("=" * 80)
        
        # 按区间统计
        summary = data_with_bins.groupby('PositionBin').agg({
            'Value': ['sum', 'mean', 'count'],
            'RelativePositionSize': ['min', 'max', 'mean']
        }).round(4)
        
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
    csv_file = os.path.join(root_dir, "Multi EMA Crypto/2023-2024/base.csv")
    
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