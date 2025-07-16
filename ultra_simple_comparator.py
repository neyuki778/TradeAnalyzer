"""
最简单的仓位对比器 - 返璞归真
只做三件事：
1. 找到相同信号的订单（时间戳+品种+方向相同）
2. 缩放到一致比例
3. 画散点图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class UltraSimpleComparator:
    """
    超简单比较器
    只做最核心的事情：找相同信号，缩放，画图
    """
    
    def __init__(self, high_pos_csv, low_pos_csv):
        self.high_data = pd.read_csv(high_pos_csv)
        self.low_data = pd.read_csv(low_pos_csv)
        
        # 转换时间格式
        self.high_data['Time'] = pd.to_datetime(self.high_data['Time'])
        self.low_data['Time'] = pd.to_datetime(self.low_data['Time'])
        
        print(f"📂 大仓位数据: {len(self.high_data)} 条订单")
        print(f"📂 小仓位数据: {len(self.low_data)} 条订单")
    
    def find_matching_orders(self):
        """第一步：找到相同信号的订单"""
        print("\n🔍 寻找相同信号的订单...")
        
        matched_pairs = []
        
        for _, high_order in self.high_data.iterrows():
            # 寻找完全匹配的订单：时间戳、品种、方向都相同
            matches = self.low_data[
                (self.low_data['Time'] == high_order['Time']) &
                (self.low_data['Symbol'] == high_order['Symbol']) &
                (np.sign(self.low_data['Quantity']) == np.sign(high_order['Quantity']))
            ]
            
            if not matches.empty:
                low_order = matches.iloc[0]  # 取第一个匹配
                
                matched_pairs.append({
                    'Time': high_order['Time'],
                    'Symbol': high_order['Symbol'],
                    'Direction': 'Buy' if high_order['Quantity'] > 0 else 'Sell',
                    
                    'High_Quantity': abs(high_order['Quantity']),
                    'High_Value': abs(high_order['Value']),
                    'High_Status': high_order['Status'],
                    
                    'Low_Quantity': abs(low_order['Quantity']),
                    'Low_Value': abs(low_order['Value']),
                    'Low_Status': low_order['Status'],
                })
        
        self.matched_orders = pd.DataFrame(matched_pairs)
        print(f"✅ 找到 {len(self.matched_orders)} 个相同信号的订单对")
        
        if len(self.matched_orders) == 0:
            print("❌ 没有找到匹配的订单")
            return False
        
        return True
    
    def calculate_scaling(self):
        """第二步：计算缩放比例"""
        print("\n📏 计算缩放比例...")
        
        # 使用有效订单计算缩放比例
        valid_pairs = self.matched_orders[
            (self.matched_orders['High_Status'] == 'Filled') & 
            (self.matched_orders['Low_Status'] == 'Filled')
        ]
        
        if len(valid_pairs) == 0:
            print("❌ 没有有效的订单对用于计算缩放比例")
            return False
        
        # 计算每对订单的缩放比例
        scaling_ratios = []
        for _, pair in valid_pairs.iterrows():
            if pair['Low_Quantity'] > 0:
                ratio = pair['High_Quantity'] / pair['Low_Quantity']
                scaling_ratios.append(ratio)
        
        if not scaling_ratios:
            print("❌ 无法计算缩放比例")
            return False
        
        self.scaling_factor = np.median(scaling_ratios)  # 使用中位数更稳健
        
        print(f"📊 缩放统计:")
        print(f"   中位数缩放比例: {self.scaling_factor:.2f}x")
        print(f"   平均缩放比例: {np.mean(scaling_ratios):.2f}x")
        print(f"   标准差: {np.std(scaling_ratios):.2f}")
        print(f"   基于 {len(scaling_ratios)} 个有效订单对")
        
        # 应用缩放到所有订单
        self.matched_orders['High_Scaled_Quantity'] = self.matched_orders['High_Quantity'].astype(float) / self.scaling_factor
        self.matched_orders['High_Scaled_Value'] = self.matched_orders['High_Value'].astype(float) / self.scaling_factor
        
        return True
    
    def plot_scatter(self, save_plot=False):
        """第三步：画散点图 + 时间序列图"""
        print("\n📊 绘制散点图和时间序列图...")
        
        data = self.matched_orders
        
        # 创建1x2的子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # === 左图：散点图 ===
        # 按仓位大小设置颜色（从大到小颜色依次变深）
        colors = []
        
        # 计算仓位大小的分位数用于颜色映射
        position_values = data['Low_Value'].astype(float)  # 使用小仓位作为参考
        
        # 使用蓝色系，从浅到深
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # 标准化仓位值到0-1范围
        norm = mcolors.Normalize(vmin=position_values.min(), vmax=position_values.max())
        colormap = cm.Blues_r  # 倒序Blues，大仓位深色，小仓位浅色
        
        for _, row in data.iterrows():
            color_intensity = norm(row['Low_Value'])
            colors.append(colormap(color_intensity))
        
        # 绘制散点图
        scatter = ax1.scatter(data['Low_Quantity'], data['High_Scaled_Quantity'], 
                   c=colors, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        # 设置对数坐标轴（因为大多数订单都很小）
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # 添加理想匹配线
        min_val = max(data['Low_Quantity'].min(), data['High_Scaled_Quantity'].min(), 1)  # 避免0值
        max_val = max(data['Low_Quantity'].max(), data['High_Scaled_Quantity'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='理想匹配线')
        
        # 设置标签和标题
        ax1.set_xlabel('小仓位订单数量（对数坐标）', fontsize=12)
        ax1.set_ylabel('大仓位订单数量-缩放后（对数坐标）', fontsize=12)
        ax1.set_title(f'订单数量对比散点图\n缩放比例: {self.scaling_factor:.2f}x', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 添加颜色条说明仓位大小
        cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
        cbar.set_label('仓位价值（深色=大仓位）', fontsize=10)
        
        # 添加理想匹配线图例
        ax1.legend(['理想匹配线'], loc='lower right', fontsize=9)
        
        # 显示统计信息 - 放在右上角
        success_both = len(data[(data['High_Status'] == 'Filled') & (data['Low_Status'] == 'Filled')])
        fail_high = len(data[(data['High_Status'] == 'Invalid') & (data['Low_Status'] == 'Filled')])
        fail_low = len(data[(data['High_Status'] == 'Filled') & (data['Low_Status'] == 'Invalid')])
        fail_both = len(data[(data['High_Status'] == 'Invalid') & (data['Low_Status'] == 'Invalid')])
        
        stats_text = f"""统计信息:
两者都成功: {success_both} ({success_both/len(data)*100:.1f}%)
大仓位失败: {fail_high} ({fail_high/len(data)*100:.1f}%)
小仓位失败: {fail_low} ({fail_low/len(data)*100:.1f}%)
两者都失败: {fail_both} ({fail_both/len(data)*100:.1f}%)"""
        
        ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
               verticalalignment='top', horizontalalignment='right', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        # === 右图：时间序列图 ===
        # 只使用成功的订单绘制时间序列
        filled_data = data[(data['High_Status'] == 'Filled') & (data['Low_Status'] == 'Filled')].copy()
        
        if len(filled_data) > 0:
            # 按时间排序
            filled_data = filled_data.sort_values('Time')
            
            # 计算仓位大小（用于圆点大小）
            # 使用对数尺度处理长尾分布
            high_size = np.log10(filled_data['High_Value'].astype(float) + 1) * 10
            low_size = np.log10(filled_data['Low_Value'].astype(float) + 1) * 10
            
            # 绘制大仓位时间序列（原始值）
            ax2.scatter(filled_data['Time'], filled_data['High_Value'], 
                       s=high_size, alpha=0.6, c='red', label='大仓位', edgecolors='darkred', linewidth=0.5)
            
            # 绘制小仓位时间序列
            ax2.scatter(filled_data['Time'], filled_data['Low_Value'], 
                       s=low_size, alpha=0.6, c='blue', label='小仓位', edgecolors='darkblue', linewidth=0.5)
            
            # 设置标签和标题
            ax2.set_xlabel('时间', fontsize=12)
            ax2.set_ylabel('盈亏', fontsize=12)
            ax2.set_title(f'时间序列盈亏图\n圆点大小=仓位大小（对数尺度）', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 图例
            ax2.legend(loc='upper left', fontsize=10)
            
            # 旋转x轴标签
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加统计信息
            high_pnl = filled_data['High_Value'].sum()
            low_pnl = filled_data['Low_Value'].sum()
            
            pnl_text = f"""盈亏统计:
大仓位总盈亏: {high_pnl:,.0f}
小仓位总盈亏: {low_pnl:,.0f}
比例: {high_pnl/low_pnl:.2f}x"""
            
            ax2.text(0.02, 0.98, pnl_text, transform=ax2.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        else:
            ax2.text(0.5, 0.5, '没有成功的订单可供展示', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        if save_plot:
            filename = "position_comparison_combined.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"💾 图表已保存: {filename}")
        else:
            plt.show()
        
        return fig
    
    def run_analysis(self, save_plot=False):
        """运行完整分析"""
        # 第一步：找匹配订单
        if not self.find_matching_orders():
            return
        
        # 第二步：计算缩放
        if not self.calculate_scaling():
            return
        
        # 第三步：画图
        self.plot_scatter(save_plot=save_plot)
        
        print("\n✅ 分析完成!")

def main():
    """主函数"""
    high_pos_csv = "MACD-long-crypto/2023-2024/biger-pos.csv"
    low_pos_csv = "MACD-long-crypto/2023-2024/less-pos.csv"
    
    try:
        comparator = UltraSimpleComparator(high_pos_csv, low_pos_csv)
        
        # 询问是否保存
        try:
            save_choice = input("\n保存散点图? (y/n, 默认 n): ").lower()
            should_save = save_choice in ['y', 'yes']
        except EOFError:
            should_save = False
        
        comparator.run_analysis(save_plot=should_save)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()