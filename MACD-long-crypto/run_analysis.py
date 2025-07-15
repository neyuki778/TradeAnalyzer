"""
MACD Long Crypto 订单分析运行脚本
运行此脚本来生成订单分析可视化图表
"""

import sys
import os

# 添加订单分析目录到路径
project_root = os.path.abspath(os.path.join(os.getcwd()))
# print(f"项目根目录: {project_root}")
orders_analysis_dir = os.path.join(project_root)
if orders_analysis_dir not in sys.path:
    sys.path.append(orders_analysis_dir)

from main_analysis import analyze_strategy_orders, create_dashboard_analysis

def main():
    """主函数"""
    print("=" * 60)
    print("MACD Long Crypto 策略订单分析")
    print("=" * 60)
    
    # 订单数据文件路径
    csv_file = os.path.join(orders_analysis_dir, "MACD-long-crypto/MACD-long-crypto-2023-2024-v1.csv")

    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: 订单数据文件不存在: {csv_file}")
        return
    
    # 切换到订单分析目录，保存图表到此目录
    save_image_path = os.path.join(orders_analysis_dir, "MACD-long-crypto")
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)
    os.chdir(save_image_path)
    
    # 询问分析类型
    print("\n📊 选择分析类型:")
    print("1. 标准分析 (多个详细图表)")
    print("2. 仪表板分析 (单个综合视图)")
    analysis_choice = input("请选择分析类型 (1-2): ").strip()
    use_dashboard = analysis_choice == '2'
    
    # 询问是否保存图表
    save_choice = input("\n是否保存图表到本地? (y/n): ").lower().strip()
    save_plots = save_choice in ['y', 'yes', '是']
    
    try:
        # 运行分析
        if use_dashboard:
            analyzer, visualizer, dashboard = create_dashboard_analysis(
                csv_file, 
                save_plot=save_plots, 
                strategy_name="MACD Long Crypto"
            )
        else:
            analyzer, visualizer = analyze_strategy_orders(
                csv_file, 
                save_plots=save_plots,
                show_plots=True,
                strategy_name="MACD Long Crypto"
            )
        
        print("\n✅ 分析完成!")
        
        if save_plots:
            current_dir = os.getcwd()
            if use_dashboard:
                print("\n生成的仪表板文件:")
                print("- macd_long_crypto_dashboard.png")
            else:
                print("\n生成的文件:")
                print("- macd_long_crypto_position_analysis.png (仓位大小分析)")
                print("- macd_long_crypto_returns_analysis.png (收益类型分析)")
                print("- macd_long_crypto_comprehensive_analysis.png (综合分析)")
                print("- macd_long_crypto_timeseries_analysis.png (时间序列分析)")
            
            # 显示文件保存位置
            print(f"\n📁 图表保存位置: {current_dir}")
        else:
            print("\n图表已显示，未保存到本地文件")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()