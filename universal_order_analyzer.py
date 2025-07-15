"""
通用订单分析可视化启动脚本
支持分析不同策略的订单数据
"""

import sys
import os
import glob

# 添加订单分析目录到路径
orders_analysis_dir = "/Users/yukiarima/Desktop/Quant/QuantFramework/orders-analysis"
if orders_analysis_dir not in sys.path:
    sys.path.append(orders_analysis_dir)

from main_analysis import analyze_strategy_orders, create_dashboard_analysis

def find_csv_files():
    """查找当前目录下的CSV文件"""
    csv_files = glob.glob("*.csv")
    return csv_files

def select_csv_file(csv_files):
    """让用户选择要分析的CSV文件"""
    if not csv_files:
        print("❌ 当前目录下没有找到CSV文件")
        return None
    
    if len(csv_files) == 1:
        print(f"📁 找到CSV文件: {csv_files[0]}")
        return csv_files[0]
    
    print("\n📁 找到以下CSV文件:")
    for i, file in enumerate(csv_files, 1):
        print(f"{i}. {file}")
    
    while True:
        try:
            choice = input(f"\n请选择要分析的文件 (1-{len(csv_files)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(csv_files):
                return csv_files[index]
            else:
                print(f"请输入1到{len(csv_files)}之间的数字")
        except ValueError:
            print("请输入有效的数字")

def get_strategy_name(csv_file):
    """从文件名推测策略名称"""
    base_name = os.path.splitext(csv_file)[0]
    return base_name.replace('-', ' ').title()

def main():
    """主函数"""
    print("=" * 60)
    print("📊 通用订单分析可视化工具")
    print("=" * 60)
    
    # 显示当前工作目录
    current_dir = os.getcwd()
    print(f"📂 当前分析目录: {os.path.basename(current_dir)}")
    
    # 查找CSV文件
    csv_files = find_csv_files()
    selected_file = select_csv_file(csv_files)
    
    if not selected_file:
        return
    
    # 获取完整路径
    csv_file_path = os.path.abspath(selected_file)
    strategy_name = get_strategy_name(selected_file)
    
    print(f"\n🎯 分析策略: {strategy_name}")
    print(f"📄 数据文件: {selected_file}")
    
    # 询问分析类型
    try:
        print("\n📊 选择分析类型:")
        print("1. 标准分析 (多个详细图表)")
        print("2. 仪表板分析 (单个综合视图)")
        analysis_choice = input("请选择分析类型 (1-2): ").strip()
        use_dashboard = analysis_choice == '2'
    except EOFError:
        # 非交互环境，默认使用标准分析
        use_dashboard = False
        print("\n📊 非交互环境，使用标准分析")
    
    # 询问是否保存图表 (在非交互环境中默认不保存)
    try:
        save_choice = input("\n💾 是否保存图表到本地? (y/n): ").lower().strip()
        save_plots = save_choice in ['y', 'yes', '是']
    except EOFError:
        # 非交互环境，默认不保存
        save_plots = False
        print("\n💾 非交互环境，默认不保存图表")
    
    try:
        # 运行分析
        print(f"\n🚀 开始分析 {strategy_name} 策略...")
        
        if use_dashboard:
            analyzer, visualizer, dashboard = create_dashboard_analysis(
                csv_file_path, 
                save_plot=save_plots, 
                strategy_name=strategy_name
            )
        else:
            analyzer, visualizer = analyze_strategy_orders(
                csv_file_path, 
                save_plots=save_plots, 
                show_plots=True,
                strategy_name=strategy_name
            )
        
        print(f"\n✅ {strategy_name} 策略分析完成!")
        
        if save_plots:
            if use_dashboard:
                print(f"\n📊 生成的仪表板文件:")
                dashboard_file = f"{strategy_name.lower().replace(' ', '_')}_dashboard.png"
                if os.path.exists(dashboard_file):
                    print(f"  - {dashboard_file}")
            else:
                print(f"\n📊 生成的图表文件:")
                # 查找生成的文件
                strategy_prefix = strategy_name.lower().replace(' ', '_').replace('-', '_')
                chart_patterns = [
                    f"{strategy_prefix}_position_analysis.png",
                    f"{strategy_prefix}_returns_analysis.png", 
                    f"{strategy_prefix}_comprehensive_analysis.png",
                    f"{strategy_prefix}_profit_source_analysis.png",
                    f"{strategy_prefix}_timeseries_analysis.png"
                ]
                
                for pattern in chart_patterns:
                    if os.path.exists(pattern):
                        print(f"  - {pattern}")
            
            print(f"\n📁 图表保存位置: {current_dir}")
        else:
            print(f"\n📊 图表已显示，未保存到本地文件")
            
        # 显示一些建议
        print(f"\n💡 分析建议:")
        print(f"  - 可以将此脚本复制到其他策略文件夹中使用")
        print(f"  - 确保CSV文件包含必要的列: Time, Symbol, Price, Quantity, Type, Status, Value")
        print(f"  - 图表支持英文显示，避免中文字体问题")
        print(f"  - 仪表板模式提供更紧凑的综合视图")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        print("\n🔍 可能的原因:")
        print("  - CSV文件格式不正确")
        print("  - 缺少必要的数据列")
        print("  - 数据中没有有效的交易记录")
        
        import traceback
        print(f"\n详细错误信息:")
        traceback.print_exc()

if __name__ == "__main__":
    main()