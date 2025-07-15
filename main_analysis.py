"""
主要分析函数
整合分析器和可视化器，提供统一的接口
"""

import matplotlib.pyplot as plt
from order_analyzer import OrderAnalyzer
from order_visualizer_charts import OrderVisualizer

def analyze_strategy_orders(csv_file_path, save_plots=False, show_plots=True, strategy_name="Strategy"):
    """
    分析策略订单数据的主函数
    
    Args:
        csv_file_path (str): 订单数据CSV文件路径
        save_plots (bool): 是否保存图表，默认False
        show_plots (bool): 是否显示图表，默认True
        strategy_name (str): 策略名称，用于文件命名
    
    Returns:
        tuple: (analyzer, visualizer) 分析器和可视化器实例
    """
    print(f"开始分析 {strategy_name} 订单数据...")
    
    try:
        # 创建分析器
        analyzer = OrderAnalyzer(csv_file_path)
        
        # 检查是否有有效数据
        if analyzer.processed_data is None or len(analyzer.processed_data) == 0:
            print("❌ 没有找到有效的交易数据，请检查数据格式和内容")
            return None, None
        
        # 创建可视化器
        visualizer = OrderVisualizer(analyzer)
        
        # 生成分析报告
        analyzer.print_summary_report()
        
        # 处理图表
        if save_plots or show_plots:
            print(f"\n生成可视化图表...")
            
            # 如果选择保存图片，就不显示交互式图表
            if save_plots:
                show_plots = False
                print("💾 保存模式：将保存图表文件，不显示交互式图表")
            
            if save_plots:
                # 保存图表
                prefix = strategy_name.lower().replace(' ', '_').replace('-', '_')
                saved_files = []
                
                # 动态调整 DPI 以避免图像过大
                def safe_savefig(fig, filename, max_dpi=300):
                    """安全保存图表，自动调整DPI避免图像过大"""
                    # 获取图表尺寸（英寸）
                    width, height = fig.get_size_inches()
                    
                    # 计算在最大DPI下的像素尺寸
                    max_pixels = 65535  # 2^16 - 1，matplotlib的限制
                    max_dpi_width = max_pixels / width
                    max_dpi_height = max_pixels / height
                    
                    # 选择安全的DPI
                    safe_dpi = min(max_dpi, max_dpi_width, max_dpi_height)
                    safe_dpi = max(72, safe_dpi)  # 最小72 DPI保证质量
                    
                    if safe_dpi < max_dpi:
                        print(f"  📏 图表 {filename} 自动调整DPI: {max_dpi} → {safe_dpi:.0f} (避免图像过大)")
                    
                    fig.savefig(filename, dpi=safe_dpi, bbox_inches='tight')
                
                # 生成各种图表
                print("生成图表...")
                fig1 = visualizer.plot_position_size_analysis()
                filename1 = f"{prefix}_position_analysis.png"
                safe_savefig(fig1, filename1)
                saved_files.append(filename1)
                
                fig2 = visualizer.plot_return_type_analysis()
                filename2 = f"{prefix}_returns_analysis.png"
                safe_savefig(fig2, filename2)
                saved_files.append(filename2)
                
                fig3 = visualizer.plot_comprehensive_analysis()
                filename3 = f"{prefix}_comprehensive_analysis.png"
                safe_savefig(fig3, filename3)
                saved_files.append(filename3)
                
                # 新增：利润来源深度分析
                fig4 = visualizer.plot_profit_source_analysis()
                filename4 = f"{prefix}_profit_source_analysis.png"
                safe_savefig(fig4, filename4)
                saved_files.append(filename4)
                
                fig5 = visualizer.plot_time_series_analysis()
                filename5 = f"{prefix}_timeseries_analysis.png"
                safe_savefig(fig5, filename5)
                saved_files.append(filename5)
                
                print(f"\n📊 已保存图表文件:")
                for file in saved_files:
                    print(f"  - {file}")
            
            if show_plots:
                # 显示图表
                print("🖥️ 显示模式：将显示交互式图表")
                visualizer.plot_position_size_analysis()
                visualizer.plot_return_type_analysis()
                visualizer.plot_comprehensive_analysis()
                visualizer.plot_profit_source_analysis()  # 新增
                visualizer.plot_time_series_analysis()
                plt.show()
        
        print(f"\n✅ {strategy_name} 策略分析完成!")
        return analyzer, visualizer
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_macd_crypto_orders(csv_file_path, save_plots=False, show_plots=True):
    """
    专门分析MACD Long Crypto策略的包装函数，保持向后兼容
    
    Args:
        csv_file_path (str): 订单数据CSV文件路径
        save_plots (bool): 是否保存图表，默认False
        show_plots (bool): 是否显示图表，默认True
    
    Returns:
        tuple: (analyzer, visualizer) 分析器和可视化器实例
    """
    return analyze_strategy_orders(
        csv_file_path=csv_file_path,
        save_plots=save_plots,
        show_plots=show_plots,
        strategy_name="MACD Long Crypto"
    )

def create_dashboard_analysis(csv_file_path, save_plot=False, strategy_name="Strategy"):
    """
    创建仪表板式综合分析
    
    Args:
        csv_file_path (str): 订单数据CSV文件路径
        save_plot (bool): 是否保存仪表板图表
        strategy_name (str): 策略名称
    
    Returns:
        tuple: (analyzer, visualizer, dashboard_fig) 分析器、可视化器和仪表板图表
    """
    print(f"创建 {strategy_name} 综合仪表板...")
    
    try:
        # 创建分析器和可视化器
        analyzer = OrderAnalyzer(csv_file_path)
        visualizer = OrderVisualizer(analyzer)
        
        # 生成仪表板
        dashboard_fig = visualizer.create_dashboard()
        
        if save_plot:
            filename = f"{strategy_name.lower().replace(' ', '_')}_dashboard.png"
            
            # 使用相同的安全保存逻辑
            width, height = dashboard_fig.get_size_inches()
            max_pixels = 65535
            max_dpi_width = max_pixels / width
            max_dpi_height = max_pixels / height
            safe_dpi = min(300, max_dpi_width, max_dpi_height)
            safe_dpi = max(72, safe_dpi)
            
            if safe_dpi < 300:
                print(f"📏 仪表板自动调整DPI: 300 → {safe_dpi:.0f} (避免图像过大)")
            
            dashboard_fig.savefig(filename, dpi=safe_dpi, bbox_inches='tight')
            print(f"💾 保存模式：仪表板已保存: {filename}")
            print("💾 保存模式：不显示交互式图表")
        else:
            print("🖥️ 显示模式：将显示交互式仪表板")
            plt.show()
        
        return analyzer, visualizer, dashboard_fig
        
    except Exception as e:
        print(f"❌ 仪表板创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    # 示例用法
    csv_file = "/Users/yukiarima/Desktop/Quant/QuantFramework/orders-analysis/MACD-long-crypto/MACD-long-crypto-2023-2024.csv"
    
    # 标准分析
    analyzer, visualizer = analyze_strategy_orders(csv_file, save_plots=False, strategy_name="MACD Long Crypto")
    
    # 创建仪表板（可选）
    # analyzer, visualizer, dashboard = create_dashboard_analysis(csv_file, save_plot=False, strategy_name="MACD Long Crypto")