"""
利润来源相关性分析器
- 分析利润与不同维度（仓位大小、持仓时间、品种等）的相关性
- 使用热力图进行可视化
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from order_analyzer import OrderAnalyzer
import os
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from mpl_toolkits.mplot3d import Axes3D # 导入3D绘图工具

# --- 辅助函数 ---
def cramers_v(x, y):
    """ 计算两个分类变量的Cramér's V关联强度 """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    # 避免除以零
    if min((k-1), (r-1)) == 0:
        return 0
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def eta_squared(categorical_var, continuous_var):
    """ 计算Eta系数，衡量分类变量对连续变量的解释度 """
    groups = continuous_var.groupby(categorical_var)
    grand_mean = continuous_var.mean()
    ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for name, group in groups)
    ss_total = sum((continuous_var - grand_mean)**2)
    return ss_between / ss_total if ss_total > 0 else 0

class ProfitCorrelationAnalyzer:
    """
    使用多种统计方法分析利润来源与各维度之间的相关性
    """
    def __init__(self, analyzer: OrderAnalyzer):
        if analyzer.processed_data is None or analyzer.processed_data.empty:
            raise ValueError("传入的分析器没有有效的已处理数据。")
        self.analyzer = analyzer
        self.analysis_df = None
        self.correlation_matrix = None
        self.numerical_cols = []
        self.categorical_cols = []

    def prepare_data(self):
        """ 准备并分类用于分析的数据 """
        assert self.analyzer.processed_data is not None, "Analyzer's processed_data is None."
        df = self.analyzer.processed_data.copy()
        
        df = df.rename(columns={'Value': 'Profit', 'Duration': 'HoldingTime'})
        df['OpenMonth'] = df['Time'].dt.month
        
        self.analysis_df = df[['Profit', 'HoldingTime', 'PositionSize', 'Symbol', 'OpenMonth']]
        
        self.numerical_cols = ['Profit', 'HoldingTime']
        self.categorical_cols = ['PositionSize', 'Symbol', 'OpenMonth']
        
        self.analysis_df['OpenMonth'] = self.analysis_df['OpenMonth'].astype('category')
        
        print("✅ Data preparation complete. Variable types have been distinguished.")
        print("Numerical variables:", self.numerical_cols)
        print("Categorical variables:", self.categorical_cols)

    def calculate_mixed_correlation(self):
        """
        构建混合类型变量的相关性矩阵
        - 数值-数值: Pearson
        - 分类-分类: Cramér's V
        - 数值-分类: Eta-squared
        """
        if self.analysis_df is None:
            self.prepare_data()
        
        assert self.analysis_df is not None, "Analysis dataframe has not been prepared."
        df = self.analysis_df
        all_cols = self.numerical_cols + self.categorical_cols
        corr_matrix = pd.DataFrame(np.zeros((len(all_cols), len(all_cols))), index=all_cols, columns=all_cols)

        for i, col1 in enumerate(all_cols):
            for j, col2 in enumerate(all_cols):
                if i == j:
                    corr_matrix.iloc[i, j] = 1.0
                elif col1 in self.numerical_cols and col2 in self.numerical_cols:
                    corr_matrix.iloc[i, j] = df[col1].corr(df[col2])
                elif col1 in self.categorical_cols and col2 in self.categorical_cols:
                    corr_matrix.iloc[i, j] = cramers_v(df[col1], df[col2])
                else:
                    num_col = col1 if col1 in self.numerical_cols else col2
                    cat_col = col2 if col1 in self.numerical_cols else col1
                    corr_matrix.iloc[i, j] = eta_squared(df[cat_col], df[num_col])
        
        self.correlation_matrix = corr_matrix
        print("\n📊 Mixed Correlation Matrix:")
        print(self.correlation_matrix)

    def plot_correlation_analysis(self, save_plot=False, strategy_name="Strategy"):
        """ 绘制3D形式的混合关联矩阵 """
        if self.correlation_matrix is None:
            self.calculate_mixed_correlation()
        
        assert self.correlation_matrix is not None, "Correlation matrix has not been calculated."

        matrix = self.correlation_matrix

        # 设置3D图
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 准备数据
        x_data, y_data = np.meshgrid(np.arange(matrix.shape[1]), np.arange(matrix.shape[0]))
        x_data = x_data.flatten()
        y_data = y_data.flatten()
        z_data = np.zeros(len(x_data))
        dx, dy = 0.8, 0.8 # 条形宽度
        dz = matrix.values.flatten()

        # 颜色映射
        cmap = plt.get_cmap('viridis')
        colors = cmap((dz - dz.min()) / (dz.max() - dz.min()))

        # 绘制3D条形图
        ax.bar3d(x_data, y_data, z_data, dx, dy, dz, color=colors) # type: ignore

        # 设置坐标轴
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_zlabel('Correlation Score', labelpad=10) # type: ignore
        
        # 设置标题
        ax.set_title(f'{strategy_name} - 3D Correlation Matrix', fontsize=18, weight='bold')

        if save_plot:
            filename = f"{strategy_name.lower().replace(' ', '_')}_3d_correlation.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n💾 3D analysis chart saved: {filename}")
        else:
            print("\n🖥️  Displaying interactive 3D analysis chart...")
            plt.show()
            
        return fig

    def run_analysis(self, save_plot=False, strategy_name="Strategy"):
        """ 运行完整的相关性分析流程 """
        self.prepare_data()
        self.calculate_mixed_correlation()
        self.plot_correlation_analysis(save_plot=save_plot, strategy_name=strategy_name)


if __name__ == "__main__":
    # --- 配置 ---
    root_dir = os.path.abspath(os.path.join(os.getcwd()))
    csv_file = os.path.join(root_dir, "MACD-long-crypto/MACD-long-crypto-2023-2024-v1.csv")
    strategy_name = "MACD Long Crypto"
    
    try:
        order_analyzer = OrderAnalyzer(csv_file_path=csv_file)
        
        if order_analyzer.processed_data is None or order_analyzer.processed_data.empty:
            print("❌ Could not process order data. Correlation analysis cannot be performed.")
        else:
            correlation_analyzer = ProfitCorrelationAnalyzer(order_analyzer)
            
            save_choice = input("Save correlation heatmap? (y/n, default n): ").lower()
            should_save_plot = save_choice in ['y', 'yes']
            
            correlation_analyzer.run_analysis(save_plot=should_save_plot, strategy_name=strategy_name)
            
            print("\n✅ Correlation analysis complete!")

    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at '{csv_file}'")
    except Exception as e:
        print(f"❌ An unexpected error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()