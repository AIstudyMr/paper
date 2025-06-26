import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """读取CSV文件"""
    print("正在读取数据...")
    df = pd.read_csv('存纸架数据汇总.csv')
    df['时间'] = pd.to_datetime(df['时间'])
    print(f"数据时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
    return df

def create_analysis_folder():
    """创建分析结果保存文件夹"""
    folder = '尾架分析结果'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def analyze_tailstock_impact(df):
    """分析尾架状态对性能指标的影响"""
    results = {}
    
    # 1. 分析尾架原纸剩余米数与折叠机速度的关系
    tailstock_columns = [col for col in df.columns if '尾架' in col and '原纸剩余米数' in col]
    
    for col in tailstock_columns:
        tailstock_num = col.split('尾架')[1][0]  # 提取尾架编号
        
        # 创建散点图
        plt.figure(figsize=(12, 6))
        plt.scatter(df[col], df['折叠机实际速度'], alpha=0.5)
        plt.xlabel(f'尾架{tailstock_num}原纸剩余米数')
        plt.ylabel('折叠机实际速度')
        plt.title(f'尾架{tailstock_num}原纸剩余米数与折叠机速度关系')
        
        # 添加趋势线
        z = np.polyfit(df[col], df['折叠机实际速度'], 1)
        p = np.poly1d(z)
        plt.plot(df[col], p(df[col]), "r--", alpha=0.8)
        
        # 计算相关系数
        corr = df[col].corr(df['折叠机实际速度'])
        plt.text(0.05, 0.95, f'相关系数: {corr:.2f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.savefig(f'尾架分析结果/尾架{tailstock_num}原纸剩余米数_折叠机速度关系.png')
        plt.close()
        
        results[f'尾架{tailstock_num}_速度相关性'] = corr

    # 2. 分析尾架状态与存纸率的关系
    def calculate_metrics_by_remaining(df, tailstock_num):
        col = f'尾架{tailstock_num}原纸剩余米数'
        
        # 将原纸剩余米数分成区间
        df['remaining_group'] = pd.qcut(df[col], q=5, labels=['极低', '低', '中', '高', '极高'])
        
        # 获取区间范围
        bins = pd.qcut(df[col], q=5)
        interval_ranges = bins.unique()
        print(f"\n尾架{tailstock_num}原纸剩余米数区间范围：")
        for label, interval in zip(['极低', '低', '中', '高', '极高'], sorted(interval_ranges)):
            print(f"{label}: {interval}")
        
        # 计算每个区间的平均存纸率和标准差
        metrics = df.groupby('remaining_group').agg({
            '存纸率': ['mean', 'std'],
            '折叠机实际速度': ['mean', 'std']
        }).round(2)
        
        return metrics, interval_ranges
    
    # 为每个尾架创建分析图表
    for col in tailstock_columns:
        tailstock_num = col.split('尾架')[1][0]
        metrics, ranges = calculate_metrics_by_remaining(df.copy(), tailstock_num)
        
        # 创建柱状图
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # 绘制存纸率柱状图
        metrics['存纸率']['mean'].plot(kind='bar', yerr=metrics['存纸率']['std'], 
                                   capsize=5, ax=ax, position=0, width=0.3,
                                   color='blue', label='存纸率')
        
        # 添加折叠机速度线图
        ax2 = ax.twinx()
        metrics['折叠机实际速度']['mean'].plot(color='red', marker='o', 
                                          label='折叠机速度', ax=ax2)
        
        plt.title(f'尾架{tailstock_num}原纸剩余量对存纸率和折叠机速度的影响')
        ax.set_xlabel('原纸剩余量区间')
        ax.set_ylabel('存纸率 (%)')
        ax2.set_ylabel('折叠机速度')
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.savefig(f'尾架分析结果/尾架{tailstock_num}原纸剩余量影响分析.png')
        plt.close()
        
        results[f'尾架{tailstock_num}_存纸率分析'] = metrics.to_dict()
        results[f'尾架{tailstock_num}_区间范围'] = {label: str(interval) for label, interval in 
                                                zip(['极低', '低', '中', '高', '极高'], sorted(ranges))}

    # 3. 分析尾架状态对小包机速度的影响
    machine_speed_cols = [col for col in df.columns if '小包机实际速度' in col or '小包机主机实际速度' in col]
    
    for speed_col in machine_speed_cols:
        machine_num = speed_col.split('#')[0]
        
        plt.figure(figsize=(15, 8))
        for col in tailstock_columns:
            tailstock_num = col.split('尾架')[1][0]
            
            # 计算相关系数
            corr = df[col].corr(df[speed_col])
            plt.scatter(df[col], df[speed_col], alpha=0.3, 
                       label=f'尾架{tailstock_num} (相关系数: {corr:.2f})')
        
        plt.xlabel('尾架原纸剩余米数')
        plt.ylabel(f'{machine_num}#小包机速度')
        plt.title(f'尾架原纸剩余量与{machine_num}#小包机速度关系')
        plt.legend()
        
        plt.savefig(f'尾架分析结果/{machine_num}#小包机速度与尾架关系.png')
        plt.close()

    # 4. 分析临界值影响
    def analyze_critical_values(df, threshold=100):
        """分析原纸剩余量低于阈值时的影响"""
        critical_stats = {}
        
        for col in tailstock_columns:
            tailstock_num = col.split('尾架')[1][0]
            
            # 分析低于阈值和高于阈值的情况
            low_remaining = df[df[col] <= threshold]
            high_remaining = df[df[col] > threshold]
            
            stats = {
                '低于阈值_存纸率': low_remaining['存纸率'].mean(),
                '高于阈值_存纸率': high_remaining['存纸率'].mean(),
                '低于阈值_折叠机速度': low_remaining['折叠机实际速度'].mean(),
                '高于阈值_折叠机速度': high_remaining['折叠机实际速度'].mean(),
                '低于阈值样本数': len(low_remaining),
                '高于阈值样本数': len(high_remaining)
            }
            
            critical_stats[f'尾架{tailstock_num}'] = stats
        
        return critical_stats
    
    results['临界值分析'] = analyze_critical_values(df)
    
    return results

def save_results(results):
    """保存分析结果"""
    # 将结果转换为DataFrame并保存为CSV
    critical_analysis = pd.DataFrame(results['临界值分析']).round(2)
    critical_analysis.to_csv('尾架分析结果/临界值分析结果.csv', encoding='utf-8')
    
    # 保存相关性分析结果
    correlations = {k: v for k, v in results.items() if '相关性' in k}
    pd.DataFrame(correlations, index=['相关系数']).round(4).to_csv(
        '尾架分析结果/尾架相关性分析.csv', encoding='utf-8-sig')
    
    print("\n分析结果已保存到'尾架分析结果'文件夹中")
    print("\n=== 临界值分析结果 ===")
    print(critical_analysis)
    
    print("\n=== 相关性分析结果 ===")
    for k, v in correlations.items():
        print(f"{k}: {v:.4f}")

def main():
    try:
        # 创建结果文件夹
        create_analysis_folder()
        
        # 读取数据
        df = load_data()
        
        # 进行分析
        results = analyze_tailstock_impact(df)
        
        # 保存结果
        save_results(results)
        
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")

if __name__ == "__main__":
    main() 