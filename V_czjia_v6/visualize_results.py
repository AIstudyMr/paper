import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_visualizations():
    """创建可视化图表分析结果"""
    
    # 读取分析结果
    df = pd.read_csv('小包机停机后存纸率斜率变化分析结果.csv', encoding='utf-8-sig')
    
    print(f"分析结果总览:")
    print(f"总共检测到 {len(df)} 个停机后斜率变化事件")
    print(f"涉及 {df['时间段'].nunique()} 个时间段")
    
    # 统计各小包机的停机影响次数
    machine_counts = {}
    for analysis in df['分析对象'].unique():
        machine_num = analysis.split('_小包机')[1]
        machine_counts[f'小包机{machine_num}'] = len(df[df['分析对象'] == analysis])
    
    print(f"\n各小包机停机影响次数:")
    for machine, count in sorted(machine_counts.items()):
        print(f"{machine}: {count} 次")
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 延迟时间分布直方图
    axes[0, 0].hist(df['延迟时间(分钟)'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('停机后存纸率斜率变化延迟时间分布')
    axes[0, 0].set_xlabel('延迟时间 (分钟)')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].axvline(df['延迟时间(分钟)'].mean(), color='red', linestyle='--', 
                      label=f'平均值: {df["延迟时间(分钟)"].mean():.3f}分钟')
    axes[0, 0].legend()
    
    # 2. 停机持续时间 vs 延迟时间散点图
    axes[0, 1].scatter(df['停机持续时间(分钟)'], df['延迟时间(分钟)'], alpha=0.6, color='orange')
    axes[0, 1].set_title('停机持续时间 vs 延迟时间')
    axes[0, 1].set_xlabel('停机持续时间 (分钟)')
    axes[0, 1].set_ylabel('延迟时间 (分钟)')
    
    # 添加相关性
    correlation = df['停机持续时间(分钟)'].corr(df['延迟时间(分钟)'])
    axes[0, 1].text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                   transform=axes[0, 1].transAxes, 
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # 3. 各小包机影响次数柱状图
    machine_data = pd.Series(machine_counts)
    machine_data.plot(kind='bar', ax=axes[0, 2], color='lightgreen')
    axes[0, 2].set_title('各小包机停机影响次数')
    axes[0, 2].set_xlabel('小包机编号')
    axes[0, 2].set_ylabel('影响次数')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. 停机前后斜率对比
    axes[1, 0].scatter(df['停机前斜率'], df['变化后斜率'], alpha=0.6, color='purple')
    axes[1, 0].set_title('停机前后存纸率斜率对比')
    axes[1, 0].set_xlabel('停机前斜率')
    axes[1, 0].set_ylabel('停机后斜率')
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # 5. 延迟时间的箱型图（按小包机分组）
    df['小包机编号'] = df['分析对象'].str.extract(r'小包机(\d+)')
    
    delay_by_machine = []
    machine_labels = []
    for machine in sorted(df['小包机编号'].unique()):
        machine_data = df[df['小包机编号'] == machine]['延迟时间(分钟)'].values
        delay_by_machine.append(machine_data)
        machine_labels.append(f'小包机{machine}')
    
    axes[1, 1].boxplot(delay_by_machine, labels=machine_labels)
    axes[1, 1].set_title('各小包机延迟时间分布')
    axes[1, 1].set_xlabel('小包机编号')
    axes[1, 1].set_ylabel('延迟时间 (分钟)')
    
    # 6. 时间段分布
    time_segment_counts = df['时间段'].value_counts().head(10)
    time_segment_counts.plot(kind='bar', ax=axes[1, 2], color='coral')
    axes[1, 2].set_title('影响最多的前10个时间段')
    axes[1, 2].set_xlabel('时间段')
    axes[1, 2].set_ylabel('影响次数')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('小包机停机影响分析图表.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 统计分析
    print(f"\n=== 详细统计分析 ===")
    print(f"延迟时间统计:")
    print(f"  平均值: {df['延迟时间(分钟)'].mean():.3f} 分钟")
    print(f"  中位数: {df['延迟时间(分钟)'].median():.3f} 分钟")
    print(f"  标准差: {df['延迟时间(分钟)'].std():.3f} 分钟")
    print(f"  最小值: {df['延迟时间(分钟)'].min():.3f} 分钟")
    print(f"  最大值: {df['延迟时间(分钟)'].max():.3f} 分钟")
    
    print(f"\n停机持续时间统计:")
    print(f"  平均值: {df['停机持续时间(分钟)'].mean():.2f} 分钟")
    print(f"  中位数: {df['停机持续时间(分钟)'].median():.2f} 分钟")
    print(f"  标准差: {df['停机持续时间(分钟)'].std():.2f} 分钟")
    
    # 按小包机分组的统计
    print(f"\n=== 各小包机详细分析 ===")
    for machine in sorted(df['小包机编号'].unique()):
        machine_df = df[df['小包机编号'] == machine]
        print(f"\n小包机{machine}:")
        print(f"  影响次数: {len(machine_df)} 次")
        print(f"  平均延迟时间: {machine_df['延迟时间(分钟)'].mean():.3f} 分钟")
        print(f"  平均停机时间: {machine_df['停机持续时间(分钟)'].mean():.2f} 分钟")
        print(f"  延迟时间范围: {machine_df['延迟时间(分钟)'].min():.3f} - {machine_df['延迟时间(分钟)'].max():.3f} 分钟")
    
    # 分析斜率变化模式
    print(f"\n=== 斜率变化模式分析 ===")
    
    # 计算斜率变化幅度
    df['斜率变化幅度'] = df['变化后斜率'] - df['停机前斜率']
    
    positive_changes = len(df[df['斜率变化幅度'] > 0])
    negative_changes = len(df[df['斜率变化幅度'] < 0])
    
    print(f"斜率增加的情况: {positive_changes} 次 ({positive_changes/len(df)*100:.1f}%)")
    print(f"斜率减少的情况: {negative_changes} 次 ({negative_changes/len(df)*100:.1f}%)")
    print(f"平均斜率变化: {df['斜率变化幅度'].mean():.4f}")
    
    # 保存详细统计结果
    with open('停机影响统计摘要.txt', 'w', encoding='utf-8') as f:
        f.write("小包机停机对存纸率斜率影响分析摘要\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"总体概况:\n")
        f.write(f"- 分析了 {df['时间段'].nunique()} 个时间段的数据\n")
        f.write(f"- 检测到 {len(df)} 个有效的停机后斜率变化事件\n")
        f.write(f"- 平均延迟时间: {df['延迟时间(分钟)'].mean():.3f} 分钟\n")
        f.write(f"- 延迟时间范围: {df['延迟时间(分钟)'].min():.3f} - {df['延迟时间(分钟)'].max():.3f} 分钟\n\n")
        
        f.write("主要发现:\n")
        f.write(f"1. 小包机停机后，存纸率斜率通常在 {df['延迟时间(分钟)'].mean():.3f} 分钟内开始发生变化\n")
        f.write(f"2. 大部分延迟时间集中在 {df['延迟时间(分钟)'].quantile(0.25):.3f} - {df['延迟时间(分钟)'].quantile(0.75):.3f} 分钟之间\n")
        f.write(f"3. 小包机2的停机影响最频繁，共 {machine_counts.get('小包机2', 0)} 次\n")
        f.write(f"4. 停机持续时间与延迟时间的相关性: {correlation:.3f}\n")
        f.write(f"5. {positive_changes} 次停机导致存纸率斜率增加，{negative_changes} 次导致斜率减少\n")

if __name__ == "__main__":
    create_visualizations() 