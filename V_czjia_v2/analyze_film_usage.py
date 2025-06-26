import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_film_usage():
    """分析包装胶膜使用情况"""
    
    print("正在读取数据...")
    # 读取数据
    df = pd.read_csv('存纸架数据汇总.csv')
    
    # 转换时间格式
    df['时间'] = pd.to_datetime(df['时间'])
    
    # 胶膜相关列
    film_columns = [
        '1#小包机包装胶膜用完',
        '2#小包机包装胶膜用完', 
        '3#小包机包装胶膜用完',
        '4#小包机包装胶膜用完'
    ]
    
    print("开始分析胶膜使用情况...")
    
    # 1. 胶膜用完次数统计
    print("\n=== 胶膜用完次数统计 ===")
    film_usage_count = {}
    
    for col in film_columns:
        machine_num = col.split('#')[0]
        count = (df[col] == 1).sum()
        film_usage_count[f"{machine_num}号机"] = count
        print(f"{machine_num}号小包机胶膜用完次数: {count}")
    
    # 2. 按日期统计胶膜用完情况
    df_daily = df.copy()
    df_daily['日期'] = df_daily['时间'].dt.date
    
    daily_film_usage = df_daily.groupby('日期')[film_columns].sum()
    
    # 3. 胶膜用完的时间分布分析
    print("\n=== 胶膜用完时间分布分析 ===")
    
    film_usage_times = {}
    for col in film_columns:
        machine_num = col.split('#')[0]
        usage_times = df[df[col] == 1]['时间']
        if len(usage_times) > 0:
            film_usage_times[f"{machine_num}号机"] = usage_times
            print(f"{machine_num}号机最近一次胶膜用完时间: {usage_times.max()}")
    
    # 4. 创建可视化图表
    create_film_usage_plots(film_usage_count, daily_film_usage, film_usage_times, df)
    
    # 5. 胶膜更换频率分析
    analyze_film_replacement_frequency(df, film_columns)
    
    # 6. 胶膜使用效率分析
    analyze_film_efficiency(df, film_columns)
    
    return film_usage_count, daily_film_usage

def create_film_usage_plots(usage_count, daily_usage, usage_times, df):
    """创建胶膜使用情况可视化图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('胶膜使用情况分析报告', fontsize=16, fontweight='bold')
    
    # 1. 各机器胶膜用完次数对比
    machines = list(usage_count.keys())
    counts = list(usage_count.values())
    
    axes[0, 0].bar(machines, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 0].set_title('各机器胶膜用完次数对比')
    axes[0, 0].set_ylabel('用完次数')
    for i, v in enumerate(counts):
        axes[0, 0].text(i, v + 0.1, str(v), ha='center')
    
    # 2. 每日胶膜用完趋势
    daily_total = daily_usage.sum(axis=1)
    axes[0, 1].plot(daily_total.index, daily_total.values, marker='o', linewidth=2)
    axes[0, 1].set_title('每日胶膜用完总数趋势')
    axes[0, 1].set_ylabel('用完次数')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 各机器每日胶膜用完堆叠图
    daily_usage.plot(kind='bar', stacked=True, ax=axes[1, 0], 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1, 0].set_title('各机器每日胶膜用完情况')
    axes[1, 0].set_ylabel('用完次数')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 胶膜用完时间分布（按小时）
    if usage_times:
        all_hours = []
        for machine, times in usage_times.items():
            hours = times.dt.hour
            all_hours.extend(hours)
        
        axes[1, 1].hist(all_hours, bins=24, range=(0, 24), alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('胶膜用完时间分布（按小时）')
        axes[1, 1].set_xlabel('小时')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig('胶膜使用情况分析.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_film_replacement_frequency(df, film_columns):
    """分析胶膜更换频率"""
    print("\n=== 胶膜更换频率分析 ===")
    
    for col in film_columns:
        machine_num = col.split('#')[0]
        
        # 找到胶膜用完的时间点
        film_empty_times = df[df[col] == 1]['时间'].sort_values()
        
        if len(film_empty_times) > 1:
            # 计算更换间隔
            intervals = film_empty_times.diff().dropna()
            avg_interval = intervals.mean()
            
            print(f"{machine_num}号机平均胶膜更换间隔: {avg_interval}")
            print(f"{machine_num}号机胶膜更换次数: {len(film_empty_times)}")
            
            if len(intervals) > 0:
                min_interval = intervals.min()
                max_interval = intervals.max()
                print(f"{machine_num}号机最短更换间隔: {min_interval}")
                print(f"{machine_num}号机最长更换间隔: {max_interval}")
        else:
            print(f"{machine_num}号机胶膜更换次数不足，无法计算间隔")
        
        print("-" * 50)

def analyze_film_efficiency(df, film_columns):
    """分析胶膜使用效率"""
    print("\n=== 胶膜使用效率分析 ===")
    
    # 获取对应的入包数列和速度列
    package_columns = [
        '1#小包机入包数',
        '2#小包机入包数', 
        '3#小包机入包数',
        '4#小包机入包数'
    ]
    
    speed_columns = [
        '1#小包机实际速度',
        '2#小包机实际速度',
        '3#小包机主机实际速度',
        '4#小包机主机实际速度'
    ]
    
    for i, film_col in enumerate(film_columns):
        machine_num = film_col.split('#')[0]
        package_col = package_columns[i]
        speed_col = speed_columns[i]
        
        # 计算胶膜用完时的生产情况
        film_empty_data = df[df[film_col] == 1]
        
        if len(film_empty_data) > 0:
            avg_packages = film_empty_data[package_col].mean()
            avg_speed = film_empty_data[speed_col].mean()
            
            print(f"{machine_num}号机胶膜用完时平均入包数: {avg_packages:.2f}")
            print(f"{machine_num}号机胶膜用完时平均速度: {avg_speed:.2f}")
            
            # 计算总生产量与胶膜消耗比率
            total_packages = df[package_col].sum()
            film_usage_count = (df[film_col] == 1).sum()
            
            if film_usage_count > 0:
                packages_per_film = total_packages / film_usage_count
                print(f"{machine_num}号机每卷胶膜平均生产包数: {packages_per_film:.2f}")
            
            print("-" * 50)

def generate_film_usage_report():
    """生成胶膜使用情况报告"""
    
    usage_count, daily_usage = analyze_film_usage()
    
    # 生成文本报告
    report_filename = f"胶膜使用情况报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("维达胶膜使用情况分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. 各机器胶膜用完次数统计:\n")
        for machine, count in usage_count.items():
            f.write(f"   {machine}: {count} 次\n")
        
        f.write(f"\n2. 总胶膜消耗量: {sum(usage_count.values())} 卷\n")
        
        f.write("\n3. 每日胶膜消耗统计:\n")
        for date, row in daily_usage.iterrows():
            total_daily = row.sum()
            f.write(f"   {date}: {total_daily} 卷\n")
        
        f.write("\n4. 建议:\n")
        f.write("   - 定期检查胶膜库存，确保生产连续性\n")
        f.write("   - 监控胶膜消耗异常，及时排查设备问题\n")
        f.write("   - 优化胶膜更换流程，减少停机时间\n")
    
    print(f"\n报告已保存至: {report_filename}")

if __name__ == "__main__":
    print("开始胶膜使用情况分析...")
    generate_film_usage_report()
    print("分析完成！") 