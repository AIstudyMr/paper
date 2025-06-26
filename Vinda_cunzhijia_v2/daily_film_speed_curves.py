import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_daily_film_speed_curves():
    """为每个小包机生成每日胶膜使用和速度对比曲线图"""
    
    print("正在读取数据...")
    df = pd.read_csv('存纸架数据汇总.csv')
    
    # 转换时间格式
    df['时间'] = pd.to_datetime(df['时间'])
    df['日期'] = df['时间'].dt.date
    df['时间_小时'] = df['时间'].dt.hour
    
    # 获取数据日期范围
    unique_dates = sorted(df['日期'].unique())
    print(f"数据日期范围: {unique_dates[0]} 到 {unique_dates[-1]}")
    print(f"总共 {len(unique_dates)} 天的数据")
    
    # 选择最近7天的数据
    recent_dates = unique_dates[-7:] if len(unique_dates) >= 7 else unique_dates
    print(f"分析最近 {len(recent_dates)} 天的数据: {recent_dates}")
    
    # 定义小包机配置
    machines_config = {
        1: {
            'film_col': '1#小包机包装胶膜用完',
            'speed_col': '1#小包机实际速度',
            'folder': '1号小包机每日胶膜速度曲线'
        },
        2: {
            'film_col': '2#小包机包装胶膜用完',
            'speed_col': '2#小包机实际速度',
            'folder': '2号小包机每日胶膜速度曲线'
        },
        3: {
            'film_col': '3#小包机包装胶膜用完',
            'speed_col': '3#小包机主机实际速度',
            'folder': '3号小包机每日胶膜速度曲线'
        },
        4: {
            'film_col': '4#小包机包装胶膜用完',
            'speed_col': '4#小包机主机实际速度',
            'folder': '4号小包机每日胶膜速度曲线'
        }
    }
    
    # 为每个小包机创建文件夹并生成图表
    for machine_num, config in machines_config.items():
        print(f"\n正在处理 {machine_num}号小包机...")
        
        # 创建文件夹
        folder_name = config['folder']
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"创建文件夹: {folder_name}")
        
        # 为每一天生成图表
        for date in recent_dates:
            generate_daily_chart(df, machine_num, config, date, folder_name)
    
    print("\n所有图表生成完成！")

def generate_daily_chart(df, machine_num, config, target_date, folder_name):
    """为指定日期和小包机生成胶膜使用和速度对比曲线图"""
    
    film_col = config['film_col']
    speed_col = config['speed_col']
    
    # 筛选当天数据
    day_data = df[df['日期'] == target_date].copy()
    
    if len(day_data) == 0:
        print(f"  {target_date} 无数据")
        return
    
    # 按时间排序，使用原始数据，不做任何汇总
    day_data = day_data.sort_values('时间').reset_index(drop=True)
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # 使用原始时间序列数据
    times = day_data['时间']
    film_usage = day_data[film_col]  # 原始胶膜数据
    speed_values = day_data[speed_col]  # 原始速度数据
    
    # 左轴 - 胶膜用完次数（红色曲线）
    color1 = '#FF4444'
    ax1.set_xlabel('时间', fontsize=12)
    ax1.set_ylabel('胶膜用完次数', color=color1, fontsize=12)
    line1 = ax1.plot(times, film_usage, linewidth=2, 
                     color=color1, label='胶膜用完次数')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 右轴 - 机器速度（蓝色曲线）
    ax2 = ax1.twinx()
    color2 = '#4444FF'
    ax2.set_ylabel('机器速度', color=color2, fontsize=12)
    line2 = ax2.plot(times, speed_values, linewidth=2, 
                     color=color2, label='机器速度')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 添加标题
    plt.title(f'{machine_num}号小包机 {target_date} 胶膜使用情况与速度对比曲线', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)
    
    # 添加数据统计信息
    total_film_usage = film_usage.sum()
    total_speed_records = len(speed_values[speed_values > 0])
    max_speed = speed_values.max()
    
    # 在图上添加统计信息
    stats_text = f'当日胶膜用完总次数: {int(total_film_usage)}\n数据记录点数: {len(day_data)}\n最高速度: {max_speed:.1f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    date_str = target_date.strftime('%Y%m%d')
    filename = f'{machine_num}号小包机_{date_str}_胶膜速度对比.png'
    filepath = os.path.join(folder_name, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  生成图表: {filename}")

def generate_summary_report():
    """生成汇总分析报告"""
    
    print("\n正在生成汇总报告...")
    
    # 读取数据
    df = pd.read_csv('存纸架数据汇总.csv')
    df['时间'] = pd.to_datetime(df['时间'])
    df['日期'] = df['时间'].dt.date
    
    # 获取最近7天数据
    unique_dates = sorted(df['日期'].unique())
    recent_dates = unique_dates[-7:] if len(unique_dates) >= 7 else unique_dates
    
    # 生成汇总报告
    report_filename = f"小包机胶膜速度分析汇总报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("小包机胶膜使用情况与速度对比分析汇总报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析日期范围: {recent_dates[0]} 到 {recent_dates[-1]}\n")
        f.write(f"分析天数: {len(recent_dates)} 天\n\n")
        
        # 为每个小包机生成统计信息
        machines_config = {
            1: {'film_col': '1#小包机包装胶膜用完', 'speed_col': '1#小包机实际速度'},
            2: {'film_col': '2#小包机包装胶膜用完', 'speed_col': '2#小包机实际速度'},
            3: {'film_col': '3#小包机包装胶膜用完', 'speed_col': '3#小包机主机实际速度'},
            4: {'film_col': '4#小包机包装胶膜用完', 'speed_col': '4#小包机主机实际速度'}
        }
        
        for machine_num, config in machines_config.items():
            f.write(f"{machine_num}号小包机分析结果\n")
            f.write("-" * 30 + "\n")
            
            # 筛选最近7天数据
            recent_data = df[df['日期'].isin(recent_dates)]
            
            # 胶膜使用统计
            total_film_usage = (recent_data[config['film_col']] == 1).sum()
            daily_film_avg = total_film_usage / len(recent_dates)
            
            # 速度统计
            speed_data = recent_data[recent_data[config['speed_col']] > 0][config['speed_col']]
            avg_speed = speed_data.mean() if len(speed_data) > 0 else 0
            max_speed = speed_data.max() if len(speed_data) > 0 else 0
            
            f.write(f"胶膜用完总次数: {total_film_usage}\n")
            f.write(f"日均胶膜用完次数: {daily_film_avg:.2f}\n")
            f.write(f"平均运行速度: {avg_speed:.2f}\n")
            f.write(f"最高运行速度: {max_speed:.2f}\n")
            
            # 每日明细
            f.write("每日明细:\n")
            for date in recent_dates:
                day_data = recent_data[recent_data['日期'] == date]
                day_film = (day_data[config['film_col']] == 1).sum()
                day_speed_data = day_data[day_data[config['speed_col']] > 0][config['speed_col']]
                day_avg_speed = day_speed_data.mean() if len(day_speed_data) > 0 else 0
                f.write(f"  {date}: 胶膜用完{day_film}次, 平均速度{day_avg_speed:.1f}\n")
            
            f.write("\n")
        
        f.write("分析说明:\n")
        f.write("1. 每个小包机生成独立的文件夹存放分析结果\n")
        f.write("2. 每天生成一个对比曲线图，显示一天内胶膜使用和速度变化\n")
        f.write("3. 红色曲线表示胶膜用完次数，蓝色曲线表示机器速度\n")
        f.write("4. 图表使用原始数据，一秒钟一个数据点，不做平均值处理\n")
    
    print(f"汇总报告已保存为: {report_filename}")

if __name__ == "__main__":
    # 生成每日对比曲线图
    create_daily_film_speed_curves()
    
    # 生成汇总报告
    generate_summary_report() 