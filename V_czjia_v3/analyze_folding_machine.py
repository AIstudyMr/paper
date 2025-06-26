#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
折叠机运行状态分析脚本
分析折叠机停机（平均速度<=2）和正常运行（平均速度>=90）的时间段
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def analyze_folding_machine_status(csv_file):
    """
    分析折叠机的运行状态
    """
    print("正在读取数据文件...")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file, encoding='gbk')
    
    print(f"数据总行数: {len(df)}")
    print(f"数据列数: {len(df.columns)}")
    
    # 查找折叠机实际速度列
    speed_column = None
    for col in df.columns:
        if '折叠机实际速度' in col:
            speed_column = col
            break
    
    if speed_column is None:
        print("未找到折叠机实际速度列，正在显示所有列名:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
        return
    
    print(f"找到折叠机速度列: {speed_column}")
    
    # 转换时间列
    df['时间'] = pd.to_datetime(df['时间'])
    
    # 获取折叠机实际速度数据
    df['折叠机速度'] = pd.to_numeric(df[speed_column], errors='coerce')
    
    # 去除空值
    df = df.dropna(subset=['折叠机速度'])
    
    print(f"有效数据行数: {len(df)}")
    print(f"折叠机速度统计:")
    print(f"  最小值: {df['折叠机速度'].min():.2f}")
    print(f"  最大值: {df['折叠机速度'].max():.2f}")
    print(f"  平均值: {df['折叠机速度'].mean():.2f}")
    print(f"  中位数: {df['折叠机速度'].median():.2f}")
    
    # 定义状态
    df['状态'] = '正常'
    df.loc[df['折叠机速度'] <= 10, '状态'] = '停机'
    df.loc[df['折叠机速度'] >= 90, '状态'] = '高速运行'
    
    # 统计各状态的数量
    status_counts = df['状态'].value_counts()
    print(f"\n状态统计:")
    for status, count in status_counts.items():
        print(f"  {status}: {count} 次 ({count/len(df)*100:.1f}%)")
    
    # 找出连续的时间段
    def find_continuous_periods(df, condition, status_name):
        """找出满足条件的连续时间段"""
        periods = []
        current_start = None
        
        for i, row in df.iterrows():
            if condition(row):
                if current_start is None:
                    current_start = row['时间']
            else:
                if current_start is not None:
                    periods.append({
                        '开始时间': current_start,
                        '结束时间': df.loc[i-1, '时间'] if i > 0 else current_start,
                        '状态': status_name,
                        '持续时间': df.loc[i-1, '时间'] - current_start if i > 0 else timedelta(0)
                    })
                    current_start = None
        
        # 处理最后一段
        if current_start is not None:
            periods.append({
                '开始时间': current_start,
                '结束时间': df.iloc[-1]['时间'],
                '状态': status_name,
                '持续时间': df.iloc[-1]['时间'] - current_start
            })
        
        return periods
    
    # 找出停机时间段
    print(f"\n=== 停机时间段分析 (速度 <= 2) ===")
    stop_periods = find_continuous_periods(df, lambda row: row['折叠机速度'] <= 25, '停机')
    
    if stop_periods:
        # 按持续时间排序用于显示
        stop_periods_display = sorted(stop_periods, key=lambda x: x['持续时间'], reverse=True)
        
        print(f"停机时间段总数: {len(stop_periods)}")
        print(f"停机时间段详情:")
        
        total_stop_time = timedelta(0)
        for i, period in enumerate(stop_periods_display):
            duration = period['持续时间']
            total_stop_time += duration
            print(f"  {i+1}. {period['开始时间'].strftime('%Y-%m-%d %H:%M:%S')} -> "
                  f"{period['结束时间'].strftime('%Y-%m-%d %H:%M:%S')} "
                  f"(持续 {duration})")
        
        print(f"总停机时间: {total_stop_time}")
        
        # 按开始时间排序导出到CSV，剔除持续时间低于1分钟的数据
        stop_periods_csv = sorted(stop_periods, key=lambda x: x['开始时间'])
        # 过滤掉持续时间小于1分钟的数据
        stop_periods_filtered = [period for period in stop_periods_csv if period['持续时间'].total_seconds() >= 60]
        stop_df = pd.DataFrame(stop_periods_filtered)
        stop_df.to_csv('折叠机停机时间段_25.csv', index=False, encoding='utf-8-sig')
        print(f"停机时间段已导出到: 折叠机停机时间段_25
              
              .csv (按开始时间排序，已剔除持续时间<1分钟的数据)")
        print(f"原始停机时间段数: {len(stop_periods_csv)}，过滤后: {len(stop_periods_filtered)}")
    else:
        print("未发现停机时间段")
    
    # 找出正常运行时间段
    print(f"\n=== 正常运行时间段分析 (速度 >= 90) ===")
    normal_periods = find_continuous_periods(df, lambda row: row['折叠机速度'] >= 90, '正常运行')
    
    if normal_periods:
        # 按持续时间排序用于显示
        normal_periods_display = sorted(normal_periods, key=lambda x: x['持续时间'], reverse=True)
        
        print(f"正常运行时间段总数: {len(normal_periods)}")
        print(f"正常运行时间段详情:")
        
        total_normal_time = timedelta(0)
        for i, period in enumerate(normal_periods_display):
            duration = period['持续时间']
            total_normal_time += duration
            print(f"  {i+1}. {period['开始时间'].strftime('%Y-%m-%d %H:%M:%S')} -> "
                  f"{period['结束时间'].strftime('%Y-%m-%d %H:%M:%S')} "
                  f"(持续 {duration})")
        
        print(f"总正常运行时间: {total_normal_time}")
        
        # 按开始时间排序导出到CSV，剔除持续时间低于1分钟的数据
        normal_periods_csv = sorted(normal_periods, key=lambda x: x['开始时间'])
        # 过滤掉持续时间小于1分钟的数据
        normal_periods_filtered = [period for period in normal_periods_csv if period['持续时间'].total_seconds() >= 60]
        normal_df = pd.DataFrame(normal_periods_filtered)
        normal_df.to_csv('折叠机正常运行时间段.csv', index=False, encoding='utf-8-sig')
        print(f"正常运行时间段已导出到: 折叠机正常运行时间段.csv (按开始时间排序，已剔除持续时间<1分钟的数据)")
        print(f"原始正常运行时间段数: {len(normal_periods_csv)}，过滤后: {len(normal_periods_filtered)}")
    else:
        print("未发现正常运行时间段")
    
    # 创建可视化图表
    print(f"\n正在生成可视化图表...")
    
    # 采样数据以提高绘图性能（如果数据量太大）
    if len(df) > 10000:
        sample_df = df.sample(n=10000).sort_values('时间')
        print(f"数据量较大，采样 {len(sample_df)} 行进行可视化")
    else:
        sample_df = df
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 第一个子图：折叠机速度时间序列
    ax1.plot(sample_df['时间'], sample_df['折叠机速度'], linewidth=0.5, alpha=0.7)
    ax1.axhline(y=10, color='red', linestyle='--', label='停机阈值 (速度 ≤ 10)')
    ax1.axhline(y=90, color='green', linestyle='--', label='正常运行阈值 (速度 ≥ 90)')
    ax1.set_ylabel('折叠机实际速度')
    ax1.set_title('折叠机实际速度时间序列')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 设置x轴时间格式
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 第二个子图：状态分布
    status_colors = {'停机': 'red', '正常': 'blue', '高速运行': 'green'}
    for status in sample_df['状态'].unique():
        status_data = sample_df[sample_df['状态'] == status]
        ax2.scatter(status_data['时间'], status_data['折叠机速度'], 
                   c=status_colors.get(status, 'gray'), s=1, alpha=0.6, label=status)
    
    ax2.set_xlabel('时间')
    ax2.set_ylabel('折叠机实际速度')
    ax2.set_title('折叠机运行状态分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 设置x轴时间格式
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('折叠机运行状态分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"可视化图表已保存到: 折叠机运行状态分析.png")

if __name__ == "__main__":
    # 分析数据
    analyze_folding_machine_status("存纸架数据汇总.csv") 