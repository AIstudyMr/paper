import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
from datetime import datetime

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 创建图片保存的文件夹
output_dir = '时间段数据分析图表_2号小包机'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取时间段数据
time_periods_df = pd.read_csv('折叠机正常运行且高存纸率时间段_最终结果_存纸率1.csv')

# 将时间列转换为datetime
time_periods_df['开始时间'] = pd.to_datetime(time_periods_df['开始时间'])
time_periods_df['结束时间'] = pd.to_datetime(time_periods_df['结束时间'])

print("正在读取大型CSV文件，请耐心等待...")

# 读取汇总数据（只读取需要的列来减少内存使用）
selected_columns = ['时间', '3#小包机实际速度', '3#瞬时切数', '进第三裁切通道纸条计数']
summary_df = pd.read_csv('存纸架数据汇总.csv', usecols=selected_columns, parse_dates=['时间'])

print(f"总共有 {len(time_periods_df)} 个时间段需要分析")

# 处理每个时间段
for idx, period in time_periods_df.iterrows():
    start_time = period['开始时间']
    end_time = period['结束时间']
    
    # 格式化时间为字符串，用于文件名
    start_str = start_time.strftime('%Y%m%d_%H%M%S')
    end_str = end_time.strftime('%Y%m%d_%H%M%S')
    
    print(f"处理时间段 {idx+1}/{len(time_periods_df)}: {start_time} 到 {end_time}")
    
    # 过滤指定时间段的数据
    period_data = summary_df[(summary_df['时间'] >= start_time) & (summary_df['时间'] <= end_time)]
    
    if len(period_data) == 0:
        print(f"警告：时间段 {idx+1} 没有对应的数据")
        continue
    
    # 创建图表
    plt.figure(figsize=(15, 8))
    
    # 绘制所选的数据列（除了时间列）
    for column in selected_columns[1:]:  # 跳过时间列
        plt.plot(period_data['时间'], period_data[column], label=column)
    
    # 设置图表格式
    plt.title(f'时间段 {idx+1}: {start_time} 到 {end_time}')
    plt.xlabel('时间')
    plt.ylabel('数值')
    plt.grid(True)
    plt.legend(loc='best')
    
    # 设置x轴时间格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    
    # 保存图表
    filename = f'时间段_{idx+1}_{start_str}_to_{end_str}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"已保存图表: {filename}")

print("所有时间段处理完毕！") 