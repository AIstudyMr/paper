import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_daily_data(input_path, output_path, time_index, column_index):
    """按天绘图（通过列序号）"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # 通过序号获取列名
    time_column = df.columns[time_index]
    column_to_plot = df.columns[column_index]
    
    # 转换时间列
    df[time_column] = pd.to_datetime(df[time_column])
    df['date'] = df[time_column].dt.date
    
    # 按天分组绘图
    for date, group in df.groupby('date'):
        plt.figure(figsize=(18, 10))
        plt.plot(group[time_column], group[column_to_plot], 
                label=column_to_plot, linewidth=2)
        
        plt.title(f'{column_to_plot} - {date}')
        plt.xlabel('时间')
        plt.ylabel('数值')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        
        output_file = os.path.join(output_path, f'{date}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

def plot_weekly_data(input_path, output_path, time_index, column_index):
    """按周绘图（通过列序号）"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    df = pd.read_csv(input_path, encoding='utf-8')
    time_column = df.columns[time_index]
    column_to_plot = df.columns[column_index]
    
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column)
    
    start_date = df[time_column].min()
    end_date = df[time_column].max()
    current_date = start_date
    
    while current_date <= end_date:
        end_week = current_date + timedelta(days=6)
        week_data = df[(df[time_column] >= current_date) & 
                      (df[time_column] <= end_week)]
        
        if not week_data.empty:
            plt.figure(figsize=(18, 10))
            plt.plot(week_data[time_column], week_data[column_to_plot],
                    label=column_to_plot, linewidth=2)
            
            plt.title(f'{column_to_plot} - {current_date.date()}至{end_week.date()}')
            plt.xticks(rotation=45)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            
            output_file = os.path.join(output_path, 
                                     f'{current_date.date()}_to_{end_week.date()}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        current_date = end_week + timedelta(days=1)

def plot_full_data(input_path, output_path, time_index, column_index):
    """绘制全周期数据（通过列序号）"""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    df = pd.read_csv(input_path, encoding='utf-8')
    time_column = df.columns[time_index]
    column_to_plot = df.columns[column_index]
    
    df[time_column] = pd.to_datetime(df[time_column])
    
    plt.figure(figsize=(25, 8))
    plt.plot(df[time_column], df[column_to_plot], linewidth=1)
    
    plt.title(f'全周期趋势 - {column_to_plot}')
    plt.xlabel('时间')
    plt.ylabel('数值')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.2)
    
    # 自动调整日期显示密度
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    output_file = os.path.join(output_path, 'full_period.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# 使用示例
if __name__ == "__main__":
    input_path = r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\存纸率.csv'
    output_daily = r'D:\Code_File\Vinda_cunzhijia\每日图_存纸率'
    output_weekly = r'D:\Code_File\Vinda_cunzhijia\每周图_存纸率'
    output_full = r'D:\Code_File\Vinda_cunzhijia\全周期图_存纸率'
    
    time_index = 0    # 时间列序号（通常是第0列）
    column_index = 1  # 要绘制的数据列序号
    
    plot_daily_data(input_path, output_daily, time_index, column_index)
    plot_weekly_data(input_path, output_weekly, time_index, column_index)
    plot_full_data(input_path, output_full, time_index, column_index)