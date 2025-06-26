import pandas as pd
import numpy as np
import os, csv, shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.colors as mcolors
import matplotlib.dates as mdates  # 新增日期处理模块

plt.rcParams['font.sans-serif'] = ['SimHei']    # 指定默认字体为黑体（需已安装）
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

# 定义归一化函数
def normalize(data, min_val, max_val):
    data_min = data.min()
    data_max = data.max()
    return min_val + (data - data_min) * (max_val - min_val) / (data_max - data_min)

# 为长文本添加换行符
def add_line_breaks(text, max_length=10):
    lines = []
    while len(text) > max_length:
        lines.append(text[:max_length])
        text = text[max_length:]
    lines.append(text)
    return '\n'.join(lines)


# 修改后的数据处理部分
def preprocess_data(df):
    # 存纸率扩大10倍
    if '折叠机实际速度' in df.columns:
        df['折叠机实际速度'] = df['折叠机实际速度'] +100

    if '1#小包机实际速度' in df.columns:
        df['1#小包机实际速度'] = df['1#小包机实际速度'] + 250

    if '2#小包机实际速度' in df.columns:
        df['2#小包机实际速度'] = df['2#小包机实际速度'] + 350

    if '3#小包机主机实际速度' in df.columns:
        df['3#小包机主机实际速度'] = df['3#小包机主机实际速度'] + 450    

    if '4#小包机主机实际速度' in df.columns:
        df['4#小包机主机实际速度'] = df['4#小包机主机实际速度'] + 550   

    return df


def load_additional_data(file_paths):
    """加载额外的存纸率数据文件"""
    additional_data = []
    colors = ['green', 'red']  # 不同文件的颜色
    labels = ['谷值', '峰值']  # 对应的标签名称
    time_cols = ['起始值时间', '结束值时间']  # 不同文件对应的时间列名
    value_cols = ['起始值', '结束值']  # 不同文件对应的数值列名
    
    for i, file_path in enumerate(file_paths):
        # 跳过None值
        if file_path is None:
            continue
            
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # 使用对应的时间列名
            current_time_col = time_cols[i]
            current_value_col = value_cols[i]
                
            df[current_time_col] = pd.to_datetime(df[current_time_col])
            
            additional_data.append({
                'data': df,
                'color': colors[i],
                'label': labels[i],
                'time_col': current_time_col,  # 使用动态确定的时间列名
                'value_col': current_value_col  # 使用动态确定的数值列名
            })
    return additional_data





def plot_hourly_data(input_path, output_path, time_name, columns_name, additional_files=None):
    """
    每小时画一张图
    :param input_path: 输入CSV文件路径
    :param output_path: 输出图片文件夹路径
    :param time_name: 时间列名称
    :param columns_name: 需要绘制的列名列表
    :param normalization_range: 各列的归一化范围列表
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.read_csv(input_path)

    df = preprocess_data(df)    # 不做归一化处理时，显示原数据

    time_column = time_name  # 时间列名称
    columns_to_plot = columns_name  # 需要绘制的列
    df[time_column] = pd.to_datetime(df[time_column])

    # 加载额外数据(绘制峰谷值)
    additional_data = []
    if additional_files:
        additional_data = load_additional_data(additional_files)



    # 按小时分组
    df['hour'] = df[time_column].dt.to_period('H')  # 按小时分组
    grouped = df.groupby('hour')

    for hour, group in grouped:
        plt.figure(figsize=(18, 10))
        
        # 绘制每条曲线
        for i, column in enumerate(columns_to_plot):
            # min_val, max_val = normalization_range[i]
            # normalized_data = normalize(group[column], min_val, max_val)
            # plt.plot(group[time_column], normalized_data, label=column)
            plt.plot(group[time_column], group[column], label=column)

        # 绘制额外数据点（峰谷值）
        for data_info in additional_data:
            # 筛选同一小时的数据
            hour_data = data_info['data'][data_info['data'][data_info['time_col']].dt.to_period('h') == hour]
            if not hour_data.empty:
                plt.scatter(
                    hour_data[data_info['time_col']], 
                    hour_data[data_info['value_col']],
                    color=data_info['color'],
                    label=data_info['label'],
                    s=50,  # 点的大小
                    zorder=5  # 确保点在最上层
                )



        # 设置标题和格式
        hour_str = hour.to_timestamp().strftime('%Y-%m-%d %H:%M')
        plt.title(f'Data from {hour_str}')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', prop={'size': 8})
        plt.xticks(rotation=45)
        plt.grid(alpha=0.2)
        
        # 调整布局并保存
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        output_file = os.path.join(output_path, f'plot_{hour_str.replace(":", "-")}.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=800)
        plt.close()

# 每天画一张图
def plot_daily_data(input_path, output_path, time_name, columns_name, additional_files=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.read_csv(input_path)

    df = preprocess_data(df)    # 不做归一化处理时，显示原数据

    time_column = time_name  # 时间列名称
    columns_to_plot = columns_name  # 需要绘制的列
    df[time_column] = pd.to_datetime(df[time_column]) # 将时间列转换为 datetime 类型

    # 加载额外数据（绘制峰谷值）
    additional_data = []
    if additional_files:
        additional_data = load_additional_data(additional_files)



    # 按天分组
    df['date'] = df[time_column].dt.date  # 提取日期部分
    grouped = df.groupby('date')


    # 遍历每一天的数据并绘图
    for date, group in grouped:
        plt.figure(figsize=(18, 10))  # 设置图表大小
        for i, column in enumerate(columns_to_plot):
            # min_val,max_val = normalization_range[i]
            # normalized_data = normalize(group[column], min_val, max_val)
            # plt.plot(group[time_column], normalized_data, label=column)  # 绘制每一列
            plt.plot(group[time_column], group[column], label=column)

        # 绘制额外数据点（峰谷值）
        for data_info in additional_data:
            # 筛选同一天的数据（按日期筛选，而不是按小时）
            date_data = data_info['data'][data_info['data'][data_info['time_col']].dt.date == date]
            if not date_data.empty:
                plt.scatter(
                    date_data[data_info['time_col']], 
                    date_data[data_info['value_col']],
                    color=data_info['color'],
                    label=data_info['label'],
                    s=50,  # 点的大小
                    zorder=5  # 确保点在最上层
                )


        # 设置图表标题和标签
        plt.title(f'Data on {date}')
        # plt.xlabel('Time')
        # plt.ylabel('Value')

        # 调整图例大小和位置
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., prop={'size': 8})

        plt.xticks(rotation=45)  # 旋转时间标签

        # 调整布局，确保图例不会被裁剪
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整右侧留出空间

        # 保存图表
        output_file = os.path.join(output_path, f'plot_{date}.png')  # 按日期命名文件
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight',dpi=800)  # 保存为图片
        plt.close()  # 关闭当前图表，释放内存

# 每周画一张图
def plot_weekly_data(input_path, output_path, time_name, columns_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.read_csv(input_path)

    df = preprocess_data(df)    # 不做归一化处理时，显示原数据

    time_column = time_name  # 时间列名称
    columns_to_plot = columns_name  # 需要绘制的列
    df[time_column] = pd.to_datetime(df[time_column]) # 将时间列转换为 datetime 类型

    # 按每七天分组
    df = df.sort_values(by=time_column)  # 按时间排序
    start_date = df[time_column].min()  # 数据的最早时间
    end_date = df[time_column].max()  # 数据的最晚时间


    current_date = start_date
    group_id = 1

    while current_date <= end_date:
        # 计算当前组的结束日期（当前日期 + 6天）
        group_end_date = current_date + timedelta(days=6)
        
        # 筛选出当前七天的数据
        group = df[(df[time_column] >= current_date) & (df[time_column] <= group_end_date)]
        
        if not group.empty:
            plt.figure(figsize=(18, 10))  # 设置图表大小
            for i, column in enumerate(columns_to_plot):
                # min_val, max_val = normalization_range[i]
                # normalized_data = normalize(group[column], min_val, max_val)
                # plt.plot(group[time_column], normalized_data, label=column)  # 绘制每一列
                plt.plot(group[time_column], group[column], label=column)

           
            # 设置图表标题和标签
            plt.title(f'Data from {current_date.date()} to {group_end_date.date()}')
            # plt.xlabel('Time')
            # plt.ylabel('Value')

            # 将图例放置在右侧外部
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=-0.1)
            
            plt.xticks(rotation=45)  # 旋转时间标签
            
            # 调整布局，确保图例不会被裁剪
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整右侧留出空间
            
            # 保存图表
            output_file = os.path.join(output_path, f'plot_{current_date.date()}_{group_end_date.date()}.png')  # 按日期命名文件
            plt.tight_layout()
            plt.grid(alpha=0.2)
            plt.savefig(output_file, bbox_inches='tight',dpi=800)  # 保存为图片
            plt.close()  # 关闭当前图表，释放内存
        
        # 移动到下一个七天
        current_date = group_end_date + timedelta(days=1)
        group_id += 1

def plot_full_data(input_path, output_path, time_name, columns_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.read_csv(input_path, encoding='utf-8')

    df = preprocess_data(df)    # 不做归一化处理时，显示原数据

    time_column = time_name
    columns_to_plot = columns_name
    df[time_column] = pd.to_datetime(df[time_column])

    # 创建超大画布适应长周期
    plt.figure(figsize=(30, 10))  # 宽度加大到30英寸

    # 绘制各参数曲线
    for i, column in enumerate(columns_to_plot):
        # min_val, max_val = normalization_range[i]
        # normalized_data = normalize(df[column], min_val, max_val)
        # plt.plot(df[time_column], normalized_data, label=column, linewidth=1)
        plt.plot(df[time_column], df[column], label=column, linewidth=1)

    # 优化时间轴显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')

    # 设置图例和标签
    plt.title('全周期生产数据趋势图', fontsize=16)
    plt.grid(alpha=0.2)
    
    # 分栏显示图例
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.,
              ncol=1, prop={'size': 8})

    # 自适应布局
    plt.tight_layout(rect=[0, 0, 0.85, 1])


    # 保存高清图片
    output_file = os.path.join(output_path, '全周期生产趋势图.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=800)
    plt.close()

# 文件路径
input_path = r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv'
output_path = r'D:\Code_File\Vinda_cunzhijia\每小时图_折叠机速度_存纸率_非归一化'
output_path1 = r'D:\Code_File\Vinda_cunzhijia\每日图_折叠机速度_存纸率_非归一化'
output_path2 = r'D:\Code_File\Vinda_cunzhijia\每周图_折叠机速度_存纸率_非归一化'
output_path3 = r'D:\Code_File\Vinda_cunzhijia\全周期图_折叠机速度_存纸率_非归一化'

# 存纸率峰谷值
file1 = None  # 替换为实际路径
file2 = r'D:\Code_File\Vinda_cunzhijia\存纸率峰值.csv'  # 替换为实际路径


time_name = '时间'

# columns_name = ['折叠机实际速度',"存纸率",'1#小包机实际速度','2#小包机实际速度','3#小包机主机实际速度','4#小包机主机实际速度']
# columns_name = ['折叠机实际速度',"存纸率",'裁切机实际速度','1#小包机实际速度','1号装箱机出包数','2号装箱机出包数','1号装箱机实际速度']
columns_name = ['折叠机实际速度',"存纸率"]
# normalization_range = [(0, 1),(1.1,2.1),(2.2,3.2),(3.3,4.3),(4.4,5.4),(5.5,6.5),(6.6,7.6)]


# 调用函数
plot_hourly_data(input_path, output_path, time_name, columns_name,additional_files=[file1, file2])
plot_daily_data(input_path, output_path1, time_name, columns_name,additional_files=[file1, file2])
# plot_weekly_data(input_path, output_path2, time_name, columns_name)
# plot_full_data(input_path, output_path3, time_name, columns_name)
