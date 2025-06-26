import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def preprocess_data(df):
    if '折叠机实际速度' in df.columns:
        df['折叠机实际速度'] = df['折叠机实际速度'] + 100
    return df

def plot_time_series_by_periods(
    ts_file_path,          # 时间序列文件路径
    periods_file_path,     # 时间段文件路径
    target_columns,        # 要绘制的列（列表形式）
    output_dir_hourly,     # 输出图片的目录（每小时）
    output_dir_daily,      # 输出图片的目录（每天）
    file1_path,            # 文件1路径（小包机数据）
    file2_path             # 文件2路径（同时停机数据）
):
    # 读取数据
    ts_data = pd.read_csv(ts_file_path, parse_dates=['时间'])
    periods_data = pd.read_csv(periods_file_path, parse_dates=['开始时间', '结束时间'])
    file1_data = pd.read_csv(file1_path, parse_dates=['开始时间', '结束时间'])
    file2_data = pd.read_csv(file2_path, parse_dates=['开始时间', '结束时间'])
    
    # 设置图形样式
    plt.style.use('ggplot')
    
    # 绘制每个时间段的组合图
    for idx, period in periods_data.iterrows():
        # 筛选时间段数据
        mask = (ts_data['时间'] >= period['开始时间']) & (ts_data['时间'] <= period['结束时间'])
        period_data = ts_data.loc[mask]
        
        if len(period_data) == 0:
            print(f"时间段 {idx+1} 无数据")
            continue
        
        # 创建图形，留出右侧空间
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0])
        
        # 绘制折叠机速度（左轴）
        color = 'tab:blue'
        ax1.set_ylabel('折叠机实际速度', color=color)
        line1 = ax1.plot(period_data['时间'], period_data['折叠机实际速度'], 
                        color=color, label='折叠机速度')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 创建右轴并绘制存纸率
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('存纸率', color=color)
        line2 = ax2.plot(period_data['时间'], period_data['存纸率'], 
                        color=color, linestyle='--', label='存纸率')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 设置标题和格式
        ax1.set_title(f"折叠机速度与存纸率对比\n{period['开始时间']} 至 {period['结束时间']}")
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 在右侧添加表格
        ax3 = fig.add_subplot(gs[1])
        ax3.axis('off')
        
        # 找到匹配的文件1和文件2数据
        file1_match = file1_data[(file1_data['开始时间'] == period['开始时间']) & 
                                (file1_data['结束时间'] == period['结束时间'])]
        file2_match = file2_data[(file2_data['开始时间'] == period['开始时间']) & 
                                (file2_data['结束时间'] == period['结束时间'])]
        
        # 创建表格数据
        table_data = []
        
        # 添加文件1数据
        if not file1_match.empty:
            file1_row = file1_match.iloc[0]
            table_data.append(['1#小包机', f"{file1_row['1#小包机']:.2f}"])
            table_data.append(['2#小包机', f"{file1_row['2#小包机']:.2f}"])
            table_data.append(['3#小包机', f"{file1_row['3#小包机']:.2f}"])
            table_data.append(['4#小包机', f"{file1_row['4#小包机']:.2f}"])
        
        
        # 添加文件2数据
        if not file2_match.empty:
            file2_row = file2_match.iloc[0]
            table_data.append(['1#&2#停机', f"{file2_row['1#&2#同时停机']:.2f}"])
            table_data.append(['1#&3#停机', f"{file2_row['1#&3#同时停机']:.2f}"])
            table_data.append(['1#&4#停机', f"{file2_row['1#&4#同时停机']:.2f}"])
            table_data.append(['2#&3#停机', f"{file2_row['2#&3#同时停机']:.2f}"])
            table_data.append(['2#&4#停机', f"{file2_row['2#&4#同时停机']:.2f}"])
            table_data.append(['3#&4#停机', f"{file2_row['3#&4#同时停机']:.2f}"])
            table_data.append(['1#&2#&3#停机', f"{file2_row['1#&2#&3#同时停机']:.2f}"])
            table_data.append(['1#&2#&4#停机', f"{file2_row['1#&2#&4#同时停机']:.2f}"])
            table_data.append(['1#&3#&4#停机', f"{file2_row['1#&3#&4#同时停机']:.2f}"])
            table_data.append(['2#&3#&4#停机', f"{file2_row['2#&3#&4#同时停机']:.2f}"])
            table_data.append(['1#&2#&3#&4#停机', f"{file2_row['1#&2#&3#&4#同时停机']:.2f}"])
        
        # 创建表格
        if table_data:
            table = ax3.table(cellText=table_data,
                             colLabels=['指标', '值'],
                             loc='center',
                             cellLoc='center',
                             bbox=[0.1, 0, 0.9, 1])  # 调整表格位置和大小
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # 设置表格标题
            ax3.set_title('停机数据统计', y=1.05)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(f"{output_dir_hourly}/时间段_{idx+1}_组合图.png", dpi=800)
        plt.close()
    
    # 绘制全天组合图
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0])
    
    # 左轴：折叠机速度
    color = 'tab:blue'
    ax1.set_ylabel('折叠机实际速度', color=color)
    line1 = ax1.plot(ts_data['时间'], ts_data['折叠机实际速度'], 
                    color=color, label='折叠机速度')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 右轴：存纸率
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('存纸率', color=color)
    line2 = ax2.plot(ts_data['时间'], ts_data['存纸率'], 
                    color=color, linestyle='--', label='存纸率')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 标记时间段
    for idx, period in periods_data.iterrows():
        ax1.axvspan(period['开始时间'], period['结束时间'], 
                    color='gray', alpha=0.2)
        # 计算时间段中点
        midpoint = period['开始时间'] + (period['结束时间'] - period['开始时间']) / 2
        ax1.text(midpoint, 
                ax1.get_ylim()[1]*0.9,
                f'时段{idx+1}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # 设置标题和格式
    date_str = ts_data['时间'].dt.date[0]
    ax1.set_title(f"折叠机速度与存纸率对比 - 全天数据 ({date_str})")
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 在右侧添加表格
    ax3 = fig.add_subplot(gs[1])
    ax3.axis('off')
    
    # 创建表格数据
    all_table_data = []
    
    for idx, period in periods_data.iterrows():
        # 找到匹配的文件1和文件2数据
        file1_match = file1_data[(file1_data['开始时间'] == period['开始时间']) & 
                                (file1_data['结束时间'] == period['结束时间'])]
        file2_match = file2_data[(file2_data['开始时间'] == period['开始时间']) & 
                                (file2_data['结束时间'] == period['结束时间'])]
        
        if not file1_match.empty and not file2_match.empty:
            file1_row = file1_match.iloc[0]
            file2_row = file2_match.iloc[0]
            
            # 添加时间段标题
            all_table_data.append([f'时段 {idx+1}', ''])
            
            # 添加文件1数据
            all_table_data.append(['1#小包机', f"{file1_row['1#小包机']:.2f}"])
            all_table_data.append(['2#小包机', f"{file1_row['2#小包机']:.2f}"])
            all_table_data.append(['3#小包机', f"{file1_row['3#小包机']:.2f}"])
            all_table_data.append(['4#小包机', f"{file1_row['4#小包机']:.2f}"])
            
            
            # 添加文件2数据
            all_table_data.append(['1#&2#停机', f"{file2_row['1#&2#同时停机']:.2f}"])
            all_table_data.append(['1#&3#停机', f"{file2_row['1#&3#同时停机']:.2f}"])
            all_table_data.append(['1#&4#停机', f"{file2_row['1#&4#同时停机']:.2f}"])
            all_table_data.append(['2#&3#停机', f"{file2_row['2#&3#同时停机']:.2f}"])
            all_table_data.append(['2#&4#停机', f"{file2_row['2#&4#同时停机']:.2f}"])
            all_table_data.append(['3#&4#停机', f"{file2_row['3#&4#同时停机']:.2f}"])
            all_table_data.append(['1#&2#&3#停机', f"{file2_row['1#&2#&3#同时停机']:.2f}"])
            all_table_data.append(['1#&2#&4#停机', f"{file2_row['1#&2#&4#同时停机']:.2f}"])
            all_table_data.append(['1#&3#&4#停机', f"{file2_row['1#&3#&4#同时停机']:.2f}"])
            all_table_data.append(['2#&3#&4#停机', f"{file2_row['2#&3#&4#同时停机']:.2f}"])
            all_table_data.append(['1#&2#&3#&4#停机', f"{file2_row['1#&2#&3#&4#同时停机']:.2f}"])
            

    
    # 创建表格
    if all_table_data:
        table = ax3.table(cellText=all_table_data,
                         colLabels=['指标', '值'],
                         loc='center',
                         cellLoc='center',
                         bbox=[0.1, 0, 0.9, 1])  # 调整表格位置和大小
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        
        # 设置表格标题
        ax3.set_title('各时段停机数据统计', y=1.05)
    
    # 保存全天图
    plt.tight_layout()
    plt.savefig(f"{output_dir_daily}/全天_组合图.png", dpi=800, bbox_inches='tight')
    plt.close()

# 文件路径
ts_file_path = r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv'    
periods_file_path1 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\low_speed_periods_v1.csv'    
periods_file_path2 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\high_speed_periods.csv'
target_columns = ['折叠机实际速度', '存纸率']    

# 新增的文件路径
file1_path = r'D:\Code_File\Vinda_cunzhijia\小包机概率计算\折叠机中各小包机个体停机概率.csv'  # 请替换为实际路径
file2_path = r'D:\Code_File\Vinda_cunzhijia\小包机概率计算\折叠机中小包机群体停机概率.csv'  # 请替换为实际路径

output_dir_hourly1 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\low_speed_hourly_v1'   
output_dir_daily1 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\low_speed_daily_v1'
output_dir_hourly2 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\high_speed_hourly'
output_dir_daily2 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\high_speed_daily'

# 创建输出目录
if not os.path.exists(output_dir_hourly1):    
    os.makedirs(output_dir_hourly1)
if not os.path.exists(output_dir_daily1):    
    os.makedirs(output_dir_daily1)
if not os.path.exists(output_dir_hourly2):    
    os.makedirs(output_dir_hourly2)
if not os.path.exists(output_dir_daily2):    
    os.makedirs(output_dir_daily2)

# 调用函数
plot_time_series_by_periods(ts_file_path, periods_file_path1, target_columns, 
                           output_dir_hourly1, output_dir_daily1, file1_path, file2_path)
# plot_time_series_by_periods(ts_file_path, periods_file_path2, target_columns, 
#                            output_dir_hourly2, output_dir_daily2, file1_path, file2_path)


exit()

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 修改后的数据处理部分
def preprocess_data(df):
    # 存纸率扩大10倍
    if '折叠机实际速度' in df.columns:
        df['折叠机实际速度'] = df['折叠机实际速度'] +100
    return df
def plot_time_series_by_periods(
    ts_file_path,          # 时间序列文件路径
    periods_file_path,     # 时间段文件路径
    target_columns,        # 要绘制的列（列表形式）
    output_dir_hourly,            # 输出图片的目录
    output_dir_daily
):
    """   
    参数:
        ts_file_path: 时间序列文件路径
        periods_file_path: 时间段文件路径
        target_columns: 要绘制的列名列表
        output_dir: 输出图片的目录
    """
    
    # 读取数据
    ts_data = pd.read_csv(ts_file_path, parse_dates=['时间'])
    periods_data = pd.read_csv(periods_file_path, parse_dates=['开始时间', '结束时间'])
    
    # 设置图形样式
    plt.style.use('ggplot')
    

    # 绘制每个时间段的组合图
    for idx, period in periods_data.iterrows():
        # 筛选时间段数据
        mask = (ts_data['时间'] >= period['开始时间']) & (ts_data['时间'] <= period['结束时间'])
        period_data = ts_data.loc[mask]
        
        if len(period_data) == 0:
            print(f"时间段 {idx+1} 无数据")
            continue
        
        # 创建双Y轴图表
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # 绘制折叠机速度（左轴）
        color = 'tab:blue'
        ax1.set_xlabel('时间')
        ax1.set_ylabel('折叠机实际速度', color=color)
        line1 = ax1.plot(period_data['时间'], period_data['折叠机实际速度'], 
                        color=color, label='折叠机速度')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 创建右轴并绘制存纸率
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('存纸率', color=color)
        line2 = ax2.plot(period_data['时间'], period_data['存纸率'], 
                        color=color, linestyle='--', label='存纸率')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 设置标题和格式
        plt.title(f"折叠机速度与存纸率对比\n{period['开始时间']} 至 {period['结束时间']}")
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(f"{output_dir_hourly}/时间段_{idx+1}_组合图.png", dpi=800)
        plt.close()
    
    # 绘制全天组合图
    fig, ax1 = plt.subplots(figsize=(18, 8))
    
    # 左轴：折叠机速度
    color = 'tab:blue'
    ax1.set_ylabel('折叠机实际速度', color=color)
    line1 = ax1.plot(ts_data['时间'], ts_data['折叠机实际速度'], 
                    color=color, label='折叠机速度')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 右轴：存纸率
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('存纸率', color=color)
    line2 = ax2.plot(ts_data['时间'], ts_data['存纸率'], 
                    color=color, linestyle='--', label='存纸率')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 标记时间段
    for idx, period in periods_data.iterrows():
        ax1.axvspan(period['开始时间'], period['结束时间'], 
                    color='gray', alpha=0.2)
        # 计算时间段中点
        midpoint = period['开始时间'] + (period['结束时间'] - period['开始时间']) / 2
        ax1.text(midpoint, 
                ax1.get_ylim()[1]*0.9,
                f'时段{idx+1}', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    # 设置标题和格式
    date_str = ts_data['时间'].dt.date[0]
    plt.title(f"折叠机速度与存纸率对比 - 全天数据 ({date_str})")
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 保存全天图
    plt.tight_layout()
    plt.savefig(f"{output_dir_daily}/全天_组合图.png", dpi=800, bbox_inches='tight')
    plt.close()


ts_file_path = r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv'    

periods_file_path1 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\low_speed_periods_v1.csv'    
periods_file_path2 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\high_speed_periods.csv'
target_columns =  ['折叠机实际速度', '存纸率']    

output_dir_hourly1 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\low_speed_hourly_v1'   
output_dir_daily1 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\low_speed_daily_v1'

output_dir_hourly2 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\high_speed_hourly'
output_dir_daily2 = r'D:\Code_File\Vinda_cunzhijia\高低速时间段\high_speed_daily'

if not os.path.exists(output_dir_hourly1):    
    os.makedirs(output_dir_hourly1)
if not os.path.exists(output_dir_daily1):    
    os.makedirs(output_dir_daily1)
if not os.path.exists(output_dir_hourly2):    
    os.makedirs(output_dir_hourly2)
if not os.path.exists(output_dir_daily2):    
    os.makedirs(output_dir_daily2)

# 调用函数
plot_time_series_by_periods(ts_file_path, periods_file_path1, target_columns, output_dir_hourly1, output_dir_daily1)
# plot_time_series_by_periods(ts_file_path, periods_file_path2, target_columns, output_dir_hourly2, output_dir_daily2)