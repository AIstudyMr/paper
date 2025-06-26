import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


df = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv')

def calculate_downtime_probability(df, machine_speed_col, time_col, threshold_seconds=10):
    """
    计算连续停机时间超过threshold_seconds的概率
    
    参数:
    df: 包含数据的DataFrame
    machine_speed_col: 小包机实际速度的列名
    time_col: 时间列名
    threshold_seconds: 视为停机的阈值秒数
    
    返回:
    连续停机超过threshold_seconds的概率
    """
    # 确保时间列是datetime类型
    df[time_col] = pd.to_datetime(df[time_col])
    
    # 按时间排序
    df = df.sort_values(time_col)
    
    # 计算时间间隔
    time_diffs = df[time_col].diff().dt.total_seconds()
    
    # 识别停机状态（速度为0）
    is_stopped = (df[machine_speed_col] == 0)
    
    # 标记停机段的开始和结束
    stop_start = (is_stopped & ~is_stopped.shift(1).fillna(False))
    stop_end = (is_stopped & ~is_stopped.shift(-1).fillna(False))
    
    # 获取所有停机段的开始和结束索引
    start_indices = df.index[stop_start].tolist()
    end_indices = df.index[stop_end].tolist()
    
    # 如果停机段没有结束，忽略最后一个开始
    if len(start_indices) > len(end_indices):
        start_indices = start_indices[:-1]
    
    # 计算每个停机段的持续时间
    long_stops = 0
    total_stops = len(start_indices)
    
    for start, end in zip(start_indices, end_indices):
        duration = time_diffs.loc[start+1:end].sum()
        if duration > threshold_seconds:
            long_stops += 1
    
    # 计算概率
    if total_stops == 0:
        return 0.0
    return long_stops / total_stops

# 使用示例
# 假设数据中时间列名为'time'，1#小包机实际速度列名为'1#小包机实际速度'
probability = calculate_downtime_probability(df, 
                                           machine_speed_col='1#小包机实际速度',
                                           time_col='时间',
                                           threshold_seconds=10)

print(f"1#小包机连续停机超过10秒的概率: {probability:.2%}")



