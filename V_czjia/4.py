import pandas as pd
import numpy as np
from collections import deque

# 读取CSV文件
df = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv')  # 请替换为实际文件名

# 提取存纸率列数据和时间列
paper_rate = df['存纸率'].values
time_stamps = df['时间'].values

# 参数设置
window_size = 30  # 滑动窗口大小（可根据数据特性调整）
min_peak_height = 0  # 最小峰高阈值（过滤小波动）
min_valley_depth = 0  # 最小谷深阈值

# 计算一阶差分（斜率）
first_diff = np.diff(paper_rate)

# 滑动窗口平滑处理
window = deque(maxlen=window_size)
smoothed_diff = []
window_mid_indices = []

for i in range(len(first_diff)):
    window.append(first_diff[i])
    if len(window) == window_size:
        # 计算窗口平均值和中值索引
        avg = np.mean(window)
        smoothed_diff.append(avg)
        mid_idx = i - window_size // 2  # 窗口中间位置的原始索引
        window_mid_indices.append(mid_idx)

# 比较相邻窗口寻找峰谷
peaks = []
valleys = []

for i in range(1, len(smoothed_diff)):
    prev_avg = smoothed_diff[i-1]
    curr_avg = smoothed_diff[i]
    
    # 检测尖峰（前窗口正斜率，当前窗口负斜率）
    if prev_avg > min_peak_height and curr_avg < -min_peak_height:
        # 取两个窗口的中值位置作为峰位置
        peak_idx = (window_mid_indices[i-1] + window_mid_indices[i]) // 2
        peaks.append(peak_idx)
    
    # 检测低谷（前窗口负斜率，当前窗口正斜率）
    elif prev_avg < -min_valley_depth and curr_avg > min_valley_depth:
        valley_idx = (window_mid_indices[i-1] + window_mid_indices[i]) // 2
        valleys.append(valley_idx)

# 创建结果DataFrame（添加存纸率为0的过滤条件）
def create_result_df(indices, source_data, time_data):
    """创建包含中值和时间戳的结果DataFrame，并过滤存纸率为0的点"""
    result = []
    for idx in indices:
        # 确定窗口边界
        start = max(0, idx - window_size // 2)
        end = min(len(source_data), idx + window_size // 2 + 1)
        
        # 计算窗口内数据的中值
        window_data = source_data[start:end]
        median_val = np.median(window_data)
        
        # 跳过存纸率为0的点
        if np.isclose(median_val, 0.0, atol=1e-6):  # 使用容差比较避免浮点误差
            continue
            
        # 获取中值对应的时间戳（取窗口中间位置）
        median_time = time_data[idx]
        
        result.append({
            '时间': median_time,
            '存纸率': median_val,
            '窗口起始索引': start,
            '窗口结束索引': end-1
        })
    return pd.DataFrame(result)

# 应用过滤条件创建结果DataFrame
peak_df = create_result_df(peaks, paper_rate, time_stamps)
valley_df = create_result_df(valleys, paper_rate, time_stamps)

# 保存结果
peak_df.to_csv('滑动窗口_存纸率尖峰.csv', index=False, encoding='utf-8')
valley_df.to_csv('滑动窗口_存纸率低谷.csv', index=False, encoding='utf-8')

print("结果已保存：")
print(f"- 滑动窗口_存纸率尖峰.csv（{len(peak_df)}个尖峰）")
print(f"- 滑动窗口_存纸率低谷.csv（{len(valley_df)}个低谷）")

# 打印检测到的峰谷信息
if not peak_df.empty:
    print("\n尖峰检测结果（前5个）：")
    print(peak_df.head())
if not valley_df.empty:
    print("\n低谷检测结果（前5个）：")
    print(valley_df.head())