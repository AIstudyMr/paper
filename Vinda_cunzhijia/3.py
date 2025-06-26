"""
计算相邻数据点之间的斜率（一阶差分）
计算斜率的变化率（二阶差分）
通过二阶差分符号变化识别拐点（尖峰/低谷）
根据拐点前后斜率变化判断是尖峰还是低谷
"""



import pandas as pd
import numpy as np
from collections import deque

# 读取CSV文件
df = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv')

# 提取存纸率列数据
paper_rate = df['存纸率'].values
time_stamps = df['时间'].values


# 计算一阶差分（斜率）
first_diff = np.diff(paper_rate)

# 计算二阶差分（斜率变化率）
second_diff = np.diff(first_diff)

# 寻找拐点（二阶差分符号变化的位置）
inflection_points = []
for i in range(1, len(second_diff)):
    if second_diff[i] * second_diff[i-1] < 0:  # 符号变化
        inflection_points.append(i+1)  # 补偿两次差分后的索引偏移

# 识别尖峰和低谷
peaks = []
valleys = []
for point in inflection_points:
    if point >= len(paper_rate)-1 or point < 1:
        continue
    
    # 判断是尖峰还是低谷
    if first_diff[point-1] > 0 and first_diff[point] < 0:
        peaks.append(point)
    elif first_diff[point-1] < 0 and first_diff[point] > 0:
        valleys.append(point)

# 输出结果
print("尖峰位置（行索引）:", peaks)
print("低谷位置（行索引）:", valleys)

# 获取对应时间戳（如果需要）
if '时间' in df.columns:
    peak_times = df.iloc[peaks]['时间'].values if peaks else []
    valley_times = df.iloc[valleys]['时间'].values if valleys else []
    print("\n尖峰时间:", peak_times)
    print("低谷时间:", valley_times)


# 创建尖峰和低谷的DataFrame，并过滤掉存纸率<=1的尖峰
peak_data = pd.DataFrame({
    '时间': time_stamps[peaks],
    '存纸率': paper_rate[peaks]
})
peak_data = peak_data[peak_data['存纸率'] > 10]  # 只保留存纸率大于1的尖峰

valley_data = pd.DataFrame({
    '时间': time_stamps[valleys],
    '存纸率': paper_rate[valleys]
})
valley_data = valley_data[valley_data['存纸率'] < 90]  # 只保留存纸率大于1的尖峰

# 保存到CSV文件
peak_data.to_csv('存纸率_尖峰.csv', index=False, encoding='utf-8')
valley_data.to_csv('存纸率_低谷.csv', index=False, encoding='utf-8')
