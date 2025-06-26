import pandas as pd

# 读取CSV文件
df = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\状态转移斜率_v2.csv')  # 请替换为实际文件路径

# 初始化列表来存储峰值和谷值
peaks = []
valleys = []

# 遍历数据行，寻找斜率符号变化
for i in range(1, len(df)):
    prev_slope = df.loc[i-1, '斜率']
    current_slope = df.loc[i, '斜率']
    
    # 从正变负，说明前一行是峰值
    if prev_slope > 0 and current_slope < 0:
        peaks.append(df.loc[i-1])
    # 从负变正，说明前一行是谷值
    elif prev_slope < 0 and current_slope > 0:
        valleys.append(df.loc[i-1])

# 将峰值和谷值转换为DataFrame
peaks_df = pd.DataFrame(peaks)
valleys_df = pd.DataFrame(valleys)

# 保存到文件
peaks_df.to_csv('存纸率峰值.csv', index=False)
valleys_df.to_csv('存纸率谷值.csv', index=False)


print(f"找到 {len(peaks_df)} 个峰值和 {len(valleys_df)} 个谷值")



# # 使用示例
# input_csv = r'D:\Code_File\Vinda_cunzhijia\状态转移斜率_v2.csv'  # 替换为你的输入文件路径
# peak_output = '存纸率峰值.csv'         # 峰值点输出文件
# valley_output = '存纸率谷值.csv'     # 谷值点输出文件

# find_peaks_valleys_by_sign_change(input_csv, peak_output, valley_output)