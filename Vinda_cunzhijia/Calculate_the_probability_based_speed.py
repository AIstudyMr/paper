import pandas as pd
import os
from itertools import combinations



# 读取文件
df1 = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\高低速时间段\low_speed_periods_v1.csv')  # 文件1
df2 = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv')  # 文件2

df2["时间"] = pd.to_datetime(df2["时间"])  # 转换为时间格式

# 初始化结果存储
results = []

# 遍历文件1的每个时间段
for _, row in df1.iterrows():
    start_time = pd.to_datetime(row["开始时间"])
    end_time = pd.to_datetime(row["结束时间"])
    
    # 筛选文件2中该时间段内的数据
    mask = (df2["时间"] >= start_time) & (df2["时间"] <= end_time)
    subset = df2[mask]
    
    # 计算四个小包机的停机概率（速度=0视为停机）
    prob_1 = (subset["1#小包机实际速度"] == 0.0).mean()
    prob_2 = (subset["2#小包机实际速度"] == 0.0).mean()
    prob_3 = (subset["3#小包机主机实际速度"] == 0.0).mean()
    prob_4 = (subset["4#小包机主机实际速度"] == 0.0).mean()
    
    # 存储结果
    results.append({
        "序号": row["序号"],
        "开始时间": row["开始时间"],
        "结束时间": row["结束时间"],
        "持续时间(秒)": row["持续时间(秒)"],
        "1#小包机": prob_1,
        "2#小包机": prob_2,
        "3#小包机": prob_3,
        "4#小包机": prob_4,
    })


outpath = r'D:\Code_File\Vinda_cunzhijia\小包机概率计算'

if not os.path.exists(outpath):
    os.makedirs(outpath)



# 转换为DataFrame并保存为CSV
result_df = pd.DataFrame(results)
result_df.to_csv(os.path.join(outpath,"折叠机中各小包机个体停机概率.csv"), index=False, float_format="%.2f")  # 保留2位小数







##########################################
# 同时停机概率
##########################################

import pandas as pd
from itertools import combinations

# 读取文件1（时间段数据）
df1 = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\高低速时间段\low_speed_periods_v1.csv')  

# 读取文件2（设备状态数据）
df2 = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv')  
df2["时间"] = pd.to_datetime(df2["时间"])  

# 定义小包机列名
pack_columns = {
    "1#": "1#小包机实际速度",
    "2#": "2#小包机实际速度",
    "3#": "3#小包机主机实际速度",
    "4#": "4#小包机主机实际速度"
}

# 初始化结果存储
results = []

# 遍历文件1的每个时间段
for _, row in df1.iterrows():
    start_time = pd.to_datetime(row["开始时间"])
    end_time = pd.to_datetime(row["结束时间"])
    
    # 筛选文件2中该时间段内的数据
    mask = (df2["时间"] >= start_time) & (df2["时间"] <= end_time)
    subset = df2[mask]
    total_records = len(subset)
    
    # 如果该时间段无数据，跳过
    if total_records == 0:
        continue
    
    # 计算每个小包机的停机状态（速度=0）
    pack_stopped = {}
    for pack, col in pack_columns.items():
        pack_stopped[pack] = (subset[col] == 0.0).values  # 返回布尔数组
    
    # 计算不同组合的停机概率
    result = {
        "序号": row["序号"],
        "开始时间": row["开始时间"],
        "结束时间": row["结束时间"],
        "持续时间(秒)": row["持续时间(秒)"],
    }
    
    # 计算2个小包机同时停机的概率（所有组合）
    for packs in combinations(pack_columns.keys(), 2):
        # 计算两个小包机同时停机的记录数
        both_stopped = (pack_stopped[packs[0]] & pack_stopped[packs[1]]).sum()
        prob = both_stopped / total_records
        result[f"{packs[0]}&{packs[1]}同时停机"] = prob
    
    # 计算3个小包机同时停机的概率（所有组合）
    for packs in combinations(pack_columns.keys(), 3):
        # 计算三个小包机同时停机的记录数
        all_stopped = (pack_stopped[packs[0]] & pack_stopped[packs[1]] & pack_stopped[packs[2]]).sum()
        prob = all_stopped / total_records
        result[f"{packs[0]}&{packs[1]}&{packs[2]}同时停机"] = prob
    
    # 计算4个小包机全部停机的概率
    all_stopped = (pack_stopped["1#"] & pack_stopped["2#"] & pack_stopped["3#"] & pack_stopped["4#"]).sum()
    prob = all_stopped / total_records
    result["1#&2#&3#&4#同时停机"] = prob
    
    results.append(result)

# 转换为DataFrame并保存为CSV
result_df = pd.DataFrame(results)
result_df.to_csv(os.path.join(outpath,"折叠机中小包机群体停机概率.csv"), index=False, float_format="%.4f")  # 保留4位小数










