import pandas as pd
import os
from datetime import datetime
from itertools import combinations


# 读取CSV文件
df = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv')  # 请替换为实际文件路径

# 确保时间列是datetime类型
df['时间'] = pd.to_datetime(df['时间'])

# 定义小包机对应的实际速度列名
speed_columns = {
    '1#小包机': '1#小包机实际速度',
    '2#小包机': '2#小包机实际速度',
    '3#小包机': '3#小包机主机实际速度',
    '4#小包机': '4#小包机主机实际速度'
}

# 创建精确到小时的日期时间列（格式：YYYY-MM-DD HH）
df['日期小时'] = df['时间'].dt.strftime('%Y-%m-%d %H:00')

# 添加其他分组列
df['日期'] = df['时间'].dt.date

# 修改周数计算（假设数据的第一天是周的第一天，即周日为一周的第一天）
# 这里我们手动计算周数
first_day = df['时间'].min()
df['周数'] = (df['时间'] - first_day).dt.days // 7 + 1  # 简单按天数划分周，可能需要调整

# 更精确的方法：自定义周数（假设数据的第一天是第1周的第一天）
df['周数'] = (df['时间'] - first_day).dt.days // 7 + 1
# 如果需要更精确的周数计算（如从特定日期开始），可以进一步调整

# 计算函数（保持不变）
def calculate_stop_probability(data, group_by=None):
    results = {}
    
    if group_by:
        grouped = data.groupby(group_by)
    else:
        grouped = [(None, data)]
    
    for group, group_data in grouped:
        group_results = {}
        for machine, col in speed_columns.items():
            stop_count = (group_data[col] == 0).sum()
            total_count = len(group_data)
            stop_prob = stop_count / total_count if total_count > 0 else 0
            group_results[machine] = stop_prob
        
        if group is not None:
            results[group] = group_results
        else:
            results = group_results
    
    return results

# 1. 计算精确到小时的停机概率（按"日期小时"分组）
hourly_prob = calculate_stop_probability(df, '日期小时')

# 其他计算保持不变
daily_prob = calculate_stop_probability(df, '日期')
weekly_prob = calculate_stop_probability(df, '周数')
overall_prob = calculate_stop_probability(df)

# 保存结果到CSV
def save_results_to_csv(results, filename, index_name):
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results.index.name = index_name
    df_results.to_csv(filename, float_format='%.4f')


outpath = r'D:\Code_File\Vinda_cunzhijia\小包机概率计算'

if not os.path.exists(outpath):
    os.makedirs(outpath)


# 保存所有结果
save_results_to_csv(hourly_prob, os.path.join(outpath,'个体事件_每小时停机概率.csv'), '日期小时')
save_results_to_csv(daily_prob, os.path.join(outpath,'个体事件_每日停机概率.csv'), '日期')
save_results_to_csv(weekly_prob, os.path.join(outpath,'个体事件_每周停机概率.csv'), '周数')
pd.DataFrame.from_dict(overall_prob, orient='index', columns=['停机概率'])\
           .to_csv(os.path.join(outpath,'个体事件_全周期停机概率.csv'), float_format='%.4f')

print("结果已保存为：")
print("- 每小时停机概率_精确到日期小时.csv（格式：'YYYY-MM-DD HH:00'）")
print("- 每日停机概率.csv")
print("- 每周停机概率.csv")
print("- 整体停机概率.csv")







#############################
# 同时停机的概率
#############################


# 假设df是已经加载的DataFrame，包含时间戳和各小包机速度数据
# speed_columns是包含各小包机速度列名的字典

# 为每个小包机创建停机状态列（速度为0时停机）
for name, col in speed_columns.items():
    df[f'{name}_停机'] = (df[col] == 0).astype(int)

# 确保时间列是 datetime 类型
df['时间'] = pd.to_datetime(df['时间'])

# 创建时间分组列
df['小时'] = df['时间'].dt.floor('h')  # 按小时分组
df['日期'] = df['时间'].dt.date        # 按日期分组

# 修改周数计算（假设数据的第一天是周的第一天，即周日为一周的第一天）
# 这里我们手动计算周数
first_day = df['时间'].min()
df['周数'] = (df['时间'] - first_day).dt.days // 7 + 1  # 简单按天数划分周，可能需要调整

# 更精确的方法：自定义周数（假设数据的第一天是第1周的第一天）
df['周数'] = (df['时间'] - first_day).dt.days // 7 + 1

# 计算组合概率的函数（按分组）
def calculate_joint_prob_by_group(df, group_col, speed_columns, n_speed_columns):
    results = []
    groups = df.groupby(group_col)
    
    for name, group in groups:
        for combo in combinations(speed_columns, n_speed_columns):
            condition = (group[[f'{m}_停机' for m in combo]].sum(axis=1) == n_speed_columns)
            prob = condition.mean()
            results.append({
                '分组': name,
                '组合': "+".join(combo),
                '概率': prob,
                '同时停机数量': n_speed_columns
            })
    return pd.DataFrame(results)

# 计算全周期概率的函数
def calculate_joint_prob_all(df, speed_columns, n_speed_columns):
    results = []
    for combo in combinations(speed_columns, n_speed_columns):
        condition = (df[[f'{m}_停机' for m in combo]].sum(axis=1) == n_speed_columns)
        prob = condition.mean()
        results.append({
            '分组': '全周期',
            '组合': "+".join(combo),
            '概率': prob,
            '同时停机数量': n_speed_columns
        })
    return pd.DataFrame(results)

# 定义时间尺度及其分组列
time_scales = {
    '每小时': '小时',
    '每天': '日期',
    '每周': '周数'
}

# 为每个时间尺度创建并保存结果
for scale_name, scale_col in time_scales.items():
    scale_results = []
    for n in range(2, len(speed_columns)+1):
        df_result = calculate_joint_prob_by_group(df, scale_col, list(speed_columns.keys()), n)
        df_result['时间尺度'] = scale_name
        scale_results.append(df_result)
    
    # 合并该时间尺度的所有结果
    final_scale_result = pd.concat(scale_results)
    final_scale_result = final_scale_result[['时间尺度', '分组', '同时停机数量', '组合', '概率']]
    
    # 保存到单独文件
    filename = f'同时停机_{scale_name}.csv'
    final_scale_result.to_csv(os.path.join(outpath, filename), 
                            index=False, 
                            float_format='%.4f')
    print(f"{scale_name}结果已保存到 {filename}")

# 处理全周期数据
all_period_results = []
for n in range(2, len(speed_columns)+1):
    df_result = calculate_joint_prob_all(df, list(speed_columns.keys()), n)
    df_result['时间尺度'] = '全周期'
    all_period_results.append(df_result)

# 合并全周期结果
final_all_result = pd.concat(all_period_results)
final_all_result = final_all_result[['时间尺度', '分组', '同时停机数量', '组合', '概率']]

# 保存全周期结果
filename = '同时停机_全周期.csv'
final_all_result.to_csv(os.path.join(outpath, filename), 
                      index=False, 
                      float_format='%.4f')
print(f"全周期结果已保存到 {filename}")

print("\n所有时间尺度的分析结果已保存完毕")