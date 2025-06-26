import pandas as pd
from datetime import datetime

# 读取文件1
df1 = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\高低速时间段\low_speed_periods.csv', parse_dates=['开始时间', '结束时间'])

# 读取文件2
df2 = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv', parse_dates=['时间'])

# 创建一个空的DataFrame来保存结果
result_df = pd.DataFrame(columns=df1.columns)

# 遍历文件1中的每一行
for index, row in df1.iterrows():
    start_time = row['开始时间']
    end_time = row['结束时间']
    
    # 在文件2中找到对应时间段的数据
    mask = (df2['时间'] >= start_time) & (df2['时间'] <= end_time)
    filtered_df2 = df2.loc[mask]
    
    # 计算存纸率的平均值
    if not filtered_df2.empty:
        avg_paper_rate = filtered_df2['存纸率'].mean()
        
        # 如果平均值大于20，则将文件1的该行添加到结果DataFrame中
        if avg_paper_rate > 20:
            result_df = pd.concat([result_df, row.to_frame().T], ignore_index=True)

# 将结果保存到新的CSV文件
if not result_df.empty:
    result_df.to_csv(r'D:\Code_File\Vinda_cunzhijia\高低速时间段\low_speed_periods_v1.csv', index=False)
    print("筛选完成，结果已保存到'筛选结果.csv'")
else:
    print("没有找到满足条件的记录")