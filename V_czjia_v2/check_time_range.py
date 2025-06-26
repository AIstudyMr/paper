import pandas as pd

# 读取CSV文件
print("正在读取数据文件，请稍候...")
df = pd.read_csv('存纸架数据汇总.csv', usecols=['时间', '裁切机实际速度'])

print(f"数据行数: {len(df)}")

# 转换时间格式
df['时间'] = pd.to_datetime(df['时间'])

# 查看时间范围
print(f"时间范围: {df['时间'].min()} 到 {df['时间'].max()}")

# 计算时间跨度
time_span = df['时间'].max() - df['时间'].min()
print(f"总时间跨度: {time_span}")

# 按日期分组，查看每天的数据量
df['日期'] = df['时间'].dt.date
daily_counts = df.groupby('日期').size()
print(f"\n每日数据点数量:")
for date, count in daily_counts.items():
    print(f"{date}: {count} 个数据点")

# 查看裁切机速度的基本统计
print(f"\n裁切机实际速度统计:")
print(df['裁切机实际速度'].describe()) 