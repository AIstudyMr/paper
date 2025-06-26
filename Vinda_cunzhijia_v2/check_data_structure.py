import pandas as pd

# 读取CSV文件的前几行
df = pd.read_csv('存纸架数据汇总.csv', nrows=10)

# 打印列名
print("CSV文件的列名：")
for i, col in enumerate(df.columns):
    print(f"{i+1}: {col}")

print("\n数据样本：")
print(df.head())

print("\n数据类型：")
print(df.dtypes)

# 查看时间范围
if '时间' in df.columns:
    df['时间'] = pd.to_datetime(df['时间'])
    print(f"\n时间范围: {df['时间'].min()} 到 {df['时间'].max()}") 