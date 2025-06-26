import pandas as pd

# 读取CSV文件
df = pd.read_csv('存纸架数据汇总.csv')

# 打印列名
print("CSV文件的列名：")
for col in df.columns:
    print(col) 