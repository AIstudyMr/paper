import pandas as pd

# 只读取前几行以获取列结构，节省内存
print("正在读取CSV文件以检查列结构...")
df_sample = pd.read_csv('存纸架数据汇总.csv', nrows=5)

# 打印列名和样例数据
print("\nCSV文件列名:", df_sample.columns.tolist())
print("\n数据样例:")
print(df_sample.head()) 