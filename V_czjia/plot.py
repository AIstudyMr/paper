import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


file_path = r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\折叠机实际速度 .csv'

df = pd.read_csv(file_path)

# 绘制曲线图
plt.figure(figsize=(12, 6))  # 设置图像大小

# 第一列作为X轴（时间），第二列作为Y轴
plt.plot(df.iloc[:, 1].values,
        #  df.iloc[:, 0], df.iloc[:, 1], 
         linewidth=0.5, 
         color='blue',
         marker='o',  # 添加数据点标记
         markersize=1,
         linestyle='-')

# 添加图表标题和标签
plt.title('时间序列数据可视化', fontsize=16, pad=20)
plt.xlabel(df.columns[0], fontsize=12)  # 使用第一列名作为X轴标签
plt.ylabel(df.columns[1], fontsize=12)  # 使用第二列名作为Y轴标签

# 优化坐标轴
plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
plt.tight_layout()  # 自动调整布局

# 显示图表
plt.show()

