import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# 读取时间序列数据
ts_data = pd.read_csv('时间序列文件.csv', parse_dates=['时间'])
# 读取时间段数据
periods_data = pd.read_csv('时间段文件.csv', parse_dates=['开始时间', '结束时间'])

# 指定要绘制的列
target_column = '折叠机实际速度'

# 设置图形样式
plt.style.use('ggplot')

# 为每个时间段创建单独的图
for idx, period in periods_data.iterrows():
    start_time = period['开始时间']
    end_time = period['结束时间']
    description = period['描述']
    
    # 筛选时间范围内的数据
    mask = (ts_data['时间'] >= start_time) & (ts_data['时间'] <= end_time)
    period_data = ts_data.loc[mask].copy()
    
    if len(period_data) == 0:
        print(f"时间段 {idx+1} 没有数据可绘制")
        continue
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制数据
    ax.plot(period_data['时间'], period_data[target_column], 
            label=f'{target_column}', color='blue')
    
    # 设置标题和标签
    ax.set_title(f"{target_column} - {description}\n{start_time} 至 {end_time}")
    ax.set_xlabel('时间')
    ax.set_ylabel(target_column)
    
    # 格式化x轴
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, len(period_data)//10)))
    plt.xticks(rotation=45)
    
    # 添加网格和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'时间段_{idx+1}_{target_column}.png', dpi=300)
    plt.close()

# 创建一天的综合图
fig, ax = plt.subplots(figsize=(15, 8))

# 绘制全天的数据
ax.plot(ts_data['时间'], ts_data[target_column], 
        label=f'全天 {target_column}', color='blue', alpha=0.5)

# 标记每个时间段
for idx, period in periods_data.iterrows():
    start_time = period['开始时间']
    end_time = period['结束时间']
    description = period['描述']
    
    # 在图上标记时间段
    ax.axvspan(start_time, end_time, color='red', alpha=0.3, 
               label=f'时间段 {idx+1}: {description}')
    
    # 添加时间段标签
    ax.text(start_time + (end_time - start_time)/2, 
            ts_data[target_column].max() * 0.95, 
            f'时间段 {idx+1}', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8))

# 设置标题和标签
ax.set_title(f"{target_column} - 全天数据 ({ts_data['时间'].dt.date[0]})")
ax.set_xlabel('时间')
ax.set_ylabel(target_column)

# 格式化x轴
ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
plt.xticks(rotation=45)

# 添加网格和图例
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f'全天_{target_column}_综合图.png', dpi=300, bbox_inches='tight')
plt.close()

print("绘图完成！")