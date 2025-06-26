import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

# 选择数据文件
data_file = r'D:\Code_File\Vinda_cunzhijia_v7\存纸架数据汇总.csv'  # 修改为你要使用的数据文件

# 读取数据
df = pd.read_csv(data_file)

# 转换时间列为datetime类型
df['时间'] = pd.to_datetime(df['时间'])

# 指定时间段
start_time = '2025-05-02 13:56:09'  # 修改为你要的开始时间
end_time = '2025-05-02 14:06:09'    # 修改为你要的结束时间

# 指定要绘制的列
columns_to_plot = ['折叠机实际速度','存纸率', '1#瞬时切数', '2#瞬时切数', '3#瞬时切数', '4#瞬时切数', 
                #    '1#小包机入包数', '2#小包机入包数', '3#小包机入包数', '4#小包机入包数',
                   '1#小包机实际速度', '2#小包机实际速度', '3#小包机主机实际速度', '4#小包机主机实际速度']  # 修改为你要绘制的列

# 过滤时间段内的数据
start_time = pd.to_datetime(start_time)
end_time = pd.to_datetime(end_time)
filtered_df = df[(df['时间'] >= start_time) & (df['时间'] <= end_time)]

# 创建plotly图表
fig = go.Figure()

# 绘制每一列
for column in columns_to_plot:
    if column in filtered_df.columns:
        # 处理缺失值
        valid_data = filtered_df.dropna(subset=[column])
        if len(valid_data) > 0:
            fig.add_trace(go.Scatter(
                x=valid_data['时间'],
                y=valid_data[column],
                mode='lines+markers',
                name=column,
                line=dict(width=2),
                marker=dict(size=4)
            ))

# 设置图表布局
fig.update_layout(
    title=f'时间段数据图表 ({start_time.strftime("%Y-%m-%d %H:%M")} - {end_time.strftime("%Y-%m-%d %H:%M")})',
    title_font_size=16,
    xaxis_title='时间',
    yaxis_title='数值',
    font_size=12,
    width=1200,
    height=600,
    showlegend=True,
    hovermode='x unified',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        tickformat='%m-%d %H:%M'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)'
    ),
    plot_bgcolor='white'
)

# 保存为HTML文件
fig.write_html('时间段数据图表.html')

# 显示图表
fig.show()

# 打印统计信息
print(f"时间段: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%Y-%m-%d %H:%M')}")
print(f"数据点数量: {len(filtered_df)}")
print(f"绘制的列: {columns_to_plot}")
print(f"每列的统计信息:")
for column in columns_to_plot:
    if column in filtered_df.columns:
        valid_data = filtered_df[column].dropna()
        if len(valid_data) > 0:
            print(f"  {column}: 最小值={valid_data.min():.2f}, 最大值={valid_data.max():.2f}, 平均值={valid_data.mean():.2f}") 