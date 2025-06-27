import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime, timedelta

def load_data(file_path):
    """加载CSV数据文件"""
    try:
        df = pd.read_csv(file_path)
        # 假设时间列名为'timestamp'或类似名称
        # 根据实际数据格式可能需要调整
        time_column = df.columns[0]  # 假设第一列是时间列
        df[time_column] = pd.to_datetime(df[time_column])
        return df
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return None

def load_events(file_path):
    """加载事件数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            events = json.load(f)
        # 将事件时间转换为datetime对象
        for event in events:
            event['timestamp'] = pd.to_datetime(event['timestamp'])
        return events
    except Exception as e:
        print(f"加载事件文件 {file_path} 时出错: {str(e)}")
        return None

def get_events_for_date(events, date):
    """获取指定日期的事件"""
    if not events:
        return []
    daily_events = [event for event in events if event['timestamp'].date() == date]
    print(f"日期 {date}: 找到 {len(daily_events)} 个事件")
    return daily_events

def create_daily_plot(date, data_dict, events):
    """为指定日期创建图表"""
    # 定义状态颜色映射
    STATUS_COLORS = {
        "生产": "rgba(75, 192, 192, 0.3)",  # 绿色
        "停机": "rgba(255, 99, 132, 0.3)",  # 红色
        "待机": "rgba(255, 205, 86, 0.3)",  # 黄色
    }
    
    fig = go.Figure()

    # 添加小包机数据
    for machine_num in range(1, 5):
        if f"{machine_num}#小包机" in data_dict:
            df = data_dict[f"{machine_num}#小包机"]
            mask = (df.iloc[:, 0].dt.date == date)
            fig.add_trace(
                go.Scatter(
                    x=df[mask].iloc[:, 0],
                    y=df[mask].iloc[:, 1],
                    name=f"{machine_num}#小包机速度",
                    mode='lines',
                    customdata=df[mask].iloc[:, 0].dt.strftime('%Y-%m-%d %H:%M:%S'),
                    hovertemplate=f"{machine_num}#小包机实际速度: %{{y:.2f}}<extra></extra>"
                )
            )

    # 添加折叠机数据
    if "折叠机" in data_dict:
        df = data_dict["折叠机"]
        mask = (df.iloc[:, 0].dt.date == date)
        fig.add_trace(
            go.Scatter(
                x=df[mask].iloc[:, 0],
                y=df[mask].iloc[:, 1],
                name="折叠机速度",
                mode='lines',
                customdata=df[mask].iloc[:, 0].dt.strftime('%Y-%m-%d %H:%M:%S'),
                hovertemplate='折叠机实际速度: %{y:.2f}<extra></extra>'
            )
        )

    # 添加裁切机数据
    if "裁切机" in data_dict:
        df = data_dict["裁切机"]
        mask = (df.iloc[:, 0].dt.date == date)
        fig.add_trace(
            go.Scatter(
                x=df[mask].iloc[:, 0],
                y=df[mask].iloc[:, 1],
                name="裁切机速度",
                mode='lines',
                customdata=df[mask].iloc[:, 0].dt.strftime('%Y-%m-%d %H:%M:%S'),
                hovertemplate='裁切机实际速度: %{y:.2f}<extra></extra>'
            )
        )

    # 添加存纸率数据 - 使用次坐标轴
    if "存纸率" in data_dict:
        df = data_dict["存纸率"]
        mask = (df.iloc[:, 0].dt.date == date)
        fig.add_trace(
            go.Scatter(
                x=df[mask].iloc[:, 0],
                y=df[mask].iloc[:, 1],
                name="存纸率",
                mode='lines',
                yaxis="y2",
                line=dict(width=3, dash='dot'),  # 使用虚线以区分
                customdata=df[mask].iloc[:, 0].dt.strftime('%Y-%m-%d %H:%M:%S'),
                hovertemplate='当前存纸率: %{y:.2f}%<extra></extra>'
            )
        )

    # 添加事件标记
    daily_events = get_events_for_date(events, date)
    if daily_events:
        print(f"处理日期 {date} 的事件标记")
        # 获取y轴范围
        y_values = []
        for trace in fig.data:
            if trace.yaxis != "y2":  # 排除存纸率数据
                y_values.extend(trace.y)
        
        if y_values:  # 确保有数据
            y_min = min(y_values)
            y_max = max(y_values)
            print(f"Y轴范围: {y_min} - {y_max}")
            
            # 为每个事件添加垂直线
            for event in daily_events:
                print(f"添加事件标记: {event['timestamp']} - {event['device']} - {event['status']}")
                # 获取状态对应的颜色
                color = STATUS_COLORS.get(event['status'], "rgba(128, 128, 128, 0.3)")  # 默认灰色
                
                # 添加事件线
                fig.add_shape(
                    type="line",
                    x0=event['timestamp'],
                    x1=event['timestamp'],
                    y0=y_min,
                    y1=y_max,
                    line=dict(
                        color=color,
                        width=1,
                        dash="dash"
                    ),
                    layer="above"
                )
                
                # 添加事件标记点
                fig.add_trace(
                    go.Scatter(
                        x=[event['timestamp']],
                        y=[y_max],
                        mode='markers',
                        name=f"事件 - {event['device']}",
                        marker=dict(
                            symbol="circle",
                            size=8,
                            color=color.replace("0.3", "1")  # 使用不透明的颜色
                        ),
                        showlegend=False,
                        hovertemplate=(
                            f"时间: {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}<br>" +
                            f"● {event['device']}: {event['status']} ({event['speed']:.1f})<extra></extra>"
                        )
                    )
                )
        else:
            print(f"警告: 日期 {date} 没有有效的Y轴数据")
    else:
        print(f"日期 {date} 没有事件")

    # 更新布局
    fig.update_layout(
        title=f"设备运行数据 - {date.strftime('%Y-%m-%d')}",
        xaxis=dict(
            title="时间",
            showspikes=True,  # 显示垂直参考线
            spikemode="across",
            spikesnap="cursor",
            showline=True,
            showgrid=True,
            spikedash="solid"
        ),
        yaxis=dict(
            title="速度",
            showspikes=True,  # 显示水平参考线
            showline=True,
            showgrid=True,
            spikedash="solid"
        ),
        yaxis2=dict(
            title="存纸率 (%)",
            overlaying="y",
            side="right",
            showspikes=True,
            showline=True,
            showgrid=False,
            spikedash="solid"
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        hovermode="x unified",  # 统一显示所有曲线在当前x位置的值
        hoverdistance=100,  # 鼠标与数据点的最大距离
        spikedistance=1000,  # 参考线的最大距离
        height=800,  # 增加图表高度
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        template='plotly_white',
        margin=dict(t=100, l=50, r=50, b=50)
    )

    return fig

def main():
    # 数据文件路径
    data_files = {
        "1#小包机": "./各点位/1#小包机实际速度.csv",
        "2#小包机": "./各点位/2#小包机实际速度.csv",
        "3#小包机": "./各点位/3#小包机实际速度.csv",
        "4#小包机": "./各点位/4#小包机实际速度.csv",
        "折叠机": "./各点位/折叠机实际速度.csv",
        "裁切机": "./各点位/裁切机实际速度.csv",
        "存纸率": "./各点位/存纸率.csv"
    }

    # 加载事件数据
    events = load_events("./script/merged_all_events.json")
    if events:
        print(f"成功加载事件数据，共 {len(events)} 条事件")
    else:
        print("警告: 无法加载事件数据，将继续生成图表但不显示事件标记")

    # 加载所有数据
    data_dict = {}
    for name, file_path in data_files.items():
        df = load_data(file_path)
        if df is not None:
            data_dict[name] = df
            print(f"成功加载 {name} 数据")

    if not data_dict:
        print("没有成功加载任何数据文件")
        return

    # 创建输出目录
    output_dir = "daily_plots"
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有数据的日期范围
    all_dates = set()
    for df in data_dict.values():
        dates = df.iloc[:, 0].dt.date.unique()
        all_dates.update(dates)
    
    print(f"找到 {len(all_dates)} 天的数据")
    
    # 按日期生成图表
    for date in sorted(all_dates):
        fig = create_daily_plot(date, data_dict, events)
        output_file = os.path.join(output_dir, f"daily_plot_{date.strftime('%Y-%m-%d')}.html")
        fig.write_html(output_file)
        print(f"生成文件: {output_file}")

    # 创建索引页面
    create_index_page(output_dir, sorted(all_dates))
    print(f"生成索引页面: {os.path.join(output_dir, 'index.html')}")

def create_index_page(output_dir, dates):
    """创建索引页面，列出所有日期的链接"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>设备运行数据日报表</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 { 
                color: #333;
                text-align: center;
                padding: 20px 0;
                border-bottom: 2px solid #eee;
            }
            .date-list { 
                list-style: none; 
                padding: 0;
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 15px;
            }
            .date-list li { 
                margin: 5px 0;
            }
            .date-list a {
                text-decoration: none;
                color: #0066cc;
                padding: 10px 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                display: block;
                text-align: center;
                transition: all 0.3s ease;
            }
            .date-list a:hover { 
                background-color: #f0f0f0;
                transform: translateY(-2px);
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <h1>设备运行数据日报表</h1>
        <ul class="date-list">
    """
    
    for date in dates:
        file_name = f"daily_plot_{date.strftime('%Y-%m-%d')}.html"
        html_content += f'<li><a href="{file_name}">{date.strftime("%Y-%m-%d")}</a></li>\n'
    
    html_content += """
        </ul>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

if __name__ == "__main__":
    main()
