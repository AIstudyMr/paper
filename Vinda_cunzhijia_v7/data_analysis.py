import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据文件"""
    try:
        # 读取调整后的时间段数据
        time_periods = pd.read_csv('折叠机正常运行且高存纸率时间段_最终结果_存纸率1_调整后.csv')
        print("时间段数据列：", time_periods.columns.tolist())
        print("时间段数据形状：", time_periods.shape)
        print("前5行时间段数据：")
        print(time_periods.head())
        
        # 读取汇总数据
        summary_data = pd.read_csv('存纸架数据汇总.csv', encoding='utf-8-sig')
        print("\n汇总数据列：", summary_data.columns.tolist())
        print("汇总数据形状：", summary_data.shape)
        print("前5行汇总数据：")
        print(summary_data.head())
        
        return time_periods, summary_data
    except Exception as e:
        print(f"读取数据时出错：{e}")
        return None, None

def convert_time_columns(time_periods, summary_data):
    """转换时间列为datetime格式"""
    try:
        # 转换时间段数据的时间列
        time_periods['开始时间'] = pd.to_datetime(time_periods['开始时间'])
        time_periods['结束时间'] = pd.to_datetime(time_periods['结束时间'])
        
        # 转换汇总数据的时间列（假设第一列是时间）
        time_col = summary_data.columns[0]  # 第一列应该是时间
        summary_data[time_col] = pd.to_datetime(summary_data[time_col])
        
        return time_periods, summary_data, time_col
    except Exception as e:
        print(f"转换时间格式时出错：{e}")
        return None, None, None

def find_matching_data(time_periods, summary_data, time_col):
    """根据时间段找出对应的汇总数据"""
    matching_data = []
    
    for idx, period in time_periods.iterrows():
        start_time = period['开始时间']
        end_time = period['结束时间']
        
        # 筛选在时间段内的数据
        mask = (summary_data[time_col] >= start_time) & (summary_data[time_col] <= end_time)
        period_data = summary_data[mask].copy()
        
        if not period_data.empty:
            period_data['时间段_索引'] = idx
            period_data['时间段_开始'] = start_time
            period_data['时间段_结束'] = end_time
            matching_data.append(period_data)
            
    if matching_data:
        all_matching_data = pd.concat(matching_data, ignore_index=True)
        return all_matching_data
    else:
        return pd.DataFrame()

def calculate_iot_delays(data, time_col):
    """计算IoT各点位之间的延时时间"""
    # 找出可能的IoT设备列（通常包含设备编号或名称）
    device_columns = []
    
    # 分析列名，寻找设备相关的列
    for col in data.columns:
        if any(keyword in str(col) for keyword in ['机', '架', '切', '包', '折', '速度', '计数', '入包', '出包']):
            device_columns.append(col)
    
    print(f"识别到的设备相关列：{len(device_columns)}个")
    print(device_columns[:10])  # 显示前10个
    
    # 计算时间间隔（相邻记录之间的时间差）
    if len(data) > 1:
        data = data.sort_values(time_col)
        data['时间间隔'] = data[time_col].diff().dt.total_seconds()
        
        # 统计延时信息
        delay_stats = {
            '平均延时(秒)': data['时间间隔'].mean(),
            '最小延时(秒)': data['时间间隔'].min(),
            '最大延时(秒)': data['时间间隔'].max(),
            '标准差(秒)': data['时间间隔'].std()
        }
        
        return delay_stats, device_columns
    else:
        return None, device_columns

def analyze_device_sequence(data, device_columns, time_col):
    """分析设备间的数据传输序列"""
    sequence_analysis = {}
    
    # 选择几个关键设备列进行分析
    key_devices = device_columns[:5] if len(device_columns) > 5 else device_columns
    
    for device in key_devices:
        if device in data.columns:
            # 计算该设备数据的变化
            device_data = data[[time_col, device]].copy()
            device_data['变化量'] = device_data[device].diff()
            device_data['有变化'] = device_data['变化量'] != 0
            
            # 找出数据变化的时间点
            change_points = device_data[device_data['有变化']]
            
            if len(change_points) > 1:
                # 计算变化间隔
                change_intervals = change_points[time_col].diff().dt.total_seconds()
                sequence_analysis[device] = {
                    '变化次数': len(change_points) - 1,
                    '平均变化间隔(秒)': change_intervals.mean(),
                    '变化间隔标准差': change_intervals.std()
                }
    
    return sequence_analysis

def main():
    print("开始分析存纸架数据...")
    
    # 加载数据
    time_periods, summary_data = load_data()
    if time_periods is None or summary_data is None:
        return
    
    # 转换时间格式
    time_periods, summary_data, time_col = convert_time_columns(time_periods, summary_data)
    if time_periods is None:
        return
    
    print(f"\n使用时间列：{time_col}")
    
    # 找出匹配的数据
    matching_data = find_matching_data(time_periods, summary_data, time_col)
    print(f"\n找到匹配数据：{len(matching_data)}条记录")
    
    if matching_data.empty:
        print("没有找到匹配的数据")
        return
    
    # 计算IoT延时
    delay_stats, device_columns = calculate_iot_delays(matching_data, time_col)
    
    if delay_stats:
        print("\n=== IoT数据传输延时分析 ===")
        for key, value in delay_stats.items():
            print(f"{key}: {value:.2f}")
    
    # 分析设备序列
    sequence_analysis = analyze_device_sequence(matching_data, device_columns, time_col)
    
    print("\n=== 设备数据变化分析 ===")
    for device, stats in sequence_analysis.items():
        print(f"\n设备：{device}")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # 保存结果
    output_file = "iot_delay_analysis_results.csv"
    matching_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n匹配的数据已保存到：{output_file}")
    
    print("\n分析完成！")

if __name__ == "__main__":
    main() 