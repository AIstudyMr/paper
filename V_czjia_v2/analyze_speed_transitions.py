#########################
#  找出4个小包机启停事件
##########################


import pandas as pd
import numpy as np
import os
from datetime import datetime

# 定义速度阈值
STOP_SPEED_MIN = 0
STOP_SPEED_MAX = 24
IDLE_SPEED_MIN = 25
IDLE_SPEED_MAX = 26
NORMAL_SPEED_MIN = 27

# 定义小包机速度列名


MACHINE_SPEED_COLUMNS = {
    1: '1#小包机实际速度',
    2: '2#小包机实际速度',
    3: '3#小包机主机实际速度',
    4: '4#小包机主机实际速度'
}

def create_result_dirs():
    """创建结果保存目录"""
    base_dirs = []
    for machine_no in range(1, 5):
        machine_dir = f"{machine_no}号小包机启停分析结果"
        if not os.path.exists(machine_dir):
            os.makedirs(machine_dir)
        base_dirs.append(machine_dir)
    return base_dirs

def calculate_slope(data):
    """计算数据的斜率"""
    try:
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return slope
    except Exception as e:
        print(f"计算斜率时出错: {e}")
        return 0

def analyze_speed_transitions(df, window_size=20, slope_threshold=0.1):
    """分析速度变化事件"""
    results = {
        "停止到待机": [],
        "停止到正常生产": [],
        "正常生产到停止": [],
        "正常生产到待机": [],
        "待机到正常生产": [],
        "待机到停止": []  # 新增事件类型
    }
    
    try:
        half_window = window_size // 2
        
        for i in range(len(df) - window_size + 1):
            try:
                window_data = df['speed'].iloc[i:i+window_size].values
                first_half = window_data[:half_window]
                second_half = window_data[half_window:]
                
                first_slope = calculate_slope(first_half)
                second_slope = calculate_slope(second_half)
                
                first_mean = np.mean(first_half)
                second_mean = np.mean(second_half)
                
                # 获取事件发生时间点和对应的速度值
                event_time = df.index[i+half_window]
                event_speed = df['speed'].iloc[i+half_window]
                
                # 低速到高速变化
                if (abs(second_slope) < slope_threshold and first_slope > slope_threshold):
                    if STOP_SPEED_MIN < first_mean <= STOP_SPEED_MAX:  # 从停止状态开始
                        if IDLE_SPEED_MIN <= second_mean < IDLE_SPEED_MAX:
                            results["停止到待机"].append((event_time, event_speed))
                        elif second_mean >= NORMAL_SPEED_MIN:
                            results["停止到正常生产"].append((event_time, event_speed))
                    elif IDLE_SPEED_MIN <= first_mean < IDLE_SPEED_MAX:  # 从待机状态开始
                        if second_mean >= NORMAL_SPEED_MIN:
                            results["待机到正常生产"].append((event_time, event_speed))
                
                # 高速到低速变化
                if (abs(first_slope) < slope_threshold and second_slope < -slope_threshold):
                    if first_mean >= NORMAL_SPEED_MIN:  # 从正常生产状态开始
                        if STOP_SPEED_MIN < second_mean <= STOP_SPEED_MAX:
                            results["正常生产到停止"].append((event_time, event_speed))
                        elif IDLE_SPEED_MIN <= second_mean < IDLE_SPEED_MAX:
                            results["正常生产到待机"].append((event_time, event_speed))
                    elif IDLE_SPEED_MIN <= first_mean < IDLE_SPEED_MAX:  # 从待机状态开始
                        if STOP_SPEED_MIN < second_mean <= STOP_SPEED_MAX:
                            results["待机到停止"].append((event_time, event_speed))
            
            except Exception as e:
                print(f"处理窗口 {i} 时出错: {e}")
                continue
        
        return results
    except Exception as e:
        print(f"分析速度变化时出错: {e}")
        return results

def analyze_machine(df, machine_no, result_dir):
    """分析单个小包机的数据"""
    print(f"\n开始分析{machine_no}号小包机...")
    
    # 重命名列以便于后续处理
    df = df.rename(columns={MACHINE_SPEED_COLUMNS[machine_no]: 'speed'})
    
    # 进行启停分析
    results = analyze_speed_transitions(df)
    
    # 保存结果
    for event_type, event_data in results.items():
        if event_data:
            # 解析时间和速度数据
            timestamps = [item[0] for item in event_data]
            speeds = [item[1] for item in event_data]
            
            # 创建DataFrame
            result_df = pd.DataFrame({
                '时间': timestamps,
                '速度': speeds
            })
            
            output_file = os.path.join(result_dir, f"{event_type}.csv")
            result_df.to_csv(output_file, index=False)
            print(f"{machine_no}号小包机 {event_type}事件数量: {len(event_data)}")

def main():
    try:
        print("开始读取数据...")
        
        # 创建结果目录
        result_dirs = create_result_dirs()
        
        # 准备所需的列名
        columns = ['时间'] + list(MACHINE_SPEED_COLUMNS.values())
        
        # 读取数据
        df = pd.read_csv("存纸架数据汇总.csv", usecols=columns)
        print(f"数据读取完成，共 {len(df)} 行")
        
        # 转换时间戳并设置为索引
        df['时间'] = pd.to_datetime(df['时间'])
        df = df.sort_values('时间')
        df = df.set_index('时间')
        
        # 分析每台机器
        for machine_no in range(1, 5):
            machine_data = df[[MACHINE_SPEED_COLUMNS[machine_no]]].copy()
            analyze_machine(machine_data, machine_no, result_dirs[machine_no-1])
        
        print("\n分析完成！")
    
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main() 