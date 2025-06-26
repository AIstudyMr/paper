import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data():
    print("正在读取数据...")
    df = pd.read_csv('存纸架数据汇总.csv')
    df['时间'] = pd.to_datetime(df['时间'])
    print(f"数据时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
    return df

def get_machine_stop_periods(df, machine_number):
    """获取指定机器的停机时间段"""
    speed_col = f"{machine_number}#小包机实际速度" if machine_number != 3 and machine_number != 4 else f"{machine_number}#小包机主机实际速度"
    
    # 标记停机状态（速度为0）
    df['is_stopped'] = df[speed_col] == 0
    
    # 创建停机组
    df['stop_group'] = (df['is_stopped'] != df['is_stopped'].shift()).cumsum()
    
    # 获取停机时间段
    stop_periods = []
    for _, group in df[df['is_stopped']].groupby('stop_group'):
        start_time = group['时间'].iloc[0]
        end_time = group['时间'].iloc[-1]
        duration = (end_time - start_time).total_seconds()
        if duration > 10:  # 只考虑超过10秒的停机
            stop_periods.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            })
    
    return stop_periods

def check_concurrent_stops(periods1, periods2):
    """检查两组停机时间段是否有重叠"""
    concurrent_stops = []
    
    for p1 in periods1:
        for p2 in periods2:
            # 检查时间段是否重叠
            if not (p1['end_time'] < p2['start_time'] or p2['end_time'] < p1['start_time']):
                # 计算重叠时间段
                overlap_start = max(p1['start_time'], p2['start_time'])
                overlap_end = min(p1['end_time'], p2['end_time'])
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                
                if overlap_duration > 10:  # 只考虑重叠超过10秒的情况
                    concurrent_stops.append({
                        'start_time': overlap_start,
                        'end_time': overlap_end,
                        'duration': overlap_duration
                    })
    
    return concurrent_stops

def analyze_concurrent_stops(df):
    # 获取每台机器的停机时间段
    machine_stops = {}
    for machine_number in [1, 2, 3, 4]:
        machine_stops[machine_number] = get_machine_stop_periods(df.copy(), machine_number)
    
    total_time = (df['时间'].max() - df['时间'].min()).total_seconds()
    
    # 分析结果
    results = {
        'single_machine': {},  # 单机停机概率
        'two_machines': {},    # 两机同时停机概率
        'three_machines': {},  # 三机同时停机概率
        'four_machines': None  # 四机同时停机概率
    }
    
    # 1. 计算单机停机后其他机器的停机概率
    for m1 in [1, 2, 3, 4]:
        for m2 in [1, 2, 3, 4]:
            if m1 != m2:
                concurrent = check_concurrent_stops(machine_stops[m1], machine_stops[m2])
                total_concurrent_time = sum(stop['duration'] for stop in concurrent)
                probability = total_concurrent_time / total_time
                results['single_machine'][f"{m1}停机后{m2}停机概率"] = probability
    
    # 2. 计算两机同时停机概率
    for m1 in [1, 2, 3, 4]:
        for m2 in range(m1 + 1, 5):
            concurrent = check_concurrent_stops(machine_stops[m1], machine_stops[m2])
            total_concurrent_time = sum(stop['duration'] for stop in concurrent)
            probability = total_concurrent_time / total_time
            results['two_machines'][f"{m1}#{m2}#同时停机概率"] = probability
    
    # 3. 计算三机同时停机概率
    for m1 in [1, 2, 3, 4]:
        for m2 in range(m1 + 1, 5):
            for m3 in range(m2 + 1, 5):
                # 先找出两台机器的同时停机
                concurrent_2 = check_concurrent_stops(machine_stops[m1], machine_stops[m2])
                # 再检查第三台机器与前两台的重叠
                concurrent_3 = []
                for period in concurrent_2:
                    temp_period = {'start_time': period['start_time'], 
                                 'end_time': period['end_time'],
                                 'duration': period['duration']}
                    concurrent_3.extend(check_concurrent_stops([temp_period], machine_stops[m3]))
                
                total_concurrent_time = sum(stop['duration'] for stop in concurrent_3)
                probability = total_concurrent_time / total_time
                results['three_machines'][f"{m1}#{m2}#{m3}#同时停机概率"] = probability
    
    # 4. 计算四机同时停机概率
    concurrent_4 = []
    concurrent_123 = check_concurrent_stops(
        check_concurrent_stops(machine_stops[1], machine_stops[2]),
        machine_stops[3]
    )
    for period in concurrent_123:
        temp_period = {'start_time': period['start_time'],
                      'end_time': period['end_time'],
                      'duration': period['duration']}
        concurrent_4.extend(check_concurrent_stops([temp_period], machine_stops[4]))
    
    total_concurrent_time = sum(stop['duration'] for stop in concurrent_4)
    results['four_machines'] = total_concurrent_time / total_time
    
    return results

def save_results_to_csv(results):
    # 保存单机触发概率
    single_machine_data = []
    for key, value in results['single_machine'].items():
        # 从key中提取触发机器和目标机器
        trigger_machine = key[0]  # 获取第一个字符作为触发机器号
        target_machine = key[key.find('后')+1]  # 获取"后"字后面的机器号
        single_machine_data.append({
            '触发机器': f"{trigger_machine}#机",
            '目标机器': f"{target_machine}#机",
            '概率': f"{value:.2%}"
        })
    pd.DataFrame(single_machine_data).to_csv(r'D:\Code_File\Vinda_cunzhijia_v2\小包机概率计算\单机触发停机概率.csv', index=False, encoding='utf-8-sig')
    
    # 保存多机同时停机概率
    multi_machine_data = []
    
    # 添加两机同时停机数据
    for key, value in results['two_machines'].items():
        machines = key.split('同时')[0]  # 获取机器组合部分
        first_machine = machines[0]  # 第一台机器作为触发机器
        multi_machine_data.append({
            '停机类型': '两机同时停机',
            '触发机器': f"{first_machine}#机",
            '停机组合': machines,
            '概率': f"{value:.2%}"
        })
    
    # 添加三机同时停机数据
    for key, value in results['three_machines'].items():
        machines = key.split('同时')[0]  # 获取机器组合部分
        first_machine = machines[0]  # 第一台机器作为触发机器
        multi_machine_data.append({
            '停机类型': '三机同时停机',
            '触发机器': f"{first_machine}#机",
            '停机组合': machines,
            '概率': f"{value:.2%}"
        })
    
    # 添加四机同时停机数据
    multi_machine_data.append({
        '停机类型': '四机同时停机',
        '触发机器': '1#机',
        '停机组合': '1#2#3#4#',
        '概率': f"{results['four_machines']:.2%}"
    })
    
    pd.DataFrame(multi_machine_data).to_csv(r'D:\Code_File\Vinda_cunzhijia_v2\小包机概率计算\多机同时停机概率.csv', index=False, encoding='utf-8-sig')
    
    print("\n分析结果已保存到CSV文件：")
    print("1. 单机触发停机概率.csv")
    print("2. 多机同时停机概率.csv")

def main():
    try:
        df = load_data()
        results = analyze_concurrent_stops(df)
        
        # 打印结果
        print("\n=== 单机停机触发其他机器停机的概率 ===")
        for key, value in results['single_machine'].items():
            print(f"{key}: {value:.2%}")
        
        print("\n=== 两机同时停机概率 ===")
        for key, value in results['two_machines'].items():
            print(f"{key}: {value:.2%}")
        
        print("\n=== 三机同时停机概率 ===")
        for key, value in results['three_machines'].items():
            print(f"{key}: {value:.2%}")
        
        print("\n=== 四机同时停机概率 ===")
        print(f"所有机器同时停机概率: {results['four_machines']:.2%}")
        
        # 保存结果到CSV
        save_results_to_csv(results)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 