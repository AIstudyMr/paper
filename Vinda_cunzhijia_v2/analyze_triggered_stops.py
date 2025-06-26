import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations

def load_data():
    print("正在读取数据...")
    df = pd.read_csv('存纸架数据汇总.csv')
    df['时间'] = pd.to_datetime(df['时间'])
    print(f"数据时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
    return df

def get_machine_stops(df, machine_number):
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
        if duration >= 10:  # 只考虑超过30秒的停机作为触发
            stop_periods.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            })
    
    return stop_periods

def check_subsequent_stops(trigger_periods, target_df, target_machines):
    """检查目标机器在触发机器停机后的停机情况"""
    subsequent_stops = []
    
    for period in trigger_periods:
        # 获取触发停机后30秒内的数据
        window_start = period['start_time']
        window_end = period['end_time'] + timedelta(seconds=30)
        window_data = target_df[(target_df['时间'] >= window_start) & 
                              (target_df['时间'] <= window_end)].copy()
        
        if window_data.empty:
            continue
            
        # 检查每个目标机器的停机情况
        machine_stops = {}
        for machine in target_machines:
            speed_col = f"{machine}#小包机实际速度" if machine != 3 and machine != 4 else f"{machine}#小包机主机实际速度"
            machine_data = window_data.copy()
            machine_data['is_stopped'] = machine_data[speed_col] == 0
            machine_data['stop_group'] = (machine_data['is_stopped'] != 
                                        machine_data['is_stopped'].shift()).cumsum()
            
            # 获取该机器的停机时段
            stops = []
            for _, group in machine_data[machine_data['is_stopped']].groupby('stop_group'):
                stop_start = group['时间'].iloc[0]
                stop_end = group['时间'].iloc[-1]
                stop_duration = (stop_end - stop_start).total_seconds()
                
                if stop_duration > 10:  # 目标机器停机超过10秒
                    stops.append({
                        'start_time': stop_start,
                        'end_time': stop_end,
                        'duration': stop_duration
                    })
            
            if stops:  # 如果有停机记录
                machine_stops[machine] = stops
        
        # 检查是否所有目标机器都有停机记录
        if len(machine_stops) == len(target_machines):
            subsequent_stops.append({
                'trigger_start': window_start,
                'machine_stops': machine_stops
            })
    
    return subsequent_stops

def analyze_triggered_stops(df):
    results = {}
    total_periods = {}
    
    # 获取每台机器的停机时间段（作为触发）
    for trigger_machine in [1, 2, 3, 4]:
        total_periods[trigger_machine] = get_machine_stops(df.copy(), trigger_machine)
    
    # 分析单机触发单机的情况
    for trigger_machine in [1, 2, 3, 4]:
        trigger_periods = total_periods[trigger_machine]
        total_trigger_count = len(trigger_periods)
        
        if total_trigger_count == 0:
            continue
            
        # 分析单机触发单机
        for target_machine in [1, 2, 3, 4]:
            if target_machine == trigger_machine:
                continue
                
            subsequent_stops = check_subsequent_stops(trigger_periods, df.copy(), [target_machine])
            probability = len(subsequent_stops) / total_trigger_count if total_trigger_count > 0 else 0
            
            key = f"{trigger_machine}停机后{target_machine}停机"
            results[key] = {
                '触发机器': f"{trigger_machine}#",
                '目标机器': f"{target_machine}#",
                '触发次数': total_trigger_count,
                '响应次数': len(subsequent_stops),
                '响应概率': probability
            }
        
        # 分析单机触发双机
        other_machines = [m for m in [1, 2, 3, 4] if m != trigger_machine]
        for target_pair in combinations(other_machines, 2):
            subsequent_stops = check_subsequent_stops(trigger_periods, df.copy(), target_pair)
            probability = len(subsequent_stops) / total_trigger_count if total_trigger_count > 0 else 0
            
            target_str = ''.join([f"{m}#" for m in target_pair])
            key = f"{trigger_machine}停机后{target_str}停机"
            results[key] = {
                '触发机器': f"{trigger_machine}#",
                '目标机器': f"{target_str}",
                '触发次数': total_trigger_count,
                '响应次数': len(subsequent_stops),
                '响应概率': probability
            }
        
        # 分析单机触发三机
        other_machines = [m for m in [1, 2, 3, 4] if m != trigger_machine]
        subsequent_stops = check_subsequent_stops(trigger_periods, df.copy(), other_machines)
        probability = len(subsequent_stops) / total_trigger_count if total_trigger_count > 0 else 0
        
        target_str = ''.join([f"{m}#" for m in other_machines])
        key = f"{trigger_machine}停机后{target_str}停机"
        results[key] = {
            '触发机器': f"{trigger_machine}#",
            '目标机器': f"{target_str}",
            '触发次数': total_trigger_count,
            '响应次数': len(subsequent_stops),
            '响应概率': probability
        }
    
    return results

def save_results_to_csv(results):
    # 将结果转换为列表
    results_list = []
    
    # 按目标机器数量分组
    single_machine_results = []
    double_machine_results = []
    triple_machine_results = []
    
    for key, data in results.items():
        result_item = {
            '触发机器': data['触发机器'],
            '目标机器': data['目标机器'],
            '触发次数': data['触发次数'],
            '响应次数': data['响应次数'],
            '响应概率': f"{data['响应概率']:.2%}"
        }
        
        # 根据目标机器数量分类
        target_count = data['目标机器'].count('#')
        if target_count == 1:  # 单机
            single_machine_results.append(result_item)
        elif target_count == 2:  # 双机
            double_machine_results.append(result_item)
        else:  # 三机
            triple_machine_results.append(result_item)
    
    # 对每个分组按触发机器排序
    for group in [single_machine_results, double_machine_results, triple_machine_results]:
        group.sort(key=lambda x: (int(x['触发机器'].strip('#')), x['目标机器']))
    
    # 合并所有结果
    results_list = single_machine_results + double_machine_results + triple_machine_results
    
    # 转换为DataFrame并保存
    df_results = pd.DataFrame(results_list)
    
    # 保存结果
    output_file = r'D:\Code_File\Vinda_cunzhijia_v2\小包机概率计算\停机事件触发概率_30秒.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n分析结果已保存到：{output_file}")
    
    # 打印分组统计信息
    print("\n=== 结果统计 ===")
    print(f"单机触发单机组合数: {len(single_machine_results)}")
    print(f"单机触发双机组合数: {len(double_machine_results)}")
    print(f"单机触发三机组合数: {len(triple_machine_results)}")
    print(f"总组合数: {len(results_list)}")

def main():
    try:
        df = load_data()
        results = analyze_triggered_stops(df)
        
        # 打印结果
        print("\n=== 触发停机分析结果 ===")
        for key, data in results.items():
            print(f"{key}:")
            print(f"  触发次数: {data['触发次数']}")
            print(f"  响应次数: {data['响应次数']}")
            print(f"  响应概率: {data['响应概率']:.2%}")
            print()
        
        # 保存结果到CSV
        save_results_to_csv(results)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 