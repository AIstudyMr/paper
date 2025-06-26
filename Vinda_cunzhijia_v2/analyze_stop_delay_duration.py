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
        if duration >= 10:  # 只考虑超过10秒的停机
            stop_periods.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            })
    
    return stop_periods

def find_next_stop(trigger_time, target_stops):
    """查找下一次停机事件"""
    for stop in target_stops:
        if stop['start_time'] > trigger_time:
            delay = (stop['start_time'] - trigger_time).total_seconds()
            return {
                'delay': delay,
                'duration': stop['duration'],
                'start_time': stop['start_time']
            }
    return None

def analyze_stop_delays_and_durations(df):
    results = []
    
    # 分析每台触发机器
    for trigger_machine in [1, 2, 3, 4]:
        print(f"\n分析 {trigger_machine}# 机器作为触发源...")
        trigger_periods = get_machine_stops(df.copy(), trigger_machine)
        
        if not trigger_periods:
            continue
            
        # 分析对单机的影响
        for target_machine in [1, 2, 3, 4]:
            if target_machine == trigger_machine:
                continue
                
            target_stops = get_machine_stops(df.copy(), target_machine)
            
            # 分析每次触发停机
            for trigger_stop in trigger_periods:
                trigger_start = trigger_stop['start_time']
                next_stop = find_next_stop(trigger_start, target_stops)
                
                if next_stop:  # 记录所有后续停机事件
                    results.append({
                        '触发机器': f"{trigger_machine}#",
                        '目标机器': f"{target_machine}#",
                        '触发开始时间': trigger_start,
                        '触发持续时间': trigger_stop['duration'],
                        '响应延迟时间': next_stop['delay'],
                        '响应持续时间': next_stop['duration']
                    })
        
        # 分析对多机组合的影响
        other_machines = [m for m in [1, 2, 3, 4] if m != trigger_machine]
        for r in range(2, len(other_machines) + 1):
            for target_combination in combinations(other_machines, r):
                for trigger_stop in trigger_periods:
                    trigger_start = trigger_stop['start_time']
                    
                    # 检查每个目标机器的响应情况
                    combination_responses = []
                    for target_machine in target_combination:
                        target_stops = get_machine_stops(df.copy(), target_machine)
                        next_stop = find_next_stop(trigger_start, target_stops)
                        
                        if next_stop:
                            combination_responses.append(next_stop)
                    
                    # 如果所有目标机器都有响应
                    if len(combination_responses) == len(target_combination):
                        max_delay = max(resp['delay'] for resp in combination_responses)
                        avg_duration = sum(resp['duration'] for resp in combination_responses) / len(combination_responses)
                        
                        results.append({
                            '触发机器': f"{trigger_machine}#",
                            '目标机器': ''.join([f"{m}#" for m in target_combination]),
                            '触发开始时间': trigger_start,
                            '触发持续时间': trigger_stop['duration'],
                            '响应延迟时间': max_delay,
                            '响应持续时间': avg_duration
                        })
    
    return results

def analyze_delay_patterns(results_df):
    """分析延迟时间的模式"""
    print("\n=== 延迟时间模式分析 ===")
    
    # 按触发机器和目标机器分组分析延迟时间
    for trigger_machine in sorted(results_df['触发机器'].unique()):
        print(f"\n{trigger_machine}触发其他机器的延迟时间分析:")
        
        for target in sorted(results_df[results_df['触发机器'] == trigger_machine]['目标机器'].unique()):
            delays = results_df[
                (results_df['触发机器'] == trigger_machine) & 
                (results_df['目标机器'] == target)
            ]['响应延迟时间']
            
            if len(delays) > 0:
                # 计算延迟时间的分布
                percentiles = np.percentile(delays, [10, 25, 50, 75, 90])
                most_common = delays.mode().iloc[0] if not delays.mode().empty else None
                
                print(f"\n目标机器 {target}:")
                print(f"  - 样本数量: {len(delays)}")
                print(f"  - 最常见延迟时间: {most_common:.2f}秒")
                print(f"  - 延迟时间范围: {delays.min():.2f}秒 - {delays.max():.2f}秒")
                print(f"  - 10%的延迟在{percentiles[0]:.2f}秒内")
                print(f"  - 25%的延迟在{percentiles[1]:.2f}秒内")
                print(f"  - 50%的延迟在{percentiles[2]:.2f}秒内（中位数）")
                print(f"  - 75%的延迟在{percentiles[3]:.2f}秒内")
                print(f"  - 90%的延迟在{percentiles[4]:.2f}秒内")
                print(f"  - 平均延迟时间: {delays.mean():.2f}秒")
                
                # 分析延迟时间的聚类
                delays_sorted = sorted(delays)
                gaps = np.diff(delays_sorted)
                significant_gaps = np.where(gaps > np.mean(gaps) + 2 * np.std(gaps))[0]
                
                if len(significant_gaps) > 0:
                    print("\n  延迟时间主要集中在以下区间:")
                    start_idx = 0
                    for gap_idx in significant_gaps:
                        print(f"    {delays_sorted[start_idx]:.2f}秒 - {delays_sorted[gap_idx]:.2f}秒")
                        start_idx = gap_idx + 1
                    print(f"    {delays_sorted[start_idx]:.2f}秒 - {delays_sorted[-1]:.2f}秒")

def save_results_to_csv(results):
    df_results = pd.DataFrame(results)
    
    # 格式化时间列
    df_results['触发开始时间'] = pd.to_datetime(df_results['触发开始时间'])
    df_results['触发开始时间'] = df_results['触发开始时间'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # 对数值列保留2位小数
    for col in ['触发持续时间', '响应延迟时间', '响应持续时间']:
        df_results[col] = df_results[col].round(2)
    
    # 按触发机器和目标机器排序
    df_results = df_results.sort_values(['触发机器', '目标机器', '触发开始时间'])
    
    # 保存结果
    output_file = r'D:\Code_File\Vinda_cunzhijia_v2\小包机概率计算\停机延迟和持续时间分析.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n分析结果已保存到：{output_file}")
    
    # 打印基础统计信息
    print("\n=== 基础统计信息 ===")
    stats = df_results.groupby(['触发机器', '目标机器']).agg({
        '响应延迟时间': ['count', 'mean', 'min', 'max'],
        '响应持续时间': ['mean', 'min', 'max']
    }).round(2)
    
    print("\n停机响应统计:")
    print(stats)
    
    # 分析延迟时间模式
    analyze_delay_patterns(df_results)

def main():
    try:
        df = load_data()
        results = analyze_stop_delays_and_durations(df)
        
        if not results:
            print("未找到符合条件的停机事件")
            return
            
        save_results_to_csv(results)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 