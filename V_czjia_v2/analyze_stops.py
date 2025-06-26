import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 读取CSV文件
def load_data():
    print("正在读取数据...")
    df = pd.read_csv('存纸架数据汇总.csv')
    df['时间'] = pd.to_datetime(df['时间'])
    print(f"数据时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
    return df

# 计算连续停机时间
def calculate_stop_probability(df, machine_number, time_window=None):
    """
    计算停机概率
    :param df: 数据框
    :param machine_number: 机器编号
    :param time_window: 时间窗口（小时），如果为None则分析全部数据
    """
    # 如果指定了时间窗口，则只分析最近的数据
    if time_window is not None:
        end_time = df['时间'].max()
        start_time = end_time - pd.Timedelta(hours=time_window)
        df = df[(df['时间'] >= start_time) & (df['时间'] <= end_time)]
        print(f"\n分析时间范围: {start_time} 到 {end_time}")

    # 获取对应小包机的速度列名
    speed_col = f"{machine_number}#小包机实际速度" if machine_number != 3 and machine_number != 4 else f"{machine_number}#小包机主机实际速度"
    
    # 确保时间是按顺序排列的
    df = df.sort_values('时间')
    
    # 计算时间差（秒）
    df['time_diff'] = df['时间'].diff().dt.total_seconds()
    
    # 标记停机状态（速度为0）
    df['is_stopped'] = df[speed_col] == 0
    
    # 创建停机组
    df['stop_group'] = (df['is_stopped'] != df['is_stopped'].shift()).cumsum()
    
    # 计算每个停机期的持续时间
    stop_periods = []
    for _, group in df[df['is_stopped']].groupby('stop_group'):
        duration = group['time_diff'].sum()
        if not np.isnan(duration):
            stop_periods.append(duration)
    
    # 计算长停机（>10秒）的概率
    total_time = (df['时间'].max() - df['时间'].min()).total_seconds()
    long_stops = [stop for stop in stop_periods if stop > 10]
    total_long_stop_time = sum(long_stops)
    
    stop_probability = total_long_stop_time / total_time if total_time > 0 else 0
    
    # 计算平均运行速度（不包括停机时间）
    avg_speed = df[~df['is_stopped']][speed_col].mean()
    
    # 计算每天的停机概率
    df['date'] = df['时间'].dt.date
    daily_stats = []
    
    for date, date_group in df.groupby('date'):
        date_total_time = (date_group['时间'].max() - date_group['时间'].min()).total_seconds()
        date_stop_periods = []
        for _, stop_group in date_group[date_group['is_stopped']].groupby('stop_group'):
            duration = stop_group['time_diff'].sum()
            if not np.isnan(duration):
                date_stop_periods.append(duration)
        
        date_long_stops = [stop for stop in date_stop_periods if stop > 10]
        date_total_long_stop_time = sum(date_long_stops)
        date_stop_probability = date_total_long_stop_time / date_total_time if date_total_time > 0 else 0
        
        daily_stats.append({
            'date': date,
            'probability': date_stop_probability,
            'total_stops': len(date_stop_periods),
            'long_stops': len(date_long_stops)
        })
    
    return {
        'probability': stop_probability,
        'total_stops': len(stop_periods),
        'long_stops': len(long_stops),
        'avg_stop_duration': np.mean(stop_periods) if stop_periods else 0,
        'max_stop_duration': max(stop_periods) if stop_periods else 0,
        'avg_running_speed': avg_speed,
        'daily_stats': daily_stats
    }

def save_results_to_csv(full_period_results, last_24h_results):
    # 保存全周期统计结果
    full_period_data = []
    for machine_number in [1, 2, 3, 4]:
        results = full_period_results[machine_number]
        full_period_data.append({
            '机器编号': f"{machine_number}#小包机",
            '停机概率': f"{results['probability']:.2%}",
            '总停机次数': results['total_stops'],
            '长停机次数': results['long_stops'],
            '平均停机时长(秒)': round(results['avg_stop_duration'], 2),
            '最长停机时长(秒)': round(results['max_stop_duration'], 2),
            '平均运行速度': round(results['avg_running_speed'], 2)
        })
    
    full_period_df = pd.DataFrame(full_period_data)
    full_period_df.to_csv(r'D:\Code_File\Vinda_cunzhijia_v2\小包机概率计算\全周期停机统计.csv', index=False, encoding='utf-8-sig')
    
    # 保存每日统计结果（新格式）
    # 首先获取所有唯一日期
    all_dates = set()
    for machine_number in [1, 2, 3, 4]:
        for daily_stat in full_period_results[machine_number]['daily_stats']:
            all_dates.add(daily_stat['date'])
    all_dates = sorted(list(all_dates))
    
    # 创建每日数据字典
    daily_data = {date: {} for date in all_dates}
    
    # 填充数据
    for machine_number in [1, 2, 3, 4]:
        for daily_stat in full_period_results[machine_number]['daily_stats']:
            date = daily_stat['date']
            daily_data[date].update({
                f"{machine_number}#小包机": f"{machine_number}#小包机",
                f"{machine_number}#停机概率": f"{daily_stat['probability']:.2%}",
                f"{machine_number}#总停机次数": daily_stat['total_stops'],
                f"{machine_number}#长停机次数": daily_stat['long_stops']
            })
    
    # 转换为DataFrame
    daily_rows = []
    for date in all_dates:
        row = {'日期': date}
        row.update(daily_data[date])
        daily_rows.append(row)
    
    # 确保列的顺序正确
    columns = ['日期']
    for machine_number in [1, 2, 3, 4]:
        columns.extend([
            f"{machine_number}#小包机",
            f"{machine_number}#停机概率",
            f"{machine_number}#总停机次数",
            f"{machine_number}#长停机次数"
        ])
    
    daily_df = pd.DataFrame(daily_rows)
    # 重新排列列顺序
    daily_df = daily_df[columns]
    daily_df.to_csv(r'D:\Code_File\Vinda_cunzhijia_v2\小包机概率计算\每日停机统计.csv', index=False, encoding='utf-8-sig')
    
    print("\n分析结果已保存到CSV文件：")
    print("1. 全周期停机统计.csv")
    print("2. 每日停机统计.csv")

def print_analysis_results(results, title):
    print(f"\n=== {title} ===")
    for machine_number in [1, 2, 3, 4]:
        print(f"\n{machine_number}#小包机:")
        print(f"停机概率（连续超过10秒）: {results[machine_number]['probability']:.2%}")
        print(f"总停机次数: {results[machine_number]['total_stops']}")
        print(f"长停机次数（>10秒）: {results[machine_number]['long_stops']}")
        print(f"平均停机时长: {results[machine_number]['avg_stop_duration']:.2f}秒")
        print(f"最长停机时长: {results[machine_number]['max_stop_duration']:.2f}秒")
        print(f"平均运行速度: {results[machine_number]['avg_running_speed']:.2f}")
        
        if title == "小包机停机分析结果（全时段）":
            print("\n每日停机概率统计:")
            for daily_stat in results[machine_number]['daily_stats']:
                print(f"日期: {daily_stat['date']}, 停机概率: {daily_stat['probability']:.2%}, "
                      f"总停机次数: {daily_stat['total_stops']}, 长停机次数: {daily_stat['long_stops']}")

def main():
    try:
        df = load_data()
        
        # 存储所有结果
        full_period_results = {}
        last_24h_results = {}
        
        # 计算全时段的停机概率
        for machine_number in [1, 2, 3, 4]:
            full_period_results[machine_number] = calculate_stop_probability(df, machine_number)
        
        # 计算最近24小时的停机概率
        for machine_number in [1, 2, 3, 4]:
            last_24h_results[machine_number] = calculate_stop_probability(df, machine_number, time_window=24)
        
        # 打印结果
        print_analysis_results(full_period_results, "小包机停机分析结果（全时段）")
        print_analysis_results(last_24h_results, "小包机停机分析结果（最近24小时）")
        
        # 保存结果到CSV文件
        save_results_to_csv(full_period_results, last_24h_results)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main() 




'''
情况1：
1#小包机停机后，2#小包机停机概率：
1#小包机停机后，3#小包机停机概率：
1#小包机停机后，4#小包机停机概率：
情况2：
2#小包机停机后，1#小包机停机概率：
2#小包机停机后，3#小包机停机概率：
2#小包机停机后，4#小包机停机概率：

情况3：
3#小包机停机后，1#小包机停机概率：
3#小包机停机后，2#小包机停机概率：
3#小包机停机后，4#小包机停机概率：

情况4：
4#小包机停机后，1#小包机停机概率：
4#小包机停机后，2#小包机停机概率：
4#小包机停机后，3#小包机停机概率：

情况5：
1#小包机停机后，2#4#小包机停机概率：
1#小包机停机后，2#3#小包机停机概率：
1#小包机停机后，2#2#小包机停机概率：

情况6：
2#小包机停机后，1#4#小包机停机概率：
2#小包机停机后，1#3#小包机停机概率：
2#小包机停机后，1#2#小包机停机概率：

情况7：
3#小包机停机后，1#4#小包机停机概率：
3#小包机停机后，1#3#小包机停机概率：
3#小包机停机后，1#2#小包机停机概率：

情况8：
4#小包机停机后，1#4#小包机停机概率：
4#小包机停机后，1#3#小包机停机概率：
4#小包机停机后，1#2#小包机停机概率：


情况9：
1#小包机停机后，2#3#4#小包机停机概率：
2#小包机停机后，1#3#4#小包机停机概率：
3#小包机停机后，1#2#4#小包机停机概率：
4#小包机停机后，1#2#3#小包机停机概率：
'''













