import pandas as pd
import numpy as np
from datetime import datetime

# 读取CSV文件
df = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv' )

# 转换时间列为datetime格式
df['时间'] = pd.to_datetime(df['时间'])

# 定义停机阈值（速度低于此值视为停机）
stop_threshold = 0

# 定义各机器的速度列名
speed_columns = {
    '1#': '1#小包机实际速度',
    '2#': '2#小包机实际速度',
    '3#': '3#小包机主机实际速度',
    '4#': '4#小包机主机实际速度'
}

# 标记各机器的停机状态
for machine, col in speed_columns.items():
    df[f'{machine}停机'] = df[col] <= stop_threshold

# 定义分析窗口大小（记录条数）
window_size = 300

def calculate_probabilities_and_times(trigger_machine, target_machines):
    """
    计算触发机器停机后，目标机器停机的概率和时间间隔
    :param trigger_machine: 触发停机的机器（如'1#'）
    :param target_machines: 目标机器列表（如['2#', '3#']）
    :return: (概率, 平均时间间隔)
    """
    trigger_stops = df[df[f'{trigger_machine}停机']].index
    total_stops = len(trigger_stops)
    
    if total_stops == 0:
        return 0, 0
    
    co_stops = 0
    time_diffs = []
    
    for i in trigger_stops:
        window = df.iloc[i:i+window_size]
        
        # 检查所有目标机器是否都在窗口内停机
        all_stopped = all(any(window[f'{m}停机']) for m in target_machines)
        
        if all_stopped:
            co_stops += 1
            # 计算最后一个停机的机器的时间差
            stop_times = []
            for m in target_machines:
                stopped = window[window[f'{m}停机']]
                if not stopped.empty:
                    stop_times.append((stopped.iloc[0]['时间'] - df.iloc[i]['时间']).total_seconds())
            if stop_times:
                time_diffs.append(max(stop_times))  # 取最晚停机的那个时间
    
    probability = co_stops / total_stops if total_stops > 0 else 0
    avg_time = np.mean(time_diffs) if time_diffs else 0
    
    return probability, avg_time

# 准备结果DataFrame
results = pd.DataFrame(columns=[
    '情况', '触发机器', '目标机器', '停机概率', '平均时间间隔(秒)'
])



# 情况1-4：单机器触发，单机器跟随
cases_1_to_4 = [
    ('1#', ['2#']), ('1#', ['3#']), ('1#', ['4#']),
    ('2#', ['1#']), ('2#', ['3#']), ('2#', ['4#']),
    ('3#', ['1#']), ('3#', ['2#']), ('3#', ['4#']),
    ('4#', ['1#']), ('4#', ['2#']), ('4#', ['3#'])
]

# 情况5-8：单机器触发，双机器跟随
cases_5_to_8 = [
    ('1#', ['2#', '3#']), ('1#', ['2#', '4#']), ('1#', ['3#', '4#']),
    ('2#', ['1#', '3#']), ('2#', ['1#', '4#']), ('2#', ['3#', '4#']),
    ('3#', ['1#', '2#']), ('3#', ['2#', '4#']), ('3#', ['1#', '4#']),
    ('4#', ['2#', '3#']), ('4#', ['1#', '2#']), ('4#', ['1#', '3#'])
]

# 情况9：单机器触发，三机器跟随
cases_9 = [
    ('1#', ['2#', '3#', '4#']),
    ('2#', ['1#', '3#', '4#']),
    ('3#', ['1#', '2#', '4#']),
    ('4#', ['1#', '2#', '3#'])
]

# 添加结果到DataFrame
def add_results(cases, case_name):
    for trigger, targets in cases:
        prob, avg_time = calculate_probabilities_and_times(trigger, targets)
        targets_str = ', '.join(targets)
        results.loc[len(results)] = [
            case_name,
            trigger,
            targets_str,
            f"{prob:.2%}",
            f"{avg_time:.2f}"
        ]


# 添加所有情况
add_results(cases_1_to_4[:3], "情况（1）")
add_results(cases_1_to_4[3:6], "情况（2）")
add_results(cases_1_to_4[6:9], "情况（3）")
add_results(cases_1_to_4[9:12], "情况（4）")
add_results(cases_5_to_8[:3], "情况（5）")
add_results(cases_5_to_8[3:6], "情况（6）")
add_results(cases_5_to_8[6:9], "情况（7）")
add_results(cases_5_to_8[9:12], "情况（8）")
add_results(cases_9, "情况（9）")

# 保存结果到CSV文件
results.to_csv(r'D:\Code_File\Vinda_cunzhijia\小包机概率计算\停机事件触发概率_{}.csv'.format(window_size), index=False, encoding='utf_8')

print("分析结果已保存到 '停机概率分析结果.csv'")

























exit()

import pandas as pd
import numpy as np
from datetime import datetime

# 读取CSV文件
df = pd.read_csv(r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv')

# 转换时间列为datetime格式
df['时间'] = pd.to_datetime(df['时间'])

# 定义停机阈值（速度低于此值视为停机）
stop_threshold = 0  # 可根据实际情况调整

# 识别1#小包机停机事件
df['1#停机'] = df['1#小包机实际速度'] <= stop_threshold

# 识别其他包机停机事件
df['2#停机'] = df['2#小包机实际速度'] <= stop_threshold
df['3#停机'] = df['3#小包机主机实际速度'] <= stop_threshold
df['4#停机'] = df['4#小包机主机实际速度'] <= stop_threshold

# 找到1#停机的时间点
stop_events = df[df['1#停机']].index

# 计算条件概率
total_1_stops = len(stop_events)
if total_1_stops == 0:
    print("数据中没有1#小包机停机事件")
else:
    # 在1#停机后，其他包机停机的次数
    after_2_stops = 0
    after_3_stops = 0
    after_4_stops = 0
    
    # 统计停机时间间隔
    time_to_2_stop = []
    time_to_3_stop = []
    time_to_4_stop = []
    
    for i in stop_events:
        # 检查后续时间点（最多查看后续10条记录）
        window = df.iloc[i:i+120]
        
        # 检查2#停机
        if any(window['2#停机']):
            after_2_stops += 1
            first_stop = window[window['2#停机']].iloc[0]
            time_diff = (first_stop['时间'] - df.iloc[i]['时间']).total_seconds()
            time_to_2_stop.append(time_diff)
        
        # 检查3#停机
        if any(window['3#停机']):
            after_3_stops += 1
            first_stop = window[window['3#停机']].iloc[0]
            time_diff = (first_stop['时间'] - df.iloc[i]['时间']).total_seconds()
            time_to_3_stop.append(time_diff)
        
        # 检查4#停机
        if any(window['4#停机']):
            after_4_stops += 1
            first_stop = window[window['4#停机']].iloc[0]
            time_diff = (first_stop['时间'] - df.iloc[i]['时间']).total_seconds()
            time_to_4_stop.append(time_diff)
    
    # 计算概率
    prob_2 = after_2_stops / total_1_stops
    prob_3 = after_3_stops / total_1_stops
    prob_4 = after_4_stops / total_1_stops
    
    # 计算平均时间间隔
    avg_time_2 = np.mean(time_to_2_stop) if time_to_2_stop else 0
    avg_time_3 = np.mean(time_to_3_stop) if time_to_3_stop else 0
    avg_time_4 = np.mean(time_to_4_stop) if time_to_4_stop else 0
    
    # 输出结果
    print(f"1#小包机停机后，2#小包机停机的概率: {prob_2:.2%}")
    print(f"1#小包机停机后，3#小包机停机的概率: {prob_3:.2%}")
    print(f"1#小包机停机后，4#小包机停机的概率: {prob_4:.2%}")
    print("\n平均停机时间间隔(秒):")
    print(f"1#到2#: {avg_time_2:.2f}")
    print(f"1#到3#: {avg_time_3:.2f}")
    print(f"1#到4#: {avg_time_4:.2f}")
    
    # # 时间分布分析
    # print("\n停机时间分布(秒):")
    # if time_to_2_stop:
    #     print("1#到2#:", sorted(time_to_2_stop))
    # if time_to_3_stop:
    #     print("1#到3#:", sorted(time_to_3_stop))
    # if time_to_4_stop:
    #     print("1#到4#:", sorted(time_to_4_stop))


