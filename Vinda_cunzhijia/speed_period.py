import csv,os
from datetime import datetime
from collections import deque

def find_speed_periods(file_path, window_size, low_threshold, high_threshold):
    # 用于存储不同速度状态的时间段
    zero_periods = []      # 速度为0的时间段
    low_speed_periods = []  # 平滑平均值≤low_threshold的时间段
    high_speed_periods = [] # 平滑平均值≥high_threshold的时间段
    
    # 平滑窗口相关变量
    window = deque(maxlen=window_size)
    current_low_start = None
    current_high_start = None
    previous_time = None
    
    # 零速度检测相关变量
    current_zero_start = None
    previous_speed_zero = False

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            try:
                time_str = row['时间']
                current_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                speed = float(row['折叠机实际速度'])
                
                # 更新平滑窗口
                window.append(speed)
                window_avg = sum(window) / len(window) if window else 0
                
                # 检测平滑平均值≤low_threshold的时间段
                if window_avg <= low_threshold:
                    if current_low_start is None:
                        current_low_start = current_time
                    # 如果同时在高速度状态，结束高速度状态
                    if current_high_start is not None:
                        if len(window) == window_size:
                            high_speed_periods.append((current_high_start, previous_time))
                        current_high_start = None
                else:
                    if current_low_start is not None:
                        if len(window) == window_size:
                            low_speed_periods.append((current_low_start, previous_time))
                        current_low_start = None
                
                # 检测平滑平均值≥high_threshold的时间段
                if window_avg >= high_threshold:
                    if current_high_start is None:
                        current_high_start = current_time
                    # 如果在低速度状态，结束低速度状态
                    if current_low_start is not None:
                        if len(window) == window_size:
                            low_speed_periods.append((current_low_start, previous_time))
                        current_low_start = None
                else:
                    if current_high_start is not None:
                        if len(window) == window_size:
                            high_speed_periods.append((current_high_start, previous_time))
                        current_high_start = None
                
                # 原功能：检测零速度时间段
                if speed == 0.0:
                    if not previous_speed_zero:
                        current_zero_start = current_time
                        previous_speed_zero = True
                else:
                    if previous_speed_zero and current_zero_start:
                        zero_periods.append((current_zero_start, previous_time))
                        current_zero_start = None
                    previous_speed_zero = False
                
                previous_time = current_time
            except (ValueError, KeyError) as e:
                print(f"处理行时出错: {row}, 错误: {e}")
                continue
        
        # 检查文件结束时是否还有未结束的段
        if current_low_start is not None and len(window) == window_size:
            low_speed_periods.append((current_low_start, previous_time))
        if current_high_start is not None and len(window) == window_size:
            high_speed_periods.append((current_high_start, previous_time))
        if previous_speed_zero and current_zero_start:
            zero_periods.append((current_zero_start, previous_time))
    
    return zero_periods, low_speed_periods, high_speed_periods


def print_periods(periods, description):
    if not periods:
        print(f"没有找到{description}的时间段")
        return
    
    print(f"{description}的时间段:")
    for i, (start, end) in enumerate(periods, 1):
        duration = end - start
        print(f"{i}. 从 {start} 到 {end}, 持续时间: {duration}")


def save_periods_to_csv(periods, output_file, description):
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['序号', '开始时间', '结束时间', '持续时间(秒)', '描述'])
        
        for i, (start, end) in enumerate(periods, 1):
            duration = (end - start).total_seconds()
            writer.writerow([i, start, end, duration, description])



# 使用示例
file_path = r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv'  # 替换为你的文件路径
zero_periods, low_speed_periods, high_speed_periods = find_speed_periods(file_path, window_size=20, low_threshold=60, high_threshold=80)

print_periods(zero_periods, "折叠机实际速度为0")
print("\n" + "="*50 + "\n")
print_periods(low_speed_periods, "折叠机实际速度20点平滑平均值≤60")
print("\n" + "="*50 + "\n")
print_periods(high_speed_periods, "折叠机实际速度20点平滑平均值≥80")


outpath = r'D:\Code_File\Vinda_cunzhijia\高低速时间段'
if not os.path.exists(outpath):
    os.makedirs(outpath)


# 保存零速度时间段
save_periods_to_csv(zero_periods, os.path.join(outpath,'zero_speed_periods.csv'), '折叠机实际速度为0')

# 保存低速时间段
save_periods_to_csv(low_speed_periods, os.path.join(outpath,'low_speed_periods.csv'), '折叠机实际速度20点平滑平均值≤60')

# 保存高速时间段
save_periods_to_csv(high_speed_periods, os.path.join(outpath,'high_speed_periods.csv'), '折叠机实际速度20点平滑平均值≥80')
