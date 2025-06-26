import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import linregress
import os

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_csv_with_encoding(filename):
    """尝试不同编码读取CSV文件"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
    for encoding in encodings:
        try:
            df = pd.read_csv(filename, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取 {filename}")
            return df
        except Exception as e:
            print(f"使用 {encoding} 编码读取失败: {e}")
            continue
    raise ValueError(f"无法读取文件 {filename}")

def smooth_data(data, window_size=5):
    """使用移动平均进行数据平滑"""
    return data.rolling(window=window_size, center=True, min_periods=1).mean()

def calculate_slope(x, y, window_size=10):
    """计算滑动窗口内的斜率"""
    slopes = []
    for i in range(len(x)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(x), i + window_size // 2 + 1)
        
        if end_idx - start_idx >= 2:
            x_window = np.array(range(end_idx - start_idx))
            y_window = y.iloc[start_idx:end_idx].values
            
            # 去除NaN值
            mask = ~np.isnan(y_window)
            if np.sum(mask) >= 2:
                slope, _, _, _, _ = linregress(x_window[mask], y_window[mask])
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        else:
            slopes.append(np.nan)
    
    return np.array(slopes)

def detect_machine_stops(speed_data, threshold=25):
    """检测小包机停机事件"""
    stops = []
    in_stop = False
    start_idx = None
    
    for i, speed in enumerate(speed_data):
        if pd.isna(speed):
            continue
            
        if speed <= threshold and not in_stop:
            # 停机开始
            in_stop = True
            start_idx = i
        elif speed > threshold and in_stop:
            # 停机结束
            in_stop = False
            if start_idx is not None:
                stops.append((start_idx, i-1))
    
    # 如果最后还在停机状态
    if in_stop and start_idx is not None:
        stops.append((start_idx, len(speed_data)-1))
    
    return stops

def analyze_slope_change_after_stop(time_data, paper_rate_smoothed, slopes, stop_events, 
                                  min_duration_minutes=2):
    """分析停机后存纸率斜率变化"""
    results = []
    
    for stop_start, stop_end in stop_events:
        # 停机持续时间
        if stop_end < len(time_data) and stop_start < len(time_data):
            stop_duration = (time_data.iloc[stop_end] - time_data.iloc[stop_start]).total_seconds() / 60
            
            # 只分析持续时间超过阈值的停机事件
            if stop_duration < min_duration_minutes:
                continue
            
            # 分析停机前后的斜率变化
            pre_stop_window = 20  # 停机前20个点
            post_stop_window = 50  # 停机后50个点
            
            pre_start = max(0, stop_start - pre_stop_window)
            post_end = min(len(slopes), stop_end + post_stop_window)
            
            if post_end > stop_end + 10:  # 确保有足够的停机后数据
                # 停机前的平均斜率
                pre_slopes = slopes[pre_start:stop_start]
                pre_mean_slope = np.nanmean(pre_slopes) if len(pre_slopes) > 0 else np.nan
                
                # 停机后寻找斜率开始显著变化的时间点
                post_slopes = slopes[stop_end:post_end]
                slope_change_idx = None
                
                if len(post_slopes) > 5:
                    # 使用滑动窗口检测斜率变化
                    window = 5
                    for i in range(window, len(post_slopes)):
                        recent_slope = np.nanmean(post_slopes[i-window:i])
                        if not np.isnan(recent_slope) and not np.isnan(pre_mean_slope):
                            # 如果斜率变化超过阈值，认为开始变化
                            if abs(recent_slope - pre_mean_slope) > abs(pre_mean_slope) * 0.5:
                                slope_change_idx = stop_end + i
                                break
                
                if slope_change_idx is not None and slope_change_idx < len(time_data):
                    change_delay = (time_data.iloc[slope_change_idx] - time_data.iloc[stop_end]).total_seconds() / 60
                    
                    results.append({
                        'stop_start_time': time_data.iloc[stop_start],
                        'stop_end_time': time_data.iloc[stop_end],
                        'stop_duration_minutes': stop_duration,
                        'slope_change_time': time_data.iloc[slope_change_idx],
                        'delay_minutes': change_delay,
                        'pre_stop_slope': pre_mean_slope,
                        'post_change_slope': np.nanmean(post_slopes[slope_change_idx-stop_end:slope_change_idx-stop_end+5])
                    })
    
    return results

def analyze_period_data(summary_data, start_time, end_time, period_idx):
    """分析单个时间段的数据"""
    print(f"\n=== 分析时间段 {period_idx}: {start_time} 到 {end_time} ===")
    
    # 转换时间列为datetime
    summary_data['时间'] = pd.to_datetime(summary_data['时间'])
    
    # 筛选时间段内的数据
    mask = (summary_data['时间'] >= start_time) & (summary_data['时间'] <= end_time)
    period_data = summary_data[mask].copy()
    
    if period_data.empty:
        print(f"时间段 {period_idx}: 没有找到数据")
        return None
    
    print(f"找到 {len(period_data)} 个数据点")
    
    # 重置索引
    period_data = period_data.reset_index(drop=True)
    
    # 查找相关列
    cols = period_data.columns.tolist()
    paper_rate_cols = [col for col in cols if '存纸率' in col]
    package_speed_cols = [col for col in cols if '小包机' in col and ('速度' in col or '实际速度' in col)]
    
    if not paper_rate_cols:
        print("未找到存纸率列")
        return None
    
    if not package_speed_cols:
        print("未找到小包机速度列")
        return None
    
    print(f"存纸率列: {paper_rate_cols}")
    print(f"小包机速度列: {package_speed_cols[:4]}")  # 只取前4个
    
    results = {}
    
    # 对每个存纸率列进行分析
    for paper_col in paper_rate_cols:
        paper_rate = pd.to_numeric(period_data[paper_col], errors='coerce')
        
        # 平滑处理
        paper_rate_smoothed = smooth_data(paper_rate, window_size=5)
        
        # 计算斜率
        slopes = calculate_slope(period_data.index, paper_rate_smoothed, window_size=10)
        
        # 分析每个小包机的停机影响
        for i, speed_col in enumerate(package_speed_cols[:4]):
            machine_speed = pd.to_numeric(period_data[speed_col], errors='coerce')
            
            # 检测停机事件
            stop_events = detect_machine_stops(machine_speed, threshold=25)
            
            if stop_events:
                print(f"小包机 {i+1} ({speed_col}): 检测到 {len(stop_events)} 个停机事件")
                
                # 分析斜率变化
                slope_changes = analyze_slope_change_after_stop(
                    period_data['时间'], paper_rate_smoothed, slopes, stop_events
                )
                
                if slope_changes:
                    results[f'{paper_col}_小包机{i+1}'] = slope_changes
                    print(f"  - 发现 {len(slope_changes)} 个有效的斜率变化事件")
                    
                    for j, change in enumerate(slope_changes):
                        print(f"    事件{j+1}: 停机{change['stop_duration_minutes']:.1f}分钟, "
                              f"延迟{change['delay_minutes']:.1f}分钟后斜率开始变化")
    
    return results

def main():
    try:
        # 读取时间段数据
        print("读取存纸率1.csv文件...")
        periods_df = pd.read_csv('折叠机正常运行且高存纸率时间段_最终结果_存纸率1_调整后.csv')
        print(f"读取到 {len(periods_df)} 个时间段")
        
        # 读取汇总数据
        print("读取汇总数据...")
        summary_df = read_csv_with_encoding('存纸架数据汇总.csv')
        print(f"读取到 {len(summary_df)} 行汇总数据")
        
        # 检查数据列
        print(f"汇总数据列名: {summary_df.columns.tolist()}")
        
        all_results = {}
        
        # 分析每个时间段
        for idx, row in periods_df.iterrows():
            start_time = pd.to_datetime(row['开始时间'])
            end_time = pd.to_datetime(row['结束时间'])
            
            period_results = analyze_period_data(summary_df, start_time, end_time, idx+1)
            if period_results:
                all_results[f'时间段_{idx+1}'] = period_results
        
        # 统计总体结果
        print("\n=== 总体统计结果 ===")
        all_delays = []
        
        for period_name, period_results in all_results.items():
            for analysis_name, slope_changes in period_results.items():
                for change in slope_changes:
                    all_delays.append(change['delay_minutes'])
        
        if all_delays:
            print(f"总共检测到 {len(all_delays)} 个有效的停机后斜率变化事件")
            print(f"平均延迟时间: {np.mean(all_delays):.2f} 分钟")
            print(f"中位数延迟时间: {np.median(all_delays):.2f} 分钟")
            print(f"最短延迟时间: {np.min(all_delays):.2f} 分钟")
            print(f"最长延迟时间: {np.max(all_delays):.2f} 分钟")
            print(f"标准差: {np.std(all_delays):.2f} 分钟")
            
            # 保存详细结果到CSV
            detailed_results = []
            for period_name, period_results in all_results.items():
                for analysis_name, slope_changes in period_results.items():
                    for i, change in enumerate(slope_changes):
                        detailed_results.append({
                            '时间段': period_name,
                            '分析对象': analysis_name,
                            '事件序号': i+1,
                            '停机开始时间': change['stop_start_time'],
                            '停机结束时间': change['stop_end_time'],
                            '停机持续时间(分钟)': change['stop_duration_minutes'],
                            '斜率变化时间': change['slope_change_time'],
                            '延迟时间(分钟)': change['delay_minutes'],
                            '停机前斜率': change['pre_stop_slope'],
                            '变化后斜率': change['post_change_slope']
                        })
            
            if detailed_results:
                results_df = pd.DataFrame(detailed_results)
                results_df.to_csv('小包机停机后存纸率斜率变化分析结果.csv', index=False, encoding='utf-8-sig')
                print(f"\n详细结果已保存到: 小包机停机后存纸率斜率变化分析结果.csv")
        else:
            print("未检测到有效的停机后斜率变化事件")
    
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 