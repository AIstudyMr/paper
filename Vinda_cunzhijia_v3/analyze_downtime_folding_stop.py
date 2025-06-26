import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def smooth_paper_rate(data, smooth_window=30, method='moving_average'):
    """
    对存纸率数据进行平滑处理
    
    Parameters:
    - data: 包含'存纸率'列的DataFrame
    - smooth_window: 平滑窗口大小，默认5
    - method: 平滑方法，支持 'moving_average', 'exponential', 'gaussian'
    
    Returns:
    - 平滑处理后的DataFrame（复制品）
    """
    from scipy import ndimage
    
    smoothed_data = data.copy()
    
    if '存纸率' not in smoothed_data.columns or len(smoothed_data) < smooth_window:
        return smoothed_data
    
    paper_rate = smoothed_data['存纸率'].values
    
    if method == 'moving_average':
        # 简单移动平均
        smoothed_rate = np.convolve(paper_rate, np.ones(smooth_window)/smooth_window, mode='same')
        
        # 处理边界效应：前后几个点使用逐渐增大的窗口
        for i in range(min(smooth_window//2, len(paper_rate))):
            # 前半部分
            if i > 0:
                smoothed_rate[i] = np.mean(paper_rate[:i*2+1])
            # 后半部分
            if len(paper_rate) - 1 - i >= 0:
                smoothed_rate[len(paper_rate) - 1 - i] = np.mean(paper_rate[len(paper_rate) - i*2 - 1:])
                
    elif method == 'exponential':
        # 指数移动平均
        alpha = 2.0 / (smooth_window + 1)
        smoothed_rate = np.zeros_like(paper_rate)
        smoothed_rate[0] = paper_rate[0]
        
        for i in range(1, len(paper_rate)):
            smoothed_rate[i] = alpha * paper_rate[i] + (1 - alpha) * smoothed_rate[i-1]
            
    elif method == 'gaussian':
        # 高斯滤波
        try:
            sigma = smooth_window / 3.0  # 标准差
            smoothed_rate = ndimage.gaussian_filter1d(paper_rate, sigma=sigma)
        except ImportError:
            # 如果scipy不可用，回退到移动平均
            smoothed_rate = np.convolve(paper_rate, np.ones(smooth_window)/smooth_window, mode='same')
    else:
        # 默认使用移动平均
        smoothed_rate = np.convolve(paper_rate, np.ones(smooth_window)/smooth_window, mode='same')
    
    smoothed_data['存纸率'] = smoothed_rate
    smoothed_data['原始存纸率'] = paper_rate  # 保留原始数据用于对比
    
    return smoothed_data

def read_csv_with_encoding(file_path):
    """尝试不同编码读取CSV文件"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件: {file_path}")
            return df
        except:
            continue
    raise Exception(f"无法读取文件 {file_path}")

def calculate_detailed_slope_stats_for_combination(paper_data, start_time, end_time, combination_time_points, window_size=30, smooth_window=30, smooth_method='moving_average'):
    """
    为特定停机组合的时间点计算详细的滑动窗口斜率统计信息
    
    Parameters:
    - paper_data: 存纸架数据
    - start_time: 开始时间
    - end_time: 结束时间
    - combination_time_points: 该组合对应的时间点索引列表
    - window_size: 滑动窗口大小
    - smooth_window: 平滑窗口大小，默认5
    - smooth_method: 平滑方法，默认'moving_average'
    
    Returns:
    - dict: 包含详细斜率统计信息的字典
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    # 过滤时间段内的数据
    mask = (paper_data['时间'] >= start_time) & (paper_data['时间'] <= end_time)
    period_data = paper_data[mask].copy()
    
    if len(period_data) == 0 or len(combination_time_points) < window_size:
        return {
            '平均斜率': 0, '斜率标准差': 0, '斜率中位数': 0, '斜率范围': 0,
            '斜率最小值': 0, '斜率最大值': 0, '正斜率比例': 0, '负斜率比例': 0,
            # '5%分位数': 0, '25%分位数': 0, '75%分位数': 0, '95%分位数': 0,'四分位距IQR': 0,
            '去除异常后下限': 0, '去除异常后上限': 0,
            '去除异常后占比': 0, '去除异常后平均斜率': 0, '去除异常后标准差': 0,
            '异常值比例': 0, '影响评估': '数据不足'
        }
    
    # 只取该组合对应的时间点
    combination_data = period_data.iloc[combination_time_points].copy()
    
    if len(combination_data) < window_size or '存纸率' not in combination_data.columns:
        return {
            '平均斜率': 0, '斜率标准差': 0, '斜率中位数': 0, '斜率范围': 0,
            '斜率最小值': 0, '斜率最大值': 0, '正斜率比例': 0, '负斜率比例': 0,
            # '5%分位数': 0, '25%分位数': 0, '75%分位数': 0, '95%分位数': 0,'四分位距IQR': 0, 
            '去除异常后下限': 0, '去除异常后上限': 0,
            '去除异常后占比': 0, '去除异常后平均斜率': 0, '去除异常后标准差': 0,
            '异常值比例': 0, '影响评估': '数据不足'
        }
    
    # 重置索引以便滑动窗口计算
    combination_data = combination_data.reset_index(drop=True)
    combination_data = combination_data.sort_values('时间').reset_index(drop=True)
    
    # 对存纸率进行平滑处理
    combination_data = smooth_paper_rate(combination_data, smooth_window=smooth_window, method=smooth_method)
    
    # 将时间转换为从0开始的分钟数
    start_timestamp = combination_data['时间'].iloc[0]
    combination_data['时间_分钟'] = (combination_data['时间'] - start_timestamp).dt.total_seconds() / 60
    
    # 滑动窗口计算斜率
    slopes = []
    for i in range(len(combination_data) - window_size + 1):
        window_data = combination_data.iloc[i:i + window_size]
        
        X = window_data['时间_分钟'].values.reshape(-1, 1)
        y = window_data['存纸率'].values
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            slopes.append(slope)
        except:
            continue
    
    if len(slopes) == 0:
        return {
            '平均斜率': 0, '斜率标准差': 0, '斜率中位数': 0, '斜率范围': 0,
            '斜率最小值': 0, '斜率最大值': 0, '正斜率比例': 0, '负斜率比例': 0,
            # '5%分位数': 0, '25%分位数': 0, '75%分位数': 0, '95%分位数': 0,'四分位距IQR': 0, 
            '去除异常后下限': 0, '去除异常后上限': 0,
            '去除异常后占比': 0, '去除异常后平均斜率': 0, '去除异常后标准差': 0,
            '异常值比例': 0, '影响评估': '无有效斜率'
        }
    
    slopes = np.array(slopes)
    
    # 排除斜率为0的值进行统计
    non_zero_slopes = slopes[slopes != 0]
    
    if len(non_zero_slopes) == 0:
        # 如果所有斜率都为0，返回0值统计
        return {
            '平均斜率': 0, '斜率标准差': 0, '斜率中位数': 0, '斜率范围': 0,
            '斜率最小值': 0, '斜率最大值': 0, '正斜率比例': 0, '负斜率比例': 0,
            '去除异常后下限': 0, '去除异常后上限': 0,
            '去除异常后占比': 0, '去除异常后平均斜率': 0, '去除异常后标准差': 0,
            '异常值比例': 0, '影响评估': '所有斜率为零'
        }
    
    # 使用非零斜率进行统计计算
    slopes_for_stats = non_zero_slopes
    
    # 基本统计
    mean_slope = np.mean(slopes_for_stats)
    std_slope = np.std(slopes_for_stats)
    median_slope = np.median(slopes_for_stats)
    min_slope = np.min(slopes_for_stats)
    max_slope = np.max(slopes_for_stats)
    slope_range = max_slope - min_slope
    
    # 正负斜率比例（基于原始所有斜率，包括0）
    positive_count = np.sum(slopes > 0)
    negative_count = np.sum(slopes < 0)
    zero_count = np.sum(slopes == 0)
    total_count = len(slopes)
    positive_ratio = positive_count / total_count
    negative_ratio = negative_count / total_count
    
    # 分位数（基于非零斜率）
    percentiles = np.percentile(slopes_for_stats, [5, 25, 75, 95])
    p5, p25, p75, p95 = percentiles
    iqr = p75 - p25
    
    # 异常值检测 (IQR方法，基于非零斜率)
    lower_bound = p25 - 1.5 * iqr
    upper_bound = p75 + 1.5 * iqr
    
    # 去除异常值（从非零斜率中）
    non_outliers = slopes_for_stats[(slopes_for_stats >= lower_bound) & (slopes_for_stats <= upper_bound)]
    outliers_count = len(slopes_for_stats) - len(non_outliers)
    outlier_ratio = outliers_count / len(slopes_for_stats)
    non_outlier_ratio = len(non_outliers) / len(slopes_for_stats)
    
    # 去除异常后的统计
    if len(non_outliers) > 0:
        clean_mean = np.mean(non_outliers)
        clean_std = np.std(non_outliers)
    else:
        clean_mean = mean_slope
        clean_std = std_slope
    
    # 影响评估
    if abs(clean_mean) < 0.001:
        impact = "影响很小"
    elif abs(clean_mean) < 0.01:
        impact = "影响较小"
    elif abs(clean_mean) < 0.05:
        impact = "影响中等"
    elif abs(clean_mean) < 0.1:
        impact = "影响较大"
    else:
        impact = "影响很大"
        
    if clean_mean > 0:
        impact += "(存纸率上升)"
    elif clean_mean < 0:
        impact += "(存纸率下降)"
    else:
        impact += "(存纸率稳定)"
    
    return {
        '平均斜率': round(clean_mean, 4),
        '斜率标准差': round(clean_std, 4),
        '斜率中位数': round(median_slope,4),
        '斜率范围': round(slope_range, 4),
        '斜率最小值': round(min_slope, 4),
        '斜率最大值': round(max_slope, 4),
        '正斜率比例': round(positive_ratio, 4),
        '负斜率比例': round(negative_ratio, 4),
        # '5%分位数': round(p5, 6),
        # '25%分位数': round(p25, 6),
        # '75%分位数': round(p75, 6),
        # '95%分位数': round(p95, 6),
        # '四分位距IQR': round(iqr, 6),
        '去除异常后下限': round(lower_bound, 4),
        '去除异常后上限': round(upper_bound, 4),
        '去除异常后占比': round(non_outlier_ratio, 4),
        '去除异常后平均斜率': round(clean_mean, 4),
        '去除异常后标准差': round(clean_std, 4),
        '异常值比例': round(outlier_ratio, 4),
        '影响评估': impact
    }

def calculate_slope_for_combination(paper_data, start_time, end_time, combination_time_points):
    """
    为特定停机组合的时间点计算简单斜率（保持向后兼容）
    """
    detailed_stats = calculate_detailed_slope_stats_for_combination(paper_data, start_time, end_time, combination_time_points)
    return detailed_stats['平均斜率']

def detect_machine_downtime_detailed(paper_data, start_time, end_time):
    """
    检测在指定时间段内所有停机组合及其持续时间比例
    根据小包机实际速度为0来判定停机状态
    返回格式：{
        'combinations': [
            {
                'combination': '停机组合描述',
                'machines': ['1#小包机', '2#小包机'],
                'count': 时间点数量,
                'percentage': 时间比例,
                'duration_minutes': 持续分钟数
            }
        ],
        'total_points': 总时间点数,
        'dominant_combination': 主要停机组合
    }
    """
    # 过滤时间段内的数据
    mask = (paper_data['时间'] >= start_time) & (paper_data['时间'] <= end_time)
    period_data = paper_data[mask].copy()
    
    if len(period_data) == 0:
        return {
            'combinations': [],
            'total_points': 0,
            'dominant_combination': '无数据'
        }
    
    # 检查4个小包机的状态
    machine_speed_columns = [
        '1#小包机实际速度',
        '2#小包机实际速度', 
        '3#小包机主机实际速度',
        '4#小包机主机实际速度'
    ]
    
    machine_names = ['1#小包机', '2#小包机', '3#小包机', '4#小包机']
    
    # 为每个时间点确定停机组合
    combination_counts = {}
    combination_time_points = {}  # 记录每种组合对应的时间点索引
    total_points = len(period_data)
    
    for point_idx, (idx, row) in enumerate(period_data.iterrows()):
        downtime_machines = []
        
        for i, speed_col in enumerate(machine_speed_columns):
            if speed_col in period_data.columns:
                # 当前时间点该小包机是否停机（速度=0）
                if pd.notna(row[speed_col]) and row[speed_col] == 0:
                    downtime_machines.append(machine_names[i])
        
        # 生成组合描述
        combination_key = get_machine_combination_description_downtime(downtime_machines)
        
        if combination_key not in combination_counts:
            combination_counts[combination_key] = {
                'machines': downtime_machines.copy(),
                'count': 0
            }
            combination_time_points[combination_key] = []
        
        combination_counts[combination_key]['count'] += 1
        combination_time_points[combination_key].append(point_idx)
    
    # 计算时间间隔（分钟）
    if len(period_data) > 1:
        time_diff = (period_data['时间'].iloc[-1] - period_data['时间'].iloc[0]).total_seconds() / 60
        avg_interval = time_diff / (len(period_data) - 1) if len(period_data) > 1 else 1
    else:
        avg_interval = 1
    
    # 整理结果
    combinations = []
    for combination_desc, info in combination_counts.items():
        count = info['count']
        percentage = (count / total_points) * 100
        duration_minutes = count * avg_interval
        
        # 计算该组合的详细斜率统计
        time_points = combination_time_points[combination_desc]
        detailed_slope_stats = calculate_detailed_slope_stats_for_combination(
            paper_data, start_time, end_time, time_points, 
            window_size=30, smooth_window=30, smooth_method='moving_average'
        )
        
        combinations.append({
            'combination': combination_desc,
            'machines': info['machines'],
            'count': count,
            'percentage': percentage,
            'duration_minutes': duration_minutes,
            'slope': detailed_slope_stats['平均斜率'],  # 保持兼容性
            'slope_stats': detailed_slope_stats  # 详细统计信息
        })
    
    # 按时间比例排序
    combinations.sort(key=lambda x: x['percentage'], reverse=True)
    
    # 确定主要停机组合（占比最高的）
    dominant_combination = combinations[0]['combination'] if combinations else '无数据'
    
    return {
        'combinations': combinations,
        'total_points': total_points,
        'dominant_combination': dominant_combination
    }

def detect_machine_downtime(paper_data, start_time, end_time):
    """
    保持向后兼容的简化版本，返回主要停机组合
    """
    detailed_result = detect_machine_downtime_detailed(paper_data, start_time, end_time)
    
    if not detailed_result['combinations']:
        return []
    
    # 返回主要停机组合中的机器列表
    dominant_combo = detailed_result['combinations'][0]
    return dominant_combo['machines']

def calculate_paper_storage_slope(paper_data, start_time, end_time, window_size=30, smooth_window=30, smooth_method='moving_average'):
    """
    使用滑动窗口计算指定时间段内的存纸率斜率
    
    Parameters:
    - paper_data: 存纸架数据
    - start_time: 开始时间
    - end_time: 结束时间  
    - window_size: 窗口大小，默认30个数据点
    - smooth_window: 平滑窗口大小，默认5
    - smooth_method: 平滑方法，默认'moving_average'
    
    Returns:
    - 平均斜率 (单位：%/分钟)
    """
    from sklearn.linear_model import LinearRegression
    
    # 过滤时间段内的数据
    mask = (paper_data['时间'] >= start_time) & (paper_data['时间'] <= end_time)
    period_data = paper_data[mask].copy()
    
    if len(period_data) < window_size:
        # 如果数据点不足窗口大小，使用传统方法
        if len(period_data) < 2:
            return 0
        
        time_diff = (period_data['时间'].max() - period_data['时间'].min()).total_seconds() / 60
        if time_diff == 0:
            return 0
            
        if '存纸率' in period_data.columns:
            paper_rate_change = period_data['存纸率'].iloc[-1] - period_data['存纸率'].iloc[0]
            return paper_rate_change / time_diff
        return 0
    
    if '存纸率' not in period_data.columns:
        return 0
    
    # 重置索引以便滑动窗口计算
    period_data = period_data.reset_index(drop=True)
    
    # 对存纸率进行平滑处理
    period_data = smooth_paper_rate(period_data, smooth_window=smooth_window, method=smooth_method)
    
    # 将时间转换为从0开始的分钟数，便于线性回归
    start_timestamp = period_data['时间'].iloc[0]
    period_data['时间_分钟'] = (period_data['时间'] - start_timestamp).dt.total_seconds() / 60
    
    slopes = []
    
    # 滑动窗口计算斜率
    for i in range(len(period_data) - window_size + 1):
        window_data = period_data.iloc[i:i + window_size]
        
        # 准备线性回归数据
        X = window_data['时间_分钟'].values.reshape(-1, 1)
        y = window_data['存纸率'].values
        
        # 线性回归计算斜率
        try:
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]  # 斜率系数，单位：%/分钟
            slopes.append(slope)
        except:
            continue
    
    if len(slopes) == 0:
        return 0
    
    # 排除斜率为0的值
    slopes = np.array(slopes)
    non_zero_slopes = slopes[slopes != 0]
    
    if len(non_zero_slopes) == 0:
        return 0
    
    # 返回非零斜率的平均值
    return np.mean(non_zero_slopes)

def calculate_detailed_slope_stats(paper_data, start_time, end_time, window_size, smooth_window=30, smooth_method='moving_average'):
    """
    使用滑动窗口计算详细的斜率统计信息
    
    Parameters:
    - paper_data: 存纸架数据
    - start_time: 开始时间
    - end_time: 结束时间
    - window_size: 滑动窗口大小
    - smooth_window: 平滑窗口大小，默认5
    - smooth_method: 平滑方法，默认'moving_average'
    
    Returns:
    - dict: 包含平均斜率、标准差、最大斜率、最小斜率、斜率变化范围等
    """
    from sklearn.linear_model import LinearRegression
    
    # 过滤时间段内的数据
    mask = (paper_data['时间'] >= start_time) & (paper_data['时间'] <= end_time)
    period_data = paper_data[mask].copy()
    
    if len(period_data) < window_size or '存纸率' not in period_data.columns:
        return {
            'mean_slope': 0,
            'std_slope': 0,
            'max_slope': 0,
            'min_slope': 0,
            'slope_range': 0,
            'window_count': 0
        }
    
    # 重置索引以便滑动窗口计算
    period_data = period_data.reset_index(drop=True)
    
    # 对存纸率进行平滑处理
    period_data = smooth_paper_rate(period_data, smooth_window=smooth_window, method=smooth_method)
    
    # 将时间转换为从0开始的分钟数
    start_timestamp = period_data['时间'].iloc[0]
    period_data['时间_分钟'] = (period_data['时间'] - start_timestamp).dt.total_seconds() / 60
    
    slopes = []
    
    # 滑动窗口计算斜率
    for i in range(len(period_data) - window_size + 1):
        window_data = period_data.iloc[i:i + window_size]
        
        X = window_data['时间_分钟'].values.reshape(-1, 1)
        y = window_data['存纸率'].values
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            slopes.append(slope)
        except:
            continue
    
    if len(slopes) == 0:
        return {
            'mean_slope': 0,
            'std_slope': 0,
            'max_slope': 0,
            'min_slope': 0,
            'slope_range': 0,
            'window_count': 0
        }
    
    slopes = np.array(slopes)
    original_window_count = len(slopes)
    
    # 排除斜率为0的值进行统计计算
    non_zero_slopes = slopes[slopes != 0]
    
    if len(non_zero_slopes) == 0:
        return {
            'mean_slope': 0,
            'std_slope': 0,
            'max_slope': 0,
            'min_slope': 0,
            'slope_range': 0,
            'window_count': original_window_count
        }
    
    return {
        'mean_slope': np.mean(non_zero_slopes),
        'std_slope': np.std(non_zero_slopes),
        'max_slope': np.max(non_zero_slopes),
        'min_slope': np.min(non_zero_slopes),
        'slope_range': np.max(non_zero_slopes) - np.min(non_zero_slopes),
        'window_count': original_window_count
    }

def get_machine_combination_description_downtime(downtime_machines):
    """
    根据停机的小包机数量和名称生成描述
    """
    count = len(downtime_machines)
    if count == 0:
        return "无小包机停机"
    elif count == 1:
        return f"单个小包机停机: {downtime_machines[0]}"
    elif count == 2:
        return f"两个小包机同时停机: {', '.join(downtime_machines)}"
    elif count == 3:
        return f"三个小包机同时停机: {', '.join(downtime_machines)}"
    elif count == 4:
        return f"四个小包机同时停机: {', '.join(downtime_machines)}"
    else:
        return f"{count}个小包机同时停机: {', '.join(downtime_machines)}"

def get_machine_combination_description(running_machines):
    """
    根据正常运行的小包机数量和名称生成描述（保持兼容性）
    """
    count = len(running_machines)
    if count == 0:
        return "无小包机正常运行"
    elif count == 1:
        return f"单个小包机正常运行: {running_machines[0]}"
    elif count == 2:
        return f"两个小包机同时正常运行: {', '.join(running_machines)}"
    elif count == 3:
        return f"三个小包机同时正常运行: {', '.join(running_machines)}"
    elif count == 4:
        return f"四个小包机同时正常运行: {', '.join(running_machines)}"
    else:
        return f"{count}个小包机同时正常运行: {', '.join(running_machines)}"

def test_smooth_function():
    """
    测试平滑处理功能
    """
    print("\n=== 测试平滑处理功能 ===")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        '时间': pd.date_range('2023-01-01', periods=20, freq='1min'),
        '存纸率': [50, 52, 48, 55, 45, 60, 40, 65, 35, 70, 
                  30, 75, 25, 80, 20, 85, 15, 90, 10, 95]  # 带噪声的递增数据
    })
    
    print("原始存纸率数据:")
    print(test_data['存纸率'].tolist())
    
    # 测试不同平滑方法
    methods = ['moving_average', 'exponential', 'gaussian']
    
    for method in methods:
        smoothed_data = smooth_paper_rate(test_data, smooth_window=30, method=method)
        print(f"\n{method} 平滑后:")
        print([round(x, 2) for x in smoothed_data['存纸率'].tolist()])
    
    print("\n=== 平滑处理功能测试完成 ===\n")

def main():
    print("开始分析折叠机停机时间段内的小包机停机状态...")
    print("注意：已启用存纸率平滑处理功能（移动平均，窗口大小=5）")
    
    # 读取折叠机停机数据
    try:
        folding_downtime = read_csv_with_encoding('折叠机停机时间段_25.csv')
        print(f"折叠机正常运行数据行数: {len(folding_downtime)}")
    except Exception as e:
        print(f"读取折叠机停机数据失败: {e}")
        return
    
    # 转换时间列
    folding_downtime['开始时间'] = pd.to_datetime(folding_downtime['开始时间'])
    folding_downtime['结束时间'] = pd.to_datetime(folding_downtime['结束时间'])
    
    # 读取存纸架数据
    try:
        paper_storage_data = read_csv_with_encoding('存纸架数据汇总.csv')
        print(f"存纸架数据行数: {len(paper_storage_data)}")
        paper_storage_data['时间'] = pd.to_datetime(paper_storage_data['时间'])
    except Exception as e:
        print(f"读取存纸架数据失败: {e}")
        return
    
    results = []
    
    print("\n开始分析每个折叠机正常运行时间段...")
    
    # 分析每个折叠机停机时间段
    for idx, row in folding_downtime.iterrows():
        start_time = row['开始时间']
        end_time = row['结束时间']
        duration = row['持续时间']
        
        # 检测在此时间段内停机的小包机（详细分析）
        downtime_detail = detect_machine_downtime_detailed(paper_storage_data, start_time, end_time)
        
        # 生成停机组合描述（使用主要组合）
        downtime_combination = downtime_detail['dominant_combination']
        
        # 生成详细停机组合信息（包含详细斜率统计）
        combinations_detail = []
        combinations_slopes = {}  # 存储每种组合的斜率
        combinations_detailed_stats = {}  # 存储每种组合的详细统计
        
        for combo in downtime_detail['combinations']:
            slope_desc = f"斜率:{combo['slope']:.4f}" if combo['slope'] != 0 else "斜率:0.0000"
            combinations_detail.append(f"{combo['combination']}({combo['percentage']:.1f}%, {slope_desc})")
            combinations_slopes[combo['combination']] = combo['slope']
            combinations_detailed_stats[combo['combination']] = combo['slope_stats']
        
        downtime_combinations_all = '; '.join(combinations_detail)
        
        # 计算存纸率斜率（滑动窗口方法，包含平滑处理）
        slope_stats = calculate_detailed_slope_stats(
            paper_storage_data, start_time, end_time, window_size=30, 
            smooth_window=30, smooth_method='moving_average'
        )
        slope = slope_stats['mean_slope']
        slope_description = f"{slope:.4f}/分钟"
        
        # 生成详细的斜率组合描述
        if slope_stats['window_count'] > 0:
            detailed_slope_desc = (f"平均:{slope:.4f}, "
                                 f"标准差:{slope_stats['std_slope']:.4f}, "
                                 f"范围:[{slope_stats['min_slope']:.4f}, {slope_stats['max_slope']:.4f}], "
                                 f"窗口数:{slope_stats['window_count']}")
        else:
            detailed_slope_desc = slope_description
        
        # 获取主要停机组合的详细斜率统计
        main_combination_slope = combinations_slopes.get(downtime_combination, 0)
        main_combination_stats = combinations_detailed_stats.get(downtime_combination, {})
        
        # 基础结果
        result_row = {
            '开始时间': start_time,
            '结束时间': end_time,
            '持续时间': duration,
            '主要停机组合': downtime_combination,
            '主要组合斜率': main_combination_slope,
            '所有停机组合': downtime_combinations_all,
            '总时间点数': downtime_detail['total_points'],
            '详细斜率统计': detailed_slope_desc,
            '全段平均斜率': slope,
            '斜率标准差': slope_stats['std_slope'],
            '最大斜率': slope_stats['max_slope'],
            '最小斜率': slope_stats['min_slope'],
            '斜率变化范围': slope_stats['slope_range'],
            '滑动窗口数量': slope_stats['window_count']
        }
        
        # 添加主要停机组合的详细斜率统计
        for stat_name, stat_value in main_combination_stats.items():
            result_row[f'主要组合_{stat_name}'] = stat_value
        
        # 添加所有停机组合的详细统计信息
        for combo in downtime_detail['combinations']:
            combo_name = combo['combination']
            combo_stats = combo['slope_stats']
            # 使用组合名称前缀避免列名冲突
            safe_combo_name = combo_name.replace(':', '_').replace('#', '号').replace(',', '').replace(' ', '').replace(':', '_')
            for stat_name, stat_value in combo_stats.items():
                result_row[f'{safe_combo_name}_{stat_name}'] = stat_value
        
        results.append(result_row)
        
        # 显示进度
        if (idx + 1) % 50 == 0:
            print(f"已处理 {idx + 1}/{len(folding_downtime)} 个时间段")
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv('折叠机停机分析结果_25_多状态_30.csv', index=False, encoding='utf-8-sig')
    print(f"\n分析完成！结果已保存到 '折叠机停机分析结果_25_多状态_30.csv'")
    print(f"共分析了 {len(results)} 个停机时间段")
    
    # 统计不同主要停机组合的数量
    print("\n主要停机组合统计:")
    combination_counts = result_df['主要停机组合'].value_counts()
    for combination, count in combination_counts.items():
        print(f"{combination}: {count} 次")
    
    # 分析不同主要停机组合的平均斜率（滑动窗口方法）
    print("\n不同主要停机组合的斜率分析:")
    slope_analysis = result_df.groupby('主要停机组合').agg({
        '主要组合斜率': ['mean', 'std', 'count'],
        '全段平均斜率': ['mean', 'std'],
        '斜率标准差': 'mean',
        '斜率变化范围': 'mean',
        '滑动窗口数量': 'mean'
    }).round(4)
    
    # 重命名列
    slope_analysis.columns = ['主要组合斜率_均值', '主要组合斜率_标准差', '计数', 
                             '全段斜率_均值', '全段斜率_标准差',
                             '平均斜率内标准差', '平均斜率变化范围', '平均窗口数量']
    print(slope_analysis)
    
    # 专门分析每种停机组合的斜率特征
    print("\n各停机组合斜率对比分析:")
    print("="*80)
    
    # 计算每种停机组合在所有时间段中的斜率分布
    for combination in result_df['主要停机组合'].unique():
        combo_data = result_df[result_df['主要停机组合'] == combination]
        combo_slopes = combo_data['主要组合斜率']
        
        print(f"\n【{combination}】:")
        print(f"  出现次数: {len(combo_data)} 次")
        print(f"  斜率统计: 均值={combo_slopes.mean():.4f}, 标准差={combo_slopes.std():.4f}")
        print(f"  斜率范围: [{combo_slopes.min():.4f}, {combo_slopes.max():.4f}]")
        
        # 斜率分布分析
        positive_count = (combo_slopes > 0).sum()
        negative_count = (combo_slopes < 0).sum()
        zero_count = (combo_slopes == 0).sum()
        
        print(f"  斜率分布: 正斜率 {positive_count} 次, 负斜率 {negative_count} 次, 零斜率 {zero_count} 次")
        
        if len(combo_data) >= 5:  # 只有足够样本才显示详细统计
            print(f"  四分位数: Q1={combo_slopes.quantile(0.25):.4f}, "
                  f"中位数={combo_slopes.median():.4f}, Q3={combo_slopes.quantile(0.75):.4f}")
    
    # 显示滑动窗口分析和平滑处理的优势
    print("\n滑动窗口分析和平滑处理优势:")
    valid_windows = result_df[result_df['滑动窗口数量'] > 0]
    print(f"有效滑动窗口分析的停机次数: {len(valid_windows)}/{len(result_df)}")
    print(f"平均每次停机的滑动窗口数量: {valid_windows['滑动窗口数量'].mean():.2f}")
    print("已对存纸率数据进行平滑处理，减少噪声对斜率计算的影响")
    
    # 显示前几行结果
    print("\n前10行详细结果预览:")
    display_columns = ['开始时间', '结束时间', '持续时间', '主要停机组合', '主要组合斜率', '所有停机组合', '总时间点数']
    print(result_df[display_columns].head(10))

if __name__ == "__main__":
    # 可以取消注释下面这行来测试平滑处理功能
    # test_smooth_function()
    
    main() 