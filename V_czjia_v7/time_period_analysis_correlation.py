import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.signal import correlate, correlation_lags
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_by_time_periods_correlation():
    """使用互相关性算法按照时间段进行并行流程分析"""
    
    # 读取时间段数据
    try:
        time_periods = pd.read_csv('折叠机正常运行且高存纸率时间段_最终结果_存纸率1.csv')
        print(f"加载时间段数据：{len(time_periods)}个时间段")
        
        # 转换时间格式
        time_periods['开始时间'] = pd.to_datetime(time_periods['开始时间'])
        time_periods['结束时间'] = pd.to_datetime(time_periods['结束时间'])
        
    except Exception as e:
        print(f"读取时间段数据出错：{e}")
        return
    
    # 读取汇总数据
    try:
        summary_data = pd.read_csv('存纸架数据汇总.csv', encoding='utf-8-sig')
        print(f"加载汇总数据：{len(summary_data)}条记录")
        
        # 转换时间格式
        summary_data['时间'] = pd.to_datetime(summary_data['时间'])
        summary_data = summary_data.sort_values('时间')
        
    except Exception as e:
        print(f"读取汇总数据出错：{e}")
        return
    
    # 定义并行流程结构
    process_structure = define_process_structure()
    
    # 对每个时间段进行分析
    period_results = []
    all_period_data = []
    
    print(f"\n=== 开始使用互相关性算法分析 {len(time_periods)} 个时间段 ===")
    
    for index, period in time_periods.iterrows():
        period_id = f"时间段{index+1:02d}"
        start_time = period['开始时间']
        end_time = period['结束时间']
        duration = period['持续时间']
        
        print(f"\n{'='*50}")
        print(f"分析 {period_id}: {start_time} ~ {end_time}")
        print(f"持续时间: {duration}")
        
        # 筛选该时间段的数据
        period_mask = (summary_data['时间'] >= start_time) & (summary_data['时间'] <= end_time)
        period_data = summary_data[period_mask].copy()
        
        if len(period_data) < 50:  # 互相关性需要更多数据点
            print(f"{period_id} 数据量不足 ({len(period_data)}条)，跳过分析")
            continue
        
        print(f"该时间段数据量: {len(period_data)}条")
        
        # 使用互相关性算法分析该时间段的流程延时
        period_result = analyze_single_period_correlation(period_data, period_id, start_time, end_time, process_structure)
        
        if period_result:
            period_results.append(period_result)
            all_period_data.extend(period_result['详细数据'])
    
    if period_results:
        # 汇总分析所有时间段
        comprehensive_period_analysis_correlation(period_results, all_period_data)
    else:
        print("没有成功分析的时间段")

def define_process_structure():
    """定义并行流程结构"""
    
    # 共同前置流程
    common_sequence = [
        ('1_折叠机入包数', '折叠机入包数'),
        ('2_折叠机出包数', '折叠机出包数'),
        ('3_外循环进内循环', '外循环进内循环纸条数量')
    ]
    
    # 四条并行生产线
    parallel_lines = {
        '生产线1': [
            ('4_第一裁切通道', '进第一裁切通道纸条计数'),
            ('5_1号有效切数', '1#有效切数'),
            ('6_1号小包机入包数', '1#小包机入包数'),
        ],
        '生产线2': [
            ('4_第二裁切通道', '进第二裁切通道纸条计数'),
            ('5_2号有效切数', '2#有效切数'),
            ('6_2号小包机入包数', '2#小包机入包数'),
        ],
        '生产线3': [
            ('4_第三裁切通道', '进第三裁切通道纸条计数'),
            ('5_3号有效切数', '3#有效切数'),
            ('6_3号小包机入包数', '3#小包机入包数'),
        ],
        '生产线4': [
            ('4_第四裁切通道', '进第四裁切通道纸条计数'),
            ('5_4号有效切数', '4#有效切数'),
            ('6_4号小包机入包数', '4#小包机入包数'),
        ]
    }
    
    return {
        'common_sequence': common_sequence,
        'parallel_lines': parallel_lines
    }

def analyze_single_period_correlation(data, period_id, start_time, end_time, process_structure):
    """使用互相关性算法分析单个时间段的流程延时"""
    
    common_sequence = process_structure['common_sequence']
    parallel_lines = process_structure['parallel_lines']
    
    period_result = {
        '时间段ID': period_id,
        '开始时间': start_time,
        '结束时间': end_time,
        '数据量': len(data),
        '共同流程': None,
        '生产线结果': {},
        '详细数据': []
    }
    
    # 分析共同前置流程
    common_results = analyze_sequence_correlation(data, common_sequence, f"{period_id}_共同流程")
    if common_results:
        period_result['共同流程'] = common_results
        for result in common_results:
            result['时间段ID'] = period_id
            result['开始时间'] = start_time
            result['结束时间'] = end_time
        period_result['详细数据'].extend(common_results)
    
    # 分析各条并行生产线
    for line_name, line_sequence in parallel_lines.items():
        # 将外循环进内循环作为起点连接到各生产线
        full_sequence = [common_sequence[-1]] + line_sequence
        line_results = analyze_sequence_correlation(data, full_sequence, f"{period_id}_{line_name}")
        
        if line_results:
            period_result['生产线结果'][line_name] = line_results
            for result in line_results:
                result['时间段ID'] = period_id
                result['开始时间'] = start_time
                result['结束时间'] = end_time
            period_result['详细数据'].extend(line_results)
    
    # 计算该时间段的生产线性能指标
    line_performance = {}
    for line_name, line_results in period_result['生产线结果'].items():
        if line_results:
            total_avg_delay = sum([r['平均延时(秒)'] for r in line_results])
            total_median_delay = sum([r['中位延时(秒)'] for r in line_results])
            avg_confidence = np.mean([r['相关性置信度'] for r in line_results])
            
            line_performance[line_name] = {
                '环节数': len(line_results),
                '总平均延时(秒)': total_avg_delay,
                '总中位延时(秒)': total_median_delay,
                '平均单环节延时(秒)': total_avg_delay / len(line_results),
                '平均相关性置信度': avg_confidence,
                '最大单环节延时(秒)': max([r['平均延时(秒)'] for r in line_results])
            }
    
    period_result['生产线性能'] = line_performance
    
    # 显示该时间段的关键指标
    if line_performance:
        print(f"\n{period_id} 生产线性能 (基于互相关性算法):")
        for line_name, performance in line_performance.items():
            print(f"  {line_name}: 总延时 {performance['总平均延时(秒)']:.1f}秒 (置信度: {performance['平均相关性置信度']:.3f})")
        
        # 找出最优和最差生产线
        best_line = min(line_performance.items(), key=lambda x: x[1]['总平均延时(秒)'])
        worst_line = max(line_performance.items(), key=lambda x: x[1]['总平均延时(秒)'])
        
        print(f"  最优: {best_line[0]} ({best_line[1]['总平均延时(秒)']:.1f}秒)")
        print(f"  最差: {worst_line[0]} ({worst_line[1]['总平均延时(秒)']:.1f}秒)")
        print(f"  差异: {worst_line[1]['总平均延时(秒)'] - best_line[1]['总平均延时(秒)']:.1f}秒")
    
    return period_result

def analyze_sequence_correlation(data, sequence, sequence_name):
    """使用互相关性算法分析单个序列的传输延时"""
    
    # 检查数据中存在的列
    available_points = []
    
    for point_name, column_name in sequence:
        if column_name in data.columns:
            available_points.append((point_name, column_name))
    
    if len(available_points) < 2:
        return None
    
    # 计算相邻点位之间的延时
    delays_results = []
    
    for i in range(len(available_points) - 1):
        current_point = available_points[i]
        next_point = available_points[i + 1]
        
        current_name, current_col = current_point
        next_name, next_col = next_point
        
        # 使用互相关性算法计算延时
        delay_result = calculate_correlation_delay(data, current_col, next_col)
        
        if delay_result:
            delay_stats = {
                '生产线': sequence_name,
                '起始点位': current_name,
                '目标点位': next_name,
                '起始列名': current_col,
                '目标列名': next_col,
                '平均延时(秒)': delay_result['平均延时'],
                '中位延时(秒)': delay_result['中位延时'],
                '最小延时(秒)': delay_result['最小延时'],
                '最大延时(秒)': delay_result['最大延时'],
                '标准差(秒)': delay_result['标准差'],
                '样本数': delay_result['样本数'],
                '最大相关系数': delay_result['最大相关系数'],
                '相关性置信度': delay_result['相关性置信度'],
                '最优延时(秒)': delay_result['最优延时']
            }
            
            delays_results.append(delay_stats)
    
    return delays_results

def calculate_correlation_delay(data, source_col, target_col, window_size=60, max_delay=300):
    """
    使用互相关性算法计算传输延时
    
    参数:
    - data: 数据框
    - source_col: 源信号列名
    - target_col: 目标信号列名  
    - window_size: 滑动窗口大小(数据点数)
    - max_delay: 最大允许延时(秒)
    """
    
    if source_col not in data.columns or target_col not in data.columns:
        return None
    
    # 获取信号数据并处理缺失值
    source_signal = data[source_col].fillna(method='ffill').fillna(method='bfill')
    target_signal = data[target_col].fillna(method='ffill').fillna(method='bfill')
    
    if len(source_signal) < window_size or len(target_signal) < window_size:
        return None
    
    # 信号预处理：标准化和去趋势
    source_signal = standardize_signal(source_signal)
    target_signal = standardize_signal(target_signal)
    
    # 计算时间间隔
    time_intervals = data['时间'].diff().dt.total_seconds().dropna()
    avg_interval = time_intervals.median()  # 使用中位数更稳健
    
    if avg_interval <= 0:
        return None
    
    # 使用滑动窗口进行互相关分析
    delays = []
    correlations = []
    
    # 计算可能的延时范围（以数据点为单位）
    max_delay_points = min(int(max_delay / avg_interval), len(source_signal) // 4)
    
    step_size = max(1, window_size // 4)  # 滑动步长
    
    for start_idx in range(0, len(source_signal) - window_size, step_size):
        end_idx = start_idx + window_size
        
        if end_idx >= len(source_signal):
            break
        
        # 提取窗口信号
        source_window = source_signal.iloc[start_idx:end_idx].values
        target_window = target_signal.iloc[start_idx:end_idx].values
        
        # 计算互相关
        correlation = correlate(target_window, source_window, mode='full')
        lags = correlation_lags(len(target_window), len(source_window), mode='full')
        
        # 限制延时范围
        valid_mask = (lags >= 0) & (lags <= max_delay_points)
        valid_correlation = correlation[valid_mask]
        valid_lags = lags[valid_mask]
        
        if len(valid_correlation) > 0:
            # 找到最大相关系数对应的延时
            max_corr_idx = np.argmax(valid_correlation)
            best_lag = valid_lags[max_corr_idx]
            best_correlation = valid_correlation[max_corr_idx]
            
            # 转换为时间延时
            delay_seconds = best_lag * avg_interval
            
            # 质量检查：相关系数必须足够高
            if best_correlation > 0.3:  # 相关性阈值
                delays.append(delay_seconds)
                correlations.append(best_correlation)
    
    if len(delays) == 0:
        return None
    
    # 异常值检测和过滤
    delays = np.array(delays)
    correlations = np.array(correlations)
    
    # 使用IQR方法去除异常值
    Q1 = np.percentile(delays, 25)
    Q3 = np.percentile(delays, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    valid_mask = (delays >= lower_bound) & (delays <= upper_bound) & (delays >= 0)
    filtered_delays = delays[valid_mask]
    filtered_correlations = correlations[valid_mask]
    
    if len(filtered_delays) == 0:
        return None
    
    # 计算最终统计结果
    result = {
        '平均延时': np.mean(filtered_delays),
        '中位延时': np.median(filtered_delays),
        '最小延时': np.min(filtered_delays),
        '最大延时': np.max(filtered_delays),
        '标准差': np.std(filtered_delays),
        '样本数': len(filtered_delays),
        '最大相关系数': np.max(filtered_correlations),
        '相关性置信度': np.mean(filtered_correlations),
        '最优延时': filtered_delays[np.argmax(filtered_correlations)]
    }
    
    return result

def standardize_signal(signal):
    """信号标准化和预处理"""
    
    # 去除异常值
    Q1 = signal.quantile(0.25)
    Q3 = signal.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    
    # 限制信号在合理范围内
    signal_clipped = signal.clip(lower_bound, upper_bound)
    
    # 标准化
    signal_std = (signal_clipped - signal_clipped.mean()) / (signal_clipped.std() + 1e-8)
    
    # 去趋势（去除线性趋势）
    x = np.arange(len(signal_std))
    if len(x) > 1:
        coeffs = np.polyfit(x, signal_std, 1)
        trend = coeffs[0] * x + coeffs[1]
        signal_detrended = signal_std - trend
    else:
        signal_detrended = signal_std
    
    return signal_detrended

def comprehensive_period_analysis_correlation(period_results, all_period_data):
    """对所有时间段进行综合分析（基于互相关性算法）"""
    
    print(f"\n{'='*60}")
    print(f"=== 时间段综合分析 (基于互相关性算法，共{len(period_results)}个时间段) ===")
    print(f"{'='*60}")
    
    # 保存所有详细数据
    if all_period_data:
        all_df = pd.DataFrame(all_period_data)
        all_df.to_csv('时间段并行流程分析_详细数据_分段_2.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
        print(f"所有时间段详细数据已保存：时间段并行流程分析_详细数据_分段_2.csv")
    
    # 创建时间段性能汇总
    period_summary = []
    
    for period_result in period_results:
        period_id = period_result['时间段ID']
        start_time = period_result['开始时间']
        line_performance = period_result['生产线性能']
        
        if line_performance:
            summary_row = {
                '时间段ID': period_id,
                '开始时间': start_time,
                '数据量': period_result['数据量']
            }
            
            # 添加各生产线的总延时和置信度
            for line_name, performance in line_performance.items():
                summary_row[f'{line_name}_总延时'] = performance['总平均延时(秒)']
                summary_row[f'{line_name}_置信度'] = performance['平均相关性置信度']
            
            # 计算该时间段的整体指标
            delays = [perf['总平均延时(秒)'] for perf in line_performance.values()]
            confidences = [perf['平均相关性置信度'] for perf in line_performance.values()]
            
            summary_row['平均延时'] = np.mean(delays)
            summary_row['最大延时'] = np.max(delays)
            summary_row['最小延时'] = np.min(delays)
            summary_row['延时差异'] = np.max(delays) - np.min(delays)
            summary_row['平均置信度'] = np.mean(confidences)
            summary_row['最低置信度'] = np.min(confidences)
            
            period_summary.append(summary_row)
    
    if period_summary:
        summary_df = pd.DataFrame(period_summary)
        summary_df.to_csv('时间段性能汇总_分段_2.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
        print(f"时间段性能汇总已保存：时间段性能汇总_分段_2.csv")
        
        # 分析时间段性能趋势
        analyze_period_trends_correlation(summary_df)
        
        # 识别最优和最差时间段
        identify_best_worst_periods_correlation(summary_df, period_results)
        
        # 创建可视化
        create_period_visualization_correlation(summary_df, period_results)
        
        # 创建流程汇总统计
        create_process_summary_correlation(all_period_data)

def analyze_period_trends_correlation(summary_df):
    """分析时间段性能趋势（基于互相关性算法）"""
    
    print(f"\n=== 时间段性能趋势分析 (互相关性算法) ===")
    
    # 整体性能统计
    print(f"平均延时统计:")
    print(f"  整体平均: {summary_df['平均延时'].mean():.2f}秒")
    print(f"  最好时段: {summary_df['平均延时'].min():.2f}秒")
    print(f"  最差时段: {summary_df['平均延时'].max():.2f}秒")
    print(f"  标准差: {summary_df['平均延时'].std():.2f}秒")
    
    # 置信度统计
    print(f"\n相关性置信度统计:")
    print(f"  整体平均置信度: {summary_df['平均置信度'].mean():.4f}")
    print(f"  最高置信度: {summary_df['平均置信度'].max():.4f}")
    print(f"  最低置信度: {summary_df['平均置信度'].min():.4f}")
    print(f"  置信度标准差: {summary_df['平均置信度'].std():.4f}")
    
    # 高置信度时间段统计
    high_confidence_threshold = 0.6
    high_conf_periods = summary_df[summary_df['平均置信度'] >= high_confidence_threshold]
    print(f"\n高置信度时间段 (置信度≥{high_confidence_threshold}):")
    print(f"  数量: {len(high_conf_periods)}/{len(summary_df)} ({len(high_conf_periods)/len(summary_df)*100:.1f}%)")
    if len(high_conf_periods) > 0:
        print(f"  平均延时: {high_conf_periods['平均延时'].mean():.2f}秒")
        print(f"  平均置信度: {high_conf_periods['平均置信度'].mean():.4f}")
    
    # 生产线稳定性分析
    line_columns = [col for col in summary_df.columns if '_总延时' in col]
    confidence_columns = [col for col in summary_df.columns if '_置信度' in col]
    
    print(f"\n各生产线在不同时间段的表现:")
    for col in line_columns:
        line_name = col.replace('_总延时', '')
        conf_col = f'{line_name}_置信度'
        
        line_data = summary_df[col].dropna()
        conf_data = summary_df[conf_col].dropna() if conf_col in summary_df.columns else None
        
        if len(line_data) > 0:
            print(f"  {line_name}:")
            print(f"    平均延时: {line_data.mean():.2f}秒")
            print(f"    变异系数: {(line_data.std()/line_data.mean()*100):.1f}%")
            print(f"    最佳表现: {line_data.min():.2f}秒")
            print(f"    最差表现: {line_data.max():.2f}秒")
            if conf_data is not None and len(conf_data) > 0:
                print(f"    平均置信度: {conf_data.mean():.4f}")

def identify_best_worst_periods_correlation(summary_df, period_results):
    """识别最优和最差时间段（基于互相关性算法）"""
    
    print(f"\n=== 最优/最差时间段识别 (互相关性算法) ===")
    
    # 综合评分：延时越小、置信度越高越好
    summary_df['综合评分'] = (1 / summary_df['平均延时']) * summary_df['平均置信度']
    
    # 按综合评分排序
    best_period_idx = summary_df['综合评分'].idxmax()
    worst_period_idx = summary_df['综合评分'].idxmin()
    
    best_period = summary_df.iloc[best_period_idx]
    worst_period = summary_df.iloc[worst_period_idx]
    
    print(f"最优时间段: {best_period['时间段ID']}")
    print(f"  开始时间: {best_period['开始时间']}")
    print(f"  平均延时: {best_period['平均延时']:.2f}秒")
    print(f"  平均置信度: {best_period['平均置信度']:.4f}")
    print(f"  综合评分: {best_period['综合评分']:.6f}")
    
    print(f"\n最差时间段: {worst_period['时间段ID']}")
    print(f"  开始时间: {worst_period['开始时间']}")
    print(f"  平均延时: {worst_period['平均延时']:.2f}秒")
    print(f"  平均置信度: {worst_period['平均置信度']:.4f}")
    print(f"  综合评分: {worst_period['综合评分']:.6f}")
    
    # 分析性能差异原因
    performance_gap = worst_period['平均延时'] - best_period['平均延时']
    confidence_gap = best_period['平均置信度'] - worst_period['平均置信度']
    
    print(f"\n性能差异分析:")
    print(f"  延时差异: {performance_gap:.2f}秒 ({performance_gap/best_period['平均延时']*100:.1f}%)")
    print(f"  置信度差异: {confidence_gap:.4f} ({confidence_gap/best_period['平均置信度']*100:.1f}%)")

def create_period_visualization_correlation(summary_df, period_results):
    """创建时间段分析可视化（基于互相关性算法）"""
    
    plt.figure(figsize=(20, 18))
    
    # 子图1: 时间段平均延时趋势
    plt.subplot(3, 4, 1)
    plt.plot(range(len(summary_df)), summary_df['平均延时'], 'bo-', alpha=0.7)
    plt.title('各时间段平均延时趋势\n(互相关性算法)')
    plt.xlabel('时间段序号')
    plt.ylabel('平均延时(秒)')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 置信度趋势
    plt.subplot(3, 4, 2)
    plt.plot(range(len(summary_df)), summary_df['平均置信度'], 'go-', alpha=0.7)
    plt.title('各时间段相关性置信度趋势')
    plt.xlabel('时间段序号')
    plt.ylabel('平均置信度')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 延时vs置信度散点图
    plt.subplot(3, 4, 3)
    plt.scatter(summary_df['平均延时'], summary_df['平均置信度'], alpha=0.6, s=50)
    plt.xlabel('平均延时(秒)')
    plt.ylabel('平均置信度')
    plt.title('延时与置信度关系')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 各生产线延时对比箱线图
    plt.subplot(3, 4, 4)
    line_columns = [col for col in summary_df.columns if '_总延时' in col]
    line_data = []
    line_labels = []
    for col in line_columns:
        data = summary_df[col].dropna()
        if len(data) > 0:
            line_data.append(data)
            line_labels.append(col.replace('_总延时', ''))
    
    if line_data:
        plt.boxplot(line_data, labels=line_labels)
        plt.title('各生产线延时分布')
        plt.ylabel('总延时(秒)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 子图5: 各生产线置信度对比
    plt.subplot(3, 4, 5)
    confidence_columns = [col for col in summary_df.columns if '_置信度' in col]
    conf_data = []
    conf_labels = []
    for col in confidence_columns:
        data = summary_df[col].dropna()
        if len(data) > 0:
            conf_data.append(data)
            conf_labels.append(col.replace('_置信度', ''))
    
    if conf_data:
        plt.boxplot(conf_data, labels=conf_labels)
        plt.title('各生产线置信度分布')
        plt.ylabel('置信度')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 子图6-8: 各生产线延时随时间变化
    for i, col in enumerate(line_columns[:3]):
        plt.subplot(3, 4, 6+i)
        line_name = col.replace('_总延时', '')
        plt.plot(range(len(summary_df)), summary_df[col], 'o-', alpha=0.7)
        plt.title(f'{line_name}延时变化')
        plt.xlabel('时间段序号')
        plt.ylabel('总延时(秒)')
        plt.grid(True, alpha=0.3)
    
    # 子图9: 延时差异趋势
    plt.subplot(3, 4, 9)
    plt.plot(range(len(summary_df)), summary_df['延时差异'], 'ro-', alpha=0.7)
    plt.title('各时间段生产线差异')
    plt.xlabel('时间段序号')
    plt.ylabel('最大-最小延时(秒)')
    plt.grid(True, alpha=0.3)
    
    # 子图10: 综合评分趋势
    plt.subplot(3, 4, 10)
    if '综合评分' in summary_df.columns:
        plt.plot(range(len(summary_df)), summary_df['综合评分'], 'mo-', alpha=0.7)
        plt.title('综合评分趋势')
        plt.xlabel('时间段序号')
        plt.ylabel('综合评分')
        plt.grid(True, alpha=0.3)
    
    # 子图11: 数据量分布
    plt.subplot(3, 4, 11)
    plt.bar(range(len(summary_df)), summary_df['数据量'], alpha=0.7)
    plt.title('各时间段数据量')
    plt.xlabel('时间段序号')
    plt.ylabel('数据条数')
    plt.grid(True, alpha=0.3)
    
    # 子图12: 性能热力图
    plt.subplot(3, 4, 12)
    if line_columns:
        heatmap_data = summary_df[line_columns].T
        im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, label='延时(秒)')
        plt.yticks(range(len(line_columns)), [col.replace('_总延时', '') for col in line_columns])
        plt.xlabel('时间段序号')
        plt.title('生产线性能热力图')
    
    plt.tight_layout()
    plt.savefig('时间段并行流程分析_互相关性_分段_1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n时间段分析可视化已保存：时间段并行流程分析_互相关性_分段_1.png")

def create_process_summary_correlation(all_period_data):
    """创建流程汇总统计（基于互相关性算法）"""
    
    print(f"\n=== 流程汇总统计分析 (互相关性算法) ===")
    
    if not all_period_data:
        print("没有数据进行流程汇总分析")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_period_data)
    
    # 按流程环节分组统计
    process_summary = []
    
    for process_name in df['起始点位'].unique():
        process_data = df[df['起始点位'] == process_name]
        
        if len(process_data) > 0:
            summary_row = {
                '流程环节': process_name,
                '起始列名': process_data['起始列名'].iloc[0],
                '目标列名': process_data['目标列名'].iloc[0],
                '平均延时(秒)': process_data['平均延时(秒)'].mean(),
                '中位延时(秒)': process_data['中位延时(秒)'].mean(),
                '最小延时(秒)': process_data['最小延时(秒)'].min(),
                '最大延时(秒)': process_data['最大延时(秒)'].max(),
                '标准差(秒)': process_data['标准差(秒)'].mean(),
                '总样本数': process_data['样本数'].sum(),
                '涉及时间段数': process_data['时间段ID'].nunique(),
                '平均相关系数': process_data['最大相关系数'].mean(),
                '平均置信度': process_data['相关性置信度'].mean(),
                '最高相关系数': process_data['最大相关系数'].max(),
                '变异系数(%)': (process_data['标准差(秒)'].mean() / process_data['平均延时(秒)'].mean() * 100) if process_data['平均延时(秒)'].mean() > 0 else 0
            }
            
            process_summary.append(summary_row)
    
    if process_summary:
        summary_df = pd.DataFrame(process_summary)
        summary_df = summary_df.sort_values('平均置信度', ascending=False)
        summary_df.to_csv('流程汇总统计_分段_2.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
        print(f"\n流程汇总统计已保存：流程汇总统计_分段_2.csv")
        
        # 显示汇总结果
        print(f"\n=== 流程环节汇总 (按置信度排序) ===")
        for _, row in summary_df.head(10).iterrows():
            print(f"{row['流程环节']}:")
            print(f"  平均延时: {row['平均延时(秒)']:.2f}秒")
            print(f"  置信度: {row['平均置信度']:.4f}")
            print(f"  相关系数: {row['平均相关系数']:.4f}")
            print(f"  样本数: {row['总样本数']}")
        
        # 识别高可信度的流程环节
        high_confidence_processes = summary_df[summary_df['平均置信度'] >= 0.6]
        print(f"\n=== 高置信度流程环节 (置信度≥0.6) ===")
        print(f"共{len(high_confidence_processes)}个流程环节达到高置信度标准:")
        for _, row in high_confidence_processes.iterrows():
            print(f"  {row['流程环节']}: 延时{row['平均延时(秒)']:.2f}秒, 置信度{row['平均置信度']:.4f}")
        
        print(f"\n=== 算法性能对比建议 ===")
        print(f"互相关性算法特点:")
        print(f"  ✓ 能够处理噪声信号")
        print(f"  ✓ 提供置信度评估")
        print(f"  ✓ 对信号同步性要求较低")
        print(f"  ✓ 适合连续信号分析")
        print(f"  ○ 需要较多数据点")
        print(f"  ○ 计算复杂度较高")

if __name__ == "__main__":
    analyze_by_time_periods_correlation()