import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def moving_average_smooth(data, columns_to_smooth, window_size=10, min_periods=None):
    """
    移动平均平滑处理 - 减少短期波动
    
    参数:
    - data: DataFrame, 需要平滑的数据
    - columns_to_smooth: list, 需要平滑的列名列表
    - window_size: int, 移动窗口大小，默认5
    - min_periods: int, 计算所需的最小观测数，默认为None（使用window_size）
    
    返回:
    - DataFrame, 平滑后的数据副本
    
    使用示例:
    smoothed_data = moving_average_smooth(summary_data, 
                                        ['折叠机实际速度', '裁切机实际速度'], 
                                        window_size=10)
    """
    
    print(f"执行移动平均平滑处理...")
    print(f"窗口大小: {window_size}")
    print(f"处理列数: {len(columns_to_smooth)}")
    
    # 创建数据副本
    smoothed_data = data.copy()
    
    # 对指定列进行移动平均平滑
    for col in columns_to_smooth:
        if col in smoothed_data.columns:
            # 保存原始列（用于对比）
            smoothed_data[f'{col}_原始'] = smoothed_data[col].copy()
            
            # 执行移动平均
            smoothed_data[col] = smoothed_data[col].rolling(
                window=window_size, 
                min_periods=min_periods if min_periods else max(1, window_size//2),
                center=True  # 居中窗口，减少延迟
            ).mean()
            
            # 处理边界值（前后几个点用原始值填充）
            mask = smoothed_data[col].isna()
            smoothed_data.loc[mask, col] = smoothed_data.loc[mask, f'{col}_原始']
            
            print(f"  ✅ 已平滑列: {col}")
        else:
            print(f"  ⚠️  列不存在: {col}")
    
    print(f"移动平均平滑完成")
    return smoothed_data


def exponential_smooth(data, columns_to_smooth, alpha=0.3, adjust_outliers=True, outlier_threshold=3):
    """
    指数平滑处理 - 保留趋势信息
    
    参数:
    - data: DataFrame, 需要平滑的数据
    - columns_to_smooth: list, 需要平滑的列名列表
    - alpha: float, 平滑参数(0-1)，越大越接近原始数据，默认0.3
    - adjust_outliers: bool, 是否先处理异常值，默认True
    - outlier_threshold: float, 异常值检测阈值（标准差倍数），默认3
    
    返回:
    - DataFrame, 平滑后的数据副本
    
    使用示例:
    smoothed_data = exponential_smooth(summary_data, 
                                     ['折叠机实际速度', '裁切机实际速度'], 
                                     alpha=0.2)
    """
    
    print(f"执行指数平滑处理...")
    print(f"平滑参数alpha: {alpha}")
    print(f"处理列数: {len(columns_to_smooth)}")
    print(f"异常值处理: {'开启' if adjust_outliers else '关闭'}")
    
    # 创建数据副本
    smoothed_data = data.copy()
    
    # 对指定列进行指数平滑
    for col in columns_to_smooth:
        if col in smoothed_data.columns:
            # 保存原始列（用于对比）
            smoothed_data[f'{col}_原始'] = smoothed_data[col].copy()
            
            # 获取原始数据
            original_series = smoothed_data[col].copy()
            
            # 异常值处理（可选）
            if adjust_outliers:
                # 计算统计指标
                mean_val = original_series.mean()
                std_val = original_series.std()
                
                # 识别异常值
                outlier_mask = np.abs((original_series - mean_val) / std_val) > outlier_threshold
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    print(f"  发现异常值: {outlier_count}个 (列: {col})")
                    
                    # 用中位数替换异常值
                    median_val = original_series.median()
                    original_series.loc[outlier_mask] = median_val
            
            # 执行指数平滑
            smoothed_series = original_series.ewm(alpha=alpha, adjust=False).mean()
            
            # 更新数据
            smoothed_data[col] = smoothed_series
            
            # 计算平滑效果统计
            original_std = smoothed_data[f'{col}_原始'].std()
            smoothed_std = smoothed_series.std()
            noise_reduction = (1 - smoothed_std/original_std) * 100 if original_std > 0 else 0
            
            print(f"  ✅ 已平滑列: {col} (噪声减少: {noise_reduction:.1f}%)")
        else:
            print(f"  ⚠️  列不存在: {col}")
    
    print(f"指数平滑完成")
    return smoothed_data


def compare_smoothing_methods(data, columns_to_compare, window_size=10, alpha=0.3):
    """
    对比两种平滑方法的效果
    
    参数:
    - data: DataFrame, 原始数据
    - columns_to_compare: list, 需要对比的列名列表
    - window_size: int, 移动平均窗口大小
    - alpha: float, 指数平滑参数
    
    返回:
    - dict, 包含对比结果的字典
    """
    
    print(f"\n{'='*50}")
    print(f"平滑方法效果对比")
    print(f"{'='*50}")
    
    # 执行两种平滑
    ma_data = moving_average_smooth(data, columns_to_compare, window_size)
    exp_data = exponential_smooth(data, columns_to_compare, alpha)
    
    comparison_results = {}
    
    for col in columns_to_compare:
        if col in data.columns:
            original = data[col].dropna()
            ma_smoothed = ma_data[col].dropna()
            exp_smoothed = exp_data[col].dropna()
            
            # 计算统计指标
            original_std = original.std()
            ma_std = ma_smoothed.std()
            exp_std = exp_smoothed.std()
            
            ma_noise_reduction = (1 - ma_std/original_std) * 100 if original_std > 0 else 0
            exp_noise_reduction = (1 - exp_std/original_std) * 100 if original_std > 0 else 0
            
            # 计算与原始数据的相关性（保留趋势能力）
            ma_correlation = np.corrcoef(original[:len(ma_smoothed)], ma_smoothed)[0,1]
            exp_correlation = np.corrcoef(original[:len(exp_smoothed)], exp_smoothed)[0,1]
            
            comparison_results[col] = {
                '原始标准差': original_std,
                '移动平均标准差': ma_std,
                '指数平滑标准差': exp_std,
                '移动平均噪声减少(%)': ma_noise_reduction,
                '指数平滑噪声减少(%)': exp_noise_reduction,
                '移动平均相关性': ma_correlation,
                '指数平滑相关性': exp_correlation
            }
            
            print(f"\n列: {col}")
            print(f"  移动平均: 噪声减少 {ma_noise_reduction:.1f}%, 相关性 {ma_correlation:.4f}")
            print(f"  指数平滑: 噪声减少 {exp_noise_reduction:.1f}%, 相关性 {exp_correlation:.4f}")
            
            # 给出推荐
            if ma_noise_reduction > exp_noise_reduction and ma_correlation > 0.9:
                print(f"  推荐: 移动平均 (更好的噪声减少)")
            elif exp_correlation > ma_correlation and exp_noise_reduction > ma_noise_reduction * 0.8:
                print(f"  推荐: 指数平滑 (更好的趋势保持)")
            else:
                print(f"  推荐: 根据具体需求选择")
    
    return comparison_results

def analyze_by_time_periods():
    """按照调整后.csv的时间段进行并行流程分析"""
    
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
        
        # 数据平滑处理 - 二选一使用
        # 方法1：移动平均平滑（减少短期波动）
        summary_data = moving_average_smooth(summary_data, 
                                           ['折叠机入包数', '折叠机出包数','外循环进内循环纸条数量','进第一裁切通道纸条计数',
                                            '进第二裁切通道纸条计数','进第三裁切通道纸条计数','进第四裁切通道纸条计数',
                                            '1#有效切数','2#有效切数','3#有效切数','4#有效切数','1#小包机入包数',
                                            '2#小包机入包数','3#小包机入包数','4#小包机入包数'], 
                                           window_size=10)
        
        # 方法2：指数平滑（保留趋势信息）
        
        # summary_data = exponential_smooth(summary_data, 
        #                                 ['折叠机入包数', '折叠机出包数','外循环进内循环纸条数量','进第一裁切通道纸条计数',
        #                                     '进第二裁切通道纸条计数','进第三裁切通道纸条计数','进第四裁切通道纸条计数',
        #                                     '1#有效切数','2#有效切数','3#有效切数','4#有效切数','1#小包机入包数',
        #                                     '2#小包机入包数','3#小包机入包数','4#小包机入包数'], 
        #                                 alpha=0.3)
        
    except Exception as e:
        print(f"读取汇总数据出错：{e}")
        return
    
    # 定义并行流程结构
    process_structure = define_process_structure()
    
    # 对每个时间段进行分析
    period_results = []
    all_period_data = []
    
    print(f"\n=== 开始分析 {len(time_periods)} 个时间段 ===")
    
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
        
        if len(period_data) < 10:  # 数据量太少
            print(f"{period_id} 数据量不足 ({len(period_data)}条)，跳过分析")
            continue
        
        print(f"该时间段数据量: {len(period_data)}条")
        
        # 分析该时间段的流程延时
        period_result = analyze_single_period(period_data, period_id, start_time, end_time, process_structure)
        
        if period_result:
            period_results.append(period_result)
            all_period_data.extend(period_result['详细数据'])
    
    if period_results:
        # 汇总分析所有时间段
        comprehensive_period_analysis(period_results, all_period_data)
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

def analyze_single_period(data, period_id, start_time, end_time, process_structure):
    """分析单个时间段的流程延时"""
    
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
    common_results = analyze_sequence(data, common_sequence, f"{period_id}_共同流程")
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
        line_results = analyze_sequence(data, full_sequence, f"{period_id}_{line_name}")
        
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
            avg_stability = np.mean([r['平均延时(秒)']/r['标准差(秒)'] if r['标准差(秒)'] > 0 else 0 for r in line_results])
            
            line_performance[line_name] = {
                '环节数': len(line_results),
                '总平均延时(秒)': total_avg_delay,
                '总中位延时(秒)': total_median_delay,
                '平均单环节延时(秒)': total_avg_delay / len(line_results),
                '平均稳定性': avg_stability,
                '最大单环节延时(秒)': max([r['平均延时(秒)'] for r in line_results])
            }
    
    period_result['生产线性能'] = line_performance
    
    # 显示该时间段的关键指标
    if line_performance:
        print(f"\n{period_id} 生产线性能:")
        for line_name, performance in line_performance.items():
            print(f"  {line_name}: 总延时 {performance['总平均延时(秒)']:.1f}秒")
        
        # 找出最优和最差生产线
        best_line = min(line_performance.items(), key=lambda x: x[1]['总平均延时(秒)'])
        worst_line = max(line_performance.items(), key=lambda x: x[1]['总平均延时(秒)'])
        
        print(f"  最优: {best_line[0]} ({best_line[1]['总平均延时(秒)']:.1f}秒)")
        print(f"  最差: {worst_line[0]} ({worst_line[1]['总平均延时(秒)']:.1f}秒)")
        print(f"  差异: {worst_line[1]['总平均延时(秒)'] - best_line[1]['总平均延时(秒)']:.1f}秒")
    
    return period_result

def analyze_sequence(data, sequence, sequence_name):
    """分析单个序列的传输延时"""
    
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
        
        # 找出数据变化点
        current_changes = find_change_points(data, current_col)
        next_changes = find_change_points(data, next_col)
        
        if len(current_changes) > 0 and len(next_changes) > 0:
            # 计算传输延时
            delays = calculate_nearest_delays(current_changes, next_changes)
            
            if len(delays) > 0:
                delay_stats = {
                    '生产线': sequence_name,
                    '起始点位': current_name,
                    '目标点位': next_name,
                    '起始列名': current_col,
                    '目标列名': next_col,
                    '平均延时(秒)': np.mean(delays),
                    '中位延时(秒)': np.median(delays),
                    '最小延时(秒)': np.min(delays),
                    '最大延时(秒)': np.max(delays),
                    '标准差(秒)': np.std(delays),
                    '样本数': len(delays)
                }
                
                delays_results.append(delay_stats)
    
    return delays_results

def comprehensive_period_analysis(period_results, all_period_data):
    """对所有时间段进行综合分析"""
    
    print(f"\n{'='*60}")
    print(f"=== 时间段综合分析 (共{len(period_results)}个时间段) ===")
    print(f"{'='*60}")
    
    # 保存所有详细数据
    if all_period_data:
        all_df = pd.DataFrame(all_period_data)
        all_df.to_csv('时间段并行流程分析_详细数据_分段_1.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
        print(f"所有时间段详细数据已保存：时间段并行流程分析_详细数据_分段_1.csv")
    
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
            
            # 添加各生产线的总延时
            for line_name, performance in line_performance.items():
                summary_row[f'{line_name}_总延时'] = performance['总平均延时(秒)']
                summary_row[f'{line_name}_稳定性'] = performance['平均稳定性']
            
            # 计算该时间段的整体指标
            delays = [perf['总平均延时(秒)'] for perf in line_performance.values()]
            summary_row['平均延时'] = np.mean(delays)
            summary_row['最大延时'] = np.max(delays)
            summary_row['最小延时'] = np.min(delays)
            summary_row['延时差异'] = np.max(delays) - np.min(delays)
            
            period_summary.append(summary_row)
    
    if period_summary:
        summary_df = pd.DataFrame(period_summary)
        summary_df.to_csv('时间段性能汇总_分段_1.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
        print(f"时间段性能汇总已保存：时间段性能汇总_分段_1.csv")
        
        # 分析时间段性能趋势
        analyze_period_trends(summary_df)
        
        # 识别最优和最差时间段
        identify_best_worst_periods(summary_df, period_results)
        
        # 创建可视化
        create_period_visualization(summary_df, period_results)
        
        # 创建流程汇总统计
        create_process_summary(all_period_data)
        
        # 创建延时拟合方程分析
        create_delay_fitting_equations(all_period_data)

def analyze_period_trends(summary_df):
    """分析时间段性能趋势"""
    
    print(f"\n=== 时间段性能趋势分析 ===")
    
    # 整体性能统计
    print(f"平均延时统计:")
    print(f"  整体平均: {summary_df['平均延时'].mean():.2f}秒")
    print(f"  最好时段: {summary_df['平均延时'].min():.2f}秒")
    print(f"  最差时段: {summary_df['平均延时'].max():.2f}秒")
    print(f"  标准差: {summary_df['平均延时'].std():.2f}秒")
    
    # 生产线稳定性分析
    line_columns = [col for col in summary_df.columns if '_总延时' in col]
    
    print(f"\n各生产线在不同时间段的表现:")
    for col in line_columns:
        line_name = col.replace('_总延时', '')
        line_data = summary_df[col].dropna()
        if len(line_data) > 0:
            print(f"  {line_name}:")
            print(f"    平均延时: {line_data.mean():.2f}秒")
            print(f"    变异系数: {(line_data.std()/line_data.mean()*100):.1f}%")
            print(f"    最佳表现: {line_data.min():.2f}秒")
            print(f"    最差表现: {line_data.max():.2f}秒")

def identify_best_worst_periods(summary_df, period_results):
    """识别最优和最差时间段"""
    
    print(f"\n=== 最优/最差时间段识别 ===")
    
    # 按平均延时排序
    best_period_idx = summary_df['平均延时'].idxmin()
    worst_period_idx = summary_df['平均延时'].idxmax()
    
    best_period = summary_df.iloc[best_period_idx]
    worst_period = summary_df.iloc[worst_period_idx]
    
    print(f"最优时间段: {best_period['时间段ID']}")
    print(f"  开始时间: {best_period['开始时间']}")
    print(f"  平均延时: {best_period['平均延时']:.2f}秒")
    print(f"  延时差异: {best_period['延时差异']:.2f}秒")
    
    print(f"\n最差时间段: {worst_period['时间段ID']}")
    print(f"  开始时间: {worst_period['开始时间']}")
    print(f"  平均延时: {worst_period['平均延时']:.2f}秒")
    print(f"  延时差异: {worst_period['延时差异']:.2f}秒")
    
    # 分析性能差异原因
    performance_gap = worst_period['平均延时'] - best_period['平均延时']
    print(f"\n性能差异: {performance_gap:.2f}秒 ({performance_gap/best_period['平均延时']*100:.1f}%)")
    
    # 找出主要差异来源
    line_columns = [col for col in summary_df.columns if '_总延时' in col]
    print(f"\n主要差异来源:")
    for col in line_columns:
        line_name = col.replace('_总延时', '')
        if not pd.isna(best_period[col]) and not pd.isna(worst_period[col]):
            diff = worst_period[col] - best_period[col]
            print(f"  {line_name}: +{diff:.2f}秒")

def create_period_visualization(summary_df, period_results):
    """创建时间段分析可视化"""
    
    plt.figure(figsize=(20, 15))
    
    # 子图1: 时间段平均延时趋势
    plt.subplot(3, 3, 1)
    plt.plot(range(len(summary_df)), summary_df['平均延时'], 'bo-', alpha=0.7)
    plt.title('各时间段平均延时趋势')
    plt.xlabel('时间段序号')
    plt.ylabel('平均延时(秒)')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 各生产线延时对比箱线图
    plt.subplot(3, 3, 2)
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
    
    # 子图3: 延时差异趋势
    plt.subplot(3, 3, 3)
    plt.plot(range(len(summary_df)), summary_df['延时差异'], 'ro-', alpha=0.7)
    plt.title('各时间段生产线差异')
    plt.xlabel('时间段序号')
    plt.ylabel('最大-最小延时(秒)')
    plt.grid(True, alpha=0.3)
    
    # 子图4-6: 各生产线随时间变化
    for i, col in enumerate(line_columns[:3]):
        plt.subplot(3, 3, 4+i)
        line_name = col.replace('_总延时', '')
        plt.plot(range(len(summary_df)), summary_df[col], 'o-', alpha=0.7)
        plt.title(f'{line_name}延时变化')
        plt.xlabel('时间段序号')
        plt.ylabel('总延时(秒)')
        plt.grid(True, alpha=0.3)
    
    # 子图7: 数据量分布
    plt.subplot(3, 3, 7)
    plt.bar(range(len(summary_df)), summary_df['数据量'], alpha=0.7)
    plt.title('各时间段数据量')
    plt.xlabel('时间段序号')
    plt.ylabel('数据条数')
    plt.grid(True, alpha=0.3)
    
    # 子图8: 性能热力图
    plt.subplot(3, 3, 8)
    if line_columns:
        heatmap_data = summary_df[line_columns].T
        plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='延时(秒)')
        plt.yticks(range(len(line_columns)), [col.replace('_总延时', '') for col in line_columns])
        plt.xlabel('时间段序号')
        plt.title('生产线性能热力图')
    
    # 子图9: 稳定性分析
    plt.subplot(3, 3, 9)
    stability_columns = [col for col in summary_df.columns if '_稳定性' in col]
    if stability_columns:
        for col in stability_columns:
            line_name = col.replace('_稳定性', '')
            plt.plot(range(len(summary_df)), summary_df[col], 'o-', alpha=0.7, label=line_name)
        plt.title('各生产线稳定性变化')
        plt.xlabel('时间段序号')
        plt.ylabel('稳定性指数')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('时间段并行流程分析_分段_1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n时间段分析可视化已保存：时间段并行流程分析_分段_1.png")

def find_change_points(data, column):
    """找出数据变化点的时间"""
    if column not in data.columns:
        return np.array([])
    
    changes = data[column].diff().fillna(0)
    change_mask = (changes != 0) & (~changes.isna())
    change_times = data.loc[change_mask, '时间'].values
    return change_times

def calculate_nearest_delays(source_times, target_times):
    """计算最近邻延时"""
    delays = []
    
    for source_time in source_times:
        # 找出在源时间之后最近的目标时间
        future_targets = target_times[target_times > source_time]
        if len(future_targets) > 0:
            nearest_target = future_targets[0]
            delay_seconds = (pd.to_datetime(nearest_target) - pd.to_datetime(source_time)).total_seconds()
            if 0 <= delay_seconds <= 1800:  # 限制在30分钟内的合理延时
                delays.append(delay_seconds)
    
    return delays

def create_process_summary(all_period_data):
    """创建流程汇总统计"""
    
    print(f"\n=== 流程汇总统计分析 ===")
    
    if not all_period_data:
        print("没有数据进行流程汇总分析")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_period_data)
    
    # 定义流程分类和起始位置
    process_categories = {
        '共同前置流程': {
            '1_折叠机入包数': {'起始位置': '折叠机入包数', '流程阶段': '前置-折叠'},
            '2_折叠机出包数': {'起始位置': '折叠机出包数', '流程阶段': '前置-折叠'},
        },
        '外循环分流连接': {
            '3_外循环→第一裁切通道': {'起始位置': '外循环进内循环纸条数量', '流程阶段': '分流-连接'},
            '3_外循环→第二裁切通道': {'起始位置': '外循环进内循环纸条数量', '流程阶段': '分流-连接'},
            '3_外循环→第三裁切通道': {'起始位置': '外循环进内循环纸条数量', '流程阶段': '分流-连接'},
            '3_外循环→第四裁切通道': {'起始位置': '外循环进内循环纸条数量', '流程阶段': '分流-连接'}
        },
        '生产线1流程': {
            '4_第一裁切通道': {'起始位置': '进第一裁切通道纸条计数', '流程阶段': '生产线1-裁切'},
            '5_1号有效切数': {'起始位置': '1#有效切数', '流程阶段': '生产线1-裁切'},
            '6_1号小包机入包数': {'起始位置': '1#小包机入包数', '流程阶段': '生产线1-包装'},
        },
        '生产线2流程': {
            '4_第二裁切通道': {'起始位置': '进第二裁切通道纸条计数', '流程阶段': '生产线2-裁切'},
            '5_2号有效切数': {'起始位置': '2#有效切数', '流程阶段': '生产线2-裁切'},
            '6_2号小包机入包数': {'起始位置': '2#小包机入包数', '流程阶段': '生产线2-包装'},
        },
        '生产线3流程': {
            '4_第三裁切通道': {'起始位置': '进第三裁切通道纸条计数', '流程阶段': '生产线3-裁切'},
            '5_3号有效切数': {'起始位置': '3#有效切数', '流程阶段': '生产线3-裁切'},
            '6_3号小包机入包数': {'起始位置': '3#小包机入包数', '流程阶段': '生产线3-包装'},

        },
        '生产线4流程': {
            '4_第四裁切通道': {'起始位置': '进第四裁切通道纸条计数', '流程阶段': '生产线4-裁切'},
            '5_4号有效切数': {'起始位置': '4#有效切数', '流程阶段': '生产线4-裁切'},
            '6_4号小包机入包数': {'起始位置': '4#小包机入包数', '流程阶段': '生产线4-包装'},

        }
    }
    
    # 创建汇总统计
    summary_data = []
    
    for category_name, processes in process_categories.items():
        print(f"\n--- {category_name} ---")
        
        for process_key, process_info in processes.items():
            # 特殊处理外循环分流连接
            if category_name == '外循环分流连接':
                # 根据process_key确定目标列名
                target_mapping = {
                    '3_外循环→第一裁切通道': '进第一裁切通道纸条计数',
                    '3_外循环→第二裁切通道': '进第二裁切通道纸条计数',
                    '3_外循环→第三裁切通道': '进第三裁切通道纸条计数',
                    '3_外循环→第四裁切通道': '进第四裁切通道纸条计数'
                }
                target_col = target_mapping.get(process_key, '')
                
                # 筛选从外循环进内循环到对应裁切通道的数据
                process_data = df[
                    (df['起始列名'] == '外循环进内循环纸条数量') & 
                    (df['目标列名'] == target_col)
                ]
            else:
                # 筛选该流程的数据
                process_data = df[df['起始点位'] == process_key]
            
            if len(process_data) > 0:
                # 计算统计指标
                avg_delay = process_data['平均延时(秒)'].mean()
                median_delay = process_data['中位延时(秒)'].mean()
                min_delay = process_data['最小延时(秒)'].min()
                max_delay = process_data['最大延时(秒)'].max()
                std_delay = process_data['标准差(秒)'].mean()
                total_samples = process_data['样本数'].sum()
                time_periods_count = process_data['时间段ID'].nunique()
                
                # 时间范围
                time_range = f"[{min_delay:.2f},{max_delay:.2f}]"
                
                # 确定目标位置
                if category_name == '外循环分流连接':
                    target_pos = target_mapping.get(process_key, '')
                else:
                    target_pos = process_data['目标列名'].iloc[0] if len(process_data) > 0 else ''
                
                summary_row = {
                    '流程类别': category_name,
                    '流程环节': process_key,
                    # '流程阶段': process_info['流程阶段'],
                    '起始位置': process_info['起始位置'],
                    '目标位置': target_pos,
                    '平均延时(秒)': avg_delay,
                    '中位延时(秒)': median_delay,
                    '时间范围': time_range,
                    '最小延时(秒)': min_delay,
                    '最大延时(秒)': max_delay,
                    '标准差(秒)': std_delay,
                    '涉及时间段数': time_periods_count,
                    '总样本数': total_samples,
                    '变异系数(%)': (std_delay / avg_delay * 100) if avg_delay > 0 else 0
                }
                
                summary_data.append(summary_row)
                
                print(f"  {process_key}: 平均{avg_delay:.2f}秒 (范围: {time_range})")
            else:
                print(f"  {process_key}: 无数据")
    
    # 保存汇总统计
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('流程汇总统计_分段_1.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
        print(f"\n流程汇总统计已保存：流程汇总统计_分段_1.csv")
        
        # 按流程类别分组统计
        category_summary = []
        for category in summary_df['流程类别'].unique():
            category_data = summary_df[summary_df['流程类别'] == category]
            
            category_row = {
                '流程类别': category,
                '环节数量': len(category_data),
                '总平均延时(秒)': category_data['平均延时(秒)'].sum(),
                '平均单环节延时(秒)': category_data['平均延时(秒)'].mean(),
                '最快环节延时(秒)': category_data['平均延时(秒)'].min(),
                '最慢环节延时(秒)': category_data['平均延时(秒)'].max(),
                '类别内差异(秒)': category_data['平均延时(秒)'].max() - category_data['平均延时(秒)'].min(),
                '总样本数': category_data['总样本数'].sum(),
                '平均变异系数(%)': category_data['变异系数(%)'].mean()
            }
            category_summary.append(category_row)
        
        # 保存流程类别汇总
        if category_summary:
            category_df = pd.DataFrame(category_summary)
            category_df.to_csv('流程类别汇总_分段_1.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
            print(f"流程类别汇总已保存：流程类别汇总_分段_1.csv")
            
            # 显示类别汇总结果
            print(f"\n=== 流程类别汇总 ===")
            for _, row in category_df.iterrows():
                print(f"{row['流程类别']}:")
                print(f"  环节数量: {row['环节数量']}")
                print(f"  总延时: {row['总平均延时(秒)']:.2f}秒")
                print(f"  平均单环节: {row['平均单环节延时(秒)']:.2f}秒")
                print(f"  类别内差异: {row['类别内差异(秒)']:.2f}秒")
                print(f"  变异系数: {row['平均变异系数(%)']:.1f}%")
        
        # 创建流程对比可视化
        create_process_comparison_chart(summary_df, category_df)
        
        # 创建完整流程路径时间总结
        create_complete_flow_summary(summary_df, category_df)
    else:
        print("没有生成汇总统计数据")

def create_process_comparison_chart(summary_df, category_df):
    """创建流程对比图表"""
    
    plt.figure(figsize=(20, 12))
    
    # 子图1: 各流程环节延时对比
    plt.subplot(2, 3, 1)
    categories = summary_df['流程类别'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, category in enumerate(categories):
        cat_data = summary_df[summary_df['流程类别'] == category]
        plt.bar(range(len(cat_data)), cat_data['平均延时(秒)'], 
                alpha=0.7, label=category, color=colors[i % len(colors)])
    
    plt.title('各流程环节平均延时对比')
    plt.ylabel('平均延时(秒)')
    plt.xlabel('流程环节')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 子图2: 流程类别总延时对比
    plt.subplot(2, 3, 2)
    plt.bar(category_df['流程类别'], category_df['总平均延时(秒)'], 
            alpha=0.7, color=colors[:len(category_df)])
    plt.title('各流程类别总延时对比')
    plt.ylabel('总延时(秒)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 子图3: 变异系数对比
    plt.subplot(2, 3, 3)
    plt.bar(category_df['流程类别'], category_df['平均变异系数(%)'], 
            alpha=0.7, color=colors[:len(category_df)])
    plt.title('各流程类别稳定性对比')
    plt.ylabel('平均变异系数(%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 子图4: 环节数量对比
    plt.subplot(2, 3, 4)
    plt.bar(category_df['流程类别'], category_df['环节数量'], 
            alpha=0.7, color=colors[:len(category_df)])
    plt.title('各流程类别环节数量')
    plt.ylabel('环节数量')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 子图5: 类别内差异对比
    plt.subplot(2, 3, 5)
    plt.bar(category_df['流程类别'], category_df['类别内差异(秒)'], 
            alpha=0.7, color=colors[:len(category_df)])
    plt.title('各流程类别内部差异')
    plt.ylabel('类别内差异(秒)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 子图6: 样本数量对比
    plt.subplot(2, 3, 6)
    plt.bar(category_df['流程类别'], category_df['总样本数'], 
            alpha=0.7, color=colors[:len(category_df)])
    plt.title('各流程类别样本数量')
    plt.ylabel('总样本数')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('流程汇总对比_分段_1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"流程汇总对比图表已保存：流程汇总对比_分段_1.png")

def create_complete_flow_summary(summary_df, category_df):
    """创建完整流程路径时间总结"""
    
    print(f"\n{'='*60}")
    print(f"=== 完整流程路径时间总结 ===")
    print(f"{'='*60}")
    
    # 安全获取各类别的总延时，如果不存在则使用默认值0
    def safe_get_category_time(category_name):
        category_data = category_df[category_df['流程类别'] == category_name]['总平均延时(秒)']
        if len(category_data) > 0:
            return category_data.iloc[0]
        else:
            print(f"  ⚠️  未找到 {category_name} 的数据，使用默认值0")
            return 0.0
    
    common_time = safe_get_category_time('共同前置流程')
    distribution_time = safe_get_category_time('外循环分流连接')
    
    line1_time = safe_get_category_time('生产线1流程')
    line2_time = safe_get_category_time('生产线2流程')
    line3_time = safe_get_category_time('生产线3流程')
    line4_time = safe_get_category_time('生产线4流程')
    
    # 安全获取各分流连接的延时
    flow_dist_df = summary_df[summary_df['流程类别'] == '外循环分流连接']
    
    def safe_get_distribution_time(flow_name):
        flow_data = flow_dist_df[flow_dist_df['流程环节'] == flow_name]['平均延时(秒)']
        if len(flow_data) > 0:
            return flow_data.iloc[0]
        else:
            print(f"  ⚠️  未找到 {flow_name} 的数据，使用默认值0")
            return 0.0
    
    dist_to_line1 = safe_get_distribution_time('3_外循环→第一裁切通道')
    dist_to_line2 = safe_get_distribution_time('3_外循环→第二裁切通道')
    dist_to_line3 = safe_get_distribution_time('3_外循环→第三裁切通道')
    dist_to_line4 = safe_get_distribution_time('3_外循环→第四裁切通道')
    
    # 计算完整流程路径时间
    complete_flows = {
        '生产线1完整流程': {
            '共同前置流程': common_time,
            '外循环→第一裁切通道': dist_to_line1,
            '生产线1流程': line1_time,
            '总时间': common_time + dist_to_line1 + line1_time
        },
        '生产线2完整流程': {
            '共同前置流程': common_time,
            '外循环→第二裁切通道': dist_to_line2,
            '生产线2流程': line2_time,
            '总时间': common_time + dist_to_line2 + line2_time
        },
        '生产线3完整流程': {
            '共同前置流程': common_time,
            '外循环→第三裁切通道': dist_to_line3,
            '生产线3流程': line3_time,
            '总时间': common_time + dist_to_line3 + line3_time
        },
        '生产线4完整流程': {
            '共同前置流程': common_time,
            '外循环→第四裁切通道': dist_to_line4,
            '生产线4流程': line4_time,
            '总时间': common_time + dist_to_line4 + line4_time
        }
    }
    
    # 显示完整流程路径分析
    print(f"\n完整流程路径分析:")
    print(f"{'='*50}")
    
    for flow_name, flow_data in complete_flows.items():
        print(f"\n📍 {flow_name}:")
        print(f"  共同前置流程: {flow_data['共同前置流程']:.2f}秒")
        
        # 找到分流连接的键名
        dist_keys = [k for k in flow_data.keys() if '外循环→' in k]
        if dist_keys:
            dist_key = dist_keys[0]
            print(f"  {dist_key}: {flow_data[dist_key]:.2f}秒")
        else:
            print(f"  外循环分流连接: 0.00秒")
            dist_key = None
        
        # 找到生产线流程的键名
        line_keys = [k for k in flow_data.keys() if '生产线' in k and '流程' in k]
        if line_keys:
            line_key = line_keys[0]
            print(f"  {line_key}: {flow_data[line_key]:.2f}秒")
        else:
            print(f"  生产线流程: 0.00秒")
            line_key = None
        
        print(f"  ➤ 总延时: {flow_data['总时间']:.2f}秒")
        
        # 计算各阶段占比（避免除0错误）
        if flow_data['总时间'] > 0:
            common_pct = (flow_data['共同前置流程'] / flow_data['总时间']) * 100
            dist_pct = (flow_data[dist_key] / flow_data['总时间']) * 100 if dist_key else 0
            line_pct = (flow_data[line_key] / flow_data['总时间']) * 100 if line_key else 0
            
            print(f"    - 前置流程占比: {common_pct:.1f}%")
            print(f"    - 分流连接占比: {dist_pct:.1f}%")
            print(f"    - 生产线流程占比: {line_pct:.1f}%")
        else:
            print(f"    - 总时间为0，无法计算占比")
    
    # 生产线性能对比
    print(f"\n{'='*50}")
    print(f"生产线性能排名:")
    print(f"{'='*50}")
    
    # 按总时间排序
    sorted_flows = sorted(complete_flows.items(), key=lambda x: x[1]['总时间'])
    
    if len(sorted_flows) > 0:
        for i, (flow_name, flow_data) in enumerate(sorted_flows, 1):
            status = "🟢 最优" if i == 1 else "🔴 最差" if i == len(sorted_flows) else f"🟡 第{i}名"
            print(f"{status} {flow_name}: {flow_data['总时间']:.2f}秒")
        
        # 性能差异分析
        best_time = sorted_flows[0][1]['总时间']
        worst_time = sorted_flows[-1][1]['总时间']
        performance_gap = worst_time - best_time
        
        print(f"\n{'='*50}")
        print(f"性能差异分析:")
        print(f"{'='*50}")
        print(f"最优生产线: {sorted_flows[0][0]} ({best_time:.2f}秒)")
        print(f"最差生产线: {sorted_flows[-1][0]} ({worst_time:.2f}秒)")
        print(f"性能差异: {performance_gap:.2f}秒")
        
        if best_time > 0:
            print(f"相对差异: {(performance_gap/best_time)*100:.1f}%")
        if worst_time > 0:
            improvement_potential = (performance_gap / worst_time) * 100
            print(f"优化潜力: {improvement_potential:.1f}%")
    else:
        print("没有生产线数据可供对比")
    
    # 瓶颈环节识别
    print(f"\n{'='*50}")
    print(f"瓶颈环节识别:")
    print(f"{'='*50}")
    
    # 找出各阶段的最大延时
    max_common = common_time
    max_dist = max(dist_to_line1, dist_to_line2, dist_to_line3, dist_to_line4)
    max_line = max(line1_time, line2_time, line3_time, line4_time)
    
    bottleneck_stage = max(
        ("共同前置流程", max_common),
        ("外循环分流连接", max_dist),
        ("生产线流程", max_line)
    )
    
    print(f"最大瓶颈阶段: {bottleneck_stage[0]} ({bottleneck_stage[1]:.2f}秒)")
    
    if bottleneck_stage[0] == "外循环分流连接":
        # 找出哪个分流连接是瓶颈
        dist_times = {
            "第一裁切通道": dist_to_line1,
            "第二裁切通道": dist_to_line2,
            "第三裁切通道": dist_to_line3,
            "第四裁切通道": dist_to_line4
        }
        worst_dist = max(dist_times.items(), key=lambda x: x[1])
        print(f"分流瓶颈: 外循环→{worst_dist[0]} ({worst_dist[1]:.2f}秒)")
    
    elif bottleneck_stage[0] == "生产线流程":
        # 找出哪个生产线是瓶颈
        line_times = {
            "生产线1": line1_time,
            "生产线2": line2_time,
            "生产线3": line3_time,
            "生产线4": line4_time
        }
        worst_line = max(line_times.items(), key=lambda x: x[1])
        print(f"生产线瓶颈: {worst_line[0]} ({worst_line[1]:.2f}秒)")
    
    # 保存完整流程路径汇总
    complete_flow_data = []
    for flow_name, flow_data in complete_flows.items():
        # 安全获取分流连接和生产线流程的值
        dist_values = [v for k, v in flow_data.items() if '外循环→' in k]
        line_values = [v for k, v in flow_data.items() if '生产线' in k and '流程' in k]
        
        dist_time = dist_values[0] if dist_values else 0.0
        line_time = line_values[0] if line_values else 0.0
        
        row = {
            '完整流程路径': flow_name,
            '共同前置流程(秒)': flow_data['共同前置流程'],
            '外循环分流连接(秒)': dist_time,
            '生产线流程(秒)': line_time,
            '总延时(秒)': flow_data['总时间']
        }
        
        # 安全计算占比（避免除0错误）
        if flow_data['总时间'] > 0:
            row['前置流程占比(%)'] = (flow_data['共同前置流程'] / flow_data['总时间']) * 100
            row['分流连接占比(%)'] = (dist_time / flow_data['总时间']) * 100
            row['生产线流程占比(%)'] = (line_time / flow_data['总时间']) * 100
        else:
            row['前置流程占比(%)'] = 0.0
            row['分流连接占比(%)'] = 0.0
            row['生产线流程占比(%)'] = 0.0
        
        complete_flow_data.append(row)
    
    # 保存到CSV
    if complete_flow_data:
        complete_flow_df = pd.DataFrame(complete_flow_data)
        complete_flow_df = complete_flow_df.sort_values('总延时(秒)')
        complete_flow_df.to_csv('完整流程路径汇总_分段_1.csv', index=False, encoding='utf-8-sig', float_format='%.2f')
        print(f"\n完整流程路径汇总已保存：完整流程路径汇总_分段_1.csv")

def create_delay_fitting_equations(all_period_data):
    """根据时间段延时数据拟合方程式"""
    
    print(f"\n{'='*60}")
    print(f"=== 延时时间拟合方程分析 ===")
    print(f"{'='*60}")
    
    if not all_period_data:
        print("没有数据进行拟合分析")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_period_data)
    
    # 读取汇总数据以获取速度信息
    try:
        summary_data = pd.read_csv('存纸架数据汇总.csv', encoding='utf-8-sig')
        summary_data['时间'] = pd.to_datetime(summary_data['时间'])
        print(f"加载汇总数据用于速度信息：{len(summary_data)}条记录")
    except Exception as e:
        print(f"读取汇总数据出错：{e}")
        return
    
    # 定义流程环节与其对应的速度列的映射
    speed_column_mapping = {
        '1_折叠机速度': '折叠机实际速度',
        '2_折叠机入包数': '折叠机实际速度',  # 使用折叠机速度作为参考
        '3_折叠机出包数': '折叠机实际速度',
        '4_存纸率': '折叠机实际速度',
        '5_外循环→第一裁切通道': '外循环进内循环纸条数量',  # 使用数量作为速度参考
        '5_外循环→第二裁切通道': '外循环进内循环纸条数量',
        '5_外循环→第三裁切通道': '外循环进内循环纸条数量',
        '5_外循环→第四裁切通道': '外循环进内循环纸条数量',
        '6_第一裁切通道': '裁切机实际速度',
        '7_裁切速度': '裁切机实际速度',
        '8_1号有效切数': '裁切机实际速度',
        '9_1号小包机入包数': '1#小包机实际速度',
        '8_2号有效切数': '裁切机实际速度',
        '9_2号小包机入包数': '2#小包机实际速度',
        '8_3号有效切数': '裁切机实际速度',
        '9_3号小包机入包数': '3#小包机主机实际速度',
        '8_4号有效切数': '裁切机实际速度',
        '9_4号小包机入包数': '4#小包机主机实际速度'
    }
    
    # 获取所有唯一的流程环节
    process_steps = df['起始点位'].unique()
    print(f"识别到 {len(process_steps)} 个流程环节需要拟合")
    
    # 存储拟合结果
    fitting_results = []
    
    for step in process_steps:
        print(f"\n--- 分析流程环节: {step} ---")
        
        # 筛选该流程环节的数据
        step_data = df[df['起始点位'] == step].copy()
        
        if len(step_data) < 5:  # 数据量太少，无法进行有效拟合
            print(f"  数据量不足 ({len(step_data)}条)，跳过拟合")
            continue
        
        # 获取对应的速度列
        speed_column = speed_column_mapping.get(step, None)
        if not speed_column:
            print(f"  未找到对应的速度列，跳过拟合")
            continue
        
        if speed_column not in summary_data.columns:
            print(f"  速度列 {speed_column} 不存在，跳过拟合")
            continue
        
        # 准备拟合数据
        fitting_data = prepare_fitting_data(step_data, summary_data, speed_column, step)
        
        if fitting_data is None or len(fitting_data) < 5:
            print(f"  准备拟合数据失败或数据量不足，跳过拟合")
            continue
        
        # 执行多种拟合方法
        fit_result = perform_multiple_fitting(fitting_data, step, speed_column)
        
        if fit_result:
            fitting_results.append(fit_result)
            print(f"  ✅ 拟合完成")
        else:
            print(f"  ❌ 拟合失败")
    
    # 保存拟合结果
    if fitting_results:
        save_fitting_results(fitting_results)
        create_fitting_visualization(fitting_results)
        analyze_fitting_patterns(fitting_results)
    else:
        print("没有成功的拟合结果")

def prepare_fitting_data(step_data, summary_data, speed_column, step_name):
    """准备拟合数据"""
    
    fitting_records = []
    
    for _, row in step_data.iterrows():
        time_period_id = row['时间段ID']
        start_time = row['开始时间']
        end_time = row['结束时间']
        delay_time = row['平均延时(秒)']
        
        # 筛选该时间段的数据
        period_mask = (summary_data['时间'] >= start_time) & (summary_data['时间'] <= end_time)
        period_summary = summary_data[period_mask]
        
        if len(period_summary) == 0:
            continue
        
        # 计算该时间段的平均速度
        speed_values = period_summary[speed_column].dropna()
        if len(speed_values) == 0:
            continue
        
        avg_speed = speed_values.mean()
        median_speed = speed_values.median()
        max_speed = speed_values.max()
        min_speed = speed_values.min()
        std_speed = speed_values.std()
        
        # 过滤异常值
        if avg_speed <= 0 or delay_time <= 0 or avg_speed > 10000 or delay_time > 3600:
            continue
        
        fitting_records.append({
            '时间段ID': time_period_id,
            '流程环节': step_name,
            '速度列': speed_column,
            '平均速度': avg_speed,
            '中位速度': median_speed,
            '最大速度': max_speed,
            '最小速度': min_speed,
            '速度标准差': std_speed,
            '延时时间': delay_time,
            '开始时间': start_time,
            '结束时间': end_time
        })
    
    if len(fitting_records) > 0:
        return pd.DataFrame(fitting_records)
    else:
        return None

def perform_multiple_fitting(data, step_name, speed_column):
    """执行多种拟合方法"""
    
    X = data['平均速度'].values.reshape(-1, 1)
    y = data['延时时间'].values
    
    fitting_methods = {}
    
    try:
        # 1. 线性拟合: y = ax + b
        linear_reg = LinearRegression()
        linear_reg.fit(X, y)
        y_pred_linear = linear_reg.predict(X)
        r2_linear = r2_score(y, y_pred_linear)
        mse_linear = mean_squared_error(y, y_pred_linear)
        
        fitting_methods['线性拟合'] = {
            '方程式': f"y = {linear_reg.coef_[0]:.4f}*x + {linear_reg.intercept_:.4f}",
            '系数a': linear_reg.coef_[0],
            '系数b': linear_reg.intercept_,
            'R²': r2_linear,
            'MSE': mse_linear,
            '预测值': y_pred_linear
        }
        
        # 2. 二次多项式拟合: y = ax² + bx + c
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, y)
        y_pred_poly = poly_reg.predict(X_poly)
        r2_poly = r2_score(y, y_pred_poly)
        mse_poly = mean_squared_error(y, y_pred_poly)
        
        fitting_methods['二次多项式拟合'] = {
            '方程式': f"y = {poly_reg.coef_[2]:.6f}*x² + {poly_reg.coef_[1]:.4f}*x + {poly_reg.intercept_:.4f}",
            '系数a': poly_reg.coef_[2],
            '系数b': poly_reg.coef_[1],
            '系数c': poly_reg.intercept_,
            'R²': r2_poly,
            'MSE': mse_poly,
            '预测值': y_pred_poly
        }
        
        # 3. 反比例拟合: y = a/x + b (当x > 0时)
        if np.all(X > 0):
            X_inv = 1 / X
            inv_reg = LinearRegression()
            inv_reg.fit(X_inv, y)
            y_pred_inv = inv_reg.predict(X_inv)
            r2_inv = r2_score(y, y_pred_inv)
            mse_inv = mean_squared_error(y, y_pred_inv)
            
            fitting_methods['反比例拟合'] = {
                '方程式': f"y = {inv_reg.coef_[0]:.4f}/x + {inv_reg.intercept_:.4f}",
                '系数a': inv_reg.coef_[0],
                '系数b': inv_reg.intercept_,
                'R²': r2_inv,
                'MSE': mse_inv,
                '预测值': y_pred_inv
            }
        
        # 4. 指数拟合: y = a*e^(bx) (当y > 0时)
        if np.all(y > 0):
            try:
                log_y = np.log(y)
                exp_reg = LinearRegression()
                exp_reg.fit(X, log_y)
                log_y_pred = exp_reg.predict(X)
                y_pred_exp = np.exp(log_y_pred)
                r2_exp = r2_score(y, y_pred_exp)
                mse_exp = mean_squared_error(y, y_pred_exp)
                
                fitting_methods['指数拟合'] = {
                    '方程式': f"y = {np.exp(exp_reg.intercept_):.4f}*exp({exp_reg.coef_[0]:.6f}*x)",
                    '系数a': np.exp(exp_reg.intercept_),
                    '系数b': exp_reg.coef_[0],
                    'R²': r2_exp,
                    'MSE': mse_exp,
                    '预测值': y_pred_exp
                }
            except:
                pass
        
        # 5. 幂函数拟合: y = a*x^b (当x > 0, y > 0时)
        if np.all(X > 0) and np.all(y > 0):
            try:
                log_X = np.log(X)
                log_y = np.log(y)
                power_reg = LinearRegression()
                power_reg.fit(log_X, log_y)
                log_y_pred = power_reg.predict(log_X)
                y_pred_power = np.exp(log_y_pred)
                r2_power = r2_score(y, y_pred_power)
                mse_power = mean_squared_error(y, y_pred_power)
                
                fitting_methods['幂函数拟合'] = {
                    '方程式': f"y = {np.exp(power_reg.intercept_):.4f}*x^{power_reg.coef_[0]:.4f}",
                    '系数a': np.exp(power_reg.intercept_),
                    '系数b': power_reg.coef_[0],
                    'R²': r2_power,
                    'MSE': mse_power,
                    '预测值': y_pred_power
                }
            except:
                pass
        
        # 选择最佳拟合方法（基于R²）
        best_method = max(fitting_methods.items(), key=lambda x: x[1]['R²'])
        
        # 计算皮尔逊相关系数
        correlation, p_value = stats.pearsonr(X.flatten(), y)
        
        result = {
            '流程环节': step_name,
            '速度列': speed_column,
            '数据点数': len(data),
            '速度范围': f"[{X.min():.2f}, {X.max():.2f}]",
            '延时范围': f"[{y.min():.2f}, {y.max():.2f}]",
            '皮尔逊相关系数': correlation,
            'P值': p_value,
            '最佳拟合方法': best_method[0],
            '最佳方程式': best_method[1]['方程式'],
            '最佳R²': best_method[1]['R²'],
            '最佳MSE': best_method[1]['MSE'],
            '所有拟合方法': fitting_methods,
            '原始数据': data
        }
        
        # 显示拟合结果
        print(f"    数据点数: {len(data)}")
        print(f"    速度范围: [{X.min():.2f}, {X.max():.2f}]")
        print(f"    延时范围: [{y.min():.2f}, {y.max():.2f}]")
        print(f"    相关系数: {correlation:.4f} (P值: {p_value:.4f})")
        print(f"    最佳拟合: {best_method[0]}")
        print(f"    最佳方程: {best_method[1]['方程式']}")
        print(f"    R²: {best_method[1]['R²']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"    拟合过程出错: {e}")
        return None

def save_fitting_results(fitting_results):
    """保存拟合结果"""
    
    # 创建详细拟合结果表
    detailed_results = []
    summary_results = []
    
    for result in fitting_results:
        # 汇总结果
        summary_row = {
            '流程环节': result['流程环节'],
            '速度列': result['速度列'],
            '数据点数': result['数据点数'],
            '速度范围': result['速度范围'],
            '延时范围': result['延时范围'],
            '皮尔逊相关系数': result['皮尔逊相关系数'],
            'P值': result['P值'],
            '最佳拟合方法': result['最佳拟合方法'],
            '最佳方程式': result['最佳方程式'],
            '最佳R²': result['最佳R²'],
            '最佳MSE': result['最佳MSE']
        }
        summary_results.append(summary_row)
        
        # 详细结果（所有拟合方法）
        for method_name, method_data in result['所有拟合方法'].items():
            detailed_row = {
                '流程环节': result['流程环节'],
                '速度列': result['速度列'],
                '拟合方法': method_name,
                '方程式': method_data['方程式'],
                'R²': method_data['R²'],
                'MSE': method_data['MSE'],
                '数据点数': result['数据点数'],
                '相关系数': result['皮尔逊相关系数']
            }
            
            # 添加系数信息
            if '系数a' in method_data:
                detailed_row['系数a'] = method_data['系数a']
            if '系数b' in method_data:
                detailed_row['系数b'] = method_data['系数b']
            if '系数c' in method_data:
                detailed_row['系数c'] = method_data['系数c']
            
            detailed_results.append(detailed_row)
    
    # 保存汇总结果
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_df = summary_df.sort_values('最佳R²', ascending=False)
        summary_df.to_csv('延时拟合方程汇总_分段_1.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
        print(f"\n延时拟合方程汇总已保存：延时拟合方程汇总_分段_1.csv")
    
    # 保存详细结果
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df = detailed_df.sort_values(['流程环节', 'R²'], ascending=[True, False])
        detailed_df.to_csv('延时拟合方程详细_分段_1.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
        print(f"延时拟合方程详细已保存：延时拟合方程详细_分段_1.csv")

def create_fitting_visualization(fitting_results):
    """创建拟合可视化"""
    
    n_results = len(fitting_results)
    if n_results == 0:
        return
    
    # 计算子图布局
    cols = min(4, n_results)
    rows = (n_results + cols - 1) // cols
    
    plt.figure(figsize=(5*cols, 4*rows))
    
    for i, result in enumerate(fitting_results[:16]):  # 最多显示16个
        plt.subplot(rows, cols, i+1)
        
        # 获取原始数据
        data = result['原始数据']
        X = data['平均速度'].values
        y = data['延时时间'].values
        
        # 绘制散点图
        plt.scatter(X, y, alpha=0.6, s=30, label='实际数据')
        
        # 绘制最佳拟合曲线
        best_method = result['最佳拟合方法']
        best_fit_data = result['所有拟合方法'][best_method]
        
        # 生成拟合曲线的x值
        x_range = np.linspace(X.min(), X.max(), 100)
        
        try:
            if best_method == '线性拟合':
                y_fit = best_fit_data['系数a'] * x_range + best_fit_data['系数b']
            elif best_method == '二次多项式拟合':
                y_fit = best_fit_data['系数a'] * x_range**2 + best_fit_data['系数b'] * x_range + best_fit_data['系数c']
            elif best_method == '反比例拟合':
                y_fit = best_fit_data['系数a'] / x_range + best_fit_data['系数b']
            elif best_method == '指数拟合':
                y_fit = best_fit_data['系数a'] * np.exp(best_fit_data['系数b'] * x_range)
            elif best_method == '幂函数拟合':
                y_fit = best_fit_data['系数a'] * (x_range ** best_fit_data['系数b'])
            else:
                y_fit = None
            
            if y_fit is not None:
                plt.plot(x_range, y_fit, 'r-', alpha=0.8, label=f'{best_method}')
        except:
            pass
        
        plt.title(f"{result['流程环节'][:15]}...\nR²={result['最佳R²']:.3f}", fontsize=10)
        plt.xlabel('平均速度')
        plt.ylabel('延时时间(秒)')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('延时拟合方程可视化_分段_1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"延时拟合方程可视化已保存：延时拟合方程可视化_分段_1.png")

def analyze_fitting_patterns(fitting_results):
    """分析拟合模式"""
    
    print(f"\n{'='*50}")
    print(f"拟合模式分析:")
    print(f"{'='*50}")
    
    # 统计拟合方法分布
    method_counts = {}
    r2_stats = []
    correlation_stats = []
    
    for result in fitting_results:
        method = result['最佳拟合方法']
        r2 = result['最佳R²']
        corr = result['皮尔逊相关系数']
        
        method_counts[method] = method_counts.get(method, 0) + 1
        r2_stats.append(r2)
        correlation_stats.append(abs(corr))
    
    print(f"\n最佳拟合方法分布:")
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(fitting_results)) * 100
        print(f"  {method}: {count}个 ({percentage:.1f}%)")
    
    print(f"\n拟合质量统计:")
    print(f"  平均R²: {np.mean(r2_stats):.4f}")
    print(f"  R²标准差: {np.std(r2_stats):.4f}")
    print(f"  最高R²: {np.max(r2_stats):.4f}")
    print(f"  最低R²: {np.min(r2_stats):.4f}")
    
    print(f"\n相关性统计:")
    print(f"  平均相关系数绝对值: {np.mean(correlation_stats):.4f}")
    print(f"  强相关(|r|>0.7): {sum(1 for c in correlation_stats if c > 0.7)}个")
    print(f"  中等相关(0.3<|r|≤0.7): {sum(1 for c in correlation_stats if 0.3 < c <= 0.7)}个")
    print(f"  弱相关(|r|≤0.3): {sum(1 for c in correlation_stats if c <= 0.3)}个")
    
    # 找出拟合效果最好的流程环节
    best_fits = sorted(fitting_results, key=lambda x: x['最佳R²'], reverse=True)[:5]
    print(f"\n拟合效果最佳的5个流程环节:")
    for i, result in enumerate(best_fits, 1):
        print(f"  {i}. {result['流程环节']}")
        print(f"     方程式: {result['最佳方程式']}")
        print(f"     R²: {result['最佳R²']:.4f}")
    
    # 找出相关性最强的流程环节
    strongest_corr = sorted(fitting_results, key=lambda x: abs(x['皮尔逊相关系数']), reverse=True)[:5]
    print(f"\n相关性最强的5个流程环节:")
    for i, result in enumerate(strongest_corr, 1):
        corr = result['皮尔逊相关系数']
        corr_type = "正相关" if corr > 0 else "负相关"
        print(f"  {i}. {result['流程环节']}")
        print(f"     相关系数: {corr:.4f} ({corr_type})")
        print(f"     方程式: {result['最佳方程式']}")

if __name__ == "__main__":
    analyze_by_time_periods() 