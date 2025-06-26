import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings
from scipy import signal
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_csv_with_encoding(file_path, encodings=['utf-8', 'gbk', 'gb2312', 'utf-8-sig']):
    """尝试不同编码读取CSV文件"""
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"使用 {encoding} 编码时发生错误: {e}")
            continue
    raise ValueError("无法读取文件，请检查文件编码")

def process_data_for_time_period(summary_df, start_time, end_time, time_interval_seconds=60):
    """
    为指定时间段处理数据
    
    参数:
    summary_df: 汇总数据DataFrame
    start_time: 开始时间
    end_time: 结束时间
    time_interval_seconds: 时间间隔（秒），默认60秒（1分钟）
                          支持任意秒数设置，例如：60, 30, 20, 10, 5, 3, 1等
    """
    # 转换时间列
    summary_df['时间'] = pd.to_datetime(summary_df['时间'])
    
    # 筛选时间段内的数据
    mask = (summary_df['时间'] >= start_time) & (summary_df['时间'] <= end_time)
    period_data = summary_df.loc[mask].copy()
    
    if period_data.empty:
        print(f"警告：时间段 {start_time} 到 {end_time} 没有数据")
        return None
    
    # 按指定时间间隔重采样
    period_data.set_index('时间', inplace=True)
    

    # 定义需要的列
    required_columns = [
        '折叠机实际速度', '折叠机入包数', '折叠机出包数', '外循环进内循环纸条数量', '存纸率',
        '裁切机实际速度', '有效总切数', '1#有效切数', '2#有效切数', '3#有效切数', '4#有效切数',
        '进第一裁切通道纸条计数', '进第二裁切通道纸条计数', '进第三裁切通道纸条计数', '进第四裁切通道纸条计数',
        '1#小包机入包数', '1#小包机实际速度', '2#小包机入包数', '2#小包机实际速度',
        '3#小包机入包数', '3#小包机主机实际速度', '4#小包机入包数', '4#小包机主机实际速度'
    ]
    
    # 检查缺失的列
    missing_cols = [col for col in required_columns if col not in period_data.columns]
    if missing_cols:
        print(f"警告：缺失以下列: {missing_cols}")
        # 使用可用的列
        available_cols = [col for col in required_columns if col in period_data.columns]
        if not available_cols:
            print("错误：没有找到任何需要的列")
            return None
        required_columns = available_cols
    
    # 创建结果字典
    result_data = {}
    
    # 累积量列（计算每分钟差值）
    cumulative_cols = [
        '折叠机入包数', '折叠机出包数', '有效总切数', '1#有效切数', '2#有效切数', 
        '3#有效切数', '4#有效切数', '1#小包机入包数', '2#小包机入包数', 
        '3#小包机入包数', '4#小包机入包数', '存纸率'
    ]
    
    # 瞬时量列
    instantaneous_cols = [
        '折叠机实际速度', '外循环进内循环纸条数量', '裁切机实际速度',
        '进第一裁切通道纸条计数', '进第二裁切通道纸条计数', '进第三裁切通道纸条计数', '进第四裁切通道纸条计数',
        '1#小包机实际速度', '2#小包机实际速度', '3#小包机主机实际速度', '4#小包机主机实际速度'
    ]
    
    # 按指定时间间隔重采样处理
    resample_freq = f'{time_interval_seconds}S'  # 生成重采样频率字符串，如'30S', '20S', '10S'
    interval_data = period_data.resample(resample_freq)
    
    # 处理累积量
    for col in cumulative_cols:
        if col in period_data.columns:
            # 计算每个时间间隔的差值
            interval_diff = interval_data[col].last() - interval_data[col].first()
            # 折叠机出包数和折叠机入包数等使用固定系数25
            if col in ['折叠机出包数', '折叠机入包数', '有效总切数', '1#有效切数', 
                       '2#有效切数', '3#有效切数', '4#有效切数', '1#小包机入包数', 
                       '2#小包机入包数', '3#小包机入包数', '4#小包机入包数']:
                # 统一使用固定系数25，不随时间间隔调整
                result_data[col] = (interval_diff / 25).values
            elif col == '存纸率':
                # 存纸率计算差值，不除以系数
                result_data[col] = interval_diff.values
            else:
                result_data[col] = interval_diff.values
    
    # 处理瞬时量
    if '折叠机实际速度' in period_data.columns:
        # 计算每个时间间隔平均值再使用固定系数
        avg_speed = interval_data['折叠机实际速度'].mean()
        # 使用固定系数9.75，不随时间间隔调整
        result_data['折叠机实际速度'] = (avg_speed / 25).round(2).values
    
    if '外循环进内循环纸条数量' in period_data.columns:
        # 计算每个时间间隔的和
        result_data['外循环进内循环纸条数量'] = interval_data['外循环进内循环纸条数量'].sum().values
    
    if '裁切机实际速度' in period_data.columns:
        # 计算每个时间间隔平均值使用固定系数
        avg_speed = interval_data['裁切机实际速度'].mean()
        # 使用固定系数9.75，不随时间间隔调整
        result_data['裁切机实际速度'] = (avg_speed / 25).round(2).values
    
    # 处理裁切通道纸条计数
    cut_channel_cols = ['进第一裁切通道纸条计数', '进第二裁切通道纸条计数', '进第三裁切通道纸条计数', '进第四裁切通道纸条计数']
    for col in cut_channel_cols:
        if col in period_data.columns:
            result_data[col] = interval_data[col].sum().values
    
    # 处理小包机速度
    packer_speed_cols = ['1#小包机实际速度', '2#小包机实际速度', '3#小包机主机实际速度', '4#小包机主机实际速度']
    packer_speeds = []
    for col in packer_speed_cols:
        if col in period_data.columns:
            avg_speed = interval_data[col].mean()
            # 使用固定系数25，不随时间间隔调整
            speed_processed = (avg_speed / 25).round(2)
            result_data[col] = speed_processed.values
            packer_speeds.append(speed_processed.values)
    
    # 计算小包机速度总和
    if packer_speeds:
        packer_speed_sum = np.sum(packer_speeds, axis=0)
        result_data['小包机速度总和'] = packer_speed_sum
    
    # 计算小包机入包数总和
    packer_input_cols = ['1#小包机入包数', '2#小包机入包数', '3#小包机入包数', '4#小包机入包数']
    packer_inputs = []
    for col in packer_input_cols:
        if col in result_data:
            packer_inputs.append(result_data[col])
    
    if packer_inputs:
        packer_input_sum = np.sum(packer_inputs, axis=0)
        result_data['小包机入包数总和'] = packer_input_sum
    
    # 存纸率已在累积量处理中计算差值
    
    # 创建时间索引
    time_index = interval_data.groups.keys()
    
    return result_data, list(time_index)

def plot_data(data_dict, time_index, start_time, end_time, output_dir, time_interval_seconds=60):
    """绘制数据图表
    
    参数:
    data_dict: 数据字典
    time_index: 时间索引
    start_time: 开始时间
    end_time: 结束时间
    output_dir: 输出目录
    time_interval_seconds: 时间间隔（秒），支持任意秒数设置
    """
    if not data_dict:
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算持续时间
    duration = end_time - start_time
    duration_str = str(duration).split('.')[0]  # 去掉微秒部分
    
    # 格式化标题，包含时间间隔信息
    interval_desc = f"{time_interval_seconds}秒间隔"
    title = f"时间段: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n持续时间: {duration_str} | 数据间隔: {interval_desc}"
    
    # 定义绘图顺序
    plot_order = [
        '折叠机实际速度', '折叠机入包数', '折叠机出包数', '外循环进内循环纸条数量', '存纸率',
        '裁切机实际速度', '有效总切数', '1#有效切数', '2#有效切数', '3#有效切数', '4#有效切数',
        '进第一裁切通道纸条计数', '进第二裁切通道纸条计数', '进第三裁切通道纸条计数', '进第四裁切通道纸条计数',
        '1#小包机入包数', '1#小包机实际速度', '2#小包机入包数', '2#小包机实际速度','小包机入包数总和',
        '3#小包机入包数', '3#小包机主机实际速度', '4#小包机入包数', '4#小包机主机实际速度',
        '小包机速度总和'
    ]
    
    # 过滤存在的列
    available_cols = [col for col in plot_order if col in data_dict]
    
    if not available_cols:
        print("没有可绘制的数据")
        return
    
    # 计算子图数量和布局
    n_cols = len(available_cols)
    n_rows = (n_cols + 2) // 3  # 每行3个子图
    
    # 创建图形，为标题留出更多空间
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows + 1))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 添加总标题
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # 扁平化axes数组
    axes_flat = axes.flatten()
    
    # 绘制每个指标
    for i, col in enumerate(available_cols):
        ax = axes_flat[i]
        
        if len(data_dict[col]) > 0:
            # 创建时间序列
            time_series = pd.Series(data_dict[col], index=time_index[:len(data_dict[col])])
            
            # 绘制曲线
            ax.plot(time_series.index, time_series.values, marker='o', markersize=3)
            ax.set_title(col, fontsize=12)
            ax.set_xlabel('时间')
            ax.set_ylabel('数值')
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            ax.tick_params(axis='x', rotation=45)
    
    # 隐藏多余的子图
    for i in range(len(available_cols), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # 调整布局，为顶部标题留出空间
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # 保存图片
    filename = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图片已保存: {filepath}")

def plot_combined_data(data_dict, time_index, start_time, end_time, output_dir, time_interval_seconds=60):
    """绘制组合数据图表 - 将指定的多列数据绘制在同一个图中，并进行时间偏移相关性分析
    
    参数:
    data_dict: 数据字典
    time_index: 时间索引
    start_time: 开始时间
    end_time: 结束时间
    output_dir: 输出目录
    time_interval_seconds: 时间间隔（秒），支持任意秒数设置
    """
    if not data_dict:
        return []
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算持续时间
    duration = end_time - start_time
    duration_str = str(duration).split('.')[0]  # 去掉微秒部分
    
    # 格式化标题，包含时间间隔信息
    interval_desc = f"{time_interval_seconds}秒间隔"
    title = f"时间段: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n持续时间: {duration_str} | 数据间隔: {interval_desc}"
    
    # ==============================================
    # 在这里定义要组合绘制的列数据组合
    # ==============================================
    plot_combinations = [
        {
            'title': '共同前置流程1',
            'columns': ['折叠机入包数','折叠机出包数'],
            'colors': ['purple', 'orange', 'brown', 'pink', 'gray', 'olive']
        },
        {
            'title': '共同前置流程2',
            'columns': ['折叠机出包数','外循环进内循环纸条数量'],
            'colors': ['red', 'blue', 'green', 'orange', 'purple']
        },
        {
            'title': '外循环分流连接1',
            'columns': ['外循环进内循环纸条数量','进第一裁切通道纸条计数'],
            'colors': ['red', 'blue', 'green', 'orange']
        },
        {
            'title': '外循环分流连接2',
            'columns': ['外循环进内循环纸条数量','进第二裁切通道纸条计数'],
            'colors': ['red', 'blue', 'green','orange','purple']
        },
        {
            'title': '外循环分流连接3',
            'columns': ['外循环进内循环纸条数量','进第三裁切通道纸条计数'],
            'colors': ['red', 'blue']
        },
        {
            'title': '外循环分流连接4',
            'columns': ['外循环进内循环纸条数量','进第四裁切通道纸条计数'],
            'colors': ['red', 'blue']
        },
        {
            'title': '生产线1流程1',
            'columns': ['进第一裁切通道纸条计数','1#有效切数'],
            'colors': ['red', 'blue']
        },
        {
            'title': '生产线1流程2',
            'columns': ['1#有效切数','1#小包机入包数'],
            'colors': ['red', 'blue']
        },
        {
            'title': '生产线2流程1',
            'columns': ['进第二裁切通道纸条计数','2#有效切数'],
            'colors': ['red', 'blue']
        },
        {
            'title': '生产线2流程2',
            'columns': ['2#有效切数','2#小包机入包数'],
            'colors': ['red', 'blue']
        },
        {
            'title': '生产线3流程1',
            'columns': ['进第三裁切通道纸条计数','3#有效切数'],
            'colors': ['red', 'blue']
        },
        {
            'title': '生产线3流程2',
            'columns': ['3#有效切数','3#小包机入包数'],
            'colors': ['red', 'blue']
        },
        {
            'title': '生产线4流程1',
            'columns': ['进第四裁切通道纸条计数','4#有效切数'],
            'colors': ['red', 'blue']
        },
        {
            'title': '生产线4流程3',
            'columns': ['4#有效切数','4#小包机入包数'],
            'colors': ['red', 'blue']
        }
    ]
    # ==============================================
    
    # 存储时间偏移分析结果
    shift_analysis_results = []
    
    # 为每个组合创建一个图表
    for combo_idx, combo in enumerate(plot_combinations):
        # 过滤存在的列
        available_cols = [col for col in combo['columns'] if col in data_dict and len(data_dict[col]) > 0]
        
        if not available_cols:
            print(f"跳过组合 '{combo['title']}'：没有可用数据")
            continue
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 添加标题
        plt.suptitle(f"{combo['title']}\n{title}", fontsize=14, fontweight='bold')
        
        # 绘制每列数据
        for i, col in enumerate(available_cols):
            if len(data_dict[col]) > 0:
                # 创建时间序列
                time_series = pd.Series(data_dict[col], index=time_index[:len(data_dict[col])])
                
                # 选择颜色
                color = combo['colors'][i % len(combo['colors'])]
                
                # 绘制曲线
                plt.plot(time_series.index, time_series.values, 
                        marker='o', markersize=4, label=col, color=color, linewidth=2)
        
        # 设置图表属性
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('数值', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 为每个组合创建独立的子文件夹
        safe_title = combo['title'].replace('/', '_').replace('\\', '_').replace(':', '_')
        combo_dir = os.path.join(output_dir, safe_title)
        os.makedirs(combo_dir, exist_ok=True)
        
        # 保存图片
        filename = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(combo_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"组合图表已保存: {filepath}")
        
        # 对于恰好有两个指标的组合，进行时间偏移相关性分析
        if len(available_cols) == 2:
            col1, col2 = available_cols
            shift_result = calculate_time_shift_correlation(
                data_dict[col1], 
                data_dict[col2], 
                col1, 
                col2, 
                time_interval_seconds
            )
            
            if shift_result:
                # 添加额外信息
                period_name = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}"
                shift_result.update({
                    'time_period': period_name,
                    'chart_title': combo['title'],
                    'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'time_interval_seconds': time_interval_seconds
                })
                shift_analysis_results.append(shift_result)
                print(f"  → 时间偏移分析: {shift_result['shift_description']}, 相关系数: {shift_result['best_correlation']:.3f}")
    
    return shift_analysis_results

def calculate_time_shift_correlation(data1, data2, col1_name, col2_name, time_interval_seconds=60, max_shift_seconds=300):
    """
    计算两个时间序列在不同时间偏移下的相关性
    
    参数:
    data1, data2: 两个时间序列数据
    col1_name, col2_name: 列名
    time_interval_seconds: 时间间隔（秒），支持任意秒数设置
    max_shift_seconds: 最大偏移时间（秒），默认5分钟
    
    返回:
    包含最佳偏移和相关性信息的字典
    """
    try:
        # 确保数据长度一致
        min_length = min(len(data1), len(data2))
        if min_length < 10:  # 数据点太少
            return None
        
        data1 = np.array(data1[:min_length])
        data2 = np.array(data2[:min_length])
        
        # 标准化数据（去除均值并归一化）
        data1_norm = (data1 - np.mean(data1)) / (np.std(data1) + 1e-8)
        data2_norm = (data2 - np.mean(data2)) / (np.std(data2) + 1e-8)
        
        # 计算最大偏移步数
        max_shift_steps = min(max_shift_seconds // time_interval_seconds, min_length // 2)
        max_shift_steps = max(1, int(max_shift_steps))
        
        correlations = []
        shifts = []
        
        # 计算不同偏移下的相关性（只考虑data2滞后于data1的情况）
        for shift in range(0, max_shift_steps + 1):
            if shift == 0:
                # 无偏移
                corr, _ = pearsonr(data1_norm, data2_norm)
            else:
                # data2 向后偏移（data2滞后于data1）
                if len(data1_norm[:-shift]) > 5 and len(data2_norm[shift:]) > 5:
                    corr, _ = pearsonr(data1_norm[:-shift], data2_norm[shift:])
                else:
                    corr = 0
            
            if not np.isnan(corr):
                correlations.append(corr)
                shifts.append(shift * time_interval_seconds)  # 转换为秒
            
        if not correlations:
            return None
            
        # 找到最高相关性
        max_corr_idx = np.argmax(np.abs(correlations))
        best_shift = shifts[max_corr_idx]
        best_correlation = correlations[max_corr_idx]
        
        # 解释偏移方向（只考虑data2滞后于data1的情况）
        if best_shift > 0:
            shift_description = f"{col2_name} 滞后 {col1_name} {abs(best_shift)} 秒"
        else:
            shift_description = "无明显时间偏移"
        
        return {
            'col1_name': col1_name,
            'col2_name': col2_name,
            'best_shift_seconds': best_shift,
            'best_correlation': best_correlation,
            'shift_description': shift_description,
            'all_shifts': shifts,
            'all_correlations': correlations,
            'data_points': min_length
        }
        
    except Exception as e:
        print(f"计算时间偏移相关性时出错: {e}")
        return None

def calculate_column_difference(data_dict, time_index, col1, col2, period_name):
    """计算两列数据的差值并统计正负比例"""
    if col1 not in data_dict or col2 not in data_dict:
        missing_cols = []
        if col1 not in data_dict:
            missing_cols.append(col1)
        if col2 not in data_dict:
            missing_cols.append(col2)
        print(f"警告：时间段 {period_name} 缺失列: {missing_cols}")
        return None
    
    # 确保两列数据长度一致
    min_length = min(len(data_dict[col1]), len(data_dict[col2]))
    if min_length == 0:
        print(f"警告：时间段 {period_name} 数据为空")
        return None
    
    col1_data = np.array(data_dict[col1][:min_length])
    col2_data = np.array(data_dict[col2][:min_length])
    
    # 计算差值 (col1 - col2)
    difference = col1_data - col2_data
    
    # 统计正负差值
    positive_count = np.sum(difference > 0)
    negative_count = np.sum(difference < 0)
    zero_count = np.sum(difference == 0)
    total_count = len(difference)
    
    # 计算比例
    positive_ratio = positive_count / total_count * 100 if total_count > 0 else 0
    negative_ratio = negative_count / total_count * 100 if total_count > 0 else 0
    zero_ratio = zero_count / total_count * 100 if total_count > 0 else 0
    
    # 创建结果字典
    result = {
        'period_name': period_name,
        'col1_name': col1,
        'col2_name': col2,
        'time_index': time_index[:min_length],
        'col1_data': col1_data,
        'col2_data': col2_data,
        'difference': difference,
        'total_count': total_count,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'zero_count': zero_count,
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'zero_ratio': zero_ratio,
        'mean_difference': np.mean(difference),
        'std_difference': np.std(difference),
        'max_difference': np.max(difference),
        'min_difference': np.min(difference)
    }
    
    return result

def calculate_compound_difference(data_dict, time_index, period_name):
    """计算复合差值：(折叠机出包数 - 外循环进内循环纸条数量) × 1.37 - 存纸率"""
    required_cols = ['折叠机出包数', '外循环进内循环纸条数量', '存纸率']
    missing_cols = [col for col in required_cols if col not in data_dict]
    
    if missing_cols:
        print(f"警告：时间段 {period_name} 缺失列: {missing_cols}")
        return None
    
    # 确保所有数据长度一致
    min_length = min(len(data_dict[col]) for col in required_cols)
    if min_length == 0:
        print(f"警告：时间段 {period_name} 数据为空")
        return None
    
    # 获取数据
    folding_out = np.array(data_dict['折叠机出包数'][:min_length])
    inner_loop = np.array(data_dict['外循环进内循环纸条数量'][:min_length])
    storage_rate = np.array(data_dict['存纸率'][:min_length])
    
    # 第一步：计算 (折叠机出包数 - 外循环进内循环纸条数量) × 1.37
    step1_diff = (folding_out - inner_loop) * 1.37
    
    # 第二步：计算 step1_diff - 存纸率
    final_diff = step1_diff - storage_rate
    
    # 统计正负差值
    positive_count = np.sum(final_diff > 0)
    negative_count = np.sum(final_diff < 0)
    zero_count = np.sum(final_diff == 0)
    total_count = len(final_diff)
    
    # 计算比例
    positive_ratio = positive_count / total_count * 100 if total_count > 0 else 0
    negative_ratio = negative_count / total_count * 100 if total_count > 0 else 0
    zero_ratio = zero_count / total_count * 100 if total_count > 0 else 0
    
    # 创建结果字典
    result = {
        'period_name': period_name,
        'calculation_formula': '(折叠机出包数 - 外循环进内循环纸条数量) × 1.37 - 存纸率',
        'time_index': time_index[:min_length],
        'folding_out_data': folding_out,
        'inner_loop_data': inner_loop,
        'storage_rate_data': storage_rate,
        'step1_diff': step1_diff,
        'final_difference': final_diff,
        'total_count': total_count,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'zero_count': zero_count,
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'zero_ratio': zero_ratio,
        'mean_difference': np.mean(final_diff),
        'std_difference': np.std(final_diff),
        'max_difference': np.max(final_diff),
        'min_difference': np.min(final_diff),
        'step1_mean': np.mean(step1_diff),
        'step1_std': np.std(step1_diff)
    }
    
    return result

def save_compound_difference_analysis(all_results, output_dir):
    """保存复合差值分析结果到CSV文件"""
    if not all_results:
        print("没有复合差值分析结果可保存")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细数据
    detailed_data = []
    for result in all_results:
        for i, (time_point, folding_val, inner_val, storage_val, step1_val, final_val) in enumerate(zip(
            result['time_index'], result['folding_out_data'], result['inner_loop_data'], 
            result['storage_rate_data'], result['step1_diff'], result['final_difference'])):
            detailed_data.append({
                '时间段': result['period_name'],
                '时间点': time_point,
                '折叠机出包数': folding_val,
                '外循环进内循环纸条数量': inner_val,
                '存纸率': storage_val,
                '第一步计算((折叠机出包数-外循环)×1.37)': step1_val,
                '最终差值(第一步-存纸率)': final_val,
                '差值符号': '正' if final_val > 0 else ('负' if final_val < 0 else '零')
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_filename = "复合差值分析_详细数据.csv"
    detailed_path = os.path.join(output_dir, detailed_filename)
    detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
    print(f"复合差值详细数据已保存: {detailed_path}")
    
    # 保存汇总统计
    summary_data = []
    for result in all_results:
        summary_data.append({
            '时间段': result['period_name'],
            '计算公式': result['calculation_formula'],
            '总数据点': result['total_count'],
            '正差值数量': result['positive_count'],
            '负差值数量': result['negative_count'],
            '零差值数量': result['zero_count'],
            '正差值比例(%)': round(result['positive_ratio'], 2),
            '负差值比例(%)': round(result['negative_ratio'], 2),
            '零差值比例(%)': round(result['zero_ratio'], 2),
            '最终差值平均值': round(result['mean_difference'], 4),
            '最终差值标准差': round(result['std_difference'], 4),
            '最终差值最大值': round(result['max_difference'], 4),
            '最终差值最小值': round(result['min_difference'], 4),
            '第一步计算平均值': round(result['step1_mean'], 4),
            '第一步计算标准差': round(result['step1_std'], 4)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = "复合差值分析_统计汇总.csv"
    summary_path = os.path.join(output_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"复合差值统计汇总已保存: {summary_path}")
    
    # 计算总体统计
    total_positive = sum(r['positive_count'] for r in all_results)
    total_negative = sum(r['negative_count'] for r in all_results)
    total_zero = sum(r['zero_count'] for r in all_results)
    total_all = sum(r['total_count'] for r in all_results)
    
    # 保存总体统计结果
    if total_all > 0:
        total_positive_ratio = (total_positive / total_all) * 100
        total_negative_ratio = (total_negative / total_all) * 100
        total_zero_ratio = (total_zero / total_all) * 100
        
        # 计算加权平均差值
        total_weighted_sum = sum(result['mean_difference'] * result['total_count'] for result in all_results)
        average_difference = total_weighted_sum / total_all
        
        # 判断整体趋势
        if total_positive_ratio > total_negative_ratio:
            trend = "复合计算结果 > 0 (正差值占主导)"
        elif total_negative_ratio > total_positive_ratio:
            trend = "复合计算结果 < 0 (负差值占主导)"
        else:
            trend = "复合计算结果 ≈ 0 (正负差值基本相等)"
        
        # 创建总体统计数据
        overall_stats = {
            '分析项目': ['(折叠机出包数 - 外循环进内循环纸条数量) × 1.37 - 存纸率'],
            '总数据点数': [total_all],
            '正差值数量': [total_positive],
            '正差值比例(%)': [round(total_positive_ratio, 2)],
            '负差值数量': [total_negative],
            '负差值比例(%)': [round(total_negative_ratio, 2)],
            '零差值数量': [total_zero],
            '零差值比例(%)': [round(total_zero_ratio, 2)],
            '平均差值': [round(average_difference, 4)],
            '总体趋势': [trend]
        }
        
        # 创建DataFrame并保存
        overall_df = pd.DataFrame(overall_stats)
        overall_filename = "复合差值分析_总体统计.csv"
        overall_path = os.path.join(output_dir, overall_filename)
        overall_df.to_csv(overall_path, index=False, encoding='utf-8-sig')
        print(f"复合差值总体统计已保存: {overall_path}")
        
        # 输出总体统计结果
        print("\n" + "="*80)
        print(f"🎯 复合差值分析总体统计结果")
        print("="*80)
        print(f"计算公式: (折叠机出包数 - 外循环进内循环纸条数量) × 1.37 - 存纸率")
        print(f"总数据点数: {total_all:,}")
        print(f"正差值: {total_positive:,} 个 ({total_positive_ratio:.2f}%)")
        print(f"负差值: {total_negative:,} 个 ({total_negative_ratio:.2f}%)")
        print(f"零差值: {total_zero:,} 个 ({total_zero_ratio:.2f}%)")
        print(f"平均差值: {average_difference:.4f}")
        print("="*80)
        print(f"📊 {trend}")
        print("="*80)
    
    print(f"\n=== 复合差值分析总结 ===")
    print(f"总数据点: {total_all}")
    print(f"正差值: {total_positive} 个 ({total_positive/total_all*100:.2f}%)")
    print(f"负差值: {total_negative} 个 ({total_negative/total_all*100:.2f}%)")
    print(f"零差值: {total_zero} 个 ({total_zero/total_all*100:.2f}%)")

def save_difference_analysis(all_results, col1, col2, output_dir):
    """保存差值分析结果到CSV文件"""
    if not all_results:
        print("没有差值分析结果可保存")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细数据
    detailed_data = []
    for result in all_results:
        for i, (time_point, diff_val, col1_val, col2_val) in enumerate(zip(
            result['time_index'], result['difference'], result['col1_data'], result['col2_data'])):
            detailed_data.append({
                '时间段': result['period_name'],
                '时间点': time_point,
                f'{col1}': col1_val,
                f'{col2}': col2_val,
                '差值': diff_val,
                '差值符号': '正' if diff_val > 0 else ('负' if diff_val < 0 else '零')
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_filename = f"{col1}_减_{col2}_详细数据.csv"
    detailed_path = os.path.join(output_dir, detailed_filename)
    detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
    print(f"详细差值数据已保存: {detailed_path}")
    
    # 保存汇总统计
    summary_data = []
    for result in all_results:
        summary_data.append({
            '时间段': result['period_name'],
            '总数据点': result['total_count'],
            '正差值数量': result['positive_count'],
            '负差值数量': result['negative_count'],
            '零差值数量': result['zero_count'],
            '正差值比例(%)': round(result['positive_ratio'], 2),
            '负差值比例(%)': round(result['negative_ratio'], 2),
            '零差值比例(%)': round(result['zero_ratio'], 2),
            '平均差值': round(result['mean_difference'], 4),
            '差值标准差': round(result['std_difference'], 4),
            '最大差值': round(result['max_difference'], 4),
            '最小差值': round(result['min_difference'], 4)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"{col1}_减_{col2}_统计汇总.csv"
    summary_path = os.path.join(output_dir, summary_filename)
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"差值统计汇总已保存: {summary_path}")
    
    # 计算总体统计
    total_positive = sum(r['positive_count'] for r in all_results)
    total_negative = sum(r['negative_count'] for r in all_results)
    total_zero = sum(r['zero_count'] for r in all_results)
    total_all = sum(r['total_count'] for r in all_results)
    
    print(f"\n=== {col1} - {col2} 差值分析总结 ===")
    print(f"总数据点: {total_all}")
    print(f"正差值: {total_positive} 个 ({total_positive/total_all*100:.2f}%)")
    print(f"负差值: {total_negative} 个 ({total_negative/total_all*100:.2f}%)")
    print(f"零差值: {total_zero} 个 ({total_zero/total_all*100:.2f}%)")
    print(f"平均差值: {np.mean([r['mean_difference'] for r in all_results]):.4f}")

def save_overall_statistics(col1, col2, total_data_points, total_positive_count, 
                           total_negative_count, total_zero_count, 
                           total_positive_ratio, total_negative_ratio, total_zero_ratio,
                           average_difference, trend, output_dir):
    """保存总体统计结果到CSV文件"""
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建总体统计数据
        overall_stats = {
            '分析项目': [f'{col1} - {col2}'],
            '总数据点数': [total_data_points],
            '正差值数量': [total_positive_count],
            '正差值比例(%)': [round(total_positive_ratio, 2)],
            '负差值数量': [total_negative_count],
            '负差值比例(%)': [round(total_negative_ratio, 2)],
            '零差值数量': [total_zero_count],
            '零差值比例(%)': [round(total_zero_ratio, 2)],
            '平均差值': [round(average_difference, 4)],
            '总体趋势': [trend]
        }
        
        # 创建DataFrame并保存
        overall_df = pd.DataFrame(overall_stats)
        filename = f"{col1}_减_{col2}_总体统计.csv"
        filepath = os.path.join(output_dir, filename)
        overall_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        print(f"总体统计结果已保存: {filepath}")
        
    except Exception as e:
        print(f"保存总体统计结果时发生错误: {e}")

def get_user_column_selection(available_columns):
    """获取用户选择的两列数据"""
    print("\n=== 差值分析功能 ===")
    print("可用的列名:")
    for i, col in enumerate(available_columns, 1):
        print(f"{i:2d}. {col}")
    
    print(f"\n请选择两列数据进行差值分析（格式：列名1,列名2）")
    print("或输入 'skip' 跳过差值分析")
    
    user_input = input("请输入: ").strip()
    
    if user_input.lower() == 'skip':
        return None, None
    
    try:
        col1, col2 = [col.strip() for col in user_input.split(',')]
        if col1 in available_columns and col2 in available_columns:
            return col1, col2
        else:
            print("错误：输入的列名不在可用列表中")
            return None, None
    except ValueError:
        print("错误：请按格式输入，例如：折叠机入包数,折叠机出包数")
        return None, None

def main():
    """主函数"""
    
    # ================================================================
    # 🔧 配置参数区域 - 您可以在这里修改时间间隔
    # ================================================================
    TIME_INTERVAL_SECONDS = 1  # 时间间隔（秒）
    """
    可设置为任意秒数，例如：
    - 60  : 1分钟间隔
    - 30  : 30秒间隔
    - 20  : 20秒间隔  
    - 10  : 10秒间隔
    - 5   : 5秒间隔
    - 3   : 3秒间隔
    - 1   : 1秒间隔（最小粒度）
    """
    # ================================================================
    
    print(f"📊 当前设置：每 {TIME_INTERVAL_SECONDS} 秒统计一次产量数据")
    
    # 读取时间段文件
    time_periods_file = "折叠机正常运行且高存纸率时间段_最终结果.csv"
    summary_file = "存纸架数据汇总.csv"
    
    try:
        # 读取时间段数据
        time_periods_df = pd.read_csv(time_periods_file)
        print(f"成功读取时间段文件，共 {len(time_periods_df)} 个时间段")
        
        # 读取汇总数据
        summary_df = read_csv_with_encoding(summary_file)
        print(f"成功读取汇总文件，共 {len(summary_df)} 行数据")
        print(f"汇总文件列名: {list(summary_df.columns)}")
        
        # 创建输出目录
        output_dir = f"时间段分析图表_{TIME_INTERVAL_SECONDS}秒"
        combined_output_dir = f"组合图表分析_{TIME_INTERVAL_SECONDS}秒"
        difference_output_dir = f"差值分析结果_{TIME_INTERVAL_SECONDS}秒"
        
        # 存储所有时间段的数据用于差值分析
        all_period_data = []
        available_columns = set()
        
        # 存储所有时间偏移分析结果
        all_shift_analysis_results = []
        
        # 处理每个时间段
        for idx, row in time_periods_df.iterrows():
            start_time = pd.to_datetime(row['开始时间'])
            end_time = pd.to_datetime(row['结束时间'])
            
            print(f"\n处理时间段 {idx+1}/{len(time_periods_df)}: {start_time} 到 {end_time}")
            
            # 处理数据
            result = process_data_for_time_period(summary_df, start_time, end_time, TIME_INTERVAL_SECONDS)
            
            if result is not None:
                data_dict, time_index = result
                
                # 收集可用列名
                available_columns.update(data_dict.keys())
                
                # 存储数据用于差值分析
                period_name = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}"
                all_period_data.append({
                    'period_name': period_name,
                    'data_dict': data_dict,
                    'time_index': time_index
                })
                
                # 绘制图表
                plot_data(data_dict, time_index, start_time, end_time, output_dir, TIME_INTERVAL_SECONDS)
                
                # 绘制组合图表并获取时间偏移分析结果
                shift_results = plot_combined_data(data_dict, time_index, start_time, end_time, combined_output_dir, TIME_INTERVAL_SECONDS)
                if shift_results:
                    all_shift_analysis_results.extend(shift_results)
            else:
                print(f"跳过时间段 {idx+1}，无数据")
        
        print(f"\n所有单项图表已保存到目录: {output_dir}")
        print(f"所有组合图表已保存到目录: {combined_output_dir}")
        
        # 保存时间偏移相关性分析结果
        if all_shift_analysis_results:
            # 创建时间偏移分析结果目录
            shift_analysis_output_dir = f"时间偏移相关性分析结果_{TIME_INTERVAL_SECONDS}秒"
            os.makedirs(shift_analysis_output_dir, exist_ok=True)
            
            # 转换为DataFrame
            df_results = pd.DataFrame(all_shift_analysis_results)
            
            # 重新排列列的顺序，便于阅读，并转换为中文列名
            columns_mapping = {
                'time_period': '时间段标识',
                'chart_title': '图表标题', 
                'start_time': '开始时间',
                'end_time': '结束时间',
                'col1_name': '第一个指标',
                'col2_name': '第二个指标',
                'best_shift_seconds': '最佳偏移时间(秒)',
                'best_correlation': '最佳相关系数',
                'shift_description': '时间偏移描述',
                'data_points': '数据点数量',
                'time_interval_seconds': '时间间隔设置(秒)'
            }
            
            # 重命名列
            df_results = df_results.rename(columns=columns_mapping)
            
            # 重新排列列的顺序
            columns_order = [
                '时间段标识', '图表标题', '开始时间', '结束时间',
                '第一个指标', '第二个指标', '最佳偏移时间(秒)', '最佳相关系数',
                '时间偏移描述', '数据点数量', '时间间隔设置(秒)'
            ]
            df_results = df_results[columns_order]
            
            # 保存汇总结果（按照最佳相关系数降序排序）
            summary_filename = f"时间偏移相关性分析_汇总结果_{TIME_INTERVAL_SECONDS}秒.csv"
            summary_filepath = os.path.join(shift_analysis_output_dir, summary_filename)
            # 按照最佳相关系数降序排序
            df_results_sorted = df_results.sort_values(by='最佳相关系数', ascending=False)
            df_results_sorted.to_csv(summary_filepath, index=False, encoding='utf-8-sig',float_format='%.3f')
            print(f"\n时间偏移相关性分析汇总结果已保存(已按相关系数排序): {summary_filepath}")
            
            # 按图表类型分组保存（每组内按照最佳相关系数降序排序）
            grouped = df_results.groupby('图表标题')
            for chart_title, group in grouped:
                safe_title = chart_title.replace('/', '_').replace('\\', '_').replace(':', '_')
                group_filename = f"时间偏移分析_{safe_title}_{TIME_INTERVAL_SECONDS}秒.csv"
                group_filepath = os.path.join(shift_analysis_output_dir, group_filename)
                # 每组内按最佳相关系数降序排序
                sorted_group = group.sort_values(by='最佳相关系数', ascending=False)
                sorted_group.to_csv(group_filepath, index=False, encoding='utf-8-sig',float_format='%.3f')
                print(f"  → {chart_title} 分析结果已保存(已按相关系数排序): {group_filepath}")
            
            # 生成分析总结并保存到文本文件
            summary_lines = []
            summary_lines.append(f"=== 时间偏移相关性分析总结 ===")
            summary_lines.append(f"总分析组合数: {len(all_shift_analysis_results)}")
            summary_lines.append(f"涉及图表类型: {len(grouped)} 种")
            summary_lines.append(f"时间间隔设置: {TIME_INTERVAL_SECONDS} 秒")
            
            # 显示相关性最高的前5个结果
            top_correlations = df_results_sorted.head(5)
            summary_lines.append(f"\n相关性最高的前5个结果:")
            for idx, row in top_correlations.iterrows():
                summary_line = f"  {row['图表标题']}: {row['时间偏移描述']}, 相关系数: {row['最佳相关系数']:.3f}"
                summary_lines.append(summary_line)
            
            # 显示延迟时间统计
            delay_stats = df_results['最佳偏移时间(秒)'].describe()
            summary_lines.append(f"\n时间偏移统计 (秒):")
            summary_lines.append(f"  平均偏移: {delay_stats['mean']:.1f} 秒")
            summary_lines.append(f"  偏移范围: {delay_stats['min']:.0f} 至 {delay_stats['max']:.0f} 秒")
            summary_lines.append(f"  标准差: {delay_stats['std']:.1f} 秒")
            
            # 保存分析总结到文本文件
            summary_text_filename = f"时间偏移分析总结_{TIME_INTERVAL_SECONDS}秒.txt"
            summary_text_filepath = os.path.join(shift_analysis_output_dir, summary_text_filename)
            
            # 添加每个组合的详细分析结果（按相关系数排序）
            detailed_summary_lines = []
            detailed_summary_lines.append("\n" + "=" * 80)
            detailed_summary_lines.append("【详细时间偏移分析结果】")
            detailed_summary_lines.append("=" * 80)
            detailed_summary_lines.append("")
            
            # 按图表名称分组显示详细结果
            for chart_title, group_data in grouped:
                detailed_summary_lines.append(f"【{chart_title}】:")
                # 按最佳相关系数排序
                sorted_group = group_data.sort_values(by=['最佳相关系数'], ascending=False)
                for i, (_, row) in enumerate(sorted_group.iterrows()):
                    detailed_summary_lines.append(f"  {i+1}. 时间段: {row['开始时间']} 至 {row['结束时间']}")
                    detailed_summary_lines.append(f"     {row['时间偏移描述']}")
                    detailed_summary_lines.append(f"     相关系数: {row['最佳相关系数']:.4f}")
                    detailed_summary_lines.append(f"     数据点数: {row['数据点数量']}")
                    detailed_summary_lines.append("")
                detailed_summary_lines.append("-" * 50)
                detailed_summary_lines.append("")
            
            # 添加流程链分析 - 按生产流程顺序分析时间传递
            flow_analysis_lines = []
            flow_analysis_lines.append("=" * 80)
            flow_analysis_lines.append("【生产流程时间传递分析】")
            flow_analysis_lines.append("=" * 80)
            flow_analysis_lines.append("")
            
            # 定义关键流程链
            process_chains = [
                # 共同前置流程链
                ["共同前置流程1", "共同前置流程2"],
                # 生产线1流程链
                ["外循环分流连接1", "生产线1流程1", "生产线1流程2"],
                # 生产线2流程链
                ["外循环分流连接2", "生产线2流程1", "生产线2流程2"],
                # 生产线3流程链
                ["外循环分流连接3", "生产线3流程1", "生产线3流程2"],
                # 生产线4流程链
                ["外循环分流连接4", "生产线4流程1", "生产线4流程2"]
            ]
            
            # 计算每条流程链的平均时间偏移
            for i, chain in enumerate(process_chains):
                chain_name = f"流程链 {i+1}" if i > 0 else "共同前置流程链"
                flow_analysis_lines.append(f"【{chain_name}】")
                
                # 收集链中图表的平均时间偏移
                chain_shifts = []
                for chart_title in chain:
                    if chart_title in grouped.groups:
                        # 获取该图表的所有时间偏移数据
                        chart_data = grouped.get_group(chart_title)
                        avg_shift = chart_data['最佳偏移时间(秒)'].mean()
                        avg_corr = chart_data['最佳相关系数'].mean()
                        chain_shifts.append((chart_title, avg_shift, avg_corr))
                        flow_analysis_lines.append(f"  {chart_title}: 平均偏移 {avg_shift:.1f} 秒, 平均相关系数 {avg_corr:.4f}")
                
                # 计算总偏移
                if chain_shifts:
                    total_shift = sum(shift for _, shift, _ in chain_shifts)
                    flow_analysis_lines.append(f"  >>> 总体延迟: {total_shift:.1f} 秒")
                    
                    # 计算平均相关系数
                    avg_chain_corr = sum(corr for _, _, corr in chain_shifts) / len(chain_shifts)
                    flow_analysis_lines.append(f"  >>> 链平均相关系数: {avg_chain_corr:.4f}")
                flow_analysis_lines.append("")
            
            # 合并所有内容
            all_summary_lines = summary_lines + detailed_summary_lines + flow_analysis_lines
            with open(summary_text_filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_summary_lines))
            
            # 在终端显示分析总结
            print(f"\n=== 时间偏移相关性分析总结 ===")
            print(f"总分析组合数: {len(all_shift_analysis_results)}")
            print(f"涉及图表类型: {len(grouped)} 种")
            print(f"时间间隔设置: {TIME_INTERVAL_SECONDS} 秒")
            
            print(f"\n相关性最高的前5个结果:")
            for idx, row in top_correlations.iterrows():
                print(f"  {row['图表标题']}: {row['时间偏移描述']}, 相关系数: {row['最佳相关系数']:.3f}")
            
            print(f"\n时间偏移统计 (秒):")
            print(f"  平均偏移: {delay_stats['mean']:.1f} 秒")
            print(f"  偏移范围: {delay_stats['min']:.0f} 至 {delay_stats['max']:.0f} 秒")
            print(f"  标准差: {delay_stats['std']:.1f} 秒")
            print(f"\n分析总结已保存到: {summary_text_filepath}")
        else:
            print(f"\n未生成时间偏移相关性分析结果")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 