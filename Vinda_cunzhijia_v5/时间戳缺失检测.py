import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

def check_timestamp_gaps(file_path, time_column='时间', time_format='%Y-%m-%d %H:%M:%S', interval_seconds=1):
    """
    检测CSV文件中时间戳是否有缺失
    
    参数:
    file_path: CSV文件路径
    time_column: 时间列名称，默认为'时间'
    time_format: 时间格式，默认为'%Y-%m-%d %H:%M:%S'
    interval_seconds: 预期的时间间隔（秒），默认为1秒
    
    返回:
    字典包含缺失信息和统计数据
    """
    
    try:
        print(f"正在读取文件: {file_path}")
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        if time_column not in df.columns:
            print(f"错误：未找到时间列 '{time_column}'")
            print(f"可用列名: {list(df.columns)}")
            return None
            
        print(f"文件读取成功，共有 {len(df):,} 行数据")
        
        # 转换时间列为datetime类型
        df[time_column] = pd.to_datetime(df[time_column], format=time_format)
        
        # 按时间排序
        df = df.sort_values(time_column).reset_index(drop=True)
        
        # 获取时间范围
        start_time = df[time_column].iloc[0]
        end_time = df[time_column].iloc[-1]

        print(f"时间范围: {start_time} 到 {end_time}")

        # 生成完整的时间序列（按指定间隔）
        expected_times = pd.date_range(start=start_time, end=end_time, freq=f'{interval_seconds}s')
        
        print(f"预期时间点数量: {len(expected_times):,}")
        print(f"实际时间点数量: {len(df):,}")
        
        # 找出缺失的时间点
        actual_times = set(df[time_column])
        expected_times_set = set(expected_times)
        missing_times = expected_times_set - actual_times
        
        # 将缺失时间转换为列表并排序
        missing_times_list = sorted(list(missing_times))
        
        # 按天分组统计缺失时间点
        daily_missing = analyze_daily_missing_points(missing_times_list, start_time, end_time)
        
        # 统计信息
        total_expected = len(expected_times)
        total_actual = len(df)
        total_missing = len(missing_times_list)
        missing_percentage = (total_missing / total_expected) * 100 if total_expected > 0 else 0
        
        # 创建结果字典
        result = {
            'file_path': file_path,
            'time_range': {
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time
            },
            'statistics': {
                'expected_count': total_expected,
                'actual_count': total_actual,
                'missing_count': total_missing,
                'missing_percentage': round(missing_percentage, 4)
            },
            'missing_timestamps': missing_times_list,
            'daily_missing': daily_missing
        }
        
        return result
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None

def analyze_daily_missing_points(missing_times, start_time, end_time):
    """
    按天分析缺失的时间点
    """
    if not missing_times:
        return {}
    
    # 按日期分组
    daily_missing = {}
    
    # 获取所有涉及的日期
    current_date = start_time.date()
    end_date = end_time.date()
    
    # 初始化每天的统计
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        daily_missing[date_str] = {
            'date': current_date,
            'missing_count': 0,
            'missing_times': [],
            'expected_seconds_in_day': 0
        }
        current_date += timedelta(days=1)
    
    # 计算每天的预期秒数
    for date_str, info in daily_missing.items():
        date = info['date']
        if date == start_time.date() and date == end_time.date():
            # 开始和结束在同一天
            info['expected_seconds_in_day'] = int((end_time - start_time).total_seconds()) + 1
        elif date == start_time.date():
            # 第一天
            end_of_day = datetime.combine(date, datetime.max.time().replace(microsecond=0))
            info['expected_seconds_in_day'] = int((end_of_day - start_time).total_seconds()) + 1
        elif date == end_time.date():
            # 最后一天
            start_of_day = datetime.combine(date, datetime.min.time())
            info['expected_seconds_in_day'] = int((end_time - start_of_day).total_seconds()) + 1
        else:
            # 完整的一天
            info['expected_seconds_in_day'] = 86400
    
    # 统计每天的缺失时间点
    for timestamp in missing_times:
        date_str = timestamp.strftime('%Y-%m-%d')
        if date_str in daily_missing:
            daily_missing[date_str]['missing_count'] += 1
            daily_missing[date_str]['missing_times'].append(timestamp)
    
    # 计算每天的缺失比例
    for date_str, info in daily_missing.items():
        if info['expected_seconds_in_day'] > 0:
            info['missing_percentage'] = (info['missing_count'] / info['expected_seconds_in_day']) * 100
        else:
            info['missing_percentage'] = 0.0
    
    return daily_missing

def print_missing_report(result):
    """
    打印缺失时间戳的详细报告
    """
    if not result:
        print("无法生成报告：结果为空")
        return
    
    print("\n" + "="*80)
    print("时间戳缺失检测报告")
    print("="*80)
    
    # 基本信息
    print(f"文件路径: {result['file_path']}")
    print(f"时间范围: {result['time_range']['start']} 到 {result['time_range']['end']}")
    print(f"总时长: {result['time_range']['duration']}")
    
    # 统计信息
    stats = result['statistics']
    print(f"\n总体统计信息:")
    print(f"  预期时间点数量: {stats['expected_count']:,}")
    print(f"  实际时间点数量: {stats['actual_count']:,}")
    print(f"  缺失时间点数量: {stats['missing_count']:,}")
    print(f"  总缺失比例: {stats['missing_percentage']:.4f}%")
    print(f"  数据完整度: {100 - stats['missing_percentage']:.4f}%")
    
    # 按天统计缺失情况
    daily_missing = result['daily_missing']
    print(f"\n" + "="*80)
    print("每日缺失统计详情")
    print("="*80)
    
    total_missing_days = 0
    for date_str, info in sorted(daily_missing.items()):
        if info['missing_count'] > 0:
            total_missing_days += 1
            print(f"\n📅 {date_str} (星期{get_weekday_chinese(info['date'].weekday())})")
            print(f"   预期时间点: {info['expected_seconds_in_day']:,}")
            print(f"   缺失时间点: {info['missing_count']:,}")
            print(f"   缺失比例: {info['missing_percentage']:.4f}%")
            print(f"   数据完整度: {100 - info['missing_percentage']:.4f}%")
            
            # 显示缺失的具体时间点
            if info['missing_times']:
                print(f"   缺失的时间点:")
                # 分组显示连续的时间点
                groups = group_consecutive_times(info['missing_times'])
                for i, group in enumerate(groups[:10], 1):  # 只显示前10组
                    if len(group) == 1:
                        print(f"     {i:2d}. {group[0].strftime('%H:%M:%S')}")
                    else:
                        print(f"     {i:2d}. {group[0].strftime('%H:%M:%S')} - {group[-1].strftime('%H:%M:%S')} (连续{len(group)}个)")
                
                if len(groups) > 10:
                    remaining_count = sum(len(group) for group in groups[10:])
                    print(f"     ... 还有 {len(groups) - 10} 组缺失时间段，共 {remaining_count} 个时间点")
    
    print(f"\n" + "="*80)
    print("汇总统计")
    print("="*80)
    print(f"总共涉及 {len([d for d in daily_missing.values() if d['missing_count'] > 0])} 天有缺失数据")
    print(f"总缺失时间点数量: {stats['missing_count']:,}")
    
    # 按天统计表格
    print(f"\n每日缺失汇总表:")
    print("-" * 80)
    print(f"{'日期':<12} {'星期':<6} {'预期点数':<10} {'缺失点数':<10} {'缺失比例':<10} {'完整度':<10}")
    print("-" * 80)
    
    for date_str, info in sorted(daily_missing.items()):
        if info['missing_count'] > 0:
            weekday = get_weekday_chinese(info['date'].weekday())
            print(f"{date_str:<12} {weekday:<6} {info['expected_seconds_in_day']:>10,} {info['missing_count']:>10,} "
                  f"{info['missing_percentage']:>9.4f}% {100 - info['missing_percentage']:>9.4f}%")

def group_consecutive_times(timestamps):
    """
    将连续的时间戳分组
    """
    if not timestamps:
        return []
    
    groups = []
    current_group = [timestamps[0]]
    
    for i in range(1, len(timestamps)):
        # 如果当前时间与前一个时间相差1秒，加入当前组
        if (timestamps[i] - timestamps[i-1]).total_seconds() == 1:
            current_group.append(timestamps[i])
        else:
            # 开始新组
            groups.append(current_group)
            current_group = [timestamps[i]]
    
    # 添加最后一组
    groups.append(current_group)
    
    return groups

def get_weekday_chinese(weekday):
    """
    获取中文星期名称
    """
    weekdays = ['一', '二', '三', '四', '五', '六', '日']
    return weekdays[weekday]

def save_missing_report_to_csv(result, output_dir='时间戳缺失分析结果'):
    """
    将缺失时间戳分析结果保存到CSV文件
    """
    if not result:
        print("没有结果需要保存")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每日缺失统计
    daily_data = []
    for date_str, info in sorted(result['daily_missing'].items()):
        daily_data.append({
            '日期': date_str,
            '星期': get_weekday_chinese(info['date'].weekday()),
            '预期时间点数': info['expected_seconds_in_day'],
            '缺失时间点数': info['missing_count'],
            '缺失比例(%)': round(info['missing_percentage'], 4),
            '数据完整度(%)': round(100 - info['missing_percentage'], 4)
        })
    
    daily_df = pd.DataFrame(daily_data)
    daily_file = os.path.join(output_dir, '每日缺失统计.csv')
    daily_df.to_csv(daily_file, index=False, encoding='utf-8-sig')
    print(f"已保存每日统计到: {daily_file}")
    
    # 保存所有缺失时间戳详单
    if result['missing_timestamps']:
        missing_detail_data = []
        for timestamp in result['missing_timestamps']:
            missing_detail_data.append({
                '缺失时间戳': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                '日期': timestamp.strftime('%Y-%m-%d'),
                '时间': timestamp.strftime('%H:%M:%S'),
                '星期': get_weekday_chinese(timestamp.weekday())
            })
        
        missing_df = pd.DataFrame(missing_detail_data)
        missing_file = os.path.join(output_dir, '缺失时间戳详单.csv')
        missing_df.to_csv(missing_file, index=False, encoding='utf-8-sig')
        print(f"已保存 {len(result['missing_timestamps'])} 个缺失时间戳详单到: {missing_file}")
    
    # 保存每日缺失详情
    for date_str, info in result['daily_missing'].items():
        if info['missing_count'] > 0:
            detail_data = []
            for timestamp in info['missing_times']:
                detail_data.append({
                    '缺失时间': timestamp.strftime('%H:%M:%S'),
                    '完整时间戳': timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            if detail_data:
                detail_df = pd.DataFrame(detail_data)
                detail_file = os.path.join(output_dir, f'{date_str}_缺失详情.csv')
                detail_df.to_csv(detail_file, index=False, encoding='utf-8-sig')
                print(f"已保存 {date_str} 的 {len(detail_data)} 个缺失时间点到: {detail_file}")

def analyze_consecutive_gaps(missing_timestamps):
    """
    分析连续缺失的时间段
    """
    if not missing_timestamps:
        return
    
    gaps = []
    current_start = missing_timestamps[0]
    current_end = missing_timestamps[0]
    
    for i in range(1, len(missing_timestamps)):
        prev_time = missing_timestamps[i-1]
        curr_time = missing_timestamps[i]
        
        # 如果当前时间与前一个时间相差1秒，说明是连续的
        if (curr_time - prev_time).total_seconds() == 1:
            current_end = curr_time
        else:
            # 连续段结束，记录当前段
            if current_start == current_end:
                gaps.append({'start': current_start, 'end': current_end, 'count': 1})
            else:
                count = int((current_end - current_start).total_seconds()) + 1
                gaps.append({'start': current_start, 'end': current_end, 'count': count})
            
            # 开始新的连续段
            current_start = curr_time
            current_end = curr_time
    
    # 处理最后一个段
    if current_start == current_end:
        gaps.append({'start': current_start, 'end': current_end, 'count': 1})
    else:
        count = int((current_end - current_start).total_seconds()) + 1
        gaps.append({'start': current_start, 'end': current_end, 'count': count})
    
    # 按缺失数量排序
    gaps.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"\n连续缺失时间段分析 (前20个最大缺失段):")
    print("-" * 80)
    for i, gap in enumerate(gaps[:20], 1):
        if gap['count'] == 1:
            print(f"{i:2d}. {gap['start'].strftime('%Y-%m-%d %H:%M:%S')} (缺失1个时间点)")
        else:
            duration = gap['end'] - gap['start']
            print(f"{i:2d}. {gap['start'].strftime('%Y-%m-%d %H:%M:%S')} 到 {gap['end'].strftime('%Y-%m-%d %H:%M:%S')} "
                  f"(连续缺失{gap['count']}个时间点，持续{duration})")

def main():
    """
    主函数
    """
    print("时间戳缺失检测工具 (增强版)")
    print("="*50)
    
    file_path = "存纸架数据汇总.csv"
    
    try:
        print(f"\n正在检测文件: {file_path}")
        
        # 执行时间戳缺失检测
        result = check_timestamp_gaps(file_path, time_column='时间', interval_seconds=1)
        
        if result:
            # 打印详细报告
            print_missing_report(result)
            
            # 分析连续缺失段
            if result['missing_timestamps']:
                analyze_consecutive_gaps(result['missing_timestamps'])
            
            # 保存结果到CSV文件
            save_missing_report_to_csv(result)
            
            print(f"\n" + "="*80)
            print("✅ 分析完成！详细结果已保存到 '时间戳缺失分析结果' 目录中")
            print("📊 包含：每日缺失统计表、缺失时间戳详单、每日缺失详情文件")
            print("="*80)
        else:
            print("❌ 分析失败")
            
    except FileNotFoundError:
        print(f"❌ 文件 {file_path} 未找到")
    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")

if __name__ == "__main__":
    main() 