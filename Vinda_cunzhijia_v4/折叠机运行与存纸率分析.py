import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def smooth_window(data, window_size=5):
    """
    使用滑动窗口对数据进行平滑处理
    """
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        if i < window_size // 2:
            # 开始部分，取前面的数据
            window_data = data[:i + window_size // 2 + 1]
        elif i >= len(data) - window_size // 2:
            # 结束部分，取后面的数据
            window_data = data[i - window_size // 2:]
        else:
            # 中间部分，取窗口大小的数据
            window_data = data[i - window_size // 2:i + window_size // 2 + 1]
        
        smoothed.append(np.mean(window_data))
    
    return smoothed

def find_continuous_periods(data, condition_column, threshold, min_duration_minutes=5):
    """
    找出满足条件的连续时间段
    """
    periods = []
    start_time = None
    data_reset = data.reset_index(drop=True)  # 重置索引以避免索引错误
    
    for i in range(len(data_reset)):
        row = data_reset.iloc[i]
        if row[condition_column] >= threshold:
            if start_time is None:
                start_time = row['时间']
        else:
            if start_time is not None:
                if i > 0:  # 确保索引有效
                    end_time = data_reset.iloc[i-1]['时间']
                    duration = end_time - start_time
                    
                    # 只保留持续时间大于最小时长的时间段
                    if duration.total_seconds() >= min_duration_minutes * 60:
                        periods.append({
                            '开始时间': start_time,
                            '结束时间': end_time,
                            '持续时间': duration
                        })
                start_time = None
    
    # 处理最后一个时间段
    if start_time is not None:
        end_time = data_reset.iloc[-1]['时间']
        duration = end_time - start_time
        if duration.total_seconds() >= min_duration_minutes * 60:
            periods.append({
                '开始时间': start_time,
                '结束时间': end_time,
                '持续时间': duration
            })
    
    return periods

def find_paper_storage_periods_in_normal_operation(data_file):
    """
    主函数：先找折叠机正常运行时间段，再在这些时间段内找存纸率>=5的时间段
    """
    print("正在读取数据...")
    
    # 读取数据
    try:
        # 尝试不同的编码方式
        try:
            df = pd.read_csv(data_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(data_file, encoding='gbk')
            except UnicodeDecodeError:
                df = pd.read_csv(data_file, encoding='gb2312')
        
        print(f"数据加载成功，共 {len(df)} 行数据")
        print("列名:", df.columns.tolist())
        
        # 转换时间列
        df['时间'] = pd.to_datetime(df['时间'])
        
        # 查找折叠机速度相关的列
        speed_columns = [col for col in df.columns if '折叠机' in col and ('速度' in col or '实际速度' in col)]
        print(f"找到折叠机速度列: {speed_columns}")
        
        if not speed_columns:
            print("错误：未找到折叠机速度相关列")
            return
        
        # 使用第一个找到的速度列
        speed_column = speed_columns[0]
        
        # 查找存纸率相关的列
        paper_storage_columns = [col for col in df.columns if '存纸' in col and '率' in col]
        print(f"找到存纸率列: {paper_storage_columns}")
        
        if not paper_storage_columns:
            print("错误：未找到存纸率相关列")
            return
        
        # 使用第一个找到的存纸率列
        paper_storage_column = paper_storage_columns[0]
        
        # 处理数据，将非数值转换为NaN
        df[speed_column] = pd.to_numeric(df[speed_column], errors='coerce')
        df[paper_storage_column] = pd.to_numeric(df[paper_storage_column], errors='coerce')
        
        # 去除空值
        df = df.dropna(subset=[speed_column, paper_storage_column])
        
        print(f"清理后数据行数: {len(df)}")
        print(f"折叠机速度范围: {df[speed_column].min():.2f} - {df[speed_column].max():.2f}")
        print(f"存纸率范围: {df[paper_storage_column].min():.2f} - {df[paper_storage_column].max():.2f}")
        
        # 第一步：找出折叠机正常运行（速度>=100）的时间段
        print("\n第一步：查找折叠机正常运行时间段（速度>=100）...")
        normal_operation_periods = find_continuous_periods(df, speed_column, 100, min_duration_minutes=5)
        
        print(f"找到 {len(normal_operation_periods)} 个折叠机正常运行时间段")
        
        # 第二步：在正常运行时间段内，找出存纸率>=5的时间段
        print("\n第二步：在正常运行时间段内查找存纸率>=5的时间段...")
        
        final_periods = []
        
        for period in normal_operation_periods:
            start_time = period['开始时间']
            end_time = period['结束时间']
            
            # 筛选出该时间段内的数据
            period_data = df[(df['时间'] >= start_time) & (df['时间'] <= end_time)].copy()
            
            if len(period_data) > 0:
                # 对存纸率数据进行平滑处理
                period_data['存纸率_平滑'] = smooth_window(period_data[paper_storage_column].values, window_size=5)
                
                # 在平滑后的数据中找出存纸率>=5的时间段
                high_storage_periods = find_continuous_periods(period_data, '存纸率_平滑', 1, min_duration_minutes=3)
                
                # 添加到最终结果
                final_periods.extend(high_storage_periods)
        
        print(f"最终找到 {len(final_periods)} 个同时满足条件的时间段")
        
        # 输出结果到CSV
        if final_periods:
            result_df = pd.DataFrame(final_periods)
            
            # 格式化持续时间为更易读的格式
            result_df['持续时间_格式化'] = result_df['持续时间'].apply(
                lambda x: f"{x.days} days {x.seconds//3600:02d}:{(x.seconds//60)%60:02d}:{x.seconds%60:02d}"
            )
            
            # 保存结果
            output_file = '折叠机正常运行且高存纸率时间段_最终结果_存纸率1.csv'
            result_df[['开始时间', '结束时间', '持续时间_格式化']].rename(
                columns={'持续时间_格式化': '持续时间'}
            ).to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"\n结果已保存到: {output_file}")
            print("\n前10个结果预览:")
            print(result_df[['开始时间', '结束时间', '持续时间_格式化']].head(10).to_string(index=False))
            
            # 统计信息
            total_duration = sum([period['持续时间'] for period in final_periods], timedelta())
            print(f"\n统计信息:")
            print(f"总时间段数量: {len(final_periods)}")
            print(f"总持续时间: {total_duration}")
            print(f"平均持续时间: {total_duration / len(final_periods) if final_periods else 0}")
            
        else:
            print("未找到同时满足条件的时间段")
    
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 数据文件路径
    data_file = "存纸架数据汇总.csv"
    
    print("=== 折叠机正常运行与高存纸率时间段分析 ===")
    print("分析目标：")
    print("1. 找出折叠机正常运行（速度>=100）的时间段")
    print("2. 在上述时间段内，找出存纸率>=5的时间段（使用平滑窗口）")
    print("3. 输出格式：CSV文件，包含开始时间、结束时间、持续时间")
    print("="*50)
    
    find_paper_storage_periods_in_normal_operation(data_file) 