import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载数据文件"""
    print("正在加载数据文件...")
    
    # 加载最终结果CSV
    try:
        result_df = pd.read_csv('折叠机正常运行且高存纸率时间段_最终结果.csv')
        result_df['开始时间'] = pd.to_datetime(result_df['开始时间'])
        result_df['结束时间'] = pd.to_datetime(result_df['结束时间'])
        print(f"成功加载最终结果文件，共 {len(result_df)} 个时间段")
    except Exception as e:
        print(f"加载最终结果文件失败: {e}")
        return None, None
    
    # 加载汇总数据CSV
    try:
        # 尝试不同编码
        try:
            summary_df = pd.read_csv('存纸架数据汇总.csv', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                summary_df = pd.read_csv('存纸架数据汇总.csv', encoding='gbk')
            except UnicodeDecodeError:
                summary_df = pd.read_csv('存纸架数据汇总.csv', encoding='gb2312')
        
        summary_df['时间'] = pd.to_datetime(summary_df['时间'])
        print(f"成功加载汇总数据文件，共 {len(summary_df)} 行数据")
        
        # 检查所需列是否存在
        required_columns = ['折叠机出包数', '外循环进内循环纸条数量', '存纸率']
        missing_columns = [col for col in required_columns if col not in summary_df.columns]
        if missing_columns:
            print(f"警告：以下列在汇总数据中未找到: {missing_columns}")
            print(f"可用列名: {summary_df.columns.tolist()}")
        
        # 将相关列转换为数值型
        for col in ['折叠机出包数', '外循环进内循环纸条数量', '存纸率']:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
        
        return result_df, summary_df
        
    except Exception as e:
        print(f"加载汇总数据文件失败: {e}")
        return result_df, None

def get_closest_data_point(summary_df, target_time, tolerance_minutes=5):
    """获取最接近目标时间的数据点"""
    # 计算时间差
    time_diff = abs(summary_df['时间'] - target_time)
    
    # 找到最小时间差的索引
    min_idx = time_diff.idxmin()
    min_diff_minutes = time_diff.loc[min_idx].total_seconds() / 60
    
    # 如果时间差超过容忍范围，返回None
    if min_diff_minutes > tolerance_minutes:
        return None
    
    return summary_df.loc[min_idx]

def calculate_period_changes(result_df, summary_df):
    """计算每个时间段的数值变化"""
    
    print("开始计算时间段数值变化...")
    
    results = []
    
    for index, row in result_df.iterrows():
        try:
            start_time = row['开始时间']
            end_time = row['结束时间']
            
            print(f"处理时间段 {index+1}/{len(result_df)}: {start_time} - {end_time}")
            
            # 获取时间段内的所有数据
            period_mask = (summary_df['时间'] >= start_time) & (summary_df['时间'] <= end_time)
            period_data = summary_df[period_mask].copy()
            
            if len(period_data) == 0:
                print(f"  警告：时间段 {index+1} 没有找到对应的数据")
                continue
            
            # 获取开始时间和结束时间的数据点
            start_data = get_closest_data_point(summary_df, start_time)
            end_data = get_closest_data_point(summary_df, end_time)
            
            if start_data is None or end_data is None:
                print(f"  警告：时间段 {index+1} 无法找到开始或结束时间的数据点")
                continue
            
            # 计算各项数值
            
            # 1. s1: 折叠机出包数的变化量（结束时间-开始时间）
            s1 = end_data['折叠机出包数'] - start_data['折叠机出包数']
            
            # 2. s2: 该时间段内外循环进内循环的纸条数量累积量
            # 计算期间的累积量（排除开始时间点，包含结束时间点）
            period_data_for_sum = period_data[period_data['时间'] > start_time]
            s2 = period_data_for_sum['外循环进内循环纸条数量'].sum()
            
            # 3. 存纸架外循环进内循环的变化 = (s1/25) - s2
            storage_change = (s1 / 25) - s2
            
            # 4. 存纸率变化 = 结束时间的存纸率 - 开始时间的存纸率
            paper_rate_change = end_data['存纸率'] - start_data['存纸率']
            
            # 5. 差值 = (3) - (4)
            difference = storage_change - paper_rate_change
            
            # 保存结果
            results.append({
                '开始时间': start_time,
                '结束时间': end_time,
                '存纸架外循环进内循环的变化': storage_change,
                '存纸率变化': paper_rate_change,
                '差值': difference
            })
            
            print(f"  s1(折叠机出包数变化): {s1:.2f}")
            print(f"  s2(纸条数量累积): {s2:.2f}")
            print(f"  存纸架外循环进内循环的变化: {storage_change:.4f}")
            print(f"  存纸率变化: {paper_rate_change:.4f}")
            print(f"  差值: {difference:.4f}")
            
        except Exception as e:
            print(f"  错误：处理时间段 {index+1} 时出现异常: {e}")
            continue
    
    return results

def save_results(results):
    """保存计算结果到CSV文件"""
    if not results:
        print("没有计算结果可保存")
        return
    
    # 创建DataFrame
    result_df = pd.DataFrame(results)
    
    # 保存到CSV文件
    output_file = '时间段数值变化分析结果.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存到: {output_file}")
    print(f"共计算了 {len(results)} 个时间段的数值变化")
    
    # 显示前几行结果
    print("\n前5个结果预览:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(result_df[['开始时间', '结束时间', '存纸架外循环进内循环的变化', '存纸率变化', '差值']].head().to_string(index=False))
    
    # 统计信息
    print(f"\n=== 统计信息 ===")
    print(f"存纸架外循环进内循环的变化:")
    print(f"  平均值: {result_df['存纸架外循环进内循环的变化'].mean():.4f}")
    print(f"  最大值: {result_df['存纸架外循环进内循环的变化'].max():.4f}")
    print(f"  最小值: {result_df['存纸架外循环进内循环的变化'].min():.4f}")
    
    print(f"\n存纸率变化:")
    print(f"  平均值: {result_df['存纸率变化'].mean():.4f}")
    print(f"  最大值: {result_df['存纸率变化'].max():.4f}")
    print(f"  最小值: {result_df['存纸率变化'].min():.4f}")
    
    print(f"\n差值:")
    print(f"  平均值: {result_df['差值'].mean():.4f}")
    print(f"  最大值: {result_df['差值'].max():.4f}")
    print(f"  最小值: {result_df['差值'].min():.4f}")
    
    # 统计'存纸架外循环进内循环的变化'的正负值比例
    storage_changes = result_df['存纸架外循环进内循环的变化']
    total_count = len(storage_changes)
    positive_count = (storage_changes > 0).sum()
    negative_count = (storage_changes < 0).sum()
    zero_count = (storage_changes == 0).sum()
    
    print(f"\n=== '存纸架外循环进内循环的变化'正负值分布 ===")
    print(f"总数量: {total_count}")
    print(f"正值: {positive_count} 个 ({positive_count/total_count*100:.2f}%)")
    print(f"负值: {negative_count} 个 ({negative_count/total_count*100:.2f}%)")
    print(f"零值: {zero_count} 个 ({zero_count/total_count*100:.2f}%)")
    
    if positive_count > 0:
        print(f"正值平均: {storage_changes[storage_changes > 0].mean():.4f}")
        print(f"正值最大: {storage_changes[storage_changes > 0].max():.4f}")
    
    if negative_count > 0:
        print(f"负值平均: {storage_changes[storage_changes < 0].mean():.4f}")
        print(f"负值最小: {storage_changes[storage_changes < 0].min():.4f}")

def main():
    """主函数"""
    print("=== 时间段数值变化分析程序 ===")
    print("功能：计算各时间段内的数值变化")
    print("计算项目：")
    print("1. s1 = 折叠机出包数变化量（结束-开始）")
    print("2. s2 = 外循环进内循环纸条数量累积量")
    print("3. 存纸架外循环进内循环的变化 = (s1/25) - s2")
    print("4. 存纸率变化 = 结束存纸率 - 开始存纸率")
    print("5. 差值 = (3) - (4)")
    print("="*60)
    
    # 加载数据
    result_df, summary_df = load_data()
    
    if result_df is None or summary_df is None:
        print("数据加载失败，程序退出")
        return
    
    # 计算数值变化
    results = calculate_period_changes(result_df, summary_df)
    
    # 保存结果
    save_results(results)

if __name__ == "__main__":
    main() 