import pandas as pd
import numpy as np
import os
from 数据分析处理 import read_csv_with_encoding, process_data_for_time_period, calculate_compound_difference, save_compound_difference_analysis

def analyze_compound_difference():
    """执行复合差值分析：(折叠机出包数 - 外循环进内循环纸条数量) × 1.37 - 存纸率"""
    print("=== 复合差值分析：(折叠机出包数 - 外循环进内循环纸条数量) × 1.37 - 存纸率 ===")
    
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
        
        # 创建输出目录
        compound_output_dir = "复合差值分析结果"
        
        # 存储所有时间段的复合差值分析结果
        compound_results = []
        
        # 处理每个时间段
        for idx, row in time_periods_df.iterrows():
            start_time = pd.to_datetime(row['开始时间'])
            end_time = pd.to_datetime(row['结束时间'])
            
            print(f"\n处理时间段 {idx+1}/{len(time_periods_df)}: {start_time} 到 {end_time}")
            
            # 处理数据
            result = process_data_for_time_period(summary_df, start_time, end_time)
            
            if result is not None:
                data_dict, time_index = result
                
                # 创建时间段标识
                period_name = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}"
                
                # 进行复合差值分析
                compound_result = calculate_compound_difference(data_dict, time_index, period_name)
                
                if compound_result:
                    compound_results.append(compound_result)
                    print(f"  -> 正差值: {compound_result['positive_count']}/{compound_result['total_count']} ({compound_result['positive_ratio']:.1f}%)")
                    print(f"  -> 负差值: {compound_result['negative_count']}/{compound_result['total_count']} ({compound_result['negative_ratio']:.1f}%)")
                    print(f"  -> 平均值: {compound_result['mean_difference']:.4f}")
            else:
                print(f"跳过时间段 {idx+1}，无数据")
        
        # 保存分析结果
        if compound_results:
            save_compound_difference_analysis(compound_results, compound_output_dir)
            return True
        else:
            print("没有成功的复合差值分析结果")
            return False
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=== 复合差值分析工具 ===")
    print("计算公式: (折叠机出包数 - 外循环进内循环纸条数量) × 1.37 - 存纸率")
    print("分析步骤:")
    print("1. 计算 折叠机出包数 - 外循环进内循环纸条数量")
    print("2. 将结果乘以 1.37")
    print("3. 用步骤2的结果减去存纸率")
    print("4. 统计最终结果的正负分布")
    
    # 执行复合差值分析
    success = analyze_compound_difference()
    if success:
        print(f"\n✅ 复合差值分析完成！结果已保存到 '复合差值分析结果' 文件夹")
    else:
        print(f"\n❌ 复合差值分析失败")

if __name__ == "__main__":
    main() 