import pandas as pd
import numpy as np
import os
from 数据分析处理 import read_csv_with_encoding, process_data_for_time_period, calculate_column_difference, save_difference_analysis, save_overall_statistics

def analyze_column_difference(col1, col2):
    """分析指定两列的差值"""
    print(f"=== 开始差值分析：{col1} - {col2} ===")
    
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
        difference_output_dir = "差值分析结果"
        
        # 存储所有时间段的差值分析结果
        difference_results = []
        
        # 总体统计变量
        total_positive_count = 0
        total_negative_count = 0
        total_zero_count = 0
        total_data_points = 0
        
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
                
                # 进行差值分析
                diff_result = calculate_column_difference(
                    data_dict, time_index, col1, col2, period_name
                )
                
                if diff_result:
                    difference_results.append(diff_result)
                    
                    # 累计总体统计数据
                    total_positive_count += diff_result['positive_count']
                    total_negative_count += diff_result['negative_count']
                    total_zero_count += diff_result['zero_count']
                    total_data_points += diff_result['total_count']
                    
                    print(f"  -> 正差值: {diff_result['positive_count']}/{diff_result['total_count']} ({diff_result['positive_ratio']:.1f}%)")
                    print(f"  -> 负差值: {diff_result['negative_count']}/{diff_result['total_count']} ({diff_result['negative_ratio']:.1f}%)")
            else:
                print(f"跳过时间段 {idx+1}，无数据")
        
        # 输出总体统计结果
        if total_data_points > 0:
            total_positive_ratio = (total_positive_count / total_data_points) * 100
            total_negative_ratio = (total_negative_count / total_data_points) * 100
            total_zero_ratio = (total_zero_count / total_data_points) * 100
            
            print("\n" + "="*60)
            print(f"🎯 总体统计结果：{col1} - {col2}")
            print("="*60)
            print(f"总数据点数: {total_data_points:,}")
            print(f"正差值: {total_positive_count:,} 个 ({total_positive_ratio:.2f}%)")
            print(f"负差值: {total_negative_count:,} 个 ({total_negative_ratio:.2f}%)")
            print(f"零差值: {total_zero_count:,} 个 ({total_zero_ratio:.2f}%)")
            print("="*60)
            
            # 判断整体趋势
            if total_positive_ratio > total_negative_ratio:
                trend = f"总体趋势：{col1} > {col2} (正差值占主导)"
            elif total_negative_ratio > total_positive_ratio:
                trend = f"总体趋势：{col1} < {col2} (负差值占主导)"
            else:
                trend = f"总体趋势：{col1} ≈ {col2} (正负差值基本相等)"
            print(f"📊 {trend}")
            print("="*60)
            
            # 计算加权平均差值
            total_weighted_sum = sum(result['mean_difference'] * result['total_count'] for result in difference_results)
            average_difference = total_weighted_sum / total_data_points if total_data_points > 0 else 0
            
            # 保存总体统计结果到CSV
            save_overall_statistics(col1, col2, total_data_points, total_positive_count, 
                                   total_negative_count, total_zero_count, 
                                   total_positive_ratio, total_negative_ratio, total_zero_ratio,
                                   average_difference, trend, difference_output_dir)
        
        # 保存分析结果
        if difference_results:
            save_difference_analysis(difference_results, col1, col2, difference_output_dir)
            return True
        else:
            print("没有成功的差值分析结果")
            return False
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def list_available_columns():
    """列出所有可用的列名"""
    try:
        # 读取一小部分数据来获取列名
        summary_df = read_csv_with_encoding("存纸架数据汇总.csv")
        
        # 定义可能用于分析的列
        analysis_columns = [
            '折叠机实际速度', '折叠机入包数', '折叠机出包数', '外循环进内循环纸条数量', '存纸率',
            '裁切机实际速度', '有效总切数', '1#有效切数', '2#有效切数', '3#有效切数', '4#有效切数',
            '进第一裁切通道纸条计数', '进第二裁切通道纸条计数', '进第三裁切通道纸条计数', '进第四裁切通道纸条计数',
            '1#小包机入包数', '1#小包机实际速度', '2#小包机入包数', '2#小包机实际速度',
            '3#小包机入包数', '3#小包机主机实际速度', '4#小包机入包数', '4#小包机主机实际速度',
            '小包机速度总和'
        ]
        
        # 过滤出存在的列
        available_cols = [col for col in analysis_columns if col in summary_df.columns or col == '小包机速度总和']
        
        print("可用于分析的列名:")
        for i, col in enumerate(available_cols, 1):
            print(f"{i:2d}. {col}")
            
        return available_cols
        
    except Exception as e:
        print(f"获取列名时发生错误: {e}")
        return []

def main():
    """主函数"""
    # ==============================================
    # 在这里直接修改要分析的两列名称
    # ==============================================
    COL1 = "进第四裁切通道纸条计数"      # 第一列名称
    COL2 = "4#有效切数"      # 第二列名称
    # ==============================================
    
    print("=== 差值分析工具 ===")
    print(f"分析列: {COL1} - {COL2}")
    
    # 检查列名是否有效
    available_columns = list_available_columns()
    if not available_columns:
        print("错误：无法获取可用列名")
        return
        
    if COL1 not in available_columns:
        print(f"错误：列名 '{COL1}' 不在可用列表中")
        print("可用的列名:")
        for i, col in enumerate(available_columns, 1):
            print(f"{i:2d}. {col}")
        return
        
    if COL2 not in available_columns:
        print(f"错误：列名 '{COL2}' 不在可用列表中")
        print("可用的列名:")
        for i, col in enumerate(available_columns, 1):
            print(f"{i:2d}. {col}")
        return
    
    # 执行差值分析
    success = analyze_column_difference(COL1, COL2)
    if success:
        print(f"\n✅ 差值分析完成！结果已保存到 '差值分析结果' 文件夹")
    else:
        print(f"\n❌ 差值分析失败")

if __name__ == "__main__":
    main() 