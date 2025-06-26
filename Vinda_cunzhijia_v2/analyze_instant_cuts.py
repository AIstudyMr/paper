import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data():
    """读取CSV文件"""
    print("正在读取数据...")
    df = pd.read_csv('存纸架数据汇总.csv')
    df['时间'] = pd.to_datetime(df['时间'])
    print(f"数据时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
    return df

def calculate_instant_cuts(df):
    """计算每一秒的瞬时切数"""
    # 确保时间是按顺序排列的
    df = df.sort_values('时间')
    
    # 计算时间差（秒）
    df['time_diff'] = df['时间'].diff().dt.total_seconds()
    
    # 对每个小包机计算瞬时切数
    for machine_number in [1, 2, 3, 4]:
        # 获取累计切数列名
        cuts_col = f"{machine_number}#有效切数"
        
        # 计算切数差值
        df[f"{machine_number}#切数差值"] = df[cuts_col].diff()
        
        # 计算瞬时切数（每秒）
        df[f"{machine_number}#瞬时切数"] = df[f"{machine_number}#切数差值"] / df['time_diff']
        
        # 处理异常值（如除以0或负值）
        df[f"{machine_number}#瞬时切数"] = df[f"{machine_number}#瞬时切数"].replace([np.inf, -np.inf], np.nan)
        df[f"{machine_number}#瞬时切数"] = df[f"{machine_number}#瞬时切数"].fillna(0)
        
        # 移除负值（这些可能是由于数据记录问题导致的）
        df.loc[df[f"{machine_number}#瞬时切数"] < 0, f"{machine_number}#瞬时切数"] = 0
        
        # 移除临时列
        df = df.drop(f"{machine_number}#切数差值", axis=1)
    
    # 移除临时列
    df = df.drop('time_diff', axis=1)
    return df

def save_results(df):
    """保存结果到CSV文件"""
    # 对瞬时切数列进行四舍五入，保留2位小数
    for machine_number in [1, 2, 3, 4]:
        col = f"{machine_number}#瞬时切数"
        df[col] = df[col].round(2)
    
    # 重新排列列顺序
    # 首先获取所有列名
    all_columns = list(df.columns)
    
    # 创建新的列顺序
    new_order = ['时间']  # 时间列始终在最前
    
    # 为每台机器添加相关列
    for machine_number in [1, 2, 3, 4]:
        # 找出该机器的所有相关列
        machine_cols = [col for col in all_columns if col.startswith(f"{machine_number}#")]
        # 确保有效切数和瞬时切数排在一起
        cuts_cols = [f"{machine_number}#有效切数", f"{machine_number}#瞬时切数"]
        other_cols = [col for col in machine_cols if col not in cuts_cols]
        new_order.extend(cuts_cols)
        new_order.extend(other_cols)
    
    # 添加剩余的列
    remaining_cols = [col for col in all_columns if col not in new_order]
    new_order.extend(remaining_cols)
    
    # 重新排序列
    df = df[new_order]
    
    # 保存回原始文件
    df.to_csv('存纸架数据汇总.csv', index=False, encoding='utf-8-sig')
    print("\n数据已保存到存纸架数据汇总.csv文件中，列已重新排序")
    print("\n列顺序示例：")
    print("时间")
    for machine_number in [1, 2, 3, 4]:
        print(f"{machine_number}#有效切数")
        print(f"{machine_number}#瞬时切数")
        print(f"... 其他{machine_number}#相关列 ...")
    
    # 打印基础统计信息
    print("\n=== 瞬时切数统计 ===")
    stats = df[[f"{i}#瞬时切数" for i in [1, 2, 3, 4]]].describe()
    print(stats)
    
    # 计算每台机器的最大瞬时切数和平均瞬时切数
    for machine_number in [1, 2, 3, 4]:
        col = f"{machine_number}#瞬时切数"
        print(f"\n{machine_number}#小包机:")
        print(f"最大瞬时切数: {df[col].max():.2f} 个/秒")
        print(f"平均瞬时切数: {df[col].mean():.2f} 个/秒")
        print(f"有效切数记录数: {(df[col] > 0).sum()} 条")

def main():
    try:
        # 读取数据
        df = load_data()
        
        # 计算瞬时切数
        df_with_instant_cuts = calculate_instant_cuts(df)
        
        # 保存结果
        save_results(df_with_instant_cuts)
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 