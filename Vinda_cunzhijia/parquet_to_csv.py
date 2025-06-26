import os
import pandas as pd
from pathlib import Path

def parquet_to_csv(input_folder, output_folder):
    """
    将指定文件夹下的所有.parquet文件转换为.csv文件
    :param input_folder: 输入文件夹路径（包含.parquet文件）
    :param output_folder: 输出文件夹路径（保存.csv文件）
    """
    # 确保输出文件夹存在
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 遍历输入文件夹中的所有.parquet文件
    for file in os.listdir(input_folder):
        if file.endswith('.parquet'):
            # 构造完整文件路径
            parquet_path = os.path.join(input_folder, file)
            csv_filename = os.path.splitext(file)[0] + '.csv'
            csv_path = os.path.join(output_folder, csv_filename)
            
            # 读取Parquet文件并转换为CSV
            try:
                df = pd.read_parquet(parquet_path)  # 默认使用pyarrow引擎
                df.to_csv(csv_path, index=False)      # 不保存行索引
                print(f"转换成功: {file} -> {csv_filename}")
            except Exception as e:
                print(f"转换失败: {file} | 错误: {e}")

# 示例调用
input_folder = r"D:\Code_File\Vinda_cunzhijia\2025-05-02\2025-05-02"  # 替换为你的输入文件夹路径
output_folder = r"D:\Code_File\Vinda_cunzhijia\csv_file"    # 替换为输出文件夹路径
parquet_to_csv(input_folder, output_folder)