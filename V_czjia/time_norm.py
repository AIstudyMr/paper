import os
import pandas as pd
from pathlib import Path

def process_csv_file(input_path, output_folder, process_func):
    """
    处理单个CSV文件并保存到输出文件夹
    :param input_path: 输入文件路径
    :param output_folder: 输出文件夹路径
    :param process_func: 自定义处理函数
    """
    df = pd.read_csv(input_path)
    df_processed = process_func(df)
    output_path = Path(output_folder) / Path(input_path).name
    df_processed.to_csv(output_path, index=False)
    print(f"处理完成: {input_path} -> {output_path}")

def batch_process_csv(input_folder, output_folder, process_func):
    """
    批量处理文件夹下所有CSV文件
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param process_func: 自定义处理函数
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            input_path = os.path.join(input_folder, file)
            try:
                process_csv_file(input_path, output_folder, process_func)
            except Exception as e:
                print(f"处理失败: {file} | 错误: {e}")

def process_timestamp(df):
    """
    直接替换原时间列为秒级精度的datetime
    :param df: 输入的DataFrame
    :return: 处理后的DataFrame
    """
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], unit='ms').dt.floor('S')
    return df

# 调用示例
batch_process_csv(
    input_folder=r'D:\Code_File\Vinda_cunzhijia\csv_file',
    output_folder=r'D:\Code_File\Vinda_cunzhijia\csv_file_time_norm',
    process_func=process_timestamp  # 直接使用处理函数
)