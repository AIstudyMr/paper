import os
import pandas as pd
from pathlib import Path

def replace_bool_by_index(df, column_index):
    """通过列序号替换布尔值"""
    if 0 <= column_index < len(df.columns):
        col_name = df.columns[column_index]
        # 统一处理字符串/布尔值（不区分大小写）
        df[col_name] = df[col_name].replace(
            {'false': 0, 'true': 1, 'False': 0, 'True': 1, False: 0, True: 1}
        )
    return df

def process_csv_file(input_path, output_folder, column_index):
    """处理单个CSV文件"""
    try:
        # 读取CSV（自动处理编码和标题行）
        df = pd.read_csv(input_path)
        
        # 替换指定列的布尔值
        df_processed = replace_bool_by_index(df, column_index)
        
        # 保存到输出文件夹
        output_path = Path(output_folder) / Path(input_path).name
        df_processed.to_csv(output_path, index=False)
        print(f"处理成功: {input_path} -> 列 {column_index}")
        
    except Exception as e:
        print(f"处理失败 {input_path}: {str(e)}")

def batch_process_csv(input_folder, output_folder, column_index):
    """批量处理所有CSV文件"""
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 遍历所有CSV文件
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):
            input_path = os.path.join(input_folder, file)
            process_csv_file(input_path, output_folder, column_index)

# 使用示例
if __name__ == "__main__":
    # 配置路径（使用原始字符串避免转义问题）
    input_folder = r"D:\Code_File\Vinda_cunzhijia\csv_file_name_norm"       # 原始CSV文件夹
    output_folder = r"D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1"   # 输出文件夹
    target_column_index = 1                # 要处理的列序号（从0开始）
    
    # 执行批量处理
    batch_process_csv(input_folder, output_folder, target_column_index)
    print("\n处理完成！输出文件保存在:", output_folder)