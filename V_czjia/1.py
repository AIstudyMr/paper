import os
import pandas as pd

def get_csv_paths(folder_path):
    """获取文件夹下所有CSV文件路径"""
    csv_paths = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            csv_paths.append(os.path.join(folder_path, file))
    return csv_paths

def read_csv_columns(folder_path):
    """读取文件夹下所有CSV文件的列名称"""
    csv_paths = get_csv_paths(folder_path)
    print(f"找到 {len(csv_paths)} 个CSV文件：")
    
    for path in csv_paths:
        try:
            # 尝试读取CSV文件，假设编码为'gbk'，如果失败则尝试'utf-8'
            try:
                df = pd.read_csv(path, encoding='gbk')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='utf-8')
            
            columns = df.columns.tolist()
            print(f"\n文件路径: {path}")
            print("列名称:")
            for col in columns:
                print(f"  - {col}")
        except Exception as e:
            print(f"\n无法读取文件 {path}，错误信息: {e}")

# 使用示例
folder = r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1'
read_csv_columns(folder)