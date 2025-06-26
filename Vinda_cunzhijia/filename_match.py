import os
import shutil
import pandas as pd
from pathlib import Path

def build_mapping_dict(mapping_file):
    """从映射文件中构建文件名映射字典"""
    df = pd.read_csv(mapping_file, header=None,encoding='gbk')
    return dict(zip(df[1], df[0]))  # {英文名: 中文名}

def process_csv_files(input_folder, output_folder, mapping_dict):
    """处理文件并保存到新文件夹"""
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            # 提取基础文件名（不含扩展名）
            base_name = os.path.splitext(filename)[0]
            
            # 在映射字典中查找对应的中文名
            if base_name in mapping_dict:
                chinese_name = mapping_dict[base_name]
                new_filename = f"{chinese_name}.csv"
                
                # 构建完整路径
                old_path = os.path.join(input_folder, filename)
                new_path = os.path.join(output_folder, new_filename)
                
                # 复制文件到新位置（使用新名称）
                try:
                    shutil.copy2(old_path, new_path)
                    print(f"处理成功: {filename} -> {new_filename}")
                    processed_count += 1
                except Exception as e:
                    print(f"处理失败 {filename}: {str(e)}")
            else:
                print(f"未找到匹配: {filename} - 跳过")
    
    print(f"\n完成！共处理 {processed_count} 个文件")

# 使用示例
if __name__ == "__main__":
    # 配置路径
    mapping_file = r"D:\Code_File\Vinda_cunzhijia\22#线与存纸架满架分析相关点位.csv"  # 你的映射文件路径
    input_folder = r"D:\Code_File\Vinda_cunzhijia\csv_file_time_norm"  # 需要重命名的文件夹
    output_folder = r"D:\Code_File\Vinda_cunzhijia\csv_file_file_norm"  # 新文件夹路径

     # 构建映射字典
    name_mapping = build_mapping_dict(mapping_file)
    
    # 执行处理
    process_csv_files(input_folder, output_folder, name_mapping)