import pandas as pd
import numpy as np
import os, shutil
# 定义文件路径和对应的列及新列名


    
file_configs = [
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\1#小包机-入包数.csv',  
        'columns': ['ts', '_196_D0102'],  
        'new_columns': ['时间', '1#小包机入包数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\1#小包机-实际速度_主机速度.csv',  
        'columns': ['_196_D0050'],  
        'new_columns': [ '1#小包机实际速度']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\1#小包机包装胶膜用完.csv',  
        'columns': ['_196_M1251'],  
        'new_columns': ['1#小包机包装胶膜用完']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\1#有效切数.csv',  
        'columns': ['_192_D0300'],  
        'new_columns': ['1#有效切数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\1号装箱机停机.csv',  
        'columns': ['_249_000010'],  
        'new_columns': ['1号装箱机停机']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\1号装箱机出包数.csv',  
        'columns': ['_249_404001'],  
        'new_columns': ['1号装箱机出包数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\1号装箱机实际速度.csv',  
        'columns': ['_249_400101'],  
        'new_columns': ['1号装箱机实际速度']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\1号装箱机待机.csv',  
        'columns': ['_249_040136'],  
        'new_columns': ['1号装箱机待机']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\1号装箱机故障.csv',  
        'columns': ['_249_000009'],  
        'new_columns': ['1号装箱机故障']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\2#小包机-入包数（人工捡入）.csv',  
        'columns': ['_199_D0108'],  
        'new_columns': ['2#小包机入包数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\2#小包机-实际速度.csv',  
        'columns': ['_199_D0050'],  
        'new_columns': ['2#小包机实际速度']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\2#小包机包装胶膜用完.csv',  
        'columns': ['_199_M1251'],  
        'new_columns': ['2#小包机包装胶膜用完']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\2#有效切数.csv',  
        'columns': ['_192_D0302'],  
        'new_columns': ['2#有效切数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\2号装箱机停机.csv',  
        'columns': ['_251_000010'],  
        'new_columns': ['2号装箱机停机']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\2号装箱机出包数.csv',  
        'columns': ['_251_404001'],  
        'new_columns': ['2号装箱机出包数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\2号装箱机实际速度.csv',  
        'columns': ['_251_400101'],  
        'new_columns': ['2号装箱机实际速度']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\2号装箱机待机.csv',  
        'columns': ['_251_040136'],  
        'new_columns': ['2号装箱机待机']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\2号装箱机故障.csv',  
        'columns': ['_251_000009'],  
        'new_columns': ['2号装箱机故障']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\3#小包机-入包数.csv',  
        'columns': ['_202_D0102'],  
        'new_columns': ['3#小包机入包数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\3#小包机-实际速度_主机速度.csv',  
        'columns': ['_202_D0050'],  
        'new_columns': ['3#小包机主机实际速度']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\3#小包机包装胶膜用完.csv',  
        'columns': ['_202_M1251'],  
        'new_columns': ['3#小包机包装胶膜用完']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\3#有效切数.csv',  
        'columns': ['_192_D0304'],  
        'new_columns': ['3#有效切数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\4#小包机-入包数.csv',  
        'columns': ['_205_D0102'],  
        'new_columns': ['4#小包机入包数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\4#小包机-实际速度_主机速度.csv',  
        'columns': ['_205_D0050'],  
        'new_columns': ['4#小包机主机实际速度']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\4#小包机包装胶膜用完.csv',  
        'columns': ['_205_M1251'],  
        'new_columns': ['4#小包机包装胶膜用完']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\4#有效切数.csv',  
        'columns': ['_192_D0306'],  
        'new_columns': ['4#有效切数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\外循环进内循环纸条数量计数.csv',  
        'columns': ['_190_X00A5'],  
        'new_columns': ['外循环进内循环纸条数量']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\存包率.csv',  
        'columns': ['_208_D0612'],  
        'new_columns': ['存包率']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\存纸率.csv',  
        'columns': ['_186_D0962'],  
        'new_columns': ['存纸率']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架1-原纸剩余米数（m） .csv',  
        'columns': ['_150_D20316'],  
        'new_columns': ['尾架1原纸剩余米数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架1-原纸外径（m，不包含辊轴）.csv',  
        'columns': ['_186_D0565'],  
        'new_columns': ['尾架1原纸外径']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架2-原纸剩余米数（m）.csv',  
        'columns': ['_150_D20318'],  
        'new_columns': ['尾架2原纸剩余米数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架2-原纸外径（m，不包含辊轴） .csv',  
        'columns': ['_186_D0575'],  
        'new_columns': ['尾架2原纸外径']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架3-原纸剩余米数（m）.csv',  
        'columns': ['_150_D20320'],  
        'new_columns': ['尾架3原纸剩余米数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架3-原纸外径（m，不包含辊轴） .csv',  
        'columns': ['_186_D0585'],  
        'new_columns': ['尾架3原纸外径']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架5-原纸剩余米数（m）.csv',  
        'columns': ['_150_D20322'],  
        'new_columns': ['尾架5原纸剩余米数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架5-原纸外径（m，不包含辊轴） .csv',  
        'columns': ['_186_D1120'],  
        'new_columns': ['尾架5原纸外径']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架6-原纸剩余米数（m）.csv',  
        'columns': ['_150_D20324'],  
        'new_columns': ['尾架6原纸剩余米数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架6-原纸外径（m，不包含辊轴） .csv',  
        'columns': ['_186_D1130'],  
        'new_columns': ['尾架6原纸外径']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架7-原纸剩余米数（m）.csv',  
        'columns': ['_150_D20326'],  
        'new_columns': ['尾架7原纸剩余米数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\尾架7-原纸外径（m，不包含辊轴） .csv',  
        'columns': ['_186_D1140'],  
        'new_columns': ['尾架7原纸外径']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\折叠机入包数.csv',  
        'columns': ['_186_D1544'],  
        'new_columns': ['折叠机入包数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\折叠机出包数.csv',  
        'columns': ['_190_D1710'],  
        'new_columns': ['折叠机出包数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\折叠机实际速度 .csv',  
        'columns': ['_186_D2001'],  
        'new_columns': ['折叠机实际速度']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\有效总切数.csv',  
        'columns': ['_192_D10056'],  
        'new_columns': ['有效总切数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\裁切机-实际速度.csv',  
        'columns': ['_192_D3100'],  
        'new_columns': ['裁切机实际速度']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\进第一裁切通道纸条计数.csv',  
        'columns': ['_190_X006D'],  
        'new_columns': ['进第一裁切通道纸条计数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\进第三裁切通道纸条计数.csv',  
        'columns': ['_190_X008D'],  
        'new_columns': ['进第三裁切通道纸条计数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\进第二裁切通道纸条计数.csv',  
        'columns': ['_190_X007D'],  
        'new_columns': ['进第二裁切通道纸条计数']  
    },
    {
        'file_path': r'D:\Code_File\Vinda_cunzhijia\csv_file_name_norm_1\进第四裁切通道纸条计数.csv',  
        'columns': ['_190_X009D'],  
        'new_columns': ['进第四裁切通道纸条计数']  
    }
    ]

# 用于存储每个文件的 DataFrame
dataframes = []

# 遍历每个文件配置
for config in file_configs:
    # 读取 CSV 文件
    df = pd.read_csv(config['file_path'],encoding='gbk')
    
    # 选择指定列
    selected_columns = df[config['columns']]
    
    # 修改列名称
    selected_columns.columns = config['new_columns']
    
    # 添加到 DataFrame 列表
    dataframes.append(selected_columns)

# 按行拼接所有 DataFrame
merged_df = pd.concat(dataframes, axis=1)


output_path = r'D:\Code_File\Vinda_cunzhijia'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 保存拼接后的 DataFrame 到新的 CSV 文件

merged_df.to_csv(os.path.join(output_path, '存纸架数据汇总.csv'), index=False,encoding='utf-8')

print(f"文件已成功拼接并保存到 {output_path}")
