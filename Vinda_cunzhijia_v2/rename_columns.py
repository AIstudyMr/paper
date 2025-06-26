import pandas as pd

def rename_columns():
    print("正在读取数据...")
    df = pd.read_csv('存纸架数据汇总.csv')
    
    # 创建列名映射字典
    column_mapping = {
        # 切数相关列重命名
        '1#有效切数': '3#有效切数',
        '1#瞬时切数': '3#瞬时切数',
        '2#有效切数': '1#有效切数',
        '2#瞬时切数': '1#瞬时切数',
        '3#有效切数': '2#有效切数',
        '3#瞬时切数': '2#瞬时切数',
        # 4#保持不变
        
        # 裁切通道相关列重命名
        '进第一裁切通道纸条计数': '进第四裁切通道纸条计数_temp',  # 使用临时名称避免冲突
        '进第二裁切通道纸条计数': '进第一裁切通道纸条计数',
        '进第三裁切通道纸条计数': '进第三裁切通道纸条计数',  # 保持不变
        '进第四裁切通道纸条计数': '进第二裁切通道纸条计数'
    }
    
    # 第一步重命名
    df = df.rename(columns=column_mapping)
    
    # 第二步：将临时名称改为最终名称
    df = df.rename(columns={'进第四裁切通道纸条计数_temp': '进第四裁切通道纸条计数'})
    
    # 保存更改后的文件
    df.to_csv('存纸架数据汇总.csv', index=False, encoding='utf-8')
    print("列名已更改并保存到文件")
    
    # 打印新的列名
    print("\n更改后的列名：")
    for col in df.columns:
        print(col)

if __name__ == "__main__":
    rename_columns() 