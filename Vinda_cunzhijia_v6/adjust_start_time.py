import pandas as pd
from datetime import datetime, timedelta

def adjust_start_time():
    # 读取CSV文件
    file_path = '折叠机正常运行且高存纸率时间段_最终结果_存纸率1.csv'
    df = pd.read_csv(file_path)
    
    print("原始数据前5行：")
    print(df.head())
    print()
    
    # 将开始时间转换为datetime格式
    df['开始时间'] = pd.to_datetime(df['开始时间'])
    
    # 将开始时间往前移10秒（减去10秒）
    df['开始时间'] = df['开始时间'] - timedelta(seconds=60)
    
    # 将datetime转换回字符串格式
    df['开始时间'] = df['开始时间'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    print("调整后的数据前5行：")
    print(df.head())
    print()
    
    # 创建新的文件名
    output_file = '折叠机正常运行且高存纸率时间段_最终结果_存纸率1_调整后.csv'
    
    # 保存修改后的数据
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"调整完成！已保存到文件: {output_file}")
    print(f"总共调整了 {len(df)} 条记录")

if __name__ == "__main__":
    adjust_start_time() 