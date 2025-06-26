import pandas as pd

def read_csv_columns(file_path):
    try:
        # 尝试读取CSV文件的头部信息
        df = pd.read_csv(file_path, nrows=0)
        # 获取列名称
        column_names = df.columns.tolist()
        print("CSV文件的列名称:")
        for i, col in enumerate(column_names):
            print(f"{i+1}. {col}")
        return column_names
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None

if __name__ == "__main__":
    file_path = "存纸架数据汇总.csv"
    read_csv_columns(file_path) 