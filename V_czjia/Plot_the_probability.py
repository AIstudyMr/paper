import pandas as pd
import matplotlib.pyplot as plt
import os


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

########################
## 个体事件
########################
def plot_and_save_downtime_probability(hourly_file, daily_file, hourly_output, daily_output):
    """
    绘制每小时和每日停机概率曲线图，并分别保存为图片
    
    参数:
        hourly_file (str): 每小时停机概率CSV文件路径
        daily_file (str): 每日停机概率CSV文件路径
        hourly_output (str): 每小时概率图的输出文件名（默认：hourly_probability.png）
        daily_output (str): 每日概率图的输出文件名（默认：daily_probability.png）
    """
    # 读取CSV文件
    df_hourly = pd.read_csv(hourly_file)
    df_hourly['日期小时'] = pd.to_datetime(df_hourly['日期小时'])
    df_hourly.set_index('日期小时', inplace=True)

    df_daily = pd.read_csv(daily_file)
    df_daily['日期'] = pd.to_datetime(df_daily['日期'])
    df_daily.set_index('日期', inplace=True)

    # 定义机器列表和颜色
    machines = ['1#小包机', '2#小包机', '3#小包机', '4#小包机']
    colors = ['b', 'g', 'r', 'm']  # 蓝、绿、红、品红

    # 绘制每小时停机概率图
    plt.figure(figsize=(12, 6))
    for i, machine in enumerate(machines):
        plt.plot(df_hourly.index, df_hourly[machine], marker='s',
                label=machine, color=colors[i], linewidth=2)
    
    plt.title('每小时停机概率（精确到日期小时）', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('停机概率', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(hourly_output,'每小时_个体事件.png'), dpi=800, bbox_inches='tight')
    plt.close()
    print(f"每小时概率图已保存到: {hourly_output}")

    # 绘制每日停机概率图
    plt.figure(figsize=(12, 6))
    for i, machine in enumerate(machines):
        plt.plot(df_daily.index, df_daily[machine], marker='o',
                label=machine, color=colors[i], linewidth=2)
    
    plt.title('每日停机概率', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('停机概率', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(daily_output,'每天_个体事件.png'), dpi=800, bbox_inches='tight')
    plt.close()
    print(f"每日概率图已保存到: {daily_output}")



########################
## 群体事件
########################
def plot_daily_shutdown_probability(input_csv_path, output_image_path):
    """
    绘制每天小包机同时停机概率趋势图
    
    参数:
        input_csv_path (str): 输入CSV文件路径（包含分组和概率数据）
        output_image_path (str): 输出图片文件路径
    """
    # 假设数据已加载为 df
    df = pd.read_csv(input_csv_path)

    # 按 "分组"（日期）和 "同时停机数量" 分组，计算平均概率
    daily_prob = df.groupby(['分组', '同时停机数量'])['概率'].mean().unstack()

    # 重置索引，方便绘图
    daily_prob = daily_prob.reset_index()
    daily_prob['分组'] = pd.to_datetime(daily_prob['分组'])  # 转换为日期格式

    # 检查数据
    print(daily_prob.head())
    plt.figure(figsize=(12, 6))

    # 绘制曲线
    plt.plot(daily_prob['分组'], daily_prob[2], marker='o', label='2台同时停机')
    plt.plot(daily_prob['分组'], daily_prob[3], marker='s', label='3台同时停机')
    plt.plot(daily_prob['分组'], daily_prob[4], marker='^', label='4台同时停机')

    # 添加标题和标签
    plt.title('每天小包机同时停机概率趋势', fontsize=15)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('停机概率', fontsize=12)
    plt.xticks(rotation=45)  # 旋转日期标签
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_image_path,'每天_群体事件.png'), dpi=800, bbox_inches='tight')
    plt.legend()

    # 显示图表
    plt.tight_layout()




def plot_hourly_shutdown_probability(input_csv_path, output_image_path):
    """
    绘制每小时小包机同时停机概率趋势图
    
    参数:
        input_csv_path (str): 输入CSV文件路径（包含分组和概率数据）
        output_dir (str): 输出图片目录路径
    """
    # 读取数据
    df = pd.read_csv(input_csv_path)
    
    # 确保"分组"列是datetime类型
    df['分组'] = pd.to_datetime(df['分组'])
    
    # 按 "分组"（小时）和 "同时停机数量" 分组，计算平均概率
    hourly_prob = df.groupby(['分组', '同时停机数量'])['概率'].mean().unstack()
    
    # 重置索引，方便绘图
    hourly_prob = hourly_prob.reset_index()
    
    # 检查数据
    print(hourly_prob.head())
    
    # 创建图表
    plt.figure(figsize=(15, 6))  # 横向拉长以更好展示小时数据
    
    # 绘制曲线
    plt.plot(hourly_prob['分组'], hourly_prob[2], marker='o', markersize=3, label='2台同时停机')
    plt.plot(hourly_prob['分组'], hourly_prob[3], marker='s', markersize=3, label='3台同时停机')
    plt.plot(hourly_prob['分组'], hourly_prob[4], marker='^', markersize=3, label='4台同时停机')
    
    # 添加标题和标签
    plt.title('每小时小包机同时停机概率趋势', fontsize=15)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('停机概率', fontsize=12)
    plt.xticks(rotation=45)  # 旋转日期标签
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 添加图例
    plt.legend()
    
    # 调整布局并保存图片
    plt.tight_layout()
    
    
    # 保存图片
    output_image_path = os.path.join(output_image_path, '每小时_群体事件.png')
    plt.savefig(output_image_path, dpi=800, bbox_inches='tight')
    plt.close()  # 关闭图形释放内存
    
    print(f"图表已保存到: {output_image_path}")



# 使用示例
if __name__ == "__main__":
    # 示例调用（请替换为实际文件路径）
    plot_and_save_downtime_probability(
        hourly_file=r'D:\Code_File\Vinda_cunzhijia\小包机概率计算\个体事件_每小时停机概率.csv',
        daily_file=r'D:\Code_File\Vinda_cunzhijia\小包机概率计算\个体事件_每日停机概率.csv',
        hourly_output=r'D:\Code_File\Vinda_cunzhijia\小包机概率计算' ,
        daily_output=r'D:\Code_File\Vinda_cunzhijia\小包机概率计算'
    )

    plot_daily_shutdown_probability(
        input_csv_path=r"D:\Code_File\Vinda_cunzhijia\小包机概率计算\同时停机_每天.csv", 
        output_image_path=r'D:\Code_File\Vinda_cunzhijia\小包机概率计算')
    
    plot_hourly_shutdown_probability(
        input_csv_path=r"D:\Code_File\Vinda_cunzhijia\小包机概率计算\同时停机_每小时.csv", 
        output_image_path=r'D:\Code_File\Vinda_cunzhijia\小包机概率计算')