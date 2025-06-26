import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_folders():
    """创建保存图表的文件夹"""
    folders = [
        '小包机概率计算/停机分析总图',
        '小包机概率计算/按小时分析'
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def identify_stop_periods(df, machine_number):
    """识别机器的停机时间段"""
    speed_col = f"{machine_number}#小包机实际速度" if machine_number != 3 and machine_number != 4 else f"{machine_number}#小包机主机实际速度"
    cuts_col = f"{machine_number}#瞬时切数"
    
    # 创建一个新的DataFrame来存储停机时间段
    stop_periods = []
    
    # 获取所有速度为0的时间点
    stopped_data = df[df[speed_col] == 0].copy()
    if stopped_data.empty:
        return pd.DataFrame()
    
    # 按时间排序
    stopped_data = stopped_data.sort_values('时间')
    
    # 初始化变量
    start_time = stopped_data.iloc[0]['时间']
    prev_time = start_time
    current_cuts = []
    current_strips = []
    
    # 遍历所有停机时间点
    for idx, row in stopped_data.iterrows():
        current_time = row['时间']
        
        # 如果时间间隔超过1秒，说明是新的停机时段
        if (current_time - prev_time).total_seconds() > 1:
            # 保存前一个停机时段
            stop_periods.append({
                '机器号': machine_number,
                '停机开始时间': start_time,
                '停机结束时间': prev_time,
                '持续时间(秒)': (prev_time - start_time).total_seconds(),
                '平均瞬时切数': sum(current_cuts) / len(current_cuts) if current_cuts else 0,
                '最大瞬时切数': max(current_cuts) if current_cuts else 0,
                '平均外循环进内循环纸条数量': sum(current_strips) / len(current_strips) if current_strips else 0,
                '最大外循环进内循环纸条数量': max(current_strips) if current_strips else 0
            })
            
            # 开始新的停机时段
            start_time = current_time
            current_cuts = []
            current_strips = []
        
        # 记录当前数据点
        current_cuts.append(row[cuts_col])
        current_strips.append(row['外循环进内循环纸条数量'])
        prev_time = current_time
    
    # 添加最后一个停机时段
    stop_periods.append({
        '机器号': machine_number,
        '停机开始时间': start_time,
        '停机结束时间': prev_time,
        '持续时间(秒)': (prev_time - start_time).total_seconds(),
        '平均瞬时切数': sum(current_cuts) / len(current_cuts) if current_cuts else 0,
        '最大瞬时切数': max(current_cuts) if current_cuts else 0,
        '平均外循环进内循环纸条数量': sum(current_strips) / len(current_strips) if current_strips else 0,
        '最大外循环进内循环纸条数量': max(current_strips) if current_strips else 0
    })
    
    return pd.DataFrame(stop_periods)

def load_data():
    """读取CSV文件"""
    print("正在读取数据...")
    df = pd.read_csv('存纸架数据汇总.csv')
    df['时间'] = pd.to_datetime(df['时间'])
    print(f"数据时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
    return df

def plot_machine_data(df, machine_number, save_path):
    """为每台机器绘制停机时的曲线图"""
    # 获取速度列名
    speed_col = f"{machine_number}#小包机实际速度" if machine_number != 3 and machine_number != 4 else f"{machine_number}#小包机主机实际速度"
    cuts_col = f"{machine_number}#瞬时切数"
    
    # 筛选停机数据（速度为0）
    stopped_data = df[df[speed_col] == 0].copy()
    
    if stopped_data.empty:
        print(f"{machine_number}#小包机没有停机数据")
        return
    
    # 创建图表
    plt.figure(figsize=(15, 8))
    
    # 创建两个Y轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # 绘制瞬时切数
    line1 = ax1.plot(stopped_data['时间'], stopped_data[cuts_col], 
                     color='blue', label='瞬时切数')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('瞬时切数 (个/秒)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # 绘制外循环进内循环纸条数量
    line2 = ax2.plot(stopped_data['时间'], stopped_data['外循环进内循环纸条数量'], 
                     color='red', label='外循环进内循环纸条数量')
    ax2.set_ylabel('外循环进内循环纸条数量', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 添加标题
    plt.title(f'{machine_number}#小包机停机时的瞬时切数和外循环进内循环纸条数量')
    
    # 添加图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='upper right')
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印统计信息
    print(f"\n{machine_number}#小包机停机统计：")
    print(f"停机次数：{len(stopped_data)}")
    print(f"平均瞬时切数：{stopped_data[cuts_col].mean():.2f}")
    print(f"最大瞬时切数：{stopped_data[cuts_col].max():.2f}")
    print(f"平均外循环进内循环纸条数量：{stopped_data['外循环进内循环纸条数量'].mean():.2f}")
    print(f"最大外循环进内循环纸条数量：{stopped_data['外循环进内循环纸条数量'].max():.2f}")

def plot_hourly_data(df, machine_number):
    """按小时绘制每台机器的数据"""
    # 获取速度列名
    speed_col = f"{machine_number}#小包机实际速度" if machine_number != 3 and machine_number != 4 else f"{machine_number}#小包机主机实际速度"
    cuts_col = f"{machine_number}#瞬时切数"
    
    # 获取数据的起止时间
    start_time = df['时间'].min()
    end_time = df['时间'].max()
    
    # 按小时遍历
    current_time = start_time
    while current_time < end_time:
        next_hour = current_time + timedelta(hours=1)
        
        # 获取这个小时的数据
        hour_data = df[(df['时间'] >= current_time) & (df['时间'] < next_hour)].copy()
        
        # 筛选停机数据
        stopped_data = hour_data[hour_data[speed_col] == 0].copy()
        
        if not stopped_data.empty:
            # 创建图表
            plt.figure(figsize=(15, 8))
            
            # 创建两个Y轴
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # 绘制瞬时切数
            line1 = ax1.plot(stopped_data['时间'], stopped_data[cuts_col], 
                           color='blue', label='瞬时切数')
            ax1.set_xlabel('时间')
            ax1.set_ylabel('瞬时切数 (个/秒)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # 绘制外循环进内循环纸条数量
            line2 = ax2.plot(stopped_data['时间'], stopped_data['外循环进内循环纸条数量'], 
                           color='red', label='外循环进内循环纸条数量')
            ax2.set_ylabel('外循环进内循环纸条数量', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # 添加标题
            hour_str = current_time.strftime('%Y-%m-%d %H:00')
            plt.title(f'{machine_number}#小包机 {hour_str} 停机分析')
            
            # 添加图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            plt.legend(lines, labels, loc='upper right')
            
            # 自动调整布局
            plt.tight_layout()
            
            # 保存图表
            save_path = f'小包机概率计算/按小时分析/{machine_number}#小包机_{hour_str.replace(":", "_")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已生成 {machine_number}#小包机 {hour_str} 的分析图表")
        
        current_time = next_hour

def main():
    try:
        # 创建文件夹
        create_folders()
        
        # 读取数据
        df = load_data()
        
        # 创建一个空的DataFrame来存储所有机器的停机时间段
        all_stop_periods = pd.DataFrame()
        
        # 为每台机器处理数据
        for machine_number in [1, 2, 3, 4]:
            # 识别停机时间段
            stop_periods = identify_stop_periods(df, machine_number)
            all_stop_periods = pd.concat([all_stop_periods, stop_periods], ignore_index=True)
            
            # 绘制总体图表
            save_path = f'小包机概率计算/停机分析总图/{machine_number}#小包机停机分析.png'
            plot_machine_data(df, machine_number, save_path)
            
            # 绘制每小时的图表
            plot_hourly_data(df, machine_number)
        
        # 保存停机时间段数据到CSV
        if not all_stop_periods.empty:
            csv_path = '小包机概率计算/停机时间段分析.csv'
            all_stop_periods.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"\n停机时间段数据已保存到: {csv_path}")
            
            # 打印汇总统计信息
            print("\n停机时间段统计信息：")
            summary = all_stop_periods.groupby('机器号').agg({
                '停机开始时间': 'count',
                '持续时间(秒)': ['mean', 'sum'],
                '平均瞬时切数': 'mean',
                '最大瞬时切数': 'max',
                '平均外循环进内循环纸条数量': 'mean',
                '最大外循环进内循环纸条数量': 'max'
            }).round(2)
            print(summary)
        
        print("\n所有图表已保存到'小包机概率计算'文件夹中")
        print("- 总体分析图表保存在'停机分析总图'子文件夹")
        print("- 按小时分析图表保存在'按小时分析'子文件夹")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 