import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_output_dir():
    """创建输出目录"""
    output_dir = "每日裁切机速度曲线图"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_daily_speed_curve(df_day, date_str, output_dir):
    """绘制单日的速度曲线图"""
    plt.figure(figsize=(15, 8))
    
    # 绘制速度曲线
    plt.plot(df_day['时间'], df_day['裁切机实际速度'], 
             linewidth=1, color='#2E86AB', alpha=0.8)
    
    # 设置标题和标签
    plt.title(f'{date_str} 裁切机实际速度曲线', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('速度 (单位/时间)', fontsize=12)
    
    # 设置时间轴格式
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    
    # 旋转时间标签
    plt.xticks(rotation=45)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加统计信息文本框
    stats_text = f"""统计信息:
平均速度: {df_day['裁切机实际速度'].mean():.2f}
最大速度: {df_day['裁切机实际速度'].max():.2f}
最小速度: {df_day['裁切机实际速度'].min():.2f}
标准差: {df_day['裁切机实际速度'].std():.2f}
数据点数: {len(df_day):,}"""
    
    plt.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.8), fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    filename = f"{date_str}_裁切机速度曲线.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存: {filepath}")

def main():
    print("正在读取数据文件...")
    
    # 读取数据
    df = pd.read_csv('存纸架数据汇总.csv', usecols=['时间', '裁切机实际速度'])
    
    # 转换时间格式
    df['时间'] = pd.to_datetime(df['时间'])
    
    # 添加日期列
    df['日期'] = df['时间'].dt.date
    
    # 创建输出目录
    output_dir = create_output_dir()
    
    print(f"数据时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
    print(f"总数据点数: {len(df):,}")
    
    # 按日期分组并绘制每日曲线
    unique_dates = sorted(df['日期'].unique())
    
    for i, date in enumerate(unique_dates, 1):
        date_str = date.strftime('%Y-%m-%d')
        print(f"\n正在处理第 {i}/{len(unique_dates)} 天: {date_str}")
        
        # 筛选当天数据
        df_day = df[df['日期'] == date].copy()
        df_day = df_day.sort_values('时间')
        
        print(f"当天数据点数: {len(df_day):,}")
        print(f"速度范围: {df_day['裁切机实际速度'].min():.2f} - {df_day['裁切机实际速度'].max():.2f}")
        
        # 绘制当天的曲线图
        plot_daily_speed_curve(df_day, date_str, output_dir)
    
    print(f"\n所有图表已保存到 {output_dir} 目录下")
    
    # 创建总览图
    print("\n正在创建总览图...")
    create_overview_plot(df, output_dir)

def create_overview_plot(df, output_dir):
    """创建所有天数的总览图"""
    plt.figure(figsize=(20, 10))
    
    # 按日期绘制不同颜色的线条  
    unique_dates = sorted(df['日期'].unique())
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD', '#27AE60', '#E67E22']
    
    for i, date in enumerate(unique_dates):
        df_day = df[df['日期'] == date]
        color = colors[i % len(colors)]
        date_str = date.strftime('%m-%d')
        
        plt.plot(df_day['时间'], df_day['裁切机实际速度'], 
                linewidth=1, color=color, alpha=0.7, label=f'{date_str}')
    
    plt.title('裁切机实际速度总览 - 所有天数', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('时间', fontsize=14)
    plt.ylabel('速度 (单位/时间)', fontsize=14)
    
    # 设置时间轴
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # 保存总览图
    overview_path = os.path.join(output_dir, "总览_裁切机速度曲线.png")
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"总览图已保存: {overview_path}")

if __name__ == "__main__":
    main() 