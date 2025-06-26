import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import numpy as np
from matplotlib.dates import DateFormatter, HourLocator, MinuteLocator
import seaborn as sns
import warnings
from matplotlib.font_manager import FontProperties
warnings.filterwarnings('ignore')

# 设置中文字体 - 使用直接路径
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"  # Microsoft YaHei字体路径
try:
    FONT_PROP = FontProperties(fname=FONT_PATH, size=12)
    print(f"成功加载字体: {FONT_PATH}")
except:
    # 备用方案
    try:
        FONT_PROP = FontProperties(family='Microsoft YaHei', size=12)
        print("使用备用字体: Microsoft YaHei")
    except:
        FONT_PROP = FontProperties(family='SimHei', size=12)
        print("使用备用字体: SimHei")

# 设置matplotlib参数
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

def create_output_folders():
    """创建输出文件夹"""
    base_folder = "有效切数每日分析"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    # 为每个小包机创建文件夹
    for i in range(1, 5):
        machine_folder = os.path.join(base_folder, f"{i}号小包机")
        if not os.path.exists(machine_folder):
            os.makedirs(machine_folder)
    
    # 创建汇总文件夹
    summary_folder = os.path.join(base_folder, "汇总对比")
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)
    
    return base_folder

def load_data():
    """加载数据"""
    print("正在加载数据...")
    try:
        df = pd.read_csv('存纸架数据汇总.csv')
        df['时间'] = pd.to_datetime(df['时间'])
        print(f"数据加载成功！时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
        print(f"数据总行数: {len(df)}")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def get_machine_columns(machine_number):
    """获取指定小包机的相关列名"""
    if machine_number in [3, 4]:
        speed_col = f"{machine_number}#小包机主机实际速度"
    else:
        speed_col = f"{machine_number}#小包机实际速度"
    
    cuts_col = f"{machine_number}#瞬时切数"
    effective_cuts_col = f"{machine_number}#有效切数"
    inner_loop_col = "外循环进内循环纸条数量"
    
    return speed_col, cuts_col, effective_cuts_col, inner_loop_col

def plot_effective_cuts_analysis(df, machine_number, date, base_folder):
    """绘制有效切数详细分析图"""
    # 设置图表样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    try:
        # 获取列名
        speed_col, cuts_col, effective_cuts_col, inner_loop_col = get_machine_columns(machine_number)
        
        # 检查列是否存在
        required_cols = [speed_col, cuts_col, effective_cuts_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"{machine_number}号小包机缺少必需的列: {missing_cols}")
            return False
        
        # 筛选指定日期的数据
        date_start = pd.Timestamp(date)
        date_end = date_start + timedelta(days=1)
        daily_data = df[(df['时间'] >= date_start) & (df['时间'] < date_end)].copy()
        
        if daily_data.empty:
            print(f"{machine_number}号小包机在 {date} 没有数据")
            return False
        
        # 筛选停机数据（速度为0或接近0）
        stopped_data = daily_data[daily_data[speed_col] <= 0.1].copy()
        
        if stopped_data.empty:
            print(f"{machine_number}号小包机在 {date} 没有停机数据")
            return False
        
        # 创建多子图布局
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 2, height_ratios=[2, 2, 1.5, 1.5], hspace=0.3, wspace=0.3)
        
        # 主标题
        fig.suptitle(f'{machine_number}号小包机 {date} 停机时有效切数与通道分析', 
                     fontsize=18, fontweight='bold', fontproperties=FONT_PROP)
        
        # 第一个子图：有效切数时间序列
        ax1 = fig.add_subplot(gs[0, :])
        
        # 绘制有效切数
        line1 = ax1.plot(stopped_data['时间'], stopped_data[effective_cuts_col], 
                        color='#1f77b4', linewidth=2, alpha=0.8, label='有效切数')
        ax1.fill_between(stopped_data['时间'], stopped_data[effective_cuts_col], 
                        alpha=0.2, color='#1f77b4')
        
        # 添加移动平均线
        if len(stopped_data) > 20:
            window = min(30, len(stopped_data) // 5)
            moving_avg = stopped_data[effective_cuts_col].rolling(window=window, center=True).mean()
            ax1.plot(stopped_data['时间'], moving_avg, 
                    color='red', linewidth=3, alpha=0.9, label=f'{window}点移动平均')
        
        ax1.set_ylabel('有效切数', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        ax1.set_title('停机时有效切数时间序列', fontsize=16, fontweight='bold', fontproperties=FONT_PROP)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12, prop=FONT_PROP)
        
        # 设置时间轴格式
        ax1.xaxis.set_major_locator(HourLocator(interval=2))
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 第二个子图：瞬时切数与有效切数对比
        ax2 = fig.add_subplot(gs[1, 0])
        ax2_twin = ax2.twinx()
        
        # 瞬时切数
        line2 = ax2.plot(stopped_data['时间'], stopped_data[cuts_col], 
                        color='#ff7f0e', linewidth=2, alpha=0.8, label='瞬时切数')
        ax2.set_ylabel('瞬时切数 (个/秒)', color='#ff7f0e', fontsize=12, fontweight='bold', fontproperties=FONT_PROP)
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        
        # 有效切数（右轴）
        line3 = ax2_twin.plot(stopped_data['时间'], stopped_data[effective_cuts_col], 
                             color='#2ca02c', linewidth=2, alpha=0.8, label='有效切数')
        ax2_twin.set_ylabel('有效切数', color='#2ca02c', fontsize=12, fontweight='bold', fontproperties=FONT_PROP)
        ax2_twin.tick_params(axis='y', labelcolor='#2ca02c')
        
        # 合并图例
        lines = line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left', prop=FONT_PROP)
        
        ax2.set_title('瞬时切数与有效切数对比', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_locator(HourLocator(interval=4))
        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 第三个子图：内外循环纸条数量
        ax3 = fig.add_subplot(gs[1, 1])
        if inner_loop_col in df.columns:
            ax3.plot(stopped_data['时间'], stopped_data[inner_loop_col], 
                    color='#d62728', linewidth=2, marker='o', markersize=3, alpha=0.8)
            ax3.fill_between(stopped_data['时间'], stopped_data[inner_loop_col], 
                            alpha=0.3, color='#d62728')
            ax3.set_ylabel('外循环进内循环纸条数量', fontsize=12, fontweight='bold', fontproperties=FONT_PROP)
            ax3.set_title('外循环进内循环纸条数量变化', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_locator(HourLocator(interval=4))
            ax3.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax3.text(0.5, 0.5, '内外循环数据不可用', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=14, fontproperties=FONT_PROP)
            ax3.set_title('内外循环数据', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        
        # 第四个子图：有效切数分布直方图
        ax4 = fig.add_subplot(gs[2, 0])
        effective_cuts_data = stopped_data[effective_cuts_col].dropna()
        n_bins = min(30, len(effective_cuts_data.unique()))
        counts, bins, patches = ax4.hist(effective_cuts_data, bins=n_bins, 
                                        color='#1f77b4', alpha=0.7, edgecolor='black')
        
        # 添加统计线
        mean_val = effective_cuts_data.mean()
        median_val = effective_cuts_data.median()
        ax4.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {mean_val:.0f}')
        ax4.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
                   label=f'中位数: {median_val:.0f}')
        
        ax4.set_xlabel('有效切数', fontsize=12, fontproperties=FONT_PROP)
        ax4.set_ylabel('频次', fontsize=12, fontproperties=FONT_PROP)
        ax4.set_title('有效切数分布直方图', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        ax4.legend(prop=FONT_PROP)
        ax4.grid(True, alpha=0.3)
        
        # 第五个子图：瞬时切数分布箱线图
        ax5 = fig.add_subplot(gs[2, 1])
        cuts_data = stopped_data[cuts_col].dropna()
        
        # 创建箱线图
        bp = ax5.boxplot([cuts_data], labels=['瞬时切数'], patch_artist=True, 
                        notch=True, showmeans=True)
        bp['boxes'][0].set_facecolor('#ff7f0e')
        bp['boxes'][0].set_alpha(0.7)
        bp['means'][0].set_marker('D')
        bp['means'][0].set_markerfacecolor('red')
        bp['means'][0].set_markeredgecolor('red')
        
        ax5.set_ylabel('瞬时切数 (个/秒)', fontsize=12, fontproperties=FONT_PROP)
        ax5.set_title('瞬时切数分布箱线图', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        ax5.grid(True, alpha=0.3)
        
        # 第六个子图：统计信息表格
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # 计算详细统计信息
        effective_cuts_stats = effective_cuts_data.describe()
        cuts_stats = cuts_data.describe()
        
        stats_data = [
            ['指标', '有效切数', '瞬时切数', '外循环进内循环'],
            ['数据点数', f'{len(effective_cuts_data):,}', f'{len(cuts_data):,}', 
             f'{len(stopped_data[inner_loop_col].dropna()):,}' if inner_loop_col in df.columns else 'N/A'],
            ['平均值', f'{effective_cuts_stats["mean"]:.2f}', f'{cuts_stats["mean"]:.3f}', 
             f'{stopped_data[inner_loop_col].mean():.3f}' if inner_loop_col in df.columns else 'N/A'],
            ['中位数', f'{effective_cuts_stats["50%"]:.2f}', f'{cuts_stats["50%"]:.3f}', 
             f'{stopped_data[inner_loop_col].median():.3f}' if inner_loop_col in df.columns else 'N/A'],
            ['标准差', f'{effective_cuts_stats["std"]:.2f}', f'{cuts_stats["std"]:.3f}', 
             f'{stopped_data[inner_loop_col].std():.3f}' if inner_loop_col in df.columns else 'N/A'],
            ['最小值', f'{effective_cuts_stats["min"]:.2f}', f'{cuts_stats["min"]:.3f}', 
             f'{stopped_data[inner_loop_col].min():.3f}' if inner_loop_col in df.columns else 'N/A'],
            ['最大值', f'{effective_cuts_stats["max"]:.2f}', f'{cuts_stats["max"]:.3f}', 
             f'{stopped_data[inner_loop_col].max():.3f}' if inner_loop_col in df.columns else 'N/A'],
            ['25%分位数', f'{effective_cuts_stats["25%"]:.2f}', f'{cuts_stats["25%"]:.3f}', 
             f'{stopped_data[inner_loop_col].quantile(0.25):.3f}' if inner_loop_col in df.columns else 'N/A'],
            ['75%分位数', f'{effective_cuts_stats["75%"]:.2f}', f'{cuts_stats["75%"]:.3f}', 
             f'{stopped_data[inner_loop_col].quantile(0.75):.3f}' if inner_loop_col in df.columns else 'N/A']
        ]
        
        # 创建表格
        table = ax6.table(cellText=stats_data[1:],
                         colLabels=stats_data[0],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.25, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式和字体
        for i in range(len(stats_data)):
            for j in range(len(stats_data[0])):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white', fontproperties=FONT_PROP)
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    cell.set_text_props(fontproperties=FONT_PROP)
        
        # 保存图表
        filename = f"{machine_number}号小包机_{date}_有效切数分析.png"
        filepath = os.path.join(base_folder, f"{machine_number}号小包机", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 打印统计信息
        print(f"\n{machine_number}号小包机 {date} 有效切数分析:")
        print(f"  停机数据点数: {len(stopped_data):,}")
        print(f"  有效切数: 平均={effective_cuts_stats['mean']:.2f}, 中位数={effective_cuts_stats['50%']:.2f}")
        print(f"  瞬时切数: 平均={cuts_stats['mean']:.3f}, 中位数={cuts_stats['50%']:.3f}")
        print(f"  图表已保存至: {filepath}")
        
        return True
        
    except Exception as e:
        print(f"绘制 {machine_number}号小包机 {date} 有效切数分析时出错: {e}")
        return False

def create_daily_comparison(df, date, base_folder):
    """创建每日四台机器有效切数对比"""
    try:
        # 筛选指定日期的数据
        date_start = pd.Timestamp(date)
        date_end = date_start + timedelta(days=1)
        daily_data = df[(df['时间'] >= date_start) & (df['时间'] < date_end)].copy()
        
        if daily_data.empty:
            return False
        
        # 收集每台机器的数据
        machine_data = {}
        for machine_number in range(1, 5):
            speed_col, cuts_col, effective_cuts_col, inner_loop_col = get_machine_columns(machine_number)
            
            if all(col in df.columns for col in [speed_col, cuts_col, effective_cuts_col]):
                stopped_data = daily_data[daily_data[speed_col] <= 0.1]
                if not stopped_data.empty:
                    machine_data[machine_number] = {
                        'effective_cuts': stopped_data[effective_cuts_col],
                        'cuts': stopped_data[cuts_col],
                        'inner_loop': stopped_data[inner_loop_col] if inner_loop_col in df.columns else None
                    }
        
        if not machine_data:
            return False
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{date} 四台小包机有效切数对比分析', fontsize=16, fontweight='bold', fontproperties=FONT_PROP)
        
        # 有效切数箱线图对比
        ax1 = axes[0, 0]
        effective_cuts_data = [machine_data[i]['effective_cuts'] for i in sorted(machine_data.keys())]
        labels = [f'{i}号机' for i in sorted(machine_data.keys())]
        bp1 = ax1.boxplot(effective_cuts_data, labels=labels, patch_artist=True, notch=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_title('有效切数分布对比', fontweight='bold', fontproperties=FONT_PROP)
        ax1.set_ylabel('有效切数', fontproperties=FONT_PROP)
        ax1.grid(True, alpha=0.3)
        # 设置X轴标签字体
        for label in ax1.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        # 平均有效切数对比
        ax2 = axes[0, 1]
        machines = sorted(machine_data.keys())
        avg_effective_cuts = [machine_data[i]['effective_cuts'].mean() for i in machines]
        bars = ax2.bar([f'{i}号机' for i in machines], avg_effective_cuts, color=colors[:len(machines)])
        ax2.set_title('平均有效切数对比', fontweight='bold', fontproperties=FONT_PROP)
        ax2.set_ylabel('平均有效切数', fontproperties=FONT_PROP)
        # 设置X轴标签字体
        for label in ax2.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        # 添加数值标签
        for bar, value in zip(bars, avg_effective_cuts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, 
                    f'{value:.0f}', ha='center', fontsize=10, fontproperties=FONT_PROP)
        
        # 有效切数变异系数对比
        ax3 = axes[1, 0]
        cv_values = [machine_data[i]['effective_cuts'].std() / machine_data[i]['effective_cuts'].mean() 
                    for i in machines]
        bars2 = ax3.bar([f'{i}号机' for i in machines], cv_values, color=colors[:len(machines)])
        ax3.set_title('有效切数变异系数对比', fontweight='bold', fontproperties=FONT_PROP)
        ax3.set_ylabel('变异系数 (标准差/平均值)', fontproperties=FONT_PROP)
        # 设置X轴标签字体
        for label in ax3.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        for bar, value in zip(bars2, cv_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.3f}', ha='center', fontsize=10, fontproperties=FONT_PROP)
        
        # 数据点数对比
        ax4 = axes[1, 1]
        data_counts = [len(machine_data[i]['effective_cuts']) for i in machines]
        bars3 = ax4.bar([f'{i}号机' for i in machines], data_counts, color=colors[:len(machines)])
        ax4.set_title('停机数据点数对比', fontweight='bold', fontproperties=FONT_PROP)
        ax4.set_ylabel('数据点数', fontproperties=FONT_PROP)
        # 设置X轴标签字体
        for label in ax4.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        for bar, value in zip(bars3, data_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{value:,}', ha='center', fontsize=10, fontproperties=FONT_PROP)
        
        plt.tight_layout()
        
        # 保存对比图表
        comparison_filename = f"{date}_四台小包机有效切数对比.png"
        comparison_filepath = os.path.join(base_folder, "汇总对比", comparison_filename)
        plt.savefig(comparison_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  四台机器有效切数对比图表已保存至: {comparison_filepath}")
        return True
        
    except Exception as e:
        print(f"创建 {date} 四台机器有效切数对比分析时出错: {e}")
        return False

def main():
    """主函数"""
    print("开始生成有效切数每日分析...")
    
    # 创建输出文件夹
    base_folder = create_output_folders()
    print(f"输出文件夹: {base_folder}")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 获取数据的日期范围
    start_date = df['时间'].dt.date.min()
    end_date = df['时间'].dt.date.max()
    
    print(f"\n开始分析日期范围: {start_date} 到 {end_date}")
    
    # 遍历每一天
    current_date = start_date
    total_success = 0
    total_attempts = 0
    
    while current_date <= end_date:
        print(f"\n{'='*50}")
        print(f"处理日期: {current_date}")
        print(f"{'='*50}")
        
        # 为每个小包机绘制有效切数分析图表
        for machine_number in range(1, 5):
            total_attempts += 1
            success = plot_effective_cuts_analysis(df, machine_number, current_date, base_folder)
            if success:
                total_success += 1
        
        # 创建当日四台机器对比
        create_daily_comparison(df, current_date, base_folder)
        
        current_date += timedelta(days=1)
    
    print(f"\n{'='*50}")
    print(f"有效切数分析完成总结:")
    print(f"  总尝试次数: {total_attempts}")
    print(f"  成功生成图表: {total_success}")
    print(f"  成功率: {total_success/total_attempts*100:.1f}%")
    print(f"  所有分析结果已保存至: '{base_folder}' 文件夹")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 