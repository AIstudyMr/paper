import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import numpy as np
from matplotlib.dates import DateFormatter, HourLocator
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
    base_folder = "待机状态分析_速度25"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    # 为每个小包机创建文件夹
    for i in range(1, 5):
        machine_folder = os.path.join(base_folder, f"{i}号小包机")
        if not os.path.exists(machine_folder):
            os.makedirs(machine_folder)
    
    # 创建汇总分析文件夹
    summary_folder = os.path.join(base_folder, "汇总分析")
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
    # 根据小包机编号确定速度列名
    if machine_number in [3, 4]:
        speed_col = f"{machine_number}#小包机主机实际速度"
    else:
        speed_col = f"{machine_number}#小包机实际速度"
    
    cuts_col = f"{machine_number}#瞬时切数"
    inner_loop_col = "外循环进内循环纸条数量"
    
    return speed_col, cuts_col, inner_loop_col

def analyze_standby_patterns(standby_data, cuts_col):
    """分析待机模式"""
    if standby_data.empty:
        return {}
    
    # 计算连续待机时段
    standby_data = standby_data.sort_values('时间')
    standby_data['time_diff'] = standby_data['时间'].diff().dt.total_seconds()
    
    # 识别待机时段（间隔超过60秒认为是新的待机时段）
    standby_segments = []
    current_segment = []
    
    for idx, row in standby_data.iterrows():
        if len(current_segment) == 0 or row['time_diff'] <= 60:
            current_segment.append(row)
        else:
            if len(current_segment) > 10:  # 至少10个数据点才算一个有效待机时段
                standby_segments.append(pd.DataFrame(current_segment))
            current_segment = [row]
    
    # 添加最后一个时段
    if len(current_segment) > 10:
        standby_segments.append(pd.DataFrame(current_segment))
    
    # 分析每个待机时段
    segment_stats = []
    for i, segment in enumerate(standby_segments):
        start_time = segment['时间'].min()
        end_time = segment['时间'].max()
        duration_minutes = (end_time - start_time).total_seconds() / 60
        segment_stats.append({
            '时段编号': i + 1,
            '开始时间': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            '结束时间': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            '持续时间(分钟)': round(duration_minutes, 2),
            '持续时间(秒)': round((end_time - start_time).total_seconds(), 1),
            '数据点数': len(segment),
            '平均瞬时切数': round(segment[cuts_col].mean(), 3),
            '最大瞬时切数': round(segment[cuts_col].max(), 3),
            '最小瞬时切数': round(segment[cuts_col].min(), 3),
            '瞬时切数标准差': round(segment[cuts_col].std(), 3),
            '瞬时切数中位数': round(segment[cuts_col].median(), 3)
        })
    
    return {
        'segments': standby_segments,
        'stats': segment_stats
    }

def plot_standby_analysis(df, machine_number, date, base_folder):
    """绘制待机状态分析图"""
    # 设置图表样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    try:
        # 获取列名
        speed_col, cuts_col, inner_loop_col = get_machine_columns(machine_number)
        
        # 检查列是否存在
        required_cols = [speed_col, cuts_col]
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
        
        # 筛选待机数据（速度为25左右，允许±2的误差）
        standby_data = daily_data[(daily_data[speed_col] >= 23) & (daily_data[speed_col] <= 27)].copy()
        
        if standby_data.empty:
            print(f"{machine_number}号小包机在 {date} 没有待机数据（速度25左右）")
            return False
        
        # 分析待机模式
        standby_analysis = analyze_standby_patterns(standby_data, cuts_col)
        
        # 创建复合图表
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 2, height_ratios=[2, 2, 1.5, 1], width_ratios=[3, 1])
        
        # 主标题
        fig.suptitle(f'{machine_number}号小包机 {date} 待机状态分析（速度25）', 
                     fontsize=18, fontweight='bold', y=0.95, fontproperties=FONT_PROP)
        
        # 第一个子图：时间序列 - 瞬时切数
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(standby_data['时间'], standby_data[cuts_col], 
                color='#2E86AB', linewidth=1.5, alpha=0.8, label='瞬时切数')
        ax1.fill_between(standby_data['时间'], standby_data[cuts_col], 
                        alpha=0.3, color='#2E86AB')
        
        ax1.set_ylabel('瞬时切数 (个/秒)', fontsize=12, fontproperties=FONT_PROP)
        ax1.set_title('待机时瞬时切数时间序列', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        ax1.grid(True, alpha=0.3)
        ax1.legend(prop=FONT_PROP)
        
        # 第二个子图：时间序列 - 内外循环数据
        ax2 = fig.add_subplot(gs[1, 0])
        if inner_loop_col in df.columns:
            ax2.plot(standby_data['时间'], standby_data[inner_loop_col], 
                    color='#F18F01', linewidth=1.5, alpha=0.8, label='外循环进内循环')
            ax2.fill_between(standby_data['时间'], standby_data[inner_loop_col], 
                            alpha=0.3, color='#F18F01')
            ax2.set_ylabel('外循环进内循环纸条数量', fontsize=12, fontproperties=FONT_PROP)
            ax2.set_title('待机时外循环进内循环时间序列', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        else:
            ax2.text(0.5, 0.5, '内外循环数据不可用', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14, fontproperties=FONT_PROP)
            ax2.set_title('外循环进内循环数据', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        
        ax2.grid(True, alpha=0.3)
        ax2.legend(prop=FONT_PROP)
        
        # 第三个子图：瞬时切数分布直方图
        ax3 = fig.add_subplot(gs[0, 1])
        cuts_data = standby_data[cuts_col].dropna()
        ax3.hist(cuts_data, bins=min(20, len(cuts_data.unique())), 
                color='#2E86AB', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('瞬时切数 (个/秒)', fontsize=12, fontproperties=FONT_PROP)
        ax3.set_ylabel('频次', fontsize=12, fontproperties=FONT_PROP)
        ax3.set_title('瞬时切数分布', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        ax3.axvline(cuts_data.mean(), color='red', linestyle='--', 
                   label=f'平均值: {cuts_data.mean():.2f}')
        ax3.legend(prop=FONT_PROP)
        
        # 第四个子图：箱线图
        ax4 = fig.add_subplot(gs[1, 1])
        box_data = [cuts_data]
        box_labels = ['瞬时切数']
        
        if inner_loop_col in df.columns:
            inner_loop_data = standby_data[inner_loop_col].dropna()
            box_data.append(inner_loop_data)
            box_labels.append('外循环进内循环')
        
        bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['#2E86AB', '#F18F01']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_title('数据分布箱线图', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        ax4.grid(True, alpha=0.3)
        # 设置X轴标签字体
        for label in ax4.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        # 第五个子图：待机时段分析
        ax5 = fig.add_subplot(gs[2, :])
        if standby_analysis['stats']:
            segments_df = pd.DataFrame(standby_analysis['stats'])
            bars = ax5.bar(range(len(segments_df)), segments_df['持续时间(分钟)'], 
                          color='#28A745', alpha=0.7)
            ax5.set_xlabel('待机时段编号', fontsize=12, fontproperties=FONT_PROP)
            ax5.set_ylabel('持续时间 (分钟)', fontsize=12, fontproperties=FONT_PROP)
            ax5.set_title('待机时段持续时间分析', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
            ax5.set_xticks(range(len(segments_df)))
            ax5.set_xticklabels([f'时段{i+1}' for i in range(len(segments_df))])
            # 设置X轴标签字体
            for label in ax5.get_xticklabels():
                label.set_fontproperties(FONT_PROP)
            
            # 添加数值标签
            for bar, duration in zip(bars, segments_df['持续时间(分钟)']):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{duration:.1f}分', ha='center', fontsize=10, fontproperties=FONT_PROP)
            
            # 保存待机时段数据为CSV
            csv_filename = f"{machine_number}号小包机_{date}_待机时段分析.csv"
            csv_filepath = os.path.join(base_folder, f"{machine_number}号小包机", csv_filename)
            segments_df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
            print(f"  待机时段数据已保存至: {csv_filepath}")
        else:
            ax5.text(0.5, 0.5, '未检测到明显的待机时段模式', 
                    transform=ax5.transAxes, ha='center', va='center', fontsize=14, fontproperties=FONT_PROP)
            ax5.set_title('待机时段分析', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        
        # 第六个子图：统计信息表格
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # 创建统计信息
        stats_data = [
            ['总待机数据点数', f'{len(standby_data):,}'],
            ['平均瞬时切数', f'{standby_data[cuts_col].mean():.3f}'],
            ['最大瞬时切数', f'{standby_data[cuts_col].max():.3f}'],
            ['最小瞬时切数', f'{standby_data[cuts_col].min():.3f}'],
            ['瞬时切数标准差', f'{standby_data[cuts_col].std():.3f}'],
            ['瞬时切数中位数', f'{standby_data[cuts_col].median():.3f}'],
            ['平均速度', f'{standby_data[speed_col].mean():.2f}'],
            ['速度范围', f'{standby_data[speed_col].min():.1f} - {standby_data[speed_col].max():.1f}']
        ]
        
        if inner_loop_col in df.columns:
            stats_data.extend([
                ['平均外循环进内循环', f'{standby_data[inner_loop_col].mean():.3f}'],
                ['最大外循环进内循环', f'{standby_data[inner_loop_col].max():.3f}'],
                ['最小外循环进内循环', f'{standby_data[inner_loop_col].min():.3f}'],
                ['外循环进内循环标准差', f'{standby_data[inner_loop_col].std():.3f}']
            ])
        
        # 创建表格
        table = ax6.table(cellText=stats_data,
                         colLabels=['统计指标', '数值'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # 设置表格样式和字体
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#28A745')
                    cell.set_text_props(weight='bold', color='white', fontproperties=FONT_PROP)
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    cell.set_text_props(fontproperties=FONT_PROP)
        
        # 设置时间轴格式
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(HourLocator(interval=2))
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        filename = f"{machine_number}号小包机_{date}_待机状态分析.png"
        filepath = os.path.join(base_folder, f"{machine_number}号小包机", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 打印详细统计信息
        print(f"\n{machine_number}号小包机 {date} 待机状态统计:")
        print(f"  待机数据点数: {len(standby_data):,}")
        print(f"  速度范围: {standby_data[speed_col].min():.1f} - {standby_data[speed_col].max():.1f} (平均: {standby_data[speed_col].mean():.2f})")
        print(f"  瞬时切数统计: 平均={standby_data[cuts_col].mean():.3f}, "
              f"最大={standby_data[cuts_col].max():.3f}, "
              f"标准差={standby_data[cuts_col].std():.3f}")
        
        if inner_loop_col in df.columns:
            print(f"  外循环进内循环统计: 平均={standby_data[inner_loop_col].mean():.3f}, "
                  f"最大={standby_data[inner_loop_col].max():.3f}, "
                  f"标准差={standby_data[inner_loop_col].std():.3f}")
        
        print(f"  检测到 {len(standby_analysis['stats'])} 个明显的待机时段")
        print(f"  待机状态分析图表已保存至: {filepath}")
        
        return True
        
    except Exception as e:
        print(f"绘制 {machine_number}号小包机 {date} 待机状态分析时出错: {e}")
        return False

def create_standby_comparison(df, date, base_folder):
    """创建多机器待机状态对比分析"""
    try:
        # 筛选指定日期的数据
        date_start = pd.Timestamp(date)
        date_end = date_start + timedelta(days=1)
        daily_data = df[(df['时间'] >= date_start) & (df['时间'] < date_end)].copy()
        
        if daily_data.empty:
            return False
        
        # 为每台机器收集数据
        machine_data = {}
        for machine_number in range(1, 5):
            speed_col, cuts_col, inner_loop_col = get_machine_columns(machine_number)
            
            if speed_col in df.columns and cuts_col in df.columns:
                # 筛选待机数据（速度为25左右）
                standby_data = daily_data[(daily_data[speed_col] >= 23) & (daily_data[speed_col] <= 27)]
                if not standby_data.empty:
                    machine_data[machine_number] = {
                        'cuts': standby_data[cuts_col],
                        'speed': standby_data[speed_col],
                        'inner_loop': standby_data[inner_loop_col] if inner_loop_col in df.columns else None
                    }
        
        if not machine_data:
            return False
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{date} 四台小包机待机状态对比分析（速度25）', fontsize=16, fontweight='bold', fontproperties=FONT_PROP)
        
        # 瞬时切数对比箱线图
        ax1 = axes[0, 0]
        cuts_data = [machine_data[i]['cuts'] for i in sorted(machine_data.keys())]
        cuts_labels = [f'{i}号机' for i in sorted(machine_data.keys())]
        bp1 = ax1.boxplot(cuts_data, labels=cuts_labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_title('待机瞬时切数分布对比', fontweight='bold', fontproperties=FONT_PROP)
        ax1.set_ylabel('瞬时切数 (个/秒)', fontproperties=FONT_PROP)
        ax1.grid(True, alpha=0.3)
        # 设置X轴标签字体
        for label in ax1.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        # 平均值对比柱状图
        ax2 = axes[0, 1]
        machines = sorted(machine_data.keys())
        avg_cuts = [machine_data[i]['cuts'].mean() for i in machines]
        bars = ax2.bar([f'{i}号机' for i in machines], avg_cuts, color=colors[:len(machines)])
        ax2.set_title('待机平均瞬时切数对比', fontweight='bold', fontproperties=FONT_PROP)
        ax2.set_ylabel('平均瞬时切数 (个/秒)', fontproperties=FONT_PROP)
        # 设置X轴标签字体
        for label in ax2.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        # 添加数值标签
        for bar, value in zip(bars, avg_cuts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', fontsize=10, fontproperties=FONT_PROP)
        
        # 待机数据点数对比
        ax3 = axes[1, 0]
        data_counts = [len(machine_data[i]['cuts']) for i in machines]
        bars2 = ax3.bar([f'{i}号机' for i in machines], data_counts, color=colors[:len(machines)])
        ax3.set_title('待机数据点数对比', fontweight='bold', fontproperties=FONT_PROP)
        ax3.set_ylabel('数据点数', fontproperties=FONT_PROP)
        # 设置X轴标签字体
        for label in ax3.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        for bar, value in zip(bars2, data_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{value:,}', ha='center', fontsize=10, fontproperties=FONT_PROP)
        
        # 外循环进内循环数据对比
        ax4 = axes[1, 1]
        if inner_loop_col in df.columns:
            # 收集有内外循环数据的机器
            inner_loop_machines = []
            inner_loop_values = []
            for i in machines:
                if machine_data[i]['inner_loop'] is not None and not machine_data[i]['inner_loop'].empty:
                    inner_loop_machines.append(i)
                    inner_loop_values.append(machine_data[i]['inner_loop'].mean())
            
            if inner_loop_machines:
                bars3 = ax4.bar([f'{i}号机' for i in inner_loop_machines], inner_loop_values, 
                               color=colors[:len(inner_loop_machines)])
                ax4.set_title('待机平均外循环进内循环对比', fontweight='bold', fontproperties=FONT_PROP)
                ax4.set_ylabel('平均外循环进内循环纸条数量', fontproperties=FONT_PROP)
                # 设置X轴标签字体
                for label in ax4.get_xticklabels():
                    label.set_fontproperties(FONT_PROP)
                
                for bar, value in zip(bars3, inner_loop_values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, 
                            f'{value:.2f}', ha='center', fontsize=10, fontproperties=FONT_PROP)
            else:
                ax4.text(0.5, 0.5, '无内外循环数据', transform=ax4.transAxes, 
                        ha='center', va='center', fontsize=14, fontproperties=FONT_PROP)
                ax4.set_title('外循环进内循环数据', fontweight='bold', fontproperties=FONT_PROP)
        else:
            ax4.text(0.5, 0.5, '内外循环数据不可用', transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=14, fontproperties=FONT_PROP)
            ax4.set_title('外循环进内循环数据', fontweight='bold', fontproperties=FONT_PROP)
        
        plt.tight_layout()
        
        # 保存对比图表
        comparison_filename = f"{date}_四台小包机待机状态对比.png"
        comparison_filepath = os.path.join(base_folder, "汇总分析", comparison_filename)
        plt.savefig(comparison_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  多机器待机状态对比分析图表已保存至: {comparison_filepath}")
        return True
        
    except Exception as e:
        print(f"创建 {date} 多机器待机状态对比分析时出错: {e}")
        return False

def create_daily_standby_summary(df, date, base_folder):
    """创建每日所有机器待机时段汇总"""
    try:
        # 筛选指定日期的数据
        date_start = pd.Timestamp(date)
        date_end = date_start + timedelta(days=1)
        daily_data = df[(df['时间'] >= date_start) & (df['时间'] < date_end)].copy()
        
        if daily_data.empty:
            return False
        
        all_segments = []
        
        # 收集所有机器的待机时段数据
        for machine_number in range(1, 5):
            speed_col, cuts_col, inner_loop_col = get_machine_columns(machine_number)
            
            if speed_col in df.columns and cuts_col in df.columns:
                # 筛选待机数据（速度为25左右）
                standby_data = daily_data[(daily_data[speed_col] >= 23) & (daily_data[speed_col] <= 27)]
                if not standby_data.empty:
                    standby_analysis = analyze_standby_patterns(standby_data, cuts_col)
                    
                    if standby_analysis['stats']:
                        for segment_stat in standby_analysis['stats']:
                            segment_stat['机器编号'] = f'{machine_number}号机'
                            all_segments.append(segment_stat)
        
        if all_segments:
            # 创建汇总DataFrame
            summary_df = pd.DataFrame(all_segments)
            
            # 重新排列列的顺序，把机器编号放在前面
            columns_order = ['机器编号', '时段编号', '开始时间', '结束时间', '持续时间(分钟)', 
                           '持续时间(秒)', '数据点数', '平均瞬时切数', '最大瞬时切数', 
                           '最小瞬时切数', '瞬时切数标准差', '瞬时切数中位数']
            summary_df = summary_df[columns_order]
            
            # 按开始时间排序
            summary_df['开始时间_datetime'] = pd.to_datetime(summary_df['开始时间'])
            summary_df = summary_df.sort_values('开始时间_datetime').drop('开始时间_datetime', axis=1)
            
            # 保存汇总CSV
            summary_filename = f"{date}_所有机器待机时段汇总.csv"
            summary_filepath = os.path.join(base_folder, "汇总分析", summary_filename)
            summary_df.to_csv(summary_filepath, index=False, encoding='utf-8-sig')
            
            print(f"  当日所有机器待机时段汇总已保存至: {summary_filepath}")
            print(f"  共检测到 {len(all_segments)} 个待机时段")
            return True
        
        return False
        
    except Exception as e:
        print(f"创建 {date} 待机时段汇总时出错: {e}")
        return False

def plot_standby_analysis(df, machine_number, date, base_folder):
    """绘制待机状态分析图"""
    # 设置图表样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    try:
        # 获取列名
        speed_col, cuts_col, inner_loop_col = get_machine_columns(machine_number)
        
        # 检查列是否存在
        required_cols = [speed_col, cuts_col]
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
        
        # 筛选待机数据（速度为25左右，允许±2的误差）
        standby_data = daily_data[(daily_data[speed_col] >= 23) & (daily_data[speed_col] <= 27)].copy()
        
        if standby_data.empty:
            print(f"{machine_number}号小包机在 {date} 没有待机数据（速度25左右）")
            return False
        
        # 分析待机模式
        standby_analysis = analyze_standby_patterns(standby_data, cuts_col)
        
        # 创建复合图表
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 2, height_ratios=[2, 2, 1.5, 1], width_ratios=[3, 1])
        
        # 主标题
        fig.suptitle(f'{machine_number}号小包机 {date} 待机状态分析（速度25）', 
                     fontsize=18, fontweight='bold', y=0.95, fontproperties=FONT_PROP)
        
        # 第一个子图：时间序列 - 瞬时切数
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(standby_data['时间'], standby_data[cuts_col], 
                color='#2E86AB', linewidth=1.5, alpha=0.8, label='瞬时切数')
        ax1.fill_between(standby_data['时间'], standby_data[cuts_col], 
                        alpha=0.3, color='#2E86AB')
        
        ax1.set_ylabel('瞬时切数 (个/秒)', fontsize=12, fontproperties=FONT_PROP)
        ax1.set_title('待机时瞬时切数时间序列', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        ax1.grid(True, alpha=0.3)
        ax1.legend(prop=FONT_PROP)
        
        # 第二个子图：时间序列 - 内外循环数据
        ax2 = fig.add_subplot(gs[1, 0])
        if inner_loop_col in df.columns:
            ax2.plot(standby_data['时间'], standby_data[inner_loop_col], 
                    color='#F18F01', linewidth=1.5, alpha=0.8, label='外循环进内循环')
            ax2.fill_between(standby_data['时间'], standby_data[inner_loop_col], 
                            alpha=0.3, color='#F18F01')
            ax2.set_ylabel('外循环进内循环纸条数量', fontsize=12, fontproperties=FONT_PROP)
            ax2.set_title('待机时外循环进内循环时间序列', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        else:
            ax2.text(0.5, 0.5, '内外循环数据不可用', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14, fontproperties=FONT_PROP)
            ax2.set_title('外循环进内循环数据', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        
        ax2.grid(True, alpha=0.3)
        ax2.legend(prop=FONT_PROP)
        
        # 第三个子图：瞬时切数分布直方图
        ax3 = fig.add_subplot(gs[0, 1])
        cuts_data = standby_data[cuts_col].dropna()
        ax3.hist(cuts_data, bins=min(20, len(cuts_data.unique())), 
                color='#2E86AB', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('瞬时切数 (个/秒)', fontsize=12, fontproperties=FONT_PROP)
        ax3.set_ylabel('频次', fontsize=12, fontproperties=FONT_PROP)
        ax3.set_title('瞬时切数分布', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        ax3.axvline(cuts_data.mean(), color='red', linestyle='--', 
                   label=f'平均值: {cuts_data.mean():.2f}')
        ax3.legend(prop=FONT_PROP)
        
        # 第四个子图：箱线图
        ax4 = fig.add_subplot(gs[1, 1])
        box_data = [cuts_data]
        box_labels = ['瞬时切数']
        
        if inner_loop_col in df.columns:
            inner_loop_data = standby_data[inner_loop_col].dropna()
            box_data.append(inner_loop_data)
            box_labels.append('外循环进内循环')
        
        bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['#2E86AB', '#F18F01']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_title('数据分布箱线图', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        ax4.grid(True, alpha=0.3)
        # 设置X轴标签字体
        for label in ax4.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        # 第五个子图：待机时段分析
        ax5 = fig.add_subplot(gs[2, :])
        if standby_analysis['stats']:
            segments_df = pd.DataFrame(standby_analysis['stats'])
            bars = ax5.bar(range(len(segments_df)), segments_df['持续时间(分钟)'], 
                          color='#28A745', alpha=0.7)
            ax5.set_xlabel('待机时段编号', fontsize=12, fontproperties=FONT_PROP)
            ax5.set_ylabel('持续时间 (分钟)', fontsize=12, fontproperties=FONT_PROP)
            ax5.set_title('待机时段持续时间分析', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
            ax5.set_xticks(range(len(segments_df)))
            ax5.set_xticklabels([f'时段{i+1}' for i in range(len(segments_df))])
            # 设置X轴标签字体
            for label in ax5.get_xticklabels():
                label.set_fontproperties(FONT_PROP)
            
            # 添加数值标签
            for bar, duration in zip(bars, segments_df['持续时间(分钟)']):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{duration:.1f}分', ha='center', fontsize=10, fontproperties=FONT_PROP)
            
            # 保存待机时段数据为CSV
            csv_filename = f"{machine_number}号小包机_{date}_待机时段分析.csv"
            csv_filepath = os.path.join(base_folder, f"{machine_number}号小包机", csv_filename)
            segments_df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
            print(f"  待机时段数据已保存至: {csv_filepath}")
        else:
            ax5.text(0.5, 0.5, '未检测到明显的待机时段模式', 
                    transform=ax5.transAxes, ha='center', va='center', fontsize=14, fontproperties=FONT_PROP)
            ax5.set_title('待机时段分析', fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        
        # 第六个子图：统计信息表格
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # 创建统计信息
        stats_data = [
            ['总待机数据点数', f'{len(standby_data):,}'],
            ['平均瞬时切数', f'{standby_data[cuts_col].mean():.3f}'],
            ['最大瞬时切数', f'{standby_data[cuts_col].max():.3f}'],
            ['最小瞬时切数', f'{standby_data[cuts_col].min():.3f}'],
            ['瞬时切数标准差', f'{standby_data[cuts_col].std():.3f}'],
            ['瞬时切数中位数', f'{standby_data[cuts_col].median():.3f}'],
            ['平均速度', f'{standby_data[speed_col].mean():.2f}'],
            ['速度范围', f'{standby_data[speed_col].min():.1f} - {standby_data[speed_col].max():.1f}']
        ]
        
        if inner_loop_col in df.columns:
            stats_data.extend([
                ['平均外循环进内循环', f'{standby_data[inner_loop_col].mean():.3f}'],
                ['最大外循环进内循环', f'{standby_data[inner_loop_col].max():.3f}'],
                ['最小外循环进内循环', f'{standby_data[inner_loop_col].min():.3f}'],
                ['外循环进内循环标准差', f'{standby_data[inner_loop_col].std():.3f}']
            ])
        
        # 创建表格
        table = ax6.table(cellText=stats_data,
                         colLabels=['统计指标', '数值'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # 设置表格样式和字体
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#28A745')
                    cell.set_text_props(weight='bold', color='white', fontproperties=FONT_PROP)
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                    cell.set_text_props(fontproperties=FONT_PROP)
        
        # 设置时间轴格式
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(HourLocator(interval=2))
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        filename = f"{machine_number}号小包机_{date}_待机状态分析.png"
        filepath = os.path.join(base_folder, f"{machine_number}号小包机", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 打印详细统计信息
        print(f"\n{machine_number}号小包机 {date} 待机状态统计:")
        print(f"  待机数据点数: {len(standby_data):,}")
        print(f"  速度范围: {standby_data[speed_col].min():.1f} - {standby_data[speed_col].max():.1f} (平均: {standby_data[speed_col].mean():.2f})")
        print(f"  瞬时切数统计: 平均={standby_data[cuts_col].mean():.3f}, "
              f"最大={standby_data[cuts_col].max():.3f}, "
              f"标准差={standby_data[cuts_col].std():.3f}")
        
        if inner_loop_col in df.columns:
            print(f"  外循环进内循环统计: 平均={standby_data[inner_loop_col].mean():.3f}, "
                  f"最大={standby_data[inner_loop_col].max():.3f}, "
                  f"标准差={standby_data[inner_loop_col].std():.3f}")
        
        print(f"  检测到 {len(standby_analysis['stats'])} 个明显的待机时段")
        print(f"  待机状态分析图表已保存至: {filepath}")
        
        return True
        
    except Exception as e:
        print(f"绘制 {machine_number}号小包机 {date} 待机状态分析时出错: {e}")
        return False

def create_standby_comparison(df, date, base_folder):
    """创建多机器待机状态对比分析"""
    try:
        # 筛选指定日期的数据
        date_start = pd.Timestamp(date)
        date_end = date_start + timedelta(days=1)
        daily_data = df[(df['时间'] >= date_start) & (df['时间'] < date_end)].copy()
        
        if daily_data.empty:
            return False
        
        # 为每台机器收集数据
        machine_data = {}
        for machine_number in range(1, 5):
            speed_col, cuts_col, inner_loop_col = get_machine_columns(machine_number)
            
            if speed_col in df.columns and cuts_col in df.columns:
                # 筛选待机数据（速度为25左右）
                standby_data = daily_data[(daily_data[speed_col] >= 23) & (daily_data[speed_col] <= 27)]
                if not standby_data.empty:
                    machine_data[machine_number] = {
                        'cuts': standby_data[cuts_col],
                        'speed': standby_data[speed_col],
                        'inner_loop': standby_data[inner_loop_col] if inner_loop_col in df.columns else None
                    }
        
        if not machine_data:
            return False
        
        # 创建对比图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{date} 四台小包机待机状态对比分析（速度25）', fontsize=16, fontweight='bold', fontproperties=FONT_PROP)
        
        # 瞬时切数对比箱线图
        ax1 = axes[0, 0]
        cuts_data = [machine_data[i]['cuts'] for i in sorted(machine_data.keys())]
        cuts_labels = [f'{i}号机' for i in sorted(machine_data.keys())]
        bp1 = ax1.boxplot(cuts_data, labels=cuts_labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_title('待机瞬时切数分布对比', fontweight='bold', fontproperties=FONT_PROP)
        ax1.set_ylabel('瞬时切数 (个/秒)', fontproperties=FONT_PROP)
        ax1.grid(True, alpha=0.3)
        # 设置X轴标签字体
        for label in ax1.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        # 平均值对比柱状图
        ax2 = axes[0, 1]
        machines = sorted(machine_data.keys())
        avg_cuts = [machine_data[i]['cuts'].mean() for i in machines]
        bars = ax2.bar([f'{i}号机' for i in machines], avg_cuts, color=colors[:len(machines)])
        ax2.set_title('待机平均瞬时切数对比', fontweight='bold', fontproperties=FONT_PROP)
        ax2.set_ylabel('平均瞬时切数 (个/秒)', fontproperties=FONT_PROP)
        # 设置X轴标签字体
        for label in ax2.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        # 添加数值标签
        for bar, value in zip(bars, avg_cuts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', fontsize=10, fontproperties=FONT_PROP)
        
        # 待机数据点数对比
        ax3 = axes[1, 0]
        data_counts = [len(machine_data[i]['cuts']) for i in machines]
        bars2 = ax3.bar([f'{i}号机' for i in machines], data_counts, color=colors[:len(machines)])
        ax3.set_title('待机数据点数对比', fontweight='bold', fontproperties=FONT_PROP)
        ax3.set_ylabel('数据点数', fontproperties=FONT_PROP)
        # 设置X轴标签字体
        for label in ax3.get_xticklabels():
            label.set_fontproperties(FONT_PROP)
        
        for bar, value in zip(bars2, data_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{value:,}', ha='center', fontsize=10, fontproperties=FONT_PROP)
        
        # 外循环进内循环数据对比
        ax4 = axes[1, 1]
        if inner_loop_col in df.columns:
            # 收集有内外循环数据的机器
            inner_loop_machines = []
            inner_loop_values = []
            for i in machines:
                if machine_data[i]['inner_loop'] is not None and not machine_data[i]['inner_loop'].empty:
                    inner_loop_machines.append(i)
                    inner_loop_values.append(machine_data[i]['inner_loop'].mean())
            
            if inner_loop_machines:
                bars3 = ax4.bar([f'{i}号机' for i in inner_loop_machines], inner_loop_values, 
                               color=colors[:len(inner_loop_machines)])
                ax4.set_title('待机平均外循环进内循环对比', fontweight='bold', fontproperties=FONT_PROP)
                ax4.set_ylabel('平均外循环进内循环纸条数量', fontproperties=FONT_PROP)
                # 设置X轴标签字体
                for label in ax4.get_xticklabels():
                    label.set_fontproperties(FONT_PROP)
                
                for bar, value in zip(bars3, inner_loop_values):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01, 
                            f'{value:.2f}', ha='center', fontsize=10, fontproperties=FONT_PROP)
            else:
                ax4.text(0.5, 0.5, '无内外循环数据', transform=ax4.transAxes, 
                        ha='center', va='center', fontsize=14, fontproperties=FONT_PROP)
                ax4.set_title('外循环进内循环数据', fontweight='bold', fontproperties=FONT_PROP)
        else:
            ax4.text(0.5, 0.5, '内外循环数据不可用', transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=14, fontproperties=FONT_PROP)
            ax4.set_title('外循环进内循环数据', fontweight='bold', fontproperties=FONT_PROP)
        
        plt.tight_layout()
        
        # 保存对比图表
        comparison_filename = f"{date}_四台小包机待机状态对比.png"
        comparison_filepath = os.path.join(base_folder, "汇总分析", comparison_filename)
        plt.savefig(comparison_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  多机器待机状态对比分析图表已保存至: {comparison_filepath}")
        return True
        
    except Exception as e:
        print(f"创建 {date} 多机器待机状态对比分析时出错: {e}")
        return False

def create_daily_standby_summary(df, date, base_folder):
    """创建每日所有机器待机时段汇总"""
    try:
        # 筛选指定日期的数据
        date_start = pd.Timestamp(date)
        date_end = date_start + timedelta(days=1)
        daily_data = df[(df['时间'] >= date_start) & (df['时间'] < date_end)].copy()
        
        if daily_data.empty:
            return False
        
        all_segments = []
        
        # 收集所有机器的待机时段数据
        for machine_number in range(1, 5):
            speed_col, cuts_col, inner_loop_col = get_machine_columns(machine_number)
            
            if speed_col in df.columns and cuts_col in df.columns:
                # 筛选待机数据（速度为25左右）
                standby_data = daily_data[(daily_data[speed_col] >= 23) & (daily_data[speed_col] <= 27)]
                if not standby_data.empty:
                    standby_analysis = analyze_standby_patterns(standby_data, cuts_col)
                    
                    if standby_analysis['stats']:
                        for segment_stat in standby_analysis['stats']:
                            segment_stat['机器编号'] = f'{machine_number}号机'
                            all_segments.append(segment_stat)
        
        if all_segments:
            # 创建汇总DataFrame
            summary_df = pd.DataFrame(all_segments)
            
            # 重新排列列的顺序，把机器编号放在前面
            columns_order = ['机器编号', '时段编号', '开始时间', '结束时间', '持续时间(分钟)', 
                           '持续时间(秒)', '数据点数', '平均瞬时切数', '最大瞬时切数', 
                           '最小瞬时切数', '瞬时切数标准差', '瞬时切数中位数']
            summary_df = summary_df[columns_order]
            
            # 按开始时间排序
            summary_df['开始时间_datetime'] = pd.to_datetime(summary_df['开始时间'])
            summary_df = summary_df.sort_values('开始时间_datetime').drop('开始时间_datetime', axis=1)
            
            # 保存汇总CSV
            summary_filename = f"{date}_所有机器待机时段汇总.csv"
            summary_filepath = os.path.join(base_folder, "汇总分析", summary_filename)
            summary_df.to_csv(summary_filepath, index=False, encoding='utf-8-sig')
            
            print(f"  当日所有机器待机时段汇总已保存至: {summary_filepath}")
            print(f"  共检测到 {len(all_segments)} 个待机时段")
            return True
        
        return False
        
    except Exception as e:
        print(f"创建 {date} 待机时段汇总时出错: {e}")
        return False

def main():
    """主函数"""
    print("开始生成待机状态分析（速度25）...")
    
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
        print(f"\n{'='*60}")
        print(f"处理日期: {current_date}")
        print(f"{'='*60}")
        
        # 为每个小包机绘制待机状态分析图表
        for machine_number in range(1, 5):
            total_attempts += 1
            success = plot_standby_analysis(df, machine_number, current_date, base_folder)
            if success:
                total_success += 1
        
        # 创建多机器待机状态对比分析
        create_standby_comparison(df, current_date, base_folder)
        
        # 创建当日所有机器待机时段汇总
        create_daily_standby_summary(df, current_date, base_folder)
        
        current_date += timedelta(days=1)
    
    print(f"\n{'='*60}")
    print(f"待机状态分析完成总结:")
    print(f"  总尝试次数: {total_attempts}")
    print(f"  成功生成图表: {total_success}")
    print(f"  成功率: {total_success/total_attempts*100:.1f}%")
    print(f"  所有分析结果已保存至: '{base_folder}' 文件夹")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 