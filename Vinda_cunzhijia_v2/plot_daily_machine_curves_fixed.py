import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import numpy as np
from matplotlib.dates import DateFormatter, HourLocator
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_font():
    """配置中文字体显示 - 最可靠的方法"""
    
    # 清理matplotlib缓存
    try:
        cache_dir = fm.get_cachedir()
        cache_file = os.path.join(cache_dir, 'fontList.json')
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print("已清理matplotlib字体缓存")
    except:
        pass
    
    # 方法1: 直接指定字体文件路径（最可靠）
    font_prop = None
    try:
        import platform
        if platform.system() == 'Windows':
            font_paths = [
                'C:/Windows/Fonts/msyh.ttc',  # Microsoft YaHei
                'C:/Windows/Fonts/simhei.ttf',  # SimHei
                'C:/Windows/Fonts/simsun.ttc',  # SimSun
                'C:/Windows/Fonts/kaiti.ttf',   # KaiTi
            ]
            
            for font_path in font_paths:
                try:
                    if os.path.exists(font_path):
                        font_prop = fm.FontProperties(fname=font_path)
                        plt.rcParams['font.family'] = font_prop.get_name()
                        print(f"成功设置字体文件: {font_path} -> {font_prop.get_name()}")
                        break
                except Exception as e:
                    continue
    except:
        pass
    
    # 方法2: 设置字体族
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.facecolor'] = 'white'
    
    # 如果没有找到字体文件，创建字体属性对象
    if font_prop is None:
        font_prop = fm.FontProperties(family='Microsoft YaHei')
    
    print("字体配置完成")
    return font_prop

# 全局字体配置
FONT_PROP = setup_chinese_font()

def create_output_folders():
    """创建输出文件夹"""
    base_folder = "修复版每日小包机停机曲线分析"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    # 为每个小包机创建文件夹
    for i in range(1, 5):
        machine_folder = os.path.join(base_folder, f"{i}号小包机")
        if not os.path.exists(machine_folder):
            os.makedirs(machine_folder)
    
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

def plot_daily_machine_curves(df, machine_number, date, base_folder):
    """绘制指定小包机在指定日期的停机时曲线图 - 修复中文字体版本"""
    
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
        
        # 筛选停机数据（速度为0或接近0）
        stopped_data = daily_data[daily_data[speed_col] <= 0.1].copy()
        
        if stopped_data.empty:
            print(f"{machine_number}号小包机在 {date} 没有停机数据")
            return False
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
        
        # 主标题 - 使用fontproperties确保中文显示
        fig.suptitle(f'{machine_number}号小包机 {date} 停机时曲线分析', 
                     fontsize=16, fontweight='bold', fontproperties=FONT_PROP)
        
        # 第一个子图：瞬时切数
        ax1.plot(stopped_data['时间'], stopped_data[cuts_col], 
                color='blue', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax1.set_ylabel('瞬时切数 (个/秒)', fontsize=12, color='blue', fontproperties=FONT_PROP)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('停机时瞬时切数变化', fontsize=14, fontproperties=FONT_PROP)
        
        # 添加统计信息到第一个子图
        mean_cuts = stopped_data[cuts_col].mean()
        max_cuts = stopped_data[cuts_col].max()
        ax1.axhline(y=mean_cuts, color='red', linestyle='--', alpha=0.7, 
                   label=f'平均值: {mean_cuts:.2f}')
        ax1.legend(prop=FONT_PROP)
        
        # 第二个子图：内外循环纸条数量（如果存在该列）
        if inner_loop_col in df.columns:
            ax2.plot(stopped_data['时间'], stopped_data[inner_loop_col], 
                    color='red', linewidth=2, marker='s', markersize=3, alpha=0.7)
            ax2.set_ylabel('外循环进内循环纸条数量', fontsize=12, color='red', fontproperties=FONT_PROP)
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(True, alpha=0.3)
            ax2.set_title('停机时外循环进内循环纸条数量变化', fontsize=14, fontproperties=FONT_PROP)
            
            # 添加统计信息
            mean_strips = stopped_data[inner_loop_col].mean()
            ax2.axhline(y=mean_strips, color='blue', linestyle='--', alpha=0.7, 
                       label=f'平均值: {mean_strips:.2f}')
            ax2.legend(prop=FONT_PROP)
        else:
            ax2.text(0.5, 0.5, '内外循环数据不可用', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=14, fontproperties=FONT_PROP)
            ax2.set_title('内外循环数据', fontsize=14, fontproperties=FONT_PROP)
        
        # 第三个子图：双轴图，同时显示瞬时切数和内外循环
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(stopped_data['时间'], stopped_data[cuts_col], 
                        color='blue', linewidth=2, label='瞬时切数')
        ax3.set_ylabel('瞬时切数 (个/秒)', fontsize=12, color='blue', fontproperties=FONT_PROP)
        ax3.tick_params(axis='y', labelcolor='blue')
        
        if inner_loop_col in df.columns:
            line2 = ax3_twin.plot(stopped_data['时间'], stopped_data[inner_loop_col], 
                                 color='red', linewidth=2, label='外循环进内循环')
            ax3_twin.set_ylabel('外循环进内循环纸条数量', fontsize=12, color='red', fontproperties=FONT_PROP)
            ax3_twin.tick_params(axis='y', labelcolor='red')
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper right', prop=FONT_PROP)
        else:
            ax3.legend(loc='upper right', prop=FONT_PROP)
        
        ax3.set_xlabel('时间', fontsize=12, fontproperties=FONT_PROP)
        ax3.set_title('停机时综合数据对比', fontsize=14, fontproperties=FONT_PROP)
        ax3.grid(True, alpha=0.3)
        
        # 设置时间轴格式
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_locator(HourLocator(interval=2))
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        filename = f"{machine_number}号小包机_{date}_停机曲线分析.png"
        filepath = os.path.join(base_folder, f"{machine_number}号小包机", filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 打印统计信息
        print(f"\n{machine_number}号小包机 {date} 停机统计:")
        print(f"  停机数据点数: {len(stopped_data)}")
        print(f"  平均瞬时切数: {stopped_data[cuts_col].mean():.2f}")
        print(f"  最大瞬时切数: {stopped_data[cuts_col].max():.2f}")
        print(f"  最小瞬时切数: {stopped_data[cuts_col].min():.2f}")
        
        if inner_loop_col in df.columns:
            print(f"  平均外循环进内循环: {stopped_data[inner_loop_col].mean():.2f}")
            print(f"  最大外循环进内循环: {stopped_data[inner_loop_col].max():.2f}")
        
        print(f"  图表已保存至: {filepath}")
        
        return True
        
    except Exception as e:
        print(f"绘制 {machine_number}号小包机 {date} 的图表时出错: {e}")
        return False

def plot_all_machines_all_days(df, base_folder):
    """为所有小包机绘制所有天的曲线图"""
    # 获取数据的日期范围
    start_date = df['时间'].dt.date.min()
    end_date = df['时间'].dt.date.max()
    
    print(f"数据日期范围: {start_date} 到 {end_date}")
    
    # 遍历每一天
    current_date = start_date
    total_success = 0
    total_attempts = 0
    
    while current_date <= end_date:
        print(f"\n处理日期: {current_date}")
        
        # 为每个小包机绘制当天的图表
        for machine_number in range(1, 5):
            total_attempts += 1
            success = plot_daily_machine_curves(df, machine_number, current_date, base_folder)
            if success:
                total_success += 1
        
        current_date += timedelta(days=1)
    
    print(f"\n总结:")
    print(f"  总尝试次数: {total_attempts}")
    print(f"  成功生成图表: {total_success}")
    print(f"  成功率: {total_success/total_attempts*100:.1f}%")

def generate_summary_report(df, base_folder):
    """生成汇总报告"""
    summary_data = []
    
    # 获取数据的日期范围
    start_date = df['时间'].dt.date.min()
    end_date = df['时间'].dt.date.max()
    
    current_date = start_date
    while current_date <= end_date:
        date_start = pd.Timestamp(current_date)
        date_end = date_start + timedelta(days=1)
        daily_data = df[(df['时间'] >= date_start) & (df['时间'] < date_end)]
        
        for machine_number in range(1, 5):
            try:
                speed_col, cuts_col, inner_loop_col = get_machine_columns(machine_number)
                
                if speed_col in df.columns and cuts_col in df.columns:
                    stopped_data = daily_data[daily_data[speed_col] <= 0.1]
                    
                    if not stopped_data.empty:
                        summary_data.append({
                            '日期': current_date,
                            '小包机号': machine_number,
                            '停机数据点数': len(stopped_data),
                            '平均瞬时切数': stopped_data[cuts_col].mean(),
                            '最大瞬时切数': stopped_data[cuts_col].max(),
                            '最小瞬时切数': stopped_data[cuts_col].min(),
                            '瞬时切数标准差': stopped_data[cuts_col].std()
                        })
            except:
                continue
        
        current_date += timedelta(days=1)
    
    # 保存汇总报告
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(base_folder, "停机数据汇总报告.csv")
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\n汇总报告已保存至: {summary_file}")

def main():
    """主函数"""
    print("开始绘制修复版每日小包机停机曲线分析...")
    print(f"使用字体: {FONT_PROP.get_name()}")
    
    # 创建输出文件夹
    base_folder = create_output_folders()
    print(f"输出文件夹: {base_folder}")
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 显示可用的列
    print("\n可用的数据列:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}: {col}")
    
    # 绘制所有机器的所有天数图表
    plot_all_machines_all_days(df, base_folder)
    
    # 生成汇总报告
    generate_summary_report(df, base_folder)
    
    print(f"\n所有图表已生成完成！请查看 '{base_folder}' 文件夹")
    print("中文字体问题已修复！")

if __name__ == "__main__":
    main() 