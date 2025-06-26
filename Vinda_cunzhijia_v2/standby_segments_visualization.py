import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import glob
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
    base_folder = "待机时段详细分析_瞬时切数与裁切通道对比"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    # 为每个小包机创建文件夹
    for i in range(1, 5):
        machine_folder = os.path.join(base_folder, f"{i}号小包机")
        if not os.path.exists(machine_folder):
            os.makedirs(machine_folder)
    
    # 创建汇总文件夹
    summary_folder = os.path.join(base_folder, "所有时段汇总")
    if not os.path.exists(summary_folder):
        os.makedirs(summary_folder)
    
    return base_folder

def load_original_data():
    """加载原始数据"""
    print("正在加载原始数据...")
    try:
        df = pd.read_csv('存纸架数据汇总.csv')
        df['时间'] = pd.to_datetime(df['时间'])
        print(f"原始数据加载成功！时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
        print(f"数据总行数: {len(df)}")
        return df
    except Exception as e:
        print(f"原始数据加载失败: {e}")
        return None

def get_machine_columns(machine_number):
    """获取指定小包机的相关列名"""
    # 根据小包机编号确定速度列名
    if machine_number in [3, 4]:
        speed_col = f"{machine_number}#小包机主机实际速度"
    else:
        speed_col = f"{machine_number}#小包机实际速度"
    
    cuts_col = f"{machine_number}#瞬时切数"
    
    # 裁切通道对应关系 - 使用中文数字
    chinese_numbers = {1: '一', 2: '二', 3: '三', 4: '四'}
    cutting_channel_col = f"进第{chinese_numbers[machine_number]}裁切通道纸条计数"
    
    return speed_col, cuts_col, cutting_channel_col

def find_csv_files():
    """查找所有待机时段CSV文件"""
    csv_files = []
    base_path = "待机状态分析_速度25"
    
    if not os.path.exists(base_path):
        print(f"错误：找不到文件夹 '{base_path}'")
        return []
    
    # 遍历各个小包机文件夹
    for machine_num in range(1, 5):
        machine_folder = os.path.join(base_path, f"{machine_num}号小包机")
        if os.path.exists(machine_folder):
            pattern = os.path.join(machine_folder, "*待机时段分析.csv")
            files = glob.glob(pattern)
            for file in files:
                csv_files.append({
                    'file_path': file,
                    'machine_number': machine_num,
                    'file_name': os.path.basename(file)
                })
    
    print(f"找到 {len(csv_files)} 个待机时段CSV文件")
    return csv_files

def plot_segment_comparison(original_df, machine_number, segment_info, base_folder):
    """绘制单个时间段的瞬时切数与裁切通道对比图"""
    try:
        # 获取列名
        speed_col, cuts_col, cutting_channel_col = get_machine_columns(machine_number)
        
        # 检查列是否存在
        required_cols = [cuts_col, cutting_channel_col]
        missing_cols = [col for col in required_cols if col not in original_df.columns]
        if missing_cols:
            print(f"  警告：{machine_number}号小包机缺少列: {missing_cols}")
            return False
        
        # 解析时间段
        start_time = pd.to_datetime(segment_info['开始时间'])
        end_time = pd.to_datetime(segment_info['结束时间'])
        
        # 提取时间段数据
        segment_data = original_df[
            (original_df['时间'] >= start_time) & 
            (original_df['时间'] <= end_time)
        ].copy()
        
        if segment_data.empty:
            print(f"  警告：时间段 {start_time} 到 {end_time} 没有数据")
            return False
        
        # 创建图表
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        # 设置主标题
        duration_minutes = segment_info['持续时间(分钟)']
        fig.suptitle(f'{machine_number}号小包机 时段{segment_info["时段编号"]} 瞬时切数与裁切通道对比\n'
                     f'时间: {start_time.strftime("%Y-%m-%d %H:%M:%S")} 到 {end_time.strftime("%H:%M:%S")} '
                     f'(持续{duration_minutes:.1f}分钟)', 
                     fontsize=14, fontweight='bold', fontproperties=FONT_PROP)
        
        # 绘制瞬时切数（左Y轴）
        color1 = '#1f77b4'
        ax1.set_xlabel('时间', fontsize=12, fontproperties=FONT_PROP)
        ax1.set_ylabel('瞬时切数 (个/秒)', color=color1, fontsize=12, fontproperties=FONT_PROP)
        
        # 绘制瞬时切数曲线和点
        line1 = ax1.plot(segment_data['时间'], segment_data[cuts_col], 
                        color=color1, linewidth=2, alpha=0.8, label='瞬时切数')
        ax1.scatter(segment_data['时间'], segment_data[cuts_col], 
                   color=color1, s=20, alpha=0.6, zorder=5)
        
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # 创建右Y轴用于裁切通道数据
        ax2 = ax1.twinx()
        color2 = '#ff7f0e'
        chinese_numbers = {1: '一', 2: '二', 3: '三', 4: '四'}
        ax2.set_ylabel(f'进第{chinese_numbers[machine_number]}裁切通道纸条计数', color=color2, fontsize=12, fontproperties=FONT_PROP)
        
        # 绘制裁切通道曲线和点
        line2 = ax2.plot(segment_data['时间'], segment_data[cutting_channel_col], 
                        color=color2, linewidth=2, alpha=0.8, label=f'进第{chinese_numbers[machine_number]}裁切通道纸条计数')
        ax2.scatter(segment_data['时间'], segment_data[cutting_channel_col], 
                   color=color2, s=20, alpha=0.6, zorder=5)
        
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 设置时间轴格式
        if duration_minutes <= 30:  # 30分钟以内，每5分钟一个刻度
            ax1.xaxis.set_major_locator(MinuteLocator(interval=5))
        elif duration_minutes <= 120:  # 2小时以内，每15分钟一个刻度
            ax1.xaxis.set_major_locator(MinuteLocator(interval=15))
        else:  # 超过2小时，每30分钟一个刻度
            ax1.xaxis.set_major_locator(MinuteLocator(interval=30))
        
        ax1.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 设置图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', prop=FONT_PROP)
        
        # 添加统计信息文本框
        stats_text = f'''数据点数: {len(segment_data)}
瞬时切数: 平均{segment_data[cuts_col].mean():.2f}, 最大{segment_data[cuts_col].max():.2f}
裁切通道: 平均{segment_data[cutting_channel_col].mean():.2f}, 最大{segment_data[cutting_channel_col].max():.2f}'''
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontproperties=FONT_PROP)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        safe_start_time = start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{machine_number}号小包机_时段{segment_info['时段编号']}_{safe_start_time}_瞬时切数与裁切通道对比.png"
        
        # 保存到机器文件夹
        machine_filepath = os.path.join(base_folder, f"{machine_number}号小包机", filename)
        plt.savefig(machine_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        
        # 同时保存到汇总文件夹
        summary_filename = f"{machine_number}号机_时段{segment_info['时段编号']}_{safe_start_time}.png"
        summary_filepath = os.path.join(base_folder, "所有时段汇总", summary_filename)
        plt.savefig(summary_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.close()
        
        print(f"    时段{segment_info['时段编号']} 图表已保存: {filename}")
        return True
        
    except Exception as e:
        print(f"    绘制时段{segment_info['时段编号']}时出错: {e}")
        return False

def main():
    """主函数"""
    print("开始生成待机时段详细分析...")
    
    # 创建输出文件夹
    base_folder = create_output_folders()
    print(f"输出文件夹: {base_folder}")
    
    # 加载原始数据
    original_df = load_original_data()
    if original_df is None:
        return
    
    # 查找所有CSV文件
    csv_files = find_csv_files()
    if not csv_files:
        print("没有找到待机时段CSV文件")
        return
    
    total_segments = 0
    total_success = 0
    
    # 处理每个CSV文件
    for csv_info in csv_files:
        machine_number = csv_info['machine_number']
        file_path = csv_info['file_path']
        file_name = csv_info['file_name']
        
        print(f"\n处理 {machine_number}号小包机: {file_name}")
        
        try:
            # 读取CSV文件
            segments_df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            if segments_df.empty:
                print(f"  {file_name} 文件为空")
                continue
            
            print(f"  找到 {len(segments_df)} 个时间段")
            
            # 处理每个时间段
            for idx, segment_info in segments_df.iterrows():
                total_segments += 1
                success = plot_segment_comparison(original_df, machine_number, segment_info, base_folder)
                if success:
                    total_success += 1
                    
        except Exception as e:
            print(f"  处理文件 {file_name} 时出错: {e}")
    
    # 输出总结
    print(f"\n{'='*60}")
    print(f"待机时段详细分析完成总结:")
    print(f"  处理的CSV文件数: {len(csv_files)}")
    print(f"  总时间段数: {total_segments}")
    print(f"  成功生成图表: {total_success}")
    print(f"  成功率: {total_success/total_segments*100:.1f}%" if total_segments > 0 else "  成功率: 0%")
    print(f"  所有图表已保存至: '{base_folder}' 文件夹")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 