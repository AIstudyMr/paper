import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================== 配置区域 ===========================
# 指定要绘制的小包机编号（1-4），可以是单个数字或列表
# 例如：SELECTED_MACHINES = [1, 3] 表示只绘制1号和3号小包机
# 例如：SELECTED_MACHINES = [2] 表示只绘制2号小包机
# 例如：SELECTED_MACHINES = [1, 2, 3, 4] 表示绘制所有小包机（默认）
SELECTED_MACHINES = [1]  # 在这里修改要绘制的小包机编号



# 是否显示小包机编号映射信息
SHOW_MACHINE_MAPPING = True
# ===============================================================

def read_csv_with_encoding(filename):
    """尝试不同编码读取CSV文件"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
    for encoding in encodings:
        try:
            df = pd.read_csv(filename, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取 {filename}")
            return df
        except Exception as e:
            print(f"使用 {encoding} 编码读取失败: {e}")
            continue
    raise ValueError(f"无法读取文件 {filename}")

def get_machine_number_from_column(col_name):
    """从列名中提取小包机编号"""
    if '1#小包机' in col_name:
        return 1
    elif '2#小包机' in col_name:
        return 2
    elif '3#小包机' in col_name:
        return 3
    elif '4#小包机' in col_name:
        return 4
    else:
        return None

def plot_time_period_data(summary_data, start_time, end_time, period_idx, output_dir):
    """为指定时间段绘制图表"""
    
    # 转换时间列为datetime
    summary_data['时间'] = pd.to_datetime(summary_data['时间'])
    
    # 筛选时间段内的数据
    mask = (summary_data['时间'] >= start_time) & (summary_data['时间'] <= end_time)
    period_data = summary_data[mask].copy()
    
    if period_data.empty:
        print(f"时间段 {period_idx}: {start_time} 到 {end_time} 没有找到数据")
        return
    
    print(f"时间段 {period_idx}: 找到 {len(period_data)} 个数据点")
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 获取时间轴
    time_axis = period_data['时间']
    
    # 定义要绘制的数据列（根据实际列名调整）
    cols = period_data.columns.tolist()
    
    # 查找折叠机速度相关列
    folding_speed_cols = [col for col in cols if '折叠机' in col and ('速度' in col or '实际速度' in col)]
    # 查找存纸率相关列
    paper_rate_cols = [col for col in cols if '存纸率' in col]
    # 查找小包机速度相关列
    package_speed_cols = [col for col in cols if '小包机' in col and ('速度' in col or '实际速度' in col)]
    
    # 根据指定的机器编号筛选小包机列
    selected_package_cols = []
    machine_mapping = {}
    
    for col in package_speed_cols:
        machine_num = get_machine_number_from_column(col)
        if machine_num and machine_num in SELECTED_MACHINES:
            selected_package_cols.append(col)
            machine_mapping[machine_num] = col
    
    if SHOW_MACHINE_MAPPING and selected_package_cols:
        print(f"选择的小包机: {SELECTED_MACHINES}")
        for machine_num in sorted(machine_mapping.keys()):
            print(f"  小包机{machine_num}: {machine_mapping[machine_num]}")
    
    print(f"找到的列: 折叠机速度: {folding_speed_cols}, 存纸率: {paper_rate_cols}, 小包机速度: {selected_package_cols}")
    
    # 创建多个y轴
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    lines = []
    labels = []
    
    # 绘制折叠机速度
    for i, col in enumerate(folding_speed_cols):
        if col in period_data.columns:
            values = pd.to_numeric(period_data[col], errors='coerce')
            if not values.isna().all():
                line = ax.plot(time_axis, values, color=colors[i % len(colors)], 
                              linewidth=2, label=f'折叠机速度: {col}')
                lines.extend(line)
                labels.append(f'折叠机速度: {col}')
    
    # 绘制存纸率
    for i, col in enumerate(paper_rate_cols):
        if col in period_data.columns:
            values = pd.to_numeric(period_data[col], errors='coerce')
            if not values.isna().all():
                # 存纸率可能需要单独的y轴，因为数值范围可能不同
                ax2 = ax.twinx()
                line = ax2.plot(time_axis, values, color=colors[(i+len(folding_speed_cols)) % len(colors)], 
                               linewidth=2, linestyle='--', label=f'存纸率: {col}')
                ax2.set_ylabel('存纸率', fontsize=12)
                lines.extend(line)
                labels.append(f'存纸率: {col}')
    
    # 绘制指定的小包机速度
    for i, col in enumerate(selected_package_cols):
        if col in period_data.columns:
            values = pd.to_numeric(period_data[col], errors='coerce')
            if not values.isna().all():
                machine_num = get_machine_number_from_column(col)
                line = ax.plot(time_axis, values, color=colors[(i+len(folding_speed_cols)+len(paper_rate_cols)) % len(colors)], 
                              linewidth=1.5, linestyle='-.', label=f'小包机{machine_num}: {col}')
                lines.extend(line)
                labels.append(f'小包机{machine_num}: {col}')
    
    # 设置图表属性
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('速度', fontsize=12)
    
    # 在标题中显示选择的小包机信息
    selected_machines_str = ', '.join([f'{num}号' for num in sorted(SELECTED_MACHINES)])
    ax.set_title(f'时间段 {period_idx}: {start_time.strftime("%Y-%m-%d %H:%M:%S")} 到 {end_time.strftime("%Y-%m-%d %H:%M:%S")}\n选择的小包机: {selected_machines_str}', 
                 fontsize=14, fontweight='bold')
    
    # 格式化时间轴
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(period_data)//10)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 添加图例
    if lines:
        ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    selected_machines_suffix = '_'.join([f'机{num}' for num in sorted(SELECTED_MACHINES)])
    filename = f"时间段_{period_idx:02d}_{start_time.strftime('%Y%m%d_%H%M%S')}_到_{end_time.strftime('%H%M%S')}_{selected_machines_suffix}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存图片: {filepath}")

def main():
    # 创建输出文件夹
    selected_machines_str = '_'.join([f'机{num}' for num in sorted(SELECTED_MACHINES)])
    output_dir = f"时间段分析结果_存纸率1_{selected_machines_str}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"配置信息:")
    print(f"  选择的小包机: {SELECTED_MACHINES}")
    print(f"  输出目录: {output_dir}")
    print("=" * 50)
    
    try:
        # 读取存纸率1.csv文件
        print("读取存纸率1.csv文件...")
        periods_df = pd.read_csv('折叠机正常运行且高存纸率时间段_最终结果_存纸率1.csv')
        print(f"读取到 {len(periods_df)} 个时间段")
        
        # 读取汇总数据
        print("读取汇总数据...")
        summary_df = read_csv_with_encoding('存纸架数据汇总.csv')
        print(f"读取到 {len(summary_df)} 行汇总数据")
        print(f"汇总数据列名: {summary_df.columns.tolist()}")
        
        # 处理每个时间段
        for idx, row in periods_df.iterrows():
            start_time = pd.to_datetime(row['开始时间'])
            end_time = pd.to_datetime(row['结束时间'])
            
            print(f"\n处理时间段 {idx+1}: {start_time} 到 {end_time}")
            
            plot_time_period_data(summary_df, start_time, end_time, idx+1, output_dir)
    
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 