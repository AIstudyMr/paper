import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def create_output_folder():
    """创建输出文件夹"""
    folder_name = "时间段图表结果_存纸率1"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"已创建文件夹: {folder_name}")
    else:
        print(f"文件夹already存在: {folder_name}")
    return folder_name

def load_data():
    """加载数据文件"""
    print("正在加载数据文件...")
    
    # 加载最终结果CSV
    try:
        result_df = pd.read_csv('折叠机正常运行且高存纸率时间段_最终结果_存纸率1.csv')
        result_df['开始时间'] = pd.to_datetime(result_df['开始时间'])
        result_df['结束时间'] = pd.to_datetime(result_df['结束时间'])
        print(f"成功加载最终结果文件，共 {len(result_df)} 个时间段")
    except Exception as e:
        print(f"加载最终结果文件失败: {e}")
        return None, None
    
    # 加载汇总数据CSV
    try:
        # 尝试不同编码
        try:
            summary_df = pd.read_csv('存纸架数据汇总.csv', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                summary_df = pd.read_csv('存纸架数据汇总.csv', encoding='gbk')
            except UnicodeDecodeError:
                summary_df = pd.read_csv('存纸架数据汇总.csv', encoding='gb2312')
        
        summary_df['时间'] = pd.to_datetime(summary_df['时间'])
        print(f"成功加载汇总数据文件，共 {len(summary_df)} 行数据")
        
        # 检查所需列是否存在
        required_columns = ['折叠机实际速度', '存纸率', '折叠机出包数', '外循环进内循环纸条数量']
        missing_columns = [col for col in required_columns if col not in summary_df.columns]
        if missing_columns:
            print(f"警告：以下列在汇总数据中未找到: {missing_columns}")
            print(f"可用列名: {summary_df.columns.tolist()}")
        
        return result_df, summary_df
        
    except Exception as e:
        print(f"加载汇总数据文件失败: {e}")
        return result_df, None

def plot_time_period(period_data, period_info, output_folder, period_index):
    """为单个时间段绘制图表"""
    
    start_time = period_info['开始时间']
    end_time = period_info['结束时间']
    duration = period_info['持续时间']
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'时间段 {period_index+1}: {start_time.strftime("%Y-%m-%d %H:%M:%S")} 至 {end_time.strftime("%Y-%m-%d %H:%M:%S")}\n持续时间: {duration}', 
                 fontsize=14, fontweight='bold')
    
    # 准备数据
    time_data = period_data['时间']
    
    # 子图1: 折叠机速度
    axes[0, 0].plot(time_data, period_data['折叠机实际速度'], 'b-', linewidth=1.5, label='折叠机速度')
    axes[0, 0].axhline(y=100, color='r', linestyle='--', alpha=0.7, label='正常运行阈值(100)')
    axes[0, 0].set_title('折叠机实际速度', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('速度', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 子图2: 存纸率
    axes[0, 1].plot(time_data, period_data['存纸率'], 'g-', linewidth=1.5, label='存纸率')
    axes[0, 1].axhline(y=5, color='r', linestyle='--', alpha=0.7, label='高存纸率阈值(5)')
    axes[0, 1].set_title('存纸率', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('存纸率', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 子图3: 折叠机出包数
    axes[1, 0].plot(time_data, period_data['折叠机出包数'], 'orange', linewidth=1.5, label='折叠机出包数')
    axes[1, 0].set_title('折叠机出包数', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('出包数', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 子图4: 外循环进内循环纸条数量
    axes[1, 1].plot(time_data, period_data['外循环进内循环纸条数量'], 'purple', linewidth=1.5, label='纸条数量')
    axes[1, 1].set_title('外循环进内循环纸条数量', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('纸条数量', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 设置X轴时间格式
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, int((end_time - start_time).total_seconds() / 600))))
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('时间', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    file_name = f"时间段_{period_index+1:03d}_{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%H%M%S')}.png"
    file_path = os.path.join(output_folder, file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return file_path

def main():
    """主函数"""
    print("=== 时间段图表绘制程序 ===")
    print("功能：根据最终结果CSV的时间段，绘制对应的数据图表")
    print("="*50)
    
    # 创建输出文件夹
    output_folder = create_output_folder()
    
    # 加载数据
    result_df, summary_df = load_data()
    
    if result_df is None or summary_df is None:
        print("数据加载失败，程序退出")
        return
    
    print(f"\n开始为 {len(result_df)} 个时间段绘制图表...")
    
    successful_plots = 0
    failed_plots = 0
    
    # 为每个时间段绘制图表
    for index, row in result_df.iterrows():
        try:
            start_time = row['开始时间']
            end_time = row['结束时间']
            
            # 从汇总数据中提取对应时间段的数据
            period_mask = (summary_df['时间'] >= start_time) & (summary_df['时间'] <= end_time)
            period_data = summary_df[period_mask].copy()
            
            if len(period_data) == 0:
                print(f"警告：时间段 {index+1} 没有找到对应的数据")
                failed_plots += 1
                continue
            
            # 处理数值列
            numeric_columns = ['折叠机实际速度', '存纸率', '折叠机出包数', '外循环进内循环纸条数量']
            for col in numeric_columns:
                if col in period_data.columns:
                    period_data[col] = pd.to_numeric(period_data[col], errors='coerce')
            
            # 绘制图表
            file_path = plot_time_period(period_data, row, output_folder, index)
            print(f"完成时间段 {index+1}/{len(result_df)}: {file_path}")
            successful_plots += 1
            
        except Exception as e:
            print(f"绘制时间段 {index+1} 失败: {e}")
            failed_plots += 1
            continue
    
    # 统计结果
    print(f"\n=== 绘制完成 ===")
    print(f"成功绘制: {successful_plots} 个图表")
    print(f"失败: {failed_plots} 个图表")
    print(f"图表保存在文件夹: {output_folder}")
    
    if successful_plots > 0:
        print(f"\n图表文件命名格式: 时间段_XXX_开始时间_结束时间.png")
        print("每个图表包含4个子图:")
        print("- 左上: 折叠机实际速度")
        print("- 右上: 存纸率") 
        print("- 左下: 折叠机出包数")
        print("- 右下: 外循环进内循环纸条数量")

if __name__ == "__main__":
    main() 