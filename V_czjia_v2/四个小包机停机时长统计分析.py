import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MachineDowntimeAnalyzer:
    def __init__(self, base_path="四个小包机启停分析结果"):
        self.base_path = base_path
        self.machines = ["1号小包机", "2号小包机", "3号小包机", "4号小包机"]
        self.downtime_files = [
            "待机到停止.csv",
            "正常生产到停止.csv",
            "停止到正常生产.csv",
            "停止到待机.csv"
        ]
        self.results = {}
        
    def load_data(self):
        """加载所有小包机的停机相关数据"""
        print("正在加载停机数据...")
        for machine in self.machines:
            machine_path = os.path.join(self.base_path, f"{machine}启停分析结果")
            self.results[machine] = {}
            
            for file_name in self.downtime_files:
                file_path = os.path.join(machine_path, file_name)
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        df['时间'] = pd.to_datetime(df['时间'])
                        self.results[machine][file_name] = df
                        print(f"已加载: {machine} - {file_name} ({len(df)} 条记录)")
                    except Exception as e:
                        print(f"加载失败: {machine} - {file_name}, 错误: {e}")
                else:
                    print(f"文件不存在: {file_path}")
    
    def calculate_downtime_segments(self):
        """计算停机时长段"""
        print("\n正在计算停机时长段...")
        downtime_segments = {}
        
        for machine in self.machines:
            downtime_segments[machine] = []
            
            # 分析"待机到停止"和"正常生产到停止"的数据
            stop_files = ["待机到停止.csv", "正常生产到停止.csv"]
            
            for file_name in stop_files:
                if file_name in self.results[machine]:
                    df = self.results[machine][file_name]
                    
                    # 识别连续的停机时间段
                    segments = self.identify_time_segments(df)
                    
                    for segment in segments:
                        start_time = segment['start']
                        end_time = segment['end']
                        duration = (end_time - start_time).total_seconds() / 60  # 转换为分钟
                        
                        downtime_segments[machine].append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration_minutes': duration,
                            'type': file_name.replace('.csv', ''),
                            'data_points': segment['count']
                        })
            
            print(f"{machine}: 识别出 {len(downtime_segments[machine])} 个停机时长段")
        
        return downtime_segments
    
    def identify_time_segments(self, df):
        """识别连续的时间段"""
        segments = []
        if len(df) < 2:
            return segments
            
        current_start = df.iloc[0]['时间']
        current_end = df.iloc[0]['时间']
        current_count = 1
        
        for i in range(1, len(df)):
            time_diff = (df.iloc[i]['时间'] - df.iloc[i-1]['时间']).total_seconds()
            
            # 如果时间间隔超过5分钟，认为是新的段
            if time_diff > 300:  # 5分钟
                segments.append({
                    'start': current_start,
                    'end': current_end,
                    'count': current_count
                })
                current_start = df.iloc[i]['时间']
                current_count = 1
            else:
                current_count += 1
            
            current_end = df.iloc[i]['时间']
        
        # 添加最后一个段
        segments.append({
            'start': current_start,
            'end': current_end,
            'count': current_count
        })
        
        return segments
    
    def analyze_duration_statistics(self, downtime_segments):
        """分析停机时长统计"""
        print("\n正在进行停机时长统计分析...")
        
        all_durations = []
        machine_durations = {}
        
        for machine in self.machines:
            durations = [seg['duration_minutes'] for seg in downtime_segments[machine]]
            machine_durations[machine] = durations
            all_durations.extend(durations)
        
        # 整体统计
        if all_durations:
            print(f"\n=== 四个小包机停机时长整体统计 ===")
            print(f"总停机次数: {len(all_durations)}")
            print(f"平均停机时长: {np.mean(all_durations):.2f} 分钟")
            print(f"最短停机时长: {np.min(all_durations):.2f} 分钟")
            print(f"最长停机时长: {np.max(all_durations):.2f} 分钟")
            print(f"停机时长中位数: {np.median(all_durations):.2f} 分钟")
            print(f"停机时长标准差: {np.std(all_durations):.2f} 分钟")
        
        # 各小包机统计
        print(f"\n=== 各小包机停机时长统计 ===")
        for machine in self.machines:
            durations = machine_durations[machine]
            if durations:
                print(f"\n{machine}:")
                print(f"  停机次数: {len(durations)}")
                print(f"  平均停机时长: {np.mean(durations):.2f} 分钟")
                print(f"  最短停机时长: {np.min(durations):.2f} 分钟")
                print(f"  最长停机时长: {np.max(durations):.2f} 分钟")
                print(f"  停机时长中位数: {np.median(durations):.2f} 分钟")
        
        return all_durations, machine_durations
    
    def analyze_duration_ranges(self, all_durations, machine_durations):
        """分析停机时长范围分布"""
        print(f"\n=== 停机时长范围分析 ===")
        
        # 定义时长范围 (分钟)
        ranges = [
            (0, 5, "0-5分钟"),
            (5, 15, "5-15分钟"),
            (15, 30, "15-30分钟"),
            (30, 60, "30-60分钟"),
            (60, 120, "1-2小时"),
            (120, 300, "2-5小时"),
            (300, 600, "5-10小时"),
            (600, float('inf'), "10小时以上")
        ]
        
        # 整体范围统计
        print(f"\n整体时长范围分布:")
        range_counts = {}
        for min_val, max_val, label in ranges:
            count = sum(1 for d in all_durations if min_val <= d < max_val)
            percentage = (count / len(all_durations) * 100) if all_durations else 0
            range_counts[label] = count
            print(f"  {label}: {count} 次 ({percentage:.1f}%)")
        
        # 各机器范围统计
        print(f"\n各小包机时长范围分布:")
        machine_range_counts = {}
        for machine in self.machines:
            durations = machine_durations[machine]
            machine_range_counts[machine] = {}
            if durations:
                print(f"\n{machine}:")
                for min_val, max_val, label in ranges:
                    count = sum(1 for d in durations if min_val <= d < max_val)
                    percentage = (count / len(durations) * 100) if durations else 0
                    machine_range_counts[machine][label] = count
                    if count > 0:
                        print(f"  {label}: {count} 次 ({percentage:.1f}%)")
        
        return range_counts, machine_range_counts
    
    def create_visualizations(self, all_durations, machine_durations, range_counts, machine_range_counts):
        """创建可视化图表"""
        print(f"\n正在生成可视化图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('四个小包机停机时长统计分析', fontsize=16, fontweight='bold')
        
        # 1. 整体停机时长分布直方图
        ax1 = axes[0, 0]
        if all_durations:
            # 使用对数刻度以更好地显示分布
            bins = np.logspace(0, np.log10(max(all_durations)), 20)
            ax1.hist(all_durations, bins=bins, alpha=0.7, edgecolor='black')
            ax1.set_xscale('log')
            ax1.set_xlabel('停机时长 (分钟, 对数刻度)')
            ax1.set_ylabel('频次')
            ax1.set_title('整体停机时长分布')
            ax1.grid(True, alpha=0.3)
        
        # 2. 各小包机停机次数对比
        ax2 = axes[0, 1]
        machine_counts = [len(machine_durations[machine]) for machine in self.machines]
        bars = ax2.bar(range(len(self.machines)), machine_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_xlabel('小包机')
        ax2.set_ylabel('停机次数')
        ax2.set_title('各小包机停机次数对比')
        ax2.set_xticks(range(len(self.machines)))
        ax2.set_xticklabels(self.machines, rotation=45)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. 停机时长范围分布饼图
        ax3 = axes[1, 0]
        if range_counts:
            labels = []
            sizes = []
            for label, count in range_counts.items():
                if count > 0:
                    labels.append(f"{label}\n({count}次)")
                    sizes.append(count)
            
            if sizes:
                colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
                wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                                  colors=colors, startangle=90)
                ax3.set_title('停机时长范围分布')
        
        # 4. 各小包机平均停机时长对比
        ax4 = axes[1, 1]
        avg_durations = []
        machine_names = []
        for machine in self.machines:
            durations = machine_durations[machine]
            if durations:
                avg_durations.append(np.mean(durations))
                machine_names.append(machine)
        
        if avg_durations:
            bars = ax4.bar(range(len(machine_names)), avg_durations, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax4.set_xlabel('小包机')
            ax4.set_ylabel('平均停机时长 (分钟)')
            ax4.set_title('各小包机平均停机时长对比')
            ax4.set_xticks(range(len(machine_names)))
            ax4.set_xticklabels(machine_names, rotation=45)
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"四个小包机停机时长统计分析_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {filename}")
        
        plt.show()
    
    def save_detailed_report(self, downtime_segments, all_durations, machine_durations, range_counts):
        """保存详细报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"四个小包机停机时长分析报告_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("四个小包机停机时长统计分析报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 整体统计
            if all_durations:
                f.write("一、整体停机时长统计\n")
                f.write("-" * 30 + "\n")
                f.write(f"总停机次数: {len(all_durations)} 次\n")
                f.write(f"平均停机时长: {np.mean(all_durations):.2f} 分钟\n")
                f.write(f"最短停机时长: {np.min(all_durations):.2f} 分钟\n")
                f.write(f"最长停机时长: {np.max(all_durations):.2f} 分钟\n")
                f.write(f"停机时长中位数: {np.median(all_durations):.2f} 分钟\n")
                f.write(f"停机时长标准差: {np.std(all_durations):.2f} 分钟\n\n")
            
            # 各机器统计
            f.write("二、各小包机停机统计\n")
            f.write("-" * 30 + "\n")
            for machine in self.machines:
                durations = machine_durations[machine]
                if durations:
                    f.write(f"\n{machine}:\n")
                    f.write(f"  停机次数: {len(durations)} 次\n")
                    f.write(f"  平均停机时长: {np.mean(durations):.2f} 分钟\n")
                    f.write(f"  最短停机时长: {np.min(durations):.2f} 分钟\n")
                    f.write(f"  最长停机时长: {np.max(durations):.2f} 分钟\n")
                    f.write(f"  停机时长中位数: {np.median(durations):.2f} 分钟\n")
            
            # 时长范围分布
            f.write(f"\n\n三、停机时长范围分布\n")
            f.write("-" * 30 + "\n")
            for label, count in range_counts.items():
                percentage = (count / len(all_durations) * 100) if all_durations else 0
                f.write(f"{label}: {count} 次 ({percentage:.1f}%)\n")
            
            # 详细停机段信息
            f.write(f"\n\n四、详细停机时长段信息\n")
            f.write("-" * 30 + "\n")
            for machine in self.machines:
                segments = downtime_segments[machine]
                if segments:
                    f.write(f"\n{machine} ({len(segments)} 个停机段):\n")
                    for i, seg in enumerate(segments, 1):
                        f.write(f"  段{i}: {seg['start_time'].strftime('%Y-%m-%d %H:%M:%S')} "
                               f"到 {seg['end_time'].strftime('%Y-%m-%d %H:%M:%S')}, "
                               f"时长 {seg['duration_minutes']:.2f} 分钟, "
                               f"类型: {seg['type']}\n")
        
        print(f"详细报告已保存: {report_filename}")
    
    def run_analysis(self):
        """运行完整分析"""
        print("开始四个小包机停机时长统计分析...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 计算停机时长段
        downtime_segments = self.calculate_downtime_segments()
        
        # 3. 统计分析
        all_durations, machine_durations = self.analyze_duration_statistics(downtime_segments)
        
        # 4. 范围分析
        range_counts, machine_range_counts = self.analyze_duration_ranges(all_durations, machine_durations)
        
        # 5. 创建可视化
        self.create_visualizations(all_durations, machine_durations, range_counts, machine_range_counts)
        
        # 6. 保存详细报告
        self.save_detailed_report(downtime_segments, all_durations, machine_durations, range_counts)
        
        print("\n分析完成！")
        return {
            'downtime_segments': downtime_segments,
            'all_durations': all_durations,
            'machine_durations': machine_durations,
            'range_counts': range_counts,
            'machine_range_counts': machine_range_counts
        }

if __name__ == "__main__":
    # 创建分析器实例并运行分析
    analyzer = MachineDowntimeAnalyzer()
    results = analyzer.run_analysis() 