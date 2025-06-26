import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_sequential_flow():
    """按照指定点位顺序分析产品流转时间"""
    
    # 读取匹配的数据
    try:
        data = pd.read_csv('iot_delay_analysis_results.csv')
        print(f"加载数据：{len(data)}条记录")
        
        # 转换时间格式
        data['时间'] = pd.to_datetime(data['时间'])
        data = data.sort_values('时间')
        
    except Exception as e:
        print(f"读取数据出错：{e}")
        return
    
    # 定义共同的前置流程
    common_sequence = [
        ('1_折叠机速度', '折叠机实际速度'),
        ('2_折叠机入包数', '折叠机入包数'),
        ('3_折叠机出包数', '折叠机出包数'),
        ('4_存纸率', '存纸率'),
        ('5_外循环进内循环', '外循环进内循环纸条数量')
    ]
    
    # 定义四条并行的生产线
    parallel_lines = {
        '生产线1': [
            ('6_第一裁切通道', '进第一裁切通道纸条计数'),
            ('7_裁切速度', '裁切机实际速度'),  # 共用裁切速度
            ('8_1号有效切数', '1#有效切数'),
            ('9_1号小包机入包数', '1#小包机入包数'),
            ('10_1号小包机速度', '1#小包机实际速度')
        ],
        '生产线2': [
            ('6_第二裁切通道', '进第二裁切通道纸条计数'),
            ('7_裁切速度', '裁切机实际速度'),  # 共用裁切速度
            ('8_2号有效切数', '2#有效切数'),
            ('9_2号小包机入包数', '2#小包机入包数'),
            ('10_2号小包机速度', '2#小包机实际速度')
        ],
        '生产线3': [
            ('6_第三裁切通道', '进第三裁切通道纸条计数'),
            ('7_裁切速度', '裁切机实际速度'),  # 共用裁切速度
            ('8_3号有效切数', '3#有效切数'),
            ('9_3号小包机入包数', '3#小包机入包数'),
            ('10_3号小包机速度', '3#小包机主机实际速度')
        ],
        '生产线4': [
            ('6_第四裁切通道', '进第四裁切通道纸条计数'),
            ('7_裁切速度', '裁切机实际速度'),  # 共用裁切速度
            ('8_4号有效切数', '4#有效切数'),
            ('9_4号小包机入包数', '4#小包机入包数'),
            ('10_4号小包机速度', '4#小包机主机实际速度')
        ]
    }
    
    print(f"\n=== 并行流程结构分析 ===")
    print(f"共同前置流程: {len(common_sequence)}个步骤")
    print(f"并行生产线: {len(parallel_lines)}条")
    
    # 分析共同前置流程
    print(f"\n=== 共同前置流程分析 ===")
    common_results = analyze_sequence(data, common_sequence, "共同前置流程")
    
    # 分析各条并行生产线
    all_parallel_results = {}
    for line_name, line_sequence in parallel_lines.items():
        print(f"\n=== {line_name}分析 ===")
        
        # 将外循环进内循环作为起点连接到各生产线
        full_sequence = [common_sequence[-1]] + line_sequence
        line_results = analyze_sequence(data, full_sequence, line_name)
        if line_results:
            all_parallel_results[line_name] = line_results
    
    # 合并所有结果进行综合分析
    return comprehensive_analysis(common_results, all_parallel_results)

def analyze_sequence(data, sequence, sequence_name):
    """分析单个序列的传输延时"""
    
    # 检查数据中存在的列
    available_points = []
    missing_points = []
    
    for point_name, column_name in sequence:
        if column_name in data.columns:
            available_points.append((point_name, column_name))
            print(f"✓ {point_name}: {column_name}")
        else:
            missing_points.append((point_name, column_name))
            print(f"✗ {point_name}: {column_name} (缺失)")
    
    print(f"可用点位: {len(available_points)}个，缺失点位: {len(missing_points)}个")
    
    if len(available_points) < 2:
        print(f"{sequence_name}可用点位不足，无法进行延时分析")
        return None
    
    # 计算相邻点位之间的延时
    delays_results = []
    detailed_delays = {}
    
    for i in range(len(available_points) - 1):
        current_point = available_points[i]
        next_point = available_points[i + 1]
        
        current_name, current_col = current_point
        next_name, next_col = next_point
        
        # 找出数据变化点
        current_changes = find_change_points(data, current_col)
        next_changes = find_change_points(data, next_col)
        
        if len(current_changes) > 0 and len(next_changes) > 0:
            # 计算传输延时
            delays = calculate_nearest_delays(current_changes, next_changes)
            
            if len(delays) > 0:
                delay_stats = {
                    '生产线': sequence_name,
                    '起始点位': current_name,
                    '目标点位': next_name,
                    '起始列名': current_col,
                    '目标列名': next_col,
                    '平均延时(秒)': np.mean(delays),
                    '中位延时(秒)': np.median(delays),
                    '最小延时(秒)': np.min(delays),
                    '最大延时(秒)': np.max(delays),
                    '标准差(秒)': np.std(delays),
                    '样本数': len(delays)
                }
                
                delays_results.append(delay_stats)
                detailed_delays[f"{current_name}→{next_name}"] = delays
                
                print(f"  {current_name} → {next_name}")
                print(f"    平均延时: {delay_stats['平均延时(秒)']:.2f}秒")
                print(f"    中位延时: {delay_stats['中位延时(秒)']:.2f}秒")
                print(f"    样本数: {delay_stats['样本数']}")
    
    return delays_results

def comprehensive_analysis(common_results, parallel_results):
    """综合分析所有流程的延时数据"""
    
    print(f"\n=== 综合分析结果 ===")
    
    # 合并所有结果
    all_results = []
    if common_results:
        all_results.extend(common_results)
    
    for line_name, line_results in parallel_results.items():
        if line_results:
            all_results.extend(line_results)
    
    if not all_results:
        print("没有足够的数据进行综合分析")
        return
    
    # 保存详细结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('并行流程传输延时分析.csv', index=False, encoding='utf-8-sig',float_format='%.2f')
    print(f"详细分析结果已保存：并行流程传输延时分析.csv")
    
    # 分析各生产线性能对比
    analyze_production_line_performance(parallel_results)
    
    # 分析关键路径
    analyze_critical_paths(all_results)
    
    # 创建可视化
    create_comprehensive_visualization(all_results, parallel_results)

def analyze_production_line_performance(parallel_results):
    """分析各生产线性能对比"""
    
    print(f"\n=== 生产线性能对比 ===")
    
    line_performance = {}
    
    for line_name, line_results in parallel_results.items():
        if line_results:
            # 计算该生产线的总延时
            total_avg_delay = sum([r['平均延时(秒)'] for r in line_results])
            total_median_delay = sum([r['中位延时(秒)'] for r in line_results])
            avg_stability = np.mean([r['平均延时(秒)']/r['标准差(秒)'] if r['标准差(秒)'] > 0 else 0 for r in line_results])
            
            line_performance[line_name] = {
                '环节数': len(line_results),
                '总平均延时(秒)': total_avg_delay,
                '总中位延时(秒)': total_median_delay,
                '平均单环节延时(秒)': total_avg_delay / len(line_results),
                '平均稳定性': avg_stability,
                '最大单环节延时(秒)': max([r['平均延时(秒)'] for r in line_results])
            }
    
    # 显示性能对比
    for line_name, performance in line_performance.items():
        print(f"\n{line_name}:")
        for key, value in performance.items():
            print(f"  {key}: {value:.2f}")
    
    # 保存性能对比
    if line_performance:
        performance_df = pd.DataFrame(line_performance).T
        performance_df.to_csv('生产线性能对比.csv', encoding='utf-8-sig',float_format='%.2f')
        print(f"\n生产线性能对比已保存：生产线性能对比.csv")
    
    # 识别最优和最差生产线
    if line_performance:
        best_line = min(line_performance.items(), key=lambda x: x[1]['总平均延时(秒)'])
        worst_line = max(line_performance.items(), key=lambda x: x[1]['总平均延时(秒)'])
        
        print(f"\n=== 生产线排名 ===")
        print(f"性能最优: {best_line[0]} (总延时: {best_line[1]['总平均延时(秒)']:.2f}秒)")
        print(f"性能最差: {worst_line[0]} (总延时: {worst_line[1]['总平均延时(秒)']:.2f}秒)")
        print(f"性能差异: {worst_line[1]['总平均延时(秒)'] - best_line[1]['总平均延时(秒)']:.2f}秒")

def analyze_critical_paths(all_results):
    """分析关键路径"""
    
    print(f"\n=== 关键路径分析 ===")
    
    # 按照延时时间排序
    sorted_delays = sorted(all_results, key=lambda x: x['平均延时(秒)'], reverse=True)
    
    print("延时最长的前10个传输环节：")
    for i, delay in enumerate(sorted_delays[:10]):
        print(f"{i+1:2d}. {delay['生产线']} - {delay['起始点位']} → {delay['目标点位']}")
        print(f"     平均延时: {delay['平均延时(秒)']:.2f}秒, 样本数: {delay['样本数']}")
    
    # 按生产线分组分析瓶颈
    line_bottlenecks = {}
    for result in all_results:
        line = result['生产线']
        if line not in line_bottlenecks:
            line_bottlenecks[line] = []
        line_bottlenecks[line].append(result)
    
    print(f"\n=== 各生产线瓶颈分析 ===")
    for line_name, line_delays in line_bottlenecks.items():
        if line_delays:
            worst_delay = max(line_delays, key=lambda x: x['平均延时(秒)'])
            print(f"\n{line_name}最大瓶颈:")
            print(f"  {worst_delay['起始点位']} → {worst_delay['目标点位']}")
            print(f"  延时: {worst_delay['平均延时(秒)']:.2f}秒")
            print(f"  稳定性: {worst_delay['平均延时(秒)']/worst_delay['标准差(秒)']:.2f}" if worst_delay['标准差(秒)'] > 0 else "  稳定性: N/A")

def create_comprehensive_visualization(all_results, parallel_results):
    """创建综合可视化"""
    
    plt.figure(figsize=(20, 12))
    
    # 子图1: 各生产线总延时对比
    plt.subplot(2, 3, 1)
    line_names = []
    line_delays = []
    
    for line_name, line_results in parallel_results.items():
        if line_results:
            total_delay = sum([r['平均延时(秒)'] for r in line_results])
            line_names.append(line_name)
            line_delays.append(total_delay)
    
    if line_names:
        plt.bar(line_names, line_delays, alpha=0.7)
        plt.title('各生产线总延时对比')
        plt.ylabel('总延时(秒)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 子图2: 延时分布热力图
    plt.subplot(2, 3, 2)
    if all_results:
        delays_by_line = {}
        for result in all_results:
            line = result['生产线']
            if line not in delays_by_line:
                delays_by_line[line] = []
            delays_by_line[line].append(result['平均延时(秒)'])
        
        # 创建热力图数据
        max_len = max(len(delays) for delays in delays_by_line.values())
        heatmap_data = []
        line_labels = []
        
        for line, delays in delays_by_line.items():
            # 补齐长度
            padded_delays = delays + [0] * (max_len - len(delays))
            heatmap_data.append(padded_delays)
            line_labels.append(line)
        
        if heatmap_data:
            plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            plt.colorbar(label='延时(秒)')
            plt.yticks(range(len(line_labels)), line_labels)
            plt.xlabel('传输环节序号')
            plt.title('各生产线延时热力图')
    
    # 子图3: 样本数统计
    plt.subplot(2, 3, 3)
    sample_counts = [r['样本数'] for r in all_results]
    point_names = [f"{r['生产线'][:4]}-{r['目标点位'][:8]}" for r in all_results]
    
    plt.bar(range(len(sample_counts)), sample_counts, alpha=0.7, color='green')
    plt.title('数据样本统计')
    plt.ylabel('样本数')
    plt.xticks(range(0, len(point_names), max(1, len(point_names)//10)), 
               [point_names[i] for i in range(0, len(point_names), max(1, len(point_names)//10))], 
               rotation=45, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 子图4: 稳定性对比
    plt.subplot(2, 3, 4)
    stability_scores = []
    labels = []
    for result in all_results:
        if result['标准差(秒)'] > 0:
            stability = result['平均延时(秒)'] / result['标准差(秒)']
            stability_scores.append(stability)
            labels.append(f"{result['生产线'][:4]}-{result['目标点位'][:8]}")
    
    if stability_scores:
        plt.bar(range(len(stability_scores)), stability_scores, alpha=0.7, color='orange')
        plt.title('传输稳定性分析')
        plt.ylabel('稳定性指数')
        plt.xticks(range(0, len(labels), max(1, len(labels)//10)), 
                   [labels[i] for i in range(0, len(labels), max(1, len(labels)//10))], 
                   rotation=45, fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # 子图5和6: 最优和最差生产线详细对比
    if len(parallel_results) >= 2:
        # 计算各生产线总延时
        line_totals = {}
        for line_name, line_results in parallel_results.items():
            if line_results:
                line_totals[line_name] = sum([r['平均延时(秒)'] for r in line_results])
        
        best_line = min(line_totals.items(), key=lambda x: x[1])
        worst_line = max(line_totals.items(), key=lambda x: x[1])
        
        # 子图5: 最优生产线
        plt.subplot(2, 3, 5)
        best_results = parallel_results[best_line[0]]
        if best_results:
            delays = [r['平均延时(秒)'] for r in best_results]
            steps = [f"{r['目标点位'][:8]}" for r in best_results]
            plt.plot(range(len(delays)), delays, 'go-', alpha=0.7)
            plt.title(f'最优生产线: {best_line[0]}')
            plt.ylabel('延时(秒)')
            plt.xticks(range(len(steps)), steps, rotation=45, fontsize=8)
            plt.grid(True, alpha=0.3)
        
        # 子图6: 最差生产线
        plt.subplot(2, 3, 6)
        worst_results = parallel_results[worst_line[0]]
        if worst_results:
            delays = [r['平均延时(秒)'] for r in worst_results]
            steps = [f"{r['目标点位'][:8]}" for r in worst_results]
            plt.plot(range(len(delays)), delays, 'ro-', alpha=0.7)
            plt.title(f'最差生产线: {worst_line[0]}')
            plt.ylabel('延时(秒)')
            plt.xticks(range(len(steps)), steps, rotation=45, fontsize=8)
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('并行流程综合分析.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n综合可视化图表已保存：并行流程综合分析.png")

def find_change_points(data, column):
    """找出数据变化点的时间"""
    if column not in data.columns:
        return np.array([])
    
    changes = data[column].diff().fillna(0)
    change_mask = (changes != 0) & (~changes.isna())
    change_times = data.loc[change_mask, '时间'].values
    return change_times

def calculate_nearest_delays(source_times, target_times):
    """计算最近邻延时"""
    delays = []
    
    for source_time in source_times:
        # 找出在源时间之后最近的目标时间
        future_targets = target_times[target_times > source_time]
        if len(future_targets) > 0:
            nearest_target = future_targets[0]
            delay_seconds = (pd.to_datetime(nearest_target) - pd.to_datetime(source_time)).total_seconds()
            if 0 <= delay_seconds <= 1800:  # 限制在30分钟内的合理延时
                delays.append(delay_seconds)
    
    return delays

if __name__ == "__main__":
    analyze_sequential_flow() 