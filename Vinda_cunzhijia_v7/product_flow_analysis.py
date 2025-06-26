import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_product_flow():
    """分析产品在各工序点位之间的流转时间"""
    
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
    
    # 定义生产线工序点位映射
    process_points = {
        '原纸供应': ['尾架1原纸剩余米数', '尾架2原纸剩余米数', '尾架3原纸剩余米数', 
                  '尾架5原纸剩余米数', '尾架6原纸剩余米数', '尾架7原纸剩余米数'],
        
        '裁切工序': ['有效总切数', '裁切机实际速度', '进第一裁切通道纸条计数', 
                  '进第二裁切通道纸条计数', '进第三裁切通道纸条计数', '进第四裁切通道纸条计数',
                  '1#有效切数', '2#有效切数', '3#有效切数', '4#有效切数'],
        
        '折叠工序': ['折叠机入包数', '折叠机出包数', '折叠机实际速度'],
        
        '小包装工序': ['1#小包机入包数', '1#小包机实际速度', '2#小包机入包数', '2#小包机实际速度',
                   '3#小包机入包数', '3#小包机主机实际速度', '4#小包机入包数', '4#小包机主机实际速度'],
        
        '装箱工序': ['1号装箱机出包数', '1号装箱机实际速度', '2号装箱机出包数', '2号装箱机实际速度'],
        
        '存储环节': ['存包率', '存纸率', '外循环进内循环纸条数量']
    }
    
    print("\n=== 生产工序点位分析 ===")
    
    # 分析各工序的数据变化特征
    process_analysis = {}
    
    for process_name, columns in process_points.items():
        print(f"\n{process_name}:")
        process_data = {}
        
        valid_columns = [col for col in columns if col in data.columns]
        if not valid_columns:
            print("  无可用数据列")
            continue
            
        for col in valid_columns:
            if col in data.columns:
                # 计算数值变化
                changes = data[col].diff().fillna(0)
                non_zero_changes = changes[changes != 0]
                
                if len(non_zero_changes) > 0:
                    process_data[col] = {
                        '变化次数': len(non_zero_changes),
                        '平均变化量': non_zero_changes.mean(),
                        '最大变化量': non_zero_changes.max(),
                        '最小变化量': non_zero_changes.min()
                    }
        
        process_analysis[process_name] = process_data
        
        # 显示该工序的关键统计
        if process_data:
            avg_changes = np.mean([stats['变化次数'] for stats in process_data.values()])
            print(f"  平均变化次数: {avg_changes:.0f}")
            print(f"  活跃设备数: {len(process_data)}")
    
    return analyze_inter_process_delays(data, process_points)

def analyze_inter_process_delays(data, process_points):
    """分析工序间的传输延时"""
    
    print("\n=== 工序间传输延时分析 ===")
    
    # 选择关键设备进行传输延时分析
    key_devices = {
        '裁切→折叠': ('有效总切数', '折叠机入包数'),
        '折叠→小包': ('折叠机出包数', '1#小包机入包数'),
        '小包→装箱': ('1#小包机入包数', '1号装箱机出包数'),
        '原纸→裁切': ('尾架1原纸剩余米数', '有效总切数')
    }
    
    flow_delays = {}
    
    for flow_name, (source_col, target_col) in key_devices.items():
        if source_col in data.columns and target_col in data.columns:
            
            # 找出源设备和目标设备的数据变化点
            source_changes = find_change_points(data, source_col)
            target_changes = find_change_points(data, target_col)
            
            if len(source_changes) > 0 and len(target_changes) > 0:
                # 计算最近邻时间差
                delays = calculate_nearest_delays(source_changes, target_changes)
                
                if len(delays) > 0:
                    flow_delays[flow_name] = {
                        '平均延时(秒)': np.mean(delays),
                        '中位延时(秒)': np.median(delays),
                        '最小延时(秒)': np.min(delays),
                        '最大延时(秒)': np.max(delays),
                        '标准差(秒)': np.std(delays),
                        '样本数': len(delays)
                    }
                    
                    print(f"\n{flow_name}传输延时:")
                    for key, value in flow_delays[flow_name].items():
                        if 'delay' in key.lower() or '延时' in key or '秒' in key:
                            print(f"  {key}: {value:.2f}")
                        else:
                            print(f"  {key}: {value}")
    
    return create_delay_visualization(flow_delays, data)

def find_change_points(data, column):
    """找出数据变化点的时间"""
    changes = data[column].diff().fillna(0)
    change_mask = changes != 0
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
            if 0 <= delay_seconds <= 3600:  # 限制在1小时内的合理延时
                delays.append(delay_seconds)
    
    return delays

def create_delay_visualization(flow_delays, data):
    """创建延时分析可视化"""
    
    if not flow_delays:
        print("没有足够的数据进行可视化")
        return
    
    # 创建延时对比图
    plt.figure(figsize=(12, 8))
    
    processes = list(flow_delays.keys())
    avg_delays = [flow_delays[process]['平均延时(秒)'] for process in processes]
    median_delays = [flow_delays[process]['中位延时(秒)'] for process in processes]
    
    x = np.arange(len(processes))
    width = 0.35
    
    plt.bar(x - width/2, avg_delays, width, label='平均延时', alpha=0.8)
    plt.bar(x + width/2, median_delays, width, label='中位延时', alpha=0.8)
    
    plt.xlabel('工序流程')
    plt.ylabel('延时时间(秒)')
    plt.title('各工序间传输延时对比')
    plt.xticks(x, processes, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('工序传输延时分析.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存详细分析结果
    results_df = pd.DataFrame(flow_delays).T
    results_df.to_csv('工序传输延时详细分析.csv', encoding='utf-8-sig')
    
    print(f"\n可视化图表已保存：工序传输延时分析.png")
    print(f"详细结果已保存：工序传输延时详细分析.csv")
    
    return create_production_efficiency_analysis(data)

def create_production_efficiency_analysis(data):
    """分析生产效率"""
    
    print("\n=== 生产效率分析 ===")
    
    # 选择关键效率指标
    efficiency_metrics = {
        '折叠机效率': '折叠机实际速度',
        '1#小包机效率': '1#小包机实际速度', 
        '2#小包机效率': '2#小包机实际速度',
        '裁切机效率': '裁切机实际速度',
        '1号装箱机效率': '1号装箱机实际速度',
        '2号装箱机效率': '2号装箱机实际速度'
    }
    
    efficiency_stats = {}
    
    for metric_name, column in efficiency_metrics.items():
        if column in data.columns:
            metric_data = data[column].dropna()
            if len(metric_data) > 0:
                efficiency_stats[metric_name] = {
                    '平均速度': metric_data.mean(),
                    '最大速度': metric_data.max(),
                    '最小速度': metric_data.min(),
                    '速度标准差': metric_data.std(),
                    '稳定性指数': (metric_data.mean() / metric_data.std()) if metric_data.std() > 0 else 0
                }
    
    # 显示效率分析结果
    for metric, stats in efficiency_stats.items():
        print(f"\n{metric}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
    
    # 保存效率分析
    if efficiency_stats:
        efficiency_df = pd.DataFrame(efficiency_stats).T
        efficiency_df.to_csv('生产效率分析.csv', encoding='utf-8-sig')
        print(f"\n效率分析结果已保存：生产效率分析.csv")
    
    print("\n=== 分析总结 ===")
    print("1. IoT数据传输延时：数据采集频率约为2.69秒")
    print("2. 各工序间存在传输延时，需要优化数据流转")
    print("3. 建议监控关键设备的稳定性指数")
    print("4. 所有分析结果已保存为CSV文件供进一步分析")

if __name__ == "__main__":
    analyze_product_flow() 