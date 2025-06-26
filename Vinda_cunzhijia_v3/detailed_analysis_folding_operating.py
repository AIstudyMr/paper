import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import re

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

def analyze_folding_operating_results():
    """分析折叠机正常运行时间段内小包机停机分析结果（滑动窗口方法）"""
    
    # 读取结果文件
    try:
        df = pd.read_csv('折叠机运行分析结果_多状态_30.csv', encoding='utf-8-sig')
        print("成功读取折叠机运行分析结果文件")
        print(f"数据行数: {len(df)}")
        print(f"数据列: {list(df.columns)}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 检查是否包含新的滑动窗口字段
    has_window_analysis = '滑动窗口数量' in df.columns
    print(f"包含滑动窗口分析: {has_window_analysis}")
    
    # 处理数据类型转换 - 适配新的字段结构
    if '全段平均斜率' in df.columns:
        # 使用全段平均斜率作为主要分析对象
        df['斜率数值'] = df['全段平均斜率']
        slope_column = '全段平均斜率'
    elif '主要组合斜率' in df.columns:
        # 使用主要组合斜率
        df['斜率数值'] = df['主要组合斜率']
        slope_column = '主要组合斜率'
    else:
        print("错误：找不到斜率数据列")
        return
    
    # 基本统计信息
    print("\n=== 基本统计信息 ===")
    print(f"总分析次数: {len(df)}")
    print(f"存纸率斜率范围: {df['斜率数值'].min():.4f} 到 {df['斜率数值'].max():.4f} /分钟")
    print(f"存纸率斜率平均值: {df['斜率数值'].mean():.4f} /分钟")
    print(f"存纸率斜率标准差: {df['斜率数值'].std():.4f} /分钟")
    
            # 滑动窗口分析统计
    if has_window_analysis:
        print("\n=== 滑动窗口分析统计 ===")
        valid_windows = df[df['滑动窗口数量'] > 0]
        print(f"有效滑动窗口分析次数: {len(valid_windows)}/{len(df)} ({len(valid_windows)/len(df)*100:.1f}%)")
        if len(valid_windows) > 0:
            print(f"平均滑动窗口数量: {valid_windows['滑动窗口数量'].mean():.2f}")
            print(f"滑动窗口数量范围: {valid_windows['滑动窗口数量'].min():.0f} - {valid_windows['滑动窗口数量'].max():.0f}")
            print(f"平均斜率内标准差: {valid_windows['斜率标准差'].mean():.4f} /分钟")
            print(f"平均斜率变化范围: {valid_windows['斜率变化范围'].mean():.4f} /分钟")
    
    # 停机组合统计（增强版）
    print("\n=== 停机组合详细统计（滑动窗口分析）===")
    
    # 判断使用哪个停机组合列
    combination_column = '主要停机组合' if '主要停机组合' in df.columns else '停机组合'
    print(f"使用停机组合列: {combination_column}")
    
    if has_window_analysis:
        # 新版本：包含滑动窗口分析
        combination_stats = df.groupby(combination_column).agg({
            '斜率数值': ['count', 'mean', 'std', 'min', 'max'],
            '斜率标准差': 'mean',
            '斜率变化范围': 'mean',
            '滑动窗口数量': ['mean', 'sum']
        }).round(4)
        
        # 重命名列
        combination_stats.columns = [
            '次数', '平均斜率', '斜率标准差', '最小斜率', '最大斜率',
            '平均斜率内标准差', '平均斜率变化范围', 
            '平均窗口数', '总窗口数'
        ]
    else:
        # 旧版本：传统分析
        combination_stats = df.groupby(combination_column).agg({
            '斜率数值': ['count', 'mean', 'std', 'min', 'max']
        }).round(4)
        
        combination_stats.columns = ['次数', '平均斜率', '斜率标准差', '最小斜率', '最大斜率']
    
    combination_stats = combination_stats.sort_values('次数', ascending=False)
    print(combination_stats)
    
    # 按停机机器数量分类分析
    print("\n=== 按停机机器数量分类分析 ===")
    
    def categorize_combination(combination):
        if '无小包机停机' in combination:
            return '0个机器停机'
        elif '单个小包机停机' in combination:
            return '1个机器停机'
        elif '两个小包机同时停机' in combination:
            return '2个机器停机'
        elif '三个小包机同时停机' in combination:
            return '3个机器停机'
        elif '四个小包机同时停机' in combination:
            return '4个机器停机'
        else:
            return '其他'
    
    df['停机数量类别'] = df[combination_column].apply(categorize_combination)
    
    category_stats = df.groupby('停机数量类别').agg({
        '斜率数值': ['count', 'mean', 'std'],
    }).round(4)
    
    category_stats.columns = ['次数', '平均斜率(/分钟)', '斜率标准差']
    category_stats = category_stats.sort_index()
    
    print(category_stats)
    
    # 计算影响程度
    print("\n=== 不同停机组合对存纸率斜率的影响分析 ===")
    
    # 基准：无小包机停机时的平均斜率
    baseline_slope = df[df['停机数量类别'] == '0个机器停机']['斜率数值'].mean()
    print(f"基准斜率（折叠机正常运行，无小包机停机）: {baseline_slope:.4f} /分钟")
    
    print("\n各停机组合相对于基准的影响:")
    for category in ['1个机器停机', '2个机器停机', '3个机器停机', '4个机器停机']:
        if category in category_stats.index:
            category_slope = category_stats.loc[category, '平均斜率(/分钟)']
            impact = category_slope - baseline_slope
            count = int(category_stats.loc[category, '次数'])
            print(f"{category}: 平均斜率 {category_slope:.4f} /分钟, 相对影响 {impact:+.4f}/分钟, 发生次数 {count}")
    
    # 定义停机组合排序函数
    def get_machine_count_priority(combination):
        """获取停机组合的排序优先级"""
        if '无小包机停机' in combination:
            return (0, 0, combination)  # 无停机
        elif '单个小包机停机' in combination:
            machine_num = int(combination.split('#')[0].split(': ')[1])
            return (1, machine_num, combination)  # 单个停机，按机器号排序
        elif '两个小包机同时停机' in combination:
            # 提取机器号进行排序
            machine_nums = [int(x.split('#')[0]) for x in combination.split(': ')[1].split(', ')]
            machine_nums.sort()
            return (2, tuple(machine_nums), combination)  # 两个停机
        elif '三个小包机同时停机' in combination:
            # 提取机器号进行排序
            machine_nums = [int(x.split('#')[0]) for x in combination.split(': ')[1].split(', ')]
            machine_nums.sort()
            return (3, tuple(machine_nums), combination)  # 三个停机
        elif '四个小包机同时停机' in combination:
            return (4, 0, combination)  # 四个停机
        else:
            return (5, 0, combination)  # 其他情况
    
    # 保存详细分析结果（增强版）
    detailed_results = []
    
    # 按停机组合统计
    for combination, group in df.groupby(combination_column):
        result = {
            '停机组合': combination,
            '发生次数': len(group),
            '平均斜率': group['斜率数值'].mean(),
            '斜率标准差': group['斜率数值'].std(),
            '最小斜率': group['斜率数值'].min(),
            '最大斜率': group['斜率数值'].max(),
            '相对基准影响': group['斜率数值'].mean() - baseline_slope,
            '排序键': get_machine_count_priority(combination)
        }
        
        # 如果有滑动窗口分析数据，添加额外信息
        if has_window_analysis:
            valid_group = group[group['滑动窗口数量'] > 0]
            if len(valid_group) > 0:
                result.update({
                    '平均斜率内标准差': valid_group['斜率标准差'].mean(),
                    '平均斜率变化范围': valid_group['斜率变化范围'].mean(),
                    '平均滑动窗口数': valid_group['滑动窗口数量'].mean(),
                    '总滑动窗口数': valid_group['滑动窗口数量'].sum(),
                    '有效窗口分析率': len(valid_group) / len(group) * 100
                })
            else:
                result.update({
                    '平均斜率内标准差': 0,
                    '平均斜率变化范围': 0,
                    '平均滑动窗口数': 0,
                    '总滑动窗口数': 0,
                    '有效窗口分析率': 0
                })
        
        detailed_results.append(result)
    
    detailed_df = pd.DataFrame(detailed_results)
    # 按停机机器数量和机器号排序
    detailed_df = detailed_df.sort_values('排序键')
    # 删除排序键列
    detailed_df = detailed_df.drop('排序键', axis=1)
    detailed_df = detailed_df.round(4)
    
    # 保存到CSV
    output_filename = '折叠机运行期间停机组合影响分析汇总_30.csv'
    detailed_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n详细分析结果已保存到 '{output_filename}'")
    
    # 显示分析质量
    if has_window_analysis:
        print(f"\n=== 滑动窗口分析质量评估 ===")
        high_quality = detailed_df[detailed_df['有效窗口分析率'] >= 80]
        medium_quality = detailed_df[(detailed_df['有效窗口分析率'] >= 50) & (detailed_df['有效窗口分析率'] < 80)]
        low_quality = detailed_df[detailed_df['有效窗口分析率'] < 50]
        
        print(f"高质量分析（≥80%有效窗口）: {len(high_quality)} 个停机组合")
        print(f"中等质量分析（50-80%有效窗口）: {len(medium_quality)} 个停机组合")
        print(f"低质量分析（<50%有效窗口）: {len(low_quality)} 个停机组合")
    
    # 显示前10行
    print("\n前10个最常见停机组合的影响分析:")
    display_cols = ['停机组合', '发生次数', '平均斜率', '相对基准影响']
    if has_window_analysis:
        display_cols.extend(['平均滑动窗口数', '有效窗口分析率'])
    print(detailed_df[display_cols].head(10).to_string(index=False))
    
    return df, detailed_df

def analyze_slope_stability():
    """分析不同停机组合的斜率稳定性"""
    
    try:
        df = pd.read_csv('折叠机运行分析结果_多状态_30.csv', encoding='utf-8-sig')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    if '滑动窗口数量' not in df.columns:
        print("数据中不包含滑动窗口分析，无法进行稳定性分析")
        return
    
    print("\n=== 斜率稳定性分析 ===")
    
    # 只分析有有效滑动窗口的数据
    valid_data = df[df['滑动窗口数量'] > 0].copy()
    
    if len(valid_data) == 0:
        print("没有有效的滑动窗口数据")
        return
    
    # 处理斜率数据类型转换（稳定性分析中也需要）
    if '全段平均斜率' in valid_data.columns:
        valid_data['斜率数值'] = valid_data['全段平均斜率']
    elif '主要组合斜率' in valid_data.columns:
        valid_data['斜率数值'] = valid_data['主要组合斜率']
    else:
        print("数据中缺少斜率字段，无法进行稳定性分析")
        return
    
    # 计算稳定性指标
    valid_data['斜率稳定性'] = 1 / (1 + valid_data['斜率标准差'])  # 标准差越小，稳定性越高
    valid_data['斜率一致性'] = 1 - (valid_data['斜率变化范围'] / (abs(valid_data['斜率数值']) + 1))  # 变化范围相对于均值的比例
    
    # 按停机组合分组分析稳定性
    # 判断使用哪个停机组合列
    combination_column = '主要停机组合' if '主要停机组合' in valid_data.columns else '停机组合'
    
    stability_analysis = valid_data.groupby(combination_column).agg({
        '斜率稳定性': 'mean',
        '斜率一致性': 'mean',
        '斜率标准差': 'mean',
        '斜率变化范围': 'mean',
        '斜率数值': 'mean',
        '滑动窗口数量': 'mean'
    }).round(4)
    
    # 添加排序键
    def get_machine_count_priority_stability(combination):
        """获取停机组合的排序优先级（稳定性分析用）"""
        if '无小包机停机' in combination:
            return (0, 0, combination)  # 无停机
        elif '单个小包机停机' in combination:
            machine_num = int(combination.split('#')[0].split(': ')[1])
            return (1, machine_num, combination)  # 单个停机，按机器号排序
        elif '两个小包机同时停机' in combination:
            # 提取机器号进行排序
            machine_nums = [int(x.split('#')[0]) for x in combination.split(': ')[1].split(', ')]
            machine_nums.sort()
            return (2, tuple(machine_nums), combination)  # 两个停机
        elif '三个小包机同时停机' in combination:
            # 提取机器号进行排序
            machine_nums = [int(x.split('#')[0]) for x in combination.split(': ')[1].split(', ')]
            machine_nums.sort()
            return (3, tuple(machine_nums), combination)  # 三个停机
        elif '四个小包机同时停机' in combination:
            return (4, 0, combination)  # 四个停机
        else:
            return (5, 0, combination)  # 其他情况
    
    # 创建包含排序键的DataFrame
    stability_df = stability_analysis.reset_index()
    stability_df['排序键'] = stability_df[combination_column].apply(get_machine_count_priority_stability)
    stability_df = stability_df.sort_values('排序键')
    stability_df = stability_df.drop('排序键', axis=1)
    stability_df = stability_df.set_index(combination_column)
    
    stability_analysis = stability_df
    
    print("各停机组合的斜率稳定性排名:")
    print(stability_analysis.head(10))
    
    # 保存稳定性分析结果
    stability_analysis.to_csv('折叠机运行期间斜率稳定性分析_30.csv', encoding='utf-8-sig')
    print(f"\n斜率稳定性分析结果已保存到 '折叠机运行期间斜率稳定性分析_30.csv'")
    
    # 识别最稳定和最不稳定的停机组合
    print(f"\n最稳定的停机组合（前3）:")
    for i, (combination, data) in enumerate(stability_analysis.head(3).iterrows()):
        print(f"{i+1}. {combination}: 稳定性={data['斜率稳定性']:.4f}, 平均斜率={data['斜率数值']:.4f}")
    
    print(f"\n最不稳定的停机组合（后3）:")
    for i, (combination, data) in enumerate(stability_analysis.tail(3).iterrows()):
        print(f"{i+1}. {combination}: 稳定性={data['斜率稳定性']:.4f}, 平均斜率={data['斜率数值']:.4f}")

def analyze_slope_statistics_by_combination():
    """按不同小包机停机组合统计详细的斜率情况"""
    
    try:
        df = pd.read_csv('折叠机运行分析结果_多状态_30.csv', encoding='utf-8-sig')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    print("\n=== 按停机组合的详细斜率统计分析 ===")
    
    # 处理数据类型转换
    if '全段平均斜率' in df.columns:
        df['斜率数值'] = df['全段平均斜率']
        slope_column = '全段平均斜率'
    elif '主要组合斜率' in df.columns:
        df['斜率数值'] = df['主要组合斜率']
        slope_column = '主要组合斜率'
    else:
        print("错误：找不到斜率数据列")
        return
    
    # 判断使用哪个停机组合列
    combination_column = '主要停机组合' if '主要停机组合' in df.columns else '停机组合'
    
    # 定义停机组合排序函数（复用）
    def get_machine_count_priority(combination):
        """获取停机组合的排序优先级"""
        if '无小包机停机' in combination:
            return (0, 0, combination)
        elif '单个小包机停机' in combination:
            machine_num = int(combination.split('#')[0].split(': ')[1])
            return (1, machine_num, combination)
        elif '两个小包机同时停机' in combination:
            machine_nums = [int(x.split('#')[0]) for x in combination.split(': ')[1].split(', ')]
            machine_nums.sort()
            return (2, tuple(machine_nums), combination)
        elif '三个小包机同时停机' in combination:
            machine_nums = [int(x.split('#')[0]) for x in combination.split(': ')[1].split(', ')]
            machine_nums.sort()
            return (3, tuple(machine_nums), combination)
        elif '四个小包机同时停机' in combination:
            return (4, 0, combination)
        else:
            return (5, 0, combination)
    
    # 计算每个停机组合的详细斜率统计
    slope_statistics = []
    
    for combination, group in df.groupby(combination_column):
        slopes = group['斜率数值'].dropna()  # 去除NaN值
        
        if len(slopes) == 0:
            continue
            
        # 基本统计量
        count = len(slopes)
        mean_slope = slopes.mean()
        std_slope = slopes.std()
        min_slope = slopes.min()
        max_slope = slopes.max()
        slope_range = max_slope - min_slope
        
        # 分位数统计
        percentiles = slopes.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        q1, q2, q3 = slopes.quantile([0.25, 0.50, 0.75])
        iqr = q3 - q1  # 四分位距
        
        # 正负斜率分布
        positive_count = (slopes > 0).sum()
        negative_count = (slopes < 0).sum()
        zero_count = (slopes == 0).sum()
        positive_ratio = positive_count / count * 100
        negative_ratio = negative_count / count * 100
        zero_ratio = zero_count / count * 100
        
        # 异常值检测（使用IQR方法）
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = slopes[(slopes < lower_bound) | (slopes > upper_bound)]
        outlier_count = len(outliers)
        outlier_ratio = outlier_count / count * 100
        
        # 斜率分类统计
        very_negative = (slopes <= -0.1).sum()  # 大幅下降
        moderate_negative = ((slopes > -0.1) & (slopes <= -0.01)).sum()  # 中度下降
        slight_negative = ((slopes > -0.01) & (slopes < 0)).sum()  # 轻微下降
        stable = (slopes == 0).sum()  # 稳定
        slight_positive = ((slopes > 0) & (slopes <= 0.01)).sum()  # 轻微上升
        moderate_positive = ((slopes > 0.01) & (slopes <= 0.1)).sum()  # 中度上升
        very_positive = (slopes > 0.1).sum()  # 大幅上升
        
        # 变异系数（相对变异程度）
        cv = (std_slope / abs(mean_slope)) * 100 if mean_slope != 0 else 0
        
        # 斜率影响评估
        if abs(mean_slope) < 0.001:
            impact_level = "极轻微"
        elif abs(mean_slope) < 0.01:
            impact_level = "轻微"
        elif abs(mean_slope) < 0.05:
            impact_level = "中等"
        elif abs(mean_slope) < 0.1:
            impact_level = "较大"
        else:
            impact_level = "很大"
        
        if mean_slope > 0:
            impact_direction = "存纸率上升"
        elif mean_slope < 0:
            impact_direction = "存纸率下降"
        else:
            impact_direction = "存纸率稳定"
        
        result = {
            '停机组合': combination,
            '数据点数': count,
            '平均斜率': round(mean_slope, 6),
            '标准差': round(std_slope, 6),
            '最小斜率': round(min_slope, 6),
            '最大斜率': round(max_slope, 6),
            '斜率范围': round(slope_range, 6),
            '中位数': round(q2, 6),
            '第一四分位数(Q1)': round(q1, 6),
            '第三四分位数(Q3)': round(q3, 6),
            '四分位距(IQR)': round(iqr, 6),
            '5%分位数': round(percentiles[0.05], 6),
            '10%分位数': round(percentiles[0.10], 6),
            '90%分位数': round(percentiles[0.90], 6),
            '95%分位数': round(percentiles[0.95], 6),
            '正斜率数量': positive_count,
            '负斜率数量': negative_count,
            '零斜率数量': zero_count,
            '正斜率比例(%)': round(positive_ratio, 2),
            '负斜率比例(%)': round(negative_ratio, 2),
            '零斜率比例(%)': round(zero_ratio, 2),
            '异常值数量': outlier_count,
            '异常值比例(%)': round(outlier_ratio, 2),
            '异常值下限': round(lower_bound, 6),
            '异常值上限': round(upper_bound, 6),
            '大幅下降数量(≤-0.1)': very_negative,
            '中度下降数量(-0.1~-0.01)': moderate_negative,
            '轻微下降数量(-0.01~0)': slight_negative,
            '稳定数量(=0)': stable,
            '轻微上升数量(0~0.01)': slight_positive,
            '中度上升数量(0.01~0.1)': moderate_positive,
            '大幅上升数量(>0.1)': very_positive,
            '变异系数(%)': round(cv, 2),
            '影响程度': impact_level,
            '影响方向': impact_direction,
            '综合评估': f"{impact_level}{impact_direction}",
            '排序键': get_machine_count_priority(combination)
        }
        
        slope_statistics.append(result)
    
    # 转换为DataFrame并排序
    stats_df = pd.DataFrame(slope_statistics)
    stats_df = stats_df.sort_values('排序键')
    stats_df = stats_df.drop('排序键', axis=1)
    
    # 保存详细统计结果
    output_filename = '折叠机运行期间小包机停机组合斜率详细统计分析_30.csv'
    stats_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"详细斜率统计分析结果已保存到 '{output_filename}'")
    
    # 显示分析摘要
    print(f"\n总共分析了 {len(stats_df)} 种不同的停机组合")
    
    # 按停机机器数量分类显示摘要
    print("\n=== 按停机机器数量分类的斜率摘要 ===")
    
    def categorize_combination_for_stats(combination):
        if '无小包机停机' in combination:
            return '0个机器停机'
        elif '单个小包机停机' in combination:
            return '1个机器停机'
        elif '两个小包机同时停机' in combination:
            return '2个机器停机'
        elif '三个小包机同时停机' in combination:
            return '3个机器停机'
        elif '四个小包机同时停机' in combination:
            return '4个机器停机'
        else:
            return '其他'
    
    stats_df['停机数量类别'] = stats_df['停机组合'].apply(categorize_combination_for_stats)
    
    category_summary = stats_df.groupby('停机数量类别').agg({
        '数据点数': 'sum',
        '平均斜率': 'mean',
        '标准差': 'mean',
        '斜率范围': 'mean',
        '正斜率比例(%)': 'mean',
        '负斜率比例(%)': 'mean',
        '异常值比例(%)': 'mean',
        '变异系数(%)': 'mean'
    }).round(4)
    
    print(category_summary)
    
    # 显示最极端的情况
    print(f"\n=== 斜率极值情况 ===")
    
    # 最大正斜率
    max_positive = stats_df.loc[stats_df['最大斜率'].idxmax()]
    print(f"最大正斜率: {max_positive['停机组合']} (斜率: {max_positive['最大斜率']:.6f})")
    
    # 最小负斜率
    min_negative = stats_df.loc[stats_df['最小斜率'].idxmin()]
    print(f"最小负斜率: {min_negative['停机组合']} (斜率: {min_negative['最小斜率']:.6f})")
    
    # 斜率范围最大
    max_range = stats_df.loc[stats_df['斜率范围'].idxmax()]
    print(f"斜率变化最大: {max_range['停机组合']} (范围: {max_range['斜率范围']:.6f})")
    
    # 最稳定（标准差最小，且数据点数>10）
    stable_combinations = stats_df[stats_df['数据点数'] >= 10]
    if len(stable_combinations) > 0:
        most_stable = stable_combinations.loc[stable_combinations['标准差'].idxmin()]
        print(f"最稳定组合: {most_stable['停机组合']} (标准差: {most_stable['标准差']:.6f})")
    
    # 显示前10个最常见停机组合的详细统计
    print(f"\n=== 前10个最常见停机组合的详细斜率统计 ===")
    top_10 = stats_df.nlargest(10, '数据点数')
    
    display_cols = ['停机组合', '数据点数', '平均斜率', '标准差', '斜率范围', 
                   '正斜率比例(%)', '负斜率比例(%)', '综合评估']
    print(top_10[display_cols].to_string(index=False))
    
    return stats_df

def analyze_all_combinations_slope_statistics():
    """统计CSV中'所有停机组合'列中所有停机组合的斜率情况"""
    
    try:
        df = pd.read_csv('折叠机运行分析结果_多状态_30.csv', encoding='utf-8-sig')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    print("\n=== '所有停机组合'列中停机组合斜率统计分析 ===")
    
    if '所有停机组合' not in df.columns:
        print("错误：找不到'所有停机组合'列")
        return
    
    # 解析所有停机组合信息
    all_combinations_data = []
    
    for idx, row in df.iterrows():
        combinations_text = row['所有停机组合']
        if pd.isna(combinations_text) or combinations_text == '':
            continue
            
        # 解析组合文本，例如：
        # "无小包机停机(63.9%, 斜率:-1.1665); 单个小包机停机: 1#小包机(12.0%, 斜率:2.3403); ..."
        combinations = combinations_text.split('; ')
        
        for combination in combinations:
            try:
                # 提取组合名称、百分比和斜率
                if '斜率:' in combination and '%' in combination:
                    # 分离组合名称和统计信息
                    parts = combination.split('(')
                    if len(parts) >= 2:
                        combination_name = parts[0].strip()
                        stats_part = '('.join(parts[1:])
                        
                        # 提取百分比
                        percent_match = re.search(r'(\d+\.?\d*)%', stats_part)
                        if not percent_match:
                            continue
                        percentage = float(percent_match.group(1))
                        
                        # 提取斜率
                        slope_match = re.search(r'斜率:([-+]?\d*\.?\d+)', stats_part)
                        if not slope_match:
                            continue
                        slope = float(slope_match.group(1))
                        
                        all_combinations_data.append({
                            '时间段开始': row['开始时间'],
                            '时间段结束': row['结束时间'],
                            '时间段持续': row['持续时间'],
                            '主要停机组合': row['主要停机组合'],
                            '停机组合名称': combination_name,
                            '时间占比(%)': percentage,
                            '斜率': slope,
                            '时间段总点数': row['总时间点数'],
                            '时间段全段平均斜率': row['全段平均斜率']
                        })
            except Exception as e:
                print(f"解析组合失败: {combination}, 错误: {e}")
                continue
    
    if not all_combinations_data:
        print("未找到有效的停机组合数据")
        return
    
    combinations_df = pd.DataFrame(all_combinations_data)
    print(f"成功解析出 {len(combinations_df)} 条停机组合记录")
    
    # 按停机组合分组进行统计
    print(f"\n=== 按停机组合分类的斜率统计 ===")
    
    combination_stats = combinations_df.groupby('停机组合名称').agg({
        '斜率': ['count', 'mean', 'std', 'min', 'max', 'median'],
        '时间占比(%)': ['mean', 'sum', 'min', 'max'],
    }).round(6)
    
    # 重命名列
    combination_stats.columns = [
        '出现次数', '平均斜率', '斜率标准差', '最小斜率', '最大斜率', '斜率中位数',
        '平均时间占比(%)', '总时间占比(%)', '最小时间占比(%)', '最大时间占比(%)'
    ]
    
    # 按出现次数排序
    combination_stats = combination_stats.sort_values('出现次数', ascending=False)
    
    print(f"发现 {len(combination_stats)} 种不同的停机组合:")
    print(combination_stats.head(15))
    
    # 详细统计每个停机组合
    detailed_combination_stats = []
    
    for combination_name, group in combinations_df.groupby('停机组合名称'):
        slopes = group['斜率'].values
        percentages = group['时间占比(%)'].values
        
        # 基本统计
        count = len(slopes)
        
        # 排除斜率为0的值进行统计计算
        non_zero_slopes = slopes[slopes != 0]
        
        if len(non_zero_slopes) == 0:
            # 如果所有斜率都为0，设置默认值
            mean_slope = 0
            std_slope = 0
            min_slope = 0
            max_slope = 0
            median_slope = 0
            slope_range = 0
            q1 = 0
            q3 = 0
            iqr = 0
            lower_bound = 0
            upper_bound = 0
            outlier_ratio = 0
        else:
            # 使用非零斜率进行统计
            mean_slope = np.mean(non_zero_slopes)
            std_slope = np.std(non_zero_slopes)
            min_slope = np.min(non_zero_slopes)
            max_slope = np.max(non_zero_slopes)
            median_slope = np.median(non_zero_slopes)
            slope_range = max_slope - min_slope
            
            # 分位数（基于非零斜率）
            q1, q3 = np.percentile(non_zero_slopes, [25, 75])
            iqr = q3 - q1
            
            # 异常值检测（基于非零斜率）
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = non_zero_slopes[(non_zero_slopes < lower_bound) | (non_zero_slopes > upper_bound)]
            outlier_ratio = len(outliers) / len(non_zero_slopes) * 100
        
        # 正负斜率分布（基于所有斜率，包括0）
        positive_count = np.sum(slopes > 0)
        negative_count = np.sum(slopes < 0)
        zero_count = np.sum(slopes == 0)
        positive_ratio = positive_count / count * 100
        negative_ratio = negative_count / count * 100
        zero_ratio = zero_count / count * 100
        
        # 时间占比统计
        mean_percentage = np.mean(percentages)
        total_percentage = np.sum(percentages)
        
        # 影响评估（基于去除零值后的平均斜率）
        if abs(mean_slope) < 0.001:
            impact_level = "极轻微"
        elif abs(mean_slope) < 0.01:
            impact_level = "轻微"
        elif abs(mean_slope) < 0.05:
            impact_level = "中等"
        elif abs(mean_slope) < 0.1:
            impact_level = "较大"
        else:
            impact_level = "很大"
        
        if mean_slope > 0:
            impact_direction = "存纸率上升"
        elif mean_slope < 0:
            impact_direction = "存纸率下降"
        else:
            impact_direction = "存纸率稳定"
        
        result = {
            '停机组合': combination_name,
            '出现次数': count,
            '平均斜率': round(mean_slope, 6),
            '斜率标准差': round(std_slope, 6),
            '最小斜率': round(min_slope, 6),
            '最大斜率': round(max_slope, 6),
            '中位数斜率': round(median_slope, 6),
            '斜率范围': round(slope_range, 6),
            '第一四分位数': round(q1, 6),
            '第三四分位数': round(q3, 6),
            '四分位距': round(iqr, 6),
            '正斜率数量': positive_count,
            '负斜率数量': negative_count,
            '零斜率数量': zero_count,
            '正斜率比例(%)': round(positive_ratio, 2),
            '负斜率比例(%)': round(negative_ratio, 2),
            '零斜率比例(%)': round(zero_ratio, 2),
            '异常值比例(%)': round(outlier_ratio, 2),
            '平均时间占比(%)': round(mean_percentage, 2),
            '累计时间占比(%)': round(total_percentage, 2),
            '影响程度': impact_level,
            '影响方向': impact_direction,
            '综合评估': f"{impact_level}{impact_direction}"
        }
        
        detailed_combination_stats.append(result)
    
    # 转换为DataFrame并按停机机器数量排序
    detailed_stats_df = pd.DataFrame(detailed_combination_stats)
    
    # 添加停机机器数量用于排序
    def get_machine_count_for_sorting(combination):
        """获取停机组合的机器数量，用于排序"""
        if '无小包机停机' in combination:
            return (0, combination)  # 无停机
        elif '单个小包机停机' in combination:
            # 提取机器号
            if ': 1#小包机' in combination:
                machine_num = 1
            elif ': 2#小包机' in combination:
                machine_num = 2
            elif ': 3#小包机' in combination:
                machine_num = 3
            elif ': 4#小包机' in combination:
                machine_num = 4
            else:
                machine_num = 0
            return (1, machine_num, combination)  # 单机停机，按机器号排序
        elif '两个小包机同时停机' in combination:
            # 提取机器号进行排序
            try:
                machine_part = combination.split(': ')[1]
                machine_nums = [int(x.split('#')[0]) for x in machine_part.split(', ')]
                machine_nums.sort()
                return (2, tuple(machine_nums), combination)  # 双机停机
            except:
                return (2, (0,), combination)
        elif '三个小包机同时停机' in combination:
            # 提取机器号进行排序
            try:
                machine_part = combination.split(': ')[1]
                machine_nums = [int(x.split('#')[0]) for x in machine_part.split(', ')]
                machine_nums.sort()
                return (3, tuple(machine_nums), combination)  # 三机停机
            except:
                return (3, (0,), combination)
        elif '四个小包机同时停机' in combination:
            return (4, (1,2,3,4), combination)  # 四机停机
        else:
            return (5, combination)  # 其他情况
    
    detailed_stats_df['排序键'] = detailed_stats_df['停机组合'].apply(get_machine_count_for_sorting)
    detailed_stats_df = detailed_stats_df.sort_values('排序键')
    detailed_stats_df = detailed_stats_df.drop('排序键', axis=1)
    
    # 保存详细统计结果
    output_filename = '折叠机运行期间所有停机组合斜率详细统计分析_30.csv'
    detailed_stats_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n详细统计结果已保存到 '{output_filename}'")
    
    # 分类统计
    print(f"\n=== 按停机类型分类统计 ===")
    
    def categorize_combination_type(combination):
        if '无小包机停机' in combination:
            return '无停机'
        elif '单个小包机停机' in combination:
            return '单机停机'
        elif '两个小包机同时停机' in combination:
            return '双机停机'
        elif '三个小包机同时停机' in combination:
            return '三机停机'
        elif '四个小包机同时停机' in combination:
            return '四机停机'
        else:
            return '其他'
    
    detailed_stats_df['停机类型'] = detailed_stats_df['停机组合'].apply(categorize_combination_type)
    
    type_summary = detailed_stats_df.groupby('停机类型').agg({
        '出现次数': 'sum',
        '平均斜率': 'mean',
        '斜率标准差': 'mean',
        '累计时间占比(%)': 'sum',
        '平均时间占比(%)': 'mean'
    }).round(4)
    
    print(type_summary)
    
    # 显示极值情况
    print(f"\n=== 斜率极值情况 ===")
    
    # 最大正斜率
    max_positive = detailed_stats_df.loc[detailed_stats_df['最大斜率'].idxmax()]
    print(f"最大正斜率: {max_positive['停机组合']} (斜率: {max_positive['最大斜率']:.6f})")
    
    # 最小负斜率
    min_negative = detailed_stats_df.loc[detailed_stats_df['最小斜率'].idxmin()]
    print(f"最小负斜率: {min_negative['停机组合']} (斜率: {min_negative['最小斜率']:.6f})")
    
    # 平均斜率最大的正面影响
    max_avg_positive = detailed_stats_df[detailed_stats_df['平均斜率'] > 0].nlargest(3, '平均斜率')
    print(f"\n平均斜率最大的正面影响组合:")
    for i, (_, row) in enumerate(max_avg_positive.iterrows()):
        print(f"{i+1}. {row['停机组合']}: 平均斜率 {row['平均斜率']:.6f}, 出现 {row['出现次数']} 次")
    
    # 平均斜率最小的负面影响
    max_avg_negative = detailed_stats_df[detailed_stats_df['平均斜率'] < 0].nsmallest(3, '平均斜率')
    print(f"\n平均斜率最小的负面影响组合:")
    for i, (_, row) in enumerate(max_avg_negative.iterrows()):
        print(f"{i+1}. {row['停机组合']}: 平均斜率 {row['平均斜率']:.6f}, 出现 {row['出现次数']} 次")
    
    # 出现频率最高的组合
    print(f"\n=== 出现频率最高的停机组合 ===")
    top_frequent = detailed_stats_df.head(10)
    display_cols = ['停机组合', '出现次数', '平均斜率', '累计时间占比(%)', '综合评估']
    print(top_frequent[display_cols].to_string(index=False))
    
    # 时间占比统计
    print(f"\n=== 时间占比统计 ===")
    print(f"各类停机组合的总时间占比:")
    for stop_type in type_summary.index:
        total_time = type_summary.loc[stop_type, '累计时间占比(%)']
        avg_time = type_summary.loc[stop_type, '平均时间占比(%)']
        count = type_summary.loc[stop_type, '出现次数']
        print(f"{stop_type}: 总占比 {total_time:.2f}%, 平均占比 {avg_time:.2f}%, 出现次数 {count}")
    
    return combinations_df, detailed_stats_df

if __name__ == "__main__":
    print("开始进行折叠机正常运行时间段内小包机停机状态的详细分析...")
    df, detailed_df = analyze_folding_operating_results()
    
    print("\n" + "="*60)
    analyze_slope_stability() 
    
    print("\n" + "="*60)
    slope_stats_df = analyze_slope_statistics_by_combination()
    
    
    print("\n" + "="*60)
    combinations_df, all_combinations_stats_df = analyze_all_combinations_slope_statistics()
    
    print(f"\n=== 分析完成摘要 ===")
    print(f"✓ 基础分析结果: '折叠机运行期间停机组合影响分析汇总_30.csv'")
    print(f"✓ 稳定性分析结果: '折叠机运行期间斜率稳定性分析_30.csv'")
    print(f"✓ 详细斜率统计结果: '折叠机运行期间小包机停机组合斜率详细统计分析_30.csv'")
    print(f"✓ 所有停机组合斜率统计: '折叠机运行期间所有停机组合斜率详细统计分析_30.csv'")
    print(f"✓ 共分析了 {len(slope_stats_df) if slope_stats_df is not None else 0} 种主要停机组合") 
    print(f"✓ 共分析了 {len(all_combinations_stats_df) if all_combinations_stats_df is not None else 0} 种所有停机组合") 