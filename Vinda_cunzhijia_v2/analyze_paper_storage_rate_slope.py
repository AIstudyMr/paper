#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不同小包机损坏情况对存纸率斜率影响分析
分析小包机1-4的不同组合损坏情况下，存纸率的变化斜率
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PaperStorageRateSlopeAnalyzer:
    def __init__(self, data_file='存纸架数据汇总.csv'):
        """初始化分析器"""
        self.data_file = data_file
        self.df = None
        self.machine_combinations = []
        self.slope_results = {}
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.data_file)
        self.df['时间'] = pd.to_datetime(self.df['时间'])
        print(f"数据加载完成，共 {len(self.df)} 条记录")
        
    def identify_machine_failures(self):
        """识别小包机损坏情况"""
        print("正在识别小包机损坏情况...")
        
        # 定义小包机状态判断阈值
        speed_threshold = 10  # 速度低于10认为停机
        
        # 为每个小包机创建损坏标志
        machines = [1, 2, 3, 4]
        failure_columns = []
        
        for machine in machines:
            if machine == 1:
                speed_col = '1#小包机实际速度'
            elif machine == 2:
                speed_col = '2#小包机实际速度'
            elif machine == 3:
                speed_col = '3#小包机主机实际速度'
            else:  # machine == 4
                speed_col = '4#小包机主机实际速度'
            
            failure_col = f'小包机{machine}_损坏'
            self.df[failure_col] = (self.df[speed_col] < speed_threshold) | self.df[speed_col].isna()
            failure_columns.append(failure_col)
        
        # 创建损坏组合标识
        self.df['损坏组合'] = self.df[failure_columns].apply(
            lambda x: ''.join([str(i+1) for i, val in enumerate(x) if val]), axis=1
        )
        self.df['损坏组合'] = self.df['损坏组合'].replace('', '正常')
        
        # 新增：计算同时损坏的机器数量
        self.df['同时损坏数量'] = self.df[failure_columns].sum(axis=1)
        
        # 创建同时损坏分类
        def categorize_simultaneous_failures(count):
            if count == 0:
                return '0台损坏(正常)'
            elif count == 1:
                return '1台损坏'
            elif count == 2:
                return '2台同时损坏'
            elif count == 3:
                return '3台同时损坏'
            else:
                return '4台同时损坏'
        
        self.df['同时损坏类别'] = self.df['同时损坏数量'].apply(categorize_simultaneous_failures)
        
        print(f"识别到的损坏组合：{self.df['损坏组合'].unique()}")
        print(f"同时损坏情况分布：")
        for category, count in self.df['同时损坏类别'].value_counts().items():
            print(f"  {category}: {count} 条记录 ({count/len(self.df)*100:.1f}%)")
        
    def calculate_storage_rate_slope(self, window_minutes=1):
        """计算存纸率斜率"""
        print(f"正在计算存纸率斜率（时间窗口：{window_minutes}分钟）...")
        
        slopes = []
        time_points = []
        combinations = []
        
        # 按时间排序
        self.df = self.df.sort_values('时间').reset_index(drop=True)
        
        # 滑动窗口计算斜率
        window_size = window_minutes * 6  # 假设每10秒一个数据点
        
        for i in range(window_size, len(self.df)):
            window_data = self.df.iloc[i-window_size:i]
            
            # 检查窗口内数据的有效性
            if window_data['存纸率'].notna().sum() < window_size * 0.8:
                continue
                
            # 计算时间序列（转换为数值）
            time_numeric = np.arange(len(window_data))
            storage_rate = window_data['存纸率'].values
            
            # 去除NaN值
            valid_indices = ~np.isnan(storage_rate)
            if valid_indices.sum() < 5:  # 至少需要5个有效点
                continue
                
            time_valid = time_numeric[valid_indices]
            storage_valid = storage_rate[valid_indices]
            
            # 线性回归计算斜率
            if len(time_valid) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_valid, storage_valid)
                
                slopes.append(slope)
                time_points.append(window_data['时间'].iloc[-1])
                combinations.append(window_data['损坏组合'].mode().iloc[0])
        
        # 创建斜率数据框，同时包含同时损坏信息
        slope_df = pd.DataFrame({
            '时间': time_points,
            '存纸率斜率': slopes,
            '损坏组合': combinations
        })
        
        # 添加同时损坏类别信息
        slope_df['同时损坏类别'] = slope_df['损坏组合'].apply(
            lambda x: f"{len(x)}台同时损坏" if x != '正常' else '0台损坏(正常)'
        )
        
        return slope_df
    
    def analyze_slope_by_combination(self, slope_df):
        """按损坏组合分析斜率"""
        print("正在分析不同损坏组合的斜率影响...")
        
        results = {}
        
        # 新增：按同时损坏数量分析
        simultaneous_results = {}
        
        for combination in slope_df['损坏组合'].unique():
            combo_data = slope_df[slope_df['损坏组合'] == combination]
            
            if len(combo_data) < 10:  # 至少需要10个数据点
                continue
                
            slopes = combo_data['存纸率斜率'].values
            
            # 计算百分位数
            percentiles = [5, 10, 25, 75, 90, 95]
            percentile_values = np.percentile(slopes, percentiles)
            
            # 计算去除异常值后的范围（去除极端值后的范围）
            q25, q75 = np.percentile(slopes, [25, 75])
            iqr = q75 - q25
            # 使用1.5倍IQR规则识别异常值
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            normal_slopes = slopes[(slopes >= lower_bound) & (slopes <= upper_bound)]
            
            results[combination] = {
                '数据点数': len(slopes),
                '平均斜率': np.mean(slopes),
                '斜率标准差': np.std(slopes),
                '斜率中位数': np.median(slopes),
                '斜率最大值': np.max(slopes),
                '斜率最小值': np.min(slopes),
                '斜率范围': np.max(slopes) - np.min(slopes),
                '正斜率比例': (slopes > 0).mean(),
                '负斜率比例': (slopes < 0).mean(),
                # 新增的详细统计
                '5%分位数': percentile_values[0],
                '10%分位数': percentile_values[1],
                '25%分位数': percentile_values[2],
                '75%分位数': percentile_values[3],
                '90%分位数': percentile_values[4],
                '95%分位数': percentile_values[5],
                'IQR(四分位距)': iqr,
                '去除异常后斜率下限': np.min(normal_slopes) if len(normal_slopes) > 0 else np.nan,
                '去除异常后斜率上限': np.max(normal_slopes) if len(normal_slopes) > 0 else np.nan,
                '绝大多数值数量': len(normal_slopes),
                '绝大多数值比例': len(normal_slopes) / len(slopes),
                '绝大多数值平均': np.mean(normal_slopes) if len(normal_slopes) > 0 else np.nan,
                '绝大多数值标准差': np.std(normal_slopes) if len(normal_slopes) > 0 else np.nan,
                '异常值数量': len(slopes) - len(normal_slopes),
                '异常值比例': 1 - len(normal_slopes) / len(slopes)
            }
        
        # 按同时损坏数量进行分析
        for category in slope_df['同时损坏类别'].unique():
            category_data = slope_df[slope_df['同时损坏类别'] == category]
            
            if len(category_data) < 10:  # 至少需要10个数据点
                continue
                
            slopes = category_data['存纸率斜率'].values
            
            # 计算百分位数
            percentiles = [5, 10, 25, 75, 90, 95]
            percentile_values = np.percentile(slopes, percentiles)
            
            # 计算去除异常值后的范围（去除极端值后的范围）
            q25, q75 = np.percentile(slopes, [25, 75])
            iqr = q75 - q25
            # 使用1.5倍IQR规则识别异常值
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            normal_slopes = slopes[(slopes >= lower_bound) & (slopes <= upper_bound)]
            
            simultaneous_results[category] = {
                '数据点数': len(slopes),
                '平均斜率': np.mean(slopes),
                '斜率标准差': np.std(slopes),
                '斜率中位数': np.median(slopes),
                '斜率最大值': np.max(slopes),
                '斜率最小值': np.min(slopes),
                '斜率范围': np.max(slopes) - np.min(slopes),
                '正斜率比例': (slopes > 0).mean(),
                '负斜率比例': (slopes < 0).mean(),
                # 详细统计
                '5%分位数': percentile_values[0],
                '10%分位数': percentile_values[1],
                '25%分位数': percentile_values[2],
                '75%分位数': percentile_values[3],
                '90%分位数': percentile_values[4],
                '95%分位数': percentile_values[5],
                'IQR(四分位距)': iqr,
                '去除异常后斜率下限': np.min(normal_slopes) if len(normal_slopes) > 0 else np.nan,
                '去除异常后斜率上限': np.max(normal_slopes) if len(normal_slopes) > 0 else np.nan,
                '绝大多数值数量': len(normal_slopes),
                '绝大多数值比例': len(normal_slopes) / len(slopes),
                '绝大多数值平均': np.mean(normal_slopes) if len(normal_slopes) > 0 else np.nan,
                '绝大多数值标准差': np.std(normal_slopes) if len(normal_slopes) > 0 else np.nan,
                '异常值数量': len(slopes) - len(normal_slopes),
                '异常值比例': 1 - len(normal_slopes) / len(slopes)
            }
        
        self.slope_results = results
        self.simultaneous_results = simultaneous_results  # 保存同时损坏分析结果
        return results, simultaneous_results
    
    def create_visualizations(self, slope_df):
        """创建可视化图表"""
        print("正在创建可视化图表...")
        
        # 创建图表布局 - 增加到3x3布局以包含同时损坏分析
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('小包机损坏对存纸率斜率影响分析 - 重点分析同时损坏情况', fontsize=18, fontweight='bold')
        
        # 1. 斜率分布箱线图
        ax1 = axes[0, 0]
        slope_df_filtered = slope_df[slope_df['损坏组合'].map(
            lambda x: x in self.slope_results.keys()
        )]
        
        sns.boxplot(data=slope_df_filtered, x='损坏组合', y='存纸率斜率', ax=ax1)
        ax1.set_title('不同损坏组合的存纸率斜率分布')
        ax1.set_xlabel('损坏组合')
        ax1.set_ylabel('存纸率斜率')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 平均斜率对比
        ax2 = axes[0, 1]
        combinations = list(self.slope_results.keys())
        mean_slopes = [self.slope_results[combo]['平均斜率'] for combo in combinations]
        
        bars = ax2.bar(combinations, mean_slopes, 
                       color=['green' if x > 0 else 'red' for x in mean_slopes])
        ax2.set_title('不同损坏组合的平均存纸率斜率')
        ax2.set_xlabel('损坏组合')
        ax2.set_ylabel('平均斜率')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, mean_slopes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 3. 斜率范围分析
        ax3 = axes[1, 0]
        ranges = [self.slope_results[combo]['斜率范围'] for combo in combinations]
        std_devs = [self.slope_results[combo]['斜率标准差'] for combo in combinations]
        
        x_pos = np.arange(len(combinations))
        ax3.bar(x_pos - 0.2, ranges, 0.4, label='斜率范围', alpha=0.7)
        ax3.bar(x_pos + 0.2, std_devs, 0.4, label='标准差', alpha=0.7)
        ax3.set_title('不同损坏组合的斜率变异程度')
        ax3.set_xlabel('损坏组合')
        ax3.set_ylabel('变异程度')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(combinations, rotation=45)
        ax3.legend()
        
        # 4. 正负斜率比例
        ax4 = axes[1, 1]
        pos_ratios = [self.slope_results[combo]['正斜率比例'] for combo in combinations]
        neg_ratios = [self.slope_results[combo]['负斜率比例'] for combo in combinations]
        
        x_pos = np.arange(len(combinations))
        ax4.bar(x_pos, pos_ratios, label='正斜率比例', color='green', alpha=0.7)
        ax4.bar(x_pos, neg_ratios, bottom=pos_ratios, label='负斜率比例', color='red', alpha=0.7)
        ax4.set_title('不同损坏组合的正负斜率分布')
        ax4.set_xlabel('损坏组合')
        ax4.set_ylabel('比例')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(combinations, rotation=45)
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        # 5. 百分位数分布图
        ax5 = axes[0, 2]
        percentiles_data = []
        labels = []
        for combo in combinations:
            percentiles_data.append([
                self.slope_results[combo]['5%分位数'],
                self.slope_results[combo]['25%分位数'],
                self.slope_results[combo]['斜率中位数'],
                self.slope_results[combo]['75%分位数'],
                self.slope_results[combo]['95%分位数']
            ])
            labels.append(combo)
        
        percentiles_data = np.array(percentiles_data)
        x_pos = np.arange(len(combinations))
        
        # 绘制百分位数线图
        percentile_labels = ['5%', '25%', '50%', '75%', '95%']
        colors = ['red', 'orange', 'blue', 'orange', 'red']
        for i, (label, color) in enumerate(zip(percentile_labels, colors)):
            ax5.plot(x_pos, percentiles_data[:, i], marker='o', label=f'{label}分位数', color=color)
        
        ax5.set_title('不同损坏组合的百分位数分布')
        ax5.set_xlabel('损坏组合')
        ax5.set_ylabel('斜率值')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(combinations, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 去除异常值后数值vs异常值比例
        ax6 = axes[1, 2]
        normal_ratios = [self.slope_results[combo]['绝大多数值比例'] for combo in combinations]
        outlier_ratios = [self.slope_results[combo]['异常值比例'] for combo in combinations]
        
        x_pos = np.arange(len(combinations))
        width = 0.35
        
        bars1 = ax6.bar(x_pos - width/2, normal_ratios, width, label='去除异常值后数值', color='lightblue', alpha=0.8)
        bars2 = ax6.bar(x_pos + width/2, outlier_ratios, width, label='异常值', color='lightcoral', alpha=0.8)
        
        ax6.set_title('去除异常值后数值与异常值比例对比')
        ax6.set_xlabel('损坏组合')
        ax6.set_ylabel('比例')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(combinations, rotation=45)
        ax6.legend()
        ax6.set_ylim(0, 1)
        
        # 在柱子上添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}', ha='center', va='bottom')
        
        # === 新增：同时损坏分析图表 ===
        
        # 7. 同时损坏数量vs平均斜率
        ax7 = axes[2, 0]
        if hasattr(self, 'simultaneous_results') and self.simultaneous_results:
            sim_categories = list(self.simultaneous_results.keys())
            sim_mean_slopes = [self.simultaneous_results[cat]['平均斜率'] for cat in sim_categories]
            
            # 按照损坏数量排序
            sorted_data = sorted(zip(sim_categories, sim_mean_slopes), 
                               key=lambda x: int(x[0][0]) if x[0][0].isdigit() else -1)
            sim_categories_sorted, sim_mean_slopes_sorted = zip(*sorted_data)
            
            bars = ax7.bar(sim_categories_sorted, sim_mean_slopes_sorted, 
                          color=['blue' if x > 0 else 'red' for x in sim_mean_slopes_sorted],
                          alpha=0.7)
            ax7.set_title('同时损坏数量对平均斜率的影响', fontweight='bold')
            ax7.set_xlabel('同时损坏的机器数量')
            ax7.set_ylabel('平均斜率')
            ax7.tick_params(axis='x', rotation=45)
            ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # 添加数值标签
            for bar, value in zip(bars, sim_mean_slopes_sorted):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.4f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 8. 同时损坏数量vs斜率范围
        ax8 = axes[2, 1]
        if hasattr(self, 'simultaneous_results') and self.simultaneous_results:
            sim_ranges = [self.simultaneous_results[cat]['斜率范围'] for cat in sim_categories_sorted]
            sim_std_devs = [self.simultaneous_results[cat]['斜率标准差'] for cat in sim_categories_sorted]
            
            x_pos = np.arange(len(sim_categories_sorted))
            width = 0.35
            
            bars1 = ax8.bar(x_pos - width/2, sim_ranges, width, label='斜率范围', alpha=0.7, color='lightcoral')
            bars2 = ax8.bar(x_pos + width/2, sim_std_devs, width, label='标准差', alpha=0.7, color='lightblue')
            
            ax8.set_title('同时损坏数量对斜率变异性的影响', fontweight='bold')
            ax8.set_xlabel('同时损坏的机器数量')
            ax8.set_ylabel('变异程度')
            ax8.set_xticks(x_pos)
            ax8.set_xticklabels(sim_categories_sorted, rotation=45)
            ax8.legend()
        
        # 9. 同时损坏趋势线图
        ax9 = axes[2, 2]
        if hasattr(self, 'simultaneous_results') and self.simultaneous_results:
            # 提取损坏数量（数字）
            damage_counts = []
            mean_slopes_for_trend = []
            std_devs_for_trend = []
            
            for cat in sim_categories_sorted:
                if cat[0].isdigit():
                    damage_counts.append(int(cat[0]))
                    mean_slopes_for_trend.append(self.simultaneous_results[cat]['平均斜率'])
                    std_devs_for_trend.append(self.simultaneous_results[cat]['斜率标准差'])
            
            if damage_counts:
                ax9.plot(damage_counts, mean_slopes_for_trend, marker='o', linewidth=2, 
                        markersize=8, label='平均斜率', color='red')
                ax9.fill_between(damage_counts, 
                               [m-s for m,s in zip(mean_slopes_for_trend, std_devs_for_trend)],
                               [m+s for m,s in zip(mean_slopes_for_trend, std_devs_for_trend)],
                               alpha=0.3, color='red', label='±1标准差范围')
                
                ax9.set_title('同时损坏数量与斜率的趋势关系', fontweight='bold')
                ax9.set_xlabel('同时损坏的机器数量')
                ax9.set_ylabel('平均斜率')
                ax9.grid(True, alpha=0.3)
                ax9.legend()
                ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'存纸率斜率影响分析_{timestamp}.png', dpi=300, bbox_inches='tight')
        print(f"图表已保存为: 存纸率斜率影响分析_{timestamp}.png")
        
        plt.show()
    
    def generate_report(self):
        """生成分析报告"""
        print("正在生成分析报告...")
        
        report = []
        report.append("=" * 80)
        report.append("小包机同时损坏对存纸率斜率影响分析报告")
        report.append("=" * 80)
        report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"数据总量: {len(self.df)} 条记录")
        report.append("")
        
        # 新增：同时损坏情况概述
        report.append("同时损坏情况概述:")
        report.append("-" * 40)
        if hasattr(self, 'simultaneous_results') and self.simultaneous_results:
            for category in sorted(self.simultaneous_results.keys(), 
                                 key=lambda x: int(x[0]) if x[0].isdigit() else -1):
                results = self.simultaneous_results[category]
                report.append(f"{category}: {results['数据点数']} 个数据点")
        report.append("")
        
        # 按影响程度排序
        sorted_combinations = sorted(self.slope_results.items(), 
                                   key=lambda x: abs(x[1]['平均斜率']), 
                                   reverse=True)
        
        report.append("各损坏组合对存纸率斜率的影响（按影响程度排序）:")
        report.append("-" * 60)
        
        for i, (combination, results) in enumerate(sorted_combinations, 1):
            if combination == '正常':
                combo_desc = "所有小包机正常运行"
            else:
                combo_desc = f"小包机 {', '.join(combination)} 号损坏"
            
            report.append(f"{i}. {combo_desc}")
            report.append(f"   - 数据点数: {results['数据点数']}")
            report.append(f"   - 平均斜率: {results['平均斜率']:.6f}")
            report.append(f"   - 斜率标准差: {results['斜率标准差']:.6f}")
            report.append(f"   - 斜率中位数: {results['斜率中位数']:.6f}")
            report.append(f"   - 斜率范围: {results['斜率范围']:.6f} ")
            report.append(f"   - 斜率最小: {results['斜率最小值']:.6f}")
            report.append(f"   - 斜率最大: {results['斜率最大值']:.6f}")
            report.append(f"   - 正斜率比例: {results['正斜率比例']:.2%}")
            report.append(f"   - 负斜率比例: {results['负斜率比例']:.2%}")
            
            # 新增详细统计信息
            report.append(f"   【详细分布统计】")
            report.append(f"   - 百分位数分布:")
            report.append(f"     * 5%分位数: {results['5%分位数']:.6f}")
            report.append(f"     * 25%分位数: {results['25%分位数']:.6f}")
            report.append(f"     * 75%分位数: {results['75%分位数']:.6f}")
            report.append(f"     * 95%分位数: {results['95%分位数']:.6f}")
            report.append(f"   - 四分位距(IQR): {results['IQR(四分位距)']:.6f}")
            report.append(f"   - 去除异常后斜率下限: {results['去除异常后斜率下限']:.6f}")
            report.append(f"   - 去除异常后斜率上限: {results['去除异常后斜率上限']:.6f}")
            report.append(f"   - 去除异常值后数值占比: {results['绝大多数值比例']:.2%} ({results['绝大多数值数量']}个)")
            report.append(f"   - 去除异常值后斜率平均值: {results['绝大多数值平均']:.6f}")
            report.append(f"   - 去除异常值后斜率标准差: {results['绝大多数值标准差']:.6f}")
            report.append(f"   - 异常值占比: {results['异常值比例']:.2%} ({results['异常值数量']}个)")
            
            # 影响评估
            if results['平均斜率'] > 0.001:
                impact = "存纸率明显上升趋势"
            elif results['平均斜率'] < -0.001:
                impact = "存纸率明显下降趋势"
            else:
                impact = "存纸率相对稳定"
            
            report.append(f"   - 影响评估: {impact}")
            report.append("")
        
        # 关键发现
        report.append("关键发现:")
        report.append("-" * 30)
        
        # 找出影响最大的损坏组合
        max_impact_combo = max(self.slope_results.items(), 
                              key=lambda x: abs(x[1]['平均斜率']))
        report.append(f"1. 对存纸率斜率影响最大的是: {max_impact_combo[0]} "
                     f"(平均斜率: {max_impact_combo[1]['平均斜率']:.6f})")
        
        # 找出最稳定的情况
        most_stable = min(self.slope_results.items(), 
                         key=lambda x: x[1]['斜率标准差'])
        report.append(f"2. 存纸率最稳定的情况是: {most_stable[0]} "
                     f"(标准差: {most_stable[1]['斜率标准差']:.6f})")
        
        # 去除异常后斜率范围分析
        widest_range_combo = max(self.slope_results.items(), 
                               key=lambda x: x[1]['去除异常后斜率上限'] - x[1]['去除异常后斜率下限'])
        narrowest_range_combo = min(self.slope_results.items(), 
                                  key=lambda x: x[1]['去除异常后斜率上限'] - x[1]['去除异常后斜率下限'])
        
        report.append(f"3. 去除异常后斜率范围最宽的损坏组合: {widest_range_combo[0]} "
                     f"(范围: [{widest_range_combo[1]['去除异常后斜率下限']:.6f}, "
                     f"{widest_range_combo[1]['去除异常后斜率上限']:.6f}])")
        
        report.append(f"4. 去除异常后斜率范围最窄的损坏组合: {narrowest_range_combo[0]} "
                     f"(范围: [{narrowest_range_combo[1]['去除异常后斜率下限']:.6f}, "
                     f"{narrowest_range_combo[1]['去除异常后斜率上限']:.6f}])")
        
        # 异常值最多的组合
        most_outliers_combo = max(self.slope_results.items(), 
                                key=lambda x: x[1]['异常值比例'])
        report.append(f"5. 异常值比例最高的损坏组合: {most_outliers_combo[0]} "
                     f"(异常值比例: {most_outliers_combo[1]['异常值比例']:.2%})")
        
        # 正常状态对比
        if '正常' in self.slope_results:
            normal_slope = self.slope_results['正常']['平均斜率']
            normal_range = (self.slope_results['正常']['去除异常后斜率上限'] - 
                          self.slope_results['正常']['去除异常后斜率下限'])
            report.append(f"6. 正常状态下的平均斜率: {normal_slope:.6f}")
            report.append(f"7. 正常状态下去除异常后斜率范围宽度: {normal_range:.6f}")
            
            # 对比其他状态
            worse_combos = [combo for combo, results in self.slope_results.items() 
                           if combo != '正常' and abs(results['平均斜率']) > abs(normal_slope)]
            if worse_combos:
                report.append(f"8. 比正常状态影响更大的损坏组合: {', '.join(worse_combos)}")
                
        # 总体分析
        report.append("")
        report.append("总体分析:")
        report.append("-" * 20)
        all_normal_ratios = [results['绝大多数值比例'] for results in self.slope_results.values()]
        avg_normal_ratio = np.mean(all_normal_ratios)
        report.append(f"• 平均去除异常值后数值占比: {avg_normal_ratio:.2%}")
        
        all_ranges = [results['斜率范围'] for results in self.slope_results.values()]
        avg_range = np.mean(all_ranges)
        report.append(f"• 平均斜率范围: {avg_range:.6f}")
        
        all_normal_ranges = [(results['去除异常后斜率上限'] - results['去除异常后斜率下限']) 
                           for results in self.slope_results.values()]
        avg_normal_range = np.mean(all_normal_ranges)
        report.append(f"• 平均去除异常后斜率范围宽度: {avg_normal_range:.6f}")
        
        # === 新增：同时损坏分析专门部分 ===
        if hasattr(self, 'simultaneous_results') and self.simultaneous_results:
            report.append("")
            report.append("=" * 60)
            report.append("重点分析：同时损坏数量对存纸率斜率的影响")
            report.append("=" * 60)
            
            # 按损坏数量排序的同时损坏分析
            sorted_sim_results = sorted(self.simultaneous_results.items(), 
                                      key=lambda x: int(x[0][0]) if x[0][0].isdigit() else -1)
            
            for i, (category, results) in enumerate(sorted_sim_results, 1):
                report.append(f"{i}. {category}")
                report.append(f"   - 数据点数: {results['数据点数']}")
                report.append(f"   - 平均斜率: {results['平均斜率']:.6f}")
                report.append(f"   - 斜率标准差: {results['斜率标准差']:.6f}")
                report.append(f"   - 斜率范围: {results['斜率范围']:.6f}")
                report.append(f"   - 去除异常后斜率范围: [{results['去除异常后斜率下限']:.6f}, {results['去除异常后斜率上限']:.6f}]")
                report.append(f"   - 异常值比例: {results['异常值比例']:.2%}")
                
                # 影响评估
                if results['平均斜率'] > 0.001:
                    impact = "存纸率明显上升趋势"
                elif results['平均斜率'] < -0.001:
                    impact = "存纸率明显下降趋势"
                else:
                    impact = "存纸率相对稳定"
                
                report.append(f"   - 影响评估: {impact}")
                report.append("")
            
            # 同时损坏关键发现
            report.append("同时损坏关键发现:")
            report.append("-" * 30)
            
            # 找出影响最严重的同时损坏数量
            max_impact_sim = max(sorted_sim_results, key=lambda x: abs(x[1]['平均斜率']))
            report.append(f"1. 对斜率影响最大: {max_impact_sim[0]} (平均斜率: {max_impact_sim[1]['平均斜率']:.6f})")
            
            # 最稳定的同时损坏情况
            most_stable_sim = min(sorted_sim_results, key=lambda x: x[1]['斜率标准差'])
            report.append(f"2. 最稳定的情况: {most_stable_sim[0]} (标准差: {most_stable_sim[1]['斜率标准差']:.6f})")
            
            # 分析同时损坏数量的递增影响
            if len(sorted_sim_results) >= 3:
                # 比较0台、1台、多台损坏的差异
                try:
                    normal_slope = next(r[1]['平均斜率'] for r in sorted_sim_results if '0台' in r[0])
                    single_slope = next(r[1]['平均斜率'] for r in sorted_sim_results if '1台' in r[0])
                    
                    multi_slopes = [r[1]['平均斜率'] for r in sorted_sim_results 
                                  if r[0][0].isdigit() and int(r[0][0]) >= 2]
                    
                    if multi_slopes:
                        avg_multi_slope = np.mean(multi_slopes)
                        
                        report.append(f"3. 损坏影响递增分析:")
                        report.append(f"   - 正常状态平均斜率: {normal_slope:.6f}")
                        report.append(f"   - 单台损坏平均斜率: {single_slope:.6f}")
                        report.append(f"   - 多台同时损坏平均斜率: {avg_multi_slope:.6f}")
                        
                        single_impact = abs(single_slope - normal_slope)
                        multi_impact = abs(avg_multi_slope - normal_slope)
                        
                        report.append(f"   - 单台损坏影响程度: {single_impact:.6f}")
                        report.append(f"   - 多台同时损坏影响程度: {multi_impact:.6f}")
                        
                        if multi_impact > single_impact * 1.5:
                            report.append(f"   - 结论: 多台同时损坏的影响显著大于单台损坏")
                        else:
                            report.append(f"   - 结论: 多台同时损坏的影响与单台损坏相近")
                except:
                    pass
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_text = '\n'.join(report)
        
        with open(f'存纸率斜率影响分析报告_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"分析报告已保存为: 存纸率斜率影响分析报告_{timestamp}.txt")
        print("\n" + report_text)
        
        return report_text
    
    def run_complete_analysis(self, window_minutes=5):
        """运行完整分析"""
        print("开始进行完整的存纸率斜率影响分析...")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 识别机器故障
        self.identify_machine_failures()
        
        # 3. 计算斜率
        slope_df = self.calculate_storage_rate_slope(window_minutes=window_minutes)
        print(f"计算得到 {len(slope_df)} 个斜率数据点")
        
        # 4. 分析结果
        self.analyze_slope_by_combination(slope_df)
        
        # 5. 创建可视化
        self.create_visualizations(slope_df)
        
        # 6. 生成报告
        self.generate_report()
        
        print("=" * 60)
        print("分析完成！")
        
        return slope_df, self.slope_results

def main():
    """主函数"""
    try:
        # 创建分析器实例
        analyzer = PaperStorageRateSlopeAnalyzer()
        
        # 运行完整分析
        slope_data, results = analyzer.run_complete_analysis(window_minutes=1)  # 设置为1分钟窗口
        
        # 返回结果供进一步分析
        return analyzer, slope_data, results
        
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    analyzer, slope_data, results = main() 