import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import os
from scipy import stats
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_output_dir():
    """创建输出目录"""
    output_dir = "高级裁切机速度分析"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def detect_anomalies(speed_data, threshold=2):
    """检测异常值（基于Z-score）"""
    z_scores = np.abs(stats.zscore(speed_data))
    anomalies = z_scores > threshold
    return anomalies

def plot_advanced_daily_analysis(df_day, date_str, output_dir):
    """绘制高级每日分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'{date_str} 裁切机详细分析', fontsize=18, fontweight='bold')
    
    # 子图1: 速度曲线和异常检测
    ax1 = axes[0, 0]
    ax1.plot(df_day['时间'], df_day['裁切机实际速度'], 
             linewidth=1, color='#2E86AB', alpha=0.8, label='正常速度')
    
    # 异常检测
    anomalies = detect_anomalies(df_day['裁切机实际速度'])
    if np.any(anomalies):
        anomaly_data = df_day[anomalies]
        ax1.scatter(anomaly_data['时间'], anomaly_data['裁切机实际速度'], 
                   color='red', s=20, alpha=0.7, label=f'异常点 ({np.sum(anomalies)}个)')
    
    ax1.set_title('速度曲线与异常检测')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('速度')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子图2: 速度分布直方图
    ax2 = axes[0, 1]
    ax2.hist(df_day['裁切机实际速度'], bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
    ax2.axvline(df_day['裁切机实际速度'].mean(), color='red', linestyle='--', 
                label=f'平均值: {df_day["裁切机实际速度"].mean():.2f}')
    ax2.axvline(df_day['裁切机实际速度'].median(), color='orange', linestyle='--', 
                label=f'中位数: {df_day["裁切机实际速度"].median():.2f}')
    ax2.set_title('速度分布')
    ax2.set_xlabel('速度')
    ax2.set_ylabel('频次')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 滑动平均和趋势
    ax3 = axes[1, 0]
    
    # 计算不同窗口的滑动平均
    window_sizes = [60, 300, 900]  # 1分钟、5分钟、15分钟
    colors = ['#F18F01', '#C73E1D', '#8E44AD']
    
    for i, window in enumerate(window_sizes):
        if len(df_day) > window:
            rolling_mean = df_day['裁切机实际速度'].rolling(window=window).mean()
            ax3.plot(df_day['时间'], rolling_mean, 
                    color=colors[i], linewidth=2, alpha=0.8, 
                    label=f'{window//60}分钟滑动平均' if window >= 60 else f'{window}秒滑动平均')
    
    ax3.set_title('滑动平均趋势')
    ax3.set_xlabel('时间')
    ax3.set_ylabel('速度')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 速度变化率
    ax4 = axes[1, 1]
    speed_diff = df_day['裁切机实际速度'].diff()
    ax4.plot(df_day['时间'][1:], speed_diff[1:], 
             linewidth=0.5, color='#27AE60', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('速度变化率')
    ax4.set_xlabel('时间')
    ax4.set_ylabel('速度变化量')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"{date_str}_高级分析.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def generate_daily_report(df_day, date_str):
    """生成每日统计报告"""
    report = {
        '日期': date_str,
        '数据点数': len(df_day),
        '平均速度': df_day['裁切机实际速度'].mean(),
        '中位数速度': df_day['裁切机实际速度'].median(),
        '最大速度': df_day['裁切机实际速度'].max(),
        '最小速度': df_day['裁切机实际速度'].min(),
        '标准差': df_day['裁切机实际速度'].std(),
        '四分位距': df_day['裁切机实际速度'].quantile(0.75) - df_day['裁切机实际速度'].quantile(0.25),
        '异常点数量': np.sum(detect_anomalies(df_day['裁切机实际速度'])),
        '零速度时间点': np.sum(df_day['裁切机实际速度'] == 0),
        '负速度时间点': np.sum(df_day['裁切机实际速度'] < 0),
        '高速运行时间点': np.sum(df_day['裁切机实际速度'] > 110),
        '稳定运行比例': np.sum((df_day['裁切机实际速度'] >= 100) & 
                           (df_day['裁切机实际速度'] <= 115)) / len(df_day) * 100
    }
    return report

def create_summary_dashboard(all_reports, output_dir):
    """创建总结仪表板"""
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('裁切机运行总结仪表板', fontsize=20, fontweight='bold')
    
    # 转换为DataFrame便于绘图
    df_reports = pd.DataFrame(all_reports)
    df_reports['日期'] = pd.to_datetime(df_reports['日期'])
    
    # 子图1: 每日平均速度趋势
    ax1 = axes[0, 0]
    ax1.plot(df_reports['日期'], df_reports['平均速度'], 
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_title('每日平均速度趋势')
    ax1.set_ylabel('平均速度')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 稳定运行比例
    ax2 = axes[0, 1]
    bars = ax2.bar(range(len(df_reports)), df_reports['稳定运行比例'], 
                   color='#27AE60', alpha=0.7)
    ax2.set_title('每日稳定运行比例')
    ax2.set_ylabel('稳定运行比例 (%)')
    ax2.set_xticks(range(len(df_reports)))
    ax2.set_xticklabels([d.strftime('%m-%d') for d in df_reports['日期']], rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 子图3: 异常点统计
    ax3 = axes[0, 2]
    ax3.bar(range(len(df_reports)), df_reports['异常点数量'], 
            color='#C73E1D', alpha=0.7)
    ax3.set_title('每日异常点数量')
    ax3.set_ylabel('异常点数量')
    ax3.set_xticks(range(len(df_reports)))
    ax3.set_xticklabels([d.strftime('%m-%d') for d in df_reports['日期']], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 速度范围（最大-最小）
    ax4 = axes[1, 0]
    speed_range = df_reports['最大速度'] - df_reports['最小速度']
    ax4.bar(range(len(df_reports)), speed_range, 
            color='#F18F01', alpha=0.7)
    ax4.set_title('每日速度波动范围')
    ax4.set_ylabel('速度范围')
    ax4.set_xticks(range(len(df_reports)))
    ax4.set_xticklabels([d.strftime('%m-%d') for d in df_reports['日期']], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 零速度和负速度时间点
    ax5 = axes[1, 1]
    width = 0.35
    x = np.arange(len(df_reports))
    ax5.bar(x - width/2, df_reports['零速度时间点'], width, 
            label='零速度', color='#8E44AD', alpha=0.7)
    ax5.bar(x + width/2, df_reports['负速度时间点'], width, 
            label='负速度', color='#A23B72', alpha=0.7)
    ax5.set_title('每日停机时间统计')
    ax5.set_ylabel('时间点数量')
    ax5.set_xticks(x)
    ax5.set_xticklabels([d.strftime('%m-%d') for d in df_reports['日期']], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 子图6: 综合评分（基于稳定运行比例和异常点少）
    ax6 = axes[1, 2]
    # 计算综合评分：稳定运行比例高且异常点少的得分高
    max_anomalies = df_reports['异常点数量'].max()
    normalized_anomalies = 1 - (df_reports['异常点数量'] / max_anomalies if max_anomalies > 0 else 0)
    composite_score = (df_reports['稳定运行比例'] / 100) * 0.7 + normalized_anomalies * 0.3
    
    bars = ax6.bar(range(len(df_reports)), composite_score * 100, 
                   color='#E67E22', alpha=0.7)
    ax6.set_title('每日运行质量评分')
    ax6.set_ylabel('质量评分 (%)')
    ax6.set_xticks(range(len(df_reports)))
    ax6.set_xticklabels([d.strftime('%m-%d') for d in df_reports['日期']], rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # 添加评分标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存仪表板
    dashboard_path = os.path.join(output_dir, "运行质量仪表板.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return dashboard_path

def main():
    print("正在进行高级裁切机速度分析...")
    
    # 读取数据
    df = pd.read_csv('存纸架数据汇总.csv', usecols=['时间', '裁切机实际速度'])
    df['时间'] = pd.to_datetime(df['时间'])
    df['日期'] = df['时间'].dt.date
    
    # 创建输出目录
    output_dir = create_output_dir()
    
    print(f"数据时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
    print(f"总数据点数: {len(df):,}")
    
    # 存储所有日报告
    all_reports = []
    
    # 按日期分析
    unique_dates = sorted(df['日期'].unique())
    
    for i, date in enumerate(unique_dates, 1):
        date_str = date.strftime('%Y-%m-%d')
        print(f"\n正在进行高级分析第 {i}/{len(unique_dates)} 天: {date_str}")
        
        # 筛选当天数据
        df_day = df[df['日期'] == date].copy()
        df_day = df_day.sort_values('时间')
        
        # 生成高级分析图
        filepath = plot_advanced_daily_analysis(df_day, date_str, output_dir)
        print(f"高级分析图已保存: {filepath}")
        
        # 生成报告
        daily_report = generate_daily_report(df_day, date_str)
        all_reports.append(daily_report)
        
        print(f"平均速度: {daily_report['平均速度']:.2f}, "
              f"异常点: {daily_report['异常点数量']}, "
              f"稳定比例: {daily_report['稳定运行比例']:.1f}%")
    
    # 保存详细报告到CSV
    df_reports = pd.DataFrame(all_reports)
    report_path = os.path.join(output_dir, "每日详细报告.csv")
    df_reports.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n详细报告已保存: {report_path}")
    
    # 创建总结仪表板
    dashboard_path = create_summary_dashboard(all_reports, output_dir)
    print(f"总结仪表板已保存: {dashboard_path}")
    
    print(f"\n所有分析结果已保存到 {output_dir} 目录下")

if __name__ == "__main__":
    main() 