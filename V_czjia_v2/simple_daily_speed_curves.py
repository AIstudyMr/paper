import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

def create_output_dir():
    """创建输出目录"""
    output_dir = "简洁版每日速度曲线"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_clean_daily_curve(df_day, date_str, output_dir):
    """绘制简洁清晰的每日速度曲线"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 绘制主要速度曲线
    ax.plot(df_day['时间'], df_day['裁切机实际速度'], 
            linewidth=1.2, color='#1f77b4', alpha=0.8)
    
    # 填充区域以突出显示
    ax.fill_between(df_day['时间'], df_day['裁切机实际速度'], 
                    alpha=0.3, color='#1f77b4')
    
    # 添加关键统计线
    mean_speed = df_day['裁切机实际速度'].mean()
    ax.axhline(y=mean_speed, color='red', linestyle='--', 
               linewidth=2, alpha=0.8, label=f'平均速度: {mean_speed:.1f}')
    
    # 标题和标签
    ax.set_title(f'{date_str} 裁切机实际速度曲线', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('时间', fontsize=14)
    ax.set_ylabel('速度 (m/min)', fontsize=14)
    
    # 时间轴格式化
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=2))
    
    # 美化网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    
    # 设置y轴范围
    y_min = max(0, df_day['裁切机实际速度'].min() - 5)
    y_max = df_day['裁切机实际速度'].max() + 5
    ax.set_ylim(y_min, y_max)
    
    # 添加统计信息框
    stats_box = f"""统计信息
━━━━━━━━━━━━━━━━━━━━━━━━━━
• 数据点数: {len(df_day):,}
• 平均速度: {df_day['裁切机实际速度'].mean():.2f}
• 最大速度: {df_day['裁切机实际速度'].max():.2f} 
• 最小速度: {df_day['裁切机实际速度'].min():.2f}
• 标准差: {df_day['裁切机实际速度'].std():.2f}
• 稳定率: {len(df_day[(df_day['裁切机实际速度'] >= 100) & (df_day['裁切机实际速度'] <= 115)]) / len(df_day) * 100:.1f}%"""
    
    ax.text(0.02, 0.98, stats_box, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                     edgecolor='#ddd', alpha=0.9),
            fontsize=11, fontfamily='monospace')
    
    # 旋转时间标签
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 添加图例
    ax.legend(loc='upper right', framealpha=0.9)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    filename = f"{date_str}_简洁速度曲线.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return filepath

def create_week_summary(df, output_dir):
    """创建周总结图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 上图：所有天数的速度曲线
    unique_dates = sorted(df['日期'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_dates)))
    
    for i, date in enumerate(unique_dates):
        df_day = df[df['日期'] == date]
        date_str = date.strftime('%m-%d')
        ax1.plot(df_day['时间'], df_day['裁切机实际速度'], 
                linewidth=1, color=colors[i], alpha=0.8, label=date_str)
    
    ax1.set_title('全周裁切机速度总览', fontsize=16, fontweight='bold')
    ax1.set_ylabel('速度 (m/min)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', ncol=len(unique_dates), 
               bbox_to_anchor=(1, 1.15))
    
    # 下图：每日平均速度柱状图
    daily_stats = []
    for date in unique_dates:
        df_day = df[df['日期'] == date]
        daily_stats.append({
            '日期': date.strftime('%m-%d'),
            '平均速度': df_day['裁切机实际速度'].mean(),
            '稳定率': len(df_day[(df_day['裁切机实际速度'] >= 100) & 
                            (df_day['裁切机实际速度'] <= 115)]) / len(df_day) * 100
        })
    
    df_stats = pd.DataFrame(daily_stats)
    
    bars = ax2.bar(df_stats['日期'], df_stats['平均速度'], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 在柱状图上添加数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        # 添加稳定率
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{df_stats.iloc[i]["稳定率"]:.1f}%', 
                ha='center', va='center', color='white', fontweight='bold')
    
    ax2.set_title('每日平均速度对比', fontsize=16, fontweight='bold')
    ax2.set_ylabel('平均速度 (m/min)', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, "周总结_速度分析.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return summary_path

def main():
    print("开始生成简洁版每日裁切机速度曲线...")
    
    # 读取数据
    df = pd.read_csv('存纸架数据汇总.csv', usecols=['时间', '裁切机实际速度'])
    df['时间'] = pd.to_datetime(df['时间'])
    df['日期'] = df['时间'].dt.date
    
    # 创建输出目录
    output_dir = create_output_dir()
    
    print(f"数据时间范围: {df['时间'].min()} 到 {df['时间'].max()}")
    print(f"总数据点数: {len(df):,}")
    
    # 按日期绘制
    unique_dates = sorted(df['日期'].unique())
    print(f"共 {len(unique_dates)} 天的数据")
    
    for i, date in enumerate(unique_dates, 1):
        date_str = date.strftime('%Y-%m-%d')
        print(f"正在处理 [{i}/{len(unique_dates)}]: {date_str}")
        
        # 筛选当天数据
        df_day = df[df['日期'] == date].copy()
        df_day = df_day.sort_values('时间')
        
        # 绘制当天曲线
        filepath = plot_clean_daily_curve(df_day, date_str, output_dir)
        
        # 显示简要统计
        avg_speed = df_day['裁切机实际速度'].mean()
        stable_rate = len(df_day[(df_day['裁切机实际速度'] >= 100) & 
                                (df_day['裁切机实际速度'] <= 115)]) / len(df_day) * 100
        print(f"  └─ 平均速度: {avg_speed:.1f}, 稳定率: {stable_rate:.1f}%, 已保存")
    
    # 生成周总结
    print("\n正在生成周总结...")
    summary_path = create_week_summary(df, output_dir)
    print(f"周总结已保存: {summary_path}")
    
    print(f"\n✅ 所有图表已生成完成！")
    print(f"📁 保存位置: {output_dir}")
    print(f"📊 共生成 {len(unique_dates)} 个每日图表 + 1 个周总结图")

if __name__ == "__main__":
    import numpy as np
    main() 