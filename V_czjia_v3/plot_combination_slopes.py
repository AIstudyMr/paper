import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 正常显示负号

def plot_combination_slopes():
    """绘制折叠机停机期间和运行期间的停机组合平均斜率对比图"""
    
    # 读取两个CSV文件
    try:
        stop_df = pd.read_csv('折叠机停机期间所有停机组合斜率详细统计分析_30.csv', encoding='utf-8-sig')
        operating_df = pd.read_csv('折叠机运行期间所有停机组合斜率详细统计分析_30.csv', encoding='utf-8-sig')
        print("成功读取两个CSV文件")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 提取需要的列
    stop_combinations = stop_df['停机组合'].tolist()
    stop_slopes = stop_df['平均斜率'].tolist()
    
    operating_combinations = operating_df['停机组合'].tolist()
    operating_slopes = operating_df['平均斜率'].tolist()
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    # fig.suptitle('折叠机停机组合平均斜率对比分析', fontsize=16, fontweight='bold')
    
    # 第一个子图：停机期间
    colors1 = ['red' if slope < 0 else 'green' for slope in stop_slopes]
    bars1 = ax1.bar(range(len(stop_combinations)), stop_slopes, color=colors1, alpha=0.7)
    ax1.set_title('折叠机停机期间 - 停机组合平均斜率', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均斜率 (/分钟)', fontsize=12)
    ax1.set_xticks(range(len(stop_combinations)))
    ax1.set_xticklabels([combo.replace(', ', ',\n') for combo in stop_combinations], 
                        rotation=45, ha='right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 在柱子上添加数值标签
    for i, (bar, slope) in enumerate(zip(bars1, stop_slopes)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height >= 0 else -0.5),
                f'{slope:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=8, rotation=0)
    
    # 第二个子图：运行期间
    colors2 = ['red' if slope < 0 else 'green' for slope in operating_slopes]
    bars2 = ax2.bar(range(len(operating_combinations)), operating_slopes, color=colors2, alpha=0.7)
    ax2.set_title('折叠机运行期间 - 停机组合平均斜率', fontsize=14, fontweight='bold')
    ax2.set_ylabel('平均斜率 (/分钟)', fontsize=12)
    ax2.set_xlabel('停机组合', fontsize=12)
    ax2.set_xticks(range(len(operating_combinations)))
    ax2.set_xticklabels([combo.replace(', ', ',\n') for combo in operating_combinations], 
                        rotation=45, ha='right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 在柱子上添加数值标签
    for i, (bar, slope) in enumerate(zip(bars2, operating_slopes)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.2),
                f'{slope:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=8, rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='正斜率 (存纸率上升趋势)'),
        Patch(facecolor='red', alpha=0.7, label='负斜率 (存纸率下降趋势)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # 保存图形
    plt.savefig('折叠机停机组合平均斜率对比图_30.png', dpi=800, bbox_inches='tight')
    print("图形已保存为 '折叠机停机组合平均斜率对比图_30.png'")
    
    # 显示图形
    # plt.show()
    
    # 打印统计信息
    print(f"\n=== 统计信息 ===")
    print(f"停机期间 - 停机组合数量: {len(stop_combinations)}")
    print(f"停机期间 - 平均斜率范围: {min(stop_slopes):.3f} 到 {max(stop_slopes):.3f}")
    print(f"停机期间 - 正斜率组合数: {sum(1 for s in stop_slopes if s > 0)}")
    print(f"停机期间 - 负斜率组合数: {sum(1 for s in stop_slopes if s < 0)}")
    
    print(f"\n运行期间 - 停机组合数量: {len(operating_combinations)}")
    print(f"运行期间 - 平均斜率范围: {min(operating_slopes):.3f} 到 {max(operating_slopes):.3f}")
    print(f"运行期间 - 正斜率组合数: {sum(1 for s in operating_slopes if s > 0)}")
    print(f"运行期间 - 负斜率组合数: {sum(1 for s in operating_slopes if s < 0)}")

def plot_combination_slopes_side_by_side():
    """绘制折叠机停机期间和运行期间的停机组合平均斜率并排对比图"""
    
    # 读取两个CSV文件
    try:
        stop_df = pd.read_csv('折叠机停机期间所有停机组合斜率详细统计分析_30.csv', encoding='utf-8-sig')
        operating_df = pd.read_csv('折叠机运行期间所有停机组合斜率详细统计分析_30.csv', encoding='utf-8-sig')
        print("成功读取两个CSV文件")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 获取所有组合的并集，保持停机期间的顺序为主
    stop_combinations = stop_df['停机组合'].tolist()
    operating_combinations = operating_df['停机组合'].tolist()
    
    # 创建组合到斜率的映射
    stop_slopes_dict = dict(zip(stop_df['停机组合'], stop_df['平均斜率']))
    operating_slopes_dict = dict(zip(operating_df['停机组合'], operating_df['平均斜率']))
    
    # 使用停机期间的组合顺序，并添加运行期间独有的组合
    all_combinations = stop_combinations.copy()
    for combo in operating_combinations:
        if combo not in all_combinations:
            all_combinations.append(combo)
    
    # 获取对应的斜率值（如果某个期间没有该组合，则为None）
    stop_slopes = [stop_slopes_dict.get(combo, None) for combo in all_combinations]
    operating_slopes = [operating_slopes_dict.get(combo, None) for combo in all_combinations]
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # 设置柱子位置
    x = np.arange(len(all_combinations))
    width = 0.35
    
    # 绘制柱状图
    bars1 = []
    bars2 = []
    
    for i, (stop_slope, operating_slope) in enumerate(zip(stop_slopes, operating_slopes)):
        # 停机期间的柱子
        if stop_slope is not None:
            color1 = 'red' if stop_slope < 0 else 'green'
            bar1 = ax.bar(x[i] - width/2, stop_slope, width, 
                         label='停机期间' if i == 0 else "", 
                         color=color1, alpha=0.9, edgecolor='black', linewidth=0.5)
            bars1.append(bar1)
            
            # 添加数值标签
            ax.text(x[i] - width/2, stop_slope + (0.3 if stop_slope >= 0 else -0.3),
                   f'{stop_slope:.2f}', ha='center', 
                   va='bottom' if stop_slope >= 0 else 'top', 
                   fontsize=8, rotation=0)
        
        # 运行期间的柱子
        if operating_slope is not None:
            color2 = 'red' if operating_slope < 0 else 'green'
            bar2 = ax.bar(x[i] + width/2, operating_slope, width, 
                         label='运行期间' if i == 0 else "", 
                         color=color2, alpha=0.5, edgecolor='black', linewidth=0.5)
            bars2.append(bar2)
            
            # 添加数值标签
            ax.text(x[i] + width/2, operating_slope + (0.2 if operating_slope >= 0 else -0.2),
                   f'{operating_slope:.2f}', ha='center', 
                   va='bottom' if operating_slope >= 0 else 'top', 
                   fontsize=8, rotation=0)
    
    # 设置图形属性
    ax.set_title('折叠机停机期间 vs 运行期间 - 停机组合平均斜率对比', fontsize=16, fontweight='bold')
    ax.set_ylabel('平均斜率 (/分钟)', fontsize=12)
    ax.set_xlabel('停机组合', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([combo.replace(', ', ',\n') for combo in all_combinations], 
                       rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # 添加详细图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.9, label='停机期间 (存纸率下降)'),
        Patch(facecolor='green', alpha=0.9, label='停机期间 (存纸率上升)'),
        Patch(facecolor='red', alpha=0.5, label='运行期间 (存纸率下降)'),
        Patch(facecolor='green', alpha=0.5, label='运行期间 (存纸率上升)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('折叠机停机组合平均斜率并排对比图_30.png', dpi=800, bbox_inches='tight')
    print("并排对比图已保存为 '折叠机停机组合平均斜率并排对比图_30.png'")
    
    # 显示图形
    # plt.show()
    
    # 打印详细对比信息
    print(f"\n=== 详细对比信息 ===")
    print(f"{'停机组合':<40} {'停机期间':<12} {'运行期间':<12} {'差值':<12}")
    print("-" * 80)
    
    for i, combo in enumerate(all_combinations):
        stop_val = stop_slopes[i] if stop_slopes[i] is not None else 0
        operating_val = operating_slopes[i] if operating_slopes[i] is not None else 0
        diff = stop_val - operating_val if stop_slopes[i] is not None and operating_slopes[i] is not None else None
        
        stop_str = f"{stop_val:.3f}" if stop_slopes[i] is not None else "N/A"
        operating_str = f"{operating_val:.3f}" if operating_slopes[i] is not None else "N/A"
        diff_str = f"{diff:.3f}" if diff is not None else "N/A"
        
        print(f"{combo:<40} {stop_str:<12} {operating_str:<12} {diff_str:<12}")

if __name__ == "__main__":
    print("绘制折叠机停机组合平均斜率对比图...")
    
    # 绘制上下排列的对比图
    plot_combination_slopes()
    
    print("\n" + "="*60)
    
    # 绘制并排对比图
    plot_combination_slopes_side_by_side()
    
    print("\n图形绘制完成！") 