import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_csv_with_encoding(file_path, encodings=['utf-8', 'gbk', 'gb2312', 'utf-8-sig']):
    """尝试不同编码读取CSV文件"""
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"使用 {encoding} 编码时发生错误: {e}")
            continue
    raise ValueError("无法读取文件，请检查文件编码")

def process_data_for_time_period(summary_df, start_time, end_time):
    """为指定时间段处理数据（复制自数据分析处理.py）"""
    # 转换时间列
    summary_df['时间'] = pd.to_datetime(summary_df['时间'])
    
    # 筛选时间段内的数据
    mask = (summary_df['时间'] >= start_time) & (summary_df['时间'] <= end_time)
    period_data = summary_df.loc[mask].copy()
    
    if period_data.empty:
        print(f"警告：时间段 {start_time} 到 {end_time} 没有数据")
        return None
    
    # 按分钟重采样
    period_data.set_index('时间', inplace=True)
    
    # 定义需要的列
    required_columns = [
        '折叠机实际速度', '折叠机入包数', '折叠机出包数', '外循环进内循环纸条数量', '存纸率',
        '裁切机实际速度', '有效总切数', '1#有效切数', '2#有效切数', '3#有效切数', '4#有效切数',
        '进第一裁切通道纸条计数', '进第二裁切通道纸条计数', '进第三裁切通道纸条计数', '进第四裁切通道纸条计数',
        '1#小包机入包数', '1#小包机实际速度', '2#小包机入包数', '2#小包机实际速度',
        '3#小包机入包数', '3#小包机主机实际速度', '4#小包机入包数', '4#小包机主机实际速度'
    ]
    
    # 检查缺失的列
    missing_cols = [col for col in required_columns if col not in period_data.columns]
    if missing_cols:
        print(f"警告：缺失以下列: {missing_cols}")
        # 使用可用的列
        available_cols = [col for col in required_columns if col in period_data.columns]
        if not available_cols:
            print("错误：没有找到任何需要的列")
            return None
        required_columns = available_cols
    
    # 创建结果字典
    result_data = {}
    
    # 累积量列（计算每分钟差值）
    cumulative_cols = [
        '折叠机入包数', '折叠机出包数', '有效总切数', '1#有效切数', '2#有效切数', 
        '3#有效切数', '4#有效切数', '1#小包机入包数', '2#小包机入包数', 
        '3#小包机入包数', '4#小包机入包数', '存纸率'
    ]
    
    # 瞬时量列
    instantaneous_cols = [
        '折叠机实际速度', '外循环进内循环纸条数量', '裁切机实际速度',
        '进第一裁切通道纸条计数', '进第二裁切通道纸条计数', '进第三裁切通道纸条计数', '进第四裁切通道纸条计数',
        '1#小包机实际速度', '2#小包机实际速度', '3#小包机主机实际速度', '4#小包机主机实际速度'
    ]
    
    # 按分钟重采样处理
    minute_data = period_data.resample('1T')
    
    # 处理累积量
    for col in cumulative_cols:
        if col in period_data.columns:
            # 计算每分钟的差值
            minute_diff = minute_data[col].last().diff().fillna(0)
            # 折叠机出包数和折叠机入包数需要除以25
            if col in ['折叠机出包数', '折叠机入包数', '有效总切数', '1#有效切数', 
                       '2#有效切数', '3#有效切数', '4#有效切数', '1#小包机入包数', 
                       '2#小包机入包数', '3#小包机入包数', '4#小包机入包数']:
                result_data[col] = (minute_diff / 25).values
            elif col == '存纸率':
                # 存纸率计算每分钟差值，不除以25
                result_data[col] = minute_diff.values
            else:
                result_data[col] = minute_diff.values
    
    # 处理瞬时量
    if '折叠机实际速度' in period_data.columns:
        # 计算每分钟平均值再除以9.75
        avg_speed = minute_data['折叠机实际速度'].mean()
        result_data['折叠机实际速度'] = (avg_speed / 9.75).round(2).values
    
    if '外循环进内循环纸条数量' in period_data.columns:
        # 计算每分钟的和
        result_data['外循环进内循环纸条数量'] = minute_data['外循环进内循环纸条数量'].sum().values
    
    if '裁切机实际速度' in period_data.columns:
        # 计算每分钟平均值再除以9.75
        avg_speed = minute_data['裁切机实际速度'].mean()
        result_data['裁切机实际速度'] = (avg_speed / 9.75).round(2).values
    
    # 处理裁切通道纸条计数
    cut_channel_cols = ['进第一裁切通道纸条计数', '进第二裁切通道纸条计数', '进第三裁切通道纸条计数', '进第四裁切通道纸条计数']
    for col in cut_channel_cols:
        if col in period_data.columns:
            result_data[col] = minute_data[col].sum().values
    
    # 处理小包机速度
    packer_speed_cols = ['1#小包机实际速度', '2#小包机实际速度', '3#小包机主机实际速度', '4#小包机主机实际速度']
    packer_speeds = []
    for col in packer_speed_cols:
        if col in period_data.columns:
            avg_speed = minute_data[col].mean()
            speed_processed = (avg_speed / 25).round(2)
            result_data[col] = speed_processed.values
            packer_speeds.append(speed_processed.values)
    
    # 计算小包机速度总和
    if packer_speeds:
        packer_speed_sum = np.sum(packer_speeds, axis=0)
        result_data['小包机速度总和'] = packer_speed_sum
    
    # 计算小包机入包数总和
    packer_input_cols = ['1#小包机入包数', '2#小包机入包数', '3#小包机入包数', '4#小包机入包数']
    packer_inputs = []
    for col in packer_input_cols:
        if col in result_data:
            packer_inputs.append(result_data[col])
    
    if packer_inputs:
        packer_input_sum = np.sum(packer_inputs, axis=0)
        result_data['小包机入包数总和'] = packer_input_sum
    
    # 创建时间索引
    time_index = minute_data.groups.keys()
    
    return result_data, list(time_index)

def analyze_correlation(data_dict, period_name):
    """分析三个变量之间的相关性"""
    target_columns = ['外循环进内循环纸条数量', '小包机入包数总和', '有效总切数']
    
    # 检查所需列是否存在
    missing_cols = [col for col in target_columns if col not in data_dict]
    if missing_cols:
        print(f"时间段 {period_name} 缺失列: {missing_cols}")
        return None
    
    # 获取数据并确保长度一致
    min_length = min(len(data_dict[col]) for col in target_columns)
    if min_length == 0:
        print(f"时间段 {period_name} 数据为空")
        return None
    
    # 构建数据矩阵
    data_matrix = np.array([data_dict[col][:min_length] for col in target_columns]).T
    
    # 创建DataFrame便于分析
    df = pd.DataFrame(data_matrix, columns=target_columns)
    
    # 计算相关系数矩阵
    correlation_matrix = df.corr()
    
    # 计算各种统计指标
    result = {
        'period_name': period_name,
        'data_count': min_length,
        'correlation_matrix': correlation_matrix,
        'data_frame': df,
        'statistics': df.describe()
    }
    
    return result

def perform_regression_analysis(combined_df):
    """执行回归分析"""
    print("\n" + "="*80)
    print("多元线性回归分析")
    print("="*80)
    
    # 准备数据
    X = combined_df[['外循环进内循环纸条数量', '小包机入包数总和']]
    y = combined_df['有效总切数']
    
    # 移除包含NaN或无穷大的行
    mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y.values)
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) == 0:
        print("警告：没有有效的数据进行回归分析")
        return None
    
    # 执行多元线性回归
    reg = LinearRegression()
    reg.fit(X_clean, y_clean)
    
    # 预测
    y_pred = reg.predict(X_clean)
    
    # 计算R²
    r2 = r2_score(y_clean, y_pred)
    
    # 计算调整后的R²
    n = len(y_clean)
    p = X_clean.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    print(f"回归方程: 有效总切数 = {reg.intercept_:.4f} + {reg.coef_[0]:.4f} × 外循环进内循环纸条数量 + {reg.coef_[1]:.4f} × 小包机入包数总和")
    print(f"R² = {r2:.4f}")
    print(f"调整后的R² = {adj_r2:.4f}")
    print(f"有效数据点: {len(X_clean)}")
    
    # 进行统计显著性检验
    from scipy.stats import f
    
    # F统计量
    mse = np.mean((y_clean - y_pred) ** 2)
    tss = np.sum((y_clean - np.mean(y_clean)) ** 2)
    f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
    f_p_value = 1 - f.cdf(f_stat, p, n - p - 1)
    
    print(f"F统计量: {f_stat:.4f}")
    print(f"F检验p值: {f_p_value:.6f}")
    
    if f_p_value < 0.05:
        print("✅ 回归模型在α=0.05水平下显著")
    else:
        print("❌ 回归模型在α=0.05水平下不显著")
    
    return {
        'model': reg,
        'r2': r2,
        'adj_r2': adj_r2,
        'coefficients': reg.coef_,
        'intercept': reg.intercept_,
        'f_stat': f_stat,
        'f_p_value': f_p_value,
        'n_samples': len(X_clean)
    }

def create_visualizations(combined_df, output_dir):
    """创建可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    target_columns = ['外循环进内循环纸条数量', '小包机入包数总和', '有效总切数']
    
    # 1. 相关系数热力图
    plt.figure(figsize=(10, 8))
    correlation_matrix = combined_df[target_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, fmt='.4f')
    plt.title('三个变量之间的相关系数热力图', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '相关系数热力图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 散点图矩阵
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, col1 in enumerate(target_columns):
        for j, col2 in enumerate(target_columns):
            ax = axes[i, j]
            
            if i == j:
                # 对角线上绘制直方图
                ax.hist(combined_df[col1].dropna(), bins=30, alpha=0.7, color='skyblue')
                ax.set_title(f'{col1} 分布', fontsize=10)
            else:
                # 非对角线绘制散点图
                valid_mask = combined_df[[col1, col2]].notna().all(axis=1)
                if valid_mask.sum() > 0:
                    x_data = combined_df.loc[valid_mask, col2]
                    y_data = combined_df.loc[valid_mask, col1]
                    
                    ax.scatter(x_data, y_data, alpha=0.6, s=20)
                    
                    # 计算相关系数
                    if len(x_data) > 1:
                        corr_coef = np.corrcoef(x_data, y_data)[0, 1]
                        ax.set_title(f'r = {corr_coef:.4f}', fontsize=10)
                    
                    # 添加趋势线
                    if len(x_data) > 1:
                        z = np.polyfit(x_data, y_data, 1)
                        p = np.poly1d(z)
                        ax.plot(x_data, p(x_data), "r--", alpha=0.8)
            
            if i == len(target_columns) - 1:
                ax.set_xlabel(col2, fontsize=9)
            if j == 0:
                ax.set_ylabel(col1, fontsize=9)
    
    plt.suptitle('三个变量之间的散点图矩阵', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '散点图矩阵.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 时间序列图
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(target_columns):
        plt.subplot(3, 1, i+1)
        
        # 为每个时间段的数据添加不同的颜色
        data_with_index = combined_df[col].dropna().reset_index(drop=True)
        plt.plot(data_with_index.index, data_with_index.values, marker='o', markersize=2, linewidth=1)
        plt.title(f'{col} 时间序列', fontsize=12)
        plt.ylabel('数值')
        plt.grid(True, alpha=0.3)
        
        if i == len(target_columns) - 1:
            plt.xlabel('数据点索引')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '时间序列图.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {output_dir}")

def save_analysis_results(all_correlations, combined_df, regression_result, output_dir):
    """保存分析结果到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存各时间段的相关系数
    correlation_results = []
    for result in all_correlations:
        if result is not None:
            period_name = result['period_name']
            corr_matrix = result['correlation_matrix']
            
            correlation_results.append({
                '时间段': period_name,
                '数据点数': result['data_count'],
                '外循环_小包机入包数_相关系数': corr_matrix.loc['外循环进内循环纸条数量', '小包机入包数总和'],
                '外循环_有效总切数_相关系数': corr_matrix.loc['外循环进内循环纸条数量', '有效总切数'],
                '小包机入包数_有效总切数_相关系数': corr_matrix.loc['小包机入包数总和', '有效总切数']
            })
    
    correlation_df = pd.DataFrame(correlation_results)
    correlation_path = os.path.join(output_dir, '各时间段相关系数分析.csv')
    correlation_df.to_csv(correlation_path, index=False, encoding='utf-8-sig')
    print(f"各时间段相关系数分析已保存: {correlation_path}")
    
    # 2. 保存整体相关系数矩阵
    target_columns = ['外循环进内循环纸条数量', '小包机入包数总和', '有效总切数']
    overall_corr = combined_df[target_columns].corr()
    overall_corr_path = os.path.join(output_dir, '整体相关系数矩阵.csv')
    overall_corr.to_csv(overall_corr_path, encoding='utf-8-sig')
    print(f"整体相关系数矩阵已保存: {overall_corr_path}")
    
    # 3. 保存描述性统计
    descriptive_stats = combined_df[target_columns].describe()
    stats_path = os.path.join(output_dir, '描述性统计.csv')
    descriptive_stats.to_csv(stats_path, encoding='utf-8-sig')
    print(f"描述性统计已保存: {stats_path}")
    
    # 4. 保存回归分析结果
    if regression_result is not None:
        regression_summary = pd.DataFrame({
            '参数': ['截距', '外循环进内循环纸条数量系数', '小包机入包数总和系数'],
            '数值': [regression_result['intercept']] + list(regression_result['coefficients']),
            '说明': ['回归方程的截距项', '外循环进内循环纸条数量的回归系数', '小包机入包数总和的回归系数']
        })
        
        regression_metrics = pd.DataFrame({
            '指标': ['R²', '调整后R²', 'F统计量', 'F检验p值', '样本量'],
            '数值': [regression_result['r2'], regression_result['adj_r2'], 
                    regression_result['f_stat'], regression_result['f_p_value'], 
                    regression_result['n_samples']],
            '说明': ['决定系数', '调整后的决定系数', 'F统计量', 'F检验的p值', '有效样本量']
        })
        
        # 创建综合回归分析结果
        comprehensive_regression = pd.DataFrame({
            '项目': ['回归方程', '截距', '外循环进内循环纸条数量系数', '小包机入包数总和系数', 
                    'R²', '调整后R²', 'F统计量', 'F检验p值', '样本量', '模型显著性'],
            '数值': [
                f"有效总切数 = {regression_result['intercept']:.4f} + {regression_result['coefficients'][0]:.4f} × 外循环进内循环纸条数量 + {regression_result['coefficients'][1]:.4f} × 小包机入包数总和",
                f"{regression_result['intercept']:.4f}",
                f"{regression_result['coefficients'][0]:.4f}",
                f"{regression_result['coefficients'][1]:.4f}",
                f"{regression_result['r2']:.4f}",
                f"{regression_result['adj_r2']:.4f}",
                f"{regression_result['f_stat']:.4f}",
                f"{regression_result['f_p_value']:.6f}",
                f"{regression_result['n_samples']}",
                "显著" if regression_result['f_p_value'] < 0.05 else "不显著"
            ],
            '说明': [
                '多元线性回归方程',
                '回归方程的截距项',
                '外循环进内循环纸条数量的回归系数',
                '小包机入包数总和的回归系数',
                '决定系数，表示模型解释的变异比例',
                '调整后的决定系数',
                'F统计量，用于检验模型整体显著性',
                'F检验的p值，<0.05表示模型显著',
                '参与回归分析的有效样本量',
                '在α=0.05水平下的模型显著性判断'
            ]
        })
        
        # 保存CSV文件
        regression_summary.to_csv(os.path.join(output_dir, '回归系数.csv'), index=False, encoding='utf-8-sig')
        regression_metrics.to_csv(os.path.join(output_dir, '回归指标.csv'), index=False, encoding='utf-8-sig')
        comprehensive_regression.to_csv(os.path.join(output_dir, '综合回归分析结果.csv'), index=False, encoding='utf-8-sig')
        
        # 同时保存Excel文件（可选）
        try:
            regression_path = os.path.join(output_dir, '回归分析结果.xlsx')
            with pd.ExcelWriter(regression_path, engine='openpyxl') as writer:
                regression_summary.to_excel(writer, sheet_name='回归系数', index=False)
                regression_metrics.to_excel(writer, sheet_name='回归指标', index=False)
                comprehensive_regression.to_excel(writer, sheet_name='综合结果', index=False)
            print(f"回归分析Excel文件已保存: {regression_path}")
        except ImportError:
            print("注意：未安装openpyxl，跳过Excel文件保存")
        
        print(f"回归系数CSV已保存: {os.path.join(output_dir, '回归系数.csv')}")
        print(f"回归指标CSV已保存: {os.path.join(output_dir, '回归指标.csv')}")
        print(f"综合回归分析结果CSV已保存: {os.path.join(output_dir, '综合回归分析结果.csv')}")
    
    # 5. 保存完整的分析数据
    complete_data_path = os.path.join(output_dir, '完整分析数据.csv')
    combined_df.to_csv(complete_data_path, index=False, encoding='utf-8-sig')
    print(f"完整分析数据已保存: {complete_data_path}")

def main():
    """主函数"""
    print("="*80)
    print("三变量关系系数分析")
    print("分析变量: 外循环进内循环纸条数量 vs 小包机入包数总和 vs 有效总切数")
    print("="*80)
    
    # 读取数据文件
    time_periods_file = "折叠机正常运行且高存纸率时间段_最终结果.csv"
    summary_file = "存纸架数据汇总.csv"
    
    try:
        # 读取时间段数据
        time_periods_df = pd.read_csv(time_periods_file)
        print(f"成功读取时间段文件，共 {len(time_periods_df)} 个时间段")
        
        # 读取汇总数据
        summary_df = read_csv_with_encoding(summary_file)
        print(f"成功读取汇总文件，共 {len(summary_df)} 行数据")
        
        # 存储所有分析结果
        all_correlations = []
        all_data = []
        
        # 处理每个时间段
        for idx, row in time_periods_df.iterrows():
            start_time = pd.to_datetime(row['开始时间'])
            end_time = pd.to_datetime(row['结束时间'])
            
            print(f"\n处理时间段 {idx+1}/{len(time_periods_df)}: {start_time} 到 {end_time}")
            
            # 处理数据
            result = process_data_for_time_period(summary_df, start_time, end_time)
            
            if result is not None:
                data_dict, time_index = result
                
                # 分析相关性
                period_name = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{end_time.strftime('%Y%m%d_%H%M%S')}"
                correlation_result = analyze_correlation(data_dict, period_name)
                
                if correlation_result is not None:
                    all_correlations.append(correlation_result)
                    
                    # 收集数据用于整体分析
                    period_df = correlation_result['data_frame'].copy()
                    period_df['时间段'] = period_name
                    all_data.append(period_df)
            else:
                print(f"跳过时间段 {idx+1}，无数据")
        
        if not all_correlations:
            print("错误：没有有效的相关性分析结果")
            return
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n合并后的总数据点: {len(combined_df)}")
        
        # 计算整体相关系数矩阵
        target_columns = ['外循环进内循环纸条数量', '小包机入包数总和', '有效总切数']
        overall_correlation = combined_df[target_columns].corr()
        
        print("\n" + "="*80)
        print("整体相关系数矩阵")
        print("="*80)
        print(overall_correlation.round(4))
        
        # 输出具体的相关系数
        print(f"\n📊 关键相关系数:")
        print(f"外循环进内循环纸条数量 vs 小包机入包数总和: {overall_correlation.loc['外循环进内循环纸条数量', '小包机入包数总和']:.4f}")
        print(f"外循环进内循环纸条数量 vs 有效总切数: {overall_correlation.loc['外循环进内循环纸条数量', '有效总切数']:.4f}")
        print(f"小包机入包数总和 vs 有效总切数: {overall_correlation.loc['小包机入包数总和', '有效总切数']:.4f}")
        
        # 进行回归分析
        regression_result = perform_regression_analysis(combined_df)
        
        # 创建输出目录
        output_dir = "关系系数分析结果"
        
        # 保存分析结果
        save_analysis_results(all_correlations, combined_df, regression_result, output_dir)
        
        # 创建可视化图表
        visualization_dir = os.path.join(output_dir, "可视化图表")
        create_visualizations(combined_df, visualization_dir)
        
        print(f"\n✅ 分析完成！结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 