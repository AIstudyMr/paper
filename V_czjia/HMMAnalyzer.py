import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import tempfile
import shutil
import os


class HMMAnalyzer:
    def __init__(self, data, doc=None):
        self.data = data
        self.doc = doc
        self.scaler = StandardScaler()
        self.model = None
        self.bic_results = None
        self.temp_dir = tempfile.mkdtemp()
        self.figures = []
        
        # 初始化数据预处理
        self._preprocess_data()
        
    def _preprocess_data(self):
        """数据预处理"""
        current_series = self.data.set_index('时间').dropna()
        self.scaled_data = self.scaler.fit_transform(
            current_series.values.reshape(-1, 1))
        self.current_series = current_series

    def _save_figure(self, fig, filename):
        """保存图片到临时目录"""
        path = os.path.join(self.temp_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        self.figures.append(path)
        return path

    def optimize_states(self, max_states=6, n_iter=1000):
        """BIC优化状态数量"""
        bic_scores = []
        state_range = range(2, max_states+1)
        
        for n in state_range:
            try:
                model = hmm.GaussianHMM(
                    n_components=n,
                    covariance_type="diag",
                    n_iter=n_iter,
                    random_state=42
                ).fit(self.scaled_data)
                bic = model.bic(self.scaled_data)
                bic_scores.append(bic)
                print(f"状态数 {n} => BIC: {bic:.2f}")
            except Exception as e:
                print(f"状态数 {n} 训练失败: {str(e)}")
                bic_scores.append(np.nan)
        
        self.bic_results = pd.DataFrame({
            '状态数': list(state_range),
            'BIC': bic_scores
        })
        
        # 绘制BIC曲线
        fig = plt.figure(figsize=(10, 6))
        plt.plot(self.bic_results['状态数'], self.bic_results['BIC'], 
                marker='o', linestyle='--', color='#2E75B6')
        plt.xlabel('隐藏状态数量')
        plt.ylabel('BIC 值')
        plt.title('BIC准则优化状态数量')
        plt.grid(True, alpha=0.3)
        plt.xticks(self.bic_results['状态数'])
        self._save_figure(fig, 'bic_curve.png')
        
        return self.bic_results

    def train_model(self):
        """训练最终模型"""
        if self.bic_results is None:
            raise ValueError("请先执行optimize_states方法")
            
        optimal_states = int(self.bic_results.loc[self.bic_results['BIC'].idxmin(), '状态数'])
        self.model = hmm.GaussianHMM(
            n_components=optimal_states,
            covariance_type="diag",
            n_iter=1000,
            random_state=42
        ).fit(self.scaled_data)
        
        # 生成特征图
        self._plot_state_characteristics()
        self._plot_transition_heatmap()
        self._plot_cycle_duration()
        return self.model

    def _plot_state_characteristics(self):
        """状态特征可视化"""
        state_means = self.scaler.inverse_transform(self.model.means_)
        state_vars = np.sqrt(self.model.covars_) * self.scaler.scale_
        
        fig = plt.figure(figsize=(10, 4))
        plt.errorbar(range(self.model.n_components),
                    state_means.flatten(),
                    yerr=state_vars.flatten(),
                    fmt='o',
                    capsize=5,
                    color='#ED7D31')
        plt.title('各状态液位特征（原始量纲）')
        plt.xlabel('隐藏状态')
        plt.ylabel('液位值')
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(range(self.model.n_components))
        self._save_figure(fig, 'state_characteristics.png')

    def _plot_transition_heatmap(self):
        """状态转移热力图"""
        transition_matrix = pd.DataFrame(
            self.model.transmat_,
            columns=[f"状态{i}" for i in range(self.model.n_components)],
            index=[f"状态{i}" for i in range(self.model.n_components)]
        )
        
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(transition_matrix,
                   annot=True,
                   cmap='Blues',
                   fmt='.2f',
                   linewidths=0.5)
        plt.title('状态转移概率矩阵')
        plt.xlabel('源状态')
        plt.ylabel('目标状态')
        self._save_figure(fig, 'transition_heatmap.png')

    def _plot_cycle_duration(self):
        """周期持续时间可视化"""
        cycle_df = self._detect_cycles()[0]
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(cycle_df['开始时间'], cycle_df['持续时间(min)'], 
                marker='o', linestyle='--', color='#2E75B6')
        plt.title('生产周期持续时间趋势')
        plt.xlabel('周期开始时间')
        plt.ylabel('持续时间（分钟）')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_figure(fig, 'cycle_duration.png')

    def _detect_cycles(self):
        """周期检测逻辑（修复索引越界问题）"""
        state_sequence = self.model.predict(self.scaled_data)
        zero_state = np.argmin(self.scaler.inverse_transform(self.model.means_))
        cycles = []
        current_start = None
        
        # 添加索引范围保护
        max_valid_idx = len(self.current_series) - 1
        
        for i in range(1, len(state_sequence)):
            # 检测起始点（0 -> 非0）
            if (state_sequence[i-1] == zero_state) and (state_sequence[i] != zero_state):
                current_start = min(i-1, max_valid_idx)  # 确保不越界
            # 检测结束点（非0 -> 0）
            if (state_sequence[i-1] != zero_state) and (state_sequence[i] == zero_state) and current_start is not None:
                # 修正索引取值
                end_idx = min(i, max_valid_idx)
                cycles.append({
                    "start_idx": current_start,
                    "end_idx": end_idx,
                    "start_time": self.current_series.index[current_start],
                    "end_time": self.current_series.index[end_idx]
                })
                current_start = None
        
        # 处理最后一个未闭合的周期
        if current_start is not None:
            end_idx = min(len(state_sequence)-1, max_valid_idx)
            cycles.append({
                "start_idx": current_start,
                "end_idx": end_idx,
                "start_time": self.current_series.index[current_start],
                "end_time": self.current_series.index[end_idx]
            })
        
        cycle_stats = []
        for cycle in cycles:
            data_segment = self.current_series.iloc[cycle['start_idx']:cycle['end_idx']+1]
            stats = {
                "周期序号": len(cycle_stats)+1,
                "开始时间": cycle['start_time'],
                "结束时间": cycle['end_time'],
                "持续时间(min)": (cycle['end_time'] - cycle['start_time']).total_seconds() / 60,
                "最大液位": data_segment.max().values[0],
                "平均液位": data_segment.mean().values[0],
                "标准差":data_segment.std().values[0],
            }
            cycle_stats.append(stats)
            
        return pd.DataFrame(cycle_stats),state_sequence,zero_state


    def __del__(self):
        """清理临时文件"""
        try:
            if self.temp_dir is not None and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except AttributeError:
            pass  # 防止temp_dir属性不存在的情况

    def calculate_state_transition_slopes(self):
        """计算状态转移时的斜率（变化率）"""
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 获取状态序列和原始数据
        state_sequence = self.model.predict(self.scaled_data)
        original_values = self.scaler.inverse_transform(self.scaled_data).flatten()
        time_index = self.current_series.index
        
        # 计算相邻时间点之间的斜率
        slopes = []
        transition_info = []
        
        for i in range(1, len(state_sequence)):
            if state_sequence[i] != state_sequence[i-1]:  # 只计算状态变化时的斜率
                delta_time = (time_index[i] - time_index[i-1]).total_seconds() / 60  # 分钟数
                delta_value = original_values[i] - original_values[i-1]
                slope = delta_value / delta_time if delta_time != 0 else float('inf')
                
                slopes.append(slope)
                transition_info.append({
                    '时间': time_index[i],
                    '起始状态': state_sequence[i-1],
                    '目标状态': state_sequence[i],
                    '斜率': slope,
                    '起始值': original_values[i-1],
                    '结束值': original_values[i],
                    '持续时间(min)': delta_time
                })
        
        # 转换为DataFrame
        transition_df = pd.DataFrame(transition_info)
        
        # 绘制斜率分布图
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(data=transition_df, 
                    x='起始状态', 
                    y='斜率', 
                    hue='目标状态',
                    palette='Blues')
        plt.title('不同状态转移间的斜率分布')
        plt.xlabel('起始状态')
        plt.ylabel('斜率（单位变化/分钟）')
        plt.grid(True, alpha=0.3)
        self._save_figure(fig, 'transition_slopes.png')
        
        return transition_df

    
    def generate_report(self, index=None, col=None, phases=None):
        """生成并打印状态统计报告（简化版）"""
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 打印状态参数
        print("\n" + "="*50)
        print("状态参数统计:")
        print("{:<10} {:<10} {:<10} {:<25}".format(
            '状态', '均值', '标准差', '95%置信区间'))
        
        means = self.scaler.inverse_transform(self.model.means_).flatten()
        stds = np.sqrt(self.model.covars_).flatten() * self.scaler.scale_
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            print("{:<10} {:<10.2f} {:<10.2f} {:<25}".format(
                f'状态{i}',
                mean,
                std,
                f"[{mean-1.96*std:.2f}, {mean+1.96*std:.2f}]"
            ))
        
        # 打印转移矩阵
        print("\n状态转移概率矩阵:")
        transmat = pd.DataFrame(self.model.transmat_, 
                            index=[f"状态{i}" for i in range(self.model.n_components)],
                            columns=[f"状态{i}" for i in range(self.model.n_components)])
        print(transmat.round(3))
        
        # 如果有周期数据则返回
        if phases is not None:
            cycle_df = self._detect_cycles()[0]
            cycle_df = cycle_df.join(phases, how="inner", rsuffix="_r")
            return cycle_df
        return None



    def merge_lowest_states(self, threshold=1.0):
        """合并所有接近零值的状态（均值绝对值小于阈值）"""
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 获取原始量纲的状态均值
        means = self.scaler.inverse_transform(self.model.means_).flatten()
        
        # 找出所有接近零值的状态（均值绝对值小于阈值）
        zero_like_states = np.where(np.abs(means) < threshold)[0]
        
        # 获取原始状态序列
        state_sequence = self.model.predict(self.scaled_data)
        self.merged_sequence = state_sequence.copy()
        self.merged_n_states = self.model.n_components
        
        if len(zero_like_states) < 2:
            print("未找到足够多的接近零值状态可合并")
            return self.merged_sequence
        
        print(f"\n将合并以下接近零值的状态: {zero_like_states} (均值分别为: {means[zero_like_states]})")
        
        # 合并状态序列（合并到最小的状态编号）
        merged_state = min(zero_like_states)
        
        for s in zero_like_states:
            self.merged_sequence = np.where(self.merged_sequence == s, merged_state, self.merged_sequence)
        
        # 调整状态编号（填补被合并状态的空缺）
        for s in range(max(zero_like_states)+1, self.model.n_components):
            self.merged_sequence = np.where(self.merged_sequence == s, s-len(zero_like_states)+1, self.merged_sequence)
        
        self.merged_n_states = self.model.n_components - len(zero_like_states) + 1
        return self.merged_sequence

    def calculate_merged_state_slopes(self):
        """计算合并状态后的转移斜率"""
        if not hasattr(self, 'merged_sequence'):
            raise ValueError("请先执行merge_lowest_states方法")
        
        original_values = self.scaler.inverse_transform(self.scaled_data).flatten()
        time_index = self.current_series.index
        
        slopes = []
        transition_info = []
        
        for i in range(1, len(self.merged_sequence)):
            if self.merged_sequence[i] != self.merged_sequence[i-1]:
                delta_time = (time_index[i] - time_index[i-1]).total_seconds() / 60
                delta_value = original_values[i] - original_values[i-1]
                slope = delta_value / delta_time if delta_time != 0 else float('inf')
                
                slopes.append(slope)
                transition_info.append({
                    '时间': time_index[i],
                    '起始状态': self.merged_sequence[i-1],
                    '目标状态': self.merged_sequence[i],
                    '斜率': slope,
                    '起始值': original_values[i-1],
                    '结束值': original_values[i],
                    '持续时间(min)': delta_time
                })
        
        transition_df = pd.DataFrame(transition_info)
        
        # 绘制合并后的斜率分布
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(data=transition_df, 
                    x='起始状态', 
                    y='斜率', 
                    hue='目标状态',
                    palette='Blues')
        plt.title('合并状态后的转移斜率分布')
        plt.xlabel('起始状态')
        plt.ylabel('斜率（单位变化/分钟）')
        plt.grid(True, alpha=0.3)
        self._save_figure(fig, 'merged_transition_slopes.png')
        
        return transition_df



    def print_merged_state_report(self):

        # 获取合并后的状态序列和原始数据
        original_values = self.scaler.inverse_transform(self.scaled_data).flatten()
        merged_states = np.unique(self.merged_sequence)
        
        print("\n" + "="*50)
        print("合并后的状态统计报告:")
        print("{:<10} {:<10} {:<10} {:<15} {:<15}".format(
            '状态', '均值', '标准差', '95%置信区间', '样本占比'))
        
        state_stats = []
        for state in merged_states:
            mask = (self.merged_sequence == state)
            state_values = original_values[mask]
            
            if len(state_values) > 0:
                mean = np.mean(state_values)
                std = np.std(state_values)
                proportion = len(state_values) / len(original_values)
                
                print("{:<10} {:<10.2f} {:<10.2f} {:<15} {:<15.1%}".format(
                    f'状态{state}',
                    mean,
                    std,
                    f"[{mean-1.96*std:.2f}, {mean+1.96*std:.2f}]",
                    proportion
                ))
                
                state_stats.append({
                    '状态': state,
                    '均值': mean,
                    '标准差': std,
                    '下限': mean - 1.96*std,
                    '上限': mean + 1.96*std,
                    '样本占比': proportion
                })
        
        # 打印合并后的转移矩阵
        print("\n合并后的状态转移矩阵:")
        trans_counts = pd.crosstab(
            pd.Series(self.merged_sequence[:-1], name='from'),
            pd.Series(self.merged_sequence[1:], name='to'),
            normalize='index'
        ).round(3)
        print(trans_counts)
        
        return pd.DataFrame(state_stats)

    def merge_near_zero_states(self, threshold=1.0):
        """合并所有接近零值的状态（包括状态0、1、4）"""
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 获取原始量纲的状态均值
        means = self.scaler.inverse_transform(self.model.means_).flatten()
        
        # 找出所有需要合并的状态（0、1、4）
        states_to_merge = []
        for i, mean in enumerate(means):
            if i in [0, 1, 4] or np.abs(mean) < threshold:  # 强制合并0/1/4状态
                states_to_merge.append(i)
        
        if len(states_to_merge) < 2:
            print("未找到足够多的接近零值状态可合并")
            return self.model.predict(self.scaled_data)
        
        print(f"\n将合并以下状态: {states_to_merge} (均值分别为: {means[states_to_merge]})")
        
        # 合并状态序列（合并到最小的状态编号）
        state_sequence = self.model.predict(self.scaled_data)
        merged_state = min(states_to_merge)
        merged_sequence = state_sequence.copy()
        
        for s in states_to_merge:
            merged_sequence = np.where(merged_sequence == s, merged_state, merged_sequence)
        
        # 调整剩余状态编号（填补空缺）
        deleted_states = sorted(set(states_to_merge) - {merged_state})
        for s in range(max(deleted_states)+1, self.model.n_components):
            offset = sum(1 for x in deleted_states if x < s)
            merged_sequence = np.where(merged_sequence == s, s-offset, merged_sequence)
        
        # 保存结果
        self.merged_sequence = merged_sequence
        self.merged_n_states = self.model.n_components - len(deleted_states)
        return merged_sequence




    def calculate_merged_state_slopes_v2(self):
        """计算合并状态后的转移斜率（基于状态持续期的中间点值）"""
        if not hasattr(self, 'merged_sequence'):
            raise ValueError("请先执行merge_lowest_states方法")
        
        original_values = self.scaler.inverse_transform(self.scaled_data).flatten()
        time_index = self.current_series.index
        
        # 找出所有状态变化的边界点
        change_points = np.where(np.diff(self.merged_sequence) != 0)[0] + 1
        segments = np.split(np.arange(len(self.merged_sequence)), change_points)
        
        transition_info = []
        
        for i in range(1, len(segments)):
            prev_segment = segments[i-1]  # 前一个状态区间
            curr_segment = segments[i]    # 当前状态区间
            
            # 计算中间点索引
            prev_mid_idx = prev_segment[len(prev_segment)//2]
            curr_mid_idx = curr_segment[len(curr_segment)//2]
            
            # 获取各种时间信息
            prev_start_time = time_index[prev_segment[0]]
            prev_end_time = time_index[prev_segment[-1]]
            curr_start_time = time_index[curr_segment[0]]
            prev_mid_time = time_index[prev_mid_idx]
            curr_mid_time = time_index[curr_mid_idx]
            
            # 计算持续时间
            prev_duration = (prev_end_time - prev_start_time).total_seconds() / 60
            transition_duration = (curr_mid_time - prev_mid_time).total_seconds() / 60
            
            # 获取值信息
            prev_mid_value = original_values[prev_mid_idx]
            curr_mid_value = original_values[curr_mid_idx]
            
            # 计算斜率
            delta_value = curr_mid_value - prev_mid_value
            slope = round(delta_value / transition_duration, 2) if transition_duration != 0 else float('inf')
            
            transition_info.append({
            '起始状态开始时间': prev_start_time,
            '起始状态结束时间': prev_end_time,
            '状态转移时间': curr_start_time,  # 实际状态切换时间
            '起始值时间': prev_mid_time,
            '结束值时间': curr_mid_time,
            '起始状态': self.merged_sequence[prev_segment[0]],
            '目标状态': self.merged_sequence[curr_segment[0]],
            '斜率': slope,
            '起始值': prev_mid_value,
            '结束值': curr_mid_value,
            '初始状态持续时间(min)': round(prev_duration, 2),
            '状态转移持续时间(min)': round(transition_duration, 2)
            })
    
        
        # 创建DataFrame并保存
        transition_df = pd.DataFrame(transition_info)

        
        # 绘制合并后的斜率分布
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(data=transition_df, 
                    x='起始状态', 
                    y='斜率', 
                    hue='目标状态',
                    palette='Blues')
        plt.title('合并状态后的转移斜率分布')
        plt.xlabel('起始状态')
        plt.ylabel('斜率（单位变化/分钟）')
        plt.grid(True, alpha=0.3)
        self._save_figure(fig, 'merged_transition_slopes.png')
        
        return transition_df















# 使用示例
if __name__ == "__main__":
    # 假设已存在DataFrame df
    file_path = r'D:\Code_File\Vinda_cunzhijia\存纸架数据汇总.csv'
    df = pd.read_csv(
        file_path,
        parse_dates=['时间'],     # 自动解析时间列
        na_values=['null']       # 将null识别为缺失值
    )

    df = df[['时间', '存纸率']]
    # ================= 可视化配置 =================
    plt.rcParams.update({
        'font.sans-serif': 'SimHei',    # 中文字体设置
        'axes.unicode_minus': False,     # 显示负号
        'figure.dpi': 800               # 显示分辨率
    })

    # ================= 初始化分析器 =================
    analyzer = HMMAnalyzer(data=df)

    # ================= 执行分析流程 =================
    # 1. 优化状态数量（只需要调用一次）
    bic_results = analyzer.optimize_states()  # 保存BIC结果供查看
    print("\nBIC优化结果:")
    print(bic_results)

    # 2. 训练模型
    trained_model = analyzer.train_model()  # 保存模型供查看

    # 3. 打印原始状态信息
    print("\n原始状态统计:")
    original_stats = analyzer.generate_report()  # 保存原始状态统计

    # ================= 合并状态 ================= 
    # 4. 合并接近零值的状态（调整threshold根据需要）
    # 强制合并状态0/1/4
    merged_sequence = analyzer.merge_near_zero_states()

    # 打印合并后报告
    print("\n合并后的状态统计:")
    merged_stats = analyzer.print_merged_state_report()

    # 计算合并后的斜率
    print("\n合并状态后的斜率统计:")
    merged_slopes = analyzer.calculate_merged_state_slopes_v2()
    print(merged_slopes.head())


    # 7. 保存关键结果
    merged_slopes.to_csv('状态转移斜率_v2.csv', index=False, encoding='utf-8')  # 支持中文


        
