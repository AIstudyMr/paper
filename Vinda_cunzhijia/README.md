# 存纸架数据分析系统

## 项目概述
本项目是一个针对纸品生产线存纸架系统的数据分析工具，主要用于分析和可视化存纸率、折叠机速度等关键生产参数，帮助优化生产效率和识别潜在问题。

## 主要功能

### 1. 数据分析模块
- **HMM状态分析** (`HMMAnalyzer.py`)
  - 使用隐马尔可夫模型进行状态识别
  - 优化状态数量（BIC准则）
  - 状态特征可视化
  - 状态转移概率分析
  - 生产周期检测与分析

### 2. 时间序列分析
- **速度周期分析** (`speed_period.py`, `speed_period_V1.py`)
  - 低速时间段识别
  - 速度变化趋势分析
  - 周期性模式识别

### 3. 数据可视化
- **多维度图表生成**
  - 按小时图表 (`plot_speed_periods.py`)
  - 按天图表
  - 按周图表
  - 全周期图表
- **图表类型**
  - 折叠机速度与存纸率对比图
  - 小包机速度分析图
  - 装箱机数据分析图
  - 状态转移热力图

### 4. 概率分析
- **设备运行概率计算** (`Calculate_the_probability.py`, `Calculate_the_probability_v2.py`)
  - 小包机运行概率分析
  - 基于速度的概率计算
  - 多设备联合概率分析

### 5. 数据预处理
- **数据标准化** (`time_norm.py`, `filename_match.py`)
  - 时间序列规范化
  - 文件名匹配与规范化
  - 布尔值替换处理

## 数据结构

### 主要数据文件
1. `存纸架数据汇总.csv`
   - 时间
   - 存纸率
   - 折叠机实际速度
   - 小包机数据
   - 装箱机数据

2. `low_speed_periods_v1.csv`
   - 序号
   - 开始时间
   - 结束时间
   - 持续时间
   - 描述

### 输出目录结构
```
├── 每小时图_折叠机速度_存纸率_非归一化/
├── 每日图_折叠机速度_存纸率_非归一化/
├── 每周图_折叠机速度_存纸率_非归一化/
├── 全周期图_折叠机速度_存纸率_非归一化/
└── 其他专题分析图表目录/
```

## 使用说明

### 环境要求
```python
pandas
numpy
matplotlib
seaborn
hmmlearn
sklearn
```

### 基本使用流程

1. **HMM分析**
```python
from HMMAnalyzer import HMMAnalyzer

# 创建分析器实例
analyzer = HMMAnalyzer(data)

# 优化状态数量
analyzer.optimize_states(max_states=6)

# 训练模型
analyzer.train_model()

# 生成分析报告
analyzer.generate_report()
```

2. **速度周期分析**
```python
# 运行速度周期分析
python speed_period_V1.py

# 生成图表
python plot_speed_periods.py
```

3. **概率分析**
```python
# 运行概率分析
python Calculate_the_probability_v2.py
```

## 注意事项
1. 数据文件需放置在正确的目录结构中
2. 确保数据格式符合系统要求
3. 大文件处理时注意内存使用
4. 图表生成时注意磁盘空间

## 维护说明
- 定期检查数据完整性
- 更新模型参数
- 优化分析算法
- 备份重要数据

## 贡献者
- 项目开发团队

## 版权信息
© 2025 All Rights Reserved 