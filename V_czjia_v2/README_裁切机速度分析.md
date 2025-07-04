# 裁切机实际速度分析报告

## 概述

本项目对 2025年4月30日 至 2025年5月6日 期间的裁切机实际速度数据进行了全面分析，生成了多种类型的可视化图表和统计报告。

## 数据统计

- **数据时间范围**: 2025-04-30 16:00:00 到 2025-05-06 12:13:00
- **总数据点数**: 494,149 个
- **数据天数**: 7 天
- **平均每日数据点**: 约 70,592 个（约每秒1个数据点）

## 生成的分析结果

### 1. 基础每日速度曲线图 📊
**目录**: `每日裁切机速度曲线图/`

包含以下文件：
- `2025-04-30_裁切机速度曲线.png` - `2025-05-06_裁切机速度曲线.png`：每日详细速度曲线
- `总览_裁切机速度曲线.png`：所有天数的速度曲线总览

**特点**：
- 显示每天24小时的速度变化
- 包含基本统计信息（平均值、最大值、最小值、标准差）
- 时间轴以2小时为间隔显示

### 2. 高级分析图表 🔬
**目录**: `高级裁切机速度分析/`

包含以下文件：
- `2025-04-30_高级分析.png` - `2025-05-06_高级分析.png`：每日四维分析图
- `运行质量仪表板.png`：综合运行质量仪表板
- `每日详细报告.csv`：详细数值统计报告

**每日四维分析包括**：
1. **速度曲线与异常检测**：标注异常速度点
2. **速度分布直方图**：显示速度分布模式
3. **滑动平均趋势**：1分钟、5分钟、15分钟滑动平均
4. **速度变化率**：反映速度变化的剧烈程度

**运行质量仪表板包括**：
- 每日平均速度趋势
- 稳定运行比例对比
- 异常点数量统计
- 速度波动范围分析
- 停机时间统计
- 综合运行质量评分

### 3. 简洁版速度曲线 ✨
**目录**: `简洁版每日速度曲线/`

包含以下文件：
- `2025-04-30_简洁速度曲线.png` - `2025-05-06_简洁速度曲线.png`：美观的每日速度曲线
- `周总结_速度分析.png`：全周速度对比分析

**特点**：
- 简洁美观的设计风格
- 突出显示关键统计信息
- 包含稳定运行比例分析
- 周总结图展示全周对比

## 关键发现 📈

### 每日运行质量评分
| 日期 | 平均速度 | 稳定率 | 异常点数 | 质量评分 |
|------|----------|--------|-----------|----------|
| 2025-04-30 | 112.0 | 98.0% | 566 | ⭐⭐⭐⭐⭐ |
| 2025-05-01 | 109.9 | 96.2% | 3227 | ⭐⭐⭐⭐ |
| 2025-05-02 | 107.9 | 94.3% | 4762 | ⭐⭐⭐ |
| 2025-05-03 | 109.3 | 95.6% | 3734 | ⭐⭐⭐⭐ |
| 2025-05-04 | 112.4 | 98.3% | 1408 | ⭐⭐⭐⭐⭐ |
| 2025-05-05 | 111.7 | 97.6% | 1971 | ⭐⭐⭐⭐ |
| 2025-05-06 | 109.3 | 95.6% | 1889 | ⭐⭐⭐⭐ |

### 主要结论
1. **最佳运行日**: 2025-05-04，平均速度112.4，稳定率98.3%
2. **需要关注日**: 2025-05-02，平均速度107.9，稳定率94.3%，异常点较多
3. **整体稳定性**: 大部分时间稳定运行比例在95%以上
4. **速度范围**: 主要在100-115范围内运行，符合正常操作范围

## 技术参数定义

### 稳定运行标准
- **稳定速度范围**: 100-115 单位/时间
- **异常检测**: 基于Z-score，阈值为2个标准差
- **停机定义**: 速度 ≤ 0
- **低速运行**: 速度在0-100之间

### 统计指标
- **平均速度**: 当日所有数据点的算术平均值
- **稳定率**: 稳定运行时间占总时间的百分比
- **异常点**: 超出正常波动范围的数据点
- **四分位距**: 75%分位数与25%分位数的差值，反映数据分散程度

## 使用的脚本文件

### 1. `draw_daily_cutting_speed_curves.py`
基础每日速度曲线生成脚本
```bash
python draw_daily_cutting_speed_curves.py
```

### 2. `advanced_daily_cutting_analysis.py`
高级分析脚本，包含异常检测和多维度分析
```bash
python advanced_daily_cutting_analysis.py
```

### 3. `simple_daily_speed_curves.py`
简洁版美观曲线生成脚本
```bash
python simple_daily_speed_curves.py
```

### 4. 辅助脚本
- `check_data_structure.py`: 查看数据结构
- `check_time_range.py`: 检查时间范围和基本统计

## 技术栈

- **Python 3.x**
- **pandas**: 数据处理和分析
- **matplotlib**: 图表绘制
- **numpy**: 数值计算
- **scipy**: 统计分析和异常检测
- **seaborn**: 图表美化

## 建议与改进方向

### 运维建议
1. **重点关注 5月2日 的运行情况**，异常点数量较多（4762个）
2. **保持 5月4日 的运行状态**，该日运行质量最佳
3. **建议定期监控稳定运行比例**，保持在95%以上

### 技术改进
1. 可增加实时监控告警功能
2. 可添加更多机器学习模型进行预测性维护
3. 可考虑与生产计划数据结合分析

## 总结

通过对7天的裁切机运行数据分析，可以看出设备整体运行状况良好，平均稳定率达到96.5%。建议继续保持当前的维护水平，并重点关注异常点较多的时段，以进一步提升运行稳定性。

---

**生成时间**: 2025年5月6日  
**数据来源**: 存纸架数据汇总.csv  
**分析工具**: Python数据分析脚本 