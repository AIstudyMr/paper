import pandas as pd
import os
from datetime import datetime

def summarize_results():
    """总结数据处理结果"""
    print("="*60)
    print("         数据分析处理结果总结")
    print("="*60)
    
    # 读取时间段文件
    time_periods_file = "折叠机正常运行且高存纸率时间段_最终结果.csv"
    time_periods_df = pd.read_csv(time_periods_file)
    
    print(f"1. 输入数据:")
    print(f"   - 时间段文件: {time_periods_file}")
    print(f"   - 汇总数据文件: 存纸架数据汇总.csv")
    print(f"   - 处理的时间段数量: {len(time_periods_df)}")
    
    # 计算总的时间跨度
    time_periods_df['开始时间'] = pd.to_datetime(time_periods_df['开始时间'])
    time_periods_df['结束时间'] = pd.to_datetime(time_periods_df['结束时间'])
    time_periods_df['持续时间_秒'] = (time_periods_df['结束时间'] - time_periods_df['开始时间']).dt.total_seconds()
    
    total_duration = time_periods_df['持续时间_秒'].sum()
    total_hours = total_duration / 3600
    
    print(f"   - 数据时间范围: {time_periods_df['开始时间'].min()} 到 {time_periods_df['结束时间'].max()}")
    print(f"   - 有效运行总时长: {total_hours:.2f} 小时")
    
    print(f"\n2. 数据处理规则:")
    print(f"   数据重采样方式: 按分钟重采样（1T）")
    
    print(f"\n   累积量列（计算每分钟差值后除以25）:")
    cumulative_cols_div25 = [
        '折叠机入包数', '折叠机出包数', '有效总切数', '1#有效切数', '2#有效切数', 
        '3#有效切数', '4#有效切数', '1#小包机入包数', '2#小包机入包数', 
        '3#小包机入包数', '4#小包机入包数'
    ]
    for col in cumulative_cols_div25:
        print(f"     - {col}: 每分钟差值 ÷ 25")
    
    print(f"\n   累积量列（仅计算每分钟差值）:")
    print(f"     - 存纸率: 每分钟差值（不除以25）")
    
    print(f"\n   瞬时量列（特殊处理）:")
    print(f"     - 折叠机实际速度: 每分钟平均值 ÷ 9.75，保留2位小数")
    print(f"     - 外循环进内循环纸条数量: 每分钟求和")
    print(f"     - 裁切机实际速度: 每分钟平均值 ÷ 9.75，保留2位小数")
    print(f"     - 裁切通道纸条计数(4个): 各自每分钟求和")
    print(f"       * 进第一裁切通道纸条计数")
    print(f"       * 进第二裁切通道纸条计数")
    print(f"       * 进第三裁切通道纸条计数")
    print(f"       * 进第四裁切通道纸条计数")
    print(f"     - 小包机实际速度(4个): 各自每分钟平均值 ÷ 25，保留2位小数")
    print(f"       * 1#小包机实际速度")
    print(f"       * 2#小包机实际速度")
    print(f"       * 3#小包机主机实际速度")
    print(f"       * 4#小包机主机实际速度")
    
    print(f"\n   计算合成指标:")
    print(f"     - 小包机速度总和: 4个小包机速度之和")
    print(f"     - 小包机入包数总和: 4个小包机入包数之和")
    
    # 检查输出结果
    output_dirs = {
        "时间段分析图表": "单项指标图表",
        "组合图表分析": "组合图表",
        "差值分析结果": "差值分析结果",
        "复合差值分析结果": "复合差值分析结果"
    }
    
    print(f"\n3. 输出结果:")
    for output_dir, description in output_dirs.items():
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            image_files = [f for f in files if f.endswith('.png')]
            csv_files = [f for f in files if f.endswith('.csv')]
            
            print(f"   - {description} ({output_dir}):")
            if image_files:
                print(f"     * 图片文件: {len(image_files)} 个")
                # 计算图片总文件大小
                total_size = 0
                for filename in image_files:
                    filepath = os.path.join(output_dir, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                total_size_mb = total_size / (1024 * 1024)
                print(f"     * 图片总大小: {total_size_mb:.2f} MB")
            
            if csv_files:
                print(f"     * CSV文件: {len(csv_files)} 个")
            
            if not image_files and not csv_files:
                print(f"     * 目录为空")
        else:
            print(f"   - {description} ({output_dir}): 目录不存在")
    
    # 显示图表文件命名规则
    print(f"\n4. 文件命名规则:")
    print(f"   - 图表文件: YYYYMMDD_HHMMSS_YYYYMMDD_HHMMSS.png")
    print(f"     (开始时间_结束时间格式)")
    print(f"   - 差值分析文件:")
    print(f"     * 详细数据: 列名1_减_列名2_详细数据.csv")
    print(f"     * 统计汇总: 列名1_减_列名2_统计汇总.csv")
    print(f"     * 总体统计: 列名1_减_列名2_总体统计.csv")
    print(f"   - 复合差值分析文件:")
    print(f"     * 复合差值详细数据.csv")
    print(f"     * 复合差值统计汇总.csv")
    
    print(f"\n5. 图表特点:")
    print(f"   - 组合图表布局: 多子图网格布局（每行3个子图）")
    print(f"   - 图表尺寸: 自适应尺寸（15×(5×行数+1)英寸）")
    print(f"   - 字体支持: 中文字体（SimHei、DejaVu Sans）")
    print(f"   - 图表分辨率: 300 DPI高分辨率")
    print(f"   - 标题信息: 包含时间段和持续时间")
    print(f"   - 布局优化: 自动调整避免标题和图表重叠")
    print(f"   - 图例和标签: 完整的轴标签和图例")
    
    print(f"\n6. 数据分析功能:")
    print(f"   主要功能模块:")
    print(f"   - 时间段数据处理: 按指定时间段筛选和处理数据")
    print(f"   - 数据重采样: 按分钟级别重采样")
    print(f"   - 多类型指标处理: 累积量、瞬时量分别处理")
    print(f"   - 组合图表生成: 一张图表包含所有关键指标")
    print(f"   - 差值分析: 支持任意两列数据的差值分析")
    print(f"   - 复合差值分析: 多对列组合的差值分析")
    print(f"   - 统计分析: 正负零差值统计和比例分析")
    
    print(f"\n   分析指标覆盖:")
    print(f"   - 折叠机相关: 实际速度、入包数、出包数")
    print(f"   - 循环系统: 外循环进内循环纸条数量")
    print(f"   - 存纸系统: 存纸率")
    print(f"   - 裁切系统: 实际速度、有效切数、通道计数")
    print(f"   - 小包机系统: 入包数、实际速度（4台设备）")
    print(f"   - 合成指标: 小包机速度和入包数总和")
    
    print(f"\n7. 数据质量:")
    # 分析时间段长度分布
    short_periods = len(time_periods_df[time_periods_df['持续时间_秒'] < 600])  # 小于10分钟
    medium_periods = len(time_periods_df[(time_periods_df['持续时间_秒'] >= 600) & (time_periods_df['持续时间_秒'] < 3600)])  # 10分钟到1小时
    long_periods = len(time_periods_df[time_periods_df['持续时间_秒'] >= 3600])  # 大于1小时
    
    print(f"   时间段长度分布:")
    print(f"   - 短时间段（<10分钟）: {short_periods} 个")
    print(f"   - 中等时间段（10分钟-1小时）: {medium_periods} 个") 
    print(f"   - 长时间段（>1小时）: {long_periods} 个")
    print(f"   - 平均时间段长度: {total_duration/len(time_periods_df)/60:.1f} 分钟")
    
    print(f"\n   数据处理保障:")
    print(f"   - 编码兼容: 支持多种文件编码（UTF-8、GBK、GB2312等）")
    print(f"   - 异常处理: 数据缺失和时间段无数据的处理")
    print(f"   - 列名检查: 自动检查和处理缺失列")
    print(f"   - 数据验证: 处理前验证数据完整性")
    
    print(f"\n" + "="*60)
    print(f"处理完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"="*60)

if __name__ == "__main__":
    summarize_results() 