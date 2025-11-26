import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
# 读取数据
parquet_path = r"C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-11-12-01-54-20_fold0\result\test_segment_scores.parquet"
df = pd.read_parquet(parquet_path)
# 定义寻优参数范围
voltage_ranges = ['4.1-4.2', '4.0-4.1', '3.9-3.8', '3.7-3.8', '3.6-3.7', '3.5-3.6', '3.3-3.5']
top_n_values = range(5, 21) # 5到20的整数
# 提取真实标签（0表示正常，1表示异常）
df['true_label'] = df['car'].apply(lambda x: 1 if x.startswith('1_') else 0)
# 存储所有参数组合的评估结果
results = []
# 遍历所有参数组合
for volt_range, top_n in product(voltage_ranges, top_n_values):
    # 筛选符合电压范围的数据
    filtered_df = df[df['volt_range'] == volt_range].copy()
    # 如果筛选后没有数据，跳过
    if filtered_df.empty:
        continue
    # 按车辆分组，获取每个车的最大top_n个rec_error的均值
    def calculate_mean(group):
        if len(group) >= top_n:
            # 明确指定按'rec_error'列排序取最大的top_n个值
            return group.nlargest(top_n, 'rec_error')['rec_error'].mean()
        # 如果数量不足，取所有值的均值
        return group['rec_error'].mean()
    car_means = filtered_df.groupby('car').apply(calculate_mean).reset_index(name='mean_rec_error')
    car_means['true_label'] = car_means['car'].apply(lambda x: 1 if x.startswith('1_') else 0)
    # 如果没有数据或只有一类标签，跳过
    if len(car_means) == 0 or len(car_means['true_label'].unique()) < 2:
        continue
    # 尝试不同阈值寻找最佳分类效果
    thresholds = np.percentile(car_means['mean_rec_error'], np.arange(5, 96, 5)) # 5%到95%的百分位数
    for threshold in thresholds:
        car_means['predicted_label'] = (car_means['mean_rec_error'] > threshold).astype(int)
        # 计算评估指标
        precision = precision_score(car_means['true_label'], car_means['predicted_label'])
        recall = recall_score(car_means['true_label'], car_means['predicted_label'])
        accuracy = accuracy_score(car_means['true_label'], car_means['predicted_label'])
        f1 = f1_score(car_means['true_label'], car_means['predicted_label'])
        # 计算混淆矩阵元素
        tp = ((car_means['true_label'] == 1) & (car_means['predicted_label'] == 1)).sum()
        fn = ((car_means['true_label'] == 1) & (car_means['predicted_label'] == 0)).sum()
        tn = ((car_means['true_label'] == 0) & (car_means['predicted_label'] == 0)).sum()
        fp = ((car_means['true_label'] == 0) & (car_means['predicted_label'] == 1)).sum()
        # 存储结果
        results.append({
            'voltage_range': volt_range,
            'top_n': top_n,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1': f1,
            'tp': tp, # 被识别为故障车的故障车总数
            'fn': fn, # 被识别为正常车的故障车总数
            'tn': tn, # 被识别为正常车的正常车总数
            'fp': fp # 被识别为故障车的正常车总数
        })
# 转换结果为DataFrame并找到最优参数组合（基于F1分数）
results_df = pd.DataFrame(results)
if not results_df.empty:
    best_idx = results_df['f1'].idxmax()
    best_params = results_df.loc[best_idx]
    # 输出最优参数下的混淆矩阵元素
    print("最优参数组合:")
    print(f"电压范围: {best_params['voltage_range']}")
    print(f"选取最大rec_error的数量: {best_params['top_n']}")
    print(f"阈值: {best_params['threshold']:.4f}")
    print("\n评估指标:")
    print(f"Precision: {best_params['precision']:.4f}")
    print(f"Recall: {best_params['recall']:.4f}")
    print(f"Accuracy: {best_params['accuracy']:.4f}")
    print(f"F1 Score: {best_params['f1']:.4f}")
    print("\n混淆矩阵元素:")
    print(f"被识别为故障车的故障车总数: {best_params['tp']}")
    print(f"被识别为正常车的故障车总数: {best_params['fn']}")
    print(f"被识别为正常车的正常车总数: {best_params['tn']}")
    print(f"被识别为故障车的正常车总数: {best_params['fp']}")
    # 生成最优情况下的详细分类结果
    best_volt_range = best_params['voltage_range']
    best_top_n = best_params['top_n']
    best_threshold = best_params['threshold']
    # 重新计算最优参数下的车辆分类
    best_filtered = df[df['volt_range'] == best_volt_range].copy()
    def best_calculate_mean(group):
        if len(group) >= best_top_n:
            return group.nlargest(best_top_n, 'rec_error')['rec_error'].mean()
        return group['rec_error'].mean()
    best_car_means = best_filtered.groupby('car').apply(best_calculate_mean).reset_index(name='mean_rec_error')
    best_car_means['true_label'] = best_car_means['car'].apply(lambda x: '异常' if x.startswith('1_') else '正常')
    best_car_means['predicted_label'] = best_car_means['mean_rec_error'].apply(
        lambda x: '异常' if x > best_threshold else '正常'
    )
    # 保存详细结果到Parquet
    output_path = "最优分类结果.parquet"
    best_car_means.to_parquet(output_path, index=False)
    print(f"\n最优情况下的详细分类结果已保存至: {output_path}")
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    confusion_matrix = np.array([[best_params['tn'], best_params['fp']],
                                 [best_params['fn'], best_params['tp']]])
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测正常', '预测异常'],
                yticklabels=['实际正常', '实际异常'])
    plt.xlabel('预测标签')
    plt.ylabel('实际标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.show()
else:
    print("没有找到有效的参数组合，请检查数据是否符合要求。")