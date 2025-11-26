import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------- 参数区域：直接修改这里即可 ----------------------
input_csv = r"D:\1研究生阶段资料\10项目\第十九届挑战杯昌平区赛道\2025年度中国青年科技创新“揭榜挂帅”擂台赛BJ02赛题(1)\DyAD(2)\DyAD\dyad_vae_save\2025-08-08-17-25-17_fold0\result\train_segment_scores.csv"  # 输入片段打分文件（train_segment_scores.csv）
output_dir = r"D:\1研究生阶段资料\10项目\第十九届挑战杯昌平区赛道\2025年度中国青年科技创新“揭榜挂帅”擂台赛BJ02赛题(1)\DyAD(2)\DyAD\dyad_vae_save\2025-08-08-17-25-17_fold0\result"  # 输出结果保存目录
score_mode = "mean"  # 可选："mean" 或 "max"
threshold = 0.06  # 使用你给定的阈值

# ------------------------------------------------------------------------

def score_vehicle(df, score_mode='mean'):
    """
    计算每辆车的分数，取每辆车的前10个片段的重构误差的平均值或最大值。
    """
    scores = []
    for car, group in df.groupby("car"):
        group_sorted = group.sort_values(by="rec_error", ascending=False)
        top_k = group_sorted.head(10)["rec_error"].values
        score = np.max(top_k) if score_mode == 'max' else np.mean(top_k)
        scores.append((car, score))
    return pd.DataFrame(scores, columns=["car", "score"])


def get_vehicle_label_from_segment(df):
    """
    从片段的标签中获取整车标签，假设整车标签是所有片段标签的最大值
    """
    car_labels = df.groupby("car")["label"].max().reset_index()
    car_labels.columns = ["car", "label"]
    return car_labels


def predict_anomalies(vehicle_scores, threshold):
    """
    根据给定的阈值预测异常车辆
    """
    vehicle_scores["predict"] = (vehicle_scores["score"] > threshold).astype(int)
    anomaly_vehicles = vehicle_scores[vehicle_scores["predict"] == 1]
    return anomaly_vehicles


def plot_score_bar(results, threshold, save_path=None):
    """
    绘制车辆得分条形图
    """
    plt.figure(figsize=(10, 5))
    sorted_results = results.sort_values(by="score", ascending=False)
    colors = ['red' if p == 1 else 'green' for p in sorted_results["predict"]]
    plt.bar(sorted_results["car"], sorted_results["score"], color=colors)
    plt.axhline(threshold, color='blue', linestyle='--', label=f"Threshold = {threshold:.4f}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Vehicle Score")
    plt.title("Vehicle Anomaly Scores")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ---------------------- 主程序 ----------------------
def main():
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"[ERROR] CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded {df.shape[0]} segment samples.")

    # 获取车辆分数
    score_df = score_vehicle(df, score_mode=score_mode)

    # 获取每辆车的标签
    label_df = get_vehicle_label_from_segment(df)
    score_df = pd.merge(score_df, label_df, on="car", how="inner")

    # 预测异常车辆
    anomaly_vehicles = predict_anomalies(score_df, threshold)

    os.makedirs(output_dir, exist_ok=True)

    # 保存异常车辆预测结果
    anomaly_vehicles.to_csv(os.path.join(output_dir, "vehicle_predictions.csv"), index=False)
    print(f"[INFO] Saved vehicle-level predictions to: {os.path.join(output_dir, 'vehicle_predictions.csv')}")

    # 绘制车辆得分条形图
    plot_score_bar(score_df, threshold=threshold,
                   save_path=os.path.join(output_dir, "vehicle_scores.png"))

    print(f"\n[INFO] Predicted Anomalous Vehicles:")
    print(anomaly_vehicles)


if __name__ == "__main__":
    main()
