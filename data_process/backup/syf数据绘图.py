import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_car_error(input_csv_path, output_img_dir):
    """
    为每个car绘制mileage与rec_error的关系图，并保存为JPG

    参数:
        input_csv_path: 输入CSV文件路径（即前一步处理后的结果文件）
        output_img_dir: 图片输出目录路径
    """
    # 设置中文字体（避免中文乱码，如无中文可删除）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # 1. 读取处理后的CSV文件
        df = pd.read_csv(input_csv_path)

        # 检查必要列是否存在
        required_cols = ['car', 'mileage', 'rec_error']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"文件缺少必要列：{', '.join(missing_cols)}")

        # 2. 创建输出目录（不存在则创建）
        if not os.path.exists(output_img_dir):
            os.makedirs(output_img_dir)
            print(f"已创建输出目录：{output_img_dir}")

        # 3. 按car分组，循环绘图
        for car, group in df.groupby('car'):
            # 排序数据（按mileage升序，使图表线条更连贯）
            group_sorted = group.sort_values('mileage').reset_index(drop=True)

            # 创建画布（设置尺寸，避免图片过小）
            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制散点图+折线图（散点显示单个数据，折线连接趋势）
            ax.plot(group_sorted['mileage'], group_sorted['rec_error'],
                    marker='o', markersize=6, linewidth=2, label=f'Car: {car}')

            # 标注最大rec_error的点（突出关键信息）
            max_error_idx = group_sorted['rec_error'].idxmax()
            max_error_mileage = group_sorted.loc[max_error_idx, 'mileage']
            max_error_value = group_sorted.loc[max_error_idx, 'rec_error']
            ax.scatter(max_error_mileage, max_error_value,
                       color='red', s=150, zorder=5, label='Max rec_error')
            ax.annotate(f'Max: {max_error_value:.2f}',
                        xy=(max_error_mileage, max_error_value),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, bbox=dict(boxstyle='round,pad=0.3', color='yellow', alpha=0.7))

            # 设置图表标题和坐标轴标签
            ax.set_title(f'Car {car}: Mileage vs Rec Error', fontsize=14, fontweight='bold')
            ax.set_xlabel('Mileage', fontsize=12)
            ax.set_ylabel('Rec Error', fontsize=12)

            # 添加网格和图例（提升可读性）
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            # 调整布局（避免标签被截断）
            plt.tight_layout()

            # 保存图片（以car为文件名，避免特殊字符）
            img_filename = f"{car}.jpg"
            img_path = os.path.join(output_img_dir, img_filename)
            plt.savefig(img_path, dpi=300, bbox_inches='tight')  # dpi=300确保图片清晰
            plt.close()  # 关闭画布，避免内存占用

            print(f"已保存图片：{img_path}")

        print(f"\n所有图片已保存至：{output_img_dir}，共生成 {len(df['car'].unique())} 张图")

    except Exception as e:
        print(f"绘图过程出错：{str(e)}")


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 1. 输入文件路径（前一步处理后的CSV文件）
    input_csv = "C:/Users/YIFSHEN/Documents/VAE/DyAD/dyad_vae_save/2025-10-27-10-17-51_fold0/result/infer2/segment_predictions.csv"  # 请替换为你的实际路径

    # 2. 图片输出目录（可自定义）
    output_dir = "C:/Users/YIFSHEN/Desktop/rec图片"  # 请替换为你的目标路径

    # 3. 执行绘图
    plot_car_error(input_csv, output_dir)