# Databricks notebook source
import pandas as pd

# 你的文件路径
file_path = '/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_new/0_3001/0_3001.parquet'

# 读取文件（只看列名，不加载完整数据）
df = pd.read_parquet(file_path, engine='pyarrow')
print("文件中的所有列名：")
print(df.columns.tolist())

# 代码要求的必需原始列名（必须全部存在）
required_cols = [
    'time', 'odo', 'bit_charging_state',
    'bms_total_voltage', 'bms_total_current', 'bms_soc',
    'bms_volt_max_value', 'bms_volt_min_value',
    'bms_temp_max_value', 'bms_temp_min_value', 'bms_tba_cells_1'
]

# 检查缺失列
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"\n❌ 缺失必需列：{missing_cols}")
else:
    print("\n✅ 所有必需列都存在！")

# COMMAND ----------

def count_files_in_volume(volume_path, recursive=True, show_progress=True):
    """
    快速统计 Databricks Volumes 路径下的文件总数
    :param volume_path: Volumes 完整路径（如 /Volumes/conf/dl/xxx/）
    :param recursive: 是否递归统计子文件夹
    :param show_progress: 是否显示每个子目录的统计进度
    :return: 总文件数
    """
    total_files = 0
    
    # 验证路径是否存在
    try:
        dbutils.fs.ls(volume_path)
    except Exception as e:
        print(f"错误：路径 {volume_path} 不存在或无访问权限！错误信息：{e}")
        return 0
    
    # 递归统计函数（内部使用，避免重复代码）
    def _recursive_count(current_path):
        nonlocal total_files
        try:
            # 列出当前目录下的所有项（文件+子目录）
            items = dbutils.fs.ls(current_path)
            # 统计当前目录的文件数（过滤掉子目录）
            current_file_count = len([item for item in items if not item.isDir()])
            total_files += current_file_count
            
            # 显示进度（可选）
            if show_progress:
                print(f"目录：{current_path} → 文件数：{current_file_count}")
            
            # 递归处理子目录（如果开启 recursive）
            if recursive:
                for item in items:
                    if item.isDir():
                        _recursive_count(item.path)
        except Exception as e:
            print(f"警告：统计目录 {current_path} 失败，跳过！错误信息：{e}")
    
    # 开始统计
    print(f"开始统计 Volumes 路径：{volume_path}")
    _recursive_count(volume_path)
    print(f"\n✅ 统计完成！")
    print(f"📁 统计范围：{'包含子文件夹' if recursive else '仅当前目录'}")
    print(f"🗂️  总文件数：{total_files}")
    
    return total_files

# ------------------- 执行统计 -------------------
# 你的 Volumes 路径（直接复制粘贴即可）
volume_path = "/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_pkl/"

# 执行统计（默认递归统计所有子文件夹，显示进度）
total_file_count = count_files_in_volume(
    volume_path=volume_path,
    recursive=True,  # 如需仅统计当前目录，改为 False
    show_progress=True  # 如需静默统计，改为 False
)

# COMMAND ----------

import os

def get_folder_size(folder_path: str) -> tuple[float, str]:
    """
    计算文件夹总占用空间（含子文件夹）
    :param folder_path: 目标文件夹路径（相对路径/绝对路径均可）
    :return: (总大小数值, 单位)，如 (2.5, "GB")、(1024, "KB")
    """
    total_size = 0  # 初始总大小（单位：字节）
    
    # 递归遍历文件夹内所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)  # 拼接文件完整路径
            # 累加文件大小（跳过符号链接，避免报错）
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    
    # 单位自动换算（从字节到最适合的单位）
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    while total_size >= 1024 and unit_index < len(units) - 1:
        total_size /= 1024
        unit_index += 1
    
    return round(total_size, 2), units[unit_index]

# ------------------- 用法示例 -------------------
if __name__ == "__main__":
    # 替换为你的目标文件夹路径（支持相对路径或绝对路径）
    target_folder = r"/Volumes/conf/dl/vol_prediction-rdeb-yifshen_common_blob/203_Validation_new/"  # Windows 示例
    # target_folder = "/Users/xxx/Documents"  # macOS/Linux 示例
    # target_folder = "./test_folder"  # 相对路径示例（当前目录下的 test_folder）
    
    # 校验路径是否存在且是文件夹
    if not os.path.exists(target_folder):
        print(f"错误：路径 {target_folder} 不存在！")
    elif not os.path.isdir(target_folder):
        print(f"错误：{target_folder} 不是文件夹！")
    else:
        size, unit = get_folder_size(target_folder)
        print(f"文件夹 {target_folder} 的总占用空间：{size} {unit}")