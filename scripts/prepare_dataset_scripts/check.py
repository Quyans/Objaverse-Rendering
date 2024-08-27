"""
检查渲染的结果是否有纯白的，即下载的模型有问题
"""

from PIL import Image
import os
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 设置根目录、输出文件路径和目标复制目录
root_dir = '/data/qys/objectverse-lvis/Lvis_rendering_cube_fixdistance'
target_dir = '/data/qys/test/checkdir/'
output_file = target_dir + 'folders_with_empty_alpha.txt'  # 保存子文件夹名字的文本文件
num_threads = 1  # 线程数，可以根据你的系统调整

# 用于记录子文件夹名字的列表
empty_alpha_folders = []

def check_alpha_channel(folder_name):
    """检查子文件夹中的 query_rgba.png 的 Alpha 通道是否全为 0，如果是则返回文件夹名"""
    folder_path = os.path.join(root_dir, folder_name)
    query_image_path = os.path.join(folder_path, 'query_rgba.png')
    if os.path.exists(query_image_path):
        # 打开图片并检查Alpha通道
        img = Image.open(query_image_path).convert('RGBA')
        alpha_channel = img.split()[-1]  # 提取Alpha通道
        if not np.array(alpha_channel).any():  # 如果没有任何非零值，表示Alpha通道全为0
            return folder_name
    return None

# 获取所有子文件夹名称
folder_names = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# 使用多线程检查每个子文件夹，并显示进度条
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {executor.submit(check_alpha_channel, folder_name): folder_name for folder_name in folder_names}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing folders"):
        result = future.result()
        if result:
            empty_alpha_folders.append(result)

# 单线程依次检查每个子文件夹，并显示进度条
# for folder_name in tqdm(folder_names, desc="Processing folders"):
#     result = check_alpha_channel(folder_name)
#     if result:
#         empty_alpha_folders.append(result)

os.makedirs(target_dir, exist_ok=True)

# 将结果保存到文本文件中
with open(output_file, 'w') as f:
    for folder_name in empty_alpha_folders:
        f.write(f"{folder_name}\n")

# 统一复制符合条件的文件夹
for folder_name in tqdm(empty_alpha_folders, desc="Copying folders"):
    folder_path = os.path.join(root_dir, folder_name)
    target_folder_path = os.path.join(target_dir, folder_name)
    shutil.copytree(folder_path, target_folder_path)

print(f"Process completed. {len(empty_alpha_folders)} folders found with query_rgba.png having all-zero alpha channel and copied.")
