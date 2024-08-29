import os
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义根目录路径
root_dir = '/data/qys/objectverse-lvis/Lvis_rendering_cube_fixdistance'  # 替换为你的根目录路径

# 读取 json 文件
json_filename = 'matching_values.json'  # 替换为你的 JSON 文件名
with open(json_filename, 'r') as jsonfile:
    directories = json.load(jsonfile)

# 使用 JSON 文件的名称（不包括扩展名）来命名目标目录
destination_dir = os.path.join(os.path.dirname(root_dir), os.path.splitext(json_filename)[0])

# 创建目标目录，如果不存在则创建
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

def copy_and_rename_image(subdir_name):
    subdir_path = os.path.join(root_dir, subdir_name)
    query_image_path = os.path.join(subdir_path, 'query.png')
    
    if os.path.exists(query_image_path):
        # 目标文件的路径
        destination_path = os.path.join(destination_dir, f'{subdir_name}.png')
        
        # 复制并重命名文件
        shutil.copy(query_image_path, destination_path)
        return f'已将 {query_image_path} 复制并重命名为 {destination_path}'
    else:
        return f'{query_image_path} 不存在，跳过'

# 使用 ThreadPoolExecutor 进行多线程处理
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(copy_and_rename_image, subdir_name): subdir_name for subdir_name in directories}

    for future in as_completed(futures):
        result = future.result()
        print(result)

print('所有匹配的图片已处理完毕。')