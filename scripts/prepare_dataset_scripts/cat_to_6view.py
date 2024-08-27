"""
    多线程
    用于将渲染后的图合并成zero123++的那种3*2的图, 保存到新的目录下，并且同时copy target image和 target_rgba
"""

import os
import shutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def concatenate_images(image_paths, output_path):
    images = [Image.open(img_path) for img_path in image_paths]

    img_width, img_height = images[0].size


    total_width = img_width * 2  # 两列
    total_height = img_height * 3  # 三行

    new_im = Image.new('RGB', (total_width, total_height))

    y_offset = 0

    for i in range(0, 6, 2):  # 外循环三次，每次处理一行
        x_offset = 0
        for j in range(2):  # 内循环两次，每次处理一列
            img_index = i + j
            im = images[img_index]
            new_im.paste(im, (x_offset, y_offset))
            x_offset += img_width
        y_offset += img_height

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    new_im.save(output_path)

def copy_queries(folder_path, new_folder_path):
    query_files = ['query.png', 'query_rgba.png']
    for file_name in query_files:
        source_file = os.path.join(folder_path, file_name)
        if os.path.exists(source_file):
            target_file = os.path.join(new_folder_path, file_name)
            shutil.copy2(source_file, target_file)
            # print(f'Copied {file_name} to {target_file}')
        else:
            print(f'File {file_name} not found in {folder_path}')

def process_folder(folder_path, new_folder_path):
    if os.path.isdir(folder_path):
        image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png') and f.startswith(('000', '001', '002', '003', '004', '005'))])
        if len(image_files) == 6:
            output_path = os.path.join(new_folder_path, 'target.png')
            concatenate_images(image_files, output_path)
            # print(f'Created concatenated image at {output_path}')
        # 复制query.png 和 query_rgba.png
        copy_queries(folder_path, new_folder_path)
    

# def process_folders_concurrently(root_dir, new_root_dir, max_workers=16):
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = []
#         for subdir in os.listdir(root_dir):
#             folder_path = os.path.join(root_dir, subdir)
#             new_folder_path = os.path.join(new_root_dir, subdir)
#             # Submit the folder for processing in a separate thread
#             futures.append(executor.submit(process_folder, folder_path, new_folder_path))

#         # Wait for all threads to complete
#         for future in futures:
#             future.result()  # This will raise exceptions from the thread if any occurred

def process_folders_concurrently(root_dir, new_root_dir, max_workers=16):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        folder_paths = [os.path.join(root_dir, subdir) for subdir in os.listdir(root_dir)]
        new_folder_paths = [os.path.join(new_root_dir, subdir) for subdir in os.listdir(root_dir)]
        # 使用tqdm创建进度条
        results = list(tqdm(executor.map(process_folder, folder_paths, new_folder_paths), total=len(folder_paths)))



root_directory = '/data/qys/objectverse-lvis/Lvis_rendering_Full'  # 修改为你的原始根目录路径
new_root_directory = '/data/qys/objectverse-lvis/deocc123_LVIS_full'  # 修改为你的新根目录路径
process_folders_concurrently(root_directory, new_root_directory)
