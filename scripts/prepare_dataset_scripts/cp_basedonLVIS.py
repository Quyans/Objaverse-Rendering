
import os
import shutil
import objaverse
from concurrent.futures import ThreadPoolExecutor, as_completed

objaverse.__version__


import multiprocessing
# processes = multiprocessing.cpu_count()
# processes
processes = 16

uids = objaverse.load_uids()
print(len(uids), type(uids))

lvis_annotations = objaverse.load_lvis_annotations()
# print(lvis_annotations)
print(len(lvis_annotations))

# 初始化一个空集合
unique_strings = set()
# 遍历字典，将每个值列表中的字符串添加到集合中
for value_list in lvis_annotations.values():
    unique_strings.update(value_list)
# 将集合转换为列表
unique_strings_list = list(unique_strings)
# print(unique_strings_list)
print("总长度是",len(unique_strings_list))   #LVIS总长度 46207




"""
单线程
"""
# root_folder = '/data/qys/objectverse-lvis/objaverse/glbs'
# # 目标文件夹
# destination_folder = '/data/qys/objavese-lvis-subset'


# subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

# # 确保目标文件夹存在
# os.makedirs(destination_folder, exist_ok=True)

# # 遍历文件名列表
# for file_name in unique_strings_list:
#     file_found = False
#     for folder in subfolders:
#         # 构造文件路径
#         file_path = os.path.join(folder, file_name + '.glb')
        
#         # 检查文件是否存在
#         if os.path.isfile(file_path):
#             # 复制文件到目标文件夹
#             shutil.copy(file_path, destination_folder)
#             file_found = True
#             print(f'Copied {file_name}.glb to {destination_folder}')
#             break
    
#     if not file_found:
#         print(f'{file_name}.glb not found in any subfolder of {root_folder}')

# print('Done!')

"""
多线程
"""


# 根目录路径
root_folder = '/data/qys/objectverse-lvis/objaverse/glbs'

# 目标文件夹
destination_folder = 'objavese-lvis-subset_full'

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 获取根目录下的所有子文件夹
subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]

def copy_file(file_name):
    """查找并复制文件的函数"""
    file_found = False
    for folder in subfolders:
        # 构造文件路径
        file_path = os.path.join(folder, file_name + '.glb')
        
        # 检查文件是否存在
        if os.path.isfile(file_path):
            # 复制文件到目标文件夹
            shutil.copy(file_path, destination_folder)
            file_found = True
            print(f'Copied {file_name}.glb to {destination_folder}')
            break
    
    if not file_found:
        print(f'{file_name}.glb not found in any subfolder of {root_folder}')

# 使用 ThreadPoolExecutor 进行多线程处理
with ThreadPoolExecutor(max_workers=10) as executor:
    # 提交所有的文件复制任务
    future_to_file = {executor.submit(copy_file, file_name): file_name for file_name in unique_strings_list}
    
    # 处理任务完成
    for future in as_completed(future_to_file):
        file_name = future_to_file[future]
        try:
            future.result()  # 如果有异常，这里会抛出
        except Exception as e:
            print(f'Error processing {file_name}: {e}')

print('Done!')