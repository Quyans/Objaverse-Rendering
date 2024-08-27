"""
用于下载指定的objaverse 模型
"""
import requests
from urllib.parse import urlparse
import os
import objaverse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 打开文件并读取
object_paths = objaverse._load_object_paths()

def download_file(url, target_folder):
    # 解析URL以获取文件名
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    
    # 拼接完整的本地文件路径
    file_path = os.path.join(target_folder, file_name)
    
    # 发送GET请求
    response = requests.get(url, stream=True)
    
    # 检查请求是否成功
    if response.status_code == 200:
        # 确保目标文件夹存在
        os.makedirs(target_folder, exist_ok=True)
        
        # 打开一个本地文件用于写入
        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=file_name) as bar:
            # 迭代写入文件
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Download complete! File saved at: {file_path}")
    else:
        print(f"Failed to download: {response.status_code}")

def main():
    with open('folders_with_empty_alpha.txt', 'r') as file:
        lines = file.readlines()

    # 去除每行末尾的换行符
    uids = [line.strip() for line in lines]

    # 创建一个线程池来管理下载任务
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for uid in uids:
            hf_url = f"https://hf-mirror.com/datasets/allenai/objaverse/resolve/main/{object_paths[uid]}"
            futures.append(executor.submit(download_file, hf_url, "./"))
        
        # 使用tqdm为所有任务生成一个总体进度条
        for future in tqdm(futures, desc="Downloading files"):
            # 检测每个任务的执行结果
            result = future.result()

if __name__ == "__main__":
    main()
