import os
import json

def collect_glb_files(directory: str, output_json: str) -> None:
    # 获取指定目录下的所有 .glb 文件的绝对路径（不包括子目录）
    glb_files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith('.glb') and os.path.isfile(os.path.join(directory, file))
    ]

    # 将路径写入JSON文件
    with open(output_json, 'w') as json_file:
        json.dump(glb_files, json_file, indent=4)

# 使用示例
directory_path = '/data/qys/objectverse-lvis/objavese-lvis-subset_full'  # 替换为你的文件夹路径
output_json_path = '/data/qys/objectverse-lvis/lvisfull.json'  # 替换为输出的JSON文件路径

collect_glb_files(directory_path, output_json_path)
