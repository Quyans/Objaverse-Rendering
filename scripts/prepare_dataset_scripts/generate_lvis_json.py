
import os
import shutil
import objaverse
import json

objaverse.__version__


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



def list_to_json(data):
    # 使用字典推导式，将列表转换为带有序号的字典
    numbered_dict = {str(i + 1): value for i, value in enumerate(data)}
    
    # 将字典转换为JSON字符串
    json_data = json.dumps(numbered_dict, indent=4)
    
    # 写入到文件
    with open('lvis_annotations.json', 'w') as json_file:
        json_file.write(json_data)


list_to_json(unique_strings_list)