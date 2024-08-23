import objaverse
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
print(unique_strings_list)
print("总长度是",len(unique_strings_list))

# objects = objaverse.load_objects(
#     uids=unique_strings_list,
#     download_processes=processes
# )