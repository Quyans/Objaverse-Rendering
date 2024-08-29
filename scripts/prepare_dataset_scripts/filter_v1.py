import csv
import json

# 读取 a.csv 中的数据
a_values = set()
with open('/data/qys/objaverse_filter/kiuisobj_v1_merged_80K.csv', mode='r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # 只取逗号后面的部分
        a_values.add(row[1])

# 读取 b.json 中的数据
with open('/data/qys/objectverse-lvis/lvis_annotations.json', 'r') as jsonfile:
    b_data = json.load(jsonfile)

# 找出 b.json 中的值哪些在 a.csv 中
matching_values = [value for value in b_data.values() if value in a_values]

# 将结果写入到一个新的 JSON 文件中
with open('/data/qys/test2/matching_values.json', 'w') as outputfile:
    json.dump(matching_values, outputfile, indent=4)

print("匹配结果已写入 matching_values.json 文件中。")
