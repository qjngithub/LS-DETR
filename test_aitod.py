import json

# 加载标注文件
with open('/mnt/hdd1/DOTA/annotations/instances_train01.json', 'r') as f:
    data = json.load(f)

# 检查类别信息
print("类别数量:", len(data['categories']))
print("类别信息:")
for category in data['categories']:
    print(f"ID: {category['id']}, Name: {category['name']}")

# 检查标注中的类别ID范围
category_ids = set()
for annotation in data['annotations']:
    category_ids.add(annotation['category_id'])

print("\n标注中出现的类别ID:", sorted(category_ids))
print("最小ID:", min(category_ids))
print("最大ID:", max(category_ids))