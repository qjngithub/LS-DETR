import json

def fix_category_ids(annotation_file, output_file):
    """将类别ID从1开始转换为从0开始"""
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # 修复类别定义（categories）
    for category in data['categories']:
        if category['id'] >= 1:
            category['id'] -= 1
    
    # 修复标注中的类别ID（annotations）
    for annotation in data['annotations']:
        if annotation['category_id'] >= 1:
            annotation['category_id'] -= 1
    
    # 保存修复后的文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"已修复类别ID并保存到: {output_file}")
    return data

# 修复训练和验证标注文件
fix_category_ids(
    '/mnt/hdd1/DOTA/annotations/instances_train2017.json',
    '/mnt/hdd1/DOTA/annotations/instances_train01.json'
)

fix_category_ids(
    '/mnt/hdd1/DOTA/annotations/instances_val2017.json',
    '/mnt/hdd1/DOTA/annotations/instances_val01.json'
)