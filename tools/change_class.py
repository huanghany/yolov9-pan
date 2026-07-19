
import os

# 原索引到新索引映射，去掉 Planting_Rack (原索引 1)
mapping = {
    2: 0,  # Ripe
    5: 1,  # Ripe7
    4: 2,  # Ripe4
    3: 3,  # Ripe2
    6: 4,  # Unripe
    7: 5,  # strawberry_flower
    0: 6  # Disease
}

input_folder = "/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-yolo/labels"  # YOLO-seg的txt标签目录
output_folder = "/home/huanghanyang/Datasets/rack_datasets/rack_datasets_v3/project-149-yolo/labels_0to6"  # 新的标签目录

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    new_lines = []

    with open(input_path, "r") as f:
        for line in f:
            data = line.strip().split()
            if not data:
                continue

            # 原类别索引
            original_class = int(data[0])

            # 跳过 Planting_Rack
            if original_class not in mapping:
                continue

            # 映射到新索引
            new_class = mapping[original_class]

            # 替换并保存
            data[0] = str(new_class)
            new_lines.append(" ".join(data))

    # 写出新标签文件
    with open(output_path, "w") as f:
        f.write("\n".join(new_lines))

print("处理完成！新标签已保存到", output_folder)