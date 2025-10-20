import os
import json
import re

# 设置数据集路径
dataset_path = "C:/Users/wangzhongliang/nnUNet/Task08_OPLL"
imagesTr_path = os.path.join(dataset_path, "imagesTr")
labelsTr_path = os.path.join(dataset_path, "labelsTr")
imagesTs_path = os.path.join(dataset_path, "imagesTs")

# 用正则表达式提取文件名中的数字部分
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

# 获取所有训练图像和标签
train_images = sorted([f for f in os.listdir(imagesTr_path) if f.endswith('.nii.gz')], key=extract_number)
train_labels = sorted([f for f in os.listdir(labelsTr_path) if f.endswith('.nii.gz')], key=extract_number)
test_images = sorted([f for f in os.listdir(imagesTs_path) if f.endswith('.nii.gz')], key=extract_number) if os.path.exists(imagesTs_path) else []

# 检查文件数量是否一致
assert len(train_images) == len(train_labels), "训练图像与标签数量不一致，请检查数据集文件。"

# 检查文件名是否一致（数字部分）
for img, lbl in zip(train_images, train_labels):
    img_num = extract_number(img)
    lbl_num = extract_number(lbl)
    assert img_num == lbl_num, f"文件不匹配：{img} 和 {lbl} 的编号不一致"

# 构建 JSON 字典
dataset_json = {
    "name": "DatasetName",  # 数据集的名称
    "description": "This is a sample dataset description",  # 数据集的描述
    "tensorImageSize": "3D",  # 图像维度 (2D 或 3D)
    "reference": "",  # 数据集的引用（如果有）
    "licence": "",  # 数据集的许可信息（如果有）
    "release": "0.0",  # 数据集版本
    "modality": {
        "0": "MRI"  # 图像的模态，例如 CT, MR 等
    },
    "labels": {
        "0": "background",
        "1": "OPLL",
    },
    "numTraining": len(train_images),
    "numTest": len(test_images),
    "training": [
        {
            "image": f"./imagesTr/{image}",
            "label": f"./labelsTr/{label}"
        }
        for image, label in zip(train_images, train_labels)
    ],
    "test": [
        f"./imagesTs/{image}"
        for image in test_images
    ]
}

# 保存 JSON 文件
output_path = os.path.join(dataset_path, "dataset.json")
with open(output_path, 'w') as outfile:
    json.dump(dataset_json, outfile, indent=4)

print(f"dataset.json 已生成并保存在 {output_path}")
