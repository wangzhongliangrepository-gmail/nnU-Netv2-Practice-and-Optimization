import os

# 设置数据集路径
imagesTr_path = "C:/Users/wangzhongliang/nnUNet/test100_spain/imagesTr"
labelsTr_path = "C:/Users/wangzhongliang/nnUNet/test100_spain/labelsTr"

# 获取所有训练图像和标签文件的基础名字（不包括扩展名）
image_files = sorted([f for f in os.listdir(imagesTr_path) if f.endswith('.nii.gz')])
label_files = sorted([f for f in os.listdir(labelsTr_path) if f.endswith('.nii.gz')])

# 提取文件名中的数字部分
def extract_number(filename):
    return os.path.splitext(filename)[0]  # 获取去掉扩展名的文件名

image_numbers = set(extract_number(f) for f in image_files)
label_numbers = set(extract_number(f) for f in label_files)

# 找出缺失的标签文件
missing_labels = image_numbers - label_numbers

if missing_labels:
    print(f"缺失的标签文件编号为: {sorted(missing_labels)}")
else:
    print("所有图像文件都有对应的标签文件。")

# 如果有缺失的标签文件，你需要手动添加这些标签文件或者从数据源重新获取
