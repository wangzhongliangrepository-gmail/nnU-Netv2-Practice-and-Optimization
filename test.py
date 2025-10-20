"""
保存为 check_spacing.py
用法:
    python check_spacing.py E:/nnUNet_raw/Dataset###
"""

import SimpleITK as sitk
from pathlib import Path
import re, sys, itertools, textwrap

root = Path(sys.argv[1]).expanduser().resolve()
img_dirs  = [p for p in (root/'imagesTr', root/'imagesTs') if p.exists()]
label_dirs = {d.name: d for d in (root/'labelsTr', root/'labelsTs') if d.exists()}

def base_id(path):
    # 去掉模态后缀 _0000  _0001 ……
    return re.sub(r'_\d{4}$', '', path.stem)

def all_image_files(img_root):
    return [p for p in img_root.rglob('*.nii*') if re.search(r'_\d{4}\.nii', p.name)]

problems = []

for img_dir in img_dirs:
    for img_file in all_image_files(img_dir):
        bid = base_id(img_file)
        seg_file = label_dirs[img_dir.name.replace('images', 'labels')]/f'{bid}.nii.gz'
        if not seg_file.exists():
            continue

        try:
            im  = sitk.ReadImage(str(img_file))
            seg = sitk.ReadImage(str(seg_file))
        except Exception as e:
            problems.append((bid, '读取失败', str(e)))
            continue

        # 1) 维度个数
        if im.GetDimension() != seg.GetDimension():
            problems.append((bid, '维度数不同',
                             f'image dim={im.GetDimension()}  label dim={seg.GetDimension()}'))
            continue                 # broadcast 错误根源，后两项可省略

        # 2) spacing 长度
        sp_i, sp_s = im.GetSpacing(), seg.GetSpacing()
        if len(sp_i) != len(sp_s):
            problems.append((bid, 'spacing tuple 长度不同',
                             f'image {sp_i}  label {sp_s}'))

        # 3) spacing 数值
        elif not all(abs(a-b) < 1e-5 for a, b in zip(sp_i, sp_s)):
            problems.append((bid, 'spacing 数值不同',
                             f'image {sp_i}  label {sp_s}'))

if not problems:
    print('✅ 未发现维度或 spacing 不一致的样本')
else:
    print('⚠️ 发现以下问题文件：')
    for bid, kind, info in problems:
        print(textwrap.indent(f'{bid}  ->  {kind}\n   {info}', '  '))
