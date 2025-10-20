import os
import pydicom
import SimpleITK as sitk


def dicom_to_nifti(dicom_folder, output_filename):
    # 读取 DICOM 文件
    reader = sitk.ImageSeriesReader()

    # 获取 DICOM 文件的序列
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)

    if len(dicom_series) == 0:
        raise ValueError(f"在 {dicom_folder} 找不到 DICOM 序列。")

    # 设置文件名
    reader.SetFileNames(dicom_series)

    # 读取 DICOM 图像序列并转换为 SimpleITK 图像对象
    image = reader.Execute()

    # 保存为 NIfTI 格式
    sitk.WriteImage(image, output_filename)
    print(f"转换完成，NIfTI 文件已保存至: {output_filename}")


# # 设置 DICOM 文件夹路径和输出 NIfTI 文件名
# dicom_folder = 'C:/Users/wangzhongliang/Desktop/Task03_OPLL/labelsTr/ScalarVolume_20'  # 替换为你的 DICOM 文件夹路径
# output_filename = 'C:/Users/wangzhongliang/Desktop/Task03_OPLL/labelsTr/OPLL_002.nii.gz'  # 替换为你的输出文件路径

# 设置 DICOM 文件夹路径和输出 NIfTI 文件名
dicom_folder = 'C:/Users/wangzhongliang/Desktop/OPLLchoose/178/MRI'  # 替换为你的 DICOM 文件夹路径
output_filename = 'C:/Users/wangzhongliang/Desktop/Task07_OPLL/imagesTr/OPLL_024.nii.gz'  # 替换为你的输出文件路径

# 运行转换函数
dicom_to_nifti(dicom_folder, output_filename)
