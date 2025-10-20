

# 数据集转化
"""
nnUNetv2_convert_MSD_dataset -i /tmp/pycharm_project_106/Task07_OPLL/
"""
# /tmp/pycharm_project_106/Task07_OPLL/
# 预处理并生成plan
"""
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
"""

# 五折交叉验证
"""
2d
nnUNetv2_train 3 2d 0 --c
nnUNetv2_train 3 2d 1 --c
nnUNetv2_train 3 2d 2 --c
nnUNetv2_train 3 2d 3 --c
nnUNetv2_train 3 2d 4 --c

3d
nnUNetv2_train 4 3d_fullres 0
nnUNetv2_train 4 3d_fullres 1
nnUNetv2_train 4 3d_fullres 2
nnUNetv2_train 4 3d_fullres 3
nnUNetv2_train 4 3d_fullres 4

"""
#验证
"""
由于有ensemble的策略，所有需要模型预测的概率图（属于某个类别的概率，在0到1之间）而不是分割结果（0，1，2。。。），先每折生成npz文件（每个模型都要，本文只有2d和3d_fullres）

nnUNetv2_train 4 2d 0  --val --npz
nnUNetv2_train 4 2d 1  --val --npz
nnUNetv2_train 4 2d 2  --val --npz
nnUNetv2_train 4 2d 3  --val --npz
nnUNetv2_train 4 2d 4  --val --npz
nnUNetv2_train 4 3d_fullres 0  --val --npz
nnUNetv2_train 4 3d_fullres 1  --val --npz
nnUNetv2_train 4 3d_fullres 2  --val --npz
nnUNetv2_train 4 3d_fullres 3  --val --npz
nnUNetv2_train 4 3d_fullres 4  --val --npz

"""

#推理
"""
nnunet可以自己找寻最优模型并给出推理命令

nnUNetv2_predict -d Dataset003_OPLL -i C:\Users\wangzhongliang\nnUNet\nnUNet_raw\Dataset003_OPLL\imagesTs -o  C:\Users\wangzhongliang\Desktop\PREDICT\number1  -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans --save_probabilities

""""""

# nnUNetv2_predict -d Dataset008_OPLL -i /tmp/pycharm_project_106/nnUNet_raw/Dataset008_OPLL/imagesTs/ -o  /tmp/pycharm_project_106/out/  -f  all -tr nnUNetTrainer -c 2d -p nnUNetPlans --save_probabilities  -chk checkpoint_best.pth