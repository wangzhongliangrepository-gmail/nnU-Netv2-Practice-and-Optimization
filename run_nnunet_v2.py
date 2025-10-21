# -*- coding: utf-8 -*-
"""
nnU-Net v2 无需命令行参数,直接运行
步骤可按开关选择：转换→预处理→训练→验证npz→推理
"""
from __future__ import annotations
from pathlib import Path
import os, sys, shlex, subprocess
from typing import Sequence, Union

# ========= 必填：将必要的三大目录结构导入到环境变量 =========
NNUNET_RAW          = r"C:\Users\wangzhongliang\nnUNet\nnUNet_raw"
NNUNET_PREPROCESSED = r"C:\Users\wangzhongliang\nnUNet\nnUNet_preprocessed"
NNUNET_RESULTS      = r"C:\Users\wangzhongliang\nnUNet\nnUNet_results"
EXPORT_NNUNET_PATHS = True             # True=把这三者注入到当前进程环境
# ========= 三大目录配置结束 =========


# ========= 配置区（按需修改） =========
DATASET_ID: Union[str, int] = "008"    # 仅填数字或字符串数字（如 "8"/"008"），不要写 DatasetXXX
TRAIN_CONFIGS: Sequence[str] = ["2d", "3d_fullres"]
FOLDS: Sequence[Union[int, str]] = [0, 1, 2, 3, 4]   # 用 ["all"] 表示全量训练（无验证划分）
TRAINER = "nnUNetTrainer"              # 自定义Trainer类名；默认 nnUNetTrainer
PLANS   = "nnUNetPlans"                # 一般默认即可
# 折如果你把 FOLDS = [0, 1, 2]，
# 脚本就只会训练 fold_0、fold_1、fold_2 ，输出对应三个（fold_0, fold_1, fold_2）。
# 推理时用 -f 0 1 2 是“降本版”的交叉验证。也可以FOLDS = ["all"]全量训练不做验证
# 用 "all" 时不会产生 checkpoint_best.pt；用 checkpoint_final.pth 推理即可。

# 训练附加参数（会原样拼接到 nnUNetv2_train 命令后）
# 常用：
#   "--c"          继续训练（从该fold结果目录最新checkpoint接着训）
#   "--fp32"       关闭混合精度（默认开启AMP；出现NaN或调试用）
#   "--use_compressed"  以压缩形式读取（更吃CPU/RAM，不推荐新手）
#   "--val --npz"  仅做验证+保存npz（训练完后）；本脚本推荐用 DO_VALIDATE_NPZ 控制，无需在训练阶段加
EXTRA_TRAIN_ARGS = "--c"                  # 建议先留空；需要时再填


# 是否执行各阶段
DO_CONVERT        = True                      # 将 MSD 原始集转换为 nnUNetv2 格式
MSD_INPUT_DIR     = r"C:\Users\wangzhongliang\nnUNet\Task08_OPLL" # 仅当 DO_CONVERT=True 时生效 放置需要转换的数据集

DO_PREPROCESS     = True                      # 预处理（首次必做）
VERIFY_DATASET    = True                      # 预处理时校验数据一致性

DO_TRAIN          = True                      # 训练所有指定折与配置
DO_VALIDATE_NPZ   = True                      # 每折生成 npz（用于后续 ensemble）

DO_PREDICT        = False                     # 推理
PRED_CONFIG       = "2d"                      # 推理配置（2d/3d_fullres）
PRED_INPUT_DIR    = r"C:\Users\wangzhongliang\nnUNet\nnUNet_raw\Dataset003_OPLL\imagesTs"
PRED_OUTPUT_DIR   = r"C:\Users\wangzhongliang\Desktop\PREDICT\number1"
PRED_FOLDS        = [0,1,2,3,4]               # 或者设为字符串 "all"
PRED_SAVE_PROB    = True                      # 保存概率图
PRED_CHECKPOINT   = "checkpoint_best.pt"      # 指定 .pt；留空用默认
PRED_PLANS        = "nnUNetPlans"             # 留空为默认；自定义则填你的 plans 名称
PRED_EXTRA_ARGS   = ""                        # 例如 "--disable_tta"
# ========= 配置区结束 =========


def run(cmd: str):
    print("\n>>>", cmd)
    # Windows 上 cmd 拆分需要 posix=False
    args = shlex.split(cmd, posix=(os.name != "nt"))
    proc = subprocess.Popen(args)
    proc.communicate()
    if proc.returncode != 0:
        sys.exit(f"[ERROR] 命令失败：{cmd}")

def stage_convert_msd(input_dir: str):
    from pathlib import Path
    from nnunetv2.dataset_conversion.convert_MSD_dataset import convert_msd_dataset
    import os

    p = Path(input_dir)
    assert p.is_dir(), f"输入目录不存在: {p}"
    for d in ["imagesTr", "labelsTr", "imagesTs"]:
        assert (p / d).is_dir(), f"缺少 {d}: {p/d}"

    # 进程数：至少 1，最多 CPU 数
    nprocs = max(1, (os.cpu_count() or 1) - 1)  # 也可直接写成 1，最稳
    # 参数顺序：(input_dir, overwrite_id, num_processes)
    # 不想覆盖编号就传 None；想强制用某个 DatasetID 就传数字（例如 8）
    convert_msd_dataset(str(p), None, nprocs)

    print(f"[OK] convert_msd_dataset 完成，进程数={nprocs}")

def stage_plan_and_preprocess(dataset_id: str, verify: bool):
    cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id}"
    if verify: cmd += " --verify_dataset_integrity"
    run(cmd)

def stage_train(dataset_id: str, configs, folds, trainer: str, extra: str):
    for c in configs:
        for f in folds:
            cmd = f"nnUNetv2_train {dataset_id} {c} {f} -tr {trainer}"
            if extra: cmd += f" {extra}"
            run(cmd)

def stage_train(dataset_id: str, configs, folds, trainer: str, extra: str):
    for c in configs:
        for f in folds:
            fold_arg = "all" if (isinstance(f, str) and f.lower()=="all") else str(f)
            cmd = f"nnUNetv2_train {dataset_id} {c} {fold_arg} -tr {trainer}"
            if extra: cmd += f" {extra}"
            run(cmd)

def stage_validate_npz(dataset_id: str, configs, folds, trainer: str):
    for c in configs:
        for f in folds:
            if isinstance(f, str) and f.lower()=="all":
                continue  # all 训练没有验证集，跳过
            run(f"nnUNetv2_train {dataset_id} {c} {f} --val --npz -tr {trainer}")

def stage_predict(dataset_id_or_name: str, inp: str, outp: str, config: str,
                  folds, trainer: str, save_prob: bool, chk: str|None,
                  plans: str|None, extra: str):
    folds_arg = "all" if (isinstance(folds, str) and folds == "all") else " ".join(map(str, folds))
    cmd = (
        f'nnUNetv2_predict -d {dataset_id_or_name} '
        f'-i "{inp}" -o "{outp}" -f {folds_arg} -tr {trainer} -c {config}'
    )
    if save_prob: cmd += " --save_probabilities"
    if chk:       cmd += f" -chk {chk}"
    if plans:     cmd += f" -p {plans}"
    if extra:     cmd += f" {extra}"
    run(cmd)

def main():
    # 导出三大目录为环境变量
    if EXPORT_NNUNET_PATHS:
        os.environ["nnUNet_raw"] = NNUNET_RAW
        os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
        os.environ["nnUNet_results"] = NNUNET_RESULTS
        print("[env] nnUNet_raw         =", os.environ["nnUNet_raw"])
        print("[env] nnUNet_preprocessed=", os.environ["nnUNet_preprocessed"])
        print("[env] nnUNet_results     =", os.environ["nnUNet_results"])

    # 可选：只用0号卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if DO_CONVERT:
        if not MSD_INPUT_DIR or not Path(MSD_INPUT_DIR).exists():
            sys.exit("[ERROR] DO_CONVERT=True 但 MSD_INPUT_DIR 无效")
        stage_convert_msd(MSD_INPUT_DIR)

    if DO_PREPROCESS:
        stage_plan_and_preprocess(DATASET_ID, VERIFY_DATASET)

    if DO_TRAIN:
        stage_train(DATASET_ID, TRAIN_CONFIGS, FOLDS, TRAINER, EXTRA_TRAIN_ARGS)

    if DO_VALIDATE_NPZ:
        stage_validate_npz(DATASET_ID, TRAIN_CONFIGS, FOLDS, TRAINER)

    if DO_PREDICT:
        if not Path(PRED_INPUT_DIR).exists():
            sys.exit("[ERROR] PRED_INPUT_DIR 不存在")
        Path(PRED_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        folds = "all" if (isinstance(PRED_FOLDS, str) and PRED_FOLDS.lower()=="all") else PRED_FOLDS
        stage_predict(DATASET_ID, PRED_INPUT_DIR, PRED_OUTPUT_DIR, PRED_CONFIG,
                      folds, TRAINER, PRED_SAVE_PROB, PRED_CHECKPOINT, PRED_PLANS, PRED_EXTRA_ARGS)

    print("\n✅ 全部完成。")

if __name__ == "__main__":
    main()
