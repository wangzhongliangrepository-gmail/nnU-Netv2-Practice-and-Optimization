#!/usr/bin/env python
"""
opp_semi_auto_annotation.py (T2-only, one-click edition)
======================================================
► 目的：让临床老师在 *PyCharm 双击 Run* 或 Windows 直接 `python opp_semi_auto_annotation.py` 即运行，
  **无需填写命令行参数**。

核心改动
--------
1. 在脚本顶部设置 **默认路径**：`DEFAULT_T2_PATH`, `DEFAULT_OUT_MASK_PATH`。
2. `argparse` 参数变为 *可选*；如未传入则回落到默认路径。
3. 若 `DEFAULT_*` 为空且未传参数 → 提示并退出。

依赖与算法流程与上一版一致（ROI → Otsu/K-means → 形态学）。

Quick Start
~~~~~~~~~~~
1. 修改下方 `DEFAULT_T2_PATH` 与 `DEFAULT_OUT_MASK_PATH`（绝对或相对路径）。
2. 运行：
   ```bash
   conda activate opp-mark
   python opp_semi_auto_annotation.py        # 一键
   ```
3. 若仍想临时改输入，可照旧写命令行 `--t2 ... --out_mask ...` 以覆盖默认。
"""
from __future__ import annotations
import argparse, pathlib, sys, numpy as np, SimpleITK as sitk
from skimage import filters, morphology, measure
from sklearn.cluster import KMeans
from scipy import ndimage as ndi

# ------------------------------------------------------------------
# ✏️  在此填写您常用的影像路径
# ------------------------------------------------------------------
DEFAULT_T2_PATH = r"C:\Users\wangzhongliang\Desktop\Task07_OPLL\imagesTr\OPLL_006.nii.gz"
DEFAULT_OUT_MASK_PATH = r"C:\Users\wangzhongliang\Desktop\Task07_OPLL\OPLL_06_draft.nii.gz"
# ------------------------------------------------------------------


def n4_bias_correct(img: sitk.Image, shrink: int = 4, iters: int = 50) -> sitk.Image:
    mask = sitk.OtsuThreshold(img, 0, 1)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([iters])
    n4.SetShrinkFactor(shrink)
    return n4.Execute(img, mask)


def zscore(arr: np.ndarray) -> np.ndarray:
    nz = arr[arr > 0]
    mu, sd = (nz.mean(), nz.std()) if nz.size else (0, 1)
    sd = max(sd, 1e-6)
    return (arr - mu) / sd


def create_roi(t2_img: sitk.Image, thick_mm: float = 2.0) -> np.ndarray:
    arr = sitk.GetArrayFromImage(t2_img)[0]
    spacing = t2_img.GetSpacing()
    dil_px = int(round(thick_mm / spacing[0]))
    edges = filters.sobel(arr)
    cortex = edges > np.percentile(edges[arr > 0], 95)
    cortex = morphology.binary_closing(cortex, morphology.disk(3))
    roi = ndi.binary_dilation(cortex, ndi.generate_binary_structure(2, 1), dil_px)
    roi = morphology.binary_closing(roi, morphology.disk(2))
    return roi.astype(np.uint8)


def local_thresh(arr: np.ndarray, roi: np.ndarray, method: str) -> np.ndarray:
    roi_vals = arr[roi > 0]
    if roi_vals.size == 0:
        return np.zeros_like(arr)
    if method == "kmeans":
        labels_all = np.zeros_like(arr, dtype=int)
        km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(roi_vals.reshape(-1, 1))
        labels_all[roi > 0] = km.labels_ + 1
    else:
        thr1, thr2 = filters.threshold_multiotsu(roi_vals, classes=3)
        labels_all = np.digitize(arr, bins=[thr1, thr2]) + 1
    return labels_all


def run(args):
    t2_path = args.t2 or DEFAULT_T2_PATH
    out_path = args.out_mask or DEFAULT_OUT_MASK_PATH

    if not t2_path or not out_path:
        sys.exit("[ERROR] --t2 and --out_mask not provided and DEFAULT_* paths are empty. Edit script header or pass parameters.")

    t2 = sitk.ReadImage(str(t2_path))
    if args.n4:
        t2 = n4_bias_correct(t2)
    arr = zscore(sitk.GetArrayFromImage(t2)[0])

    roi = create_roi(t2, thick_mm=args.roi_thickness) if args.roi_mask is None else \
          (sitk.GetArrayFromImage(sitk.ReadImage(str(args.roi_mask)))[0] > 0)

    labels = local_thresh(arr, roi, args.method)
    lowest = labels == labels[roi > 0].min()

    grad = filters.sobel(arr)
    cortex = grad > np.percentile(grad[roi > 0], 95)
    cortex = morphology.binary_closing(cortex, morphology.disk(3))
    candidate = np.logical_and(lowest, np.logical_not(cortex))

    cleaned = morphology.remove_small_objects(candidate, min_size=args.min_area)
    lbl = measure.label(cleaned)
    final = np.zeros_like(cleaned, np.uint8)
    for reg in measure.regionprops(lbl):
        if reg.major_axis_length >= args.min_major:
            final[lbl == reg.label] = 1

    out_img = sitk.GetImageFromArray(final[np.newaxis, ...])
    out_img.CopyInformation(t2)
    sitk.WriteImage(out_img, str(out_path))
    print(f"Draft mask saved ➜ {out_path}")


def get_parser():
    p = argparse.ArgumentParser(description="One-click OPLL draft mask generator (T2 sagittal)")
    p.add_argument("--t2", type=pathlib.Path, help="T2 sagittal .nii/.nii.gz path (optional if default set)")
    p.add_argument("--out_mask", type=pathlib.Path, help="output mask path (optional if default set)")
    p.add_argument("--roi_mask", type=pathlib.Path, help="optional custom ROI mask")
    p.add_argument("--method", choices=["otsu", "kmeans"], default="kmeans", help="threshold method")
    p.add_argument("--n4", action="store_true", help="apply N4 bias field correction")
    p.add_argument("--roi_thickness", type=float, default=2.0, help="posterior dilation thickness (mm)")
    p.add_argument("--min_area", type=int, default=15, help="min connected area (px)")
    p.add_argument("--min_major", type=int, default=4, help="min major-axis len (px)")
    return p


if __name__ == "__main__":
    run(get_parser().parse_args())