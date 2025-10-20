#!/usr/bin/env python
"""
opp_semi_auto_annotation.py  (T2‑only edition)
=============================================
Semi‑automatic pipeline to draft pixel‑level masks of cervical OPLL on *single* mid‑sagittal **T2‑weighted** MRI slices.
The script follows the workflow discussed in chat:
  ① ROI constraint (posterior vertebral band)  →
  ② Local multi‑threshold (Otsu / K‑means)      →
  ③ Morphological filtering (remove bone cortex & noise) →
  ④ Save draft mask for expert refinement.

*Changes vs previous dual‑modality version*
-----------------------------------------
* removed all T1 optional branches → simpler dependency chain.
* ROI derived from T2 Sobel edges + posterior dilation.
* Cortex removal uses gradient magnitude only.

Dependencies
------------
conda install -c conda-forge simpleitk scikit-image numpy opencv-python tqdm

Usage (single slice)
--------------------
python opp_semi_auto_annotation.py \
       --t2  /path/OPLL_001_T2_midSag.nii.gz \
       --out_mask  masks_auto/OPLL_001_draft.nii.gz \
       --method  otsu   # or kmeans

Author: ChatGPT‑o3  |  2025‑06‑12
"""
from __future__ import annotations
import argparse, pathlib, numpy as np, SimpleITK as sitk
from skimage import filters, morphology, measure
from sklearn.cluster import KMeans
from scipy import ndimage as ndi

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def n4_bias_correct(img: sitk.Image, shrink: int = 4, iters: int = 50) -> sitk.Image:
    mask = sitk.OtsuThreshold(img, 0, 1)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([iters])
    n4.SetShrinkFactor(shrink)
    return n4.Execute(img, mask)


def zscore(arr: np.ndarray) -> np.ndarray:
    nonzero = arr[arr > 0]
    mu, sd = nonzero.mean(), nonzero.std() if nonzero.size else (0, 1)
    sd = max(sd, 1e-6)
    return (arr - mu) / sd

# ---------------------------------------------------------------------
# ROI generation (posterior 2 mm band)
# ---------------------------------------------------------------------

def create_roi(t2_img: sitk.Image, thickness_mm: float = 2.0) -> np.ndarray:
    spacing = t2_img.GetSpacing()  # (x, y)
    dilate_px = int(round(thickness_mm / spacing[0]))
    arr = sitk.GetArrayFromImage(t2_img)[0]

    # strong Sobel edge ≈ bone cortex
    edge = filters.sobel(arr)
    cortex = edge > np.percentile(edge[arr > 0], 95)
    cortex = morphology.binary_closing(cortex, morphology.disk(3))

    # dilation posterior (downwards in x-axis) – assume image orient → scalp top
    struct = ndi.generate_binary_structure(2, 1)
    roi = ndi.binary_dilation(cortex, structure=struct, iterations=dilate_px)
    roi = morphology.binary_closing(roi, morphology.disk(2))
    return roi.astype(np.uint8)

# ---------------------------------------------------------------------
# Local multi-thresholding
# ---------------------------------------------------------------------

def local_threshold(arr: np.ndarray, roi: np.ndarray, method: str = "otsu") -> np.ndarray:
    roi_vals = arr[roi > 0]
    if roi_vals.size == 0:
        return np.zeros_like(arr)
    if method == "kmeans":
        km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(roi_vals.reshape(-1, 1))
        labels = np.zeros_like(arr, dtype=int)
        labels[roi > 0] = km.labels_ + 1
    else:  # Otsu (2 thresholds -> 3 classes)
        thr1, thr2 = filters.threshold_multiotsu(roi_vals, classes=3)
        labels = np.digitize(arr, bins=[thr1, thr2]) + 1
    return labels

# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run(args):
    t2 = sitk.ReadImage(str(args.t2))
    if args.n4:
        t2 = n4_bias_correct(t2)

    arr = zscore(sitk.GetArrayFromImage(t2)[0])

    # ROI
    roi = create_roi(t2, thickness_mm=args.roi_thickness) if args.roi_mask is None else \
          (sitk.GetArrayFromImage(sitk.ReadImage(str(args.roi_mask)))[0] > 0)

    # multi‑threshold
    labels = local_threshold(arr, roi, method=args.method)
    lowest = labels == labels[roi > 0].min()  # candidate OPLL + cortex

    # Remove bone cortex by gradient magnitude
    grad = filters.sobel(arr)
    cortex_edge = (grad > np.percentile(grad[roi > 0], 95))
    cortex_edge = morphology.binary_closing(cortex_edge, morphology.disk(3))
    candidate = np.logical_and(lowest, np.logical_not(cortex_edge))

    # Morphological cleanup
    cleaned = morphology.remove_small_objects(candidate, min_size=args.min_area)
    lbl = measure.label(cleaned)
    final = np.zeros_like(cleaned, dtype=np.uint8)
    for r in measure.regionprops(lbl):
        if r.major_axis_length >= args.min_major:
            final[lbl == r.label] = 1

    # Save
    out_img = sitk.GetImageFromArray(final[np.newaxis, ...])
    out_img.CopyInformation(t2)
    sitk.WriteImage(out_img, str(args.out_mask))
    print(f"Draft mask saved → {args.out_mask}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def get_parser():
    p = argparse.ArgumentParser(description="Semi‑auto OPLL draft mask generator (T2 sagittal)")
    p.add_argument("--t2", type=pathlib.Path, required=True, help="mid‑sagittal T2 .nii/.nii.gz")
    p.add_argument("--out_mask", type=pathlib.Path, required=True, help="output draft mask path")
    p.add_argument("--roi_mask", type=pathlib.Path, help="optional precomputed ROI mask")
    p.add_argument("--method", choices=["otsu", "kmeans"], default="otsu", help="threshold method")
    p.add_argument("--n4", action="store_true", help="apply N4 bias correction before thresholding")
    p.add_argument("--roi_thickness", type=float, default=2.0, help="posterior dilation thickness (mm)")
    p.add_argument("--min_area", type=int, default=15, help="min connected area (pixels) to keep")
    p.add_argument("--min_major", type=int, default=4, help="min major‑axis length (pixels)")
    return p

if __name__ == "__main__":
    run(get_parser().parse_args())
