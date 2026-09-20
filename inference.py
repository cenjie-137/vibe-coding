import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
def list_cases(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No input files found in {input_dir} with pattern '{pattern}'")
    return files


def case_id_from_path(path: Path, input_suffix: str) -> str:
    name = path.name
    if name.endswith(input_suffix):
        return name[: -len(input_suffix)]
    return path.stem


def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = image.get_fdata().astype(np.float32)
    return data, image


def save_nifti_like(reference: nib.Nifti1Image, data: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data.astype(np.float32), reference.affine, reference.header)
    nib.save(out, str(output_path))


def normalize_volume(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Generic T1-style normalization. Adapt if your model expects a different scheme."""
    valid = volume[np.isfinite(volume)]
    if valid.size == 0:
        return np.zeros_like(volume, dtype=np.float32)

    nonzero = valid[valid != 0]
    if nonzero.size == 0:
        nonzero = valid

    lo, hi = np.percentile(nonzero, [0.5, 99.5])
    clipped = np.clip(volume, lo, hi)

    mean = clipped[clipped != 0].mean() if np.any(clipped != 0) else clipped.mean()
    std = clipped[clipped != 0].std() if np.any(clipped != 0) else clipped.std()
    std = float(std) if std > eps else 1.0
    normalized = (clipped - mean) / std
    return normalized.astype(np.float32)


def dice_score(gt: np.ndarray, pred: np.ndarray, empty_value: float = 1.0) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    gt_sum = gt.sum()
    pred_sum = pred.sum()
    if gt_sum == 0 and pred_sum == 0:
        return empty_value
    intersection = np.logical_and(gt, pred).sum()
    return float((2.0 * intersection) / (gt_sum + pred_sum + 1e-8))


def absolute_volume_difference_ml(gt: np.ndarray, pred: np.ndarray, voxel_spacing_mm: Sequence[float]) -> float:
    voxel_volume_ml = float(np.prod(voxel_spacing_mm[:3]) / 1000.0)
    return float(abs(gt.astype(bool).sum() - pred.astype(bool).sum()) * voxel_volume_ml)


def lesion_count(mask: np.ndarray) -> int:
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    _, num = ndimage.label(mask.astype(bool), structure=structure)
    return int(num)


def lesion_wise_f1(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.25, empty_value: float = 1.0) -> float:
    from scipy import ndimage

    gt = gt.astype(bool)
    pred = pred.astype(bool)
    gt_structure = ndimage.generate_binary_structure(gt.ndim, gt.ndim)
    pred_structure = ndimage.generate_binary_structure(pred.ndim, pred.ndim)
    gt_lab, gt_num = ndimage.label(gt, structure=gt_structure)
    pred_lab, pred_num = ndimage.label(pred, structure=pred_structure)

    if gt_num == 0 and pred_num == 0:
        return empty_value
    if gt_num == 0 or pred_num == 0:
        return 0.0

    gt_slices = ndimage.find_objects(gt_lab)
    pred_slices = ndimage.find_objects(pred_lab)

    # Greedy matching by IoU
    matched_pred = set()
    tp = 0
    for gt_idx, gt_slice in enumerate(gt_slices, start=1):
        if gt_slice is None:
            continue
        gt_component = gt_lab[gt_slice] == gt_idx
        best_pred = None
        best_iou = 0.0
        for pred_idx, pred_slice in enumerate(pred_slices, start=1):
            if pred_slice is None or pred_idx in matched_pred:
                continue
            # quick bounding-box overlap check
            overlap = []
            for a, b in zip(gt_slice, pred_slice):
                start = max(a.start, b.start)
                stop = min(a.stop, b.stop)
                overlap.append(start < stop)
            if not all(overlap):
                continue
            common_slice = tuple(slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(gt_slice, pred_slice))
            gt_sub = (gt_lab[common_slice] == gt_idx)
            pred_sub = (pred_lab[common_slice] == pred_idx)
            inter = np.logical_and(gt_sub, pred_sub).sum()
            if inter == 0:
                continue
            union = gt_component.sum() + (pred_lab[pred_slice] == pred_idx).sum() - inter
            iou = float(inter / (union + 1e-8))
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_idx
        if best_pred is not None and best_iou >= iou_threshold:
            tp += 1
            matched_pred.add(best_pred)

    fp = pred_num - tp
    fn = gt_num - tp
    return float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))


def pr_auc_score(gt: np.ndarray, soft_map: np.ndarray, empty_value: float = 1.0) -> float:
    from sklearn.metrics import auc, precision_recall_curve

    gt_flat = gt.astype(bool).ravel()
    pred_flat = soft_map.astype(np.float32).ravel()
    if gt_flat.shape != pred_flat.shape:
        raise ValueError("ground truth and prediction must have the same shape")
    if gt_flat.sum() == 0:
        if np.all(pred_flat == pred_flat[0]):
            return empty_value
        return 0.0
    precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
    return float(auc(recall, precision))


def build_model(model_name: str, model_kwargs: dict) -> torch.nn.Module:
    """
    Template hook: swap this to your actual ISLES model.
    If your model class lives in `models.py`, this reflection helper can instantiate it.
    """
    import models

    if not hasattr(models, model_name):
        raise AttributeError(f"models.py does not define '{model_name}'")
    model_cls = getattr(models, model_name)
    return model_cls(**model_kwargs)


@torch.no_grad()
def predict_case(model: torch.nn.Module, volume: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Template inference path for a single 3D case.
    Replace this if your model expects patches, multi-channel input, or a different layout.
    """
    x = normalize_volume(volume)
    x = torch.from_numpy(x[None, None]).float().to(device)  # [B, C, D, H, W]
    logits = model(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    return probs.astype(np.float32)


def evaluate_case(
    gt_path: Path,
    soft_map: np.ndarray,
    binary_mask: np.ndarray,
    reference_image: nib.Nifti1Image,
) -> dict:
    gt, _ = load_nifti(gt_path)
    gt = (gt > 0).astype(np.uint8)
    spacing = reference_image.header.get_zooms()[:3]
    return {
        "Dice": dice_score(gt, binary_mask),
        "AbsVolumeDiff_ml": absolute_volume_difference_ml(gt, binary_mask, spacing),
        "LesionCountDiff": abs(lesion_count(gt) - lesion_count(binary_mask)),
        "LesionWiseF1": lesion_wise_f1(gt, binary_mask),
        "PRAUC": pr_auc_score(gt, soft_map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ISLES inference template")
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing NIfTI cases")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder to save predictions")
    parser.add_argument("--labels-dir", type=Path, default=None, help="Optional folder with GT labels")
    parser.add_argument("--input-pattern", default="*_0000.nii.gz", help="Input file glob pattern")
    parser.add_argument("--label-suffix", default=".nii.gz", help="Label filename suffix to strip from case id")
    parser.add_argument("--model-name", default="UNet", help="Model class name from models.py")
    parser.add_argument("--model-kwargs", default="{}", help="JSON dict passed to model constructor")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask")
    parser.add_argument("--save-softmap", action="store_true", help="Save soft probability map as NIfTI")
    parser.add_argument("--save-metrics", action="store_true", help="Save metrics if labels are available")
    args = parser.parse_args()

    model_kwargs = json.loads(args.model_kwargs)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Loading model: {args.model_name} {model_kwargs}", flush=True)
    model = build_model(args.model_name, model_kwargs).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}", flush=True)

    input_files = list_cases(args.input_dir, args.input_pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "preds_raw"
    post_dir = args.output_dir / "preds_post"
    soft_dir = args.output_dir / "softmaps"
    raw_dir.mkdir(parents=True, exist_ok=True)
    post_dir.mkdir(parents=True, exist_ok=True)
    if args.save_softmap:
        soft_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    label_lookup = {}
    if args.labels_dir and args.labels_dir.exists():
        for label_path in args.labels_dir.glob(f"*{args.label_suffix}"):
            case_id = case_id_from_path(label_path, args.label_suffix)
            label_lookup[case_id] = label_path

    for idx, image_path in enumerate(input_files, start=1):
        case_id = case_id_from_path(image_path, "_0000.nii.gz")
        volume, reference = load_nifti(image_path)

        print(f"[{idx}/{len(input_files)}] Processing {case_id}", flush=True)
        soft_map = predict_case(model, volume, device)
        binary_mask = (soft_map >= args.threshold).astype(np.uint8)

        raw_path = raw_dir / f"{case_id}.nii.gz"
        post_path = post_dir / f"{case_id}.nii.gz"
        save_nifti_like(reference, soft_map.astype(np.float32), raw_path)
        save_nifti_like(reference, binary_mask.astype(np.float32), post_path)

        if args.save_softmap:
            save_nifti_like(reference, soft_map.astype(np.float32), soft_dir / f"{case_id}.nii.gz")

        if args.save_metrics and case_id in label_lookup:
            metrics = evaluate_case(label_lookup[case_id], soft_map, binary_mask, reference)
            metrics["case_id"] = case_id
            metrics_rows.append(metrics)
            print(
                f"      Dice={metrics['Dice']:.4f} "
                f"PR-AUC={metrics['PRAUC']:.4f} "
                f"CountDiff={metrics['LesionCountDiff']} "
                f"F1={metrics['LesionWiseF1']:.4f}",
                flush=True,
            )

    if args.save_metrics and metrics_rows:
        csv_path = args.output_dir / "metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

        summary = {}
        for key in ["Dice", "AbsVolumeDiff_ml", "LesionCountDiff", "LesionWiseF1", "PRAUC"]:
            values = np.array([row[key] for row in metrics_rows], dtype=np.float32)
            summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Saved metrics to {csv_path}", flush=True)
        print(f"[INFO] Saved summary to {summary_path}", flush=True)

    print(f"[DONE] Predictions saved to {post_dir}", flush=True)
    if args.save_softmap:
        print(f"[DONE] Soft maps saved to {soft_dir}", flush=True)


if __name__ == "__main__":
    main()
