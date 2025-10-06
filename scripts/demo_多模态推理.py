# Optional config for better memory efficiency
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from mapanything.models import MapAnything
from mapanything.utils.hf_utils.viz import predictions_to_glb
from mapanything.utils.image import preprocess_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MapAnything inference on saved NBV views and optionally export a GLB."
    )
    parser.add_argument(
        "views_dir",
        type=str,
        help="Path to a directory containing view_XX.pt files (e.g. .../images/step_xxxxx/batch_xxx)",
    )
    parser.add_argument(
        "--model",
        default="facebook/map-anything",
        type=str,
        help="Model identifier to load via MapAnything.from_pretrained",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save a merged GLB (.glb or .gltf) with colored point cloud and cameras.",
    )
    parser.add_argument(
        "--as_mesh",
        action="store_true",
        help="Export the GLB as a triangle mesh instead of a point cloud (default: point cloud).",
    )
    return parser.parse_args()


def load_views_from_directory(directory: str, device: torch.device) -> List[Dict[str, torch.Tensor]]:
    """Load per-view tensors from the trainer export directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Input directory does not exist: {directory}")

    view_files = [
        os.path.join(directory, fname)
        for fname in sorted(os.listdir(directory))
        if fname.startswith("view_") and fname.endswith(".pt")
    ]

    if not view_files:
        raise FileNotFoundError(f"No view_XX.pt files found in {directory}")

    views: List[Dict[str, torch.Tensor]] = []
    for path in view_files:
        payload = torch.load(path, map_location="cpu")

        img: torch.Tensor = payload["img"].to(torch.float32)
        intrinsics: torch.Tensor = payload["intrinsics"].to(torch.float32)
        camera_poses: torch.Tensor = payload["camera_poses"].to(torch.float32)
        is_metric_scale: torch.Tensor = payload.get("is_metric_scale", torch.tensor([True]))

        views.append(
            {
                "img": img,
                "intrinsics": intrinsics,
                "camera_poses": camera_poses,
                "is_metric_scale": is_metric_scale,
            }
        )
        print(f"Loaded view {path} with intrinsics {intrinsics} and camera poses {camera_poses}")

    return views


def collate_predictions_for_glb(
    predictions: List[Dict[str, torch.Tensor]]
) -> Dict[str, np.ndarray]:
    """Convert model predictions into numpy arrays expected by predictions_to_glb."""
    world_points: List[np.ndarray] = []
    images: List[np.ndarray] = []
    final_masks: List[np.ndarray] = []
    extrinsics: List[np.ndarray] = []
    conf_values: List[np.ndarray] = []
    has_conf = any("conf" in pred for pred in predictions)

    for pred_idx, pred in enumerate(predictions):
        required_keys = ("pts3d", "img_no_norm", "mask", "camera_poses")
        for key in required_keys:
            if key not in pred:
                raise KeyError(f"Prediction {pred_idx} is missing required key '{key}'")

        pts3d_np = pred["pts3d"].detach().cpu().numpy()
        images_np = pred["img_no_norm"].detach().cpu().numpy()
        mask_np = pred["mask"].detach().cpu().numpy()
        poses_np = pred["camera_poses"].detach().cpu().numpy()
        conf_np = pred.get("conf")
        if conf_np is not None:
            conf_np = conf_np.detach().cpu().numpy()

        batch_size = pts3d_np.shape[0]
        for batch_idx in range(batch_size):
            pts_frame = pts3d_np[batch_idx]
            image_frame = images_np[batch_idx]
            mask_frame = mask_np[batch_idx]
            pose_frame = poses_np[batch_idx]

            if image_frame.ndim == 3 and image_frame.shape[0] in {3, 4} and image_frame.shape[-1] != 3:
                image_frame = np.transpose(image_frame, (1, 2, 0))
            if mask_frame.ndim == 3 and mask_frame.shape[-1] == 1:
                mask_frame = mask_frame[..., 0]

            world_points.append(np.asarray(pts_frame, dtype=np.float32))
            images.append(np.asarray(image_frame, dtype=np.float32))
            final_masks.append(np.asarray(mask_frame, dtype=bool))
            extrinsics.append(np.linalg.inv(pose_frame).astype(np.float32))

            if conf_np is not None:
                conf_frame = conf_np[batch_idx]
                conf_values.append(np.asarray(conf_frame, dtype=np.float32))

    if not world_points:
        raise ValueError("No predictions available for GLB export")

    result = {
        "world_points": np.stack(world_points, axis=0).astype(np.float32),
        "images": np.stack(images, axis=0).astype(np.float32),
        "final_mask": np.stack(final_masks, axis=0).astype(bool),
        "extrinsic": np.stack(extrinsics, axis=0).astype(np.float32),
    }

    if has_conf and conf_values:
        result["conf"] = np.stack(conf_values, axis=0).astype(np.float32)

    return result


def export_glb(
    predictions: List[Dict[str, torch.Tensor]],
    output_path: Path,
    as_mesh: bool,
) -> Path:
    """Aggregate predictions and export a GLB file containing colored geometry and cameras."""
    output_path = output_path.expanduser()
    if output_path.suffix.lower() not in {".glb", ".gltf"}:
        raise ValueError("--output_path must end with .glb or .gltf")
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_np = collate_predictions_for_glb(predictions)
    scene = predictions_to_glb(
        predictions_np,
        show_cam=True,
        as_mesh=as_mesh,
        mask_ambiguous=True,
    )
    scene.export(str(output_path))
    return output_path


def main() -> None:
    args = parse_args()

    # Get inference device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init model - This requires internet access or a local huggingface cache
    model = MapAnything.from_pretrained(args.model).to(device)

    views_example = load_views_from_directory(args.views_dir, device)
    
    # Preprocess inputs to the expected format
    processed_views = preprocess_inputs(views_example)
    # print(processed_views)
    # Run inference with any combination of inputs
    predictions = model.infer(
        processed_views,
        memory_efficient_inference=False,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
        confidence_percentile=10,
        ignore_calibration_inputs=False,
        ignore_depth_inputs=False,
        ignore_pose_inputs=False,
        ignore_depth_scale_inputs=False,
        ignore_pose_scale_inputs=False,
    )

    if not isinstance(predictions, list):
        raise TypeError(
            f"Expected MapAnything.infer to return a list, got {type(predictions)}"
        )

    print("Inference finished! Per-view outputs:")
    for view_idx, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            raise TypeError(
                f"Expected prediction for view {view_idx} to be a dict, got {type(pred)}"
            )

        print(f"  View {view_idx:02d} ({len(pred)} fields):")
        for key, value in pred.items():
            if isinstance(value, torch.Tensor):
                print(f"    - {key}: Tensor{tuple(value.shape)} {value.dtype}")
            else:
                print(f"    - {key}: {type(value).__name__}")

    if args.output_path is not None:
        saved_path = export_glb(predictions, Path(args.output_path), args.as_mesh)
        print(f"Saved GLB reconstruction to: {saved_path.resolve()}")
    else:
        print("GLB export skipped (provide --output_path to save a reconstruction).")


if __name__ == "__main__":
    main()

'''
示例输入：
python demo_多模态推理.py \
/mnt/sdb/chenmohan/VGGT-NBV/runs/dataset-house3k_bs-1_initv-3_pom-position_only_20251006-154923/images/step_000174/batch_000 \
--output_path /mnt/sdb/chenmohan/VGGT-NBV/map-anything/scripts/test.glb
示例输出：
Inference finished! Per-view outputs:
  View 00 (15 fields):
    - pts3d: Tensor(1, 518, 518, 3) torch.float32
    - pts3d_cam: Tensor(1, 518, 518, 3) torch.float32
    - ray_directions: Tensor(1, 518, 518, 3) torch.float32
    - depth_along_ray: Tensor(1, 518, 518, 1) torch.float32
    - cam_trans: Tensor(1, 3) torch.float32
    - cam_quats: Tensor(1, 4) torch.float32
    - metric_scaling_factor: Tensor(1, 1) torch.float32
    - conf: Tensor(1, 518, 518) torch.float32
    - non_ambiguous_mask: Tensor(1, 518, 518) torch.bool
    - non_ambiguous_mask_logits: Tensor(1, 518, 518) torch.float32
    - img_no_norm: Tensor(1, 518, 518, 3) torch.float32
    - depth_z: Tensor(1, 518, 518, 1) torch.float32
    - intrinsics: Tensor(1, 3, 3) torch.float32
    - camera_poses: Tensor(1, 4, 4) torch.float32
    - mask: Tensor(1, 518, 518, 1) torch.bool
  View 01 (15 fields):
    - pts3d: Tensor(1, 518, 518, 3) torch.float32
    - pts3d_cam: Tensor(1, 518, 518, 3) torch.float32
    - ray_directions: Tensor(1, 518, 518, 3) torch.float32
    - depth_along_ray: Tensor(1, 518, 518, 1) torch.float32
    - cam_trans: Tensor(1, 3) torch.float32
    - cam_quats: Tensor(1, 4) torch.float32
    - metric_scaling_factor: Tensor(1, 1) torch.float32
    - conf: Tensor(1, 518, 518) torch.float32
    - non_ambiguous_mask: Tensor(1, 518, 518) torch.bool
    - non_ambiguous_mask_logits: Tensor(1, 518, 518) torch.float32
    - img_no_norm: Tensor(1, 518, 518, 3) torch.float32
    - depth_z: Tensor(1, 518, 518, 1) torch.float32
    - intrinsics: Tensor(1, 3, 3) torch.float32
    - camera_poses: Tensor(1, 4, 4) torch.float32
    - mask: Tensor(1, 518, 518, 1) torch.bool
'''
