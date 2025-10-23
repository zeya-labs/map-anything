"""
Rebuild MapAnything predicted point clouds using the exact training pipeline.

Usage:
    python -m map-anything.scripts.repro_training_pointcloud \
        <views_dir> --model facebook/map-anything
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from mapanything.models import MapAnything
from mapanything.utils.geometry import rotation_matrix_to_quaternion
from mapanything.utils.image import IMAGE_NORMALIZATION_DICT
from mapanything.utils.inference import (
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from nbv_framework.training.loss.reconstruction import ReconstructionLoss
from nbv_framework.utils.tensorboard_mesh import log_point_clouds_to_tensorboard
from pytorch3d.structures import Pointclouds

_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the training-time MapAnything predicted point cloud export "
            "for a saved view directory."
        )
    )
    parser.add_argument(
        "views_dir",
        type=str,
        help="Directory containing view_XX.pt files produced by prepare_mapanything_views.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/map-anything",
        help="Model identifier or local checkpoint passed to MapAnythingWrapper.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to write the reconstructed GLB. Defaults to <views_dir>_training_predicted.glb.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=4096,
        help="Maximum number of points per cloud when exporting the GLB.",
    )
    parser.add_argument(
        "--black_bg_threshold",
        type=float,
        default=0.05,
        help="Intensity threshold used to filter black background pixels (matches training default).",
    )
    return parser.parse_args()


def _load_views(directory: Path) -> List[Dict[str, torch.Tensor]]:
    view_files = sorted(
        directory.glob("view_*.pt"),
        key=lambda p: p.name,
    )
    if not view_files:
        raise FileNotFoundError(f"No view_*.pt files found in {directory}")

    views: List[Dict[str, torch.Tensor]] = []
    for path in view_files:
        payload = torch.load(path, map_location="cpu")
        views.append(payload)
    return views


def _assemble_batch(
    views: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Match the tensor shapes used during training."""
    if not views:
        raise ValueError("No views provided.")

    images_list: List[torch.Tensor] = []
    pose_vec_list: List[torch.Tensor] = []
    depth_list: List[torch.Tensor] = []
    gt_mask_list: List[torch.Tensor] = []
    gt_map_list: List[torch.Tensor] = []

    for view in views:
        image = view["img"]
        if image.dim() == 3:
            if image.shape[0] == 3:
                image = image.unsqueeze(0)
            elif image.shape[-1] == 3:
                image = image.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported img shape {tuple(image.shape)}")
        elif image.dim() == 4:
            if image.shape[1] == 3:
                pass
            elif image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported img shape {tuple(image.shape)}")
        else:
            raise ValueError(f"Unsupported img rank {image.dim()}")

        image = image.to(device=device, dtype=torch.float32)
        if torch.amax(image) > 1.0:
            image = image / 255.0
        image = image.clamp(0.0, 1.0)
        images_list.append(image)

        pose_mat = view["camera_poses"]
        if pose_mat.dim() == 2:
            pose_mat = pose_mat.view(4, 4).unsqueeze(0)
        pose_mat = pose_mat.to(device=device, dtype=torch.float32)
        rot = pose_mat[:, :3, :3]
        quat = rotation_matrix_to_quaternion(rot)
        trans = pose_mat[:, :3, 3]
        pose_vec = torch.cat([trans, quat], dim=-1)
        pose_vec_list.append(pose_vec)

        depth = view.get("depth_z")
        if depth is not None:
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
            elif depth.dim() == 4 and depth.shape[-1] == 1:
                depth = depth.squeeze(-1)
            elif depth.dim() != 3:
                raise ValueError(f"Unsupported depth shape {tuple(depth.shape)}")
            depth_list.append(depth.to(device=device, dtype=torch.float32))

        gt_mask = view.get("gt_valid_mask")
        if gt_mask is not None:
            if gt_mask.dim() == 2:
                gt_mask = gt_mask.unsqueeze(0)
            gt_mask_list.append(gt_mask.to(device=device, dtype=torch.bool))

        gt_map = view.get("gt_point_map")
        if gt_map is not None:
            if gt_map.dim() == 3:
                gt_map = gt_map.unsqueeze(0)
            gt_map_list.append(gt_map.to(device=device, dtype=torch.float32))

    images_tensor = torch.stack(images_list, dim=1).contiguous()
    pose_vec_tensor = torch.stack(pose_vec_list, dim=1).contiguous()
    depth_tensor = (
        torch.stack(depth_list, dim=1).contiguous() if depth_list else None
    )
    gt_valid_masks = (
        torch.stack(gt_mask_list, dim=1).contiguous() if gt_mask_list else None
    )
    gt_point_maps = (
        torch.stack(gt_map_list, dim=1).contiguous() if gt_map_list else None
    )
    return images_tensor, pose_vec_tensor, depth_tensor, gt_valid_masks, gt_point_maps


def _get_normalization_tensors(
    data_norm_type: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = IMAGE_NORMALIZATION_DICT.get(data_norm_type)
    if cfg is not None:
        mean = torch.as_tensor(cfg.mean, dtype=dtype, device=device)
        std = torch.as_tensor(cfg.std, dtype=dtype, device=device)
    else:
        mean = _DEFAULT_MEAN.to(device=device, dtype=dtype)
        std = _DEFAULT_STD.to(device=device, dtype=dtype)
    return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)


def _prepare_views_for_mapanything(
    raw_views: List[Dict[str, torch.Tensor]],
    images_batch: torch.Tensor,
    depth_tensor: Optional[torch.Tensor],
    *,
    device: torch.device,
    data_norm_type: str = "dinov2",
) -> List[Dict[str, torch.Tensor]]:
    if not raw_views:
        raise ValueError("raw_views must contain at least one element")

    batch_size, num_views = images_batch.shape[:2]
    mean_tensor, std_tensor = _get_normalization_tensors(
        data_norm_type,
        device=device,
        dtype=images_batch.dtype,
    )

    processed_views: List[Dict[str, torch.Tensor]] = []
    for view_idx in range(num_views):
        raw_view = raw_views[view_idx]

        image_tensor = images_batch[:, view_idx].to(device=device, dtype=torch.float32)
        normalized = (image_tensor - mean_tensor) / std_tensor

        intrinsics = raw_view["intrinsics"]
        if intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0)
        intrinsics = intrinsics.to(device=device, dtype=torch.float32)
        if intrinsics.shape[0] == 1 and batch_size > 1:
            intrinsics = intrinsics.expand(batch_size, -1, -1)

        cam_pose = raw_view["camera_poses"]
        if cam_pose.dim() == 2:
            cam_pose = cam_pose.unsqueeze(0)
        cam_pose = cam_pose.to(device=device, dtype=torch.float32)
        if cam_pose.shape[0] == 1 and batch_size > 1:
            cam_pose = cam_pose.expand(batch_size, -1, -1)

        is_metric_scale = raw_view.get("is_metric_scale")
        if is_metric_scale is None:
            is_metric_tensor = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            if is_metric_scale.dim() == 0:
                is_metric_tensor = is_metric_scale.bool().unsqueeze(0).to(device=device)
            else:
                is_metric_tensor = is_metric_scale.to(device=device, dtype=torch.bool)
            if is_metric_tensor.shape[0] == 1 and batch_size > 1:
                is_metric_tensor = is_metric_tensor.expand(batch_size)

        view_dict: Dict[str, Any] = {
            "img": normalized,
            "data_norm_type": [data_norm_type],
            "intrinsics": intrinsics,
            "camera_poses": cam_pose,
            "is_metric_scale": is_metric_tensor,
        }

        if depth_tensor is not None:
            depth_per_view = depth_tensor[:, view_idx]
            view_dict["depth_z"] = depth_per_view.to(device=device, dtype=torch.float32)

        processed_views.append(view_dict)

    validated_views = validate_input_views_for_inference(processed_views)
    ready_views = preprocess_input_views_for_inference(validated_views)
    print("views keys:", ready_views[0].keys())
    return ready_views


def _stack_predictions(predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    stacked: Dict[str, List[torch.Tensor]] = {}
    for view_pred in predictions:
        for key, value in view_pred.items():
            if torch.is_tensor(value):
                stacked.setdefault(key, []).append(value)

    result: Dict[str, torch.Tensor] = {}
    for key, tensors in stacked.items():
        result[key] = torch.stack(tensors, dim=1)

    if "pts3d" in result:
        pts3d = result["pts3d"]
        result["world_points_from_depth"] = pts3d
        result["world_points"] = pts3d
    if "conf" in result:
        conf = result["conf"]
        result["depth_conf"] = conf
        result["world_points_conf"] = conf
    if "pts3d_cam" in result and "depth" not in result:
        result["depth"] = result["pts3d_cam"][..., 2:3]
    return result

def _export_pointcloud_glb(
    point_clouds: Pointclouds,
    output_path: Path,
    max_points: int,
) -> None:
    point_specs = [
        ("predicted", point_clouds, np.array([0, 0, 255], dtype=np.uint8)),
    ]
    log_point_clouds_to_tensorboard(
        writer=None,
        tag="Repro/Predicted",
        point_cloud_specs=point_specs,
        step=0,
        batch_index=0,
        max_points_per_cloud=max_points,
        glb_output_path=str(output_path),
    )


def main() -> None:
    args = parse_args()
    views_path = Path(args.views_dir)
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else views_path.parent / f"{views_path.name}_training_predicted.glb"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    views = _load_views(views_path)
    (
        images_batch,
        _camera_pose_vec,
        depth_tensor,
        gt_valid_masks,
        gt_point_maps,
    ) = _assemble_batch(views, device)

    model = MapAnything.from_pretrained(args.model).to(device)
    model.eval()

    prepared_views = _prepare_views_for_mapanything(
        views,
        images_batch,
        depth_tensor,
        device=device,
    )

    use_depth = depth_tensor is not None
    try:
        if hasattr(model, "_configure_geometric_input_config"):
            model._configure_geometric_input_config(
                use_calibration=True,
                use_depth=use_depth,
                use_pose=True,
                use_depth_scale=False,
                use_pose_scale=False,
            )
        predictions = model.forward(
            prepared_views,
            memory_efficient_inference=False,
        )
    finally:
        if hasattr(model, "_restore_original_geometric_input_config"):
            model._restore_original_geometric_input_config()

    if not isinstance(predictions, list) or not predictions:
        raise RuntimeError("MapAnything forward did not return per-view predictions list")

    print("predictions keys:", predictions[0].keys())
    recon = _stack_predictions(predictions)

    reconstruction_loss = ReconstructionLoss(
        save_point_clouds=False,
        log_tensorboard=False,
    )
    pred_pointclouds, _ = reconstruction_loss.extract_point_cloud_from_reconstruction(
        recon_data=recon,
        combined_images_batch=images_batch,
        confidence_threshold=0.0,
        source="vggt",
        gt_valid_masks=gt_valid_masks,
    )

    _export_pointcloud_glb(pred_pointclouds, output_path, args.max_points)
    print(f"Wrote predicted point cloud to {output_path.resolve()}")


if __name__ == "__main__":
    main()
