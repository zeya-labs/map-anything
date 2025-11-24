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
import torch.nn.functional as F

from mapanything.models import MapAnything
from mapanything.utils.geometry import rotation_matrix_to_quaternion
from mapanything.utils.image import IMAGE_NORMALIZATION_DICT, preprocess_inputs
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


def _to_hw3_tensor(image: torch.Tensor) -> torch.Tensor:
    """Convert stored image tensor to channel-last (H, W, 3) format on CPU."""
    if image.dim() == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.dim() != 3:
        raise ValueError(f"Unsupported img shape {tuple(image.shape)}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.permute(1, 2, 0)
    elif image.shape[-1] != 3:
        raise ValueError(f"Expected image channel dimension size 3, got {tuple(image.shape)}")
    return image.detach().cpu().to(torch.float32)


def _prepare_preprocess_inputs(
    raw_views: List[Dict[str, torch.Tensor]],
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, Optional[torch.Tensor]]]]:
    """Convert raw saved views to the format expected by preprocess_inputs."""
    formatted: List[Dict[str, torch.Tensor]] = []
    gt_metadata: List[Dict[str, Optional[torch.Tensor]]] = []

    for view in raw_views:
        formatted_view: Dict[str, torch.Tensor] = {}

        image_tensor = _to_hw3_tensor(view["img"])
        formatted_view["img"] = image_tensor

        intrinsics = view.get("intrinsics")
        if intrinsics is not None:
            formatted_view["intrinsics"] = intrinsics.detach().cpu().to(torch.float32)

        camera_poses = view.get("camera_poses")
        if camera_poses is not None:
            camera_poses_cpu = camera_poses.detach().cpu().to(torch.float32)
            if camera_poses_cpu.dim() == 3 and camera_poses_cpu.shape[0] == 1:
                camera_poses_cpu = camera_poses_cpu.squeeze(0)
            formatted_view["camera_poses"] = camera_poses_cpu

        depth_z = view.get("depth_z")
        if depth_z is not None:
            depth_tensor = depth_z.detach().cpu().to(torch.float32)
            if depth_tensor.dim() == 3 and depth_tensor.shape[0] == 1:
                depth_tensor = depth_tensor.squeeze(0)
            formatted_view["depth_z"] = depth_tensor

        is_metric_scale = view.get("is_metric_scale")
        if is_metric_scale is not None:
            if isinstance(is_metric_scale, torch.Tensor):
                formatted_view["is_metric_scale"] = is_metric_scale.detach().cpu()
            else:
                formatted_view["is_metric_scale"] = torch.tensor(is_metric_scale, dtype=torch.bool)

        formatted.append(formatted_view)

        gt_metadata.append(
            {
                "gt_valid_mask": view.get("gt_valid_mask"),
                "gt_point_map": view.get("gt_point_map"),
            }
        )

    return formatted, gt_metadata


def _resize_gt_metadata(
    processed_views: List[Dict[str, torch.Tensor]],
    gt_metadata: List[Dict[str, Optional[torch.Tensor]]],
) -> List[Dict[str, Optional[torch.Tensor]]]:
    """Resize optional ground truth maps to match processed resolution."""
    resized_metadata: List[Dict[str, Optional[torch.Tensor]]] = []

    for view, extras in zip(processed_views, gt_metadata):
        height = int(view["img"].shape[-2])
        width = int(view["img"].shape[-1])

        resized_entry: Dict[str, Optional[torch.Tensor]] = {"gt_valid_mask": None, "gt_point_map": None}

        mask = extras.get("gt_valid_mask")
        if mask is not None:
            mask_tensor = torch.as_tensor(mask).detach().cpu()
            if mask_tensor.dim() == 4 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)
            if mask_tensor.dim() == 3 and mask_tensor.shape[-1] == 1:
                mask_tensor = mask_tensor.permute(2, 0, 1)
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            elif mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.unsqueeze(0)
            else:
                raise ValueError(f"Unsupported gt_valid_mask shape {tuple(mask_tensor.shape)}")

            mask_resized = F.interpolate(mask_tensor.float(), size=(height, width), mode="nearest")
            resized_entry["gt_valid_mask"] = mask_resized.squeeze(0).squeeze(0).to(torch.bool)

        point_map = extras.get("gt_point_map")
        if point_map is not None:
            point_tensor = torch.as_tensor(point_map).detach().cpu().to(torch.float32)
            if point_tensor.dim() == 4 and point_tensor.shape[0] == 1:
                point_tensor = point_tensor.squeeze(0)
            if point_tensor.dim() == 4:
                raise ValueError(f"Unsupported gt_point_map shape {tuple(point_tensor.shape)}")
            if point_tensor.dim() == 3 and point_tensor.shape[0] == 3:
                point_tensor = point_tensor
            elif point_tensor.dim() == 3 and point_tensor.shape[-1] == 3:
                point_tensor = point_tensor.permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported gt_point_map shape {tuple(point_tensor.shape)}")

            point_tensor = point_tensor.unsqueeze(0)
            point_resized = F.interpolate(point_tensor, size=(height, width), mode="bilinear", align_corners=False)
            resized_entry["gt_point_map"] = point_resized.squeeze(0).permute(1, 2, 0)

        resized_metadata.append(resized_entry)

    while len(resized_metadata) < len(processed_views):
        resized_metadata.append({"gt_valid_mask": None, "gt_point_map": None})

    return resized_metadata


def _assemble_batch(
    views: List[Dict[str, torch.Tensor]],
    device: torch.device,
    gt_metadata: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
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

    for view_idx, view in enumerate(views):
        metadata_entry = gt_metadata[view_idx] if gt_metadata is not None else None
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
        norm_type = view.get("data_norm_type", ["dinov2"])[0]
        mean_tensor, std_tensor = _get_normalization_tensors(
            norm_type,
            device=device,
            dtype=image.dtype,
        )
        image_denorm = (image * std_tensor) + mean_tensor
        image_denorm = image_denorm.clamp(0.0, 1.0)
        images_list.append(image_denorm)

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
        elif metadata_entry is not None:
            mask_meta = metadata_entry.get("gt_valid_mask")
            if mask_meta is not None:
                mask_tensor = mask_meta
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
                gt_mask_list.append(mask_tensor.to(device=device, dtype=torch.bool))

        gt_map = view.get("gt_point_map")
        if gt_map is not None:
            if gt_map.dim() == 3:
                gt_map = gt_map.unsqueeze(0)
            gt_map_list.append(gt_map.to(device=device, dtype=torch.float32))
        elif metadata_entry is not None:
            gt_point_map_meta = metadata_entry.get("gt_point_map")
            if gt_point_map_meta is not None:
                gt_map_tensor = gt_point_map_meta
                if gt_map_tensor.dim() == 3 and gt_map_tensor.shape[-1] == 3:
                    gt_map_tensor = gt_map_tensor.unsqueeze(0)
                elif gt_map_tensor.dim() == 3 and gt_map_tensor.shape[0] == 3:
                    gt_map_tensor = gt_map_tensor.unsqueeze(0)
                gt_map_list.append(gt_map_tensor.to(device=device, dtype=torch.float32))

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
    processed_views: List[Dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    if not processed_views:
        raise ValueError("processed_views must contain at least one element")

    device_ready_views: List[Dict[str, Any]] = []
    for view in processed_views:
        view_dict: Dict[str, Any] = {
            "img": view["img"].to(device=device, dtype=torch.float32).contiguous(),
            "data_norm_type": view.get("data_norm_type", ["dinov2"]),
        }

        intrinsics = view.get("intrinsics")
        if intrinsics is not None:
            intrinsics_tensor = intrinsics.to(device=device, dtype=torch.float32)
            view_dict["intrinsics"] = intrinsics_tensor

        cam_pose = view.get("camera_poses")
        if cam_pose is not None:
            cam_pose_tensor = cam_pose.to(device=device, dtype=torch.float32)
            view_dict["camera_poses"] = cam_pose_tensor

        depth_tensor = view.get("depth_z")
        if depth_tensor is not None:
            depth_on_device = depth_tensor.to(device=device, dtype=torch.float32)
            view_dict["depth_z"] = depth_on_device

        is_metric_scale = view.get("is_metric_scale")
        if is_metric_scale is not None:
            if isinstance(is_metric_scale, torch.Tensor):
                is_metric_tensor = is_metric_scale.to(device=device, dtype=torch.bool)
            else:
                is_metric_tensor = torch.as_tensor(
                    is_metric_scale,
                    device=device,
                    dtype=torch.bool,
                )
            view_dict["is_metric_scale"] = is_metric_tensor

        device_ready_views.append(view_dict)

    validated_views = validate_input_views_for_inference(device_ready_views)
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

    raw_views = _load_views(views_path)
    preprocess_input_views, raw_gt_metadata = _prepare_preprocess_inputs(raw_views)
    processed_views = preprocess_inputs(preprocess_input_views)
    resized_gt_metadata = _resize_gt_metadata(processed_views, raw_gt_metadata)
    (
        images_batch,
        _camera_pose_vec,
        depth_tensor,
        gt_valid_masks,
        gt_point_maps,
    ) = _assemble_batch(processed_views, device, resized_gt_metadata)

    model = MapAnything.from_pretrained(args.model).to(device)
    model.eval()

    prepared_views = _prepare_views_for_mapanything(
        processed_views,
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
        default_device=device,
        tensor_dtype=torch.float32,
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
