# Optional config for better memory efficiency
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Required imports
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import rerun as rr
import torch

from mapanything.models import MapAnything
from mapanything.utils.hf_utils.viz import predictions_to_glb
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.viz import script_add_rerun_args, script_setup_with_port


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
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable visualization with Rerun.",
    )
    parser.add_argument(
        "--filter_black_bg",
        action="store_true",
        help="Filter near-black pixels from outputs to remove background noise.",
    )
    parser.add_argument(
        "--black_bg_threshold",
        type=float,
        default=0.2,
        help="Threshold in [0, 1] used with --filter_black_bg to drop pixels whose max RGB is below this value.",
    )
    script_add_rerun_args(parser)
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
        depth_z: torch.Tensor = payload["depth_z"].to(torch.float32)
        is_metric_scale: torch.Tensor = payload.get("is_metric_scale", torch.tensor([True]))

        views.append(
            {
                "img": img,
                "intrinsics": intrinsics,
                "camera_poses": camera_poses,
                "depth_z": depth_z,
                "is_metric_scale": is_metric_scale,
            }
        )
        # print({
        #         "img": img,
        #         "intrinsics": intrinsics,
        #         "camera_poses": camera_poses,
        #         "depth_z": depth_z,
        #         "is_metric_scale": is_metric_scale,
        #     })
        print(f"Loaded view {path} with is_metric_scale {is_metric_scale}")

    return views


def collate_predictions_for_glb(
    predictions: List[Dict[str, torch.Tensor]],
    filter_black_bg: bool = False,
    black_bg_threshold: float = 0.0,
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

            frame_mask = np.asarray(mask_frame, dtype=bool)
            image_frame_np = np.asarray(image_frame, dtype=np.float32)
            if filter_black_bg and image_frame_np.ndim == 3 and image_frame_np.shape[-1] >= 3:
                color_for_filter = image_frame_np[..., :3]
                max_val = float(color_for_filter.max()) if color_for_filter.size else 0.0
                if max_val > 1.5:
                    color_for_filter = color_for_filter / 255.0
                frame_mask &= np.max(color_for_filter, axis=-1) > black_bg_threshold

            world_points.append(np.asarray(pts_frame, dtype=np.float32))
            images.append(image_frame_np)
            final_masks.append(frame_mask)
            # MapAnything camera_poses are already cam2world; exporting them directly keeps orientation correct.
            extrinsics.append(np.asarray(pose_frame, dtype=np.float32))

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


def log_prediction_to_rerun(
    pred: Dict[str, torch.Tensor],
    view_idx: int,
    filter_black_bg: bool,
    black_bg_threshold: float,
) -> None:
    """Log prediction tensors to a Rerun timeline for visualization."""
    image_tensor = pred["img_no_norm"][0]
    image_np = image_tensor.detach().cpu().numpy()
    if image_np.ndim == 3 and image_np.shape[0] in {3, 4} and image_np.shape[-1] not in {3, 4}:
        image_np = np.transpose(image_np, (1, 2, 0))
    image_np = image_np.astype(np.float32, copy=False)

    depth_tensor = pred["depth_z"][0]
    depth_np = depth_tensor.detach().cpu().numpy()
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    depth_np = depth_np.astype(np.float32, copy=False)

    pose_np = pred["camera_poses"][0].detach().cpu().numpy()
    intrinsics_np = pred["intrinsics"][0].detach().cpu().numpy()

    pts3d_np = pred["pts3d"][0].detach().cpu().numpy()
    pts3d_np = pts3d_np.astype(np.float32, copy=False)

    mask_tensor = pred["mask"][0]
    mask_np = mask_tensor.detach().cpu().numpy()
    if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
        mask_np = mask_np[..., 0]
    mask_np = mask_np.astype(bool, copy=False)

    if filter_black_bg and image_np.ndim == 3 and image_np.shape[-1] >= 3:
        color_for_filter = image_np[..., :3]
        max_val = float(color_for_filter.max()) if color_for_filter.size else 0.0
        if max_val > 1.5:
            color_for_filter = color_for_filter / 255.0
        mask_np &= np.max(color_for_filter, axis=-1) > black_bg_threshold

    base_name = f"mapanything/view_{view_idx:02d}"
    pts_name = f"mapanything/pointcloud_view_{view_idx:02d}"

    height, width = image_np.shape[0], image_np.shape[1]

    rr.log(
        base_name,
        rr.Transform3D(
            translation=pose_np[:3, 3],
            mat3x3=pose_np[:3, :3],
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics_np,
            height=height,
            width=width,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1.0,
        ),
    )
    rr.log(f"{base_name}/pinhole/rgb", rr.Image(image_np))
    rr.log(f"{base_name}/pinhole/depth", rr.DepthImage(depth_np))
    rr.log(f"{base_name}/pinhole/mask", rr.SegmentationImage(mask_np.astype(int)))

    if mask_np.any():
        filtered_pts = pts3d_np[mask_np]
        filtered_cols = image_np[mask_np]
        rr.log(
            pts_name,
            rr.Points3D(
                positions=filtered_pts.reshape(-1, 3).astype(np.float32, copy=False),
                colors=filtered_cols.reshape(-1, filtered_cols.shape[-1]).astype(np.float32, copy=False),
            ),
        )
    else:
        rr.log(
            pts_name,
            rr.Points3D(
                positions=np.zeros((0, 3), dtype=np.float32),
            ),
        )


def export_glb(
    predictions: List[Dict[str, torch.Tensor]],
    output_path: Path,
    input_path: Path,
    as_mesh: bool,
    filter_black_bg: bool,
    black_bg_threshold: float,
) -> Path:
    """Aggregate predictions and export a GLB file containing colored geometry and cameras."""
    if output_path is None:
        # 使用输入路径生成默认输出路径
        input_path_obj = Path(input_path)
        output_path = input_path_obj.parent / f"{input_path_obj.name}_mapanything.glb"
    else:
        output_path = output_path.expanduser()
        if output_path.suffix.lower() not in {".glb", ".gltf"}:
            raise ValueError("--output_path must end with .glb or .gltf")
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_np = collate_predictions_for_glb(
        predictions,
        filter_black_bg=filter_black_bg,
        black_bg_threshold=black_bg_threshold,
    )
    scene = predictions_to_glb(
        predictions_np,
        show_cam=True,
        as_mesh=as_mesh,
        mask_black_bg=True,
        mask_ambiguous=True,
    )
    scene.export(str(output_path))
    return output_path


def main() -> None:
    args = parse_args()

    if getattr(args, "save", None):
        save_path = Path(args.save).expanduser()
        if save_path.parent:
            save_path.parent.mkdir(parents=True, exist_ok=True)

    if args.filter_black_bg:
        args.black_bg_threshold = float(np.clip(args.black_bg_threshold, 0.0, 1.0))

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
        edge_normal_threshold=5.0,
        edge_depth_threshold=0.03,
        apply_confidence_mask=False,
        confidence_percentile=10,
        ignore_calibration_inputs=False,
        ignore_depth_inputs=False,
        ignore_pose_inputs=False,
        ignore_depth_scale_inputs=False,
        ignore_pose_scale_inputs=False,
    )

    default_rerun_url = "rerun+http://127.0.0.1:2004/proxy"

    if args.viz and args.serve and args.connect and args.url == default_rerun_url:
        print(
            "Detected --serve without an explicit --url; skipping default remote connection."
        )
        args.connect = False

    if args.viz:
        viz_identifier = "MapAnything_Multimodal_Inference"
        try:
            script_setup_with_port(args, viz_identifier)
        except RuntimeError as err:
            if "Address already in use" in str(err):
                raise RuntimeError(
                    f"Failed to launch Rerun web viewer on port {args.web_port}. "
                    "Free the port or rerun with --web_port <PORT> to use a different one."
                ) from err
            raise
        rr.set_time("stable_time", sequence=0)
        rr.log("mapanything", rr.ViewCoordinates.RDF, static=True)

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

        if args.viz:
            log_prediction_to_rerun(
                pred,
                view_idx,
                filter_black_bg=args.filter_black_bg,
                black_bg_threshold=args.black_bg_threshold,
            )

    if args.viz:
        print("Visualization complete! Check the Rerun viewer.")

    # 总是导出GLB文件，如果没有指定输出路径则使用默认路径
    output_path = Path(args.output_path) if args.output_path is not None else None
    saved_path = export_glb(
        predictions,
        output_path,
        Path(args.views_dir),
        args.as_mesh,
        args.filter_black_bg,
        args.black_bg_threshold,
    )
    print(f"Saved GLB reconstruction to: {saved_path.resolve()}")

    if args.viz:
        rr.script_teardown(args)
        if getattr(args, "save", None):
            print(f"Saved Rerun recording to: {Path(args.save).expanduser().resolve()}")

if __name__ == "__main__":
    main()

'''
示例输入：
python map-anything/scripts/demo_多模态推理.py \
runs/dataset-house3k_bs-8_initv-3_pom-position_only_20251022-121242/images/step_000001/batch_000 \
--viz --serve True --grpc_port 9876 --web_port 9098
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
