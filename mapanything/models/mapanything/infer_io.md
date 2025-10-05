# MapAnything.infer 输入输出指南

## 概述
`MapAnything.infer` 封装了模型前向推理流程，自动完成输入校验、预处理、后处理，并通过 `torch.inference_mode()` 关闭梯度计算。调用者按视角传入数据，即可获得已经对齐的几何预测结果。

## 输入
- `views`: 视角字典列表，每个元素对应一个视角，必须共享相同的 batch 维度 `B`。
  - 必填字段：
    - `img`: 形状 `(B, 3, H, W)` 的归一化 RGB 图像。
    - `data_norm_type`: 字符串，表明 `img` 使用的归一化方式，必须与 `self.encoder.data_norm_type` 相同。
  - 标定信息（二选一或都不提供）：
    - `intrinsics`: `(B, 3, 3)` 的针孔相机内参矩阵，推理时会被转换为视线方向。
    - `ray_directions`: `(B, H, W, 3)` 的相机坐标系射线方向。
  - 可选深度信息：
    - `depth_z`: `(B, H, W, 1)` 的相机坐标系 Z 深度；若提供，必须同时给出 `intrinsics` 或 `ray_directions`。
  - 可选位姿信息（所有视角需保持同一种表示方式）：
    - `camera_poses`: `(B, 4, 4)` 的齐次位姿矩阵，或元组 `(quats, trans)`，其中 `quats` 为 `(B, 4)` 四元数，`trans` 为 `(B, 3)` 平移向量。
    - `is_metric_scale`: 布尔值或 `(B,)` 张量，指示提供的深度/位姿是否已为米制，缺省为 `True`。
  - 透传元数据（不会被修改）：
    - `instance`: 长度为 `B` 的字符串列表。
    - `idx`: 长度为 `B` 的整数列表。
    - `true_shape`: 长度为 `B` 的 `(H, W)` 元组列表，用于记录图像的真实尺寸。

- 控制参数：
  - `memory_efficient_inference`（默认 `False`）：为稠密预测头启用低显存模式，速度会变慢。
  - `use_amp`（默认 `True`）：启用自动混合精度。
  - `amp_dtype`（默认 `"bf16"`）：当 `use_amp=True` 时请求的精度，可选 `"fp16"`、`"bf16"`、`"fp32"`；若设备不支持 `bf16`，会回退到 `fp16`。
  - `apply_mask`（默认 `True`）：对输出应用非歧义掩膜。
  - `mask_edges`（默认 `True`）：根据法向和深度计算边缘掩膜并应用。
  - `edge_normal_threshold`（默认 `5.0`）：法向边缘检测的角度容差（度）。
  - `edge_depth_threshold`（默认 `0.03`）：深度边缘检测的相对容差。
  - `apply_confidence_mask`（默认 `False`）：基于置信度阈值过滤低置信像素。
  - `confidence_percentile`（默认 `10`）：`apply_confidence_mask=True` 时使用的分位数。
  
  - `ignore_calibration_inputs`（默认 `False`）：忽略 `intrinsics` 与 `ray_directions`。
  - `ignore_depth_inputs`（默认 `False`）：忽略深度观测。
  - `ignore_pose_inputs`（默认 `False`）：忽略位姿观测。
  - `ignore_depth_scale_inputs`（默认 `False`）：忽略深度尺度因子。
  - `ignore_pose_scale_inputs`（默认 `False`）：忽略位姿尺度因子。

### 校验规则
- 同一视角不可同时提供 `intrinsics` 与 `ray_directions`。
- 若提供 `depth_z`，必须配套 `intrinsics` 或 `ray_directions`。
- 只要任意视角包含 `camera_poses`，第 0 个视角也必须提供，以固定世界坐标系。
- `validate_input_views_for_inference` 会检查维度和 batch 大小，之后所有张量会被移动到模型所在设备（忽略 `instance`、`idx`、`true_shape`、`data_norm_type`）。

## 输出
返回列表与输入 `views` 长度一致，逐视角给出结果，每个字典仍保留 batch 维度 `B`：
- `img_no_norm`: `(B, H, W, 3)`，还原到原始数值范围的 RGB 图像。
- `pts3d`: `(B, H, W, 3)`，应用度量尺度后的世界坐标点云。
- `pts3d_cam`: `(B, H, W, 3)`，相机坐标系下的点云（射线方向乘以沿射线深度）。
- `ray_directions`: `(B, H, W, 3)`，相机坐标系射线方向。
- `intrinsics`: `(B, 3, 3)`，由预测射线方向恢复的针孔内参。
- `depth_along_ray`: `(B, H, W, 1)`，沿射线方向的深度。
- `depth_z`: `(B, H, W, 1)`，相机坐标系 Z 深度，由 `depth_along_ray` 与内参转换得到。
- `cam_trans`: `(B, 3)`，世界坐标系下的相机平移。
- `cam_quats`: `(B, 4)`，世界坐标系下的相机姿态四元数。
- `camera_poses`: `(B, 4, 4)`，由四元数和平移组成的齐次位姿矩阵。
- `metric_scaling_factor`: `(B,)`，应用在所有几何预测上的尺度因子。
- `mask`: `(B, H, W, 1)`，综合非歧义掩膜、可选边缘掩膜与置信度掩膜的最终掩膜。
- `non_ambiguous_mask`: `(B, H, W)`，基础非歧义掩膜。
- `non_ambiguous_mask_logits`: `(B, H, W)`，生成非歧义掩膜的原始 logits。
- `conf`: `(B, H, W)`，置信度图（仅当预测头输出置信度时存在）。

### 后处理说明
- `postprocess_model_outputs_for_inference` 负责反归一化图像、从射线方向重建内参、在点云与位姿之间转换，并根据参数应用掩膜。
- 通过 `ignore_*` 参数修改的几何输入配置会在推理结束后恢复。

## 错误行为
若输入存在冲突、缺少必填字段、batch 维度不一致或违反位姿约束，`infer` 会抛出 `ValueError`。
