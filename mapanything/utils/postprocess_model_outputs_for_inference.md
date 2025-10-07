# `postprocess_model_outputs_for_inference`

该文档说明 `mapanything/utils/inference.py` 中同名函数的用途及关键逻辑。

## 功能概览
该函数在模型对每个视角生成原始预测结果后，对其进行补充整理：保留所有原始张量，同时附加反归一化后的彩色图、深度信息、相机内参、位姿矩阵，以及可选的综合有效性掩膜，方便后续可视化或分析。

## 处理流程
1. 复制原始预测字典，确保下游仍能访问未经改动的结果。
2. 利用输入视图中的 `data_norm_type` 对图像做反归一化，得到 `img_no_norm`（通道顺序为 HWC，便于直接展示）。
3. 若包含相机坐标系下的点云 `pts3d_cam`，计算其 Z 轴深度并存为 `depth_z`。
4. 若预测中有光线方向 `ray_directions`，通过 `recover_pinhole_intrinsics_from_ray_directions` 恢复针孔相机内参矩阵。
5. 当同时具备相机平移 `cam_trans` 和四元数 `cam_quats` 时，组合成 4×4 的姿态矩阵并写入 `camera_poses`。
6. 按需构建综合掩膜 `mask`：
   - 先采用模型给出的 `non_ambiguous_mask`，
   - 如果开启置信度过滤（`apply_confidence_mask=True`），按给定分位数阈值剔除低置信度像素，
   - 若启用 `mask_edges`，再基于法线梯度和深度突变抑制边缘噪声。
7. 若生成了最终掩膜，将其应用到稠密几何输出（如 `pts3d`、`pts3d_cam`、`depth_along_ray`、`depth_z`）上，对无效区域置零，并把掩膜本身返回。

## 主要参数
- `raw_outputs`：模型按视角输出的预测字典列表。
- `input_views`：推理时的原始输入视图列表，为后处理提供图像、归一化信息以及相机属性。
- `apply_mask`：是否构造并应用综合掩膜，默认开启。
- `mask_edges`：在掩膜中加入边缘抑制，默认开启。
- `edge_normal_threshold`、`edge_depth_threshold`：边缘筛除时的法线和深度阈值。
- `apply_confidence_mask`、`confidence_percentile`：是否基于置信度分位数进一步过滤。

## 返回结果
返回与 `raw_outputs` 等长的列表。每个字典除了原始预测外，还包含上述派生张量，以及可选的 `mask`，用于标识稠密几何中可信的像素。

## 使用建议
- 需要看未经掩膜的稠密几何时，可将 `apply_mask=False`。
- 输入视图的 `data_norm_type` 必须正确，否则无法准确还原原始图像。
- 只要提供 `ray_directions` 就能还原内参；若已有外部标定，也可直接把内参写进预测字典。
- 边缘掩膜对噪声比较敏感，若定性可视化更重要，可调低阈值或直接关闭。
