# MapAnything `downstream_head` 架构解析

本文档详细说明 `MapAnything` 模型中 `self.downstream_head(...)` 的整体结构与实现细节。

## 1. 调用位置与输入
- `MapAnything.forward` 在调用 `self.downstream_head(...)` 前，会准备四类参数：`dense_head_inputs`、`scale_head_inputs`、`img_shape` 与 `memory_efficient_inference`（`map-anything/mapanything/models/mapanything/model.py:1623-1669`）。
- `dense_head_inputs` 的形态取决于 `pred_head_type`：
  - `"linear"`：单个 BCHW 张量，已拼接全部视角。
  - `"dpt"` 或 `"dpt+pose"`：包含四个 BCHW 特征图的列表（编码器输出及多视角 Transformer 各层特征）。

## 2. 密集预测分支
- `downstream_head` 先调用 `downstream_dense_head`（`model.py:1323-1359`），将输入送入配置好的密集头与适配器。
- **线性头路径**（`pred_head_type == "linear"`）：
  - 密集头为 `LinearFeature`（`uniception/models/prediction_heads/linear.py`），通过 1×1 卷积将通道扩展到 `output_dim × patch_size²`，再用 `pixel_shuffle` 恢复到像素分辨率。
  - 返回 `PixelTaskOutput(decoded_channels=...)`。
- **DPT 头路径**（`"dpt"` 或 `"dpt+pose"`）：
  - `self.dense_head` 是 `nn.Sequential(DPTFeature, DPTRegressionProcessor)`。
    - `DPTFeature` 利用 CroCo 风格的多尺度解码模块，将四层 Transformer 表示经过 1×1/反卷积与级联融合模块后上采样 8×（`dpt.py:58-197`）。
    - `DPTRegressionProcessor` 进一步内插到 `img_shape` 并用两层卷积映射到目标通道数（`dpt.py:214-279`）。
  - 同样返回 `PixelTaskOutput(decoded_channels=...)`。
- 密集适配器 `self.dense_adaptor` 会将通道转换成语义化输出（见 `_initialize_adaptors`，`model.py:430-608`）。常见类型：
  - `PointMapAdaptor` → `RegressionAdaptorOutput(value=xyz)`。
  - `PointMapWithConfidenceAdaptor` → `RegressionWithConfidenceAdaptorOutput(value=xyz, confidence=score)`。
  - `PointMapPlusRayDirectionsPlusDepthWithConfidenceAndMaskAdaptor` → 同时输出 `value/confidence/logits/mask`。
  - `RayDirectionsPlusDepthAdaptor`、`RayMapPlusDepthAdaptor` 等会拆分射线和深度分量。
- 适配器始终接受 BCHW 张量及目标尺寸 `(H, W)`，通过 `AdaptorInput` 传入。

## 3. 位姿分支（仅 `"dpt+pose"`）
- 当启用位姿头时，`downstream_head` 会将列表中最后一层特征（PoseHead 需要的分辨率）按批送入 `PoseHead`（`model.py:1416-1427`）。
- `PoseHead`（`pose_head.py`）结构：
  - 先用 1×1 卷积将输入通道压到 `4 × patch_size²`。
  - 堆叠若干残差 1×1 卷积块与 ReLU。
  - 自适应平均池化后接两层全连接 + ReLU。
  - 最后两个线性层分别预测平移（3）与四元数（4），拼成 7 维向量（`SummaryTaskOutput.decoded_channels`）。
- `CamTranslationPlusQuatsAdaptor` 将 7 通道拆分为平移与四元数，施加缩放/归一化规则并返回 `AdaptorOutput(value=[tx, ty, tz, qw, qx, qy, qz])`。
- 如果未启用位姿头，`pose_final_outputs` 为 `None`。

## 4. 尺度分支
- 尺度预测开销小，每次都会执行一次（`model.py:1481-1490`）。
- `MLPHead`（`mlp_head.py`）处理 Transformer 的尺度 token（形状 `B × C × 1`）：
  - 先换轴为 `B × 1 × C`，随后使用输入线性层、`num_mlp_layers` 个带 ReLU 的全连接块，再映射到目标维度。
  - 返回 `SummaryTaskOutput(decoded_channels=...)`。
- `ScaleAdaptor` 按配置（线性/平方/指数及裁剪）变换输出，最终得到 `(B, 1)` 的度量尺度因子。

## 5. 内存高效模式
- 当 `memory_efficient_inference=True` 时，密集头（及可选的位姿头）会按小批次运行（`model.py:1376-1459`）：
  - `_compute_adaptive_minibatch_size` 会依据当前 CUDA 空闲显存估算安全的批大小（上限约 680 MB/样本）。
  - 对每个小批复用相同流程，最后按字段拼接 dataclass 属性，恢复完整输出。
  - 每轮小批后清理 CUDA cache 以降低峰值显存。
- 若该标志为 False，则直接整批推理。

## 6. 输出约定
- `downstream_head` 返回 `(dense_final_outputs, pose_final_outputs, scale_final_output)` 三元组：
  - `dense_final_outputs`：与场景表示类型对应的适配器 dataclass。常见字段：
    - `value`：主输出（点云、射线、射线+深度等），形状 `(B_total, C, H, W)`。
    - 若配置中启用了 `confidence` 或 `mask`，会附加对应字段。
  - `pose_final_outputs`：位姿头关闭时为 `None`；开启时为 `AdaptorOutput`，其 `value` 形状约为 `(B_total, 7, 1, 1)`，后续会 reshape 成 `(B, 7)`。
  - `scale_final_output`：形状 `(B, 1)` 的尺度因子，稍后会按视角广播。
- 在 `forward` 中，这些输出会按视角拆分、乘以尺度因子，并封装成每视角的结果字典（`model.py:1671-1863`）。

## 7. 配置与扩展性提示
- `pred_head_config` 中的参数完全驱动密集头与适配器的构造，切换 `adaptor_type` 会自动调整输出字段。
- 若要新增场景表示：
  1. 实现一个新的适配器子类，返回标准的 adaptor output dataclass。
  2. 在 `_initialize_adaptors` 中注册它，并根据需要设置 `scene_rep_type`。
- 尺度预测保持独立，因此上层无论用何种几何表示都能统一缩放结果。

## 8. 汇总表

| 阶段 | 模块 | 关键输出 |
| --- | --- | --- |
| 密集头 | `LinearFeature` 或 `DPTFeature → DPTRegressionProcessor` | BCHW 特征图 |
| 密集适配器 | 例如 `PointMapAdaptor`、`RayMapPlusDepthAdaptor`、`PointMapPlusRayDirectionsPlusDepthWithConfidenceAdaptor` | 几何/置信度/mask 等 dataclass |
| 位姿头（可选） | `PoseHead → CamTranslationPlusQuatsAdaptor` | `(tx, ty, tz, qw, qx, qy, qz)` |
| 尺度头 | `MLPHead → ScaleAdaptor` | `(B, 1)` 尺度因子 |

以上内容即为 `self.downstream_head(...)` 输出分支的完整实现梳理。

