# MapAnything `forward` 全流程解析

## 记号约定
- `B`: 每个视角的 batch 大小。
- `V`: 视角数量 (`len(views)`)。
- `H_img × W_img`: 输入图像分辨率。
- `p`: 编码器 patch 大小 (`self.encoder.patch_size`)。
- `H_feat = H_img / p`, `W_feat = W_img / p`: 编码器/Transformer 的 patch 网格分辨率。
- `C_enc`: 图像编码器输出通道 (= Transformer 输入/输出通道)。

所有形状均以 `NCHW` 表示（batch、通道、高、宽），除非特别说明。

## 输入结构
- `views`: 长度为 `V` 的列表。每个元素至少包含：
  - `img`: `(B, 3, H_img, W_img)`，已按编码器要求归一化。
  - `data_norm_type`: `[encoder.data_norm_type]`。所有视角需一致。
  可选几何模态键：`ray_directions_cam (B, H_img, W_img, 3)`、`depth_along_ray (B, H_img, W_img, 1)`、`camera_pose_quats (B, 4)`、`camera_pose_trans (B, 3)`、`is_metric_scale (B, 1)` 等。
- `memory_efficient_inference`: `bool`，为 `True` 时稠密头会分块运行以控制显存。

假设所有视角共享相同的 `H_img × W_img` 与几何模态维度。

## 阶段 1：图像编码
1. 读取 `views[0]["img"]` 确认 `B、H_img、W_img`，记录视角数 `V`。
2. `_encode_n_views(views)`：将 `V` 个 `(B, 3, H_img, W_img)` 拼接为 `(B·V, 3, H_img, W_img)`，送入 `self.encoder`。
3. 编码器输出被按视角拆回，得到列表 `all_encoder_features_across_views_list`，长度 `V`，其中每个元素形状为 `(B, C_enc, H_feat, W_feat)`——这也是几何模态融合的起点。

## 阶段 2：几何模态编码与特征融合（`_encode_and_fuse_optional_geometric_inputs`）
这一阶段在 `forward` 函数中、图像编码 (`_encode_n_views`) 结束后立即执行，并放在 `torch.autocast("cuda", enabled=False)` 的上下文里运行，以避免 LayerNorm NaN。这里的输入是图像特征列表，输出仍然是按视角分组的特征列表。

### 2.1 特征准备
- 将上一步的列表按 batch 维拼接成 `all_encoder_features_across_views`，形状 `(B·V, C_enc, H_feat, W_feat)`。
- 依据配置采样三个 mask：
  - `per_sample_geometric_input_mask (B·V,)`：全局控制是否使用任何几何模态，后续每种模态的掩码都会与它按位与；若全局掩码为 False，则射线/深度/姿态都会被禁用。
  - `per_sample_ray_dirs_input_mask / per_sample_depth_input_mask / per_sample_cam_input_mask (B·V,)`：在全局掩码基础上额外随机子采样各模态的样本，生成最终掩码。

### 2.2 射线方向编码 `_encode_and_fuse_ray_dirs`
1. 对每个视角，先切片得到 `per_sample_ray_dirs_input_mask_for_curr_view`（形状 `(B,)`），它来自阶段 2.1 中构造的全局掩码 `per_sample_ray_dirs_input_mask`。
   - 若掩码中对应样本为 True 且该视角提供了 `ray_directions_cam`，就把这些样本的射线复制进缓存 `(B, H_img, W_img, 3)`。
   - 否则保持零张量，并把 `per_sample_ray_dirs_input_mask` 在当前视角的段落整体设为 False，确保后续步骤不会误用。
2. 拼接所有视角的缓存，得到 `(B·V, H_img, W_img, 3)`，并转置为卷积友好的 `(B·V, 3, H_img, W_img)`。
3. 将该张量包装成 `ViTEncoderNonImageInput`，送入 `self.ray_dirs_encoder`，输出 `(B·V, C_enc, H_feat, W_feat)`，与图像特征保持相同的空间/通道分辨率。
4. 把全局掩码 `per_sample_ray_dirs_input_mask` reshape 成 `(B·V, 1, 1, 1)` 与射线编码特征逐元素相乘，屏蔽掉掩码为 False 的样本，然后与 `all_encoder_features_across_views` 做逐元素加法完成融合。

### 2.3 深度编码 `_encode_and_fuse_depths`
1. 深度张量初始化为 `(B, H_img, W_img, 1)`；对有效样本：
   - 可选稀疏采样；
   - 归一化非零深度，记录缩放因子 `depth_norm_factor (B,)`；
   - 将结果写入深度栈，否则掩码置零。
2. 拼接为 `(B·V, H_img, W_img, 1)` → 施加 log 缩放 → 转置为 `(B·V, 1, H_img, W_img)`。
3. `self.depth_encoder` 输出 `(B·V, C_enc, H_feat, W_feat)`，再以掩码 `(B·V, 1, 1, 1)` 抑制。
4. 额外将 `depth_norm_factor` 拼接为 `(B·V,)`，经 `self.depth_scale_encoder` 产生 `(B·V, C_enc)`，再 unsqueeze 成 `(B·V, C_enc, 1, 1)` 加入特征。
5. 若 `is_metric_scale` 为 `False`，对应样本的尺度特征被清零。

### 2.4 姿态编码 `_encode_and_fuse_cam_quats_and_trans`
1. `_compute_pose_quats_and_trans_for_across_views_in_ref_view`：
   - 将所有姿态转换到参考视角 0 的坐标系。
   - 返回 `pose_quats_across_views (B·V, 4)`、`pose_trans_across_views (B·V, 3)` 以及更新后的掩码。
2. `self.cam_rot_encoder` 处理四元数 → `(B·V, C_enc)` → `(B·V, C_enc, 1, 1)`。
3. 平移向量：
   - 先 reshape 为 `(B, V, 3)` 并归一化以获得 `pose_trans_norm_factors (B,)`；
   - 复原为 `(B·V, 3)` 后输入 `self.cam_trans_encoder` → `(B·V, C_enc)`；
   - 同时将 `log(pose_trans_norm_factors)` 送入 `self.cam_trans_scale_encoder` → `(B·V, C_enc)`。
4. 将上述三份 `(B·V, C_enc)` 全部扩展为 `(B·V, C_enc, 1, 1)` 与特征求和；若样本不在掩码或没有度量尺度，同步归零。

### 2.5 LayerNorm 融合
- 将特征换轴为 `(B·V, H_feat, W_feat, C_enc)`，用 `fusion_norm_layer`（LayerNorm）沿最后一维标准化，再转回 `(B·V, C_enc, H_feat, W_feat)`。
- 最终按视角拆分成长度 `V` 的列表，每项 `(B, C_enc, H_feat, W_feat)`。这份列表作为后续 Transformer 的输入。

## 阶段 3：多视角 Transformer 信息共享
1. `scale_token`：`(C_enc,)` → `unsqueeze(0)` → `(1, C_enc)` → `unsqueeze(-1)` → `(1, C_enc, 1)` → `repeat(B, 1, 1)`，得到 `(B, C_enc, 1)`。
2. 构造 `MultiViewTransformerInput`：
   - `features`: 上一步的列表，元素 `(B, C_enc, H_feat, W_feat)`；
   - `additional_input_tokens`: `(B, C_enc, 1)`。
3. 调用 `self.info_sharing`：
   - 若 `info_sharing_return_type == "no_intermediate_features"`：返回 `MultiViewTransformerOutput`，其中 `features` 仍为长度 `V` 的列表，每项 `(B, C_enc, H_feat, W_feat)`；`additional_token_features` 为 `(B, C_enc, 1)`。
   - 若为 `"intermediate_features"`：额外返回若干 `MultiViewTransformerOutput`，用于 DPT 多尺度分支。布局与最终输出一致。

## 阶段 4：拼装下游预测头输入
- `pred_head_type == "linear"`：
  - `dense_head_inputs = torch.cat(final_info_sharing_multi_view_feat.features, dim=0)` → `(B·V, C_enc, H_feat, W_feat)`。
- `pred_head_type in {"dpt", "dpt+pose"}`：
  - 构建列表 `dense_head_inputs_list`。
  - 若 `self.use_encoder_features_for_dpt` 为真：按顺序追加
    1. 编码器输出 `(B·V, C_enc, H_feat, W_feat)`；
    2-3. 两个中间层 `(B·V, C_enc, H_feat, W_feat)`；
    4. 最终层 `(B·V, C_enc, H_feat, W_feat)`。
  - 否则使用 Transformer 返回的三个中间层 + 最终层（同形状）。
- 尺度分支：`scale_head_inputs = final_info_sharing_multi_view_feat.additional_token_features`，形状 `(B, C_enc, 1)`。

## 阶段 5：预测头执行与特征解码
整段再次禁用 AMP。

### 5.1 稠密预测头 (`self.dense_head` + `self.dense_adaptor`)
- `pred_head_type == "linear"`：
  1. `LinearFeature` 接收 `(B·V, C_enc, H_feat, W_feat)`；
  2. 先 `Conv2d` 输出 `(B·V, C_out·p², H_feat, W_feat)`；
  3. `pixel_shuffle` 还原到 `(B·V, C_out, H_img, W_img)`；
  4. 交由 `dense_adaptor`（与 `scene_rep_type` 对应）转换为结构化结果，比如 point map（三通道）、ray map（七通道）等。
- `pred_head_type in {"dpt", "dpt+pose"}`：
  - `DPTFeature` + `DPTRegressionProcessor` 逐级操作，最终 `dense_adaptor` 同样接收 `(B·V, C_out, H_img, W_img)` 并产出 `RegressionAdaptorOutput`，属性依 `scene_rep_type` 而定：
    - `value`: `(B·V, channels, H_img, W_img)`；
    - 可选 `confidence` / `mask` / `logits` 等附加张量，布局一致。

### 5.2 姿态预测头（仅 `dpt+pose`）
- 取 `dense_head_inputs[-1]`（Transformer 最后一层特征，形状 `(B·V, C_enc, H_feat, W_feat)`）。
- `PoseHead` 通过 1×1 卷积 + ResConv + 全局池化，输出 `(B·V, 7)`（3 个平移量 + 4 个四元数分量）。
- `pose_adaptor` 将其解释为 `(B·V, 7)`，分别执行平移与旋转的后处理，得到 `AdaptorOutput.value`，形状 `(B·V, 7)`（或拆分视角后 `(B, 7)`）。

### 5.3 尺度预测头
- `scale_head` 接收 `(B, C_enc, 1)`，输出 `decoded_channels` `(B, C_scale)`。
- `scale_adaptor` 将其映射到 `(B, 1, 1)`，最后 squeeze 得到 `(B, 1)` 的度量缩放因子，并在随后广播到空间维度。

### 5.4 `memory_efficient_inference=True` 的分块逻辑
- 通过 `_compute_adaptive_minibatch_size()` 动态估算 mini-batch。
- 稠密头与（必要时）姿态头在 mini-batch 上循环，输出沿 batch 维拼接回 `(B·V, …)`。
- 每个 mini-batch 结束后清理 CUDA cache 以腾出显存。

## 阶段 6：按 `scene_rep_type` 组装最终输出
所有场景类型都会先将 `(B·V, ·, H_img, W_img)` 按视角拆分成 `V` 份 `(B, ·, H_img, W_img)`，并与 `scale_final_output (B, 1)` 结合。常见类型与通道解释如下：

| `scene_rep_type` 前缀 | `dense_final_outputs.value` 通道布局 | 主要输出字段（逐视角，均为 `(B, H_img, W_img, ·)`） |
| --- | --- | --- |
| `pointmap` | `[3]` | `pts3d`（乘尺度后得到世界坐标）。
| `raymap+depth` | `[3(origin)+3(dir)+1(depth)]` | `ray_origins`、`ray_directions`、`depth_along_ray`、推导出的 `pts3d`。
| `raydirs+depth+pose` | `[3(dir)+1(depth)]` + 姿态 `(B, 7)` | 结合姿态解算世界/相机坐标点云 `pts3d / pts3d_cam`，并保留射线信息与相机位姿。
| `campointmap+pose` | `[3]`（相机系点云） + 姿态 | 通过姿态转换生成 `pts3d`、`pts3d_cam`。
| `pointmap+raydirs+depth+pose` | `[3(point)+3(dir)+1(depth)]` + 姿态 | 可同时读取世界点云、射线信息与相机姿态。

- 名称包含 `+confidence`：读取 `dense_final_outputs.confidence (B·V, 1, H_img, W_img)` → `(B, H_img, W_img)` 赋给 `conf`。
- 名称包含 `+mask`：阈值化 `dense_final_outputs.mask` （同形状）得到 `non_ambiguous_mask`，并附上 `logits`（未阈值化）以便训练。
- 姿态分支额外返回 `cam_trans (B, 3)`、`cam_quats (B, 4)`，并乘以尺度因子（平移乘 `(B,1)`，点云乘 `(B,1,1,1)`）。

## 阶段 7：封装返回值
最终返回长度 `V` 的列表 `res`，其中第 `i` 个字典包含：
- `pts3d`: `(B, H_img, W_img, 3)`，世界坐标点云（所有类型都至少提供）。
- `metric_scaling_factor`: `(B, 1)`，同一 batch 内共享。
- 依据 `scene_rep_type`，附加 `pts3d_cam`、`ray_origins`、`ray_directions`、`depth_along_ray`、`cam_trans`、`cam_quats`、`conf`、`non_ambiguous_mask` 等键。

## 核心辅助模块速览
- `_encode_n_views`：图像批拼接 → ViT 编码 → 列表 `(B, C_enc, H_feat, W_feat)`。
- `_encode_and_fuse_optional_geometric_inputs`：多模态编码器（射线/深度/姿态） → 与图像特征同形状的增量，然后做 LayerNorm。
- `MultiView*Transformer`：接受 `V` 份 `(B, C_enc, H_feat, W_feat)` 与尺度 token `(B, C_enc, 1)`，输出同步形状的跨视角特征，必要时附带中间层。
- `downstream_head`：协调稠密头、姿态头、尺度头，并在需要时执行 mini-batch 切分。
- 各类 `Adaptor`：将 `(B·V, channels, H_img, W_img)` 或 `(B·V, channels)` 的张量解码成点云、射线、姿态等结构化字段。

通过上述形状清晰的流水线，`forward` 在多视角输入上完成几何模态融合、跨视角注意力聚合，并产出可直接用于三维重建/位姿估计的多种场景表示。
