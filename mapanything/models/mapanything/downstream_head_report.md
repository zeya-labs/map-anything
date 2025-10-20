# MapAnything `downstream_head` Architecture

This note documents how the downstream prediction heads are wired inside `MapAnything` and explains what happens inside `self.downstream_head(...)`.

## 1. Call Site and Inputs
- `MapAnything.forward` prepares four tensors/lists: image-space features (`dense_head_inputs`), the scale token (`scale_head_inputs`), the image resolution (`img_shape`), and the `memory_efficient_inference` flag before invoking `self.downstream_head(...)` (`model.py:1623-1669`).
- The content of `dense_head_inputs` depends on `pred_head_type`:
  - `"linear"` → a single BCHW tensor, already stacked across views.
  - `"dpt"` or `"dpt+pose"` → a list of BCHW tensors (encoder residual blocks, transformer intermediates, and final features) packed in canonical order.

## 2. Dense Prediction Branch
- `downstream_head` immediately delegates to `downstream_dense_head` (`model.py:1323-1359`), which standardises inputs for the configured dense head and adaptor.
- **Linear head path** (`pred_head_type == "linear"`):
  - Uses `LinearFeature` (`uniception/models/prediction_heads/linear.py`) — a 1×1 convolution that expands channels to `output_dim × patch_size²`, then `torch.nn.functional.pixel_shuffle` to lift tokens back to pixel resolution.
  - Output is wrapped as `PixelTaskOutput(decoded_channels=...)`.
- **DPT head path** (`"dpt"` or `"dpt+pose"`):
  - `self.dense_head` is a `nn.Sequential(DPTFeature, DPTRegressionProcessor)`.
    - `DPTFeature` performs CroCo-style multi-scale decoding: it processes four transformer layers via learned 1×1/transpose convolutions, successive fusion blocks, and upsamples by 8× (`dpt.py:58-197`).
    - `DPTRegressionProcessor` interpolates to the requested `img_shape` and projects to the configured output channel count with two convolutional stages (`dpt.py:214-279`).
  - Again returns `PixelTaskOutput(decoded_channels=...)`.
- The dense adaptor (`self.dense_adaptor`) converts raw channels into semantically meaningful tensors according to `pred_head_config["adaptor_type"]` (`model.py:430-608`). Examples:
  - `PointMapAdaptor` → `RegressionAdaptorOutput(value=xyz)` (BCHW).
  - `PointMapWithConfidenceAdaptor` → `RegressionWithConfidenceAdaptorOutput(value=xyz, confidence=score)`.
  - `PointMapPlusRayDirectionsPlusDepthWithConfidenceAndMaskAdaptor` → `RegressionWithConfidenceAndMaskAdaptorOutput` containing `value`, `confidence`, `logits`, and `mask`.
  - `RayDirectionsPlusDepthAdaptor`, `RayMapPlusDepthAdaptor`, etc. split channels into ray origins, directions, and depths as needed.
- The adaptor stage always receives BCHW features and the desired `(H, W)` resolution via `AdaptorInput`.

## 3. Pose Branch (only for `"dpt+pose"`)
- When the pose head is enabled, `downstream_head` feeds the last transformer feature map (before DPT decoding) to `PoseHead` in sync with dense prediction batches (`model.py:1416-1427`).
- `PoseHead` (`pose_head.py`) architecture:
  - 1×1 convolution projects encoder channels to `4 × patch_size²`, the hidden width used throughout the head.
  - A stack of residual 1×1 convolution blocks (ReLU activations) refines the tensor.
  - Global average pooling collapses spatial dimensions, followed by two fully-connected layers with ReLU.
  - Two linear heads regress translation (3) and quaternion components (4), concatenated into a 7D vector (`SummaryTaskOutput.decoded_channels`).
- `CamTranslationPlusQuatsAdaptor` splits those 7 channels, applies scale/non-linearity rules per component, normalises quaternions, and returns `AdaptorOutput(value=[tx, ty, tz, qw, qx, qy, qz])`.
- The pose adaptor output is only present when the pose head is active; otherwise `pose_final_outputs` is `None`.

## 4. Scale Branch
- Scale prediction is always executed once per `downstream_head` call because it is lightweight (`model.py:1481-1490`).
- `MLPHead` (`mlp_head.py`) consumes the transformer scale token (`B × C × 1`):
  - Permutes to `B × 1 × C`, applies an input linear projection, runs `num_mlp_layers` fully-connected blocks with ReLU, and projects back to the requested output dimension.
  - Returns `SummaryTaskOutput(decoded_channels=...)`.
- `ScaleAdaptor` handles exponentiation/squaring/clamping per configuration, producing a `(B, 1)` metric scaling factor (`AdaptorOutput.value.squeeze(-1)`).

## 5. Memory-Efficient Mode
- When `memory_efficient_inference=True`, dense (and optional pose) heads are evaluated in mini-batches (`model.py:1376-1459`):
  - `_compute_adaptive_minibatch_size` inspects current CUDA memory and selects a safe per-batch size (upper bound ≈ 680 MB per sample).
  - Each mini-batch reuses the same helper methods, and outputs are concatenated field-wise (by enumerating dataclass attributes) to rebuild full-size adaptor outputs.
  - CUDA cache is cleared between mini-batches to stabilise peak usage.
- Without this flag the heads run in a single pass.

## 6. Output Contract
- `downstream_head` returns a triple `(dense_final_outputs, pose_final_outputs, scale_final_output)`:
  - `dense_final_outputs`: adaptor dataclass aligned with the chosen scene representation. Common attributes include:
    - `value` → primary tensor (point map, ray map, ray directions + depth, etc.) shaped `(B_total, C, H, W)`.
    - Optional `confidence`, `mask`, and `logits` for configurations that request uncertainty or ambiguity outputs.
  - `pose_final_outputs`: either `None` or an `AdaptorOutput` with `value` `(B_total, 7, 1, 1)` that is later reshaped to `(B, 7)` per view.
  - `scale_final_output`: plain tensor `(B, 1)` broadcast across views during post-processing.
- In `forward`, these outputs are repartitioned per view, multiplied by the predicted scale, and packed into per-view dictionaries (`model.py:1671-1863`). Different `scene_rep_type` blocks decide how to assemble point clouds, rays, and masks.

## 7. Configuration and Extensibility Notes
- `pred_head_config` supplies the constructor arguments for all heads and adaptors. Changing adaptor types automatically adjusts available output keys because the adaptor classes advertise their required channel counts.
- Adding a new dense representation requires:
  1. Implementing a compatible adaptor subclass (returning one of the adaptor output dataclasses).
  2. Plugging it into `_initialize_adaptors` so `scene_rep_type` and downstream packaging know how to handle it.
- The design keeps scale prediction decoupled so downstream consumers can uniformly rescale geometry regardless of dense head choice.

## 8. Summary Table

| Stage | Module(s) | Key Output |
| --- | --- | --- |
| Dense head | `LinearFeature` *or* `DPTFeature → DPTRegressionProcessor` | BCHW latent map |
| Dense adaptor | e.g., `PointMapAdaptor`, `RayMapPlusDepthAdaptor`, `PointMapPlusRayDirectionsPlusDepthWithConfidenceAdaptor` | Geometry/masks/confidence dataclasses |
| Pose head (optional) | `PoseHead → CamTranslationPlusQuatsAdaptor` | `(tx, ty, tz, qw, qx, qy, qz)` |
| Scale head | `MLPHead → ScaleAdaptor` | `(B, 1)` metric scale |

Together, these components form the full downstream output branch triggered by `self.downstream_head(...)`.

