# demo_images_only_inference.py 使用说明

## 功能概述
该脚本用于在本地对一组图像执行 MapAnything 的纯图像推理流程，输出每个视角的深度、相机参数以及点云，并可选地：
- 通过 [Rerun](https://www.rerun.io/) 进行交互式可视化；
- 将结果导出为 GLB 格式的三维模型，方便在 Blender、MeshLab 等工具中查看。

## 环境准备
1. **安装依赖**
   ```bash
   # 推荐使用 README 中的标准环境
   conda create -n mapanything python=3.12 -y
   conda activate mapanything
   pip install -e .
   ```
   - 若需可视化，请确保安装了 `rerun`（`pip install rerun-sdk`）。
   - 首次运行会从 Hugging Face 下载模型权重，需要网络访问或提前将权重缓存到本机。

2. **硬件要求**
   - 优先使用支持 CUDA 的 GPU；若无 GPU，会自动退回 CPU，但推理时间会明显增加。
   - 如显存不足，可启用 `--memory_efficient_inference` 选项，以牺牲速度换取更低的显存占用。

## 输入数据
- 使用 `--image_folder` 指定包含待重建图像的文件夹。
- 支持常见的 RGB 图像格式（JPEG/PNG 等）。
- 相邻视角差异越大、数量越多，重建效果越好；建议至少 3 张以上的同一场景图像。

## 基本用法
```bash
python map-anything/scripts/demo_images_only_inference.py \
   --image_folder /mnt/sdb/chenmohan/VGGT-NBV/TEMP/view/render_output_volume_10/images \
   --viz --serve --web_port 9098 --connect False
  #  --save outputs/run.rrd \
  #  --save_glb --output_path outputs/scene.glb
```
运行过程将：
- 自动检测 GPU/CPU 并加载对应模型；
- 在终端输出加载图像数量与推理进度；
- 若指定 `--save_glb`，在 `--output_path` 路径处生成 GLB 模型文件。

## 常用参数说明
- `--image_folder` *(必填)*：输入图像文件夹路径。
- `--memory_efficient_inference`：启用内存优化模式（速度较慢，显存占用更低）。
- `--apache`：改用 Apache 2.0 许可的模型权重 `facebook/map-anything-apache`。
- `--model_path`：指定本地权重路径或自定义 Hugging Face 模型 ID，优先级高于 `--apache`。
- `--hf_endpoint`：自定义 Hugging Face 镜像地址，例如 `https://hf-mirror.com`，避免跨境下载瓶颈。
- `--hf_cache_dir`：指定 Hugging Face 缓存目录，建议使用高速磁盘并在多次运行时复用已缓存权重。
- `--viz`：启动 Rerun 可视化窗口，实时查看点云、深度和相机姿态。
  - 结合 Rerun 额外参数：`--headless`（无界面运行）、`--connect`、`--serve`、`--url`、`--save`、`--stdout` 等，具体含义可通过 `python scripts/demo_images_only_inference.py --help` 查看。
- `--web_port`：当使用 `--serve` 启动网页端 Rerun Viewer 时指定监听端口，默认 9090。
- `--grpc_port`：Rerun gRPC 服务端口，默认 9876；远程访问时需同步转发该端口。
- `--save_glb`：根据推理结果生成 GLB 文件。
- `--output_path`：GLB 输出路径，默认为 `output.glb`；支持 `.glb/.gltf`。若需保存 Rerun 回放文件请改用 `--save <path>.rrd`。
- `--save`：输出 Rerun `.rrd` 录制文件，可与 `--connect/--serve` 同时使用；脚本结束后会提示保存路径。

## 可视化提示
- 使用 `--viz` 时，脚本会自动调用 Rerun Viewer；首次启动可能需要等待模型加载。
- 若在服务器上无显示环境，可配合 `--headless` 与 `--serve` 参数，将可视化结果通过浏览器访问。

## 常见问题
- **模型下载缓慢或失败**：检查网络连接，或提前执行 `huggingface-cli login` 获取下载权限。
  - 可通过参数 `--hf_endpoint https://hf-mirror.com` 使用镜像站点。
  - 预先在联网机器上下载模型后，通过 `--model_path /path/to/local/model` 指向本地路径。
  - 搭配 `--hf_cache_dir /fast/cache` 将缓存放在高速磁盘，加快重复运行装载速度。
- **Rerun 提示端口被占用**：使用 `--web_port 9100` 等更换端口，或停止占用默认 9090 端口的进程后重试。
- **Web Viewer 提示 `Failed to fetch`**：说明浏览器无法访问 gRPC 服务，确认在远程环境下同时映射/开放 `--grpc_port`（默认 9876）与 `--web_port`，或改用本地 `rerun` 客户端配合 `--connect`。
- **使用本地 Rerun Viewer 但无数据显示**：保持 Viewer 进程运行，并通过 `ssh -R 9876:localhost:9876` 等反向端口转发将远端 gRPC 端口映射回本地；脚本需携带 `--viz --connect --url rerun+http://127.0.0.1:9876/proxy`，完成推理后请确保在 Viewer 中重置 Blueprint 或从 Log Explorer 展开 `mapanything/...` 节点。
- **显存不足导致 OOM**：使用 `--memory_efficient_inference`，或减少输入图像数量。
- **Rerun Viewer 无法打开**：确认本地已安装 `rerun-sdk`，并检查防火墙/远程桌面配置。

如需更多推理和部署场景，请参考仓库根目录的 `README.md`。
