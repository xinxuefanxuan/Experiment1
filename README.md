# Experiment1 - Canonical UV Position Map (Step A)

这个仓库当前实现了你要求的第一步：

- 输出 `1024x1024x3`（可配置）的 **Canonical UV Position Map**。
- 每个有效 UV 像素存储 FLAME 网格上的 3D 坐标。
- 空白区域可写 `0`（默认）或指定固定背景值。

## 1. 代码结构

- `src/uv_position_map/canonical.py`
  - 核心逻辑（UV 光栅化 + barycentric 插值 + position map 生成）
- `tools/generate_canonical_uv_position_map.py`
  - 命令行入口
- `docs/uv_position_map_design.md`
  - 设计文档（之前的方案）

## 2. 你需要补充的依赖

目前核心只依赖：

- Python >= 3.10
- `numpy`

安装示例：

```bash
pip install numpy
```

> 如果你希望直接输出 `.png`，后续可加 `imageio` 或 `Pillow`。

## 3. 你需要准备并放置的输入文件

请先把你已有流程中的 mesh/uv 数据导成下面 4 个 `.npy` 文件（路径可自定义）：

- `data/vertices.npy`
  - shape: `(N, 3)`，float32/float64
  - 含义：FLAME 顶点坐标（本帧 canonical/object space）
- `data/faces.npy`
  - shape: `(M, 3)`，int32/int64
  - 含义：几何三角面索引（对应 `vertices`）
- `data/uv_vertices.npy`
  - shape: `(Nt, 2)`，float32/float64，范围建议 `[0, 1]`
  - 含义：UV 顶点坐标
- `data/uv_faces.npy`
  - shape: `(M, 3)`，int32/int64
  - 含义：UV 三角面索引（对应 `uv_vertices`）

> 关键要求：`faces.shape[0] == uv_faces.shape[0]`，并且第 `i` 个几何面与第 `i` 个 UV 面语义对应。

## 4. 运行方式

先把 `src` 加到 `PYTHONPATH`，然后运行：

```bash
PYTHONPATH=src python tools/generate_canonical_uv_position_map.py \
  --vertices data/vertices.npy \
  --faces data/faces.npy \
  --uv-vertices data/uv_vertices.npy \
  --uv-faces data/uv_faces.npy \
  --resolution 1024 \
  --background 0.0 \
  --out-npy outputs/uv_position_map.npy \
  --out-mask outputs/uv_valid_mask.npy \
  --out-pix2face outputs/uv_pix2face.npy \
  --out-bary outputs/uv_barycentric.npy \
  --out-vis outputs/uv_position_map_vis.npy
```

## 5. 输出说明

- `outputs/uv_position_map.npy`
  - shape `(H, W, 3)`，float32，核心结果
- `outputs/uv_valid_mask.npy`
  - shape `(H, W)`，bool，有效 UV 区域
- `outputs/uv_pix2face.npy`
  - shape `(H, W)`，int32，像素命中的 triangle id
- `outputs/uv_barycentric.npy`
  - shape `(H, W, 3)`，float32，重心坐标
- `outputs/uv_position_map_vis.npy`
  - shape `(H, W, 3)`，uint8，便于可视化检查

## 6. 如何对接你后续会补充的文件

你提到的这些文件：

- `flame_param`
- `flame2023.pkl`
- `transformer.json`
- `intrs.npy`
- `rgb.npy`
- `mask.npy`
- `landmark2d.npz`

在 Step A 里，**直接必需**的是：

1. 从 `flame_param + flame2023.pkl` 生成本帧 `vertices`（`N x 3`）；
2. 准备 FLAME 对应拓扑 `faces`；
3. 准备 UV 拓扑 `uv_vertices + uv_faces`。

也就是你只要先把上述 4 个 `.npy` 给到脚本，就能跑通第一步。

`intrs/rgb/mask/landmark/transformer` 会在你做“可见区域 UV map（Step B）”时更关键。
