# Experiment1 - Canonical UV Position Map (Step A)

你碰到的报错是：

```text
ValueError: OBJ face has no vt index, cannot build uv_faces. Bad token: 4
```

这说明：`flame2023_from_pkl.obj` 这个文件里只有 `f v1 v2 v3`，没有 `f v1/vt1 ...`，所以不能直接拿来做 UV 拓扑。

---

## 1) Step A 为什么必须 4 个输入

Step A 的数学本质是：
- 在 UV 三角形里算 barycentric；
- 用同样权重插值 3D 三角形顶点坐标。

因此最小输入必须是：
1. `vertices.npy` `(N,3)`
2. `faces.npy` `(M,3)`
3. `uv_vertices.npy` `(Nt,2)`
4. `uv_faces.npy` `(M,3)`

并且 `faces.shape[0] == uv_faces.shape[0]`（第 i 个 UV 面要对应第 i 个 3D 面）。

---

## 2) 针对你这个数据结构的可行方案

你现在有两类 OBJ：

- 模板 OBJ：`/home/yuanyuhao/VHAP/asset/flame/flame2023_from_pkl.obj`（可能无 vt）
- 帧级 OBJ：`/home/yuanyuhao/FastAvatar/data/sequence_EXP-1-head_part-2/flame/091/0000/flame.obj`（通常有纹理与 vt）

所以推荐：
- `--flame-obj` 用模板 OBJ（提供几何面 `faces`）
- `--uv-obj` 用帧级 flame.obj（提供 `uv_vertices/uv_faces`）

我已经把脚本升级支持这个模式。

---

## 3) 新脚本能力（已更新）

`tools/prepare_step_a_inputs_from_fastavatar.py` 现在支持：

1. 如果 `--flame-obj` 自带 vt，则直接使用。
2. 如果 `--flame-obj` 无 vt，则可通过 `--uv-obj` 读取 UV。
3. `vertices` 支持两种来源：

4. 当 `--flame-obj` 和 `--uv-obj` 面片数量不一致时，脚本会自动尝试按“几何面顶点索引”重映射 UV 面（适配你遇到的 `9976 vs 10144` 场景）。
   - 若可映射则继续；
   - 若仍有缺失面，会提示两份 OBJ 拓扑不兼容。
   - 直接 `--vertices-npy` 指定（推荐，最稳）
   - 或从 `--frame-param / --canonical-param` 的 npz 里自动尝试读取 `vertices/verts/v/mesh_vertices`

---

## 4) 你可以直接试这个命令

```bash
PYTHONPATH=src python tools/prepare_step_a_inputs_from_fastavatar.py \
  --flame-obj /home/yuanyuhao/VHAP/asset/flame/flame2023_from_pkl.obj \
  --uv-obj /home/yuanyuhao/FastAvatar/data/sequence_EXP-1-head_part-2/flame/091/0000/flame.obj \
  --frame-param /home/yuanyuhao/FastAvatar/data/nersemble_fastavatar/017/cam_220700191/EXP-1-head_part-1/flame_param/00000.npz \
  --canonical-param /home/yuanyuhao/FastAvatar/data/nersemble_fastavatar/017/cam_220700191/EXP-1-head_part-1/canonical_flame_param.npz \
  --out-dir data/step_a_input
```

如果你已有每帧顶点（例如在 `vertices` 目录里某个 `.npy`），建议直接：

```bash
PYTHONPATH=src python tools/prepare_step_a_inputs_from_fastavatar.py \
  --flame-obj /home/yuanyuhao/VHAP/asset/flame/flame2023_from_pkl.obj \
  --uv-obj /home/yuanyuhao/FastAvatar/data/sequence_EXP-1-head_part-2/flame/091/0000/flame.obj \
  --vertices-npy /home/yuanyuhao/FastAvatar/data/sequence_EXP-1-head_part-2/vertices/091/0000.npy \
  --out-dir data/step_a_input
```

> 第二条命令是最推荐的，因为不用猜 npz 的 key。

---

## 5) 生成 canonical UV position map

准备好 `data/step_a_input/*.npy` 后：

```bash
PYTHONPATH=src python tools/generate_canonical_uv_position_map.py \
  --vertices data/step_a_input/vertices.npy \
  --faces data/step_a_input/faces.npy \
  --uv-vertices data/step_a_input/uv_vertices.npy \
  --uv-faces data/step_a_input/uv_faces.npy \
  --resolution 1024 \
  --background 0.0
```

输出：
- `outputs/uv_position_map.npy`
- `outputs/uv_valid_mask.npy`
- `outputs/uv_pix2face.npy`
- `outputs/uv_barycentric.npy`
- `outputs/uv_position_map_vis.npy`

---

## 6) 依赖

- Python >= 3.10
- `numpy`

```bash
pip install numpy
```

---

## 7) 你现在这个状态（四个输入已生成）下一步直接跑

你已经拿到：

- `data/step_a_input/faces.npy`
- `data/step_a_input/uv_vertices.npy`
- `data/step_a_input/uv_faces.npy`
- `data/step_a_input/vertices.npy`

那么直接执行：

```bash
PYTHONPATH=src python tools/generate_canonical_uv_position_map.py \
  --vertices data/step_a_input/vertices.npy \
  --faces data/step_a_input/faces.npy \
  --uv-vertices data/step_a_input/uv_vertices.npy \
  --uv-faces data/step_a_input/uv_faces.npy \
  --resolution 1024 \
  --background 0.0 \
  --out-npy outputs/uv_position_map.npy \
  --out-mask outputs/uv_valid_mask.npy \
  --out-pix2face outputs/uv_pix2face.npy \
  --out-bary outputs/uv_barycentric.npy \
  --out-vis outputs/uv_position_map_vis.npy
```

如果你想确认结果维度：

```bash
python - <<'PY'
import numpy as np
x=np.load('outputs/uv_position_map.npy')
m=np.load('outputs/uv_valid_mask.npy')
print('uv_position_map:', x.shape, x.dtype)
print('uv_valid_mask:', m.shape, m.dtype, 'valid_ratio=', m.mean())
PY
```

---

## 8) 只输入一张图片的“一键系统”

你问的这个需求可以做成下面这个统一入口（已提供）：

- `tools/run_single_image_to_uv_map.py`

它会按顺序自动做三件事：
1. （可选）调用外部 FLAME 拟合器，从 `--image` 产出 `vertices.npy`；
2. 调 `prepare_step_a_inputs_from_fastavatar.py` 生成 `faces/uv_vertices/uv_faces/vertices`；
3. 调 `generate_canonical_uv_position_map.py` 输出 UV position map。

### A. 你已经有 vertices.npy（最简单）

```bash
PYTHONPATH=src python tools/run_single_image_to_uv_map.py \
  --image /path/to/frame.jpg \
  --flame-obj /home/yuanyuhao/VHAP/asset/flame/flame2020_from_pkl.obj \
  --uv-obj /home/yuanyuhao/FastAvatar/data/sequence_EXP-1-head_part-2/flame/091/0000/flame.obj \
  --vertices-npy /home/yuanyuhao/FastAvatar/data/sequence_EXP-1-head_part-2/vertices/091/image_220700191/image_220700191.npy \
  --step-a-dir data/step_a_input \
  --outputs-dir outputs
```

### B. 你只有图片，没有 vertices.npy

你需要准备一个外部估计器（DECA / EMOCA / 你自己的VHAP脚本），只要它能把 `vertices.npy` 写到指定路径即可。

示例（命令模板，按你的工程替换）：

```bash
PYTHONPATH=src python tools/run_single_image_to_uv_map.py \
  --image /path/to/frame.jpg \
  --flame-obj /home/yuanyuhao/VHAP/asset/flame/flame2020_from_pkl.obj \
  --uv-obj /home/yuanyuhao/FastAvatar/data/sequence_EXP-1-head_part-2/flame/091/0000/flame.obj \
  --estimator-cmd "python /path/to/your_estimator.py --input {image} --out-vertices {vertices_out}" \
  --estimator-vertices-out work/vertices.npy \
  --step-a-dir data/step_a_input \
  --outputs-dir outputs
```

> 这里的 `{image}`、`{work_dir}`、`{vertices_out}` 会由脚本自动替换。
