# Matformer-CSP：基于 Matformer 骨干与 GMM 耦合流匹配的晶体结构预测

本目录把 CrystalFlow 的完整晶体生成实验框架移植到 Matformer 仓库，并新增：

- **Matformer 解码器**：用 `MatformerConv` 替换 `CSPNet` 的消息传递骨干，接口与 `CSPNet` 对齐。
- **GMM 耦合流匹配**：晶格 6D 贝叶斯 GMM + 分数坐标 wrapped GMM + 置换/平移群平均 + 模式耦合流匹配（Stage 0–4）。
- 原始 CrystalFlow 的 flow / diffusion / diffcsppp / energy / 条件生成 / 评估 / 按组分采样实验全部保留。

原 `matformer/` 属性预测代码未改动。

---

## 1. 目录与文件说明

```
data/                          # 完整数据集 CSV（mp_20 / mpts_52 / perov_5 / carbon_24）
matformer_csp/
├── common/                     # 从 CrystalFlow 移植的公共工具
│   ├── data_utils.py           # CIF 预处理、晶格矩阵/极分解、PBC 建图、scaler
│   ├── constants.py            # 组分 scaler 常量（COV 评估）
│   ├── utils.py                # PROJECT_ROOT、Hydra/Lightning 日志工具
│   ├── lattice_score.py        # 晶格打分（Opt 评估）
│   ├── ocp_graph_utils.py      # OCP 图工具
│   └── gemnet/                 # GemNet 骨干（可选 decoder）
├── pl_data/
│   ├── dataset.py              # CrystDataset；自动加载 *_gmm_resp.pt
│   └── datamodule.py           # CrystalFlow 原数据模块
├── pl_modules/
│   ├── flow.py                 # 原 CrystalFlow 流模型 + GMM 先验钩子
│   ├── flow_w_type.py          # DNG 流模型 + GMM 先验钩子
│   ├── flow_gmm.py             # 【新】GMM 耦合流（CSP）
│   ├── flow_gmm_w_type.py      # 【新】GMM 耦合流（DNG）
│   ├── matformer_cspnet.py     # 【新】Matformer 解码器（CSPNet 接口）
│   ├── matformer_conv.py       # 【新】MatformerConv（PyG-only 副本）
│   ├── rbf.py                  # 【新】RBFExpansion（Matformer 原实现）
│   ├── edge_utils.py           # 【新】从 CSPNet 抽出的按需建图
│   ├── cspnet.py               # 原 CSPNet；建图改为调用 edge_utils
│   ├── diffusion.py / diffusion_w_type.py
│   ├── diffcsppp_diffusion.py / diffcsppp_diffusion_w_type.py / diffcsppp_cspnet.py
│   ├── energy_model.py         # Energy 模型（guidance / 优化）
│   ├── gemnet.py / gnn.py / flow_cls_guide.py
│   ├── diff_utils.py / ode_solvers.py / symmetrize.py / lattice_utils.py
│   ├── conditioning.py / type_module.py / hungarian.py / time_scheduler.py / model.py
│   └── lattice/                # 晶格相关模块
├── gmm/                        # 【新】Stage 0–2
│   ├── padding.py              # pad/unpad、mask、wrap
│   ├── lattice_gmm.py          # 6D 贝叶斯 GMM（sklearn DP-BGMM）
│   ├── wrapped_gmm.py          # 对角 wrapped GMM（有限镜像 VB/EM）
│   ├── symmetry.py             # 同元素置换 + 平移 MC 群平均
│   └── fit_priors.py           # 拟合 GMM 先验、保存 *_gmm_resp.pt
├── scripts/
│   ├── evaluate.py             # CSP 测试集采样评估
│   ├── eval_utils.py           # 模型/数据加载、指标工具
│   ├── compute_metrics.py      # match rate / RMS / validity / COV
│   ├── generation.py           # DNG 生成评估
│   ├── sample.py               # 按化学式采样
│   ├── sample_ode.py / evaluate_ode.py / extract_gen.py
│   ├── optimization.py / readout_opt.py / compute_metrics_rec.py
│   ├── calypso_run_mlp.py / prepare_dataset.py / prepare_mp_default_relax.py
│   ├── make_debug_data.py      # 从 mp_20 截取几十条样本，生成 data/mp_20_debug
│   ├── check_env.py            # 环境自检（torch/CUDA/PyG 扩展版本与 ABI）
│   └── smoke.py                # 轻量 smoke test（无需 pymatgen/lightning）
├── conf/                       # Hydra 配置
│   ├── default.yaml
│   ├── data/*.yaml             # mp_20 / mpts_52 / perov_5 / carbon_24 / calypso_60 / ...
│   ├── model/*.yaml            # flow / flow_polar / flow_w_type / diffusion / ...
│   │   ├── flow_polar_matformer.yaml
│   │   ├── flow_polar_w_type_matformer.yaml
│   │   ├── flow_gmm.yaml / flow_gmm_matformer.yaml
│   │   ├── flow_gmm_w_type.yaml / flow_gmm_w_type_matformer.yaml
│   │   └── decoder/{cspnet, matformer, matformer_wide, diffcsppp_cspnet, gemnet}.yaml
│   ├── gmm/default.yaml        # GMM 先验拟合超参数
│   ├── optim/ train/ logging/ hydra/
├── prop_models/                # 预训练属性模型（COV/Opt 评估，约 22MB）
├── run.py                      # 训练入口
├── finetune.py                 # 微调入口
└── README.md
```

根目录环境文件：

- `environment-cpu.yml` / `environment-gpu.yml`：conda 的 CPU / NVIDIA CUDA 12.1 环境
- `requirements-csp.txt`：公共 pip 依赖
- `requirements-pyg-cpu.txt` / `requirements-pyg-gpu.txt`：PyG 扩展（分别匹配 torch 2.5.0 CPU / CUDA 12.1）
- `.gitignore`

---

## 2. 方法简述

### 2.1 原始 CrystalFlow 流匹配

```
x_T = 真实结构；x_0 ~ 先验；t ~ U(0,1)
x_t = x_0 + t (x_T - x_0)
(v_l, v_f) = v_θ(A, F_t, k_t, t)
L = cost_lattice ||v_l - (k_T-k_0)||² + cost_coord ||v_f - Δ_F||²
采样：x_{t+1/N} = x_t + v_t / N
```

### 2.2 GMM 耦合流匹配（Stage 0–4）

- **Stage 0**：从训练集读取 `(A_i, F_i, L_i)`；对晶格极分解得到 6D `k_i`；`F_i` pad 到数据集实际 `max_n`。
- **Stage 1**：
  - 晶格 `k`：截断 Dirichlet Process 贝叶斯 GMM（`sklearn` VB）。
  - 坐标 `vec(F)`：对角 wrapped GMM，有限周期镜像 `n ∈ {-1,0,1}`，变分/EM 近似；padding 维不参与。
- **Stage 2**：
  - 采样同元素置换 `P ∈ P_A`、平移 `τ ~ U([0,1)^3)`，计算群平均 `gbar_c^F(F|A)`。
  - 联合混合 `q0(k,F|A) = Σ_c π_c N(k; μ_c^k, Σ_c^k) gbar_c^F(F|A)`。
  - 用联合责任 `r_ic` 迭代更新 `π`、晶格参数、坐标参数（联合 EM）。
- **Stage 3**：
  - 目标晶体 `(A,F_1,k_1)`；`c ~ Categorical(r_1c)`。
  - `k_0 ~ N(μ_c^k, Σ_c^k)`；`y_0 ~ N(μ_c^F, diag(σ_c^F))`。
  - 对源、目标施加同一个 `(P, τ)`；线性插值得到 `k_t, F_t`；Matformer 预测速度；MSE 损失。
- **Stage 4**：
  - 输入组成 `A*`；`c ~ Categorical(π)`；从对应分量采样 `k_0, F_0`，再做一次随机 `(P, τ)`。
  - 用训练好的 Matformer 速度场做 Euler ODE 积分，输出 `(A*, F_1, L_1)`。

**实现近似（重要）**：
- 坐标 GMM 是 **pad 到 max_n + 对角协方差 + 有限镜像 VB/EM**，不是精确 3N 维 wrapped VB。
- 群平均是 **J 个蒙特卡洛样本**（默认 4），不是精确枚举。
- padding 维不参与密度；A 条件通过置换组 `P_A` 与平移体现。
- GMM 流要求 `lattice_polar=true`（6D）。

---

## 3. 模型参数

### 3.1 Matformer 解码器

| 参数 | 默认值 | 说明 |
|---|---|---|
| `hidden_dim` | 128 | Matformer 原 `node_features`；不照搬 CSPNet 的 512 |
| `num_layers` | 5 | Matformer 原 `conv_layers` |
| `heads` | 4 | 注意力头数 |
| `edge_rbf_bins` | 128 | 键长 RBF 展开维数 |
| `edge_style` | `knn` | `fc / knn / knn_cart / knn_frac` |
| `cutoff` | 7.0 | kNN 半径（来自 `model.radius`） |
| `max_neighbors` | 20 | 每原子最大邻居数 |
| `lattice_dim` | 6 | 极分解实验设为 6；矩阵晶格为 9 |
| `ip` | false | 极分解实验 false；矩阵晶格 true |
| `pred_type` | false | DNG 实验为 true |
| `smooth` | false | DNG/Energy 连续类型输入为 true |
| `pred_scalar` | false | Energy 模型为 true |
| `ln` | true | 最终 LayerNorm |

放大版 `conf/model/decoder/matformer_wide.yaml`：`hidden_dim=256, num_layers=6`。128/5 欠拟合时使用。

> Matformer 的 `MatformerConv` 每层含 `3H→3H` 消息更新，H=512 时参数量远大于 CSPNet；因此默认用论文规模 128/5。

### 3.2 流模型

| 参数 | 默认值 | 说明 |
|---|---|---|
| `time_dim` | 256 | 时间正弦嵌入维数 |
| `latent_dim` | 0 | 额外隐变量维数（通常 0） |
| `cost_coord` | 10.0 | 坐标损失权重 |
| `cost_lattice` | 1.0 | 晶格损失权重 |
| `cost_type` | 1.0/10.0 | 类型损失权重（DNG） |
| `lattice_polar` | true | 6D 极分解 |
| `lattice_polar_sigma` | 0.1 | 原流先验的高斯尺度（GMM 流不使用） |
| `guide_threshold` | 无 | 条件引导开关 |
| `max_neighbors` / `radius` | 20 / 7.0 | 建图参数 |

### 3.3 GMM 先验（`conf/gmm/default.yaml`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `K_max` | 10 | 截断分量上限 |
| `eps` | 0.01 | 有效模式阈值 |
| `alpha` | 0.1 | Dirichlet 先验浓度（<1 抑制无效分量） |
| `group_samples` | 4 | 群平均 MC 样本数 J |
| `joint_iters` | 20 | 联合 EM 迭代次数 |
| `wrapped_iters` | 30 | 坐标 GMM 单独拟合迭代次数 |
| `sigma_prior` | 1e-3 | 方差先验 |
| `min_sigma` / `max_sigma` | 1e-3 / 0.5 | 坐标标准差范围 |
| `max_fit_samples` | null | 只用子集拟合（仍为全部样本算 resp） |
| `max_atoms` | null | 覆盖从数据统计的 max_n |
| `device` | cuda | 无 GPU 自动回退 cpu |

### 3.4 优化与训练（`conf/optim`、`conf/train`）

| 参数 | 默认值 |
|---|---|
| 优化器 | Adam，`lr=1e-3`，`weight_decay=0` |
| 调度器 | ReduceLROnPlateau，`factor=0.6`，`patience=30` |
| batch size | 256 / 128 / 128（train/val/test） |
| max epochs | `${data.train_max_epochs}`（mp_20 为 3000） |
| accelerator | `auto`（有 GPU 自动用 GPU） |
| gradient clip | 0.5 |

---

## 4. 环境安装（CPU / GPU，uv 与 conda 两套）

### 4.0 硬件与版本矩阵

- 当前这台机器是 **Intel Arc 130T**，没有 NVIDIA GPU；CUDA 仅适用于 NVIDIA。本机建议用 CPU 环境做 smoke/debug，正式训练放到 NVIDIA 服务器/云。
- NVIDIA 用户先运行 `nvidia-smi` 确认驱动与 CUDA 版本；cu121 需要驱动约 >= 527.41（Windows）。
- macOS 使用 CPU 方案；MPS 未验证，PyG 扩展可能没有 macOS wheel。
- `environment-gpu.yml` / `requirements-pyg-gpu.txt` 面向 **NVIDIA CUDA 12.1**；需要驱动支持。若驱动只支持 CUDA 11.8，把 `cu121` 换成 `cu118`，并同步改 `requirements-pyg-gpu.txt` 的 URL（`torch-2.5.0+cu118.html`）。
- Intel Arc 可用 IPEX/XPU 尝试，但 PyG 扩展只提供 CPU/CUDA 版，本项目不建议；用 CPU 或 NVIDIA。
- 推荐版本矩阵：

| 组件 | CPU | GPU |
|---|---|---|
| Python | 3.10 | 3.10 |
| torch | 2.5.0+cpu | 2.5.0+cu121 |
| PyG 扩展 | torch-2.5.0+cpu | torch-2.5.0+cu121 |
| numpy | >=1.26,<2 | >=1.26,<2 |
| CUDA 驱动 | - | NVIDIA >= 527.41 (Windows) |

### 4.1 方式 A：uv

安装 uv（任选其一）：

```bash
pip install uv
# 或 Windows PowerShell:
# irm https://astral.sh/uv/install.ps1 | iex
```

#### CPU

```bash
uv venv --python 3.10 .venv-cpu
# Windows: .venv-cpu\Scripts\activate
source .venv-cpu/bin/activate

uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.0"
uv pip install -r requirements-csp.txt
uv pip install -r requirements-pyg-cpu.txt
```

#### GPU（NVIDIA CUDA 12.1）

```bash
uv venv --python 3.10 .venv-gpu
# Windows: .venv-gpu\Scripts\activate
source .venv-gpu/bin/activate

uv pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.5.0"
uv pip install -r requirements-csp.txt
uv pip install -r requirements-pyg-gpu.txt
```

> `--index-url` 只作用于当前这条命令，不会影响后续 `uv pip install`。
> PyG 扩展的 `--find-links` 必须与 torch 版本、CUDA 版本完全一致；换 torch 版本时同步改 `requirements-pyg-*.txt`。
> 如果 uv 版本较老不支持 requirements 文件里的 `--find-links`，可直接执行：
> `uv pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html`（GPU 把 `+cpu` 换成 `+cu121`）。

### 4.2 方式 B：conda

conda 只负责 Python + PyTorch，其余依赖用 pip 安装，避免 conda/pip 混装冲突。

#### CPU

```bash
conda env create -f environment-cpu.yml
conda activate matformer-csp-cpu
python -m pip install -r requirements-csp.txt
python -m pip install -r requirements-pyg-cpu.txt
```

#### GPU（NVIDIA CUDA 12.1）

```bash
conda env create -f environment-gpu.yml
conda activate matformer-csp-gpu
python -m pip install -r requirements-csp.txt
python -m pip install -r requirements-pyg-gpu.txt
```

### 4.3 环境自检

```bash
python matformer_csp/scripts/check_env.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip check
```

`check_env.py` 会打印 torch/CUDA/PyG 扩展版本，并实际调用一次 `torch_scatter`；如果出现 `DLL load failed`，说明 torch 与 PyG 扩展版本不匹配。

### 4.4 数据路径

完整数据集已放在仓库的 `data/` 下（`mp_20`、`mpts_52`、`perov_5`、`carbon_24`），默认直接使用，无需设置环境变量。若要改用其他目录：

```bash
# Linux/macOS
export MATFORMER_CSP_DATA_ROOT=/path/to/data
# Windows PowerShell
$env:MATFORMER_CSP_DATA_ROOT="D:/path/to/data"
```

默认值为 `<仓库根>/data`。数据目录需要可写（会生成 `*_ori.pt` 缓存和 `*_gmm_resp.pt`，这些派生文件已在 `.gitignore` 中忽略）。

---

## 5. 训练命令

> 所有命令都在**仓库根目录**执行。用 `hydra.run.dir=outputs/<name>` 固定输出目录，便于评估。

### 5.1 原始 CrystalFlow 流 + Matformer

```bash
# CSP MP-20
python matformer_csp/run.py data=mp_20 model=flow_polar_matformer \
  expname=csp-flow-matformer hydra.run.dir=outputs/csp-flow-matformer

# 也可以直接覆盖 decoder
python matformer_csp/run.py data=mp_20 model=flow_polar model/decoder=matformer \
  expname=csp-flow-matformer2 hydra.run.dir=outputs/csp-flow-matformer2

# DNG MP-20
python matformer_csp/run.py data=mp_20 model=flow_polar_w_type_matformer \
  +model.type_encoding=table expname=dng-flow-matformer hydra.run.dir=outputs/dng-flow-matformer

# 压力条件 CSP（CALYPSO-60，需要 calypso_60 数据）
python matformer_csp/run.py data=calypso_60 model=flow_polar_matformer \
  +model.guide_threshold=-1 expname=csp-calypso60 hydra.run.dir=outputs/csp-calypso60

# 能量条件 DNG（需要 mp_20_chgnet 数据）
python matformer_csp/run.py data=mp_20_chgnet model=flow_polar_w_type_matformer \
  +model.type_encoding=table +model.guide_threshold=-1 \
  expname=dng-energy hydra.run.dir=outputs/dng-energy

# 放大容量
python matformer_csp/run.py data=mp_20 model=flow_polar_matformer \
  model/decoder=matformer_wide expname=csp-wide hydra.run.dir=outputs/csp-wide
```

### 5.2 GMM 耦合流 + Matformer

```bash
# Stage 0–2：拟合 GMM 先验（先跑一次）
python matformer_csp/gmm/fit_priors.py data=mp_20 gmm.K_max=10 \
  gmm.out=cache/gmm/mp_20.pt gmm.device=cuda

# Stage 3：GMM 流训练（CSP）
python matformer_csp/run.py data=mp_20 model=flow_gmm_matformer \
  model.gmm_prior_path=cache/gmm/mp_20.pt \
  expname=csp-gmm-matformer hydra.run.dir=outputs/csp-gmm-matformer

# GMM 流训练（DNG）
python matformer_csp/run.py data=mp_20 model=flow_gmm_w_type_matformer \
  model.gmm_prior_path=cache/gmm/mp_20.pt +model.type_encoding=table \
  expname=dng-gmm-matformer hydra.run.dir=outputs/dng-gmm-matformer
```

> `gmm_prior_path` 也可以写成绝对路径；训练/评估时都需要能访问该文件。

### 5.3 CPU / 单卡 / 多卡

```bash
# CPU 调试
python matformer_csp/run.py data=mp_20 model=flow_polar_matformer \
  train.pl_trainer.accelerator=cpu train.pl_trainer.fast_dev_run=true

# 单卡 GPU
python matformer_csp/run.py data=mp_20 model=flow_polar_matformer \
  train.pl_trainer.accelerator=gpu train.pl_trainer.devices=1

# 多卡 GPU
python matformer_csp/run.py data=mp_20 model=flow_polar_matformer \
  train.pl_trainer.accelerator=gpu train.pl_trainer.devices=4 \
  +train.pl_trainer.strategy=ddp_find_unused_parameters_true
```

### 5.4 原始基线（不换解码器）

```bash
python matformer_csp/run.py data=mp_20 model=flow_polar expname=flow-polar
python matformer_csp/run.py data=mp_20 model=diffusion expname=diffusion
python matformer_csp/run.py data=mp_20 model=diffcsppp_diffusion expname=diffcsppp
```

---

## 6. 评估与生成命令

### 6.1 CSP 评估（单次）

```bash
python matformer_csp/scripts/evaluate.py \
  --model_path outputs/csp-flow-matformer \
  --ode-int-steps 100 --dataset mp_20 \
  --anneal_coords --anneal_slope 5 --label csp

python matformer_csp/scripts/compute_metrics.py \
  --root_path outputs/csp-flow-matformer \
  --tasks csp --gt_file data/mp_20/test.csv --label csp
```

### 6.2 CSP 评估（多次采样）

```bash
python matformer_csp/scripts/evaluate.py \
  --model_path outputs/csp-flow-matformer \
  --ode-int-steps 100 --dataset mp_20 --num_evals 20 \
  --anneal_coords --anneal_slope 5 --label csp20

python matformer_csp/scripts/compute_metrics.py \
  --root_path outputs/csp-flow-matformer \
  --tasks csp --gt_file data/mp_20/test.csv \
  --multi_eval --label csp20
```

### 6.3 提取生成结果

```bash
python matformer_csp/scripts/extract_gen.py eval_gen_csp.pt --task eval
```

### 6.4 DNG 生成评估

```bash
python matformer_csp/scripts/generation.py \
  --model_path outputs/dng-flow-matformer \
  --ode-int-steps 100 --dataset mp_20 --label dng

python matformer_csp/scripts/compute_metrics.py \
  --root_path outputs/dng-flow-matformer \
  --tasks gen --gt_file data/mp_20/test.csv --label dng
```

### 6.5 按化学式采样

```bash
python matformer_csp/scripts/sample.py \
  --model_path outputs/csp-flow-matformer \
  --ode-int-steps 100 --save_path outputs/samples/mp20 \
  --formula Li2MnO3 --num_evals 20 --anneal_coords --anneal_slope 5
```

---

## 7. 数据要求

| 数据集 | 状态 | 说明 |
|---|---|---|
| `mp_20` / `mpts_52` / `perov_5` / `carbon_24` | **已包含在 `data/`** | 直接可用 |
| `mp_20_debug` | 由脚本从 mp_20 截取 | `python matformer_csp/scripts/make_debug_data.py --source data/mp_20` |
| `calypso_60` / `mp_20_chgnet` / `*_sym` / `alex_*` | 需外部提供 | CSV 需包含 `material_id`、`cif`、目标属性列 |
| GMM 缓存 | 自动生成 | `cache/gmm/<dataset>.pt` 与各 split 的 `*_gmm_resp.pt` |

CSV 预处理会写入 `<split>_ori.pt`；首次运行较慢，之后直接读缓存。若旧缓存缺少 `lattice_polar`，删除对应 `*_ori.pt` 重新预处理。

---

## 8. 与 CrystalFlow 的差异

1. **骨干**：`MatformerCSPNet` 用 MatformerConv；默认 128/5/4，而非 CSPNet 的 512/6。
2. **GMM 流**：新增 `flow_gmm` / `flow_gmm_w_type`；原 flow/diffusion 实验保持可运行。
3. **坐标 GMM**：pad 到 max_n、对角协方差、有限镜像 VB/EM 近似；不是精确 3N wrapped VB。
4. **群平均**：置换 + 平移的 J 样本蒙特卡洛，不是精确枚举。
5. **生成阶段**：`sample()` 只用全局混合权重 `π`，不使用测试集的模式后验 `r_ic`，避免真值泄漏。
6. **diffcsppp**：接口不同，保留原 `CSPNet` 解码器。
7. **CSPNet 专属参数**：`num_freqs / rec_emb / num_millers / na_emb / use_angles / periodic_norm` 对 Matformer 无效，可原样传入但会被忽略。
8. **BatchNorm**：MatformerConv 含 BN，评估/采样前已补 `model.eval()`。
9. **路径**：`PROJECT_ROOT` 默认仓库根；数据集根用 `MATFORMER_CSP_DATA_ROOT`；Hydra 输出到 `outputs/`。
10. **日志**：wandb/swanlab 改为可选；未安装时自动跳过。

---

## 9. 快速自检

```bash
# 0) 环境自检：torch/CUDA/PyG 扩展版本与 ABI
python matformer_csp/scripts/check_env.py

# 1) 轻量 smoke：Matformer 解码器各分支 + GMM 数学（只需 torch + torch_geometric）
python matformer_csp/scripts/smoke.py

# 2) 配置组合检查（需要 hydra-core）
python - <<'PY'
import os; os.environ["PROJECT_ROOT"] = os.getcwd()
from hydra import initialize_config_dir, compose
with initialize_config_dir(config_dir=os.path.join(os.getcwd(), "matformer_csp", "conf"), version_base="1.3"):
    cfg = compose(config_name="default", overrides=["data=mp_20", "model=flow_polar_matformer"])
print(cfg.model._target_, cfg.model.decoder._target_)
PY

# 3) 生成极小调试数据（16/8/8 条）
python matformer_csp/scripts/make_debug_data.py --source data/mp_20

# 4) 原始流 + Matformer 的 fast_dev_run
python matformer_csp/run.py data=mp_20_debug model=flow_polar_matformer \
  model.decoder.edge_style=fc \
  train.pl_trainer.fast_dev_run=true train.pl_trainer.accelerator=cpu \
  expname=debug-flow hydra.run.dir=outputs/debug-flow

# 5) GMM 先验拟合（小样本）
python matformer_csp/gmm/fit_priors.py data=mp_20_debug \
  gmm.K_max=2 gmm.max_fit_samples=12 gmm.group_samples=2 \
  gmm.joint_iters=2 gmm.wrapped_iters=2 gmm.device=cpu \
  gmm.out=cache/gmm/mp_20_debug.pt

# 6) GMM 流 + Matformer 的 fast_dev_run
python matformer_csp/run.py data=mp_20_debug model=flow_gmm_matformer \
  model.decoder.edge_style=fc \
  model.gmm_prior_path=cache/gmm/mp_20_debug.pt \
  train.pl_trainer.fast_dev_run=true train.pl_trainer.accelerator=cpu \
  expname=debug-gmm hydra.run.dir=outputs/debug-gmm
```

> `edge_style=fc` 只用于最小依赖验证；正式训练用默认 `knn`（需要安装匹配版本的 `torch_scatter`）。

完整训练前的建议顺序：
1. 安装环境（第 4 节）。
2. `python matformer_csp/scripts/check_env.py`。
3. `python matformer_csp/scripts/smoke.py`。
4. 按上面 3–6 步跑通 debug 数据。
5. 正式训练 + 评估。
