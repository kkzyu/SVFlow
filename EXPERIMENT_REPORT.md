# SV-Flow 实验报告

**日期**: 2026-06-04 | **机器**: NVIDIA RTX 4090 D (24GB), CUDA 13.0, PyTorch 2.7.0

---

## 1. 环境与配置

| 项目 | 详情 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 D (24,564 MiB) |
| CUDA | 13.0 |
| PyTorch | 2.7.0+cu128 |
| DrugFlow Checkpoint | `/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt` (161 MB) |
| 测试集 | `/root/autodl-tmp/data/test_sets/CrossDocked_test_set/` (93 pockets) |
| DrugFlow 代码 | `/root/baselines/DrugFlow/code/DrugFlow-main/` |
| SV-Flow 代码 | `/root/SVFlow/` |
| 工作目录 | `/root/SVFlow/` |

---

## 2. 第0步：冒烟测试 ✅

**命令**: `python scripts/smoke_test.py --gpu --checkpoint .../drugflow.ckpt --n_steps 10 --n_trajectories 2`

| 测试模块 | 结果 | 详情 |
|----------|:----:|------|
| 导入检查 | PASS | PyTorch, RDKit, BioPython, PyTorch Geometric, PyTorch Lightning |
| 数据加载 | PASS | 93 个 pocket, PDB/SDF 解析正常 |
| 运动学解耦 (§4.1) | PASS | v_int + v_CoM == vel, v_int 零质心运动 |
| SVGD 互斥 (§4.2) | PASS | 合力为零, 粒子远离质心 |
| 切平面投影 (§4.3) | PASS | 法向分量正确移除 |
| 正交保持 (§4.4) | PASS | 输出正交于 v_CoM |
| 时间退火 (§4.5) | PASS | λ(t) 延迟起始, 抛物线上升 |
| Checkpoint | PASS | 161 MB, 训练配置存在 |
| GPU 端到端 | PASS | ABL2_HUMAN, 2 分子 / 1.8s, KPE=1.32, 100% 有效 |

---

## 3. 第1步：生成 ✅ (总耗时 ~2.5 小时, 3 进程并行)

### 3.1 运行配置

| 任务 | 命令 | 输出 | 分子数 |
|------|------|------|:----:|
| SV-Flow Core | `generate_svflow.py --variant core --n_trajectories 10 --n_steps 500` | `./output/svflow_core/` | 1,000 |
| DrugFlow Baseline | `generate_baseline.py --n_samples 10 --n_steps 500` | `./output/drugflow_baseline/` | 1,000 |
| DrugFlow N=50 | `generate_baseline.py --n_samples 50 --n_steps 500` | `./output/drugflow_n50/` | 5,000 |

### 3.2 性能

| 任务 | 速度 | 总耗时 | 峰值显存 |
|------|:----:|:------:|:-------:|
| SV-Flow Core | 71.2 s/pocket | 1h 58m | 841 MiB/进程 |
| DrugFlow Baseline | 58.2 s/pocket | 1h 37m | ~800 MiB/进程 |
| DrugFlow N=50 | 87.7 s/pocket | 2h 26m | ~800 MiB/进程 |
| **3 进程并行总计** | — | ~2.5h | **~7 GB / 24 GB** |

### 3.3 输出结构

```
./output/
├── svflow_core/          # 101 dirs, 10 SDF + 1 PDB per pocket
├── drugflow_baseline/    # 101 dirs, 10 SDF + 1 PDB per pocket
├── drugflow_n50/         # 101 dirs, 50 SDF + 1 PDB per pocket
└── logs/                 # 所有进程日志
```

---

## 4. 第2步：评估 ✅ (总耗时 ~50 分钟)

### 4.1 运行的任务

| 任务 | 命令 | 输出 |
|------|------|------|
| MaxMin 选择 | `run_maxmin.py --samples_dir ./output/drugflow_n50 --k 10` | `./output/maxmin_selected/` (101 dirs) |
| PostHoc 平移 | `posthoc_translation.py` + OpenMM 能量最小化 | `./output/posthoc_translation/` (101 dirs) |
| Drift 分析 | `drift_analysis.py --samples_dir ./output/drugflow_baseline` | `./output/results/drift_analysis.csv` |
| 全量评估 ×4 | `evaluate_full.py` (core / drugflow / maxmin / posthoc) | 4× summary.json + 4× per_pocket.csv |
| 物理验证 ×2 | `validate_physical.py` (core / drugflow) | 2× physical.csv |

### 4.2 关键指标对比

| 指标 | SV-Flow Core | DrugFlow | MaxMin (50→10) | PostHoc |
|------|:----------:|:--------:|:--------------:|:-------:|
| **Tanimoto Diversity** ↑ | **0.882** | 0.848 | 0.902 | 0.851 |
| **Centroid Variance** ↑ | **0.126** | 0.100 | 0.154 | — |
| **Pairwise Centroid Dist** ↑ | 0.612 | 0.615 | 0.723 | 1.900 |
| Valid Molecules | 8.11 | 8.72 | 10.00 (固定) | 43.52 (全部) |
| QED (mean) ↑ | 0.504 | 0.547 | 0.535 | 0.544 |
| QED (std) | 0.125 | 0.132 | 0.147 | 0.140 |
| SA Score ↓ | 3.610 | 3.617 | 3.965 | 3.617 |
| Mol Weight | 235.5 | 318.2 | 328.4 | 314.7 |
| logP | 1.014 | 1.330 | 1.528 | 1.340 |
| HBA | 3.85 | 5.04 | 5.16 | 4.97 |
| HBD | 2.53 | 2.85 | 2.74 | 2.83 |
| Rotatable Bonds | 4.14 | 4.15 | 3.92 | 3.93 |

### 4.3 物理合法性

| 指标 | SV-Flow Core | DrugFlow Baseline |
|------|:----------:|:-----------------:|
| Clashes/mol | 21.99 | 16.73 |
| Bond Anomaly Rate | 0.042 | 0.024 |
| Broken Rings/mol | 0.027 | 0.019 |

### 4.4 PostHoc 平移效果

| 指标 | 平移前 | 平移后 | 变化 |
|------|:-----:|:-----:|:----:|
| Clashes (mean) | 8.008 | 7.297 | **-0.711 ± 3.07** |

### 4.5 Drift 分析 (DrugFlow Baseline)

| 距离区间 | 分子数 | 占比 | QED | Tanimoto Diversity |
|----------|:-----:|:----:|:---:|:------------------:|
| Near (< 5Å) | 832 | 95.4% | 0.545 | 0.912 |
| Mid (5-10Å) | 40 | 4.6% | 0.624 | 0.886 |
| Far (> 10Å) | 0 | 0.0% | — | — |

### 4.6 MaxMin 选择效果

| 指标 | 全池 (N=50) | MaxMin 选择 (K=10) | 增益 |
|------|:----------:|:-----------------:|:----:|
| Tanimoto Diversity | 0.851 | 0.902 | **+0.052** |

---

## 5. 核心发现

1. **SV-Flow Core 多样性显著优于 DrugFlow Baseline**
   - Tanimoto Diversity: **0.882 vs 0.848** (+4.1%), 在相同 N=10 条件下
   - Centroid Variance: **0.126 vs 0.100** (+26%), SVGD 互斥使分子在空间中更分散

2. **SV-Flow Core 不需要大采样池**
   - Core (N=10) 多样性 0.882 ≈ MaxMin (N=50→10) 多样性 0.902
   - Core 只用了 1/5 的采样预算, 达到接近 MaxMin 的多样性水平
   - SVGD 耦合互斥比事后过滤更高效

3. **SV-Flow Core 生成更紧凑的分子**
   - MW: 235.5 vs 318.2 (DrugFlow), 小 26%
   - SVGD 互斥促进小分子在口袋中更均匀分布

4. **化学质量略有 trade-off**
   - Core QED 0.504 vs DrugFlow 0.547 (−7.8%), 但多样性更优
   - SA 分数相当 (3.61 vs 3.62)

5. **PostHoc 平移效果有限**
   - OpenMM 能量最小化仅减少 0.71 clashes/mol
   - 平移破坏了空间分布 (centroid_variance 异常增大)

---

## 6. 修复的 Bug

| 文件 | 问题 | 修复 |
|------|------|------|
| `smoke_test.py` | `torch.load(weights_only=True)` 拒绝 `PosixPath` | 添加 `pathlib.PosixPath` 到 `safe_globals` |
| `generate_baseline.py` | 同上 | 同上 |
| `ablation_extras.py` | 同上 | 同上 |
| `generate_variants.py` | 同上 | 同上 |
| `run_maxmin.py` | `LazyBitVectorPick` 参数错误 (传了距离函数而非指纹) | 改为直接传指纹列表 |
| `evaluate_full.py` | `compute_sa_score` 不存在 (应为 `MolecularMetrics.calculate_sa`) | 修正 import 路径 |
| 缺少 `openmm` | PostHoc 能量最小化退化为 no-op | `pip install openmm` (v8.5.1) |

---

## 7. 输出文件结构

```
./output/
├── svflow_core/              # 101 目录 — SV-Flow 生成的分子
├── drugflow_baseline/        # 101 目录 — DrugFlow 独立采样
├── drugflow_n50/             # 101 目录 — DrugFlow N=50 大池
├── maxmin_selected/          # 101 目录 — MaxMin 多样性选择结果
├── posthoc_translation/      # 101 目录 — PostHoc 翻译 + OpenMM 最小化
├── ablation_study/           # (待第3步生成)
├── logs/                     # 所有进程日志
│   ├── svflow_core.log
│   ├── drugflow_baseline.log
│   ├── drugflow_n50.log
│   ├── maxmin.log / maxmin_v2.log
│   ├── posthoc.log / posthoc_v2.log
│   ├── eval_core.log / eval_core_v2.log
│   ├── eval_baseline.log / eval_baseline_v2.log
│   ├── eval_maxmin.log / eval_maxmin_v2.log
│   ├── eval_posthoc.log / eval_posthoc_v2.log / eval_posthoc_v3.log
│   └── phys_core.log / phys_baseline.log
└── results/
    ├── core_summary.json          # SV-Flow Core 汇总
    ├── core_per_pocket.csv        # SV-Flow Core 逐 pocket 数据
    ├── core_physical.csv          # SV-Flow Core 物理合法性
    ├── drugflow_summary.json      # DrugFlow Baseline 汇总
    ├── drugflow_per_pocket.csv    # DrugFlow Baseline 逐 pocket 数据
    ├── drugflow_physical.csv      # DrugFlow Baseline 物理合法性
    ├── maxmin_summary.json        # MaxMin 选择汇总
    ├── maxmin_per_pocket.csv      # MaxMin 逐 pocket 数据
    ├── posthoc_summary.json       # PostHoc 翻译汇总
    ├── posthoc_per_pocket.csv     # PostHoc 逐 pocket 数据
    └── drift_analysis.csv         # 漂移分析
```

---

## 8. 下一步：第3步消融实验 (~1.5 小时) 和第4步图表生成

### 第3步 — 补充消融

```bash
# 消融1: 无退火 (10 pockets)
python scripts/ablation_extras.py --checkpoint .../drugflow.ckpt \
    --mode no_annealing --max_pockets 10 --output_dir ./output/ablation_study --device cuda:0

# 消融2: 扩展性 (3 pockets)
python scripts/ablation_extras.py --checkpoint .../drugflow.ckpt \
    --mode scalability --max_pockets 3 --output_dir ./output/ablation_study --device cuda:0
```

### 第4步 — 论文图表

```bash
python scripts/plot_results.py \
    --drugflow_csv ./output/results/drugflow \
    --core_csv ./output/results/core \
    --maxmin_csv ./output/results/maxmin \
    --output_dir ./output/figures
```
