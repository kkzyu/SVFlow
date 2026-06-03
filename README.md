# SV-Flow: Lightweight Inference-Time Diversity Enhancement for SBDD

**Stein Variational Flow Matching** — 通过运动学解耦将SVGD互斥限制在质心空间，在保护化学完整性的前提下实现结合模式的空间多样性提升。

## 一句话概述

SV-Flow 将流匹配ODE的独立采样升级为耦合多粒子变分推断系统。核心包含三个轻量级机制：

| 机制 | 说明 |
|------|------|
| **运动学解耦** (§4.1) | 将速度场分解为 v_int（内部构象）+ v_CoM（质心平动），SVGD仅作用于v_CoM |
| **SVGD核互斥** (§4.2) | 在ℝ³质心空间通过RBF核+Stein算子最大化分布熵 |
| **时间退火调度** (§4.3) | 晚期介入（t ≤ 0.5），避免早期噪声态引入伪影 |

**关键发现："Less is More"** — 最初设计的切平面投影和正交动能保护经消融实验证明反而破坏空间多样性。纯SVGD互斥即是最优解。

## 快速开始

```bash
# 冒烟测试（CPU可用）
python scripts/smoke_test.py

# GPU快速端到端测试
python scripts/smoke_test.py --gpu --n_steps 10 --n_trajectories 2

# 完整生成（需要GPU，~35分钟/100 pockets）
python scripts/generate_svflow.py \
    --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt \
    --n_trajectories 10 --n_steps 500 \
    --output_dir ./svflow_samples

# DrugFlow基线生成
python scripts/generate_baseline.py \
    --checkpoint /root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt \
    --n_samples 10 --n_steps 500 \
    --output_dir ./drugflow_baseline

# 评估
python scripts/evaluate_full.py \
    --samples_dir ./svflow_samples \
    --output_prefix ./results/core

# 一键运行全部实验
bash scripts/run_all_experiments.sh
```

## 项目结构

```
SVFlow/
├── svflow/                     # 核心代码
│   ├── sampler.py              #   多轨迹SV-Flow采样编排
│   ├── kinematics.py           #   运动学解耦
│   ├── svgd.py                 #   SVGD核互斥 + 各向同性基线
│   ├── time_scheduler.py       #   时间退火调度 λ(t)
│   ├── tangent_projection.py   #   [归档] 切平面投影（消融证明破坏多样性）
│   └── orthogonal_preservation.py  # [归档] 正交动能保护（消融证明破坏多样性）
├── scripts/                    # 实验脚本
│   ├── generate_svflow.py      #   主生成脚本
│   ├── generate_baseline.py    #   DrugFlow基线生成
│   ├── generate_variants.py    #   变体批量运行
│   ├── evaluate_full.py        #   综合评估管线
│   ├── validate_physical.py    #   物理合法性验证
│   ├── run_maxmin.py           #   MaxMin后处理基线
│   ├── posthoc_translation.py  #   事后平移对比实验
│   ├── drift_analysis.py       #   漂移验证实验
│   ├── ablation_extras.py      #   无退火消融 + N扩展性
│   ├── plot_results.py         #   论文图表生成
│   ├── smoke_test.py           #   冒烟测试
│   └── run_all_experiments.sh  #   主实验编排器
└── README.md                   # 本文件
```

## 依赖

- PyTorch ≥ 2.6 + CUDA
- PyTorch Geometric
- PyTorch Lightning
- RDKit
- BioPython
- NumPy, SciPy, Pandas

基座模型：[DrugFlow](https://github.com/) — pretrained checkpoint (161 MB)

## 与Metadiffusion的对比

| 维度 | Metadiffusion | SV-Flow |
|------|--------------|---------|
| 互斥空间 | 全原子 ℝ³ᴺ | 质心 ℝ³ |
| 对化学键影响 | 有破坏风险 | **零影响（数学保证）** |
| 计算开销 | O(N²·N_atoms²) | O(N²·3) |
| 适用场景 | 蛋白质构象 | **小分子SBDD** |

## 引用

```bibtex
@article{svflow2026,
  title={SV-Flow: Lightweight Inference-Time Diversity Enhancement for
         Structure-Based Drug Design via Kinematic Decoupling},
  author={},
  journal={},
  year={2026}
}
```
