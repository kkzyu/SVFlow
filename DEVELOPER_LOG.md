# SV-Flow Developer Log

## Important Files

| File | Description |
|------|-------------|
| `README/方案.md` | **UPDATED** — Simplified positioning, "Less is More" narrative |
| `/root/baselines/DrugFlow/code/DrugFlow-main/` | Base model (DrugFlow) source code |
| `/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt` | Pretrained DrugFlow checkpoint |
| `/root/autodl-tmp/data/test_sets/CrossDocked_test_set/` | 93 CrossDocked test pockets |

### Core Modules (simplified — 3 mechanisms only)

| File | Description |
|------|-------------|
| `svflow/kinematics.py` | Kinematic decoupling: v_int + v_CoM decomposition (§4.1) |
| `svflow/svgd.py` | SVGD kernel + isotropic repulsion baseline (§4.2) |
| `svflow/time_scheduler.py` | Time-annealed scheduling λ(t) (§4.3) |
| `svflow/sampler.py` | **SIMPLIFIED** — No TP/OP, just: decouple → SVGD → anneal → broadcast |
| `svflow/__init__.py` | Package exports (TP/OP removed from main API) |

### Retained for Ablation Reference Only

| File | Description |
|------|-------------|
| `svflow/tangent_projection.py` | Tangent plane projection (REMOVED from pipeline, kept for archival ablation) |
| `svflow/orthogonal_preservation.py` | Orthogonal preservation (REMOVED from pipeline, kept for archival ablation) |

### Scripts

| File | Description |
|------|-------------|
| `scripts/generate_svflow.py` | **SIMPLIFIED** — Only `--variant core` or `--variant isotropic` |
| `scripts/generate_baseline.py` | DrugFlow native independent sampling (no SVGD) |
| `scripts/generate_variants.py` | **UPDATED** — Now tests: core, isotropic, no_annealing |
| `scripts/evaluate.py` | Original evaluation script |
| `scripts/evaluate_full.py` | Comprehensive evaluation (QED, Tanimoto, spatial, physical, Vina) |
| `scripts/validate_physical.py` | Physical validity: clashes, bond anomalies, broken rings |
| `scripts/smoke_test.py` | Smoke test (CPU + optional GPU quick test) |
| `scripts/run_maxmin.py` | MaxMin diversity selection baseline (N=50 → K=10) |
| `scripts/drift_analysis.py` | **NEW** — Drift verification (Near/Mid/Far grouping) |
| `scripts/posthoc_translation.py` | **NEW** — Post-hoc translation + minimization baseline |
| `scripts/ablation_extras.py` | **NEW** — No-annealing ablation + N scalability |
| `scripts/plot_results.py` | Paper figure generation (Pareto, spatial, ablation, physical) |
| `scripts/run_all_experiments.sh` | **UPDATED** — Master orchestrator for full experiment pipeline |

## Key Design Decisions (Updated)

1. **SVGD operates only on CoM space (R^3)**: v_int is NEVER modified. Core innovation.

2. **No tangent plane projection, no orthogonal preservation**: Systematic ablation proved these
   constraints DESTROY spatial diversity (centroid variance drops to 7-44% of DrugFlow).
   Pure SVGD repulsion + time annealing is the optimal configuration. "Less is More."

3. **Late-onset scheduling**: λ(t)=0 for t > t_on=0.5, then parabolic ramp to λ_max at t=0.

4. **Batched dynamics**: All N trajectories processed in single _forward() call per ODE step.

5. **No self-conditioning in SV-Flow**: Bypass to avoid cross-trajectory contamination.

6. **Differentiation from Metadiffusion**: Full-atom repulsion (Metadiffusion) breaks bonds in small
   molecules; COM-only repulsion (SV-Flow) mathematically guarantees zero chemical bond impact.

## Core Algorithm (Final)
```
for each ODE step:
    1. v_θ = model.predict(x_t, t, pocket)         # batched forward
    2. v_CoM = per_molecule_mean(v_θ)               # kinematic decoupling
    3. c = predict_clean_COM(x_t)                   # predicted clean CoM
    4. ΔV = SVGD_repulsion(c, RBF, median_h)        # CoM-space only
    5. ΔV = λ(t) · ΔV                               # time annealing
    6. x_{t+dt} = x_t + dt · (v_int + v_CoM + ΔV)  # direct broadcast
    # NOTE: No projection, no orthogonalization.
```

## GPU Requirements
- **VRAM**: ~8-12 GB (N=10, ~30 atoms/mol)
- **RAM**: ~16 GB
- **RTX 5090 time**: ~35 min for 100 pockets (N=10, T=500)

## Experiment Status

| Experiment | Status | Script |
|-----------|--------|--------|
| SV-Flow Core 100-pocket gen | Ready to run | `generate_svflow.py --variant core` |
| DrugFlow baseline 100-pocket | Ready to run | `generate_baseline.py` |
| DrugFlow N=50 (for MaxMin) | Ready to run | `generate_baseline.py --n_samples 50` |
| MaxMin selection | Ready to run | `run_maxmin.py` |
| Post-hoc translation | **NEW** — Ready to run | `posthoc_translation.py` |
| Drift verification | **NEW** — Ready to run | `drift_analysis.py` |
| No-annealing ablation | **NEW** — Ready to run | `ablation_extras.py --mode no_annealing` |
| N scalability | **NEW** — Ready to run | `ablation_extras.py --mode scalability` |
| Physical validation | Ready to run | `validate_physical.py` |
| Full evaluation | Ready to run | `evaluate_full.py` |
| Paper figures | Ready to run | `plot_results.py` |

## Change Log

### 2026-06-03: Major Simplification
- Removed tangent_projection and orthogonal_preservation from sampler pipeline
- Updated scheme document to "Less is More" narrative
- Added post-hoc translation, drift analysis, no-annealing ablation, scalability experiments
- Cleaned up variant configuration (only Core + Isotropic remain)
- All script paths now reference `svflow/` directly (fixed import paths)
- Added `aa_atom_type_tensor` and `aa_atom_mask_tensor` to DrugFlow constants
- Fixed 9 documented bugs (time convention, self-conditioning, device mismatch, etc.)
