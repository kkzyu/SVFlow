#!/bin/bash
# ============================================================================
# SV-Flow Master Experiment Runner (Updated — Simplified Architecture)
#
# Orchestrates the complete experiment pipeline once GPU is available.
# All paths are configured for the RTX 5090 environment.
#
# New experiment structure:
#   Phase 1: Smoke test
#   Phase 2: Core generation (SV-Flow Core + DrugFlow baseline + DrugFlow N=50)
#   Phase 3: Evaluation + Post-hoc translation + Drift analysis
#   Phase 4: Supplementary ablations (no-annealing + scalability)
#   Phase 5: Figures and tables
#
# Usage:
#   bash scripts/run_all_experiments.sh              # Run all
#   bash scripts/run_all_experiments.sh 1            # Smoke test only
#   bash scripts/run_all_experiments.sh 2            # Generation only
#   bash scripts/run_all_experiments.sh 3            # Evaluation only
# ============================================================================

set -e

# ---- Configuration ----
CHECKPOINT="/root/autodl-tmp/checkpoints/DrugFlow/drugflow.ckpt"
TEST_DIR="/root/autodl-tmp/data/test_sets/CrossDocked_test_set"
DEVICE="cuda:0"
N_TRAJECTORIES=10
N_STEPS=500
MAX_POCKETS=""  # empty = all 93 pockets
SEED=42

# Output directories
OUTPUT_DIR="./output"
CORE_DIR="${OUTPUT_DIR}/svflow_core"
BASELINE_DIR="${OUTPUT_DIR}/drugflow_baseline"
N50_DIR="${OUTPUT_DIR}/drugflow_n50"
MAXMIN_DIR="${OUTPUT_DIR}/maxmin_selected"
POSTHOC_DIR="${OUTPUT_DIR}/posthoc_translation"
ABLATION_DIR="${OUTPUT_DIR}/ablation_study"
RESULTS_DIR="${OUTPUT_DIR}/results"
FIGURES_DIR="${OUTPUT_DIR}/figures"

PHASE="${1:-all}"

# ---- Helpers ----
log_section() {
    echo ""
    echo "============================================================================"
    echo "  $1"
    echo "============================================================================"
    echo ""
}

run_cmd() {
    echo "  → $*"
    python "$@"
}

# ---- Ensure auxiliary files ----
ensure_aux_files() {
    log_section "Ensuring auxiliary files"
    mkdir -p processed_crossdocked
    if [ ! -f processed_crossdocked/size_distribution.npy ]; then
        cp /root/baselines/DrugFlow/code/DrugFlow-main/src/default/size_distribution.npy \
           processed_crossdocked/ 2>/dev/null || true
        echo "  ✓ size_distribution.npy copied"
    else
        echo "  ✓ size_distribution.npy exists"
    fi
}

# ---- Phase 1: Smoke Test (~5 min) ----
run_phase1() {
    log_section "Phase 1: Smoke Test"
    run_cmd scripts/smoke_test.py \
        --gpu --checkpoint "$CHECKPOINT" \
        --n_steps 10 --n_trajectories 2
}

# ---- Phase 2: Core Generation (~2 hours on RTX 5090) ----
run_phase2() {
    local max_arg=""
    [ -n "$MAX_POCKETS" ] && max_arg="--max_pockets $MAX_POCKETS"

    # 2a. SV-Flow Core (the ONLY SV-Flow variant — simplified algorithm)
    log_section "Phase 2a: SV-Flow Core Generation"
    run_cmd scripts/generate_svflow.py \
        --checkpoint "$CHECKPOINT" --test_dir "$TEST_DIR" \
        --output_dir "$CORE_DIR" --variant core \
        --n_trajectories "$N_TRAJECTORIES" --n_steps "$N_STEPS" \
        --device "$DEVICE" --seed "$SEED" $max_arg

    # 2b. DrugFlow baseline (independent sampling, N=10)
    log_section "Phase 2b: DrugFlow Baseline (N=10)"
    run_cmd scripts/generate_baseline.py \
        --checkpoint "$CHECKPOINT" --test_dir "$TEST_DIR" \
        --output_dir "$BASELINE_DIR" \
        --n_samples "$N_TRAJECTORIES" --n_steps "$N_STEPS" \
        --device "$DEVICE" --seed "$SEED" $max_arg

    # 2c. DrugFlow N=50 (for MaxMin + post-hoc translation)
    log_section "Phase 2c: DrugFlow N=50 (for MaxMin & PostHoc)"
    run_cmd scripts/generate_baseline.py \
        --checkpoint "$CHECKPOINT" --test_dir "$TEST_DIR" \
        --output_dir "$N50_DIR" \
        --n_samples 50 --n_steps "$N_STEPS" \
        --device "$DEVICE" --seed "$SEED" $max_arg
}

# ---- Phase 3: Evaluation + PostHoc + Drift (~1 hour) ----
run_phase3() {
    mkdir -p "$RESULTS_DIR"

    # 3a. Evaluate SV-Flow Core
    log_section "Phase 3a: Evaluate SV-Flow Core"
    run_cmd scripts/evaluate_full.py \
        --samples_dir "$CORE_DIR" --label "core" \
        --output_prefix "${RESULTS_DIR}/core"

    # 3b. Evaluate DrugFlow baseline
    log_section "Phase 3b: Evaluate DrugFlow Baseline"
    run_cmd scripts/evaluate_full.py \
        --samples_dir "$BASELINE_DIR" --label "drugflow" \
        --output_prefix "${RESULTS_DIR}/drugflow"

    # 3c. MaxMin selection from N=50
    log_section "Phase 3c: MaxMin Selection (N=50 → K=10)"
    run_cmd scripts/run_maxmin.py \
        --samples_dir "$N50_DIR" --k 10 --output_dir "$MAXMIN_DIR"
    run_cmd scripts/evaluate_full.py \
        --samples_dir "$MAXMIN_DIR" --label "maxmin" \
        --output_prefix "${RESULTS_DIR}/maxmin"

    # 3d. Post-hoc translation baseline (KEY NEW EXPERIMENT)
    log_section "Phase 3d: Post-hoc Translation Baseline"
    run_cmd scripts/posthoc_translation.py \
        --drugflow_samples "$N50_DIR" \
        --target_distribution "$CORE_DIR" \
        --output_dir "$POSTHOC_DIR"
    run_cmd scripts/evaluate_full.py \
        --samples_dir "$POSTHOC_DIR" --label "posthoc" \
        --output_prefix "${RESULTS_DIR}/posthoc"

    # 3e. Drift verification (KEY NEW EXPERIMENT)
    log_section "Phase 3e: Drift Verification"
    run_cmd scripts/drift_analysis.py \
        --samples_dir "$BASELINE_DIR" \
        --output "${RESULTS_DIR}/drift_analysis.csv"

    # 3f. Physical validation for all methods
    log_section "Phase 3f: Physical Validation"
    for dir in "$CORE_DIR" "$BASELINE_DIR" "$MAXMIN_DIR" "$POSTHOC_DIR"; do
        if [ -d "$dir" ]; then
            local label=$(basename "$dir")
            run_cmd scripts/validate_physical.py \
                --samples_dir "$dir" \
                --output "${RESULTS_DIR}/${label}_physical.csv"
        fi
    done
}

# ---- Phase 4: Supplementary Ablations (~2 hours) ----
run_phase4() {
    log_section "Phase 4a: No-Annealing Ablation (t_on=0.5 vs t_on=1.0)"
    run_cmd scripts/ablation_extras.py \
        --checkpoint "$CHECKPOINT" --test_dir "$TEST_DIR" \
        --mode no_annealing --max_pockets 10 \
        --output_dir "$ABLATION_DIR" \
        --device "$DEVICE" --seed "$SEED"

    log_section "Phase 4b: N-Trajectory Scalability (N=2,4,8,16,32)"
    run_cmd scripts/ablation_extras.py \
        --checkpoint "$CHECKPOINT" --test_dir "$TEST_DIR" \
        --mode scalability --max_pockets 3 \
        --output_dir "$ABLATION_DIR" \
        --device "$DEVICE" --seed "$SEED"

    # 4c. Quick variant comparison (Core vs Isotropic on 3 pockets)
    log_section "Phase 4c: Repulsion Type Comparison"
    run_cmd scripts/generate_variants.py \
        --checkpoint "$CHECKPOINT" --test_dir "$TEST_DIR" \
        --variants "core,isotropic" --max_pockets 3 \
        --output_dir "$ABLATION_DIR" \
        --device "$DEVICE" --seed "$SEED"
}

# ---- Phase 5: Figures and Tables ----
run_phase5() {
    log_section "Phase 5: Paper Figures & Tables"
    mkdir -p "$FIGURES_DIR"

    run_cmd scripts/plot_results.py \
        --drugflow_csv "${RESULTS_DIR}/drugflow" \
        --core_csv "${RESULTS_DIR}/core" \
        --maxmin_csv "${RESULTS_DIR}/maxmin" \
        --output_dir "$FIGURES_DIR"

    echo "Figures saved to $FIGURES_DIR/"
}

# ---- Main ----
main() {
    ensure_aux_files

    case "$PHASE" in
        all)  run_phase1; run_phase2; run_phase3; run_phase4; run_phase5 ;;
        1)    run_phase1 ;;
        2)    run_phase2 ;;
        3)    run_phase3 ;;
        4)    run_phase4 ;;
        5)    run_phase5 ;;
        *)
            echo "Usage: $0 [phase]"
            echo "  1: Smoke test"
            echo "  2: Generation (Core + DrugFlow + N50)"
            echo "  3: Evaluation + PostHoc + Drift + Physical"
            echo "  4: Supplementary ablations"
            echo "  5: Figures & tables"
            echo "  (no arg): Run all phases"
            exit 1 ;;
    esac

    log_section "Done!"
}

main
