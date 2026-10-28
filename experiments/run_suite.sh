#!/usr/bin/env bash
# Exact commands behind results/. Each (benchmark, method) runs as its own single-thread
# process; JOBS of them in parallel.
# Usage: bash experiments/run_suite.sh [pmnist|pmnist_stress|pmnist_stress_val|pmnist_few|cifar10|cifar100|compose_val|compose|selection|cifar_full|cifar_curves|cifar_select|judge_pilot|judge]
# The v0.1 results were produced with the judge on training samples and 100 repair steps;
# V01 pins those settings (the current defaults are --feedback-source val --repair-steps 0).
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
SEEDS=${SEEDS:-"0 1 2"}
LOGS=results/logs
V01="--feedback-source train --repair-steps 100"
mkdir -p "$LOGS"

run_many() {  # $1 = common args, rest = methods
  local common="$1"; shift
  printf '%s\n' "$@" | xargs -P "$JOBS" -I{} sh -c \
    "python experiments/run_cl.py $common --methods {} --seeds $SEEDS --threads 1 > $LOGS/\$(echo '$common' | tr -cd '[:alnum:]_')_{}.log 2>&1"
}

case "${1:-pmnist}" in
  pmnist)       # 10 permuted-MNIST tasks, full data, 1 epoch; hyperparameters chosen on validation
    run_many "--benchmark pmnist $V01 --tasks 10 --ewc-lambda 100 --grace-eps ${GRACE_EPS:-1}" \
      finetune ewc er grace independent mntdp vcm_freeze vcm_lazy vcm_eager vcm_lazy_norepair \
      vcm_lazy_promo vcm_lazy_promo_keep vcm_freeze_promo
    python experiments/plot_cl.py --raw results/raw/pmnist --out results/pmnist --title "Permuted MNIST (10 tasks)" ;;
  pmnist_stress)  # aggressive in-place updates of shared units, so the judge has work to do
    run_many "--benchmark pmnist $V01 --tasks 10 --shared-lr-scale 1.0 --tag stress" \
      vcm_freeze vcm_lazy vcm_eager vcm_lazy_norepair vcm_nojudge vcm_lazy_promo vcm_lazy_promo_keep
    python experiments/plot_cl.py --raw results/raw/pmnist_stress --out results/pmnist_stress \
      --reference results/raw/pmnist --title "Permuted MNIST, shared units updated at full learning rate" ;;
  pmnist_stress_val)  # same, but the judge's feedback buffer is held-out validation data
    run_many "--benchmark pmnist --repair-steps 100 --tasks 10 --shared-lr-scale 1.0 --feedback-source val --tag stress_val" \
      vcm_lazy vcm_eager vcm_lazy_norepair vcm_lazy_promo
    python experiments/plot_cl.py --raw results/raw/pmnist_stress_val --out results/pmnist_stress_val \
      --reference results/raw/pmnist \
      --title "Permuted MNIST, full-rate shared updates, judge on held-out data" ;;
  pmnist_few)   # forward-transfer regime: task 0 full, later tasks 1000 samples
    run_many "--benchmark pmnist $V01 --tasks 10 --train-per-task 1000 --keep-first-full --epochs 5 --tag few" \
      finetune independent mntdp vcm_freeze vcm_lazy vcm_lazy_promo vcm_freeze_promo
    python experiments/plot_cl.py --raw results/raw/pmnist_few --out results/pmnist_few \
      --title "Permuted MNIST, 1000 samples per later task" ;;
  cifar10)
    run_many "--benchmark split_cifar10 $V01 --ewc-lambda ${CIFAR_EWC:-100} --grace-eps ${GRACE_EPS:-1}" \
      finetune ewc er grace independent mntdp vcm_freeze vcm_lazy vcm_eager vcm_lazy_promo
    python experiments/plot_cl.py --raw results/raw/split_cifar10 --out results/split_cifar10 \
      --title "Split CIFAR-10 (5 tasks, task-incremental)" ;;
  cifar100)
    run_many "--benchmark split_cifar100 $V01 --ewc-lambda ${CIFAR_EWC:-100} --grace-eps ${GRACE_EPS:-1}" \
      finetune ewc er grace independent mntdp vcm_freeze vcm_lazy vcm_eager vcm_lazy_promo
    python experiments/plot_cl.py --raw results/raw/split_cifar100 --out results/split_cifar100 \
      --title "Split CIFAR-100 (10 tasks, task-incremental)" ;;
  compose_val)  # selections on validation (seed 0 unless noted): step budget, probe length (2 seeds), EWC lambda
    SW=results/raw/sweeps
    for N in 250 1000; do for S in 250 500 1000; do
      python experiments/run_cl.py --benchmark mnist_compose --eval-split val --train-per-task $N --min-steps $S \
        --methods independent finetune vcm_freeze vcm_freeze_promo_comp --seeds 0 --threads 1 --tag n${N}_s$S --out $SW &
    done; done; wait
    for N in 100 250 1000; do for P in 50 100 200; do
      python experiments/run_cl.py --benchmark mnist_compose --eval-split val --train-per-task $N --min-steps 500 \
        --probe-steps $P --methods vcm_freeze vcm_freeze_promo_comp --seeds 0 1 --threads 1 --tag n${N}_p$P --out $SW &
    done; done; wait
    for L in 1 10 100 1000 10000 100000; do
      python experiments/run_cl.py --benchmark mnist_compose --eval-split val --train-per-task 1000 --min-steps 500 \
        --methods ewc --ewc-lambda $L --seeds 0 --threads 1 --tag ewc$L --out $SW &
    done; wait
    python experiments/compose_val.py ;;
  compose)      # read digits, then two-image tasks; later tasks get N samples each (5 seeds)
    SEEDS=${COMPOSE_SEEDS:-"0 1 2 3 4"}
    for N in ${SIZES:-100 250 1000 4000}; do
      run_many "--benchmark mnist_compose --train-per-task $N --min-steps 500 --probe-steps 100 --ewc-lambda ${COMPOSE_EWC:-100000} --tag n$N" \
        independent finetune ewc er mntdp vcm_freeze vcm_freeze_promo vcm_freeze_promo_comp vcm_freeze_promo_comp0 \
        vcm_lazy vcm_lazy_promo_comp vcm_lazy_promo_comp_keep
      run_many "--benchmark mnist_compose --train-per-task $N --min-steps 500 --probe-steps 50 --tag n${N}_p50" \
        vcm_freeze vcm_freeze_promo_comp
    done
    python experiments/compose_readers.py --seeds $SEEDS --threads "$JOBS"
    python experiments/compose_report.py --sizes ${SIZES:-100 250 1000 4000} ;;
  selection)    # candidate selection (v0.3): learning curves on validation, then learner variants on test
    for N in ${SIZES:-100 250 1000 4000}; do
      python experiments/run_cl.py --benchmark mnist_compose --train-per-task $N --min-steps 500 --selection full \
        --curve-every 25 --eval-split val --methods vcm_freeze_promo_comp --seeds 0 1 2 3 4 5 6 7 --threads 1 \
        --tag curves_n$N --out results/raw/selection > $LOGS/selection_curves_n$N.log 2>&1 &
    done; wait
    python experiments/selection_curves.py --raw results/raw/selection --out results/selection
    SEEDS=${COMPOSE_SEEDS:-"0 1 2 3 4"}
    for N in ${SIZES:-100 250 1000 4000}; do   # needs the probe-100 runs of `compose` for the comparison
      run_many "--benchmark mnist_compose --train-per-task $N --min-steps 500 --probe-steps 50 --selection halving --tag n${N}_halving" \
        vcm_freeze vcm_freeze_promo_comp vcm_lazy vcm_lazy_promo_comp
      run_many "--benchmark mnist_compose --train-per-task $N --min-steps 500 --selection full --tag n${N}_full" \
        vcm_freeze vcm_freeze_promo_comp
      run_many "--benchmark mnist_compose --train-per-task $N --min-steps 500 --probe-steps 100 --selection halving --halving-eta 10 --tag n${N}_probewarm" \
        vcm_freeze vcm_freeze_promo_comp
    done
    python experiments/selection_report.py --sizes ${SIZES:-100 250 1000 4000} ;;
  cifar_full)   # VCM freeze with every candidate trained fully: the same selection budget as MNTDP-D
    for B in split_cifar10 split_cifar100; do
      python experiments/run_cl.py --benchmark $B $V01 --selection full --tag full --methods vcm_freeze \
        --seeds $SEEDS --threads 2 > $LOGS/${B}_full_vcm_freeze.log 2>&1 &
    done; wait
    for B in split_cifar10 split_cifar100; do
      python experiments/plot_cl.py --raw results/raw/${B}_full --out results/${B}_full --reference results/raw/$B \
        --title "$B, VCM freeze with every candidate trained fully"
    done ;;
  cifar_curves)  # v0.5: cheaper selection on Split CIFAR. Learning curves on validation (20% of each pool
                 # held out, so the selection split keeps its usual size and the rest scores the choice)
    for B in split_cifar10 split_cifar100; do
      PV=$([ $B = split_cifar100 ] && echo 500 || echo 1000)
      printf '%s\n' 0 1 2 3 4 5 6 7 | xargs -P "$JOBS" -I{} sh -c \
        "python experiments/run_cl.py --benchmark $B $V01 --val-fraction 0.2 --probe-val $PV --selection full \
          --curve-every 25 --eval-split val --methods vcm_freeze --seeds {} --threads 1 --tag curves \
          > $LOGS/${B}_curves_s{}.log 2>&1"
    done
    python experiments/selection_cifar.py ;;
  cifar_select)  # v0.5, test: the two rules fixed on the curves (halving at the probe's cost; a 25-step probe,
                 # then the best 2 train fully), paired by seed with the runs of cifar10, cifar100 and cifar_full
    for B in split_cifar10 split_cifar100; do for M in vcm_freeze vcm_lazy; do for S in $SEEDS; do
      printf '%s\n' "$B $M $S halving" "$B $M $S finalists"
    done; done; done | xargs -P "$JOBS" -L 1 sh -c "python experiments/run_cl.py --benchmark \$0 $V01 \
      --probe-steps 25 --selection \$3 --finalists 2 --methods \$1 --seeds \$2 --threads 1 --tag \$3 \
      > $LOGS/\$0_\$3_\$1_s\$2.log 2>&1"
    python experiments/selection_cifar.py ;;
  judge_pilot)   # v0.6 falsification test, step 1: always-accept pilots (validation split only) whose
                 # judge statistics place each judge's thresholds (quantiles); writes results/judge/grid.tsv
    P=results/raw/judge_pilot
    COMMON="--tolerance 1.0 --judge-log --settle --eval-split val --methods vcm_lazy --seeds 0 --threads 1 --out $P"
    python experiments/run_cl.py --benchmark pmnist --tasks 10 --shared-lr-scale 1.0 --tag pilot_stress $COMMON &
    python experiments/run_cl.py --benchmark split_cifar100 --shared-lr-scale 1.0 --tag pilot_stress $COMMON &
    python experiments/run_cl.py --benchmark split_cifar100 --tag pilot $COMMON &
    wait
    python experiments/judge_study.py grid --pilot $P > results/judge/grid.tsv ;;
  judge)         # v0.6, step 2: every configuration in results/judge/grid.tsv on the test split
    while IFS=$'\t' read -r setting config judge flag value; do
      case $setting in
        pmnist_stress)   B="--benchmark pmnist --tasks 10 --shared-lr-scale 1.0" ;;
        cifar100_stress) B="--benchmark split_cifar100 --shared-lr-scale 1.0" ;;
        cifar100)        B="--benchmark split_cifar100" ;;
      esac
      for S in $SEEDS; do
        echo "$LOGS/judge_${setting}_${config}_s$S.log $B --judge $judge $flag $value --judge-log --settle" \
             "--methods vcm_lazy --seeds $S --threads 1 --tag $config --out results/raw/judge/$setting"
      done
    done < results/judge/grid.tsv | xargs -P "$JOBS" -L 1 sh -c 'log=$1; shift; python experiments/run_cl.py "$@" > "$log" 2>&1' _
    python experiments/judge_study.py report ;;
esac
