#!/usr/bin/env bash
# Exact commands behind results/. Each (benchmark, method) runs as its own single-thread
# process; JOBS of them in parallel. Usage: bash experiments/run_suite.sh [pmnist|pmnist_few|cifar10|cifar100]
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-4}
SEEDS=${SEEDS:-"0 1 2"}
LOGS=results/logs
mkdir -p "$LOGS"

run_many() {  # $1 = common args, rest = methods
  local common="$1"; shift
  printf '%s\n' "$@" | xargs -P "$JOBS" -I{} sh -c \
    "python experiments/run_cl.py $common --methods {} --seeds $SEEDS --threads 1 > $LOGS/\$(echo '$common' | tr -cd '[:alnum:]_')_{}.log 2>&1"
}

case "${1:-pmnist}" in
  pmnist)       # 10 permuted-MNIST tasks, full data, 1 epoch; hyperparameters chosen on validation
    run_many "--benchmark pmnist --tasks 10 --ewc-lambda 100 --grace-eps ${GRACE_EPS:-1}" \
      finetune ewc er grace independent vcm_freeze vcm_lazy vcm_eager vcm_lazy_norepair \
      vcm_lazy_promo vcm_lazy_promo_keep vcm_freeze_promo
    python experiments/plot_cl.py --raw results/raw/pmnist --out results/pmnist --title "Permuted MNIST (10 tasks)" ;;
  pmnist_stress)  # aggressive in-place updates of shared units, so the judge has work to do
    run_many "--benchmark pmnist --tasks 10 --shared-lr-scale 1.0 --tag stress" \
      vcm_freeze vcm_lazy vcm_eager vcm_lazy_norepair vcm_nojudge vcm_lazy_promo vcm_lazy_promo_keep
    python experiments/plot_cl.py --raw results/raw/pmnist_stress --out results/pmnist_stress \
      --reference results/raw/pmnist --title "Permuted MNIST, shared units updated at full learning rate" ;;
  pmnist_stress_val)  # same, but the judge's feedback buffer is held-out validation data
    run_many "--benchmark pmnist --tasks 10 --shared-lr-scale 1.0 --feedback-source val --tag stress_val" \
      vcm_lazy vcm_eager vcm_lazy_norepair vcm_lazy_promo
    python experiments/plot_cl.py --raw results/raw/pmnist_stress_val --out results/pmnist_stress_val \
      --reference results/raw/pmnist \
      --title "Permuted MNIST, full-rate shared updates, judge on held-out data" ;;
  pmnist_few)   # forward-transfer regime: task 0 full, later tasks 1000 samples
    run_many "--benchmark pmnist --tasks 10 --train-per-task 1000 --keep-first-full --epochs 5 --tag few" \
      finetune independent vcm_freeze vcm_lazy vcm_lazy_promo vcm_freeze_promo
    python experiments/plot_cl.py --raw results/raw/pmnist_few --out results/pmnist_few \
      --title "Permuted MNIST, 1000 samples per later task" ;;
  cifar10)
    run_many "--benchmark split_cifar10 --ewc-lambda ${CIFAR_EWC:-100} --grace-eps ${GRACE_EPS:-1}" \
      finetune ewc er grace independent vcm_freeze vcm_lazy vcm_eager vcm_lazy_promo
    python experiments/plot_cl.py --raw results/raw/split_cifar10 --out results/split_cifar10 \
      --title "Split CIFAR-10 (5 tasks, task-incremental)" ;;
  cifar100)
    run_many "--benchmark split_cifar100 --ewc-lambda ${CIFAR_EWC:-100} --grace-eps ${GRACE_EPS:-1}" \
      finetune ewc er grace independent vcm_freeze vcm_lazy vcm_eager vcm_lazy_promo
    python experiments/plot_cl.py --raw results/raw/split_cifar100 --out results/split_cifar100 \
      --title "Split CIFAR-100 (10 tasks, task-incremental)" ;;
esac
