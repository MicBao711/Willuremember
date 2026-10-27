#!/usr/bin/env bash
# Remaining experiments after the Permuted MNIST suites, in order. Hyperparameters are
# selected on validation splits (seed 0) before the test-set runs.
set -euo pipefail
cd "$(dirname "$0")/.."
SW=results/raw/sweeps
MD=results/sweeps/selected.md
mkdir -p "$SW" results/sweeps results/logs
STAGE=${STAGE:-1}   # resume from a later stage: STAGE=3 bash experiments/queue_rest.sh
[ "$STAGE" -le 1 ] && : > "$MD"

sweep() {  # $1 benchmark, $2 flag, $3 key, $4 method, rest = values
  local bench=$1 flag=$2 key=$3 method=$4; shift 4
  printf '%s\n' "$@" | xargs -P 4 -I{} sh -c \
    "python experiments/run_cl.py --benchmark $bench --eval-split val --methods $method $flag {} \
       --threads 1 --tag ${method}{} --out $SW > results/logs/sweep_${bench}_${method}{}.log 2>&1"
  python experiments/select_hparam.py $(printf "$SW/${bench}_${method}%s " "$@") --key "$key" --md "$MD"
}

# 1. Judge on held-out data under heavy in-place updates
if [ "$STAGE" -le 1 ]; then
bash experiments/run_suite.sh pmnist_stress_val
fi

# 2. GRACE-CL with the codebook rules of the official implementation: re-select eps, rerun
G=-
if [ "$STAGE" -le 2 ]; then
G=$(sweep pmnist --grace-eps grace_eps grace 0.5 1 2 4 | tail -n 1)
rm -f results/raw/pmnist/grace_s*.json
python experiments/run_cl.py --benchmark pmnist --tasks 10 --methods grace --grace-eps "$G" \
  --seeds 0 1 2 --threads 4 > results/logs/pmnist_grace_rerun.log 2>&1
python experiments/plot_cl.py --raw results/raw/pmnist --out results/pmnist --title "Permuted MNIST (10 tasks)"
fi

# 3. Split CIFAR-10: select EWC lambda and GRACE eps on validation, then the suite
L=$(sweep split_cifar10 --ewc-lambda ewc_lambda ewc 1 10 100 1000 | tail -n 1)
E=$(sweep split_cifar10 --grace-eps grace_eps grace 0.5 1 2 4 | tail -n 1)
CIFAR_EWC=$L GRACE_EPS=$E bash experiments/run_suite.sh cifar10

# 4. Split CIFAR-100 with the CIFAR-10 selections (no separate sweep, to save time)
CIFAR_EWC=$L GRACE_EPS=$E bash experiments/run_suite.sh cifar100
echo "QUEUE DONE (pmnist grace eps=$G, cifar ewc lambda=$L, cifar grace eps=$E)"
