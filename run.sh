#!/bin/bash
# run_targets.sh

targets=("5ht1a" "a2a")

for target in "${targets[@]}"; do
  echo "Running target: $target"
  python train_dnn.py \
    --target $target \
    --epochs 25 \
    --batch_size 32 \
    --splits 4 \
    --repeats 5
done
