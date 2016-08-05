#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID seq16-hg-pf-clstm \
  -nThreads 1 \
  -nEpochs 4 \
  -batchSize 2 \
  -testInt 10000 \
  -netType hg-pf-clstm \
  -hgModel pose-hg-train/exp/penn_action_cropped/hg-256-ft/best_model.t7
