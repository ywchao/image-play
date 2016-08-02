#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID seq16-hg-single-clstm \
  -nEpochs 4 \
  -batchSize 1 \
  -testInt 20000 \
  -netType hg-single-clstm \
  -hgModel pose-hg-train/exp/penn_action_cropped/hg-single-ft/best_model.t7
