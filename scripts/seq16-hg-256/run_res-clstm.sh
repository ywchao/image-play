#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

th main.lua \
  -GPU $gpu_id \
  -expID seq16-hg-256-res-clstm \
  -nThreads 1 \
  -nEpochs 4 \
  -batchSize 3 \
  -testInt 6600 \
  -netType hg-256-res-clstm \
  -hgModel pose-hg-train/exp/penn_action_cropped/hg-256-ft/best_model.t7
