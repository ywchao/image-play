#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

th main.lua \
  -GPU $gpu_id \
  -expID seq2-hg-no-skip-clstm-init1 \
  -nEpochs 10 \
  -netType hg-single-no-skip-clstm \
  -hgModel pose-hg-train/exp/penn_action_cropped/hg-single-no-skip-clstm-ft/best_model.t7
