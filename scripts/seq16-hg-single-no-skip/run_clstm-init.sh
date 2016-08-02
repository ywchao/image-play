#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID seq16-hg-single-no-skip-clstm-init \
  -nEpochs 4 \
  -netType hg-single-no-skip-clstm \
  -hgModel pose-hg-train/exp/penn_action_cropped/hg-single-no-skip-clstm-ft/best_model.t7
