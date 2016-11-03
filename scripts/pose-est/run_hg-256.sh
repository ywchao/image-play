#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID pose-est-hg-256 \
  -nThreads 4 \
  -nEpochs 0 \
  -netType hg-256 \
  -seqLength 1 \
  -hgModel pose-hg-train/exp/penn_action_cropped/hg-256-ft/best_model.t7
