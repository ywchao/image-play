#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID seq16-rnn-vae-hg-256-no-skip-res-clstm \
  -nThreads 4 \
  -nEpochs 4 \
  -batchSize 8 \
  -testInt 2500 \
  -netType rnn-vae-hg-256-no-skip-res-clstm \
  -numScales 1 \
  -hgModel pose-hg-train/exp/penn_action_cropped/hg-256-no-skip-ft/best_model.t7
