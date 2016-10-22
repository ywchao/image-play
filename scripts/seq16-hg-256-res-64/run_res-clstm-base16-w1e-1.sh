#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID seq16-hg-256-res-clstm-res-64-base16-w1e-1 \
  -nThreads 4 \
  -nEpochs 1 \
  -batchSize 3 \
  -currBase 16 \
  -testInt 6600 \
  -weightProj 1e-1 \
  -netType hg-256-res-clstm-res-64 \
  -hgModel pose-hg-train/exp/penn_action_cropped/hg-256-ft/best_model.t7 \
  -s3Model skeleton2d3d/exp/h36m/res-64-t2/model_best.t7
