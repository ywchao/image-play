#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

CUDA_VISIBLE_DEVICES=$gpu_id th main.lua \
  -expID seq16-hg-256-res-clstm-res-64-fts3-hg-base16-proj-only-lr1.0e-5 \
  -nThreads 4 \
  -nEpochs 1 \
  -batchSize 3 \
  -currBase 16 \
  -testInt 6600 \
  -weightHMap 0 \
  -LR 1.0e-5 \
  -netType hg-256-res-clstm-res-64 \
  -hgs3Model skeleton2d3d/exp/h36m/hg-256-res-64-h36m-hgfix-w1/model_best.t7 \
  -evalOut hg
