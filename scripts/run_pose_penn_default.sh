#!/bin/bash

if [ $# -eq 0 ]; then
  gpu_id=0
else
  gpu_id=$1
fi

th ./tools/run_pose_penn.lua \
  $gpu_id \
  default
