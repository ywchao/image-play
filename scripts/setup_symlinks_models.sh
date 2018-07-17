#!/bin/bash

echo "Setting up symlinks for precomputed implay models and prediction..."

dir_name=( "penn-crop" )

cd exp

for k in "${dir_name[@]}"; do
  if [ -L $k ]; then
    rm $k
  fi
  if [ -d $k ]; then
    echo "Failed: exp/$k already exists as a folder..."
    continue
  fi
  ln -s precomputed_implay_models_prediction/$k $k
done

cd ..

echo "Done."
