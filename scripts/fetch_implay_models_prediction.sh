#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )/exp"
mkdir -p $DIR && cd $DIR

FILE=precomputed_implay_models_prediction.tar.gz
URL=http://napoli18.eecs.umich.edu/public_html/data/cvpr_2017/precomputed_implay_models_prediction.tar.gz

if [ -f $FILE ]; then
  echo "File already exists..."
  exit 0
fi

echo "Downloading precomputed implay models (2.4G)..."

wget $URL -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done."
