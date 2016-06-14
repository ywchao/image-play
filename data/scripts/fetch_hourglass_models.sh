#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
URL=http://www-personal.umich.edu/~alnewell/pose/umich-stacked-hourglass.zip
FILE=$DIR/umich-stacked-hourglass.zip
CHECKSUM=0ddc66c047f442b43e8c0616aa09ee8b

if [ -f "$FILE" ]; then
  echo "File already exists. Checking md5..."
  checksum=`md5sum $FILE | awk '{ print $1 }'`
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading finetuned CaffeNet models (186M)..."
wget $URL -P $DIR;

echo "Unzipping..."
unzip $FILE -d $DIR

echo "Done."
