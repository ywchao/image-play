## Setup

0. Download trained binaries.
  ```Shell
  cd caffe
  ./scripts/download_model_binary.py models/bvlc_reference_caffenet
  ```

0. Create symlinks for the cropped Penn Action dataset.
  ```Shell
  ln -s $PENN_CROP_ROOT data/penn-crop
  ```