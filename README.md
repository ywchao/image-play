# image-play

## Setup

0. Download and extract the Penn Action dataset from `https://upenn.box.com/PennAction`.

0. Clone the image-play repository.
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive git@git.corp.adobe.com:chao/image-play.git
  ```

0. Create symlinks for the downloaded Penn Action dataset. `PENN_ROOT` should contain `frames`, `labels`, and `README`.
  ```Shell
  cd $IMPLAY_ROOT
  ln -s $PENN_ROOT ./external/Penn_Action
  ```

0. Crop the Penn Action dataset.
  ```Shell
  cd $IMPLAY_ROOT
  matlab -r "prepare_cropped_data"
  ```

#### Running pre-trained hourglass models using pose-hg-demo

0. Download pre-trained hourglass models.
  ```Shell
  cd $IMPLAY_ROOT
  ./data/scripts/fetch_hourglass_models.sh
  ```

  This will populate the `$IMPLAY_ROOT/data` folder with `umich-stacked-hourglass`.

#### Training new hourglass models using pose-hg-train

0. Download and extract the MPII Human Pose dataset from `http://human-pose.mpi-inf.mpg.de`.

0. Create symlinks for the downloaded MPII dataset. `MPII_ROOT` should contain `images`.
  ```Shell
  cd $IMPLAY_ROOT
  ln -s $MPII_ROOT/images ./pose-hg-train/data/mpii/images
  ```

#### Training hourglass-LSTM

0. Generate validation set and preprocess data.

  ```Shell
  cd $IMPLAY_ROOT
  matlab -r "generate_valid"
  python ./tools/preprocess.py
  ```
