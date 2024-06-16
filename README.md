# Image-Play

Code for reproducing the results in the following paper:

This repo, together with [skeleton2d3d](https://github.com/ywchao/skeleton2d3d) and [pose-hg-train (branch `image-play`)](https://github.com/ywchao/pose-hg-train/tree/image-play), hold the code for reproducing the results in the following paper:

**Forecasting Human Dynamics from Static Images**  
Yu-Wei Chao, Jimei Yang, Brian Price, Scott Cohen, Jia Deng  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017  

Check out the [project site](https://umich-ywchao-image-play.github.io/) for more details.

### Role

- The main content of this repo is for implementing **training step 3** (Sec. 3.3), i.e. training the full 3D-PFNet (hourglass + RNNs + 3D skeleton converter).

- For the implementation of **training step 1**, please refer to submodule [pose-hg-train (branch `image-play`)](https://github.com/ywchao/pose-hg-train/tree/image-play).

- For the implementation of **training step 2**, please refer to submodule [skeleton2d3d](https://github.com/ywchao/skeleton2d3d).

### Citing Image-Play

Please cite Image-Play if it helps your research:

    @INPROCEEDINGS{chao:cvpr2017,
      author = {Yu-Wei Chao and Jimei Yang and Brian Price and Scott Cohen and Jia Deng},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      title = {Forecasting Human Dynamics from Static Images},
      year = {2017},
    }

### Clone the Repository

This repo contains three submodules (`pose-hg-train`, `skeleton2d3d`, and `Deep3DPose`), so make sure you clone with `--recursive`:

  ```Shell
  git clone --recursive https://github.com/ywchao/image-play.git
  ```

### Contents

1. [Download Pre-Computed Models and Prediction](#download-pre-computed-models-and-prediction)
2. [Dependencies](#dependencies)
3. [Setting Up Penn Action](#setting-up-penn-action)
4. [Training to Forecast 2D Pose](#training-to-forecast-2d-pose)
5. [Training to Forecast 3D Pose](#training-to-forecast-3d-pose)
6. [Comparison with NN Baselines](#comparison-with-nn-baselines)
7. [Evaluation](#evaluation)
8. [Human Character Rendering](#human-character-rendering)

## Download Pre-Computed Models and Prediction

If you just want to run prediction or evaluation, you can simply download the pre-computed models and prediction (2.4G) and skip the training sections.

  ```Shell
  ./scripts/fetch_implay_models_prediction.sh
  ./scripts/setup_symlinks_models.sh
  ```

This will populate the `exp` folder with `precomputed_implay_models_prediction` and set up a set of symlinks.

You can now [set up Penn Action](#setting-up-penn-action) and [run the evaluation demo](#evaluation) with the downloaded prediction. This will ensure exact reproduction of the paper's results.

## Dependencies

To proceed to the remaining content, make sure the following are installed.

- [Torch7](https://github.com/torch/distro)
    - We used [commit bd5e664](https://github.com/torch/distro/commit/bd5e664194953539e928546e987c615a481a8eee) (2016-10-17) with CUDA 8.0.27 RC and cuDNN v5.1 (cudnn-8.0-linux-x64-v5.1).
    - All our models were trained on a GeForce GTX TITAN X GPU.
- [matio-ffi](https://github.com/soumith/matio-ffi.torch)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)
- [MATLAB](https://www.mathworks.com/products/matlab.html)
- [Blender](https://www.blender.org/)
    - This is only required for [human character rendering](#human-character-rendering).
    - We used release [blender-2.78a-linux-glibc211-x86_64](http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2) (2016-10-26).

## Setting Up Penn Action

The Penn Action dataset is used for training and evaluation.

1. Download the [Penn Action dataset](https://upenn.box.com/PennAction) to `external`. `external` should contain `Penn_Action.tar.gz`. Extract the files:

    ```Shell
    tar zxvf external/Penn_Action.tar.gz -C external
    ```

    This will populate the `external` folder with a folder `Penn_Action` with `frames`, `labels`, `tools`, and `README`.

2. Preprocess Penn Action by cropping the images:

    ```Shell
    matlab -r "prepare_penn_crop; quit"
    ```

    This will populate the `data/penn-crop` folder with `frames` and `labels`.

3. Generate validation set and preprocess annotations:

    ```Shell
    matlab -r "generate_valid_penn; quit"
    python tools/preprocess.py
    ```

    This will populate the `data/penn-crop` folder with `valid_ind.txt`, `train.h5`, `val.h5`, and `test.h5`.

4. **Optional:** Visualize statistics:

    ```Shell
    matlab -r "vis_data_stats; quit"
    ```

    The output will be saved in `output/vis_dataset`.

5. **Optional:** Visualize annotations:

    ```Shell
    matlab -r "vis_data_anno; quit"
    ```

    The output will be saved in `output/vis_dataset`.

6. **Optional:** Visualize frame skipping. As mentioned in the paper (Sec 4.1), we generated training and evaluation sequences by skipping frames. The following MATLAB script visualizes a subset of the generated sequences after frame skipping:

    ```Shell
    matlab -r "vis_action_phase; quit"
    ```

    The output will be saved in `output/vis_action_phase`.

## Training to Forecast 2D Pose

We begin with training a minimal model (hourglass + RNNs) which does just 2D pose forecasting.

1. Before starting, make sure to remove the symlinks from the download section, if any:

    ```Shell
    find exp -type l -delete
    ```

2. Obtain a trained hourglass model. This is done with the submodule `pose-hg-train`. 

    **Option 1:** [Download pre-computed hourglass models (50M)](https://github.com/ywchao/pose-hg-train/tree/image-play#downloading-pre-computed-hourglass-models): **(recommended)**

    ```Shell
    cd pose-hg-train
    ./scripts/fetch_hg_models.sh
    ./scripts/setup_symlinks_models.sh
    cd ..
    ```

    This will populate the `pose-hg-train/exp` folder with `precomputed_hg_models` and set up a set of symlinks.

    **Option 2:** [Train your own models](https://github.com/ywchao/pose-hg-train/tree/image-play#training-your-own-models).

3. Start training:

    ```Shell
    ./scripts/penn-crop/hg-256-res-clstm.sh $GPU_ID
    ```

    The output will be saved in `exp/penn-crop/hg-256-res-clstm`.

4. **Optional:** Visualize training loss and accuracy:

    ```Shell
    matlab -r "exp_name = 'hg-256-res-clstm'; plot_loss_err_acc; quit"
    ```

    The output will be saved to `output/plot_hg-256-res-clstm.pdf`.

5. **Optional:** Visualize prediction on a subset of the test set:

    ```Shell
    matlab -r "vis_preds_2d; quit"
    ```

    The output will be saved in `output/vis_hg-256-res-clstm`.

## Training to Forecast 3D Pose

Now we train the full 3D-PFNet (hourglass + RNNs + 3D skeleton converter), which also converts each 2D pose into 3D.

1. Obtain a trained hourglass model if you have not (see the section above).

2. Obtain a trained 3d skeleton converter. This is done with the submodule `skeleton2d3d`. 

    **Option 1:** [Download pre-computed s2d3d models (108M)](https://github.com/ywchao/skeleton2d3d#download-pre-computed-models-and-prediction): **(recommended)**

    ```Shell
    cd skeleton2d3d
    ./scripts/fetch_s2d3d_models_prediction.sh
    ./scripts/setup_symlinks_models.sh
    cd ..
    ```

    This will populate the `skeleton2d3d/exp` folder with `precomputed_s2d3d_models_prediction` and set up a set of symlinks.

    **Option 2:** [Train your own models (on ground-truth heatmaps)](https://github.com/ywchao/skeleton2d3d#training-3d-skeleton-converter-on-ground-truth-heatmaps).

3. Start training:

    ```Shell
    ./scripts/penn-crop/hg-256-res-clstm-res-64.sh $GPU_ID
    ```

    The output will be saved in `exp/penn-crop/hg-256-res-clstm-res-64`.

4. **Optional:** Visualize training loss and accuracy:

    ```Shell
    matlab -r "exp_name = 'hg-256-res-clstm-res-64'; plot_loss_err_acc; quit"
    ```

    The output will be saved to `output/plot_hg-256-res-clstm-res-64.pdf`.

5. **Optional:** Visualize prediction on a subset of the test set. Here we leverage Human3.6M's 3D pose visualizing routine.

    First, download the Human3.6M dataset code:

    ```Shell
    cd skeleton2d3d
    ./h36m_utils/fetch_h36m_code.sh
    cd ..
    ```

    This will populate the `skeleton2d3d/h36m_utils` folder with `Release-v1.1`.

    Then run the visualization script:

    ```Shell
    matlab -r "vis_preds_3d; quit"
    ```

    If you run this for the first time, the script will ask you to set two paths. Set the data path to `skeleton2d3d/external/Human3.6M` and the config file directory to `skeleton2d3d/h36m_utils/Release-v1.1`. This will create a new file `H36M.conf` under `image-play`.

    The output will be saved in `output/vis_hg-256-res-clstm-res-64`.

## Comparison with NN Baselines

This demo reproduces the nearest neighbor (NN) baselines reported in the paper (Sec. 4.1).

1. Obtain a trained hourglass model if you have not (see the section above).

2. Run pose estimation on input images.

    ```Shell
    ./scripts/penn-crop/hg-256.sh $GPU_ID
    ```

    The output will be saved in `exp/penn-crop/hg-256`.

3. Run the NN baselines:

    ```Shell
    matlab -r "nn_run; quit"
    ```

    The output will be saved in `exp/penn-crop/nn-all-th09` and `exp/penn-crop/nn-oracle-th09`.

4. **Optional:** Visualize prediction on a subset of the test set:

    ```Shell
    matlab -r "nn_vis; quit"
    ```

    The output will be saved in `output/vis_nn-all-th09` and `output/vis_nn-oracle-th09`.

## Evaluation

This demo runs the MATLAB evaluation script and reproduces our results in the paper (Tab. 1 and Fig. 7). If you are using [pre-computed prediction](#download-pre-computed-models-and-prediction), and want to also evaluate the NN baselines, make sure to first run step 3 in the last section.

Compute Percentage of Correct Keypoints (PCK):

  ```Shell
  matlab -r "eval_run; quit"
  ```

This will print out the PCK values with threshold 0.05 (PCK@0.05) and also show the PCK curves.

## Human Character Rendering

Finally, we show how we rendered human characters from the forecasted 3D skeletal poses using the method developed by [Chen et al. [4]](http://irc.cs.sdu.edu.cn/Deep3DPose/). This relies on the submodule `Deep3DPose`.

1. Obtain forecasted 3D poses by either [downloading pre-computed prediction](#download-pre-computed-models-and-prediction) or [generating your own](#training-to-forecast-3d-pose).

2. Set Blender path. Edit the following line in `tools/render_scape.m`:

    ```Shell
    blender_path = '$BLENDER_PATH/blender-2.78a-linux-glibc211-x86_64/blender';
    ```

3. Run rendering. We provide demo for both rendering without and with textures.

    Render body shape without textures:

    ```Shell
    matlab -r "texture = 0; render_scape; quit"
    ```

    Render body shape with textures:

    ```Shell
    matlab -r "texture = 1; render_scape; quit"
    ```

    The output will be saved in `output/render_hg-256-res-clstm-res-64`.