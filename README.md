# image-play

### Setup

0. Download and extract the Penn Action dataset from `https://upenn.box.com/PennAction`.

0. Clone the image-play repository.
  ```
  git clone https://git.corp.adobe.com/chao/image-play.git
  ```

0. Create symlinks for the downloaded Penn Action dataset. `PENN_ROOT` should contain `frames`, `labels`, and `README`.
  ```
  cd image-play
  ln -s $PENN_ROOT ./external/Penn_Action
  ```
