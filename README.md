# camera-style

## Installation

Python 3.7.10
Cuda 11.2

1. Setup a conda environment:
```
conda create --name movie-style python=3.7 pip
conda activate movie-style
```

2. To install required modules and load pre-trained models launch the following commands (`LIB_DIR` is a **global** path to a directory that will contain all downloaded external modules and must be outside of the current repository):
```
./setup/setup_env.sh
./setup/load_models.sh
LIB_DIR=path/to/dir ./setup/load_lib.sh
python ./setup/confifg_test.py
```

**Note**: the version of `matplolib` can be conflicting with `sort` repository. To avoid such issue, the script `config/comment_lines.py` launched in `config/setup.sh` will comment lines 22 and 23 of `lib/sort/sort.py`:
```
import matplotlib
matplotlib.use('TkAgg')
```

## External modules

This project use the following external modules:
 - [DOPE](https://github.com/naver/dope)
 - [LCR-Net V2](https://github.com/naver/lcrnet-v2-improved-ppi)
 - [mannequinchallenge](https://github.com/google/mannequinchallenge)
 - [camera-control](https://github.com/jianghd1996/Camera-control)
 - [SORT](https://github.com/abewley/sort)
 - [depth-distillation](https://github.com/vinthony/depth-distillation)
 - [RAFT](https://github.com/princeton-vl/RAFT)
 - [motion-detection](https://github.com/robincourant/motion-detection)
