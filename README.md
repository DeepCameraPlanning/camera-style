# camera-style

## Installation

Python 3.9.7
Cuda 11.5

1. Setup a conda environment:
```
conda create --name camera-style python=3.9 pip
conda activate camera-style
```

2. To install required modules and load pre-trained models launch the following commands (`LIB_DIR` is a **global** path to a directory that will contain all downloaded external modules and must be outside of the current repository):
```
./setup/setup_env.sh
./setup/load_models.sh
LIB_DIR=/path/to/dir ./setup/load_lib.sh
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

# Dataset structure

```
data_unity/
├── flow_*flow-type*_pk/           # pickle external flows
├── flow_*flow-type*_pth/          # pth external flows
├── flow_*flow-type*_maxnorm_pth/  # max-norm preprocessed pth external flows
├── flow_*flow-type*_unit_pth/     # unit preprocessed pth external flows
├── flow_*flow-type*_videos/       # mp4 external flows
│
├── flow_unity_pth/                # pth unity flows
├── flow_unity_maxnorm_pth/        # max-norm preprocessed pth unity flows
├── flow_unity_unit_pth/           # unit preprocessed pth unity flows
├── flow_unity_videos/             # mp4 unity flows
│
├── raw_unity/
│   ├── archives/                  # archives files
│   ├── flow_frames/               # png unity flows
│   ├── raw_all/                   # unity depth, OF, raw frames
│   └── raw_frames/                # png unity raw frames
│
├── raw_videos/                    # mp4 raw frames
│
└── splits/                        # train/val/test splits
    ├── test.csv
    ├── train.csv
    └── val.csv
```

Run the following command to preprocess flows:
```
ROOT_DIR=/path/to/data_unity ./setup/preprocess_flows.sh
```