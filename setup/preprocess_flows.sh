# DATA_DIR=/home/robin/Work/camera-planning/camera-style/data/MovingObjDataset
DATA_DIR=/home/robin/Work/camera-planning/camera-style/data/MotionSet/flows
ROOT_DIR=/home/robin/Work/camera-planning/camera-style
echo $ROOT_DIR

Structure dataset directory from raw Unity outputs:
python $ROOT_DIR/flow_encoder/src/datamodules/preprocessing/structure_dir.py $DATA_DIR

# Split clips into train and validation set:
python $ROOT_DIR/utils/dataset_split.py \
    $DATA_DIR/raw_frames $DATA_DIR/splits

# Export RGB Unity flows to pth files:
python $ROOT_DIR/flow_encoder/src/datamodules/preprocessing/unity_to_pth.py \
    $DATA_DIR/flow_frames/ $DATA_DIR/flow_unity_pth

# # Compute external optical flow (here [RAFT](https://github.com/princeton-vl/RAFT)):
python $ROOT_DIR/raw_features/scripts/compute_flows.py \
    $DATA_DIR/raw_frames $DATA_DIR/flow_raft_pk/ -m raft

# Export external pkl flow files to pth:
python $ROOT_DIR/flow_encoder/src/datamodules/preprocessing/pkl_to_pth.py \
    $DATA_DIR/flow_raft_pk $DATA_DIR/flow_raft_pth

python $ROOT_DIR/flow_encoder/src/datamodules/preprocessing/pth_to_rgb.py \
    $DATA_DIR/flow_raft_pth $DATA_DIR/flow_raft_png

python $ROOT_DIR/flow_encoder/src/datamodules/preprocessing/pth_to_rgb.py \
    $DATA_DIR/flow_unity_pth $DATA_DIR/flow_unity_png

Preprocess pth Unity and external flows:
python /home/robin/Work/camera-planning/camera-style/flow_encoder/src/datamodules/preprocessing/flow_transforms.py --max-module MAX_MODULE \
$DATA_DIR/flow_unity_pth $DATA_DIR/flow_unity_maxnorm_pth
# \
unity_max_module
python $ROOT_DIR/flow_encoder/src/datamodules/preprocessing/flow_transforms.py --unit-module \
    $DATA_DIR/flow_raft_pth $DATA_DIR/flow_raft_unit_pth

python $ROOT_DIR/flow_encoder/src/datamodules/preprocessing/flow_transforms.py --unit-module \
    $DATA_DIR/flow_unity_pth $DATA_DIR/flow_unity_unit_pth
    \
    raft_max_module