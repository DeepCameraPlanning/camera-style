# Structure dataset directory from raw Unity outputs:
python src/datamodules/preprocessing/structure_dir.py $ROOT_DIR/raw_unity/

# Split clips into train and validation set:
python src/datamodules/preprocessing/dataset_split.py \
    $ROOT_DIR/raw_unity/raw_frames $ROOT_DIR/splits

# Export RGB Unity flows to pth files:
python src/datamodules/preprocessing/unity_to_pth.py \
    $ROOT_DIR/raw_unity/flow_frames/ $ROOT_DIR/flow_unity_pth

# Compute external optical flow (here [RAFT](https://github.com/princeton-vl/RAFT)):
python features/scripts/compute_flows.py \
    $ROOT_DIR/raw_unity/raw_frames $ROOT_DIR/flow_raft_pk/ -m raft

# Export external pkl flow files to pth:
python src/datamodules/preprocessing/pkl_to_pth.py \
    $ROOT_DIR/flow_raft_pk $ROOT_DIR/flow_raft_pth

# Preprocess pth Unity and external flows:
python src/datamodules/preprocessing/flow_transforms.py \
    $ROOT_DIR/flow_unity_pth $ROOT_DIR/flow_unity_maxnorm_pth \
    unity_max_module
python src/datamodules/preprocessing/flow_transforms.py \
    $ROOT_DIR/flow_raft_pth $ROOT_DIR/flow_raft_maxnorm_pth \
    raft_max_module