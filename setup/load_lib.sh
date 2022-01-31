# Clone external modules
git clone https://github.com/naver/dope.git $LIB_DIR/dope
git clone https://github.com/naver/lcrnet-v2-improved-ppi.git $LIB_DIR/lcrnet-v2-improved-ppi
git clone https://github.com/google/mannequinchallenge.git $LIB_DIR/mannequinchallenge
git clone https://github.com/jianghd1996/Camera-control.git $LIB_DIR/Camera-control
git clone https://github.com/abewley/sort.git $LIB_DIR/sort
git clone https://github.com/vinthony/depth-distillation.git $LIB_DIR/depth-distillation
git clone https://github.com/princeton-vl/RAFT.git $LIB_DIR/RAFT
git clone https://github.com/robincourant/motion-detection.git $LIB_DIR/motion-detection

# Create symbolic links for external modules
mkdir lib
ln -s $LIB_DIR/dope lib/dope
ln -s $LIB_DIR/lcrnet-v2-improved-ppi lib/lcrnet_v2_improved_ppi
ln -s $LIB_DIR/mannequinchallenge lib/mannequinchallenge
ln -s $LIB_DIR/Camera-control/SIGGRAPH_2020/Movie_analysis lib/camera_control
ln -s $LIB_DIR/sort lib/sort
ln -s $LIB_DIR/depth-distillation lib/depth_distillation
ln -s $LIB_DIR/RAFT lib/RAFT
ln -s $LIB_DIR/motion-detection/motion_detection lib/motion_detection

# Comment line 22 and 23 of the file lib/sort/sort.py to avoid dependency conflicts
python ./setup/comment_lines.py