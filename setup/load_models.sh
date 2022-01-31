# Download pre-trained models
mkdir models
curl -L https://storage.googleapis.com/mannequinchallenge-data/checkpoints/best_depth_Ours_Bilinear_inc_3_net_G.pth -o models/depth_estimation.pth
wget http://download.europe.naverlabs.com/ComputerVision/DOPE_models/DOPErealtime_v1_0_0.pth.tgz -P models
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip -P models && unzip models/models.zip
gdown https://drive.google.com/uc\?id\=1PpAKJk8OYqP1m_oMr4DhfHDZiGrN7MQV -O models/toric_estimation.tar
gdown https://drive.google.com/uc\?id\=1VigqrPdiIF18VALo92L9WCuASnpzu7qa -O models/defocus_estimation_vgg.pth
gdown https://drive.google.com/uc\?id\=1IZSo6NhcSKzljQIF_JDqu0IKyzk-lGQ5 -O models/motion.pth