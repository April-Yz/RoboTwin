# 环境安装
  source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
  conda activate RoboTwin_openvla
  cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft
  ./install_robotwin_openvla.sh

# 下载数据
  source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
  conda activate RoboTwin_openvla
  cd /home/zaijia001/ssd/RoboTwin
  rm -rf data/beat_block_hammer/demo_clean/.cache
  bash collect_data.sh beat_block_hammer demo_clean 0
