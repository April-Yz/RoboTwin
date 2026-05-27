# SSD 指令库（中文）

> 约定：每条都采用 **一行注释 + 一行命令**。可直接复制到终端执行。


# 新的人手数据
rclone copy  gdrive:piper/human/stack_cups/ /home/zaijia001/ssd/data/piper/hand -P --dry-run
rclone copy  gdrive:piper/human/pick_diverse_bottles/pick_diverse_bottles-human-101.zip  /home/zaijia001/ssd/data/piper/hand -P --dry-run
rclone copy  gdrive:piper/human/place_bread_basket/human_place_bread_basket.zip  /home/zaijia001/ssd/data/piper/hand/place_bread_basket -P --dry-run
rclone copy  gdrive:piper/human/pnp_bread/pnp_bread-7.zip /home/zaijia001/ssd/data/piper/hand/pnp_bread/origin -P --dry-run

rclone copy  gdrive:piper/human/handover_bottle/human_handover_bottle.zip  /home/zaijia001/ssd/data/piper/hand/handover_bottle -P --dry-run
rclone copy  gdrive:piper/human/pnp_bread/human_pnp_bread.zip  /home/zaijia001/ssd/data/piper/hand/pnp_bread -P --dry-run
rclone copy  gdrive:piper/human/pnp_tray/human_pnp_tray.zip  /home/zaijia001/ssd/data/piper/hand/pnp_tray -P --dry-run

## A. HaMeR（人手）

### A1. 数据转换（pnp_star_pear -> HaMeR 输入）

# 将整个 pnp_star_pear 转成 HaMeR 可直接读取的 RealR1 格式（rgb_*.mp4 / depth_*.mp4 / params_*.json）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py --input_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input --use_meta_fps --overwrite

# place_bread_basket任务
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py --input_root /home/zaijia001/ssd/data/piper/hand/place_bread_basket/origin --output_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_input --use_meta_fps --overwrite

# pick_diverse_bottles任务
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py --input_root /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/origin --output_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_input --use_meta_fps --overwrite

# pnp_bread任务
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py --input_root /home/zaijia001/ssd/data/piper/hand/pnp_bread/origin --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_bread/harmer_input --use_meta_fps --overwrite

# stack_cups任务
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py --input_root /home/zaijia001/ssd/data/piper/hand/stack_cups/origin --output_dir /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_input --use_meta_fps --overwrite

# handover_bottle任务
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py --input_root /home/zaijia001/ssd/data/piper/hand/handover_bottle/origin --output_dir /home/zaijia001/ssd/data/piper/hand/handover_bottle/harmer_input --use_meta_fps --overwrite

# pnp_tray任务
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py --input_root /home/zaijia001/ssd/data/piper/hand/pnp_tray/origin --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_tray/harmer_input --use_meta_fps --overwrite

# 只转换一个 episode（例如 episode0）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py --input_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input --episodes 0 --use_meta_fps --overwrite

### A2. 人手关键点检测

# 易错点：在 Blackwell GPU 上如果误用 hamer-r1（非 hamer-r1-gpu）跑 CUDA，常见现象是流程跑完但 hand_detected 全 0。

# CPU 跑单个视频（video_id=0），输出 hand_detections_0.* 与可视化视频 hand_vis_0.mp4
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output --video_ids 0 --device cpu

# CPU 跑全部视频（省显存，速度较慢）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output --device cpu

# GPU 跑单个视频（Blackwell 卡请用 hamer-r1-gpu；先清空 LD_LIBRARY_PATH）
unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=1 conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output --video_ids 0 --device cuda

# GPU 跑全部视频（Blackwell 卡请用 hamer-r1-gpu；先清空 LD_LIBRARY_PATH）
unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=1 conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output --device cuda

# GPU 跑全部视频（可改卡号，GPU_ID=0/1/2...；建议先写入新输出目录避免覆盖）
GPU_ID=2; unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2 --device cuda

GPU_ID=2; unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_input --output_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output --device cuda

GPU_ID=2; unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_input --output_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output --device cuda

GPU_ID=2; unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_input --output_dir /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output --device cuda

GPU_ID=2; unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_bread/harmer_input --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_bread/harmer_output --device cuda

GPU_ID=1; unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_tray/harmer_input --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_tray/harmer_output --device cuda

GPU_ID=0; unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/handover_bottle/harmer_input --output_dir /home/zaijia001/ssd/data/piper/hand/handover_bottle/harmer_output --device cuda

# 仅导出 npz/npy，不生成可视化视频（更快）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output --video_ids 0 --device cpu --no_visualize

### A3. 检测结果可视化与检查

# 播放关节可视化视频（骨架）
ffplay -autoexit /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output/hand_vis_0.mp4

# 播放带 gripper 轴可视化视频
ffplay -autoexit /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output/hand_vis_gripper_0.mp4

# 快速查看输出文件是否齐全
ls -lh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output

# 快速统计某个 npz 的手检测帧数（用于确认不是 0/0）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python -c "import numpy as np; d=np.load('/home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output/hand_detections_0.npz'); print('left',int(d['left_hand_detected'].sum()),'/',len(d['left_hand_detected'])); print('right',int(d['right_hand_detected'].sum()),'/',len(d['right_hand_detected']))"

# Debug 基准：已验证在 hamer-r1-gpu + CUDA_VISIBLE_DEVICES=2 下，video_id=0 可达 128/128 双手检测
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=2 conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_dbg_gpuenv --video_ids 0 --device cuda --no_visualize

### A4. 在 RoboTwin 重演手轨迹（hand_detections）

# 将单个 hand_detections_0.npz 做 RoboTwin URDFIK 回放（你这条命令可用）
CUDA_VISIBLE_DEVICES=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output/hand_detections_0.npz /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_pnp_star_pear 5 --lighting_mode front_no_shadow

## B. FoundationPose（双物体位姿估计）

### B1. 准备 FoundationPose 输入

# 把 piper 数据转换成 FoundationPose 可用输入（rgb_<id>.mp4 + depth_<id>/*.npy + params_<id>.json）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py --input_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_input --use_meta_fps --overwrite

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py --input_root /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/origin --output_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_input --use_meta_fps --overwrite

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py --input_root /home/zaijia001/ssd/data/piper/hand/place_bread_basket/origin --output_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_input --use_meta_fps --overwrite

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py --input_root /home/zaijia001/ssd/data/piper/hand/stack_cups/origin --output_dir /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_input --use_meta_fps --overwrite

#### 
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py --input_root /home/zaijia001/ssd/data/piper/hand/handover_bottle/origin --output_dir /home/zaijia001/ssd/data/piper/hand/handover_bottle/foundation_input --use_meta_fps --overwrite

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py --input_root /home/zaijia001/ssd/data/piper/hand/pnp_bread/origin --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_input --use_meta_fps --overwrite

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py --input_root /home/zaijia001/ssd/data/piper/hand/pnp_tray/origin --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_tray/foundation_input --use_meta_fps --overwrite


### B2. 运行 FoundationPose（pear + star fruit）

# 跑单个序列（video_id=0），同时检测 pear 和 star fruit（杨桃）
source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=3 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_input --output_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis --video_ids 0 --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj --object "star fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45

source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=2 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_input  --output_root /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_vis/obs_vis --video_ids 0 --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj" --object "left bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45

# 跑全部序列（默认会跳过已存在 poses.npz 的项）
source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=3 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_input --output_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj --object "star fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45

source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=2 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_input  --output_root /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_vis/obs_vis --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj" --object "left bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45

source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=2 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_input  --output_root /home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_vis/obs_vis --object "basket=/home/zaijia001/ssd/data/R1/hand/obj_mesh/basket/basket.obj" --object "bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_y/bread_y.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45

source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=2 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_input  --output_root /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_vis/obs_vis  --object "right dark red cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj" --object "left light pink cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45


#####  
source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=1 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/handover_bottle/foundation_input  --output_root /home/zaijia001/ssd/data/piper/hand/handover_bottle/foundation_vis/obs_vis  --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45

source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=2 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_input  --output_root /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_vis/obs_vis  --object "right bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_yerong/bread_yerong.obj" --object "left bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_niujiao/bread_niujiao.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45

source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=2 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_tray/foundation_input  --output_root /home/zaijia001/ssd/data/piper/hand/pnp_tray/foundation_vis/obs_vis  --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj"  --object "left dark red cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45



# 使用专门的 piper 包装脚本跑（可加 --video_ids 0 1 2 或 --gpu 2）
bash /home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh

## C. Foundation 结果在 RoboTwin 重演（轨迹 / pos）

### C0. 当前 Piper 标定参数（new_table / 2026-05-15）

> 当前 C/D/E 中所有 Piper replay 命令使用 0515 new_table 标定。下次重新标定时，优先改这里的路径和值，并同步修改下面列出的 wrapper 默认值。
>
> 标定源文件：
> - head D435：`/home/zaijia001/ssd/data/piper/calibration/handeye/head_d435_new_table_0515_head_from_wrist.json`
> - 左右 base：`/home/zaijia001/ssd/data/piper/calibration/handeye/left_base_T_right_base_new_table.json`
> - 左腕相机：`/home/zaijia001/ssd/data/piper/calibration/handeye/left_wrist_new_table_eye_in_hand.json`
> - 右腕相机：`/home/zaijia001/ssd/data/piper/calibration/handeye/right_wrist_new_table_eye_in_hand.json`
>
> 注意：同目录下还有旧的 `head_d435_new_table_head_from_wrist.json`，本轮不用它；当前使用的是带 `0515` 的 head 标定。
>
> 当前派生参数：
> - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json`
> - `--head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325`
> - `--head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399`
> - 注意：handeye JSON 中的 head 是 raw/optical 相机坐标；replay 命令使用 `--camera_cv_axis_mode legacy_r1` 时，命令和 bundle 里保存的是 render/SAPIEN 相机位姿，即 `raw_optical @ legacy_r1.T`。如果直接把 raw quaternion 写进 replay，会出现“位置对但朝向偏”的现象。
> - 左腕 `gripper_T_camera`：pos `-0.1493144547324326 0.020709178309353107 0.04361262758934953`，quat_wxyz `-0.4327524246505378 0.557101905823548 -0.43171951730171426 0.562121929715988`
> - 右腕 `gripper_T_camera`：pos `-0.13501234680848043 -0.027382233580855064 0.0394202891823076`，quat_wxyz `0.6579499408322479 -0.2646744381656339 0.6467791787566072 -0.2805815586731893`
>
> 当前摆放估算（基于 `robot_config_PiperPika_agx_dual_table_0515.json` 中保留的 left base 世界位姿，并应用 `left_base_T_right_base_new_table.json`）：
> - left base world：`[-0.300000, -0.250000, 0.750000]`
> - right base world：`[0.556208, -0.271842, 0.769842]`
> - head camera world：`[0.091894, -0.137896, 1.225389]`
> - 两个 base 间距约 `0.857 m`；从 left 到 right 主要是世界 `+x 0.856 m`，同时 `-y 0.022 m`、`+z 0.020 m`
> - head 相对 left base：`dx=+0.392 m, dy=+0.112 m, dz=+0.475 m`，水平距离约 `0.408 m`，三维距离约 `0.626 m`
> - head 相对 right base：`dx=-0.464 m, dy=+0.134 m, dz=+0.456 m`，水平距离约 `0.483 m`，三维距离约 `0.664 m`
>
> 下次要同步修改的文件：
> - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`：用户侧可复制命令。
> - `/home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json`：左右 base 位姿；当前由 `left_base_T_right_base_new_table.json` 派生。
> - `/home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh`：D 批处理默认 `ROBOT_CONFIG/HEAD_POS/HEAD_QUAT`。
> - `/home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`：E 批处理默认 `ROBOT_CONFIG/HEAD_POS/HEAD_QUAT`。
> - `/home/zaijia001/ssd/RoboTwin/code_painting/plot_piper_gripper_wrist_object_axis_distances.py`：debug 距离曲线默认 `robot_config/head_camera`。
> - 如果后续启用真实 wrist camera 渲染，再把 `left_wrist_new_table_eye_in_hand.json` 和 `right_wrist_new_table_eye_in_hand.json` 的 `gripper_T_camera` 分别接到左右 wrist camera local pose 参数；当前 D/E 主 replay 主要使用 head camera，不依赖 wrist camera 标定。

# 从四个 handeye 标定文件生成一个自包含 Piper 标定 bundle；后续 replay 可以只指定 CALIBRATION_BUNDLE，避免散改 robot_config/head/wrist 参数
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/build_piper_calibration_bundle.py --head /home/zaijia001/ssd/data/piper/calibration/handeye/head_d435_new_table_0515_head_from_wrist.json --base /home/zaijia001/ssd/data/piper/calibration/handeye/left_base_T_right_base_new_table.json --left_wrist /home/zaijia001/ssd/data/piper/calibration/handeye/left_wrist_new_table_eye_in_hand.json --right_wrist /home/zaijia001/ssd/data/piper/calibration/handeye/right_wrist_new_table_eye_in_hand.json --template_robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table.json --output /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json

# 可视化 base/head camera 坐标轴，并把旧 head 外参一起画出来对比；红=x，绿=y，蓝=z，SAPIEN 相机通常看向 -z
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/visualize_piper_calibration_bundle.py --bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --output_png /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_calibration_bundle_0515/axes_compare_old_head.png --old_head_camera_local_pos 0.107882 -0.2693875 0.464396 --old_head_camera_local_quat_wxyz 0.85401166 0.01255256 0.51885652 -0.0359783

### C1. 双物体一起回放（默认推荐）

##### 回放 FoundationPose 结果（Piper 标定 head cam 版本，无机器人，且保存姿态debug；建议先跑单个id）
CUDA_VISIBLE_DEVICES=3 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_pnp_star_pear_norobot 5 --ids 0 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj --object star_fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj

#### 批处理
CUDA_VISIBLE_DEVICES=3 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_pnp_star_pear_norobot 5 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj --object star_fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj

#####  pick_diverse_bottles：FoundationPose 双物体 replay，使用 Piper 0515 head/base 标定；先用 --ids 0 单条验证
CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_replay 5 --ids 0 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj" --object "left bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj"
```bash
CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_replay 5 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj" --object "left bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj"

zip -r pick_diverse_bottles_foundation_replay.zip foundation_replay

CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_replay 5 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "basket=/home/zaijia001/ssd/data/R1/hand/obj_mesh/basket/basket.obj" --object "bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_y/bread_y.obj" 

zip -r place_bread_basket_foundation_replay.zip foundation_replay

CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_replay 5 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right dark red cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj" --object "left light pink cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj" 

zip -r stack_cups_foundation_replay.zip foundation_replay



##### 
CUDA_VISIBLE_DEVICES=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/handover_bottle/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/handover_bottle/foundation_replay 5 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj"

zip -r handover_bottle_foundation_replay.zip foundation_replay

CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_replay 5 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399  --object "right bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_yerong/bread_yerong.obj" --object "left bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_niujiao/bread_niujiao.obj" 

zip -r pnp_bread_foundation_replay.zip foundation_replay

CUDA_VISIBLE_DEVICES=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_tray/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/pnp_tray/foundation_replay 5 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj"  --object "left dark red cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj" 
zip -r pnp_tray_foundation_replay.zip foundation_replay

```




### C2. 只看 pear 的轨迹和位姿

# 只重演 pear（用于单物体检查）
python /home/zaijia001/ssd/RoboTwin/code_painting/render_multi_object_pose_r1_npz_batch.py --input_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis --output_root /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_pnp_star_pear_only --fps 5 --head_only 1 --hide_robot 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --objects pear --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj

### C3. 只看 star_fruit（杨桃）的轨迹和位姿

# 只重演 star_fruit（用于单物体检查）
CUDA_VISIBLE_DEVICES=3 python /home/zaijia001/ssd/RoboTwin/code_painting/render_multi_object_pose_r1_npz_batch.py --input_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis --output_root /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_pnp_star_fruit_only --fps 5 --head_only 1 --hide_robot 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --objects star_fruit --object star_fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj

## D. HaMeR 结果在 RoboTwin 按 HaMeR 轴原样 replay（Piper 标定 head cam）

> 最终确认规则：直接使用 NPZ 中的 `gripper_position` + `gripper_rotation_matrix`，不做轴 remap，不做 post-rot。即 `--orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0`。当前语义：蓝色 `+Z` 是夹爪前进/接近方向，绿色 `+Y` 是开合轴，红色 `+X` 是侧向/法向。
>
> 夹爪朝向 debug、扫图、post-rot 和历史 remap 指令已迁移到：`/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md`。
>
> Piper 规则总结见：`/home/zaijia001/ssd/RoboTwin/agent-read/PIPER_GRIPPER_ORIENTATION_RULES.zh.md`。

### D0. place_bread_basket：HaMeR 检测结果与人手夹爪可视化

# 跑 place_bread_basket 的 HaMeR 检测；输出 hand_detections_*.npz、hand_vis_*.mp4、hand_vis_gripper_*.mp4
GPU_ID=2; unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_input --output_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output --device cuda

# 只跑部分视频 ID 做 debug：例子只跑 id0-id10；会输出 hand_detections_0..10 和 hand_vis/hand_vis_gripper_0..10
GPU_ID=2; unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=${GPU_ID} conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py --data_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_input --output_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output --video_ids $(seq 0 10) --device cuda

# 快速统计每个 npz 的左右手检测帧数
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n hamer-r1 python - <<'PY'
import glob, os
import numpy as np
for p in sorted(glob.glob('/home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_detections_*.npz')):
    d = np.load(p)
    n = len(d['left_hand_detected'])
    l = int(d['left_hand_detected'].sum())
    r = int(d['right_hand_detected'].sum())
    print(f"{os.path.basename(p)} left={l}/{n} right={r}/{n}")
PY

# 查看 HaMeR 自带的人手可视化与“夹爪轴/夹爪点”可视化视频
ls -lh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_*.mp4 /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_gripper_*.mp4
ffplay -autoexit /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_gripper_0.mp4

# 把 HaMeR 检测出来的 gripper_position + gripper_rotation_matrix 放进 Piper 场景 replay；用于检查夹爪轴在 0515 标定场景中的表现
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes


CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes

# 批处理全部 place_bread_basket 检测结果；每个 id 输出到 output_place_bread_basket_piper_hamer_axes/id_<id>
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=3000 ARMS=both KEEP_ONLY_ZED_THIRD=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes

# 只 replay id0-id10 做 debug；每个 id 输出到 output_place_bread_basket_piper_hamer_axes/id_<id>
```bash
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=3000 ARMS=both KEEP_ONLY_ZED_THIRD=0 ID_FILTER=0-10 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes

CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=3000 ARMS=both KEEP_ONLY_ZED_THIRD=0 ID_FILTER=0-10 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/place_bread_basket

CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=3000 ARMS=both KEEP_ONLY_ZED_THIRD=0 ID_FILTER=0-10 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/pick_diverse_bottles

CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=3000 ARMS=both KEEP_ONLY_ZED_THIRD=0 ID_FILTER=0-10 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_bread/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/pnp_bread

CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=3000 ARMS=both KEEP_ONLY_ZED_THIRD=0 ID_FILTER=0-10 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/stack_cups
```

# 查看 Piper replay 的 zed/third 视角视频与 PNG 帧
ls -lh /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_0/*replay.mp4
find /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_0/frames -maxdepth 1 \( -name '*zed*.png' -o -name '*third*.png' \) | sort | head -n 40

# 使用统一标定 bundle 跑 id0；推荐后续使用这个形式，避免 robot_config/head camera 参数散落在命令中。DEBUG_VISUALIZE_CAMERAS=1 会在 third 视角里画出 head camera 的白色机身、红绿蓝 xyz 轴、黄色 -Z 光轴。
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json DEBUG_VISUALIZE_CAMERAS=1 DEBUG_CAMERA_AXIS_LENGTH=0.22 GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes_bundle

# 打印 head camera 世界位姿/forward/right/up，快速判断相机朝向是否符合预期
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=1 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0 DEBUG_MODE=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_camera_debug

# 打开 SAPIEN viewer 交互检查 head/third 视角；不要设置 CUDA_VISIBLE_DEVICES=2。SAPIEN viewer 需要能看到驱动 VNC/X display 的 GPU，否则会报 Renderer does not support display。
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && echo "DISPLAY=$DISPLAY WAYLAND_DISPLAY=$WAYLAND_DISPLAY XDG_SESSION_TYPE=$XDG_SESSION_TYPE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_detections_0.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_viewer_probe --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --fps 5 --max_frames 1 --arms both --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 1 --debug_post_execute 1 --save_png_frames 1 --lighting_mode front_no_shadow --debug_visualize_cameras 1 --debug_camera_axis_length 0.22 --enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02

# 如果上面没有弹窗，先在同一个 VNC 终端跑这个最小 SAPIEN viewer 探针；它不加载机器人，只验证 viewer 能不能创建和显示。
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && echo "DISPLAY=$DISPLAY WAYLAND_DISPLAY=$WAYLAND_DISPLAY XDG_SESSION_TYPE=$XDG_SESSION_TYPE" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py

# 如果想确认是不是 CUDA_VISIBLE_DEVICES=2 导致 viewer 失败，用这个反例探针；如果这里失败而上面不带 CUDA_VISIBLE_DEVICES 的 probe 成功，就说明 display GPU 被 CUDA mask 隐藏了。
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && echo "DISPLAY=$DISPLAY WAYLAND_DISPLAY=$WAYLAND_DISPLAY XDG_SESSION_TYPE=$XDG_SESSION_TYPE CUDA_VISIBLE_DEVICES=2" && CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py

# 生成 pre-0515 new_table bundle：同一套 base/左右 wrist，但 head 使用旧的 head_d435_new_table_head_from_wrist.json
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/build_piper_calibration_bundle.py --head /home/zaijia001/ssd/data/piper/calibration/handeye/head_d435_new_table_head_from_wrist.json --base /home/zaijia001/ssd/data/piper/calibration/handeye/left_base_T_right_base_new_table.json --left_wrist /home/zaijia001/ssd/data/piper/calibration/handeye/left_wrist_new_table_eye_in_hand.json --right_wrist /home/zaijia001/ssd/data/piper/calibration/handeye/right_wrist_new_table_eye_in_hand.json --template_robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table.json --output /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_pre0515.json

# 三版本 head camera marker 对比 1：最早手写 head 参数 + 旧 robot_config（不使用 bundle）
ROBOT_CONFIG=/home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table.json HEAD_POS="0.107882 -0.2693875 0.464396" HEAD_QUAT="0.85401166 0.01255256 0.51885652 -0.0359783" DEBUG_VISUALIZE_CAMERAS=1 DEBUG_CAMERA_AXIS_LENGTH=0.22 GPU=2 FPS=5 MAX_FRAMES=1 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_camera_compare/old_manual

# 三版本 head camera marker 对比 2：pre-0515 new_table head bundle
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_pre0515.json DEBUG_VISUALIZE_CAMERAS=1 DEBUG_CAMERA_AXIS_LENGTH=0.22 GPU=2 FPS=5 MAX_FRAMES=1 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_camera_compare/new_table_pre0515

# 三版本 head camera marker 对比 3：0515 new_table head bundle（当前推荐）
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json DEBUG_VISUALIZE_CAMERAS=1 DEBUG_CAMERA_AXIS_LENGTH=0.22 GPU=2 FPS=5 MAX_FRAMES=1 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_camera_compare/new_table_0515

# 查看三版本 third 视角 PNG；head camera marker 会画在 third_0000.png 里
find /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_camera_compare -path '*/id_0/frames/third_0000.png' | sort

### D1. 单个 id0：最终确认可用 replay（双手）

# 单个 id0：严格使用 NPZ 中 HaMeR/重算得到的 gripper_position + gripper_rotation_matrix；不做 remap，不做 post-rot，左右手一起 replay。推荐使用 piper_calibration_bundle，避免 robot_config/head camera 参数散写。
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_id0_both --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 1 --debug_post_execute 1 --save_png_frames 1 --lighting_mode front_no_shadow --debug_frame_limit -1

# 查看 replay 视频和保存的 zed/third PNG 帧
ls -lh /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_id0_both/*replay.mp4
find /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_id0_both/frames -maxdepth 1 \( -name '*zed*.png' -o -name '*third*.png' \) | sort | head -n 40

### D2. 批处理：全部 hand_detections_*.npz 按最终 HaMeR 轴 replay

# 批处理全部 hand_detections_*.npz；每个 id 输出到 output_piper_replay_hamer_axes_all/id_<id>，KEEP_ONLY_ZED_THIRD=0 会保留 zed/third/depth/wrist 全部输出
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2 /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_all

# 沿夹爪自身蓝色 +Z 前进轴“后退”指定距离再 replay：正数表示 target position -= distance * local(+Z)，例如 0.05 表示后退 5cm；该偏移跟随每帧夹爪朝向，不是固定世界 XYZ。
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json TARGET_LOCAL_FORWARD_RETREAT_M=0.05 GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/stack_cups—z-005

# 底层单条命令格式：等价于上面的 wrapper 环境变量；可把 0.05 改成 0.02/0.08/0.12 做距离扫描
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_retreat_blue_z/id_0_direct --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 1 --debug_post_execute 1 --save_png_frames 1 --lighting_mode front_no_shadow --debug_frame_limit -1

# 只跑 id0（用于快速复查）
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2 /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_all

# 只跑多个 ID + 范围（示例：0,2,5-8）
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0,2,5-8 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2 /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_all

# 如果只想检查右手，把 ARMS 改成 right
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json GPU=2 FPS=5 MAX_FRAMES=300 ARMS=right KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2 /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_right

### D3. 批处理输出检查

# 查看某个 id 的 replay 视频
ls -lh /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_all/id_0/*replay.mp4

# 查看某个 id 的 PNG 帧；KEEP_ONLY_ZED_THIRD=1 时会删除 depth/wrist PNG 和 depth/wrist mp4，只保留 zed/third 视角 RGB 帧
find /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_all/id_0/frames -maxdepth 1 -type f | sort | head -n 40

# 检查 frames 中是否只剩 zed/third 两类 RGB 帧
find /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_all/id_0/frames -maxdepth 1 -type f | sed 's#.*/##' | cut -d_ -f1 | sort | uniq -c

### D4. HaMeR 可视化视频 + Piper retarget replay 横向拼接对比

# 单个 id0：左=HaMeR hand_vis_gripper，右=Piper zed_replay；输出 compare_hamer_gripper_vs_piper_zed_0.mp4
ID=0; H=/home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_gripper_${ID}.mp4; P=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/zed_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/compare_hamer_gripper_vs_piper_zed_${ID}.mp4; ffmpeg -y -i "$H" -i "$P" -filter_complex "[0:v]scale=640:360,setsar=1,drawtext=text='HaMeR gripper id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]scale=640:360,setsar=1,drawtext=text='Piper zed replay id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[v0][v1]hstack=inputs=2[v]" -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "$O"

# 单个 id0：左=HaMeR hand_vis_gripper，中=Piper zed_replay，右=Piper third_replay；输出三联对比视频
ID=0; H=/home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/compare_hamer_gripper_piper_zed_third_${ID}.mp4; ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]scale=640:360,setsar=1,drawtext=text='HaMeR gripper id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3[v]" -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "$O"

# 对齐版：HaMeR 通常是 30fps，Piper replay 是 5fps；这里自动按 zed_replay 时长拉伸 HaMeR，再横向拼接，避免左侧播放过快。
ID=0; H=/home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/compare_aligned_hamer_gripper_piper_zed_third_${ID}.mp4; R=$(python3 - <<PY
import subprocess
def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',p]).decode().strip())
h = dur('$H')
z = dur('$Z')
print(z / h)
PY
); ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]setpts=${R}*PTS,fps=5,scale=640:360,setsar=1,drawtext=text='HaMeR gripper aligned id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]" -map "[v]" -an -r 5 -c:v libx264 -pix_fmt yuv420p "$O"


# 批量拼接 id0-id10：生成每个 id 的三联对比视频；需要先跑完 HaMeR 检测和 Piper replay
for ID in $(seq 0 10); do H=/home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/compare_hamer_gripper_piper_zed_third_${ID}.mp4; if [[ -f "$H" && -f "$Z" && -f "$T" ]]; then ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]scale=640:360,setsar=1,drawtext=text='HaMeR gripper id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3[v]" -map "[v]" -an -c:v libx264 -pix_fmt yuv420p "$O"; else echo "[skip] id=${ID} missing one of: $H $Z $T"; fi; done

# 批量对齐版 id0-id10：自动按每个 id 的 zed_replay 时长拉伸 HaMeR，再生成三联对比视频。
for ID in $(seq 0 10); do H=/home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes/id_${ID}/compare_aligned_hamer_gripper_piper_zed_third_${ID}.mp4; if [[ -f "$H" && -f "$Z" && -f "$T" ]]; then R=$(python3 - <<PY
import subprocess
def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',p]).decode().strip())
print(dur('$Z') / dur('$H'))
PY
); ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]setpts=${R}*PTS,fps=5,scale=640:360,setsar=1,drawtext=text='HaMeR gripper aligned id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]" -map "[v]" -an -r 5 -c:v libx264 -pix_fmt yuv420p "$O"; else echo "[skip] id=${ID} missing one of: $H $Z $T"; fi; done



```bash
for ID in $(seq 0 10); do H=/home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/place_bread_basket/id_${ID}/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/place_bread_basket/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/place_bread_basket/id_${ID}/compare_aligned_hamer_gripper_piper_zed_third_${ID}.mp4; if [[ -f "$H" && -f "$Z" && -f "$T" ]]; then R=$(python3 - <<PY
import subprocess
def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',p]).decode().strip())
print(dur('$Z') / dur('$H'))
PY
); ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]setpts=${R}*PTS,fps=5,scale=640:360,setsar=1,drawtext=text='HaMeR gripper aligned id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]" -map "[v]" -an -r 5 -c:v libx264 -pix_fmt yuv420p "$O"; else echo "[skip] id=${ID} missing one of: $H $Z $T"; fi; done


for ID in $(seq 0 10); do H=/home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/pick_diverse_bottles/id_${ID}/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/pick_diverse_bottles/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/pick_diverse_bottles/id_${ID}/compare_aligned_hamer_gripper_piper_zed_third_${ID}.mp4; if [[ -f "$H" && -f "$Z" && -f "$T" ]]; then R=$(python3 - <<PY
import subprocess
def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',p]).decode().strip())
print(dur('$Z') / dur('$H'))
PY
); ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]setpts=${R}*PTS,fps=5,scale=640:360,setsar=1,drawtext=text='HaMeR gripper aligned id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]" -map "[v]" -an -r 5 -c:v libx264 -pix_fmt yuv420p "$O"; else echo "[skip] id=${ID} missing one of: $H $Z $T"; fi; done


for ID in $(seq 0 10); do H=/home/zaijia001/ssd/data/piper/hand/pnp_bread/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/pnp_bread/id_${ID}/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/pnp_bread/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/pnp_bread/id_${ID}/compare_aligned_hamer_gripper_piper_zed_third_${ID}.mp4; if [[ -f "$H" && -f "$Z" && -f "$T" ]]; then R=$(python3 - <<PY
import subprocess
def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',p]).decode().strip())
print(dur('$Z') / dur('$H'))
PY
); ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]setpts=${R}*PTS,fps=5,scale=640:360,setsar=1,drawtext=text='HaMeR gripper aligned id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]" -map "[v]" -an -r 5 -c:v libx264 -pix_fmt yuv420p "$O"; else echo "[skip] id=${ID} missing one of: $H $Z $T"; fi; done

for ID in $(seq 0 10); do H=/home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/stack_cups/id_${ID}/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/stack_cups/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/stack_cups/id_${ID}/compare_aligned_hamer_gripper_piper_zed_third_${ID}.mp4; if [[ -f "$H" && -f "$Z" && -f "$T" ]]; then R=$(python3 - <<PY
import subprocess
def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',p]).decode().strip())
print(dur('$Z') / dur('$H'))
PY
); ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]setpts=${R}*PTS,fps=5,scale=640:360,setsar=1,drawtext=text='HaMeR gripper aligned id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]" -map "[v]" -an -r 5 -c:v libx264 -pix_fmt yuv420p "$O"; else echo "[skip] id=${ID} missing one of: $H $Z $T"; fi; done

for ID in $(seq 0 10); do H=/home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/stack_cups—z-005/id_${ID}/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/stack_cups—z-005/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/stack_cups—z-005/id_${ID}/compare_aligned_hamer_gripper_piper_zed_third_${ID}.mp4; if [[ -f "$H" && -f "$Z" && -f "$T" ]]; then R=$(python3 - <<PY
import subprocess
def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',p]).decode().strip())
print(dur('$Z') / dur('$H'))
PY
); ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]setpts=${R}*PTS,fps=5,scale=640:360,setsar=1,drawtext=text='HaMeR gripper aligned id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]" -map "[v]" -an -r 5 -c:v libx264 -pix_fmt yuv420p "$O"; else echo "[skip] id=${ID} missing one of: $H $Z $T"; fi; done
```


# 查看拼接结果
find /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes -path '*/compare_hamer_gripper_piper_zed_third_*.mp4' | sort

# 查看对齐版拼接结果
find /home/zaijia001/ssd/RoboTwin/code_painting/output_place_bread_basket_piper_hamer_axes -path '*/compare_aligned_hamer_gripper_piper_zed_third_*.mp4' | sort

## E. HaMeR 手 + FoundationPose 物体同场 replay（Piper 标定 head cam）

> 用途：在 D 的最终 Piper/HaMeR 手 replay 场景里，同时加载 C 的 FoundationPose 物体轨迹。手和物体共用同一套 PiperPika robot config、`legacy_r1` camera axis、head camera local pos/quat 标定。
>
> 手的最终规则仍然是：`--require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0`。

### E1. 批处理：全部 hand_detections_*.npz + FoundationPose 物体轨迹

# 批处理全部 ID：每个 id 输出到 output_piper_replay_hamer_axes_with_objects_all/id_<id>；KEEP_ONLY_ZED_THIRD=0 会保留 zed/third/depth/wrist 全部视频和 PNG，方便检查
GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2 /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_with_objects_all

# 只跑 id0（快速复查手和 pear/star_fruit 是否在同一坐标系内对齐）
GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=0 ID_FILTER=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2 /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_with_objects_all

# 只跑多个 ID + 范围（示例：0,2,5-8）
GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=0 ID_FILTER=0,2,5-8 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2 /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_with_objects_all

### E2. 单条命令：手 + 指定 FoundationPose video dir

# 单个 id0：直接指定 hand_detections_0.npz 和对应 FoundationPose video-level 目录；如果物体某帧缺失，默认 hide
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_with_objects_id0 --fps 5 --max_frames 300 --arms both --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 1 --debug_frame_limit -1 --save_png_frames 1 --object_replay_input_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis/pnp_star_pear_foundation_input_0 --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj --object star_fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj --object_missing_frame_policy hide --lighting_mode front_no_shadow

#### E0. 三个 H2O 任务：pure Piper replay（只保留 zed/third RGB，无文字/坐标轴）

用途：为后续 SAM repaint 准备干净机器人视频。输出每个 id 只保留 `zed_replay.mp4` 和 `third_replay.mp4`；`--clean_output 1 --overlay_text_enable 0` 关闭左上角文字，`--debug_visualize_targets 0 --debug_visualize_cameras 0` 关闭目标/相机坐标轴可视化，最后用 ffmpeg 转成 VS Code 更稳定支持的 `h264/yuv420p/faststart`。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && GPU=3; FPS=5; MAX_FRAMES=3000; RETREAT=0.05; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 10 120); do IN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; OUT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK}/id${ID}_z005; [[ -f "$IN" ]] || { echo "[skip] missing $IN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz "$IN" --output_dir "$OUT" --fps ${FPS} --max_frames ${MAX_FRAMES} --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m ${RETREAT} --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 0 --debug_frame_limit -1 --debug_visualize_targets 0 --debug_visualize_cameras 0 --clean_output 1 --overlay_text_enable 0 --save_png_frames 0 --lighting_mode front_no_shadow; rm -f "$OUT"/zed_depth.mp4 "$OUT"/left_wrist_replay.mp4 "$OUT"/right_wrist_replay.mp4 "$OUT"/smooth_*.mp4; rm -rf "$OUT"/frames; for V in zed_replay third_replay; do [[ -f "$OUT/${V}.mp4" ]] || continue; ffmpeg -y -i "$OUT/${V}.mp4" -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT/${V}.tmp.mp4" && mv "$OUT/${V}.tmp.mp4" "$OUT/${V}.mp4"; done; done; done
```

查看输出：

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure -maxdepth 3 -type f \( -name 'zed_replay.mp4' -o -name 'third_replay.mp4' \) | sort
```

#### E2.0 三个 H2O 任务：只 replay 人手，不加载 FoundationPose 物体

> 用途：排除 FoundationPose 物体加载/渲染对画面的影响，只检查 HaMeR gripper pose 到 Piper 双臂 replay 的结果。下面三条命令仍使用 0515 Piper calibration bundle、identity gripper 朝向、沿夹爪蓝色 +Z 反方向后退 5cm。`--save_png_frames 0` 表示只保存 replay mp4/npz 等主输出，不保存 `frames/` 下的逐帧 PNG。

```bash
# pick_diverse_bottles：只回放人手 id0-id10
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=3 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output/hand_detections_${ID}.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2/pick_diverse_bottles/id${ID}_z005 --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 1 --debug_frame_limit -1 --save_png_frames 0 --lighting_mode front_no_shadow; done

# place_bread_basket：只回放人手 id0-id10
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=3 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_detections_${ID}.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2/place_bread_basket/id${ID}_z005 --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 1 --debug_frame_limit -1 --save_png_frames 0 --lighting_mode front_no_shadow; done

# stack_cups：只回放人手 id0-id10
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=3 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_detections_${ID}.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2/stack_cups/id${ID}_z005 --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 1 --debug_frame_limit -1 --save_png_frames 0 --lighting_mode front_no_shadow; done
```

如果 `zed_replay.mp4` / `third_replay.mp4` 不能在 VS Code 里预览，通常是原始 mp4 的 codec/pixel format 不被 VS Code 的 Chromium 播放器接受。用 ffmpeg 转成 H.264 + yuv420p 后一般可以直接预览：

```bash
ID=0; I=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2/place_bread_basket/id${ID}_z005/zed_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2/place_bread_basket/id${ID}_z005/zed_replay_vscode.mp4; ffmpeg -y -i "$I" -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$O"
```

#### E2.1 pick_diverse_bottles：手 + right_bottle/left_bottle

```bash
# 非 viewer 版本：批量跑 id0-id10，沿夹爪蓝色 +Z 前进轴反方向后退 5cm；把 --target_local_forward_retreat_m 0.05 改成 0.00/0.02/0.08 可做距离扫描
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output/hand_detections_${ID}.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/pick_diverse_bottles/id${ID}_z005 --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 1 --debug_frame_limit -1 --save_png_frames 1 --object_replay_input_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_vis/obs_vis/foundation_input_${ID} --object right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj --object left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_missing_frame_policy hide --lighting_mode front_no_shadow; done
```

# viewer 版本：建议先把 `seq 0 10` 改成单个 ID（例如 `seq 0 0`），再在 python 参数末尾追加这三个参数；无显示环境不要开
# --enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02

#### E2.2 place_bread_basket：手 + basket/bread

# 非 viewer 版本：批量跑 id0-id10，默认后退 5cm
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 10 11); do CUDA_VISIBLE_DEVICES=3 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_detections_${ID}.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/place_bread_basket/id${ID}_z005 --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 1 --debug_frame_limit -1 --save_png_frames 1 --object_replay_input_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_vis/obs_vis/foundation_input_${ID} --object basket=/home/zaijia001/ssd/data/R1/hand/obj_mesh/basket/basket.obj --object bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_y/bread_y.obj --object_missing_frame_policy hide --lighting_mode front_no_shadow; done

# viewer 版本：建议先把 `seq 0 10` 改成单个 ID（例如 `seq 0 0`），再在 python 参数末尾追加这三个参数；无显示环境不要开
# --enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02

#### E2.3 stack_cups：手 + right_dark_red_cup/left_light_pink_cup

# 非 viewer 版本：批量跑 id0-id10，默认后退 5cm
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 10 100); do CUDA_VISIBLE_DEVICES=3 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_detections_${ID}.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/stack_cups/id${ID}_z005 --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 1 --debug_frame_limit -1 --save_png_frames 1 --object_replay_input_dir /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_vis/obs_vis/foundation_input_${ID} --object right_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj --object left_light_pink_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj --object_missing_frame_policy hide --lighting_mode front_no_shadow; done

# viewer 版本：建议先把 `seq 0 10` 改成单个 ID（例如 `seq 0 0`），再在 python 参数末尾追加这三个参数；无显示环境不要开
# --enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02

#### E2.4 D435 正常 head 视角：人手数据 pure Piper replay

用途：给后续 repaint 生成更接近真实 head D435 的机器人视频，并与前面的默认 640x360 / `fovy_deg=90` replay 并列保存。原始人手数据的 D435 参数可从 `/home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/origin/episode35/head_d435_rgbd_meta.json` 和 `/home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_input/params_35.json` 交叉确认：`camera_name=headD435`，`rgb_topic=/camera/color/image_raw`，`depth_topic=/camera/aligned_depth_to_color/image_raw`，`fx=617.0489501953125`，`fy=617.160888671875`，`cx=312.47467041015625`，`cy=245.45556640625`，`width=640`，`height=480`。`42.499880046655484°` 不是只看图片尺寸猜出来的，而是由实际录制写出的 camera_info 内参按 `fovy = 2 * atan(height / (2 * fy))` 换算得到；只知道 `640x480` 不能推断 FOV。按该内参换算，当前录制 RGB 流的有效水平 FOV 约 `54.822°`，垂直 FOV 约 `42.500°`，因此 SAPIEN replay 应优先用 `--image_width 640 --image_height 480 --fovy_deg 42.499880046655484`。

官方 D435 规格需要区分 depth 与 color：Intel D435 产品规格页写的 depth FOV 是 `85.2 x 58`；RealSense D400 系列 datasheet 中 D435/D435i 的 color camera FOV 常见标称为 `H:69 / V:42 / D:77`。本 replay 目标是匹配 `/camera/color/image_raw` 人手 RGB 视频，且 SAPIEN `add_camera(..., fovy=...)` 使用垂直 FOV，所以这里使用实际 RGB camera_info 的 `42.5°`，不是使用 depth 的 `58°`。

原因记录：当前默认 replay 只用了 D435 head 外参，但仍用 `640x360 + fovy_deg=90` 的虚拟广角内参，所以画面比真实 D435 更广，Piper 投影更小。

先跑一个正常 D435 id35 检查视角：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && TASK=pick_diverse_bottles; ID=35; CUDA_VISIBLE_DEVICES=3 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005 --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 0 --debug_frame_limit -1 --debug_visualize_targets 0 --debug_visualize_cameras 0 --clean_output 1 --overlay_text_enable 0 --save_png_frames 1 --lighting_mode front_no_shadow
```

批量生成三个 H2O 任务的 D435 pure replay，输出目录和文件名都带 `d435`，避免和旧版广角 replay 混淆：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && GPU=3; FPS=5; MAX_FRAMES=3000; RETREAT=0.05; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 0 30); do IN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; OUT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005; [[ -f "$IN" ]] || { echo "[skip] missing $IN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz "$IN" --output_dir "$OUT" --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --fps ${FPS} --max_frames ${MAX_FRAMES} --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m ${RETREAT} --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 0 --debug_frame_limit -1 --debug_visualize_targets 0 --debug_visualize_cameras 0 --clean_output 1 --overlay_text_enable 0 --save_png_frames 0 --lighting_mode front_no_shadow; rm -f "$OUT"/zed_depth.mp4 "$OUT"/left_wrist_replay.mp4 "$OUT"/right_wrist_replay.mp4 "$OUT"/smooth_*.mp4; rm -rf "$OUT"/frames; for V in zed_replay third_replay; do [[ -f "$OUT/${V}.mp4" ]] || continue; ffmpeg -y -i "$OUT/${V}.mp4" -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT/${V}_d435.tmp.mp4" && mv "$OUT/${V}_d435.tmp.mp4" "$OUT/${V}_d435.mp4"; done; done; done


source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && GPU=3; FPS=5; MAX_FRAMES=3000; RETREAT=0.05; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 30 60); do IN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; OUT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005; [[ -f "$IN" ]] || { echo "[skip] missing $IN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz "$IN" --output_dir "$OUT" --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --fps ${FPS} --max_frames ${MAX_FRAMES} --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m ${RETREAT} --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 0 --debug_frame_limit -1 --debug_visualize_targets 0 --debug_visualize_cameras 0 --clean_output 1 --overlay_text_enable 0 --save_png_frames 0 --lighting_mode front_no_shadow; rm -f "$OUT"/zed_depth.mp4 "$OUT"/left_wrist_replay.mp4 "$OUT"/right_wrist_replay.mp4 "$OUT"/smooth_*.mp4; rm -rf "$OUT"/frames; for V in zed_replay third_replay; do [[ -f "$OUT/${V}.mp4" ]] || continue; ffmpeg -y -i "$OUT/${V}.mp4" -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT/${V}_d435.tmp.mp4" && mv "$OUT/${V}_d435.tmp.mp4" "$OUT/${V}_d435.mp4"; done; done; done


source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && GPU=3; FPS=5; MAX_FRAMES=3000; RETREAT=0.05; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 60 90); do IN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; OUT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005; [[ -f "$IN" ]] || { echo "[skip] missing $IN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz "$IN" --output_dir "$OUT" --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --fps ${FPS} --max_frames ${MAX_FRAMES} --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m ${RETREAT} --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 0 --debug_frame_limit -1 --debug_visualize_targets 0 --debug_visualize_cameras 0 --clean_output 1 --overlay_text_enable 0 --save_png_frames 0 --lighting_mode front_no_shadow; rm -f "$OUT"/zed_depth.mp4 "$OUT"/left_wrist_replay.mp4 "$OUT"/right_wrist_replay.mp4 "$OUT"/smooth_*.mp4; rm -rf "$OUT"/frames; for V in zed_replay third_replay; do [[ -f "$OUT/${V}.mp4" ]] || continue; ffmpeg -y -i "$OUT/${V}.mp4" -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT/${V}_d435.tmp.mp4" && mv "$OUT/${V}_d435.tmp.mp4" "$OUT/${V}_d435.mp4"; done; done; done


source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && GPU=3; FPS=5; MAX_FRAMES=3000; RETREAT=0.05; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 90 120); do IN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; OUT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005; [[ -f "$IN" ]] || { echo "[skip] missing $IN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz "$IN" --output_dir "$OUT" --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --fps ${FPS} --max_frames ${MAX_FRAMES} --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m ${RETREAT} --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 0 --debug_frame_limit -1 --debug_visualize_targets 0 --debug_visualize_cameras 0 --clean_output 1 --overlay_text_enable 0 --save_png_frames 0 --lighting_mode front_no_shadow; rm -f "$OUT"/zed_depth.mp4 "$OUT"/left_wrist_replay.mp4 "$OUT"/right_wrist_replay.mp4 "$OUT"/smooth_*.mp4; rm -rf "$OUT"/frames; for V in zed_replay third_replay; do [[ -f "$OUT/${V}.mp4" ]] || continue; ffmpeg -y -i "$OUT/${V}.mp4" -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT/${V}_d435.tmp.mp4" && mv "$OUT/${V}_d435.tmp.mp4" "$OUT/${V}_d435.mp4"; done; done; done
```

输出检查：

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435 -maxdepth 3 -type f \( -name 'zed_replay_d435.mp4' -o -name 'third_replay_d435.mp4' \) | sort
```

```bash

for ID in $(seq 0 10); do H=/home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/stack_cups/id${ID}_z005/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/stack_cups/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/stack_cups/id${ID}_z005/compare_aligned_hamer_gripper_piper_zed_third_${ID}.mp4; if [[ -f "$H" && -f "$Z" && -f "$T" ]]; then R=$(python3 - <<PY
import subprocess
def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',p]).decode().strip())
print(dur('$Z') / dur('$H'))
PY
); ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]setpts=${R}*PTS,fps=5,scale=640:360,setsar=1,drawtext=text='HaMeR gripper aligned id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]" -map "[v]" -an -r 5 -c:v libx264 -pix_fmt yuv420p "$O"; else echo "[skip] id=${ID} missing one of: $H $Z $T"; fi; done

for ID in $(seq 0 10); do H=/home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/place_bread_basket/id${ID}_z005/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/place_bread_basket/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/place_bread_basket/id${ID}_z005/compare_aligned_hamer_gripper_piper_zed_third_${ID}.mp4; if [[ -f "$H" && -f "$Z" && -f "$T" ]]; then R=$(python3 - <<PY
import subprocess
def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',p]).decode().strip())
print(dur('$Z') / dur('$H'))
PY
); ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]setpts=${R}*PTS,fps=5,scale=640:360,setsar=1,drawtext=text='HaMeR gripper aligned id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]" -map "[v]" -an -r 5 -c:v libx264 -pix_fmt yuv420p "$O"; else echo "[skip] id=${ID} missing one of: $H $Z $T"; fi; done

for ID in $(seq 0 10); do H=/home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output/hand_vis_gripper_${ID}.mp4; Z=/home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/pick_diverse_bottles/id${ID}_z005/zed_replay.mp4; T=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/pick_diverse_bottles/id_${ID}/third_replay.mp4; O=/home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/pick_diverse_bottles/id${ID}_z005/compare_aligned_hamer_gripper_piper_zed_third_${ID}.mp4; if [[ -f "$H" && -f "$Z" && -f "$T" ]]; then R=$(python3 - <<PY
import subprocess
def dur(p):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',p]).decode().strip())
print(dur('$Z') / dur('$H'))
PY
); ffmpeg -y -i "$H" -i "$Z" -i "$T" -filter_complex "[0:v]setpts=${R}*PTS,fps=5,scale=640:360,setsar=1,drawtext=text='HaMeR gripper aligned id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v0];[1:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper zed id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v1];[2:v]fps=5,scale=640:360,setsar=1,drawtext=text='Piper third id${ID}':x=12:y=12:fontsize=24:fontcolor=lime:box=1:boxcolor=black@0.45[v2];[v0][v1][v2]hstack=inputs=3:shortest=1[v]" -map "[v]" -an -r 5 -c:v libx264 -pix_fmt yuv420p "$O"; else echo "[skip] id=${ID} missing one of: $H $Z $T"; fi; done
```
### E3. 输出检查

# 查看同场 replay 视频（zed/third 是主要检查视角）
ls -lh /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_with_objects_all/id_0/*replay.mp4

# 查看同场 replay 的 PNG 帧；KEEP_ONLY_ZED_THIRD=0 时会保留 depth 和 wrist，KEEP_ONLY_ZED_THIRD=1 时只留 zed/third
find /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_with_objects_all/id_0/frames -maxdepth 1 -type f | sort | head -n 40

## F. 常见排错

# 看 HaMeR 模型文件是否存在（MANO 是必需文件）
ls -lh /home/zaijia001/ssd/hamer_r1/submodules/phantom-hamer/_DATA/data/mano/MANO_RIGHT.pkl

# 如果你希望跑 GPU 版本，可先查看当前 GPU 占用
nvidia-smi

# 输出 id0 的 gripper/wrist-retreat 到物体的世界轴向距离曲线
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && CUDA_VISIBLE_DEVICES=3 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plot_piper_gripper_wrist_object_axis_distances.py --hand_npz /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz --object_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis/pnp_star_pear_foundation_input_0 --output_png /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_with_objects_all/id_0/gripper_wrist_object_axis_distance_id0.png --max_frames 300

## G. H2O 人手/夹爪点到 FoundationPose 物体的世界轴向距离曲线

> 用途：判断“物体 replay 和人手 replay 在 z 轴上整体偏低”到底更像 FoundationPose 检测/深度偏差，还是 RoboTwin replay 坐标转换偏差。图中每个任务每个 id 输出一张 PNG 和同名 CSV；曲线是 `手点 - 物体中心` 的世界坐标轴向差值，蓝线 `dz` 是高度差。
>
> 可视化规则：脚本默认 `--plot_clip_abs_m 0.5`，即 PNG 中超过 `±0.5m` 的异常值会被压到边界显示，便于观察 0.5m 以内的主要趋势；CSV 仍保留原始未裁剪数值。若要关闭压缩显示，追加 `--plot_clip_abs_m 0`。
>
> 读图规则：如果多个任务/多个 id 的 `gripper_dz` 和 `wrist_dz` 同时出现稳定同向偏移，优先怀疑 head/depth/camera-to-world 或 replay 标定；如果只在某个物体或某些帧跳变，优先怀疑 FoundationPose 物体 pose/depth/mesh 估计；如果 z 偏差和 x/y 偏差一起随帧漂移，更像检测跟踪或相机位姿链路问题。

### G1. pick_diverse_bottles：id0-id10 距离曲线

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=3 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plot_piper_gripper_wrist_object_axis_distances.py --hand_npz /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output/hand_detections_${ID}.npz --object_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_vis/obs_vis/foundation_input_${ID} --left_object left_bottle --right_object right_bottle --output_png /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/pick_diverse_bottles/id${ID}_z005/gripper_wrist_object_axis_distance_id${ID}.png --max_frames 300; done
```

### G2. place_bread_basket：id0-id10 距离曲线

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=3 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plot_piper_gripper_wrist_object_axis_distances.py --hand_npz /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_detections_${ID}.npz --object_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_vis/obs_vis/foundation_input_${ID} --left_object basket --right_object bread --output_png /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/place_bread_basket/id${ID}_z005/gripper_wrist_object_axis_distance_id${ID}.png --max_frames 300; done
```

### G3. stack_cups：id0-id10 距离曲线

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=3 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plot_piper_gripper_wrist_object_axis_distances.py --hand_npz /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_detections_${ID}.npz --object_dir /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_vis/obs_vis/foundation_input_${ID} --left_object left_light_pink_cup --right_object right_dark_red_cup --output_png /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/stack_cups/id${ID}_z005/gripper_wrist_object_axis_distance_id${ID}.png --max_frames 300; done
```

### G4. 快速查看输出

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o -path '*/gripper_wrist_object_axis_distance_id*.png' | sort
find /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o -path '*/gripper_wrist_object_axis_distance_id*.csv' | sort
```

## H. HaMeR 原始手点 + FoundationPose 原始物体点对比

> 用途：直接在原始检测可视化层面检查“人手拇指/食指中点”和 FoundationPose 物体中心是否已经存在系统性偏差。H 部分不经过 Piper IK/replay，只对齐 HaMeR `hand_vis_gripper_${ID}.mp4` 与 FoundationPose 每个物体的 `mesh_overlay.mp4`，因此更适合判断偏差来自检测本身还是后续 RoboTwin replay/标定链路。
>
> 可视化点：左侧面板标出 thumb tip、index tip、thumb/index midpoint；物体面板标出 FoundationPose `poses.npz` 的物体中心投影。CSV 记录相机坐标系下 `hand_midpoint - object_center` 的 `dx/dy/dz`。

### H1. G 部分 33 个 replay CSV 的统计摘要

统计口径：三任务 id0-id10，总计 33 个 CSV；正常帧统计只使用 `|value| <= 0.5m` 的轴向差值，超过 0.5m 记为 outlier，原始 CSV 不裁剪。

| 任务 | 点 | abs dx median | abs dy median | abs dz median | signed dx median | signed dy median | signed dz median | 主要 outlier |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pick_diverse_bottles | gripper | 1.8cm | 10.2cm | 14.6cm | -0.1cm | +10.0cm | +14.5cm | z outlier 4.33%，raw min -10.04m |
| pick_diverse_bottles | wrist-retreat | 6.6cm | 5.6cm | 16.5cm | -0.3cm | +2.3cm | +16.3cm | z outlier 4.49%，raw min -9.96m |
| place_bread_basket | gripper | 16.5cm | 4.1cm | 18.8cm | -7.4cm | +2.5cm | +18.1cm | z outlier 2.19%，raw min -3.88m |
| place_bread_basket | wrist-retreat | 23.0cm | 8.2cm | 18.6cm | -12.4cm | -3.9cm | +18.1cm | z outlier 2.19%，raw min -3.91m |
| stack_cups | gripper | 1.9cm | 9.2cm | 14.7cm | +0.2cm | +9.2cm | +14.7cm | 无 `>0.5m` outlier |
| stack_cups | wrist-retreat | 9.0cm | 3.1cm | 16.8cm | +3.8cm | +1.5cm | +16.8cm | 无 `>0.5m` outlier |
| ALL | gripper | 2.9cm | 8.5cm | 15.1cm | -0.3cm | +8.2cm | +15.0cm | z outlier 1.73%，raw min -10.04m |
| ALL | wrist-retreat | 9.5cm | 4.3cm | 17.0cm | -3.8cm | +0.9cm | +16.9cm | z outlier 1.77%，raw min -9.96m |

读数结论：正常帧里 `dz` 的中位数整体在 `+15cm` 到 `+17cm`，说明手点相对 FoundationPose 物体中心在世界 z 轴上有稳定正偏差；这不是单帧执行不到位能解释的问题。pick/place 的少量米级异常更像 FoundationPose 物体 pose track 或缺帧/跳帧问题，stack_cups 没有这种大 outlier。

当前 H 部分原始检测 CSV 统计（已生成 `pick_diverse_bottles` id0-id10，共 11 个 CSV；相机坐标系 `hand_midpoint - object_center`，正常帧仍按 `|value|<=0.5m`）：

| 任务 | 点 | abs dx median | abs dy median | abs dz median | signed dx median | signed dy median | signed dz median | 主要 outlier |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pick_diverse_bottles | left hand vs left_bottle | 1.6cm | 4.0cm | 6.3cm | +0.5cm | -1.7cm | -6.2cm | z outlier 5.12%，raw max +9.63m |
| pick_diverse_bottles | right hand vs right_bottle | 1.8cm | 2.6cm | 3.9cm | -0.9cm | -1.0cm | -3.2cm | z outlier 4.09%，raw max +3.35m |
| pick_diverse_bottles | both | 1.7cm | 3.4cm | 5.1cm | -0.1cm | -1.2cm | -4.6cm | z outlier 4.60%，raw max +9.63m |

H 原始 CSV 与 G replay/world CSV 的差异：H 是相机坐标系原始检测差值，且不包含 `--target_world_offset_xyz 0 0.1 0.1`、Piper 标定 camera-to-world 旋转、机器人 replay 目标偏移和 wrist-retreat 点；G 是 world 坐标系下的 gripper/wrist-retreat 到物体差值。因此 H 中 pick 的正常帧 `abs dz median` 约 `5.1cm`，明显小于 G 中 pick 的 gripper `14.6cm` / wrist `16.5cm`；G 的 z 偏差主要来自 replay/世界坐标转换和人为 target offset/retreat 定义，而不是 HaMeR 与 FoundationPose 原始相机坐标里已经有 15cm 级 z 偏差。

### H2. 原始 HaMeR + FoundationPose 点位对比脚本

入口脚本：

```bash
/home/zaijia001/ssd/RoboTwin/code_painting/make_hamer_foundation_point_compare_video.py
```

输出：

```bash
# 每个 id 输出一个横向拼接视频、同名 CSV、同 stem 的距离曲线 PNG
# 视频：HaMeR hand_vis_gripper + 每个物体 mesh_overlay，并叠加手指/中点/物体中心
# CSV：相机坐标系下 hand_midpoint - object_center 的 dx/dy/dz，单位 m
# PNG：和 G 部分类似的距离曲线；默认 --plot_clip_abs_m 0.5，只压缩 PNG 显示，CSV 保留原始值
```

常用参数：

```bash
# 默认会输出：*_hamer_foundation_points_distance.png
# 如需关闭曲线压缩显示，在任意 H3/H4/H5 命令末尾追加：
--plot_clip_abs_m 0

# 如需指定曲线图路径，追加：
--output_plot /path/to/custom_distance.png
```

### H3. pick_diverse_bottles：id0-id10 原始点位对比

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/make_hamer_foundation_point_compare_video.py --hand_npz /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output/hand_detections_${ID}.npz --hand_video /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output/hand_vis_gripper_${ID}.mp4 --object left_bottle=/home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_vis/obs_vis/foundation_input_${ID}/left_bottle --object right_bottle=/home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_vis/obs_vis/foundation_input_${ID}/right_bottle --left_object left_bottle --right_object right_bottle --output_video /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o_compare_points/pick_diverse_bottles/id${ID}_hamer_foundation_points.mp4 --max_frames 300; done
```

### H4. place_bread_basket：id0-id10 原始点位对比

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/make_hamer_foundation_point_compare_video.py --hand_npz /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_detections_${ID}.npz --hand_video /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_vis_gripper_${ID}.mp4 --object basket=/home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_vis/obs_vis/foundation_input_${ID}/basket --object bread=/home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_vis/obs_vis/foundation_input_${ID}/bread --left_object basket --right_object bread --output_video /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o_compare_points/place_bread_basket/id${ID}_hamer_foundation_points.mp4 --max_frames 300; done
```

说明：如果某个 id 的 `bread/poses.npz` 或物体视频缺失，脚本会输出 warning，并在对应面板/CSV 中保留空值，不会中断整个批处理。

### H5. stack_cups：id0-id10 原始点位对比

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/make_hamer_foundation_point_compare_video.py --hand_npz /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_detections_${ID}.npz --hand_video /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_vis_gripper_${ID}.mp4 --object left_light_pink_cup=/home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_vis/obs_vis/foundation_input_${ID}/left_light_pink_cup --object right_dark_red_cup=/home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_vis/obs_vis/foundation_input_${ID}/right_dark_red_cup --left_object left_light_pink_cup --right_object right_dark_red_cup --output_video /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o_compare_points/stack_cups/id${ID}_hamer_foundation_points.mp4 --max_frames 300; done
```

### H6. 查看 H 部分输出

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o_compare_points -name '*_hamer_foundation_points.mp4' | sort
find /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o_compare_points -name '*_hamer_foundation_points.csv' | sort
find /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o_compare_points -name '*_hamer_foundation_points_distance.png' | sort
```

## I. SAM repaint：H2O 人手抠除 + pure Piper 机械臂贴回

说明：本节模仿 `/home/zaijia001/usage.sh` 的两阶段流程。I1 先对原始人手视频做 Stage-1 抠除，正常 Stage-1 背景位于 `stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4`；I2 使用 E0 的 pure `zed_replay.mp4` 作为 robot video，把机械臂贴回同一个背景。默认处理三个任务 `id0-id10`。

注意：I1 只生成手抠除背景，`run_human_robot_inpaint_repaint.py` 仍要求传入 `--robot_video`，所以这里用一个已存在的 dummy robot 视频满足脚本参数；I1 的 Stage-1 背景不依赖该 dummy robot 是否对应同一个 task/id。I2 才要求每个 task/id 的 E0 pure robot 视频实际存在。

### I1. 三个任务：只做人手抠除背景缓存

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; DUMMY_ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/pick_diverse_bottles/id0_z005/zed_replay.mp4; [[ -f "$DUMMY_ROBOT" ]] || DUMMY_ROBOT=$(find /home/zaijia001/ssd/RoboTwin/code_painting/human_replay /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay -path '*id*' -name zed_replay.mp4 2>/dev/null | sort | head -n 1); [[ -f "$DUMMY_ROBOT" ]] || { echo "[error] no robot_video found; run E0 or E2.0 first"; exit 1; }; echo "[stage1] dummy robot_video=$DUMMY_ROBOT"; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 10 120); do HUMAN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input/rgb_${ID}.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; [[ -f "$HUMAN" ]] || { echo "[skip] task=${TASK} id=${ID} missing HUMAN=$HUMAN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --human_video "$HUMAN" --robot_video "$DUMMY_ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --human_save_debug_artifacts 0 --robot_save_removed_video 0 --robot_save_mask_artifacts 0 --robot_save_debug_videos 0 --robot_save_composite_video 0; done; done
```

输出检查：

```bash
find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1 -path '*/stage1_human_inpaint/removed_w_mask_*.mp4' | sort
```

### I2. 三个任务：把 E0 pure replay 贴回 I1 背景

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 0 10); do BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK}/id${ID}_z005/zed_replay.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/${TASK}/id_${ID}; [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG under ${BG_ROOT}/stage1_human_inpaint; run I1 first"; continue; }; [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing pure ROBOT=$ROBOT; run E0 for this id first"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1; done; done

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 40 80); do BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK}/id${ID}_z005/zed_replay.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/${TASK}/id_${ID}; [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG under ${BG_ROOT}/stage1_human_inpaint; run I1 first"; continue; }; [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing pure ROBOT=$ROBOT; run E0 for this id first"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1; done; done

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 80 120); do BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK}/id${ID}_z005/zed_replay.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/${TASK}/id_${ID}; [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG under ${BG_ROOT}/stage1_human_inpaint; run I1 first"; continue; }; [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing pure ROBOT=$ROBOT; run E0 for this id first"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1; done; done
```

输出检查：

```bash
find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot -type f \( -name '*target_with_original*.mp4' -o -name '*repaint*.mp4' -o -name '*.mp4' \) | sort | head -n 80
```

### I3. D435 视角：把 E2.4 pure replay 贴回 I1 背景

用途：沿用 I1 的真实人手 D435 背景，但机器人视频改用 E2.4 的 `h2_pure_d435` 输出。输出根目录使用 `results_repaint_piper_h2_d435`，文件路径中保留 `d435`，便于和默认广角 replay/repaint 并列比较。

#### I3.0 三套 D435 robot repaint 指令区别

| 小节 | 入口 | 后端 | 初始化逻辑 | 适用场景 | 输出根目录 |
| --- | --- | --- | --- | --- | --- |
| I3 | `inpainting_sam2_robot/run_human_robot_inpaint_repaint.py` | 原 SAM2 环境 | 固定第 0 帧 | 和旧版 SAM2 流程对齐 | `results_repaint_piper_h2_d435` |
| I3.3 | `inpainting_sam3_robot/remove_anything_video_sam3_robot.py` | SAM3 项目自动后端；本机当前 fallback 到 SAM2/DINO2 | 固定第 0 帧 | 对比 SAM3 项目目录下的旧逻辑 | `results_repaint_piper_h2_d435_sam3` |
| I3.4 | `inpainting_sam3_robot/remove_anything_video_sam3_robot_visible_reinit.py` | SAM3 项目自动后端；本机当前 fallback 到 SAM2/DINO2 | 第一帧可见 robot 才初始化，lost 后可重初始化 | D435 首帧/中途无 robot 的主要推荐路径 | `results_repaint_piper_h2_d435_sam3_visible_reinit` |

本机目前只检测到 `/home/zaijia001/ssd/inpainting_sam2_robot/Grounded_SAM_2`，没有检测到 `Grounded_SAM_3` 目录。因此 I3.3/I3.4 在当前环境日志中会显示 `[backend] SAM=sam2, DINO=dino2`。如果后续安装了真正的 `Grounded_SAM_3`，可在命令前加 `export GROUNDED_SAM3_DIR=/path/to/Grounded_SAM_3`，启动日志显示 `[backend] SAM=sam3, DINO=dino3` 时才是真正 SAM3/DINO3。

#### I3.0.1 原 SAM2 首帧初始化：单 id 调试命令

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; TASK=pick_diverse_bottles; ID=0; BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005/zed_replay_d435.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/${TASK}/id_${ID}_d435; [[ -f "$BG" ]] || { echo "[skip] missing BG under ${BG_ROOT}"; exit 1; }; [[ -f "$ROBOT" ]] || { echo "[skip] missing D435 ROBOT=$ROBOT"; exit 1; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1
```

#### I3.0.2 SAM3 项目首帧初始化：单 id 调试命令

当前本机这条命令会 fallback 到 SAM2/DINO2；如果要强制使用真正 SAM3，需要先安装 `Grounded_SAM_3` 并设置 `GROUNDED_SAM3_DIR`。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && GPU=3; FPS=5; TASK=pick_diverse_bottles; ID=0; LEGACY=/home/zaijia001/ssd/inpainting_sam2_robot; BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005/zed_replay_d435.mp4; OUT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3/e0_robot/${TASK}/id_${ID}_d435; [[ -f "$BG" ]] || { echo "[skip] missing BG under ${BG_ROOT}"; exit 1; }; [[ -f "$ROBOT" ]] || { echo "[skip] missing D435 ROBOT=$ROBOT"; exit 1; }; CUDA_VISIBLE_DEVICES=${GPU} python remove_anything_video_sam3_robot.py --input_video "$ROBOT" --target_video "$BG" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --dilate_kernel_size 0 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --mask_idx 0 --fps ${FPS} --device cuda --sam_ckpt ${LEGACY}/pretrained_models/sam_vit_h_4b8939.pth --lama_config ${LEGACY}/lama/configs/prediction/default.yaml --lama_ckpt ${LEGACY}/pretrained_models/big-lama --tracker_ckpt vitb_384_mae_ce_32x4_ep300 --vi_ckpt ${LEGACY}/pretrained_models/sttn.pth --save_removed_video 0 --save_mask_frames 1 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1 && cp "$OUT/target_with_original_zed_replay_d435.mp4" "$OUT/final_repainted.mp4"
```

真正 SAM3 backend 模板：

```bash
export GROUNDED_SAM3_DIR=/path/to/Grounded_SAM_3
# 然后运行 I3.0.2 或 I3.4 命令；确认启动日志为 [backend] SAM=sam3, DINO=dino3
```

#### I3.0.3 新逻辑可见帧重初始化：单 id 调试命令

当前本机这条命令会使用 SAM2/DINO2 backend，但逻辑是 I3.4 的“可见帧初始化 + lost 后重初始化”，不是旧的第 0 帧固定初始化。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && GPU=3; FPS=5; TASK=pick_diverse_bottles; ID=0; BG=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}/stage1_human_inpaint/removed_w_mask_rgb_${ID}.mp4; ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005/zed_replay_d435.mp4; OUT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK}/id_${ID}_d435; CUDA_VISIBLE_DEVICES=${GPU} python remove_anything_video_sam3_robot_visible_reinit.py --input_video "$ROBOT" --target_video "$BG" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --init_policy first_visible --reinit_policy on_lost --detector_stride 1 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --fps ${FPS} --device cuda --save_mask_frames 1 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1
```

#### I3.0.4 新逻辑可见帧重初始化：批处理复用 checkpoint

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && CUDA_VISIBLE_DEVICES=0 python batch_visible_reinit_d435_repaint.py --tasks pick_diverse_bottles place_bread_basket stack_cups --id_start 10 --id_end 40 --fps 5 --device cuda --init_policy first_visible --reinit_policy on_lost --detector_stride 1 --redetect_every_n 0 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --save_removed_video 0 --save_mask_frames 0 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1 --overwrite 0 --continue_on_error 1

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && CUDA_VISIBLE_DEVICES=3 python batch_visible_reinit_d435_repaint.py --tasks pick_diverse_bottles place_bread_basket stack_cups --id_start 70 --id_end 100 --fps 5 --device cuda --init_policy first_visible --reinit_policy on_lost --detector_stride 1 --redetect_every_n 0 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --save_removed_video 0 --save_mask_frames 0 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1 --overwrite 0 --continue_on_error 1

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && CUDA_VISIBLE_DEVICES=1 python batch_visible_reinit_d435_repaint.py --tasks pick_diverse_bottles place_bread_basket stack_cups --id_start 100 --id_end 120 --fps 5 --device cuda --init_policy first_visible --reinit_policy on_lost --detector_stride 1 --redetect_every_n 0 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --save_removed_video 0 --save_mask_frames 0 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1 --overwrite 0 --continue_on_error 1
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 0 30); do BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005/zed_replay_d435.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/${TASK}/id_${ID}_d435; [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG under ${BG_ROOT}/stage1_human_inpaint; run I1 first"; continue; }; [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing D435 pure ROBOT=$ROBOT; run E2.4 first"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1; done; done

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 30 60); do BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005/zed_replay_d435.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/${TASK}/id_${ID}_d435; [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG under ${BG_ROOT}/stage1_human_inpaint; run I1 first"; continue; }; [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing D435 pure ROBOT=$ROBOT; run E2.4 first"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1; done; done

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 60 90); do BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005/zed_replay_d435.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/${TASK}/id_${ID}_d435; [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG under ${BG_ROOT}/stage1_human_inpaint; run I1 first"; continue; }; [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing D435 pure ROBOT=$ROBOT; run E2.4 first"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1; done; done

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 90 120); do BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005/zed_replay_d435.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/${TASK}/id_${ID}_d435; [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG under ${BG_ROOT}/stage1_human_inpaint; run I1 first"; continue; }; [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing D435 pure ROBOT=$ROBOT; run E2.4 first"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1; done; done
```

输出检查：

```bash
find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot -type f \( -name '*target_with_original*.mp4' -o -name '*repaint*.mp4' -o -name '*.mp4' \) | sort | head -n 80
```

#### I3.1 D435 Stage-2 异常诊断记录

问题现象：D435 版本 `results_repaint_piper_h2_d435/e0_robot/<TASK>/id_<ID>_d435/final_repainted.mp4` 不是只把机械臂贴到 I1 的人手抠除背景上，而是把 D435 robot replay 里的大片仿真白底/桌面背景也贴回去了。

确认过 I3 的 `BG` 路径本身没有选错。以 `pick_diverse_bottles id0` 为例，`pipeline_meta.json` 中 `stage1_output` 是：

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/pick_diverse_bottles/id_0/stage1_human_inpaint/removed_w_mask_rgb_0.mp4
```

真正出错的位置在 Stage-2 robot mask。`run_human_robot_inpaint_repaint.py` 会把 `--robot_video` 作为 `remove_anything_video_sam2_robot.py --input_video`，把 `--stage1_bg_video` 作为 `--target_video`；合成时不是直接 alpha overlay 整个 robot 视频，而是在 robot mask 区域执行：

```python
target_frame[mask_3ch] = orig_frame_resized[mask_3ch]
```

因此只要 D435 robot mask 把仿真背景也选进去，最终就会把 `zed_replay_d435.mp4` 中的白底/桌面背景一起拷贝到 Stage-1 BG 上。

id0 对比结论：

```text
原始人手视频: 640x480, 约 3.565s, 106 frames
I1 Stage-1 BG: 640x480, 约 3.565s, 106 frames
旧版 robot replay: 640x360, 21.2s, 106 frames
D435 robot replay: 640x480, 21.2s, 106 frames
旧版 final: 640x480, 21.2s, 106 frames
D435 final: 640x480, 21.2s, 106 frames
```

所以“人手视频是不是 640x480”的答案是：是，原始 head D435 人手视频是 `640x480`；D435 replay 也是 `640x480`。旧版非 D435 replay 是 `640x360`，但 Stage-2 合成时会 resize 到 target BG 的 `640x480`。D435 出错不是因为 BG 尺寸不对，而是因为 D435 窄视角下机器人在画面里更大、更贴边，原来适配旧版 `640x360/fovy=90` 的 `--robot_text_prompt`、`--robot_box_threshold 0.20`、`--robot_text_threshold 0.20`、`--mask_idx 2` 不再稳定，GroundingDINO/SAM2 在 D435 robot 源上选到了包含大块背景的 box/mask。

复查 id0 输入/输出元数据：

```bash
for F in /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_input/rgb_0.mp4 /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/pick_diverse_bottles/id_0/stage1_human_inpaint/removed_w_mask_rgb_0.mp4 /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/pick_diverse_bottles/id0_z005/zed_replay.mp4 /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/pick_diverse_bottles/id0_d435_z005/zed_replay_d435.mp4 /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/pick_diverse_bottles/id_0/final_repainted.mp4 /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/pick_diverse_bottles/id_0_d435/final_repainted.mp4; do echo "==== $F"; ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,duration,avg_frame_rate -of default=noprint_wrappers=1 "$F" 2>/dev/null || true; done
```

生成旧版/D435 的 Stage-2 mask 对比图：

```bash
TMP=/tmp/d435_i3_probe_id0_f30; rm -rf "$TMP"; mkdir -p "$TMP"; extract(){ ffmpeg -y -v error -i "$1" -vf "select='eq(n,30)'" -frames:v 1 "$2" || ffmpeg -y -v error -i "$1" -frames:v 1 "$2"; }; extract /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_input/rgb_0.mp4 "$TMP/01_human.png"; extract /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/pick_diverse_bottles/id_0/stage1_human_inpaint/removed_w_mask_rgb_0.mp4 "$TMP/02_bg.png"; extract /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/pick_diverse_bottles/id0_z005/zed_replay.mp4 "$TMP/03_old_robot.png"; extract /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/pick_diverse_bottles/id_0/stage2_robot_repaint/w_mask_zed_replay.mp4 "$TMP/04_old_wmask.png"; extract /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/pick_diverse_bottles/id_0/stage2_robot_repaint/w_box_zed_replay.mp4 "$TMP/05_old_wbox.png"; extract /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/pick_diverse_bottles/id_0/final_repainted.mp4 "$TMP/06_old_final.png"; extract /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/pick_diverse_bottles/id0_d435_z005/zed_replay_d435.mp4 "$TMP/07_d435_robot.png"; extract /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/pick_diverse_bottles/id_0_d435/stage2_robot_repaint/w_mask_zed_replay_d435.mp4 "$TMP/08_d435_wmask.png"; extract /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/pick_diverse_bottles/id_0_d435/stage2_robot_repaint/w_box_zed_replay_d435.mp4 "$TMP/09_d435_wbox.png"; extract /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/pick_diverse_bottles/id_0_d435/final_repainted.mp4 "$TMP/10_d435_final.png"; for f in "$TMP"/*.png; do base=$(basename "$f" .png); ffmpeg -y -v error -i "$f" -vf "scale=320:240,drawbox=x=0:y=0:w=190:h=26:color=black@0.65:t=fill,drawtext=text='${base}':x=6:y=6:fontsize=16:fontcolor=yellow" "$TMP/${base}_lbl.jpg"; done; ffmpeg -y -v error -i "$TMP/01_human_lbl.jpg" -i "$TMP/02_bg_lbl.jpg" -i "$TMP/03_old_robot_lbl.jpg" -i "$TMP/04_old_wmask_lbl.jpg" -i "$TMP/05_old_wbox_lbl.jpg" -i "$TMP/06_old_final_lbl.jpg" -i "$TMP/07_d435_robot_lbl.jpg" -i "$TMP/08_d435_wmask_lbl.jpg" -i "$TMP/09_d435_wbox_lbl.jpg" -i "$TMP/10_d435_final_lbl.jpg" -filter_complex "[0][1][2][3][4]hstack=inputs=5[top];[5][6][7][8][9]hstack=inputs=5[bot];[top][bot]vstack=inputs=2" "$TMP/contact_labeled.jpg"; echo "$TMP/contact_labeled.jpg"
```

读图规则：旧版 `04_old_wmask/05_old_wbox` 主要覆盖机械臂和夹爪，`06_old_final` 因此只把机器人贴到人手抠除背景；D435 `08_d435_wmask/09_d435_wbox` 会覆盖大片仿真背景，`10_d435_final` 就会把这些背景一起贴回去。

后续修复方向：不要直接复用旧版 Stage-2 参数跑 D435。优先为 D435 单独调 `--robot_text_prompt`、`--robot_box_threshold`、`--robot_text_threshold`、`--max_selected_boxes`、`--arm_split_ratio`、`--mask_idx`，或增加对 robot-source 背景颜色/深度的几何 mask 约束，再批量跑 I3。

#### I3.2 D435 首帧无机器人/误检背景的参数建议

进一步确认的根因：D435 的竖直 FOV 从旧 replay 的 `90 deg` 缩到真实 head D435 的约 `42.5 deg` 后，部分 id 的前几帧或个别帧里机器人没有进入画面，或者只露出很小一部分。当前 SAM2/SAM3 的 robot Stage-2 都是从第 0 帧初始化 mask，脚本里也固定 `key_frame_idx == 0`。如果第 0 帧没有机器人，GroundingDINO/SAM 就容易把仿真白底、桌面边缘或大块背景当成目标；后续 mask propagation 会沿着这个错误目标传播，所以最终看起来像“把 robot replay 的背景叠加到了人手 inpainting 背景上”，本质上是 robot mask 选错了。

最有效的处理不是单纯换 SAM3，而是保证 Stage-2 初始化帧里机器人可见：

- 优先方案：对 D435 robot replay 和对应 Stage-1 BG 同步裁掉前 N 帧，让 Stage-2 的第 0 帧就是机器人清楚可见的帧。当前脚本按帧列表合成，输出长度取 target/source 的较短长度；如果只裁 robot 不裁 BG，时序会偏移。
- 更稳的代码方向：给 Stage-2 加 `--key_frame_idx` 或 `--stage2_frame_start`，从机器人可见的帧初始化，然后从该帧开始传播/合成。当前 SAM2/SAM3 脚本还没有这个能力。
- replay 侧方案：生成 D435 pure replay 时让第 0 帧就能看到 Piper，例如减少进入画面前的空白帧，或在执行前加入几帧静止但可见的机器人 warm-up。

只靠参数可以降低误检背景概率，但不能解决“首帧完全没有机器人”的情况。D435 建议从下面这组更保守参数开始调：

```text
--robot_box_threshold 0.30~0.40
--robot_text_threshold 0.25~0.35
--robot_max_mask_area_ratio 0.20~0.35
--robot_text_prompt "robot arm, robotic gripper, robot wrist, robot forearm."
--robot_save_mask_artifacts 1
```

参数含义：

- `--robot_max_mask_area_ratio` 不要继续用 `1.0`。如果误选白底/桌面，首帧 mask 面积通常很大，设成 `0.20~0.35` 可以直接丢掉这类候选。
- `--robot_box_threshold/--robot_text_threshold` 从旧版 `0.20/0.20` 提高到 `0.30/0.30` 附近，减少弱匹配背景 box。
- `--robot_text_prompt` 尽量写机器人本体词，不要太宽泛；`end effector`、`forearm` 在部分画面中可能帮助召回，但也可能引入非机器人区域，D435 先用更短更明确的 prompt。
- `--mask_idx 2` 是旧广角 replay 的经验值，D435 下不稳定。必须看 `w_mask_*.mp4`、`w_box_*.mp4` 或保存 mask artifacts 后重新选。
- `--robot_erode_kernel_size`、`--robot_composite_erode_kernel_size` 只能收缩边界/减少白边，不能修复首帧选错目标。

快速找某个 id 的可见起始帧：

```bash
TASK=pick_diverse_bottles; ID=0; ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005/zed_replay_d435.mp4; TMP=/tmp/d435_visible_${TASK}_${ID}; rm -rf "$TMP"; mkdir -p "$TMP"; ffmpeg -y -v error -i "$ROBOT" -vf "select='not(mod(n,5))',scale=320:240,tile=5x5" -frames:v 1 "$TMP/contact_every5.jpg"; echo "$TMP/contact_every5.jpg"
```

#### I3.3 当前 SAM3 项目首帧初始化：直接复用 I1 背景做 D435 robot repaint

说明：`/home/zaijia001/ssd/inpainting_sam3_robot/run_human_robot_inpaint_repaint.py` 会重新跑 Stage-1 human inpainting，当前没有 `--stage1_bg_video`/`--reuse_stage1` 参数；如果目标是沿用 I1 已经生成的人手抠除背景，直接调用 `remove_anything_video_sam3_robot.py` 更清楚。SAM3 版本也仍然从第 0 帧初始化，所以运行前要先确认 D435 robot replay 的第 0 帧能看到机器人；否则先同步裁剪 robot/BG 或修改代码支持非 0 初始化帧。

注意：这里的“当前 SAM3 项目”指入口目录是 `/home/zaijia001/ssd/inpainting_sam3_robot`。实际后端以启动日志为准：如果日志显示 `[backend] SAM=sam2, DINO=dino2`，说明当前环境没有使用真正的 SAM3/DINO3，而是 fallback 到 `Grounded_SAM_2` 的 SAM2/GroundingDINO2；如果日志显示 `[backend] SAM=sam3, DINO=dino3`，才是真正 SAM3/DINO3。

单条调试命令，默认 `pick_diverse_bottles id0`，输出到 `results_repaint_piper_h2_d435_sam3/e0_robot/...`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && GPU=3; FPS=5; TASK=pick_diverse_bottles; ID=0; LEGACY=/home/zaijia001/ssd/inpainting_sam2_robot; BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005/zed_replay_d435.mp4; OUT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3/e0_robot/${TASK}/id_${ID}_d435; [[ -f "$BG" ]] || { echo "[skip] missing BG under ${BG_ROOT}"; exit 1; }; [[ -f "$ROBOT" ]] || { echo "[skip] missing D435 ROBOT=$ROBOT"; exit 1; }; CUDA_VISIBLE_DEVICES=${GPU} python remove_anything_video_sam3_robot.py --input_video "$ROBOT" --target_video "$BG" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --dilate_kernel_size 0 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --mask_idx 0 --fps ${FPS} --device cuda --sam_ckpt ${LEGACY}/pretrained_models/sam_vit_h_4b8939.pth --lama_config ${LEGACY}/lama/configs/prediction/default.yaml --lama_ckpt ${LEGACY}/pretrained_models/big-lama --tracker_ckpt vitb_384_mae_ce_32x4_ep300 --vi_ckpt ${LEGACY}/pretrained_models/sttn.pth --save_removed_video 0 --save_mask_frames 1 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1 && cp "$OUT/target_with_original_zed_replay_d435.mp4" "$OUT/final_repainted.mp4"
```

如果首帧已经确认有机器人但仍误选背景，先看：

```bash
ls -lh /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3/e0_robot/pick_diverse_bottles/id_0_d435/*zed_replay_d435*.mp4
```

重点检查 `w_box_zed_replay_d435.mp4` 和 `w_mask_zed_replay_d435.mp4`。如果框/mask 仍覆盖白底，把 `--max_mask_area_ratio` 继续降到 `0.20~0.25`，并尝试 `--mask_idx 1/2/3`；如果完全没有机器人候选，说明必须换可见起始帧，不能继续靠阈值硬调。

#### I3.4 新逻辑：可见帧重初始化 SAM2/SAM3 模式

用途：这是给 D435 窄视角 replay 准备的新 Stage-2 模式。它不修改原来的 SAM2/SAM3 脚本和接口，而是新增独立脚本：

```text
/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot_visible_reinit.py
/home/zaijia001/ssd/inpainting_sam3_robot/batch_visible_reinit_d435_repaint.py
```

目标是解决“很难保证机械臂第 0 帧一定出现在视频里”的问题。单视频脚本用于调试一个 id；批处理脚本会在一个 Python 进程里只加载一次 DINO/SAM checkpoint，然后循环处理多个 task/id，避免每个视频重复加载模型。

命名关系：

- 原来 SAM2 指令：I3 的 `/home/zaijia001/ssd/inpainting_sam2_robot/run_human_robot_inpaint_repaint.py --stage1_bg_video ... --robot_video ... --reuse_stage1`，固定第 0 帧初始化。
- 当前 SAM3 项目指令：I3.3 的 `/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot.py --target_video ...`，也是固定第 0 帧初始化；实际是 SAM3 还是 fallback SAM2，要看日志 `[backend] SAM=...`。
- 新逻辑指令：I3.4 的 `remove_anything_video_sam3_robot_visible_reinit.py` / `batch_visible_reinit_d435_repaint.py`，不是固定第 0 帧，而是等待第一帧可见 robot 后初始化，并允许中途 lost 后重新初始化。

本机当前日志示例：`Grounded-SAM dir: /home/zaijia001/ssd/inpainting_sam2_robot/Grounded_SAM_2`、`[backend] SAM=sam2, DINO=dino2`，所以 I3.3/I3.4 在当前环境下实际跑的是 SAM2/GroundingDINO2 backend，只是入口脚本位于 `inpainting_sam3_robot`。

兼容修复：如果运行时报 `AttributeError: 'BertModel' object has no attribute 'get_head_mask'` 或 `TypeError: to() received an invalid combination of arguments - got (dtype=torch.device, )`，原因是旧 GroundingDINO 的 `BertModelWarper` 依赖 transformers 旧版 BERT helper API，但当前 `inpainting-sam3-dino3` 环境里的 transformers 版本已经移除 `get_head_mask`，并把 `get_extended_attention_mask` 第三个参数从旧版 `device` 改成新版 `dtype`。现在 `remove_anything_video_sam3_robot.py` 和 `remove_anything_video_sam3_robot_visible_reinit.py` 都会在自身入口中动态 patch 这两个 BERT helper，不修改第三方 Grounded_SAM 源码，也不改变原来的命令行接口。

当前 SAM Stage-2 和新模式的区别：

| 项目 | 当前 SAM2/SAM3 Stage-2 | 可见帧重初始化模式 |
| --- | --- | --- |
| 初始化帧 | 固定第 0 帧，脚本中约束 `key_frame_idx == 0` | 从第一帧开始逐帧检测，等检测到有效 robot 后再初始化 SAM |
| 第 0 帧无机器人 | 容易误检白底/桌面，错误 mask 会传播到后续帧 | 第 0 帧输出空 mask，不贴 robot；直到 robot 出现才开始贴 |
| 中间帧 robot 消失 | 仍沿着上一帧 mask 传播，容易漂到背景 | 判断为 lost 后输出空 mask，暂停传播 |
| robot 再次出现 | 旧 mask 可能已经漂移，难以自动恢复 | 再次通过 detector 找到 robot 后重新初始化 SAM |
| 主要代价 | 快，只需首帧检测/初始化 | 慢，需要按帧或按 stride 重跑 detector，并增加 mask 有效性判断 |

建议状态机：

```text
state = inactive
for frame t:
  candidates = detect_robot(frame[t])
  valid = select_valid_robot_candidate(candidates)

  if state == inactive:
    if valid:
      init_sam_on_frame(t, valid)
      mask[t] = sam_mask[t]
      state = tracking
    else:
      mask[t] = empty

  elif state == tracking:
    mask[t] = propagate_from_previous_valid_frame(t)
    if mask_invalid(mask[t]) or robot_detector_confirms_lost(frame[t], mask[t]):
      mask[t] = empty
      state = inactive
    else:
      keep tracking
```

关键判断：

- `select_valid_robot_candidate` 不应该只看最高分 box；需要同时检查 prompt 分数、box 面积、mask 面积、是否覆盖大块白底、是否超过 `max_mask_area_ratio`。
- `mask_invalid` 可以用面积过大/过小、mask bbox 突然跳变、连续帧 IoU 过低、mask 贴到纯白背景比例过高来判断。
- `robot_detector_confirms_lost` 建议允许 `lost_patience`，例如连续 2-3 帧没有有效 robot box 再切回 `inactive`，避免单帧 detector 漏检导致闪烁。
- 重初始化时不要继承已经漂移的旧 mask；一旦进入 `inactive`，后续只根据当前帧 detector 的有效 robot candidate 重新开始。
- 合成时 `inactive` 帧应输出空 robot mask，也就是只保留 Stage-1 人手抠除背景，不从 robot replay 拷贝任何像素。

单条调试命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && GPU=3; FPS=5; TASK=pick_diverse_bottles; ID=0; BG=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}/stage1_human_inpaint/removed_w_mask_rgb_${ID}.mp4; ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005/zed_replay_d435.mp4; OUT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK}/id_${ID}_d435; CUDA_VISIBLE_DEVICES=${GPU} python remove_anything_video_sam3_robot_visible_reinit.py --input_video "$ROBOT" --target_video "$BG" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --init_policy first_visible --reinit_policy on_lost --detector_stride 1 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --fps ${FPS} --device cuda --save_mask_frames 1 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1
```

批处理命令：一次加载模型，处理三个 task 的 id0-id120。`--overwrite 0` 会跳过已经存在的 `final_repainted.mp4`。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && CUDA_VISIBLE_DEVICES=3 python batch_visible_reinit_d435_repaint.py --tasks pick_diverse_bottles place_bread_basket stack_cups --id_start 0 --id_end 120 --fps 5 --device cuda --init_policy first_visible --reinit_policy on_lost --detector_stride 1 --redetect_every_n 0 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --save_removed_video 0 --save_mask_frames 0 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1 --overwrite 0 --continue_on_error 1
```

每个 id 的主输出包括 `target_with_original_zed_replay_d435.mp4` 和复制出的 `final_repainted.mp4`；调试输出包括 `mask_zed_replay_d435.mp4`、`w_mask_zed_replay_d435.mp4`、`w_box_zed_replay_d435.mp4`、`visible_reinit_meta.json`。

快速 dry-run 检查可用输入，不加载模型：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && python batch_visible_reinit_d435_repaint.py --tasks pick_diverse_bottles place_bread_basket stack_cups --id_start 0 --id_end 10 --dry_run
```

输出目录不覆盖当前 SAM2/SAM3 结果，便于 A/B 对比：

```text
当前输出:
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/<TASK>/id_<ID>_d435/
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3/e0_robot/<TASK>/id_<ID>_d435/

新模式输出:
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435/
```

对比时优先看这几类帧：

- 开头 robot 还没进入画面的帧：新模式应只显示 I1 背景，不应贴白底。
- robot 第一次进入画面的帧：新模式应从该帧开始初始化并贴 robot。
- robot 中途离开画面的帧：新模式应自动停止贴 robot，mask 变空。
- robot 重新进入画面的帧：新模式应重新检测并初始化，而不是沿用旧漂移 mask。

实现要点：

- `inactive` 状态按 `--detector_stride` 运行 GroundingDINO；只有 candidate 通过面积、白色背景比例、阈值检查后，才用 SAM image predictor 初始化 mask。
- `tracking` 状态不再依赖第 0 帧 video predictor，而是用上一帧有效 bbox 扩张后作为当前帧 SAM box prompt，符合“依据前一帧进行 SAM 检测”的逻辑。
- 当前帧 mask 如果过大、过小、白色背景比例过高，或者 bbox 中心跳变超过 `--max_bbox_jump_ratio`，则视为 lost。
- lost 后根据 `--empty_mask_when_lost 1` 输出空 mask，避免把 robot replay 背景拷贝到 Stage-1 BG；后续 frame 再次检测到 robot 时重新初始化。
- 批处理脚本复用同一个 `VisibleReinitRobotSegmenter` 实例，因此不会每个视频都重新加载 DINO/SAM checkpoint。

## J. AnyGrasp 候选筛选：找离人手朝向/目标物最近的候选

说明：三个任务的 AnyGrasp 结果位于 `/home/zaijia001/ssd/data/piper/hand/<TASK>/<TASK>_output/foundation_input_<ID>`。J0 先检查 id0-id10 需要的 AnyGrasp、Foundation replay、HaMeR NPZ 是否齐全；J1 生成 ranked preview 和 `summary.json`，其中 `orientation_rank` 更偏向“和人手局部轴接近”，`fused_rank` 同时考虑 AnyGrasp score 和人手朝向。

### J0. 三个任务：筛选可用 id0-id10 数据

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups; do REPORT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/${TASK}_availability_id0_10.txt; mkdir -p "$(dirname "$REPORT")"; : > "$REPORT"; for ID in $(seq 0 10); do A=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output/foundation_input_${ID}; R=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay/foundation_input_${ID}/head_anygrasp_frames; H=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; if [[ -d "$A/grasps" && -d "$R" && -f "$H" ]]; then echo "OK id=${ID} anygrasp=$A replay=$R hand=$H" | tee -a "$REPORT"; else echo "MISS id=${ID} anygrasp=$([[ -d "$A/grasps" ]] && echo 1 || echo 0) replay=$([[ -d "$R" ]] && echo 1 || echo 0) hand=$([[ -f "$H" ]] && echo 1 || echo 0)" | tee -a "$REPORT"; fi; done; done
```

### J1. 三个任务：生成和人手最接近的候选 preview/summary

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for TASK in pick_diverse_bottles place_bread_basket stack_cups; do case "$TASK" in pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle ;; place_bread_basket) LEFT_OBJ=basket; RIGHT_OBJ=bread ;; stack_cups) LEFT_OBJ=left_light_pink_cup; RIGHT_OBJ=right_dark_red_cup ;; esac; for ID in $(seq 0 10); do A=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output/foundation_input_${ID}; R=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay/foundation_input_${ID}; H=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; O=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/${TASK}/foundation_input_${ID}; [[ -d "$A/grasps" && -d "$R/head_anygrasp_frames" && -f "$H" ]] || { echo "[skip] task=${TASK} id=${ID} missing input"; continue; }; CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_anygrasp_ranked_preview.py --anygrasp_dir "$A" --replay_dir "$R" --hand_npz "$H" --base_image_dir "$R/head_anygrasp_frames" --base_image_mode raw --output_dir "$O" --frames 1 22 -10 --left_target_object "$LEFT_OBJ" --right_target_object "$RIGHT_OBJ" --anygrasp_score_weight 0.25 --orientation_score_weight 0.75 --max_rotation_distance_deg 90 --candidate_target_local_x_offset_m -0.05 --draw_object_overlay 1 --draw_hand_reference 1 --debug_dump_object_distances 1 --top_k 20 --camera_cv_axis_mode legacy_r1; done; done
```

输出检查：

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview -name summary.json | sort
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview -name '*orientation_rank.png' | sort | head -n 30
```

## K. AnyGrasp replay + SAM repaint：使用 J 的候选贴回人手背景

说明：K1 使用 J1 的 `summary.json` 选择候选，跑 Piper AnyGrasp 规划/执行预览；K2 把规划出的 `head_cam_plan.mp4` 贴回 I1 的人手抠除背景。若想看交互 viewer，在 K1 命令末尾把 `--enable_viewer 0 --viewer_wait_at_end 0` 改成 `--enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02`。

### K0. 人工筛选关键帧与废弃 bad id

用途：K1 使用 `--reuse_preview_frame_mode annotated_json_keyframes` 时，需要 preview summary 里带人工关键帧信息。旧的 `ffplay/mpv` 命令只是临时看视频，不会自动写标注；正式流程改为使用交互式标注脚本逐个打开 `hand_vis_gripper_*.mp4`，把结果写成 `hand_keyframes_all.json`。通常第一个关键帧选“手刚接近/准备抓取”，第二个关键帧选“动作目标/搬运目标”。

#### K0.1 交互标注关键帧与废弃视频

脚本入口：

- `/home/zaijia001/ssd/RoboTwin/code_painting/annotate_hand_keyframes.py`

按键逻辑：

- `Space`：把当前帧加入/移除整体关键帧，写入 `keyframes`
- `l` / `L`：把当前帧加入/移除左手关键帧，写入 `left_keyframes`
- `r`：把当前帧加入/移除右手关键帧，写入 `right_keyframes`
- `Left/Right`：逐帧前后移动
- `s`：播放/暂停
- `d`：把当前视频标记为 `reject`，再次按 `d` 恢复为 `in_progress`
- `R`：从当前视频第 0 帧重新播放
- `n`：保存当前视频并进入下一个；如果没有被 reject，会写成 `status=done`
- `p`：保存当前视频并回上一个
- `q` / `Esc`：保存退出

三个任务批量人工标注入口如下；每次只打开一个任务，建议先改 `TASK=...` 单独跑：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && TASK=pick_diverse_bottles; python /home/zaijia001/ssd/RoboTwin/code_painting/annotate_hand_keyframes.py --video-dir /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output --pattern 'hand_vis_gripper_*.mp4' --output-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --json-video-name-mode hand_vis --delay-ms 120

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && TASK=place_bread_basket; python /home/zaijia001/ssd/RoboTwin/code_painting/annotate_hand_keyframes.py --video-dir /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output --pattern 'hand_vis_gripper_*.mp4' --output-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --json-video-name-mode hand_vis --delay-ms 120

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && TASK=stack_cups; python /home/zaijia001/ssd/RoboTwin/code_painting/annotate_hand_keyframes.py --video-dir /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output --pattern 'hand_vis_gripper_*.mp4' --output-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --json-video-name-mode hand_vis --delay-ms 120


source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && TASK=handover_bottle; python /home/zaijia001/ssd/RoboTwin/code_painting/annotate_hand_keyframes.py --video-dir /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output --pattern 'hand_vis_gripper_*.mp4' --output-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --json-video-name-mode hand_vis --delay-ms 120

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && TASK=pnp_bread; python /home/zaijia001/ssd/RoboTwin/code_painting/annotate_hand_keyframes.py --video-dir /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output --pattern 'hand_vis_gripper_*.mp4' --output-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --json-video-name-mode hand_vis --delay-ms 120

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && TASK=pnp_tray; python /home/zaijia001/ssd/RoboTwin/code_painting/annotate_hand_keyframes.py --video-dir /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output --pattern 'hand_vis_gripper_*.mp4' --output-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --json-video-name-mode hand_vis --delay-ms 120
```

输出 JSON 结构和旧版保持兼容，核心字段是：

```json
{
  "videos": {
    "hand_vis_0.mp4": {
      "keyframes": [15, 31],
      "left_keyframes": [15],
      "right_keyframes": [31],
      "status": "done"
    },
    "hand_vis_3.mp4": {
      "keyframes": [],
      "status": "reject",
      "notes": "discarded_by_operator"
    }
  }
}
```

说明：`--json-video-name-mode hand_vis` 会把正在看的 `hand_vis_gripper_${ID}.mp4` 归一化保存成 `hand_vis_${ID}.mp4`。这是下游 `render_anygrasp_ranked_preview.py` 当前读取的 key 格式。

#### K0.2 用人工关键帧重跑 preview summary

下游逻辑：`render_anygrasp_ranked_preview.py --frame_selection_mode hand_keyframes_json` 会读取 `hand_keyframes_all.json`，把帧列表变成 `[0] + 标注关键帧`；其中 `0` 是上下文预览帧。planner 用 `--reuse_preview_frame_mode annotated_json_keyframes` 时，只取 `frame_selection.annotated_keyframes[:2]` 作为真正执行的两个关键帧。`run_render_anygrasp_ranked_preview_keyframes_batch.sh` 会自动跳过 `status=reject/discard/bad` 或少于两个关键帧的 id。

```bash
# 整任务批处理：旧版 d_pour_blue 使用同一个 wrapper；这里 H2O 目录名前缀是 foundation_input，所以加 VIDEO_PREFIX=foundation_input。
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for TASK in pick_diverse_bottles place_bread_basket stack_cups; do case "$TASK" in pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle ;; place_bread_basket) LEFT_OBJ=basket; RIGHT_OBJ=bread ;; stack_cups) LEFT_OBJ=left_light_pink_cup; RIGHT_OBJ=right_dark_red_cup ;; esac; ANN=/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json; [[ -f "$ANN" ]] || { echo "[skip] missing annotation $ANN"; continue; }; VIDEO_PREFIX=foundation_input CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh /home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output /home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/${TASK} --hand_keyframes_json "$ANN" --left_target_object "$LEFT_OBJ" --right_target_object "$RIGHT_OBJ" --anygrasp_score_weight 0.25 --orientation_score_weight 0.75 --max_rotation_distance_deg 90 --candidate_target_local_x_offset_m -0.05 --draw_object_overlay 1 --draw_hand_reference 1 --debug_dump_object_distances 1 --top_k 20 --camera_cv_axis_mode legacy_r1; done
```

查看 preview 输出：

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview -name summary.json | sort | head -n 30
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview -name '*orientation_rank.png' | sort | head -n 30
```

#### K0.3 可选：物理移动废弃人手数据

标注脚本里的 `status=reject` 已足够让 K0.2/K1 跳过坏 id。只有在你确认要把坏视频从原始目录移走时，才使用下面命令。默认 `APPLY=0` 只打印计划，不移动文件；确认无误后把 `APPLY=1`。这里只移动人手相关输入/输出，不动 FoundationPose/AnyGrasp/robot replay 结果，避免误删物体侧数据。

```bash
cat > /tmp/h2o_reject_ids.tsv <<'EOF'
# task id reason
pick_diverse_bottles 3 bad_hand_detection
place_bread_basket 5 bad_keyframes
stack_cups 7 bad_hand_motion
EOF

APPLY=0 python3 - <<'PY'
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

apply = os.environ.get("APPLY", "0") == "1"
tsv = Path("/tmp/h2o_reject_ids.tsv")
data_root = Path("/home/zaijia001/ssd/data/piper/hand")
reject_root = data_root / "_rejected_human_ids"
record = {"generated_at": datetime.now().isoformat(timespec="seconds"), "apply": apply, "items": []}

patterns = [
    ("harmer_input", "rgb_{id}.mp4"),
    ("harmer_output", "hand_detections_{id}.npz"),
    ("harmer_output", "hand_vis_{id}.mp4"),
    ("harmer_output", "hand_vis_gripper_{id}.mp4"),
]

for raw in tsv.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    task, sid, *rest = line.split(maxsplit=2)
    reason = rest[0] if rest else ""
    item = {"task": task, "id": int(sid), "reason": reason, "moved": []}
    for subdir, pattern in patterns:
        src = data_root / task / subdir / pattern.format(id=int(sid))
        if not src.exists():
            continue
        dst = reject_root / task / subdir / src.name
        item["moved"].append({"src": str(src), "dst": str(dst)})
        print(("[move]" if apply else "[dry-run]"), src, "->", dst)
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    record["items"].append(item)

reject_root.mkdir(parents=True, exist_ok=True)
json_path = reject_root / "rejected_ids.json"
old = {"history": []}
if json_path.exists():
    old = json.loads(json_path.read_text(encoding="utf-8"))
old.setdefault("history", []).append(record)
json_path.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[record] {json_path}")
PY
```

### K1. 三个任务：AnyGrasp 候选规划与 replay

```bash
# 安全续跑版：只规划已经存在 K0.2 preview summary 的 id；用 heredoc 生成 bash 脚本，避免 zsh 不支持 mapfile 或误用中文引号。
cat > /tmp/run_h2o_k1_preview_resume.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

for TASK in pick_diverse_bottles place_bread_basket stack_cups; do
  PREVIEW_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/${TASK}
  mapfile -t IDS < <(find "$PREVIEW_ROOT" -mindepth 2 -maxdepth 2 -name summary.json 2>/dev/null | sed -E 's#.*/foundation_input_([0-9]+)/summary.json#\1#' | sort -n)
  if ((${#IDS[@]} == 0)); then
    echo "[skip] task=${TASK} no preview summary under ${PREVIEW_ROOT}; run K0.2 first"
    continue
  fi
  echo "[K1] task=${TASK} ids=${IDS[*]}"

  case "$TASK" in
    pick_diverse_bottles)
      LEFT_OBJ=left_bottle
      RIGHT_OBJ=right_bottle
      MESH_ARGS=(--object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj)
      ;;
    place_bread_basket)
      LEFT_OBJ=basket
      RIGHT_OBJ=bread
      MESH_ARGS=(--object_mesh_override basket=/home/zaijia001/ssd/data/R1/hand/obj_mesh/basket/basket.obj --object_mesh_override bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_y/bread_y.obj)
      ;;
    stack_cups)
      LEFT_OBJ=left_light_pink_cup
      RIGHT_OBJ=right_dark_red_cup
      MESH_ARGS=(--object_mesh_override left_light_pink_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj --object_mesh_override right_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj)
      ;;
  esac

  CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_batch.sh \
    /home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output \
    /home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay \
    /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output \
    /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/${TASK} \
    --ids "${IDS[@]}" \
    --skip_existing 1 \
    --continue_on_error 1 \
    --reuse_preview_summary_root "$PREVIEW_ROOT" \
    --reuse_preview_frame_mode annotated_json_keyframes \
    --reuse_preview_candidate_group orientation \
    --reuse_preview_top_rank 1 \
    --arm auto \
    --execute_both_arms 1 \
    --planner_backend urdfik \
    --urdfik_trajectory_mode cartesian_interp_ik \
    --urdfik_cartesian_interp_steps 20 \
    --urdfik_cartesian_interp_auto_step_m 0.05 \
    --candidate_selection_mode planner \
    --left_target_object "$LEFT_OBJ" \
    --right_target_object "$RIGHT_OBJ" \
    --candidate_target_local_x_offset_m -0.05 \
    --approach_offset_m 0.20 \
    --reach_error_pose_source tcp \
    --replan_until_reached 1 \
    --replan_until_reached_max_attempts 3 \
    --save_debug_preview 1 \
    --save_debug_execution_preview 1 \
    --reach_pos_tol_m 0.03 \
    --reach_rot_tol_deg 20 \
    --pure_scene_output 1 \
    --overlay_text 0 \
    --debug_visualize_targets 0 \
    --debug_visualize_ik_waypoints 0 \
    --debug_visualize_cameras 0 \
    --debug_camera_axis_length 0.16 \
    --debug_camera_axis_thickness 0.006 \
    --target_local_forward_retreat_m 0.0 \
    --third_person_view 0 \
    --head_only 1 \
    --lighting_mode front_no_shadow \
    --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json \
    --camera_cv_axis_mode legacy_r1 \
    --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 \
    --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 \
    --enable_viewer 0 \
    --viewer_wait_at_end 0 \
    "${MESH_ARGS[@]}"
done
BASH
bash /tmp/run_h2o_k1_preview_resume.sh
```

输出检查：

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan -name head_cam_plan.mp4 | sort
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan -name plan_summary.json | sort
```

#### K1-debug：pick_diverse_bottles id 0-10 单任务调试（joint_interp 模式）

诊断结果（2026-05-22）：

**BUG 1（已修复）— DOF 不匹配崩溃**：`plan_anygrasp_keyframes_piper.py` 的 `HandRetargetPiperURDFIKRenderer`
继承自 R1 基类，后者在 `_plan_path_cartesian_interp_ik` 等函数中硬编码了 10-DOF
（`np.concatenate([self.torso_qpos, current_arm])` = 4 torso + 6 arm）。
Piper 只有 6-DOF（没有 torso），导致 IK solver 收到 size 10 的 seed 而崩溃：
`RuntimeError: shape '[-1, 1, 6]' is invalid for input of size 10`。

**为什么之前 human retarget 管线没看到**：human retarget 走的是
`render_hand_retarget_piper_dual_npz_urdfik.py` → `HandRetargetPiperDualURDFIKRenderer`
（从 `PiperDualReplayRenderer` 继承，自带正确的 6-DOF 方法）。
AnyGrasp 管线走 `plan_anygrasp_keyframes_piper.py` → R1 基类，是两条不同的代码路径。

修复内容（已写入 `plan_anygrasp_keyframes_piper.py`）：
1. 覆写 `_solve_ik_best_candidate`、`_plan_path_joint_interp`、`_plan_path_cartesian_interp_ik`、
   `_solution_error_to_ee_target` 四个方法，去掉 torso_qpos 拼接，全部改用 6-DOF
2. `DEFAULT_PIPER_INIT_ARM_JOINTS` 从全零 `[0,0,0,0,0,0]` 改为 `[0.0, 0.8, 1.2, 0.0, -0.4, 0.0]`
   （全零意味着机械臂完全伸展、末端朝天 z≈1.38m，离桌面物体非常远；新值来自 human retarget 管线，
   肘关节弯曲后末端更靠近桌面工作空间）

**与标准 K1 命令对比**——本调试版采用 `joint_interp` 模式：

| 参数 | 标准 K1 | 本调试版 | 原因 |
|------|---------|----------|------|
| `urdfik_trajectory_mode` | cartesian_interp_ik | **joint_interp** | 关节空间插值 = 中间路径全部可达，不会出现 Cartesian 路径点 IK 失败。与 human retarget 管线一致 |
| `urdfik_joint_interp_waypoints` | — | 20 | 更多分段 = 更平缓 |
| `approach_offset_m` | 0.20 | 0.12 | 降低预抓取后退 |
| `replan_max_attempts` | 3 | 3 | 不变（IK 收敛瓶颈不在重试次数） |
| `settle_steps` | 4 | 10 | 更长稳定等待 |
| `joint_target_wait_steps` | 60 | (default) | 使用默认值 |
| `save_pose_debug` | 0 | 1 | 输出位姿日志 |
| `debug_visualize_targets` | 0 | 1 | 渲染目标标记 |

```bash
# 先跑 id 0 单条验证，确认机器人能动起来且视频时长正常（应 > 3 秒）：
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py \
  --anygrasp_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/pick_diverse_bottles_output/foundation_input_0 \
  --replay_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_replay/foundation_input_0 \
  --hand_npz /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output/hand_detections_0.npz \
  --output_dir /tmp/k1_debug_joint_interp_id0 \
  --reuse_preview_summary_json /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/pick_diverse_bottles/foundation_input_0/summary.json \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --keyframes 38 78 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode joint_interp \
  --urdfik_joint_interp_waypoints 40 \
  --candidate_selection_mode planner \
  --left_target_object left_bottle \
  --right_target_object right_bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.12 \
  --reach_error_pose_source tcp \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 3 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --save_pose_debug 1 \
  --reach_pos_tol_m 0.03 \
  --reach_rot_tol_deg 20 \
  --settle_steps 100 \
  --joint_target_wait_steps 70 \
  --pure_scene_output 1 \
  --overlay_text 0 \
  --debug_visualize_targets 1 \
  --head_only 1 \
  --lighting_mode front_no_shadow \
  --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json \
  --camera_cv_axis_mode legacy_r1 \
  --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 \
  --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 \
  --enable_viewer 0 --viewer_wait_at_end 0 \
  --object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj \
  --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj

# id 0 验证通过后，批量跑 id 0-10：
cat > /tmp/run_h2o_k1_debug_bottles_0_10.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

TASK=pick_diverse_bottles
PREVIEW_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/${TASK}
IDS=($(seq 0 10))
LEFT_OBJ=left_bottle
RIGHT_OBJ=right_bottle
MESH_ARGS=(--object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj)

echo "[K1-debug] task=${TASK} ids=${IDS[*]} mode=joint_interp"

CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_batch.sh \
  /home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output \
  /home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay \
  /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/${TASK} \
  --ids "${IDS[@]}" \
  --skip_existing 0 \
  --continue_on_error 1 \
  --reuse_preview_summary_root "$PREVIEW_ROOT" \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode joint_interp \
  --urdfik_joint_interp_waypoints 20 \
  --candidate_selection_mode planner \
  --left_target_object "$LEFT_OBJ" \
  --right_target_object "$RIGHT_OBJ" \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.12 \
  --reach_error_pose_source tcp \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 3 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --save_pose_debug 1 \
  --reach_pos_tol_m 0.03 \
  --reach_rot_tol_deg 20 \
  --settle_steps 10 \
  --settle_steps 100 \
  --pure_scene_output 1 \
  --overlay_text 0 \
  --debug_visualize_targets 1 \
  --head_only 1 \
  --lighting_mode front_no_shadow \
  --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json \
  --camera_cv_axis_mode legacy_r1 \
  --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 \
  --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 \
  --enable_viewer 0 --viewer_wait_at_end 0 \
  "${MESH_ARGS[@]}"
BASH
bash /tmp/run_h2o_k1_debug_bottles_0_10.sh
```

输出检查：

```bash
# 视频时长（应 > 3 秒，远大于之前的 0.1~2.0 秒）
for i in $(seq 0 10); do
  echo -n "id=$i: "; ffprobe -v error -show_entries format=duration -of csv=p=0 "/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/pick_diverse_bottles/foundation_input_${i}/head_cam_plan.mp4" 2>/dev/null || echo "missing"
done
# 检查 IK 收敛失败
grep -l "Failed to converge\|execution_failed.*True" /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/pick_diverse_bottles/foundation_input_*/plan_summary.json 2>/dev/null
# 位姿日志行数
for i in $(seq 0 10); do
  f="/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/pick_diverse_bottles/foundation_input_${i}/pose_debug.jsonl"
  echo -n "id=$i lines: "; wc -l < "$f" 2>/dev/null || echo "missing"
done
# 各阶段距离误差
python3 -c "
import json, glob
for p in sorted(glob.glob('/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/pick_diverse_bottles/foundation_input_*/plan_summary.json')):
    d = json.load(open(p))
    stages = d.get('stages', {})
    fid = p.split('foundation_input_')[1].split('/')[0]
    for sname in ['pregrasp', 'grasp', 'action']:
        s = stages.get(sname, {})
        if s:
            print(f'id={fid} {sname}: reached={s.get(\"reached\")} pos_err={s.get(\"pos_err_m\",0):.3f}m rot_err={s.get(\"rot_err_deg\",0):.1f}deg')
" 2>/dev/null
```
### K2. 三个任务：把 AnyGrasp replay 贴回 I1 背景

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=0; FPS=5; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 0 10); do BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/${TASK}/foundation_input_${ID}/head_cam_plan.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/${TASK}/id_${ID}; [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG under ${BG_ROOT}/stage1_human_inpaint"; continue; }; [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing ROBOT=$ROBOT"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1; done; done
```

输出检查：

```bash
find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp -type f \( -name '*target_with_original*.mp4' -o -name '*repaint*.mp4' -o -name '*.mp4' \) | sort | head -n 80
```

## L. pi0 训练数据整理：原始人手、pure replay、AnyGrasp replay

这一节只记录“训练数据从哪里来、怎么转成 `policy/pi0/processed_data`、再怎么转成 LeRobot 数据集”。三类数据不要混用：

- 原始人手数据：真实人手录制视频，常见路径是 `/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_input/rgb_<ID>.mp4`，以及同任务下的 `origin/` 原始目录。
- 未使用 AnyGrasp 的机器人 replay：由人手 retarget 得到的 pure replay，动作来自 `world_targets_and_status.npz`，wrist 视角来自 `left_wrist_replay.mp4` 和 `right_wrist_replay.mp4`。
- AnyGrasp replay：动作来自 planner 的 `pose_debug.jsonl`，训练转换脚本还要求同一 planner episode 下存在 `left_wrist_cam_plan.mp4` 和 `right_wrist_cam_plan.mp4`。
- 三类训练转换都应先用 AnyGrasp 第一阶段人工标注结果过滤坏人手数据：`/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/<TASK>/hand_keyframes_all.json`。其中 `status=reject/discard/bad` 会被排除。

### L1. 原始人手 head + pure replay action/wrist 转训练

原始人手 head 视频检查：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups; do
  echo "===== ${TASK} human rgb ====="
  find /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input -maxdepth 1 -type f -name 'rgb_*.mp4' | sort | head -n 20
done
```

原始目录检查：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups; do
  echo "===== ${TASK} origin ====="
  find /home/zaijia001/ssd/data/piper/hand/${TASK}/origin -maxdepth 2 | sort | head -n 40
done
```

已经转换成 pi0 中间训练格式的数据在：

```bash
find /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data -maxdepth 3 -type f -name '*.hdf5' | sort | head -n 80
```

这里的“单纯人手数据”不是只用 `rgb_<ID>.mp4`，而是：

- head/cam_high：原始人手视频 `/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_input/rgb_<ID>.mp4`
- action/state：第二种 pure replay 的 `world_targets_and_status.npz`
- wrist：第二种 pure replay 的 `left_wrist_replay.mp4`、`right_wrist_replay.mp4`
- 过滤：`h2o_manual_review/<TASK>/hand_keyframes_all.json` 排除 `reject/discard/bad`

单任务转换到 pi0 `processed_data`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; INSTRUCTION="pick diverse bottles"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_human_head_pure_action" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input --head-dir-template '.' --head-video-name 'rgb_{id}.mp4' --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK} --retarget-dir-template 'id{id}_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-${N}
```

如果某个 `processed_data/<DATASET>` 已经存在，可以直接转成 LeRobot 数据集：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; N=120; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-${N}; uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir "$DATASET" --repo_id "local/h2o_${TASK}_human_head_pure_action" --task "pick diverse bottles"
```

说明：单独的原始人手 `rgb_<ID>.mp4` 不能直接变成 pi0 训练 HDF5，因为 pi0 HDF5 还需要 action/state 以及 wrist camera。L1 的 action/wrist 来自 L2 的 pure replay 计算方式。

### L2. 未使用 AnyGrasp：repaint 后 head + pure replay 机械臂数据转训练

默认广角 pure replay 的输入对应关系：

- head 视频：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/<TASK>/id_<ID>/final_repainted.mp4`
- 动作和 wrist：`/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/<TASK>/id<ID>_z005/`
- 必需文件：`world_targets_and_status.npz`、`left_wrist_replay.mp4`、`right_wrist_replay.mp4`

单任务转换到 pi0 `processed_data`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; INSTRUCTION="pick diverse bottles"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/${TASK} --head-dir-template 'id_{id}' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK} --retarget-dir-template 'id{id}_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-${N}
```

再转 LeRobot：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; N=120; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-${N}; uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir "$DATASET" --repo_id "local/h2o_${TASK}_pure_repaint" --task "pick diverse bottles"
```

D435 / visible-reinit 版本只替换 head 和 retarget 根目录模板：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; INSTRUCTION="pick diverse bottles"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_d435_visible_reinit" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK} --head-dir-template 'id_{id}_d435' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK} --retarget-dir-template 'id{id}_d435_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-${N}
```

### L3. AnyGrasp：repaint 后 head + planner 机械臂数据转训练

AnyGrasp 转训练时，head 视频可以用 K2 的 repaint 结果，但 action/state 必须来自 planner 的 `pose_debug.jsonl`，wrist 视频也应该来自同一个 planner episode。

期望输入路径：

- planner 原始 head：`/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/head_cam_plan.mp4`
- repaint 后训练 head：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/<TASK>/id_<ID>/final_repainted.mp4`
- planner action/state：`/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/pose_debug.jsonl`
- 训练转换还需要的 planner wrist：
  - `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/left_wrist_cam_plan.mp4`
  - `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/right_wrist_cam_plan.mp4`

如果 planner episode 已经补齐 `left_wrist_cam_plan.mp4` 和 `right_wrist_cam_plan.mp4`，用这个命令转成 pi0 `processed_data`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; INSTRUCTION="pick diverse bottles"; N=60; python scripts/process_repainted_planner_outputs.py "h2o_${TASK}_anygrasp_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/${TASK} --head-dir-template 'id_{id}' --head-video-name final_repainted.mp4 --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/${TASK} --planner-dir-template 'foundation_input_{id}' --left-wrist-video-name left_wrist_cam_plan.mp4 --right-wrist-video-name right_wrist_cam_plan.mp4 --pose-debug-name pose_debug.jsonl --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_anygrasp_repaint-${N}
```

再转 LeRobot：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; N=60; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_anygrasp_repaint-${N}; uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir "$DATASET" --repo_id "local/h2o_${TASK}_anygrasp_repaint" --task "pick diverse bottles"
```

### L4. 第 3 步目前还没处理好的地方：缺少 planner wrist 视频

当前检查命令：

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan -maxdepth 3 -type f \( -name 'head_cam_plan.mp4' -o -name 'pose_debug.jsonl' -o -name '*wrist*plan*.mp4' \) | sort | head -n 80
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan -maxdepth 3 -type f -name '*wrist*plan*.mp4' | wc -l
```

截至本次检查，`anygrasp_h2o_plan` 下能看到 `head_cam_plan.mp4` 和 `pose_debug.jsonl`，但没有 `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4`。因此 L3 的转换命令目前会因为缺少 wrist 输入而 skip episode 或最终报 `No usable episodes were processed`。

推荐补齐方式：重新运行或修改 AnyGrasp planner replay，让每个 episode 同步保存：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/head_cam_plan.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/left_wrist_cam_plan.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/right_wrist_cam_plan.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/pose_debug.jsonl
```

不推荐把 `h2_pure/<TASK>/id<ID>_z005/left_wrist_replay.mp4` 直接当作 AnyGrasp wrist，因为它对应的是人手 retarget pure replay，不是 AnyGrasp planner 的 `pose_debug.jsonl`，wrist 图像和 action/state 可能不对齐。临时调试可以这样做，但不建议作为最终训练数据。

### L5. 三个任务分别转换：L1 原始人手 head + pure replay action/wrist

用途：`cam_high` 使用真实人手原始视频 `rgb_<ID>.mp4`，action/state 与左右 wrist 使用 L2 pure replay 计算结果。三条命令都用 `hand_keyframes_all.json` 过滤 `reject/discard/bad`。

```bash
# pick_diverse_bottles
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; INSTRUCTION="pick diverse bottles"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_human_head_pure_action" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input --head-dir-template '.' --head-video-name 'rgb_{id}.mp4' --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK} --retarget-dir-template 'id{id}_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-${N}

# place_bread_basket
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=place_bread_basket; INSTRUCTION="place bread basket"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_human_head_pure_action" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input --head-dir-template '.' --head-video-name 'rgb_{id}.mp4' --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK} --retarget-dir-template 'id{id}_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-${N}

# stack_cups
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=stack_cups; INSTRUCTION="stack cups"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_human_head_pure_action" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input --head-dir-template '.' --head-video-name 'rgb_{id}.mp4' --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK} --retarget-dir-template 'id{id}_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-${N}
```

### L6. 三个任务分别转换：L2 repaint robot head + pure replay action/wrist

用途：`cam_high` 使用 Stage-2 最终合成的 `final_repainted.mp4`，action/state 与左右 wrist 使用同一条 pure replay 结果。

```bash
# pick_diverse_bottles
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; INSTRUCTION="pick diverse bottles"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/${TASK} --head-dir-template 'id_{id}' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK} --retarget-dir-template 'id{id}_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-${N}

# place_bread_basket
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=place_bread_basket; INSTRUCTION="place bread basket"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/${TASK} --head-dir-template 'id_{id}' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK} --retarget-dir-template 'id{id}_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-${N}

# stack_cups
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=stack_cups; INSTRUCTION="stack cups"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/${TASK} --head-dir-template 'id_{id}' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK} --retarget-dir-template 'id{id}_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-${N}
```

### L7. 检查和可视化 pi0 processed_data

统计每个数据集实际生成了多少个 episode：

```bash
for DATASET in h2o_pick_diverse_bottles_human_head_pure_action-120 h2o_place_bread_basket_human_head_pure_action-120 h2o_stack_cups_human_head_pure_action-120 h2o_pick_diverse_bottles_pure_repaint-120 h2o_place_bread_basket_pure_repaint-120 h2o_stack_cups_pure_repaint-120; do ROOT=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/${DATASET}; echo "===== ${DATASET} ====="; find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | sort | wc -l; done
```

检查 HDF5 结构、state/action 维度和三路相机帧数：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && python - <<'PY'
from pathlib import Path
import h5py

datasets = [
    "h2o_pick_diverse_bottles_human_head_pure_action-120",
    "h2o_place_bread_basket_human_head_pure_action-120",
    "h2o_stack_cups_human_head_pure_action-120",
    "h2o_pick_diverse_bottles_pure_repaint-120",
    "h2o_place_bread_basket_pure_repaint-120",
    "h2o_stack_cups_pure_repaint-120",
]
base = Path("/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data")
for name in datasets:
    files = sorted((base / name).glob("episode_*/episode_*.hdf5"))
    print(f"===== {name}: {len(files)} episodes =====")
    for p in files[:3]:
        with h5py.File(p, "r") as f:
            imgs = f["observations/images"]
            counts = {k: len(imgs[k]) for k in ["cam_high", "cam_left_wrist", "cam_right_wrist"]}
            print(p.name, "state", f["observations/state"].shape, "action", f["action"].shape, "images", counts)
PY
```

把一个 episode 的 `cam_high / cam_left_wrist / cam_right_wrist` 拼成 review mp4：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_pick_diverse_bottles_pure_repaint-120 && python scripts/visualize_processed_hdf5_episode.py --dataset-dir "$DATASET" --episode 0 --output-video "$DATASET/episode_0/review_cam_high_left_right.mp4" --fps 5 --max-frames 200
```

播放检查：

```bash
ffplay -autoexit /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_pick_diverse_bottles_pure_repaint-120/episode_0/review_cam_high_left_right.mp4
```

### L8. 三个任务分别转换：D435 visible-reinit head + D435 pure replay action/wrist

用途：`cam_high` 使用 D435 视角 visible-reinit 的 Stage-2 输出，action/state 与左右 wrist 使用 `h2_pure_d435`。如果某个 id 的 D435 visible-reinit 输出不存在，会被转换脚本 skip。

```bash
# pick_diverse_bottles
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; INSTRUCTION="pick diverse bottles"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_d435_visible_reinit" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK} --head-dir-template 'id_{id}_d435' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK} --retarget-dir-template 'id{id}_d435_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-${N}

# place_bread_basket
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=place_bread_basket; INSTRUCTION="place bread basket"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_d435_visible_reinit" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK} --head-dir-template 'id_{id}_d435' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK} --retarget-dir-template 'id{id}_d435_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-${N}

# stack_cups
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=stack_cups; INSTRUCTION="stack cups"; N=120; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_d435_visible_reinit" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK} --head-dir-template 'id_{id}_d435' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK} --retarget-dir-template 'id{id}_d435_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-${N}
```

### L9. 三个任务分别转换：AnyGrasp repaint head + planner action/wrist

前提：每个 planner episode 已经补齐 `left_wrist_cam_plan.mp4` 和 `right_wrist_cam_plan.mp4`。如果还没补齐，下面命令会逐 id skip，并最终可能报 `No usable episodes were processed`。

```bash
# pick_diverse_bottles
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; INSTRUCTION="pick diverse bottles"; N=60; python scripts/process_repainted_planner_outputs.py "h2o_${TASK}_anygrasp_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/${TASK} --head-dir-template 'id_{id}' --head-video-name final_repainted.mp4 --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/${TASK} --planner-dir-template 'foundation_input_{id}' --left-wrist-video-name left_wrist_cam_plan.mp4 --right-wrist-video-name right_wrist_cam_plan.mp4 --pose-debug-name pose_debug.jsonl --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_anygrasp_repaint-${N}

# place_bread_basket
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=place_bread_basket; INSTRUCTION="place bread basket"; N=60; python scripts/process_repainted_planner_outputs.py "h2o_${TASK}_anygrasp_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/${TASK} --head-dir-template 'id_{id}' --head-video-name final_repainted.mp4 --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/${TASK} --planner-dir-template 'foundation_input_{id}' --left-wrist-video-name left_wrist_cam_plan.mp4 --right-wrist-video-name right_wrist_cam_plan.mp4 --pose-debug-name pose_debug.jsonl --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_anygrasp_repaint-${N}

# stack_cups
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=stack_cups; INSTRUCTION="stack cups"; N=60; python scripts/process_repainted_planner_outputs.py "h2o_${TASK}_anygrasp_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/${TASK} --head-dir-template 'id_{id}' --head-video-name final_repainted.mp4 --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/${TASK} --planner-dir-template 'foundation_input_{id}' --left-wrist-video-name left_wrist_cam_plan.mp4 --right-wrist-video-name right_wrist_cam_plan.mp4 --pose-debug-name pose_debug.jsonl --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_anygrasp_repaint-${N}
```

L8/L9 的输出也用 L7 的统计、HDF5 结构检查和 `visualize_processed_hdf5_episode.py` 生成 review mp4。把 L7 里的 `DATASET` 替换成：

```text
h2o_<TASK>_pure_d435_visible_reinit-120
h2o_<TASK>_anygrasp_repaint-60
```

### L10. LeRobot 转换：3 种已可用数据模式 x 3 个任务

为什么前面有 4 种模式：

- L5：真实人手原始 head + pure replay action/wrist。用于保留真实人手视频画面，但 action 仍来自 pure replay 计算。
- L6：默认广角 robot repaint head + pure replay action/wrist。用于训练“默认广角模拟 robot head”。
- L8：D435 visible-reinit robot repaint head + D435 pure replay action/wrist。用于训练“更接近真实 D435 视角”的 robot head。
- L9：AnyGrasp repaint head + planner action/wrist。这个依赖 `left_wrist_cam_plan.mp4/right_wrist_cam_plan.mp4`，当前还没完整补齐，因此不放进下面 3x3 正式转换。

L6 和 L8 的核心区别：L6 使用默认广角 `h2_pure` 与 `results_repaint_piper_h2/e0_robot`；L8 使用 D435 窄视角 `h2_pure_d435` 与 `results_repaint_piper_h2_d435_sam3_visible_reinit`。两者 action/wrist 来源也分别对应各自 replay 根目录，不建议混用。

后续 HDF5 -> LeRobot 使用 `convert_aloha_data_to_lerobot_R1.py`，因为它读取的是当前 H2O processed HDF5 的 `/observations/state` 和 `/action`。命令显式加 `--use-wrist --mode video`，会把 `cam_high/cam_left_wrist/cam_right_wrist` 都写入 LeRobot dataset。

#### L10.1 L5：原始人手 head + pure replay action/wrist

```bash
# pick_diverse_bottles
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-120; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_human_head_pure_action" --task "pick up one bottle with one arm, and pick up another bottle with the other arm." --use-wrist --mode video

# place_bread_basket
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=place_bread_basket; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-120; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_human_head_pure_action" --task "place bread basket" --use-wrist --mode video

# stack_cups
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=stack_cups; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-120; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_human_head_pure_action" --task "stack cups" --use-wrist --mode video
```

#### [无意义]L10.2 L6：默认广角 robot repaint head + pure replay action/wrist

```bash
# pick_diverse_bottles
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-120; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_pure_repaint" --task "pick up one bottle with one arm, and pick up another bottle with the other arm." --use-wrist --mode video

# place_bread_basket
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=place_bread_basket; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-120; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_pure_repaint" --task "place bread basket" --use-wrist --mode video

# stack_cups
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=stack_cups; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-120; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_pure_repaint" --task "stack cups" --use-wrist --mode video
```

#### L10.3 L8：D435 visible-reinit robot repaint head + D435 pure replay action/wrist

```bash
# pick_diverse_bottles
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=pick_diverse_bottles; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-120; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_pure_d435_visible_reinit" --task "pick up one bottle with one arm, and pick up another bottle with the other arm." --use-wrist --mode video

# place_bread_basket
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=place_bread_basket; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-120; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_pure_d435_visible_reinit" --task "place bread basket" --use-wrist --mode video

# stack_cups
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=stack_cups; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-120; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_pure_d435_visible_reinit" --task "stack cups" --use-wrist --mode video
```

输出检查：LeRobot 数据默认写到 `$HF_LEROBOT_HOME/<repo-id>`。如果没有设置 `HF_LEROBOT_HOME`，按当前 LeRobot 包默认目录检查。

### L11. 从已生成 LeRobot 数据中抽取指定 episode 并重新编号

用途：不重新跑 HDF5 -> LeRobot 转换，直接从 `/home/zaijia001/.cache/huggingface/lerobot/local/<dataset>` 里抽取指定 episode，例如 `0,1-5,7`，输出到新的 repo-id。脚本会先按旧 episode id 去重并升序排序，再把新数据集连续重排为 `0..N-1`。

抽取前 25 个 episode：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_pick_diverse_bottles_human_head_pure_action --output-repo-id local/h2o_pick_diverse_bottles_human_head_pure_action_25ep --episodes '0-24' --overwrite


source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_place_bread_basket_human_head_pure_action --output-repo-id local/h2o_place_bread_basket_human_head_pure_action_25ep --episodes '0-24' --overwrite

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_stack_cups_human_head_pure_action --output-repo-id local/h2o_stack_cups_human_head_pure_action_25ep --episodes '0-24' --overwrite

zip -r human_ori_3task_25ep.zip *_25ep
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/human_ori_3task_25ep.zip  gdrive:piper/multi/3task/human_ori -P --drive-chunk-size 64M --transfers 4 --dry-run

```

按任意 id 列表抽取，例如 `0,1-5,7`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_pick_diverse_bottles_human_head_pure_action --output-repo-id local/h2o_pick_diverse_bottles_human_head_pure_action_subset --episodes '0,1-5,7' --overwrite
```

输出路径：

```text
/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action_25ep
```

检查新数据 episode 数量与重新编号：

```bash
ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action_25ep; python3 - <<'PY'
import json
from pathlib import Path
root = Path("/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action_25ep")
print(json.load(open(root / "meta/info.json"))["total_episodes"])
print((root / "meta/episodes.jsonl").read_text().splitlines()[:30])
print(sorted(p.name for p in (root / "data/chunk-000").glob("episode_*.parquet"))[:30])
PY
find "$ROOT/videos/chunk-000" -maxdepth 2 -type f -name 'episode_*.mp4' | sort | head -n 20
```

后续 3x3 其他数据集只需要替换 `--source` 和 `--output-repo-id`，例如：

```text
local/h2o_<TASK>_human_head_pure_action
local/h2o_<TASK>_pure_repaint
local/h2o_<TASK>_pure_d435_visible_reinit
```

### L12. 直接修改已生成 LeRobot cache 的 task 文本

用途：已经生成好的 LeRobot 数据不想重新转换时，直接把 `meta/tasks.jsonl` 和 `meta/episodes.jsonl` 里的旧任务名替换成新任务名。当前 parquet 里只存 `task_index=0`，不存任务文本，所以不需要改 parquet。

`pick_diverse_bottles` 示例：

```bash
ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action; OLD='pick diverse bottles'; NEW='pick up one bottle with one arm, and pick up another bottle with the other arm.'; cp "$ROOT/meta/tasks.jsonl" "$ROOT/meta/tasks.jsonl.bak" && cp "$ROOT/meta/episodes.jsonl" "$ROOT/meta/episodes.jsonl.bak" && OLD="$OLD" NEW="$NEW" perl -0pi -e 's/\Q$ENV{OLD}\E/$ENV{NEW}/g' "$ROOT/meta/tasks.jsonl" "$ROOT/meta/episodes.jsonl"
```

检查：

```bash
ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action; sed -n '1,5p' "$ROOT/meta/tasks.jsonl"; sed -n '1,5p' "$ROOT/meta/episodes.jsonl"
```

### L13. AnyGrasp Piper keyframe planner：pick_diverse_bottles id0-id10

用途：把单条 `foundation_input_0` 的 AnyGrasp Piper keyframe planner 命令扩展到 `id0-id10`。每个 id 自动替换 `anygrasp_dir`、`replay_dir`、`hand_npz`、`reuse_preview_summary_json` 和 `output_dir`。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && GPU=2; TASK=pick_diverse_bottles; for ID in $(seq 0 10); do ANYGRASP_DIR=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output/foundation_input_${ID}; REPLAY_DIR=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay/foundation_input_${ID}; HAND_NPZ=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; SUMMARY=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/${TASK}/foundation_input_${ID}/summary.json; OUT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_v3/id${ID}; [[ -d "$ANYGRASP_DIR" ]] || { echo "[skip] id=${ID} missing ANYGRASP_DIR=$ANYGRASP_DIR"; continue; }; [[ -d "$REPLAY_DIR" ]] || { echo "[skip] id=${ID} missing REPLAY_DIR=$REPLAY_DIR"; continue; }; [[ -f "$HAND_NPZ" ]] || { echo "[skip] id=${ID} missing HAND_NPZ=$HAND_NPZ"; continue; }; [[ -f "$SUMMARY" ]] || { echo "[skip] id=${ID} missing SUMMARY=$SUMMARY"; continue; }; echo "[run] id=${ID} out=${OUT}"; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py --anygrasp_dir "$ANYGRASP_DIR" --replay_dir "$REPLAY_DIR" --hand_npz "$HAND_NPZ" --output_dir "$OUT" --reuse_preview_summary_json "$SUMMARY" --reuse_preview_frame_mode annotated_json_keyframes --reuse_preview_candidate_group orientation --reuse_preview_top_rank 1 --keyframes 38 78 --arm auto --execute_both_arms 1 --planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik --urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.01 --candidate_selection_mode planner --left_target_object left_bottle --right_target_object right_bottle --candidate_target_local_x_offset_m -0.05 --approach_offset_m 0.12 --reach_error_pose_source tcp --replan_until_reached 1 --replan_until_reached_max_attempts 3 --save_debug_preview 1 --save_debug_execution_preview 1 --save_pose_debug 1 --debug_visualize_targets 1 --debug_visualize_ik_waypoints 1 --reach_pos_tol_m 0.03 --reach_rot_tol_deg 20 --settle_steps 100 --joint_target_wait_steps 100 --pure_scene_output 1 --overlay_text 0 --head_only 1 --lighting_mode front_no_shadow --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --enable_viewer 0 --viewer_wait_at_end 0 --object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj; done
```

检查输出：

```bash
for ID in $(seq 0 10); do OUT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_v3/id${ID}; echo "===== id${ID} ====="; ls -lh "$OUT"/head_cam_plan.mp4 "$OUT"/debug_execution_preview.mp4 "$OUT"/pose_debug.jsonl "$OUT"/plan_summary.json 2>/dev/null || true; python3 - "$OUT/plan_summary.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print("missing plan_summary")
    raise SystemExit
d = json.load(open(p))
print("execution_success:", d.get("execution_success"))
print("execution_failed:", d.get("execution_failed"))
print("failed_stage_records:", d.get("failed_stage_records"))
print("joint_target_wait_steps:", d.get("joint_target_wait_steps"))
print("settle_steps:", d.get("settle_steps"))
PY
done
```

注意：`--settle_steps` 和 `--joint_target_wait_steps` 主要用于推进 physics/controller 收敛，不等价于在输出视频里多写 100 帧。当前代码只在阶段完成后记录一帧最终状态；如果要让视频中明显停留，需要增加 `--hold_frames_after_stage`，如果要每个 waypoint 更充分收敛，可以增加 `--joint_command_scene_steps` 或 `--execute_interp_steps`。

### L13. Piper hand 原始 origin 数据可视化审核并移动 bad episode

用途：在 AnyGrasp / foundation 预处理之前，直接审核 `/home/zaijia001/ssd/data/piper/hand/<TASK>/origin/episode*` 下的原始 D435 RGB 帧。按 `b` 或 `d` 会把当前整个 episode 目录从 `origin/` 移到同级 `bad/`，因此后续处理不会再从 `origin/` 读到坏数据。按 `g`、`n` 或回车表示保留并进入下一个 episode。

脚本入口：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/review_piper_hand_origin.py
```

交互按键：

```text
b / d        标为 bad，并 mv 到 <TASK>/bad/
g / n /回车  保留当前 episode，进入下一个
p            回到上一个 episode
左右方向键    逐帧前后查看
空格 / s      暂停或继续播放
r            从当前 episode 第 0 帧重播
q / Esc      保存审核日志并退出
```

输出：

```text
/home/zaijia001/ssd/data/piper/hand/<TASK>/origin_bad_review.json
/home/zaijia001/ssd/data/piper/hand/<TASK>/bad/episode*
```

`pnp_tray`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task pnp_tray --delay-ms 5
```

`handover_bottle`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task handover_bottle --delay-ms 5
```

`pnp_bread`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task pnp_bread --delay-ms 5
```

只检查不移动目录：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task pnp_tray --dry-run
```

本次已按要求把 `handover_bottle/origin/episode*` 直接移动到 `handover_bottle/bad/`：

```bash
mkdir -p /home/zaijia001/ssd/data/piper/hand/handover_bottle/bad && shopt -s nullglob; for d in /home/zaijia001/ssd/data/piper/hand/handover_bottle/origin/episode*; do mv "$d" /home/zaijia001/ssd/data/piper/hand/handover_bottle/bad/; done
```

### L14. Piper AnyGrasp 两关键帧规划：使用 Piper dual base 和 Cartesian waypoint

用途：对 H2O Piper 数据使用 AnyGrasp rank preview 的两个关键帧候选做双臂执行。当前入口已经改为复用 Piper direct replay 的正确 dual renderer：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py
```

关键实现点：

- AnyGrasp Piper 入口现在使用 `PiperDualReplayRenderer` 和 `HandRetargetPiperDualURDFIKRenderer`。
- 左右臂分别保留 `robot_config_PiperPika_agx_dual_table_0515.json` 中的独立 base pose。
- IK 前按 arm 调用 `world_pose_to_base_pose_for_arm(...)`，避免 R1 单 base 逻辑把右臂目标转到错误坐标系。
- 建议使用 `cartesian_interp_ik`，让 TCP 在当前位姿和目标位姿之间生成中间 waypoint，并逐点求 IK；不要优先用 `joint_interp` 判断目标是否可达。

单条 id0 调试命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py \
  --anygrasp_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/pick_diverse_bottles_output/foundation_input_0 \
  --replay_dir /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_replay/foundation_input_0 \
  --hand_npz /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output/hand_detections_0.npz \
  --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_v3/1 \
  --reuse_preview_summary_json /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/pick_diverse_bottles/foundation_input_0/summary.json \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --keyframes 38 78 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps -1 \
  --urdfik_cartesian_interp_auto_step_m 0.01 \
  --candidate_selection_mode planner \
  --left_target_object left_bottle \
  --right_target_object right_bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.12 \
  --reach_error_pose_source tcp \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 3 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --save_pose_debug 1 \
  --debug_visualize_targets 1 \
  --debug_visualize_ik_waypoints 1 \
  --reach_pos_tol_m 0.03 \
  --reach_rot_tol_deg 20 \
  --settle_steps 100 \
  --joint_target_wait_steps 100 \
  --pure_scene_output 1 \
  --overlay_text 0 \
  --head_only 1 \
  --lighting_mode front_no_shadow \
  --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json \
  --camera_cv_axis_mode legacy_r1 \
  --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 \
  --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 \
  --enable_viewer 0 --viewer_wait_at_end 0 \
  --object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj \
  --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj
```

检查输出：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/piper_anygrasp_dual_base_cart_id0/plan_summary.json')
d = json.load(open(p))
print('execution_success:', d.get('execution_success'))
print('execution_failed:', d.get('execution_failed'))
print('failed_stage_records:', d.get('failed_stage_records'))
print('rank_preview_images:', d.get('rank_preview_images'))
PY
ls -lh /tmp/piper_anygrasp_dual_base_cart_id0/head_cam_plan.mp4 /tmp/piper_anygrasp_dual_base_cart_id0/debug_execution_preview.mp4
```

批量时保持同样参数，只需要按 id 替换四个输入路径和 `--reuse_preview_summary_json`。注意不要拿 `foundation_input_2/rank_previews` 判断 `foundation_input_0` 的执行结果；每个 id 的 `summary.json` 和关键帧不同。
