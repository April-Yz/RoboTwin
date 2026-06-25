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
> - Foundation O.1/O.1.1/O.1.2 已启用真实 wrist 手眼外参：左右 `gripper_T_camera` 分别写入 calibration bundle，仿真侧再使用 `piper_pika_agx` 平移净空 adapter；D/E 主 replay 仍主要使用 head camera。

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

# pnp_bread
CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_replay_d435 5 --skip_existing 0  --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json  --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right_bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_yr/bread_yerong.obj" --object "left_bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_nj/bread_niujiao.obj"
```


### C1.2. 六个 H2O 任务：D435 相机内参版 FoundationPose replay

说明：本节复用 C1 的 FoundationPose 结果、Piper 0515 head D435 外参和 `legacy_r1` 坐标转换，但把渲染相机内参改成和后面 E2.4 机器人 pure replay 一致的真实 D435 color 设置：`--image_width 640 --image_height 480 --fovy_deg 42.499880046655484`。输出目录统一使用 `foundation_replay_d435`，避免覆盖 C1 的默认广角/旧内参 replay。

```bash
CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_replay_d435 5 --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj" --object "left bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj"

cd /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles && zip -r pick_diverse_bottles_foundation_replay_d435.zip foundation_replay_d435

CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_replay_d435 5 --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "basket=/home/zaijia001/ssd/data/R1/hand/obj_mesh/basket/basket.obj" --object "bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_y/bread_y.obj"

cd /home/zaijia001/ssd/data/piper/hand/place_bread_basket && zip -r place_bread_basket_foundation_replay_d435.zip foundation_replay_d435

CUDA_VISIBLE_DEVICES=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_replay_d435 5 --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right dark red cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj" --object "left light pink cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj"

cd /home/zaijia001/ssd/data/piper/hand/stack_cups && zip -r stack_cups_foundation_replay_d435.zip foundation_replay_d435

CUDA_VISIBLE_DEVICES=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/handover_bottle/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/handover_bottle/foundation_replay_d435 5 --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj"

cd /home/zaijia001/ssd/data/piper/hand/handover_bottle && zip -r handover_bottle_foundation_replay_d435.zip foundation_replay_d435

CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_replay_d435 5 --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_yerong/bread_yerong.obj" --object "left bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_niujiao/bread_niujiao.obj"

CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/pnp_bread/foundation_replay_d435 5 --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_yr/bread_yerong.obj" --object "left bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_nj/bread_niujiao.obj"

cd /home/zaijia001/ssd/data/piper/hand/pnp_bread && zip -r pnp_bread_foundation_replay_d435.zip foundation_replay_d435

CUDA_VISIBLE_DEVICES=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_tray/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/pnp_tray/foundation_replay_d435 5 --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj" --object "left dark red cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj"

cd /home/zaijia001/ssd/data/piper/hand/pnp_tray && zip -r pnp_tray_foundation_replay_d435.zip foundation_replay_d435
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

#### new 3 task
```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && GPU=1; FPS=5; MAX_FRAMES=3000; RETREAT=0.05; for TASK in pnp_tray pnp_bread handover_bottle; do for ID in $(seq 0 30); do IN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; OUT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK}/id${ID}_d435_z005; [[ -f "$IN" ]] || { echo "[skip] missing $IN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz "$IN" --output_dir "$OUT" --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --fps ${FPS} --max_frames ${MAX_FRAMES} --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m ${RETREAT} --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 0 --debug_frame_limit -1 --debug_visualize_targets 0 --debug_visualize_cameras 0 --clean_output 1 --overlay_text_enable 0 --save_png_frames 0 --lighting_mode front_no_shadow; rm -f "$OUT"/zed_depth.mp4 "$OUT"/left_wrist_replay.mp4 "$OUT"/right_wrist_replay.mp4 "$OUT"/smooth_*.mp4; rm -rf "$OUT"/frames; for V in zed_replay third_replay; do [[ -f "$OUT/${V}.mp4" ]] || continue; ffmpeg -y -i "$OUT/${V}.mp4" -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT/${V}_d435.tmp.mp4" && mv "$OUT/${V}_d435.tmp.mp4" "$OUT/${V}_d435.mp4"; done; done; done
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

### I1.1 新三任务：只做人手抠除背景缓存

用途：给 `handover_bottle / pnp_bread / pnp_tray` 生成 Stage-1 人手抠除背景，供后续 I3.5 D435 robot visible-reinit repaint 使用。主输出位于：

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4
```

运行命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; DUMMY_ROBOT=$(find /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435 -path '*id*_d435_z005/zed_replay_d435.mp4' 2>/dev/null | sort | head -n 1); [[ -f "$DUMMY_ROBOT" ]] || { echo "[error] missing dummy D435 robot video"; exit 1; }; echo "[stage1] dummy robot_video=$DUMMY_ROBOT"; for TASK in handover_bottle pnp_bread pnp_tray; do for ID in $(seq 20 40); do HUMAN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input/rgb_${ID}.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; [[ -f "$HUMAN" ]] || { echo "[skip] task=${TASK} id=${ID} missing HUMAN=$HUMAN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --human_video "$HUMAN" --robot_video "$DUMMY_ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --human_save_debug_artifacts 0 --robot_save_removed_video 0 --robot_save_mask_artifacts 0 --robot_save_debug_videos 0 --robot_save_composite_video 0; done; done

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=2; FPS=5; DUMMY_ROBOT=$(find /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435 -path '*id*_d435_z005/zed_replay_d435.mp4' 2>/dev/null | sort | head -n 1); [[ -f "$DUMMY_ROBOT" ]] || { echo "[error] missing dummy D435 robot video"; exit 1; }; echo "[stage1] dummy robot_video=$DUMMY_ROBOT"; for TASK in handover_bottle pnp_bread pnp_tray; do for ID in $(seq 40 60); do HUMAN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input/rgb_${ID}.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; [[ -f "$HUMAN" ]] || { echo "[skip] task=${TASK} id=${ID} missing HUMAN=$HUMAN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --human_video "$HUMAN" --robot_video "$DUMMY_ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --human_save_debug_artifacts 0 --robot_save_removed_video 0 --robot_save_mask_artifacts 0 --robot_save_debug_videos 0 --robot_save_composite_video 0; done; done
```

检查：

```bash
for TASK in handover_bottle pnp_bread pnp_tray; do echo "===== ${TASK} ====="; find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK} -path '*/stage1_human_inpaint/removed_w_mask_*.mp4' 2>/dev/null | wc -l; done
```

#### I1.1.1 新三任务 Stage-1 缺失 id 补跑

用途：如果 I3.5/L8.2 发现新三任务只有少量 episode，先检查 Stage-1 背景是否缺失。`L8.2` 只会读取已经存在的 D435 repaint final，不会自动补 Stage-1 或 repaint。

当前检查逻辑：

```bash
for TASK in handover_bottle pnp_bread pnp_tray; do echo "===== ${TASK} ====="; echo -n "h2_pure_d435 ids: "; find /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK} -maxdepth 1 -type d -name 'id*_d435_z005' 2>/dev/null | wc -l; echo -n "stage1 BG: "; find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK} -path '*/stage1_human_inpaint/removed_w_mask_*.mp4' 2>/dev/null | wc -l; echo -n "D435 final: "; find /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK} -maxdepth 2 -type f -name final_repainted.mp4 2>/dev/null | wc -l; done
```

只补缺失 Stage-1 BG，不覆盖已存在输出：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=2; FPS=5; DUMMY_ROBOT=$(find /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435 -path '*id*_d435_z005/zed_replay_d435.mp4' 2>/dev/null | sort | head -n 1); [[ -f "$DUMMY_ROBOT" ]] || { echo "[error] missing dummy D435 robot video"; exit 1; }; for TASK in handover_bottle pnp_bread pnp_tray; do for ID in $(seq 0 80); do HUMAN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input/rgb_${ID}.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; EXISTING=$(find "$OUT/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | head -n 1); [[ -n "$EXISTING" ]] && { echo "[skip-existing] task=${TASK} id=${ID} BG=$EXISTING"; continue; }; [[ -f "$HUMAN" ]] || { echo "[skip] task=${TASK} id=${ID} missing HUMAN=$HUMAN"; continue; }; echo "[stage1-missing] task=${TASK} id=${ID}"; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --human_video "$HUMAN" --robot_video "$DUMMY_ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --human_save_debug_artifacts 0 --robot_save_removed_video 0 --robot_save_mask_artifacts 0 --robot_save_debug_videos 0 --robot_save_composite_video 0; done; done
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

### I3.5.1 新三任务：D435 visible-reinit robot repaint

用途：给 `handover_bottle / pnp_bread / pnp_tray` 生成 D435 robot replay 版本，用于和 L5.2 的真实人手 head baseline 对比。前提是 I1.1 已经生成 Stage-1 人手抠除背景，且 E2.4/new 3 task 已经生成 `h2_pure_d435/<TASK>/id<ID>_d435_z005/zed_replay_d435.mp4`。

d435 final 的来源：

```text
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435/final_repainted.mp4
```

它不是 L8.2 生成的。L8.2 只读取这个 `final_repainted.mp4` 并转换成 processed HDF5。这个 final 由下面的 `batch_visible_reinit_d435_repaint.py` 生成；单个视频内部由 `remove_anything_video_sam3_robot_visible_reinit.py` 生成 `target_with_original_zed_replay_d435.mp4`，再复制成 `final_repainted.mp4`。

虽然脚本目录叫 `inpainting_sam3_robot`，但当前本机没有 `Grounded_SAM_3`，启动日志会显示：

```text
[backend] SAM=sam2, DINO=dino2
[model] loading DINO once: ...
[model] loading SAM image predictor once: ...
```

因此当前这条批处理实际就是 SAM2/DINO2 后端，并且只加载一次 DINO/SAM checkpoint 后循环处理所有 task/id；不是每个视频都重新加载 ckpt。

先 dry-run 检查输入，不加载模型：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && python batch_visible_reinit_d435_repaint.py --tasks handover_bottle pnp_bread pnp_tray --id_start 0 --id_end 120 --dry_run
```

正式运行：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && CUDA_VISIBLE_DEVICES=1 python batch_visible_reinit_d435_repaint.py --tasks handover_bottle pnp_bread pnp_tray --id_start 0 --id_end 20 --fps 5 --device cuda --init_policy first_visible --reinit_policy on_lost --detector_stride 1 --redetect_every_n 0 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --save_removed_video 0 --save_mask_frames 0 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1 --overwrite 0 --continue_on_error 1
```

补跑缺失 id：`--overwrite 0` 会跳过已经有 `final_repainted.mp4` 的输出；如果某个 id 缺 Stage-1 BG，会继续 skip，所以先跑 I1.1.1。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && CUDA_VISIBLE_DEVICES=1 python batch_visible_reinit_d435_repaint.py --tasks handover_bottle pnp_bread pnp_tray --id_start 0 --id_end 80 --fps 5 --device cuda --init_policy first_visible --reinit_policy on_lost --detector_stride 1 --redetect_every_n 0 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --save_removed_video 0 --save_mask_frames 0 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1 --overwrite 0 --continue_on_error 1
```

如果目标只是先让三个任务都达到至少 25 个 `final_repainted.mp4`，当前已有 Stage-1 的候选已经够用，可以先直接跑下面这条 SAM2/DINO2 fallback 批处理命令；如果后续发现某些 id 因 Stage-1 缺失被 skip，再回到 I1.1.1 补 Stage-1。

当前已存在 final / 还需要补的数量大致是：

```text
handover_bottle: final=12，还需至少 13；已有 Stage-1 可补 id 4,5,6,7,8,10,12,17,18,21...
pnp_bread: final=2，还需至少 23；已有 Stage-1 可补 id 21,22,23,...,41,44,58
pnp_tray: final=14，还需至少 11；已有 Stage-1 可补 id 21,22,23,...,40,45,50
```

先补到至少 25 个 final 的命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam3-dino3 && cd /home/zaijia001/ssd/inpainting_sam3_robot && CUDA_VISIBLE_DEVICES=1 python batch_visible_reinit_d435_repaint.py --tasks handover_bottle pnp_bread pnp_tray --id_start 0 --id_end 60 --fps 5 --device cuda --init_policy first_visible --reinit_policy on_lost --detector_stride 1 --redetect_every_n 0 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." --box_threshold 0.35 --text_threshold 0.30 --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 --save_removed_video 0 --save_mask_frames 0 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1 --overwrite 0 --continue_on_error 1
```

检查输出：

```bash
for TASK in handover_bottle pnp_bread pnp_tray; do echo "===== ${TASK} ====="; find /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK} -maxdepth 2 -type f -name final_repainted.mp4 2>/dev/null | wc -l; done
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

### I3.5.2 L16 六任务：人手+物体 Stage-1 inpaint + robot/object prompt repaint debug（每任务 5 个 id）

用途：给 L16 Human Replay 的机器人+物体 repaint 准备真实背景。和 I1/I1.1 不同，本节 Stage-1 会把**人手/手臂 + 当前任务物体**一起抠除，避免后续把 L16 里的仿真物体贴回时和真实原始物体重影。输出目录不覆盖旧 I1/I3.5：

```text
Stage-1 BG:
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4

Stage-2 final:
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_visible_reinit/e0_robot_object/<TASK>/id_<ID>_l16/final_repainted.mp4
```

对应 L16 小节和任务物体：

| 任务 | 对应 L16 | debug id | Stage-1 物体 prompt |
| --- | --- | --- | --- |
| `pick_diverse_bottles` | L16.1 | `0 1 2 3 4` | `left bottle, right bottle, bottles` |
| `place_bread_basket` | L16.2 | `0 1 2 3 4` | `bread, basket` |
| `stack_cups` | L16.3 | `0 1 2 3 4` | `left light pink cup, right dark red cup` |
| `handover_bottle` | L16.4 | `1 2 3 4 5` | `right bottle, bottle` |
| `pnp_bread` | L16.5 | `7 8 9 10 11` | `bread` |
| `pnp_tray` | L16.6 | `0 1 2 3 4` | `left red cup, right bottle, cup, bottle` |

`pnp_bread` 先使用泛化 `bread`，因为真实画面中 `left/right bread` 未必比 `bread` 更稳；如果 debug 发现漏检，再把 prompt 改为 `left bread, right bread, bread.` 重跑对应 id。

`stack_cups` 不再追加泛化 `cups`，只使用 `left light pink cup, right dark red cup`，避免把绿色杯子也一起 inpaint。

直接运行：

```bash
bash <<'BASH'
set -eo pipefail

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

L16=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean
STAGE1=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object
OUTROOT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_visible_reinit/e0_robot_object
SAM2=/home/zaijia001/ssd/inpainting_sam2_robot
SAM3=/home/zaijia001/ssd/inpainting_sam3_robot
GPU_STAGE1=3
GPU_STAGE2=3
FPS=5

ids() {
  case "$1" in
    pick_diverse_bottles|place_bread_basket|stack_cups|pnp_tray) echo 0 1 2 3 4 ;;
    handover_bottle) echo 1 2 3 4 5 ;;
    pnp_bread) echo 7 8 9 10 11 ;;
  esac
}

human_prompt() {
  case "$1" in
    pick_diverse_bottles) echo "arms, hands, wrists, watch, left bottle, right bottle, bottles." ;;
    place_bread_basket) echo "arms, hands, wrists, watch, bread, basket." ;;
    stack_cups) echo "arms, hands, wrists, watch, left light pink cup, right dark red cup." ;;
    handover_bottle) echo "arms, hands, wrists, watch, right bottle, bottle." ;;
    pnp_bread) echo "arms, hands, wrists, watch, bread." ;;
    pnp_tray) echo "arms, hands, wrists, watch, left red cup, right bottle, cup, bottle." ;;
  esac
}

repaint_prompt() {
  case "$1" in
    pick_diverse_bottles) echo "robot arm, robotic gripper, robot wrist, robot forearm, left bottle, right bottle, bottles." ;;
    place_bread_basket) echo "robot arm, robotic gripper, robot wrist, robot forearm, bread, basket." ;;
    stack_cups) echo "robot arm, robotic gripper, robot wrist, robot forearm, left light pink cup, right dark red cup." ;;
    handover_bottle) echo "robot arm, robotic gripper, robot wrist, robot forearm, right bottle, bottle." ;;
    pnp_bread) echo "robot arm, robotic gripper, robot wrist, robot forearm, bread." ;;
    pnp_tray) echo "robot arm, robotic gripper, robot wrist, robot forearm, left red cup, right bottle, cup, bottle." ;;
  esac
}

conda activate inpainting-sam2-r1
cd "$SAM2"
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  for ID in $(ids "$TASK"); do
    HUMAN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input/rgb_${ID}.mp4
    S1OUT=$STAGE1/${TASK}/id_${ID}/stage1_human_inpaint
    BG=$S1OUT/removed_w_mask_rgb_${ID}.mp4
    [[ -f "$BG" ]] && { echo "[stage1 skip] task=${TASK} id=${ID} bg=${BG}"; continue; }
    [[ -f "$HUMAN" ]] || { echo "[stage1 skip] task=${TASK} id=${ID} missing HUMAN=${HUMAN}"; continue; }
    mkdir -p "$S1OUT"
    echo "[stage1 run] task=${TASK} id=${ID}"
    CUDA_VISIBLE_DEVICES=$GPU_STAGE1 python remove_anything_video_sam2.py \
      --input_video "$HUMAN" \
      --coords_type key_in --point_coords 10 80 --point_labels 1 \
      --dilate_kernel_size 100 \
      --text_prompt "$(human_prompt "$TASK")" \
      --box_threshold 0.35 --text_threshold 0.25 \
      --output_dir "$S1OUT" \
      --sam_ckpt "$SAM2/pretrained_models/sam_vit_h_4b8939.pth" \
      --lama_config "$SAM2/lama/configs/prediction/default.yaml" \
      --lama_ckpt "$SAM2/pretrained_models/big-lama" \
      --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
      --vi_ckpt "$SAM2/pretrained_models/sttn.pth" \
      --mask_idx 2 --fps $FPS --device cuda \
      --save_mask_frames 1 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1
  done
done

conda activate inpainting-sam3-dino3
cd "$SAM3"
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  for ID in $(ids "$TASK"); do
    BG=$STAGE1/${TASK}/id_${ID}/stage1_human_inpaint/removed_w_mask_rgb_${ID}.mp4
    ROBOT=$L16/${TASK}/foundation_input_${ID}/head_cam_plan.mp4
    OUT=$OUTROOT/${TASK}/id_${ID}_l16
    [[ -f "$BG" ]] || { echo "[repaint skip] task=${TASK} id=${ID} missing BG=${BG}"; continue; }
    [[ -f "$ROBOT" ]] || { echo "[repaint skip] task=${TASK} id=${ID} missing ROBOT=${ROBOT}"; continue; }
    echo "[repaint run] task=${TASK} id=${ID}"
    CUDA_VISIBLE_DEVICES=$GPU_STAGE2 python remove_anything_video_sam3_robot_visible_reinit.py \
      --input_video "$ROBOT" --target_video "$BG" --output_dir "$OUT" \
      --coords_type key_in --point_coords 10 80 --point_labels 1 \
      --init_policy first_visible --reinit_policy on_lost \
      --detector_stride 1 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 \
      --text_prompt "$(repaint_prompt "$TASK")" \
      --box_threshold 0.35 --text_threshold 0.30 \
      --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 \
      --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 \
      --fps $FPS --device cuda \
      --save_removed_video 0 --save_mask_frames 1 --save_mask_video 1 \
      --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1
  done
done
BASH
```

检查 debug 输出：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do echo "===== ${TASK} ====="; find /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_visible_reinit/e0_robot_object/${TASK} -maxdepth 2 -type f -name final_repainted.mp4 2>/dev/null | sort; done
```

重点查看每个 id 的：

```text
w_box_head_cam_plan.mp4
w_mask_head_cam_plan.mp4
final_repainted.mp4
```

### I3.5.3 L16 六任务：人手+物体 Stage-1 + L16 robot/object prompt repaint 批处理

用途：在 I3.5.2 debug 检查通过后，自动枚举 L16 中已经存在 `head_cam_plan.mp4` 的全部 id。`--overwrite` 逻辑由脚本中的 skip 实现：已有 Stage-1 BG 会跳过，已有 `final_repainted.mp4` 也会跳过；因此可以反复运行用于补齐缺失项。

直接运行：

```bash
bash <<'BASH'
set -eo pipefail

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

L16=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean
STAGE1=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object
OUTROOT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_visible_reinit/e0_robot_object
SAM2=/home/zaijia001/ssd/inpainting_sam2_robot
SAM3=/home/zaijia001/ssd/inpainting_sam3_robot
GPU_STAGE1=3
GPU_STAGE2=3
FPS=5

ids() {
  find "$L16/$1" -path '*/head_cam_plan.mp4' 2>/dev/null \
    | sed 's#.*/foundation_input_\([0-9]*\)/head_cam_plan.mp4#\1#' \
    | sort -n
}

human_prompt() {
  case "$1" in
    pick_diverse_bottles) echo "arms, hands, wrists, watch, left bottle, right bottle, bottles." ;;
    place_bread_basket) echo "arms, hands, wrists, watch, bread, basket." ;;
    stack_cups) echo "arms, hands, wrists, watch, left light pink cup, right dark red cup." ;;
    handover_bottle) echo "arms, hands, wrists, watch, right bottle, bottle." ;;
    pnp_bread) echo "arms, hands, wrists, watch, bread." ;;
    pnp_tray) echo "arms, hands, wrists, watch, left red cup, right bottle, cup, bottle." ;;
  esac
}

repaint_prompt() {
  case "$1" in
    pick_diverse_bottles) echo "robot arm, robotic gripper, robot wrist, robot forearm, left bottle, right bottle, bottles." ;;
    place_bread_basket) echo "robot arm, robotic gripper, robot wrist, robot forearm, bread, basket." ;;
    stack_cups) echo "robot arm, robotic gripper, robot wrist, robot forearm, left light pink cup, right dark red cup." ;;
    handover_bottle) echo "robot arm, robotic gripper, robot wrist, robot forearm, right bottle, bottle." ;;
    pnp_bread) echo "robot arm, robotic gripper, robot wrist, robot forearm, bread." ;;
    pnp_tray) echo "robot arm, robotic gripper, robot wrist, robot forearm, left red cup, right bottle, cup, bottle." ;;
  esac
}

conda activate inpainting-sam2-r1
cd "$SAM2"
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  for ID in $(ids "$TASK"); do
    HUMAN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input/rgb_${ID}.mp4
    S1OUT=$STAGE1/${TASK}/id_${ID}/stage1_human_inpaint
    BG=$S1OUT/removed_w_mask_rgb_${ID}.mp4
    [[ -f "$BG" ]] && { echo "[stage1 skip] task=${TASK} id=${ID} bg=${BG}"; continue; }
    [[ -f "$HUMAN" ]] || { echo "[stage1 skip] task=${TASK} id=${ID} missing HUMAN=${HUMAN}"; continue; }
    mkdir -p "$S1OUT"
    echo "[stage1 run] task=${TASK} id=${ID}"
    CUDA_VISIBLE_DEVICES=$GPU_STAGE1 python remove_anything_video_sam2.py \
      --input_video "$HUMAN" \
      --coords_type key_in --point_coords 10 80 --point_labels 1 \
      --dilate_kernel_size 100 \
      --text_prompt "$(human_prompt "$TASK")" \
      --box_threshold 0.35 --text_threshold 0.25 \
      --output_dir "$S1OUT" \
      --sam_ckpt "$SAM2/pretrained_models/sam_vit_h_4b8939.pth" \
      --lama_config "$SAM2/lama/configs/prediction/default.yaml" \
      --lama_ckpt "$SAM2/pretrained_models/big-lama" \
      --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
      --vi_ckpt "$SAM2/pretrained_models/sttn.pth" \
      --mask_idx 2 --fps $FPS --device cuda \
      --save_mask_frames 0 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1
  done
done

conda activate inpainting-sam3-dino3
cd "$SAM3"
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  for ID in $(ids "$TASK"); do
    BG=$STAGE1/${TASK}/id_${ID}/stage1_human_inpaint/removed_w_mask_rgb_${ID}.mp4
    ROBOT=$L16/${TASK}/foundation_input_${ID}/head_cam_plan.mp4
    OUT=$OUTROOT/${TASK}/id_${ID}_l16
    FINAL=$OUT/final_repainted.mp4
    [[ -f "$FINAL" ]] && { echo "[repaint skip] task=${TASK} id=${ID} final=${FINAL}"; continue; }
    [[ -f "$BG" ]] || { echo "[repaint skip] task=${TASK} id=${ID} missing BG=${BG}"; continue; }
    [[ -f "$ROBOT" ]] || { echo "[repaint skip] task=${TASK} id=${ID} missing ROBOT=${ROBOT}"; continue; }
    echo "[repaint run] task=${TASK} id=${ID}"
    CUDA_VISIBLE_DEVICES=$GPU_STAGE2 python remove_anything_video_sam3_robot_visible_reinit.py \
      --input_video "$ROBOT" --target_video "$BG" --output_dir "$OUT" \
      --coords_type key_in --point_coords 10 80 --point_labels 1 \
      --init_policy first_visible --reinit_policy on_lost \
      --detector_stride 1 --min_visible_consecutive 1 --lost_patience 2 --empty_mask_when_lost 1 \
      --text_prompt "$(repaint_prompt "$TASK")" \
      --box_threshold 0.35 --text_threshold 0.30 \
      --max_mask_area_ratio 0.35 --min_mask_area_ratio 0.002 --max_white_pixel_ratio_in_mask 0.60 \
      --exclude_bottom_ratio 0.14 --erode_kernel_size 3 --composite_erode_kernel_size 1 --blend_alpha_sigma 1.0 \
      --fps $FPS --device cuda \
      --save_removed_video 0 --save_mask_frames 0 --save_mask_video 1 \
      --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 1
  done
done
BASH
```

批处理数量检查：

```bash
echo "stage1 human-object counts"; for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do printf '%s ' "$TASK"; find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object/$TASK -path '*/stage1_human_inpaint/removed_w_mask_*.mp4' 2>/dev/null | wc -l; done
echo "final repaint counts"; for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do printf '%s ' "$TASK"; find /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_visible_reinit/e0_robot_object/$TASK -maxdepth 2 -type f -name final_repainted.mp4 2>/dev/null | wc -l; done
```


### I3.6 L16 六任务：白色背景 SAM + 反选 mask repaint（不改现有代码）

用途：这是 I3.5.2/I3.5.3 的对照路线。旧路线在 Stage-2 直接 prompt `robot arm + object`，如果第一帧没有机械臂/物体，mask 可能从一开始就漂到背景。本节改为 prompt L16 源视频里的白色背景，然后使用 `--invert_mask` 得到“非白背景区域”（通常就是机械臂 + 物体）作为贴回区域。

注意：`remove_anything_video_sam3_robot.py --invert_mask` 会反转保存出来的 `mask_head_cam_plan/`、`mask_head_cam_plan.mp4`、`w_mask_head_cam_plan.mp4`；脚本自带的 `target_with_original_head_cam_plan.mp4` 仍然用原始白背景 mask 合成，所以本节命令会关闭脚本自带合成，并在命令末尾用反选后的 `mask_head_cam_plan/*.jpg` 重新合成 `target_with_original_head_cam_plan.mp4` 和 `final_repainted.mp4`。合成时输出帧数跟随 robot/mask 帧数；如果 Stage-1 背景更短，会按比例采样背景帧来拉伸到同样长度。

默认背景使用 I3.5.2/I3.5.3 的“人手+对应物体” Stage-1：

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4
```

如果只是想和“只抠除人手”的旧背景做对照，可以把命令开头改成 `BG_MODE=hand_only`；正式检查 L16 robot/object 合成时不要用 hand-only 背景，否则真实原始物体会留在背景里：

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4
```

debug 运行六任务各 5 个 id：

```bash
RUN_MODE=debug BG_MODE=human_object OVERWRITE=1 bash <<'BASH'
set -eo pipefail

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam3-dino3
cd /home/zaijia001/ssd/inpainting_sam3_robot

L16=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean
BGROOT_HAND=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1
BGROOT_OBJECT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object
OUTROOT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object
LEGACY=/home/zaijia001/ssd/inpainting_sam2_robot
GPU=${GPU:-3}
FPS=${FPS:-5}
RUN_MODE=${RUN_MODE:-debug}
BG_MODE=${BG_MODE:-human_object}
OVERWRITE=${OVERWRITE:-1}
MASK_IDX=${MASK_IDX:-0}
WHITE_PROMPT=${WHITE_PROMPT:-white background, white floor, white table, blank white area.}
COMPOSITE_ERODE=${COMPOSITE_ERODE:-0}
BLEND_ALPHA_SIGMA=${BLEND_ALPHA_SIGMA:-1.0}

ids_debug() {
  case "$1" in
    pick_diverse_bottles|place_bread_basket|stack_cups|pnp_tray) echo 0 1 2 3 4 ;;
    handover_bottle) echo 1 2 3 4 5 ;;
    pnp_bread) echo 7 8 9 10 11 ;;
  esac
}

ids_all() {
  [[ -d "$L16/$1" ]] || return 0
  find "$L16/$1" -path '*/head_cam_plan.mp4' 2>/dev/null \
    | sed 's#.*/foundation_input_\([0-9]*\)/head_cam_plan.mp4#\1#' \
    | sort -n
}

ids() {
  case "$RUN_MODE" in
    debug) ids_debug "$1" ;;
    batch) ids_all "$1" ;;
    *) echo "[error] RUN_MODE must be debug or batch, got $RUN_MODE" >&2; return 1 ;;
  esac
}

bg_path() {
  TASK=$1
  ID=$2
  if [[ "$BG_MODE" == "human_object" ]]; then
    echo "$BGROOT_OBJECT/${TASK}/id_${ID}/stage1_human_inpaint/removed_w_mask_rgb_${ID}.mp4"
  else
    BG_ROOT="$BGROOT_HAND/${TASK}/id_${ID}"
    BG="$BG_ROOT/human_hand_bg.mp4"
    if [[ ! -f "$BG" && -d "$BG_ROOT/stage1_human_inpaint" ]]; then
      BG=$(find "$BG_ROOT/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1 || true)
    fi
    echo "$BG"
  fi
}

for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  for ID in $(ids "$TASK"); do
    ROBOT=$L16/${TASK}/foundation_input_${ID}/head_cam_plan.mp4
    BG=$(bg_path "$TASK" "$ID")
    OUT=$OUTROOT/${TASK}/id_${ID}_l16_whitebg_${BG_MODE}
    FINAL=$OUT/final_repainted.mp4
    [[ "$OVERWRITE" == "1" || ! -f "$FINAL" ]] || { echo "[skip] task=${TASK} id=${ID} final=${FINAL}"; continue; }
    [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing ROBOT=${ROBOT}"; continue; }
    [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG=${BG}"; continue; }
    mkdir -p "$OUT"
    echo "[whitebg-sam] task=${TASK} id=${ID} bg_mode=${BG_MODE} mask_idx=${MASK_IDX}"
    CUDA_VISIBLE_DEVICES=$GPU python remove_anything_video_sam3_robot.py \
      --input_video "$ROBOT" --target_video "$BG" --output_dir "$OUT" \
      --coords_type key_in --point_coords 10 80 --point_labels 1 \
      --dilate_kernel_size 0 \
      --text_prompt "$WHITE_PROMPT" \
      --box_threshold 0.20 --text_threshold 0.20 \
      --max_mask_area_ratio 1.0 --exclude_bottom_ratio 0.0 \
      --erode_kernel_size 0 --composite_erode_kernel_size 0 --blend_alpha_sigma 0.0 \
      --invert_mask --mask_idx $MASK_IDX --fps $FPS --device cuda \
      --sam_ckpt "$LEGACY/pretrained_models/sam_vit_h_4b8939.pth" \
      --lama_config "$LEGACY/lama/configs/prediction/default.yaml" \
      --lama_ckpt "$LEGACY/pretrained_models/big-lama" \
      --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
      --vi_ckpt "$LEGACY/pretrained_models/sttn.pth" \
      --save_removed_video 0 --save_mask_frames 1 --save_mask_video 1 \
      --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 0

    python - "$ROBOT" "$BG" "$OUT" "$FPS" "$COMPOSITE_ERODE" "$BLEND_ALPHA_SIGMA" <<'PYCODE'
import shutil
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

source_p, bg_p, out_p, fps_s, erode_s, sigma_s = sys.argv[1:7]
out = Path(out_p)
mask_dir = out / "mask_head_cam_plan"
mask_frames = sorted(mask_dir.glob("*.jpg"))
if not mask_frames:
    raise SystemExit(f"[compose error] no inverted mask frames under {mask_dir}")
fps = int(float(fps_s))
erode = int(erode_s)
sigma = float(sigma_s)
source = iio.get_reader(source_p)
bg_frames = iio.mimread(bg_p, memtest=False)
if not bg_frames:
    raise SystemExit(f"[compose error] no background frames in {bg_p}")
out_len = len(mask_frames)
bg_len = len(bg_frames)
print(f"[compose info] output_frames={out_len} bg_frames={bg_len}; background is sampled proportionally")
frames = []
for idx, mask_p in enumerate(mask_frames):
    src = source.get_data(idx)
    if bg_len == 1 or out_len == 1:
        bg_idx = 0
    else:
        bg_idx = int(round(idx * (bg_len - 1) / (out_len - 1)))
    bg = bg_frames[bg_idx]
    if src.ndim == 2:
        src = np.repeat(src[:, :, None], 3, axis=2)
    if bg.ndim == 2:
        bg = np.repeat(bg[:, :, None], 3, axis=2)
    src = src[:, :, :3]
    bg = bg[:, :, :3]
    h, w = bg.shape[:2]
    src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"[compose error] failed to read mask {mask_p}")
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)
    if erode > 0:
        kernel = np.ones((erode, erode), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    alpha = mask.astype(np.float32)
    if sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
        alpha *= mask.astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)
    alpha = alpha[:, :, None]
    frames.append((src.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8))
source.close()
if len(frames) != out_len:
    raise SystemExit(f"[compose error] composed {len(frames)} frames, expected {out_len}")
out_video = out / "target_with_original_head_cam_plan.mp4"
iio.mimwrite(out_video, frames, fps=fps, macro_block_size=1)
shutil.copyfile(out_video, out / "final_repainted.mp4")
print(f"[compose ok] {out_video}")
print(f"[final ok] {out / 'final_repainted.mp4'}")
PYCODE
  done
done
BASH
```

全量批处理：复制上面 debug 的整段脚本运行，只把第一行改成下面这一行；`OVERWRITE=0` 会跳过已经存在的 `final_repainted.mp4`。

```bash
RUN_MODE=batch BG_MODE=human_object OVERWRITE=0 bash <<'BASH'
set -eo pipefail

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam3-dino3
cd /home/zaijia001/ssd/inpainting_sam3_robot

L16=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean
BGROOT_HAND=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1
BGROOT_OBJECT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object
OUTROOT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object
LEGACY=/home/zaijia001/ssd/inpainting_sam2_robot
GPU=${GPU:-3}
FPS=${FPS:-5}
RUN_MODE=${RUN_MODE:-debug}
BG_MODE=${BG_MODE:-human_object}
OVERWRITE=${OVERWRITE:-1}
MASK_IDX=${MASK_IDX:-0}
WHITE_PROMPT=${WHITE_PROMPT:-white background, white floor, white table, blank white area.}
COMPOSITE_ERODE=${COMPOSITE_ERODE:-0}
BLEND_ALPHA_SIGMA=${BLEND_ALPHA_SIGMA:-1.0}

ids_debug() {
  case "$1" in
    pick_diverse_bottles|place_bread_basket|stack_cups|pnp_tray) echo 0 1 2 3 4 ;;
    handover_bottle) echo 1 2 3 4 5 ;;
    pnp_bread) echo 7 8 9 10 11 ;;
  esac
}

ids_all() {
  [[ -d "$L16/$1" ]] || return 0
  find "$L16/$1" -path '*/head_cam_plan.mp4' 2>/dev/null \
    | sed 's#.*/foundation_input_\([0-9]*\)/head_cam_plan.mp4#\1#' \
    | sort -n
}

ids() {
  case "$RUN_MODE" in
    debug) ids_debug "$1" ;;
    batch) ids_all "$1" ;;
    *) echo "[error] RUN_MODE must be debug or batch, got $RUN_MODE" >&2; return 1 ;;
  esac
}

bg_path() {
  TASK=$1
  ID=$2
  if [[ "$BG_MODE" == "human_object" ]]; then
    echo "$BGROOT_OBJECT/${TASK}/id_${ID}/stage1_human_inpaint/removed_w_mask_rgb_${ID}.mp4"
  else
    BG_ROOT="$BGROOT_HAND/${TASK}/id_${ID}"
    BG="$BG_ROOT/human_hand_bg.mp4"
    if [[ ! -f "$BG" && -d "$BG_ROOT/stage1_human_inpaint" ]]; then
      BG=$(find "$BG_ROOT/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1 || true)
    fi
    echo "$BG"
  fi
}

for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  for ID in $(ids "$TASK"); do
    ROBOT=$L16/${TASK}/foundation_input_${ID}/head_cam_plan.mp4
    BG=$(bg_path "$TASK" "$ID")
    OUT=$OUTROOT/${TASK}/id_${ID}_l16_whitebg_${BG_MODE}
    FINAL=$OUT/final_repainted.mp4
    [[ "$OVERWRITE" == "1" || ! -f "$FINAL" ]] || { echo "[skip] task=${TASK} id=${ID} final=${FINAL}"; continue; }
    [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing ROBOT=${ROBOT}"; continue; }
    [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG=${BG}"; continue; }
    mkdir -p "$OUT"
    echo "[whitebg-sam] task=${TASK} id=${ID} bg_mode=${BG_MODE} mask_idx=${MASK_IDX}"
    CUDA_VISIBLE_DEVICES=$GPU python remove_anything_video_sam3_robot.py \
      --input_video "$ROBOT" --target_video "$BG" --output_dir "$OUT" \
      --coords_type key_in --point_coords 10 80 --point_labels 1 \
      --dilate_kernel_size 0 \
      --text_prompt "$WHITE_PROMPT" \
      --box_threshold 0.20 --text_threshold 0.20 \
      --max_mask_area_ratio 1.0 --exclude_bottom_ratio 0.0 \
      --erode_kernel_size 0 --composite_erode_kernel_size 0 --blend_alpha_sigma 0.0 \
      --invert_mask --mask_idx $MASK_IDX --fps $FPS --device cuda \
      --sam_ckpt "$LEGACY/pretrained_models/sam_vit_h_4b8939.pth" \
      --lama_config "$LEGACY/lama/configs/prediction/default.yaml" \
      --lama_ckpt "$LEGACY/pretrained_models/big-lama" \
      --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
      --vi_ckpt "$LEGACY/pretrained_models/sttn.pth" \
      --save_removed_video 0 --save_mask_frames 1 --save_mask_video 1 \
      --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 0

    python - "$ROBOT" "$BG" "$OUT" "$FPS" "$COMPOSITE_ERODE" "$BLEND_ALPHA_SIGMA" <<'PYCODE'
import shutil
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

source_p, bg_p, out_p, fps_s, erode_s, sigma_s = sys.argv[1:7]
out = Path(out_p)
mask_dir = out / "mask_head_cam_plan"
mask_frames = sorted(mask_dir.glob("*.jpg"))
if not mask_frames:
    raise SystemExit(f"[compose error] no inverted mask frames under {mask_dir}")
fps = int(float(fps_s))
erode = int(erode_s)
sigma = float(sigma_s)
source = iio.get_reader(source_p)
bg_frames = iio.mimread(bg_p, memtest=False)
if not bg_frames:
    raise SystemExit(f"[compose error] no background frames in {bg_p}")
out_len = len(mask_frames)
bg_len = len(bg_frames)
print(f"[compose info] output_frames={out_len} bg_frames={bg_len}; background is sampled proportionally")
frames = []
for idx, mask_p in enumerate(mask_frames):
    src = source.get_data(idx)
    if bg_len == 1 or out_len == 1:
        bg_idx = 0
    else:
        bg_idx = int(round(idx * (bg_len - 1) / (out_len - 1)))
    bg = bg_frames[bg_idx]
    if src.ndim == 2:
        src = np.repeat(src[:, :, None], 3, axis=2)
    if bg.ndim == 2:
        bg = np.repeat(bg[:, :, None], 3, axis=2)
    src = src[:, :, :3]
    bg = bg[:, :, :3]
    h, w = bg.shape[:2]
    src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"[compose error] failed to read mask {mask_p}")
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)
    if erode > 0:
        kernel = np.ones((erode, erode), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    alpha = mask.astype(np.float32)
    if sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
        alpha *= mask.astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)
    alpha = alpha[:, :, None]
    frames.append((src.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8))
source.close()
if len(frames) != out_len:
    raise SystemExit(f"[compose error] composed {len(frames)} frames, expected {out_len}")
out_video = out / "target_with_original_head_cam_plan.mp4"
iio.mimwrite(out_video, frames, fps=fps, macro_block_size=1)
shutil.copyfile(out_video, out / "final_repainted.mp4")
print(f"[compose ok] {out_video}")
print(f"[final ok] {out / 'final_repainted.mp4'}")
PYCODE
  done
done
BASH
```

检查输出数量：

```bash
echo "white-bg invert final counts"; for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do printf '%s ' "$TASK"; find /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/$TASK -maxdepth 2 -type f -name final_repainted.mp4 2>/dev/null | wc -l; done
```

合成后视频看这里：

```text
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_human_object/final_repainted.mp4
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_human_object/target_with_original_head_cam_plan.mp4
```

SAM 掉背景/反选效果看这里：

```text
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_human_object/w_box_head_cam_plan.mp4       # 原始白背景检测框
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_human_object/w_mask_head_cam_plan.mp4      # 反选后贴图区域，可理解为机器人+物体候选
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_human_object/mask_head_cam_plan.mp4        # 反选后的二值 mask 视频
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_human_object/mask_head_cam_plan/000000.jpg     # 逐帧反选 mask
```

调参建议：如果 `w_mask_head_cam_plan.mp4` 反选后包含太多白底，先尝试 `MASK_IDX=1` 或 `MASK_IDX=2` 重跑；如果边缘漏掉机械臂/物体，先保持 `COMPOSITE_ERODE=0`，只调 `BLEND_ALPHA_SIGMA`；如果想确认物体残影来自背景来源，可用 `BG_MODE=hand_only` 对照跑同一批 id；正式结果应优先使用 `BG_MODE=human_object`。

#### I3.6.1 L16 白背景反选：按任务并行重跑入口

用途：把 I3.6 拆成两个可复用脚本，便于按任务分配不同 GPU 并行运行：

```text
Stage-1 人手+物体 inpaint:
/home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stage1_human_object_task.sh

Stage-2 白背景 SAM + 反选 repaint:
/home/zaijia001/ssd/RoboTwin/code_painting/run_l16_whitebg_repaint_task.sh
```

Stage-2 脚本会按 robot/mask 帧数输出最终视频；如果 Stage-1 BG 更短，会按比例采样 BG 帧拉伸到同样帧数。`remove_anything_video_sam3_robot.py` 在 `--save_removed_video 0` 时已跳过无用 STTN inpainting，只保存 mask/box 并交给脚本内的 compose 步骤合成，避免 Stage-2 OOM。

先补/重跑五个非 `stack_cups` 任务的 Stage-1。`OVERWRITE=0` 只补缺失；如果要强制重做某个任务，把对应命令改成 `OVERWRITE=1`：

```bash
tmux new-session -d -s l16_s1_pick_gpu0 'TASK=pick_diverse_bottles GPU=0 OVERWRITE=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stage1_human_object_task.sh'
tmux new-session -d -s l16_s1_place_gpu1 'TASK=place_bread_basket GPU=1 OVERWRITE=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stage1_human_object_task.sh'
tmux new-session -d -s l16_s1_handover_gpu2 'TASK=handover_bottle GPU=2 OVERWRITE=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stage1_human_object_task.sh'
tmux new-session -d -s l16_s1_pnpbread_gpu3 'TASK=pnp_bread GPU=3 OVERWRITE=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stage1_human_object_task.sh'
# 第二波：等任意一张 GPU 空出来后再跑 pnp_tray；这里示例用 GPU=0。
tmux new-session -d -s l16_s1_pnptray_gpu0 'TASK=pnp_tray GPU=0 OVERWRITE=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stage1_human_object_task.sh'
```

Stage-1 检查：

```bash
for TASK in pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray; do printf '%s ' "$TASK"; find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object/$TASK -path '*/stage1_human_inpaint/removed_w_mask_*.mp4' 2>/dev/null | wc -l; done
```

再跑五个非 `stack_cups` 任务的 Stage-2 repaint。这里建议 `OVERWRITE=1`，因为旧输出可能是 hand-only 背景或未做比例拉伸：

```bash
tmux new-session -d -s l16_rp_pick_gpu0 'TASK=pick_diverse_bottles GPU=0 OVERWRITE=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_whitebg_repaint_task.sh'
tmux new-session -d -s l16_rp_place_gpu1 'TASK=place_bread_basket GPU=1 OVERWRITE=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_whitebg_repaint_task.sh'
tmux new-session -d -s l16_rp_handover_gpu2 'TASK=handover_bottle GPU=2 OVERWRITE=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_whitebg_repaint_task.sh'
tmux new-session -d -s l16_rp_pnpbread_gpu3 'TASK=pnp_bread GPU=3 OVERWRITE=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_whitebg_repaint_task.sh'
# 第二波：等任意一张 GPU 空出来后再跑 pnp_tray；这里示例用 GPU=0。
tmux new-session -d -s l16_rp_pnptray_gpu0 'TASK=pnp_tray GPU=0 OVERWRITE=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_whitebg_repaint_task.sh'
```

Stage-2 检查：

```bash
for TASK in pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray; do printf '%s ' "$TASK"; find /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/$TASK -maxdepth 2 -type f -name final_repainted.mp4 -path '*whitebg_human_object*' 2>/dev/null | wc -l; done
```

单条输出检查：

```text
Stage-1 BG:
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4

Stage-2 final:
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_human_object/final_repainted.mp4
```

#### I3.6.2 L16 stack_cups 绿色杯保护 Stage-1 debug 与 B 方案全量

背景：`stack_cups` 用 DINO 文本框检测 `left light pink cup, right dark red cup` 时，容易产生接近全屏的大框；绿色杯在两个红杯中间，因此会被一起送入 SAM2/video propagation，导致 Stage-1 inpaint 把绿色杯也去掉。

已测四个方案：

- A `A_protect_dino`：DINO remove mask 减 DINO `green cup` protect mask。当前效果不稳定，是错误方法；原因是 remove/protect 都依赖 DINO，大框错误会继续传导。
- B `B_points_negative`：SAM2 正点标注左右红杯和双手，绿色杯中心作为负点。当前 `id_0..4` debug 可用，优先采用。
- C `C_hsv_green_protect`：DINO remove mask 减 HSV green protect mask。当前 `id_0..4` debug 也可用，作为 B 的备选。
- D `D_tight_dino`：只提高 DINO prompt/threshold。当前效果不行，是错误方法；原因是 DINO 仍可能给出覆盖绿色杯的大框。

##### I3.6.2.1 四方案 debug 输出位置

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/A_protect_dino/stack_cups/id_<ID>/stage1_human_inpaint/
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/B_points_negative/stack_cups/id_<ID>/stage1_human_inpaint/
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/C_hsv_green_protect/stack_cups/id_<ID>/stage1_human_inpaint/
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/D_tight_dino/stack_cups/id_<ID>/stage1_human_inpaint/
```

重点看：

```text
w_mask_rgb_<ID>.mp4
w_box_rgb_<ID>.mp4
removed_w_mask_rgb_<ID>.mp4
w_protect_mask_rgb_<ID>.mp4   # A/C 才有
debug_summary.json
```

##### I3.6.2.2 只跑 B 方案全量 Stage-1

```bash
IDS=$(find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/stack_cups -path '*/head_cam_plan.mp4' 2>/dev/null | sed 's#.*/foundation_input_\([0-9]*\)/head_cam_plan.mp4#\1#' | sort -n | tr '\n' ' ')
tmux new-session -d -s l16_stack_B_stage1_gpu1 "IDS=\"$IDS\" GPU=1 VARIANTS=\"B_points_negative\" MAX_FRAMES=300 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stack_cups_debug_variants.sh"
```

看进度：

```bash
tmux capture-pane -pt l16_stack_B_stage1_gpu1 -S -80
for V in B_points_negative C_hsv_green_protect; do printf '%s ' "$V"; find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/$V/stack_cups -maxdepth 3 -type f -name 'removed_w_mask_rgb_*.mp4' 2>/dev/null | wc -l; done
```

##### I3.6.2.3 B 方案 Stage-2 repaint

Stage-2 读取 B 方案 Stage-1 背景，不覆盖默认 `e0_robot_object` 输出，单独写到 `e0_robot_object_b_points_negative`。

```bash
tmux new-session -d -s l16_stack_B_stage2_after_s1_gpu1 'while tmux has-session -t l16_stack_B_stage1_gpu1 2>/dev/null; do sleep 60; done; TASK=stack_cups GPU=1 OVERWRITE=1 STAGE1=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/B_points_negative OUTROOT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_whitebg_repaint_task.sh'
```

最终结果位置：

```text
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative/stack_cups/id_<ID>_l16_whitebg_human_object/final_repainted.mp4
```

计数检查：

```bash
find /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative/stack_cups -maxdepth 2 -type f -name final_repainted.mp4 2>/dev/null | wc -l
```

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

### J0.1 六任务 D435：筛选可用 AnyGrasp / D435 replay / HaMeR 数据

用途：D435 版本的候选筛选使用 D435 replay 的原始 head 图像作为底图，对应路径是：

```text
AnyGrasp: /home/zaijia001/ssd/data/piper/hand/<TASK>/<TASK>_output/foundation_input_<ID>
D435 replay: /home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435/foundation_input_<ID>
HaMeR: /home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_output/hand_detections_<ID>.npz
```

注意：`place_bread_basket` 当前只检查到 `place_bread_basket_output_old_cam`，所以命令里对这个任务做了 fallback。如果你已经把 D435 AnyGrasp 结果整理成 `place_bread_basket_output`，可以把 `ANY_ROOT` 改回 `${TASK}_output`。

`seq 0 120` 是粗扫，不代表每个任务都有 121 个 episode。当前本机可用基础输入统计大致是：

```text
pick_diverse_bottles: AnyGrasp/D435 replay/HaMeR 交集 102 个，D435 summary 已生成 102 个
place_bread_basket: AnyGrasp/D435 replay/HaMeR 交集 92 个，按左右手 effective keyframes 可生成 43 个
stack_cups: AnyGrasp/D435 replay/HaMeR 交集 47 个，按左右手 effective keyframes 可生成 41 个
handover_bottle: AnyGrasp/D435 replay/HaMeR 交集 47 个，按左右手 effective keyframes 可生成 47 个
pnp_bread: AnyGrasp/D435 replay/HaMeR 交集 72 个，按左右手 effective keyframes 可生成 72 个
pnp_tray: AnyGrasp/D435 replay/HaMeR 交集 51 个，D435 summary 已生成 51 个
```

如果看到很多 `MISS`，先看是哪一列为 0：

- `anygrasp=0`: 对应 `foundation_input_<ID>/grasps` 不存在，说明 AnyGrasp 检测没生成或该 id 本来不存在。
- `replay=0`: 对应 `foundation_replay_d435/foundation_input_<ID>/head_anygrasp_frames` 不存在，说明 D435 FoundationPose replay 没生成。
- `hand=0`: 对应 HaMeR `hand_detections_<ID>.npz` 不存在。
- 三列都是 0 且 id 超过任务实际最大 id，通常只是 `seq 0 120` 扫得太宽。

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do REPORT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}_availability.txt; mkdir -p "$(dirname "$REPORT")"; : > "$REPORT"; ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output; [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam; for ID in $(seq 0 120); do A=${ANY_ROOT}/foundation_input_${ID}; R=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435/foundation_input_${ID}; H=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; if [[ -d "$A/grasps" && -d "$R/head_anygrasp_frames" && -f "$H" ]]; then echo "OK task=${TASK} id=${ID} anygrasp=$A replay=$R hand=$H" | tee -a "$REPORT"; else echo "MISS task=${TASK} id=${ID} anygrasp=$([[ -d "$A/grasps" ]] && echo 1 || echo 0) replay=$([[ -d "$R/head_anygrasp_frames" ]] && echo 1 || echo 0) hand=$([[ -f "$H" ]] && echo 1 || echo 0) A=$A R=$R H=$H" | tee -a "$REPORT"; fi; done; done
```

更推荐的交集统计命令：

```bash
python3 - <<'PY'
from pathlib import Path
import json, re
TASKS = ["pick_diverse_bottles", "place_bread_basket", "stack_cups", "handover_bottle", "pnp_bread", "pnp_tray"]
BASE = Path("/home/zaijia001/ssd/data/piper/hand")
PREVIEW = Path("/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435")
ANNROOT = Path("/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review")
def ids_any(task):
    root = BASE / task / f"{task}_output"
    if not root.exists():
        root = BASE / task / f"{task}_output_old_cam"
    return {int(m.group(1)) for p in root.glob("foundation_input_*/grasps") if (m := re.search(r"foundation_input_(\d+)", str(p)))}
def ids_replay(task):
    return {int(m.group(1)) for p in (BASE / task / "foundation_replay_d435").glob("foundation_input_*/head_anygrasp_frames") if (m := re.search(r"foundation_input_(\d+)", str(p)))}
def ids_hand(task):
    return {int(m.group(1)) for p in (BASE / task / "harmer_output").glob("hand_detections_*.npz") if (m := re.search(r"hand_detections_(\d+)\.npz", p.name))}
def ids_summary(task):
    return {int(m.group(1)) for p in (PREVIEW / task).glob("foundation_input_*/summary.json") if (m := re.search(r"foundation_input_(\d+)", str(p)))}
def dedup(xs):
    out, seen = [], set()
    for x in xs:
        x = int(x)
        if x not in seen:
            seen.add(x); out.append(x)
    return out
def effective(info, arm):
    global_k = [int(v) for v in info.get("keyframes", [])]
    arm_k = [int(v) for v in info.get(f"{arm}_keyframes", [])]
    return arm_k[:2] if len(arm_k) >= 2 else dedup(arm_k + global_k)[:2]
for task in TASKS:
    ann = ANNROOT / task / "hand_keyframes_all.json"
    videos = json.load(open(ann)).get("videos", {}) if ann.exists() else {}
    annotated = set()
    for name, info in videos.items():
        m = re.search(r"hand_vis_(\d+)", name)
        if not m or str(info.get("status", "done")).lower() in {"reject", "discard", "bad"}:
            continue
        if len(effective(info, "left")) >= 2 and len(effective(info, "right")) >= 2:
            annotated.add(int(m.group(1)))
    base_ok = ids_any(task) & ids_replay(task) & ids_hand(task)
    ready = base_ok & annotated
    summaries = ids_summary(task)
    print(f"{task}: base_ok={len(base_ok)} annotated_ready={len(ready)} d435_summary={len(summaries)} missing_summary={sorted(ready - summaries)[:20]}")
PY
```

### J1.1 六任务 D435：用人工关键帧生成候选 preview/summary

用途：和 K0.2 一样读取 `hand_keyframes_all.json`，但输入 replay 换成 D435 的 `foundation_replay_d435`，输出写入独立目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json
```

下游 D435 AnyGrasp planner 应使用这个 `summary.json`，不要再复用默认广角的 `anygrasp_h2o_preview`。当前 preview 脚本支持两类标注：

- `keyframes` 有两帧：作为全局两关键帧。
- `keyframes` 不足两帧但 `left_keyframes/right_keyframes` 可补足：写入 `frame_selection.effective_keyframes_by_arm`，下游 planner 会按左右手各自两帧执行。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle ;; place_bread_basket) LEFT_OBJ=basket; RIGHT_OBJ=bread ;; stack_cups) LEFT_OBJ=left_light_pink_cup; RIGHT_OBJ=right_dark_red_cup ;; handover_bottle) LEFT_OBJ=right_bottle; RIGHT_OBJ=right_bottle ;; pnp_bread) LEFT_OBJ=left_bread; RIGHT_OBJ=right_bread ;; pnp_tray) LEFT_OBJ=left_dark_red_cup; RIGHT_OBJ=right_bottle ;; esac; ANN=/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json; ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output; [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam; REPLAY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435; HAND_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output; OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}; [[ -f "$ANN" ]] || { echo "[skip] task=${TASK} missing annotation $ANN"; continue; }; [[ -d "$ANY_ROOT" ]] || { echo "[skip] task=${TASK} missing ANY_ROOT=$ANY_ROOT"; continue; }; [[ -d "$REPLAY_ROOT" ]] || { echo "[skip] task=${TASK} missing REPLAY_ROOT=$REPLAY_ROOT"; continue; }; VIDEO_PREFIX=foundation_input CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh "$ANY_ROOT" "$REPLAY_ROOT" "$HAND_ROOT" "$OUT_ROOT" --hand_keyframes_json "$ANN" --left_target_object "$LEFT_OBJ" --right_target_object "$RIGHT_OBJ" --anygrasp_score_weight 0.25 --orientation_score_weight 0.75 --max_rotation_distance_deg 90 --candidate_target_local_x_offset_m -0.05 --draw_object_overlay 1 --draw_hand_reference 1 --debug_dump_object_distances 1 --top_k 20 --camera_cv_axis_mode legacy_r1; done
```

输出检查：

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435 -name summary.json | sort | head -n 50
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435 -name '*orientation_rank.png' | sort | head -n 50
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do echo "===== ${TASK} ====="; find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK} -name summary.json 2>/dev/null | wc -l; done
```

### J1.2 D435：只重跑指定 id 的候选 preview/summary

用途：如果 J1.1 只在部分 id 上生成了正确的 D435 AnyGrasp 检测结果，可以只重跑这些 id 的关键帧候选选择，避免覆盖或等待全任务。下面命令以你当前看到有 D435 preview 图的 `pick_diverse_bottles` id 为例。输出仍然是：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/pick_diverse_bottles/foundation_input_<ID>/summary.json
```

重要：这一步只生成候选 preview/summary；后面的 planner 必须使用 `L15.3` 的 D435 路径，不要混用默认广角的 `foundation_replay` 或 `anygrasp_h2o_preview`。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && TASK=pick_diverse_bottles; IDS="2 17 18 19 20 21"; case "$TASK" in pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle ;; pnp_tray) LEFT_OBJ=left_dark_red_cup; RIGHT_OBJ=right_bottle ;; *) echo "[error] add object mapping for TASK=$TASK"; exit 1 ;; esac; ANN=/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json; ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output; [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam; REPLAY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435; HAND_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output; OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}; [[ -f "$ANN" ]] || { echo "[error] missing annotation $ANN"; exit 1; }; [[ -d "$ANY_ROOT" ]] || { echo "[error] missing ANY_ROOT=$ANY_ROOT"; exit 1; }; [[ -d "$REPLAY_ROOT" ]] || { echo "[error] missing REPLAY_ROOT=$REPLAY_ROOT"; exit 1; }; VIDEO_PREFIX=foundation_input CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh "$ANY_ROOT" "$REPLAY_ROOT" "$HAND_ROOT" "$OUT_ROOT" --ids ${IDS} --hand_keyframes_json "$ANN" --left_target_object "$LEFT_OBJ" --right_target_object "$RIGHT_OBJ" --anygrasp_score_weight 0.25 --orientation_score_weight 0.75 --max_rotation_distance_deg 90 --candidate_target_local_x_offset_m -0.05 --draw_object_overlay 1 --draw_hand_reference 1 --debug_dump_object_distances 1 --top_k 20 --camera_cv_axis_mode legacy_r1
```

指定 id 输出检查：

```bash
TASK=pick_diverse_bottles; for ID in 2 17 18 19 20 21; do echo "===== id=${ID} ====="; ls -lh /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}/foundation_input_${ID}/summary.json /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}/foundation_input_${ID}/*orientation_rank.png 2>/dev/null || true; done
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
set -eo pipefail
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
set -eo pipefail
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

### L0. H2O pi0 三类数据处理总览和运行顺序

本段是索引，避免把 human 数据、普通 robot replay 数据、AnyGrasp replay 数据混在一起。三类数据最终都要走：

```text
视频/轨迹源 -> processed HDF5 -> LeRobot cache -> 25 episode subset/zip
```

#### L0.1 Human 数据：真实人手 head + replay action/wrist

用途：`cam_high` 保留真实人手视频，action/state/wrist 来自 replay 计算结果。适合作为真实视觉 baseline。

旧三任务默认广角 action/wrist：

```text
L5 或 L5.1 -> L10.4 的 human_head_pure_action -> L11
```

新三任务当前已有的是 D435 action/wrist：

```text
L5.2 -> L10.5 -> L11.1.3
```

注意：`handover_bottle / pnp_bread / pnp_tray` 当前没有 `h2_pure/<TASK>/id<ID>_z005`，所以不要对这三个任务跑 L5.1；应跑 L5.2。

#### L0.2 Robot replay 数据：repaint robot head + pure replay action/wrist

默认广角 robot replay：

```text
旧三任务：已有 L6，六任务批量用 L6.1 -> L10.4 的 pure_repaint -> L11.2
```

D435 robot replay / visible-reinit：

```text
旧三任务：I1 -> I3.4 -> L8 或 L8.2 -> L10.3 或 L10.6 -> L11.2.4
新三任务：I1.1 -> I3.5 -> L8.1 或 L8.2 -> L10.6 -> L11.2.4
```

当前新三任务还缺 Stage-1 和 D435 robot repaint，所以要先跑 I1.1，再跑 I3.5，最后跑 L8.1。

#### L0.3 AnyGrasp replay 数据：AnyGrasp repaint head + planner action/wrist

用途：`cam_high` 使用 AnyGrasp planner/repaint 的 robot 视频，action/state/wrist 来自 planner 输出。

```text
L9 或 L9.1 -> L10.4 的 anygrasp_repaint -> L11.2
```

前提是每个 planner episode 已有：

```text
pose_debug.jsonl
left_wrist_cam_plan.mp4
right_wrist_cam_plan.mp4
```

如果 wrist plan 视频没补齐，L9/L9.1 会 skip，严重时会报 `No usable episodes were processed`。

#### L0.4 task prompt 设置位置

prompt 应在 processed HDF5 生成阶段设置，也就是 L5/L5.1/L5.2/L6.1/L8.1/L9.1 的 `INSTRUCTION="..."`。当前 `convert_aloha_data_to_lerobot_R1.py --task` 不会覆盖已经写进 `episode_*/instructions.json` 的文本；如果 processed data 已经生成但还没转 LeRobot，先按 L11.3 改 `instructions.json`。

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

### L5.1 六个任务转换：原始人手 head + pure replay action/wrist

用途：这是 L5 的 6-task 版本。`cam_high` 使用真实人手原始视频 `harmer_input/rgb_<ID>.mp4`，action/state 与左右 wrist 使用 `h2_pure/<TASK>/id<ID>_z005` 的 pure replay 结果。生成后续 LeRobot cache：

```text
/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_<TASK>_human_head_pure_action-120
```

前提：每个 task/id 已经存在：

```text
/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_input/rgb_<ID>.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/<TASK>/id<ID>_z005/world_targets_and_status.npz
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/<TASK>/id<ID>_z005/left_wrist_replay.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/<TASK>/id<ID>_z005/right_wrist_replay.mp4
```

运行命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && N=120; for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in pick_diverse_bottles) INSTRUCTION="pick up one bottle with one arm, and pick up another bottle with the other arm." ;; place_bread_basket) INSTRUCTION="Use one arm to pick up the bread, put it into the basket, and use another arm to lift the basket." ;; stack_cups) INSTRUCTION="Stack the dark red and light red cups onto the green cup." ;; handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; echo "===== process human_head_pure_action TASK=${TASK} ====="; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_human_head_pure_action" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input --head-dir-template '.' --head-video-name 'rgb_{id}.mp4' --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK} --retarget-dir-template 'id{id}_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-${N}; done
```

检查 6 个 processed HDF5 数量：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-120; echo "===== ${TASK} ====="; find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | sort | wc -l; done
```

### L5.2 新三任务当前可用版本：原始人手 head + D435 pure replay action/wrist

用途：`handover_bottle / pnp_bread / pnp_tray` 当前检查到已经生成的是：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005
```

而不是：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/<TASK>/id<ID>_z005
```

所以直接跑 L5.1 会在新三任务上因为缺少 `h2_pure/<TASK>/id<ID>_z005` 全部 skip。若要先使用当前已有结果处理新三任务的人手视频，应使用下面的 D435 action/wrist 版本：`cam_high` 仍是真实人手 `rgb_<ID>.mp4`，action/state 与 wrist 来自 `h2_pure_d435`。

如果目标是后续和 D435 robot replay 对比，推荐顺序：

1. 先跑 L5.2，得到真实人手 head + D435 action/wrist 的 baseline。
2. 再跑 I1.1，生成新三任务 Stage-1 人手抠除背景。
3. 再跑 I3.5，生成新三任务 D435 visible-reinit robot repaint。
4. 最后跑 L8.1，把新三任务 D435 robot repaint 转成 processed HDF5。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && N=120; for TASK in handover_bottle pnp_bread pnp_tray; do case "$TASK" in handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; echo "===== process human_head_pure_d435_action TASK=${TASK} ====="; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_human_head_pure_d435_action" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input --head-dir-template '.' --head-video-name 'rgb_{id}.mp4' --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK} --retarget-dir-template 'id{id}_d435_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_d435_action-${N}; done
```

检查：

```bash
for TASK in handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_d435_action-120; echo "===== ${TASK} ====="; find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | sort | wc -l; done
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

### L6.1 六个任务转换：repaint robot head + pure replay action/wrist

用途：这是 L11.2 抽取 `local/h2o_<TASK>_pure_repaint` 前必须先跑的前置步骤之一。它会把 6 个 task 的 robot repaint head 和 pure replay action/wrist 先转换成 pi0 intermediate HDF5：

```text
/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_<TASK>_pure_repaint-120
```

前提：每个 task/id 已经存在：

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/<TASK>/id_<ID>/final_repainted.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/<TASK>/id<ID>_z005/world_targets_and_status.npz
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/<TASK>/id<ID>_z005/left_wrist_replay.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/<TASK>/id<ID>_z005/right_wrist_replay.mp4
```

重要：L6.1 是默认广角 `h2_pure` 流程，不是 D435 流程。当前检查结果是：

```text
旧三任务 pick_diverse_bottles/place_bread_basket/stack_cups：有 h2_pure、默认 repaint head，也有 h2_pure_d435。
新三任务 handover_bottle/pnp_bread/pnp_tray：只有 h2_pure_d435，没有 h2_pure，也没有默认广角 results_repaint_piper_h2/e0_robot/<TASK>/id_<ID>/final_repainted.mp4。
```

所以对新三任务运行 L6.1 会全部 skip，并最终报：

```text
RuntimeError: No usable episodes were processed. Check directory templates and video names.
```

这不是 `process_repainted_headcam_with_wrist.py` 的逻辑错误，而是输入路径不属于同一条 pipeline。新三任务要处理 robot replay，应走 D435 流程：`I1.1 -> I3.5 -> L8.1/L8.2 -> L10.6 -> L11.2.4`。

运行命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && N=120; for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in pick_diverse_bottles) INSTRUCTION="pick up one bottle with one arm, and pick up another bottle with the other arm." ;; place_bread_basket) INSTRUCTION="Use one arm to pick up the bread, put it into the basket, and use another arm to lift the basket." ;; stack_cups) INSTRUCTION="Stack the dark red and light red cups onto the green cup." ;; handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; echo "===== process pure_repaint TASK=${TASK} ====="; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/${TASK} --head-dir-template 'id_{id}' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK} --retarget-dir-template 'id{id}_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-${N}; done
```

检查 6 个 processed HDF5 数量：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-120; echo "===== ${TASK} ====="; find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | sort | wc -l; done
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

### L8.1 新三任务：D435 visible-reinit robot repaint head + D435 pure replay action/wrist

用途：把 I3.5 的 D435 robot repaint 结果转成 pi0 processed HDF5。输出可直接和 L5.2 的 `h2o_<TASK>_human_head_pure_d435_action-120` 对比。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && N=120; for TASK in handover_bottle pnp_bread pnp_tray; do case "$TASK" in handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; echo "===== process pure_d435_visible_reinit TASK=${TASK} ====="; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_d435_visible_reinit" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK} --head-dir-template 'id_{id}_d435' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK} --retarget-dir-template 'id{id}_d435_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-${N}; done
```

检查：

```bash
for TASK in handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-120; echo "===== ${TASK} ====="; find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | sort | wc -l; done
```

### L8.2 六个任务 D435 robot replay：visible-reinit head + D435 action/wrist 转 processed HDF5

用途：这是 D435 robot replay 的六任务统一转换入口，输出：

```text
/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_<TASK>_pure_d435_visible_reinit-120
```

前置关系：

```text
旧三任务：E2.4 D435 retarget -> I1 Stage-1 BG -> I3.4 D435 visible-reinit repaint -> L8.2
新三任务：E2.5 D435 retarget -> I1.1 Stage-1 BG -> I3.5 D435 visible-reinit repaint -> L8.2
```

每个 id 需要同时存在：

```text
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435/final_repainted.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005/world_targets_and_status.npz
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005/left_wrist_replay.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005/right_wrist_replay.mp4
```

当前前置完成情况可以先用这个命令检查：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do echo "===== ${TASK} ====="; echo -n "h2_pure_d435 dirs: "; find /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK} -maxdepth 1 -type d -name 'id*_d435_z005' 2>/dev/null | wc -l; echo -n "D435 repaint finals: "; find /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK} -maxdepth 2 -type f -name final_repainted.mp4 2>/dev/null | wc -l; done
```

运行命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && N=120; for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in pick_diverse_bottles) INSTRUCTION="pick up one bottle with one arm, and pick up another bottle with the other arm." ;; place_bread_basket) INSTRUCTION="Use one arm to pick up the bread, put it into the basket, and use another arm to lift the basket." ;; stack_cups) INSTRUCTION="Stack the dark red and light red cups onto the green cup." ;; handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; echo "===== process pure_d435_visible_reinit TASK=${TASK} ====="; python scripts/process_repainted_headcam_with_wrist.py "h2o_${TASK}_pure_d435_visible_reinit" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/${TASK} --head-dir-template 'id_{id}_d435' --head-video-name final_repainted.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/${TASK} --retarget-dir-template 'id{id}_d435_z005' --world-targets-name world_targets_and_status.npz --left-wrist-video-name left_wrist_replay.mp4 --right-wrist-video-name right_wrist_replay.mp4 --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-${N}; done
```

检查 6 个 processed HDF5 数量：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-120; echo "===== ${TASK} ====="; find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | sort | wc -l; done
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

### L9.1 六个任务转换：AnyGrasp repaint head + planner action/wrist

用途：这是 L11.2 抽取 `local/h2o_<TASK>_anygrasp_repaint` 前必须先跑的前置步骤之一。它会把 6 个 task 的 AnyGrasp repaint head 和 planner action/wrist 转换成 pi0 intermediate HDF5：

```text
/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_<TASK>_anygrasp_repaint-60
```

前提：每个 task/id 已经存在：

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/<TASK>/id_<ID>/final_repainted.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/pose_debug.jsonl
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/left_wrist_cam_plan.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/right_wrist_cam_plan.mp4
```

如果 wrist plan 视频还没补齐，命令会 skip 对应 id，严重时最后会报 `No usable episodes were processed`。

运行命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && N=60; for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in pick_diverse_bottles) INSTRUCTION="pick up one bottle with one arm, and pick up another bottle with the other arm." ;; place_bread_basket) INSTRUCTION="place the bread into the basket." ;; stack_cups) INSTRUCTION="stack the cups." ;; handover_bottle) INSTRUCTION="handover the bottle." ;; pnp_bread) INSTRUCTION="pick and place the bread." ;; pnp_tray) INSTRUCTION="pick and place the objects on the tray." ;; esac; echo "===== process anygrasp_repaint TASK=${TASK} ====="; python scripts/process_repainted_planner_outputs.py "h2o_${TASK}_anygrasp_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/${TASK} --head-dir-template 'id_{id}' --head-video-name final_repainted.mp4 --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/${TASK} --planner-dir-template 'foundation_input_{id}' --left-wrist-video-name left_wrist_cam_plan.mp4 --right-wrist-video-name right_wrist_cam_plan.mp4 --pose-debug-name pose_debug.jsonl --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_anygrasp_repaint-${N}; done
```

检查 6 个 AnyGrasp processed HDF5 数量：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_anygrasp_repaint-60; echo "===== ${TASK} ====="; find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | sort | wc -l; done
```

L8/L9 的输出也用 L7 的统计、HDF5 结构检查和 `visualize_processed_hdf5_episode.py` 生成 review mp4。把 L7 里的 `DATASET` 替换成：

```text
h2o_<TASK>_pure_d435_visible_reinit-120
h2o_<TASK>_anygrasp_repaint-60
```

### L9.2 L16 whitebg repaint head + L16 planner action/wrist 转 processed HDF5

用途：把 I3.6/I3.6.1 生成的 L16 whitebg repaint head 视频，和 L16 planner 目录里的机器人状态、左右 wrist 视频整合成 pi0 intermediate HDF5。

注意：L16 目录是 planner-style 输出，不是 L8.2 的 D435 pure replay 输出。当前 L16 每个 episode 目录里没有 `world_targets_and_status.npz`，因此不要使用 `process_repainted_headcam_with_wrist.py`。这里必须使用 `process_repainted_planner_outputs.py`，输入是：

```text
head repaint:
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_human_object/final_repainted.mp4

L16 planner:
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/<TASK>/foundation_input_<ID>/pose_debug.jsonl
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/<TASK>/foundation_input_<ID>/left_wrist_cam_plan.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/<TASK>/foundation_input_<ID>/right_wrist_cam_plan.mp4
```

六任务默认 L16 whitebg repaint 转 processed HDF5：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && N=120; for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in pick_diverse_bottles) INSTRUCTION="pick up one bottle with one arm, and pick up another bottle with the other arm." ;; place_bread_basket) INSTRUCTION="Use one arm to pick up the bread, put it into the basket, and use another arm to lift the basket." ;; stack_cups) INSTRUCTION="Stack the dark red and light red cups onto the green cup." ;; handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; echo "===== process l16_whitebg_repaint TASK=${TASK} ====="; python scripts/process_repainted_planner_outputs.py "h2o_${TASK}_l16_whitebg_repaint" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/${TASK} --head-dir-template 'id_{id}_l16_whitebg_human_object' --head-video-name final_repainted.mp4 --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/${TASK} --planner-dir-template 'foundation_input_{id}' --left-wrist-video-name left_wrist_cam_plan.mp4 --right-wrist-video-name right_wrist_cam_plan.mp4 --pose-debug-name pose_debug.jsonl --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_l16_whitebg_repaint-${N}; done
```

如果 `stack_cups` 使用 I3.6.2 的 B 方案结果，单独跑这一条覆盖成独立数据集名：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=stack_cups; INSTRUCTION="Stack the dark red and light red cups onto the green cup."; N=120; python scripts/process_repainted_planner_outputs.py "h2o_${TASK}_l16_whitebg_b_points_negative" "$INSTRUCTION" ${N} --head-root /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative/${TASK} --head-dir-template 'id_{id}_l16_whitebg_human_object' --head-video-name final_repainted.mp4 --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/${TASK} --planner-dir-template 'foundation_input_{id}' --left-wrist-video-name left_wrist_cam_plan.mp4 --right-wrist-video-name right_wrist_cam_plan.mp4 --pose-debug-name pose_debug.jsonl --review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_l16_whitebg_b_points_negative-${N}
```

检查 processed HDF5：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_l16_whitebg_repaint-120; echo "===== ${TASK} ====="; find "$ROOT" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | sort | wc -l; done
find /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_stack_cups_l16_whitebg_b_points_negative-120 -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | wc -l
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

#### L10.4 六个任务：把 L6.1/L9.1 的 processed HDF5 转成 LeRobot cache

用途：这是 L11.2 抽取 `_25ep` 前必须先跑的前置步骤。你刚才遇到的 `Dataset not found as path or repo id: local/h2o_<TASK>_pure_repaint`，就是因为这一步还没生成对应 LeRobot cache。

注意：当前 `convert_aloha_data_to_lerobot_R1.py` 的 `--task` 不会覆盖 processed episode 里已有的 `instructions.json`。prompt 要在 L6.1/L9.1 的 `INSTRUCTION="..."` 设置，或者按 L11.3 先批量改 `episode_*/instructions.json` 后再运行这里。

六任务原始人手 head + pure action：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_action-120; case "$TASK" in pick_diverse_bottles) INSTRUCTION="pick up one bottle with one arm, and pick up another bottle with the other arm." ;; place_bread_basket) INSTRUCTION="Use one arm to pick up the bread, put it into the basket, and use another arm to lift the basket." ;; stack_cups) INSTRUCTION="Stack the dark red and light red cups onto the green cup." ;; handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; [[ -d "$DATASET" ]] || { echo "[skip] missing DATASET=$DATASET; run L5.1 first"; continue; }; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_human_head_pure_action" --task "$INSTRUCTION" --use-wrist --mode video; done
```

六任务默认广角 robot replay：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_repaint-120; case "$TASK" in pick_diverse_bottles) INSTRUCTION="pick up one bottle with one arm, and pick up another bottle with the other arm." ;; place_bread_basket) INSTRUCTION="Use one arm to pick up the bread, put it into the basket, and use another arm to lift the basket." ;; stack_cups) INSTRUCTION="Stack the dark red and light red cups onto the green cup." ;; handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; [[ -d "$DATASET" ]] || { echo "[skip] missing DATASET=$DATASET; run L6.1 first"; continue; }; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_pure_repaint" --task "$INSTRUCTION" --use-wrist --mode video; done
```

六任务 AnyGrasp robot：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_anygrasp_repaint-60; case "$TASK" in pick_diverse_bottles) INSTRUCTION="pick up one bottle with one arm, and pick up another bottle with the other arm." ;; place_bread_basket) INSTRUCTION="Use one arm to pick up the bread, put it into the basket, and use another arm to lift the basket." ;; stack_cups) INSTRUCTION="Stack the dark red and light red cups onto the green cup." ;; handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; [[ -d "$DATASET" ]] || { echo "[skip] missing DATASET=$DATASET; run L9.1 first"; continue; }; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_anygrasp_repaint" --task "$INSTRUCTION" --use-wrist --mode video; done
```

#### L10.5 新三任务 L5.2：human head + D435 action/wrist 转 LeRobot

用途：L5.2 的输出数据集名是 `h2o_<TASK>_human_head_pure_d435_action-120`，不能用 L10.4 里的 `human_head_pure_action` 命令转换。跑完 L5.2 后，使用这里的命令转成 LeRobot cache：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in handover_bottle pnp_bread pnp_tray; do DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_d435_action-120; case "$TASK" in handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; [[ -d "$DATASET" ]] || { echo "[skip] missing DATASET=$DATASET; run L5.2 first"; continue; }; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_human_head_pure_d435_action" --task "$INSTRUCTION" --use-wrist --mode video; done
```

检查：

```bash
for TASK in handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_${TASK}_human_head_pure_d435_action; echo "===== h2o_${TASK}_human_head_pure_d435_action ====="; if [[ -f "$ROOT/meta/info.json" ]]; then python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
info = json.load(open(root / "meta/info.json"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
PY
else echo "missing"; fi; done
```

#### L10.6 六个任务 L8.2：D435 robot replay 转 LeRobot cache

用途：把 L8/L8.1/L8.2 生成的六任务 D435 robot replay processed HDF5 转成 LeRobot cache：

```text
local/h2o_<TASK>_pure_d435_visible_reinit
```

注意：如果某个 task 的 L8.2 processed HDF5 还是 0 个 episode，这一步会失败或生成无效 cache。先用 L8.2 的数量检查确认前置完成。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-120; case "$TASK" in pick_diverse_bottles) INSTRUCTION="pick up one bottle with one arm, and pick up another bottle with the other arm." ;; place_bread_basket) INSTRUCTION="Use one arm to pick up the bread, put it into the basket, and use another arm to lift the basket." ;; stack_cups) INSTRUCTION="Stack the dark red and light red cups onto the green cup." ;; handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; COUNT=$(find "$DATASET" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | wc -l); [[ "$COUNT" -gt 0 ]] || { echo "[skip] ${TASK}: no HDF5 under $DATASET; run I1/I1.1 -> I3.4/I3.5 -> L8.2 first"; continue; }; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_pure_d435_visible_reinit" --task "$INSTRUCTION" --use-wrist --mode video; done
```

检查：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_${TASK}_pure_d435_visible_reinit; echo "===== h2o_${TASK}_pure_d435_visible_reinit ====="; if [[ -f "$ROOT/meta/info.json" ]]; then python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
info = json.load(open(root / "meta/info.json"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
PY
else echo "missing"; fi; done
```

#### L10.7 六个任务 L9.2：L16 whitebg repaint 转 LeRobot cache

用途：把 L9.2 生成的 L16 planner-style processed HDF5 转成 LeRobot cache。默认六任务输出名：

```text
local/h2o_<TASK>_l16_whitebg_repaint
```

如果 `stack_cups` 采用 I3.6.2 的 B 方案，额外输出：

```text
local/h2o_stack_cups_l16_whitebg_b_points_negative
```

六任务默认 L16 whitebg repaint：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_l16_whitebg_repaint-120; case "$TASK" in pick_diverse_bottles) INSTRUCTION="pick up one bottle with one arm, and pick up another bottle with the other arm." ;; place_bread_basket) INSTRUCTION="Use one arm to pick up the bread, put it into the basket, and use another arm to lift the basket." ;; stack_cups) INSTRUCTION="Stack the dark red and light red cups onto the green cup." ;; handover_bottle) INSTRUCTION="Use the right arm to grasp the bottle on the table, handover it to the left arm." ;; pnp_bread) INSTRUCTION="Pick up two breads, then place them onto the blue plate." ;; pnp_tray) INSTRUCTION="Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;; esac; COUNT=$(find "$DATASET" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | wc -l); [[ "$COUNT" -gt 0 ]] || { echo "[skip] ${TASK}: no HDF5 under $DATASET; run L9.2 first"; continue; }; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_l16_whitebg_repaint" --task "$INSTRUCTION" --use-wrist --mode video; done
```

`stack_cups` B 方案单独转换：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=stack_cups; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_l16_whitebg_b_points_negative-120; INSTRUCTION="Stack the dark red and light red cups onto the green cup."; COUNT=$(find "$DATASET" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | wc -l); [[ "$COUNT" -gt 0 ]] || { echo "[skip] no HDF5 under $DATASET; run L9.2 B first"; exit 0; }; uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw-dir "$DATASET" --repo-id "local/h2o_${TASK}_l16_whitebg_b_points_negative" --task "$INSTRUCTION" --use-wrist --mode video
```

检查：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_${TASK}_l16_whitebg_repaint; echo "===== h2o_${TASK}_l16_whitebg_repaint ====="; if [[ -f "$ROOT/meta/info.json" ]]; then python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
info = json.load(open(root / "meta/info.json"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
PY
else echo "missing"; fi; done
```

检查 LeRobot cache 是否生成：

```bash
for MODE in human_head_pure_action pure_repaint anygrasp_repaint; do for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_${TASK}_${MODE}; echo "===== h2o_${TASK}_${MODE} ====="; if [[ -f "$ROOT/meta/info.json" ]]; then python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
info = json.load(open(root / "meta/info.json"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
PY
else echo "missing"; fi; done; done
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

### L11.1 Robot 数据：从已生成 LeRobot cache 抽取 25 episode、打包、上传

用途：L11 已经给出原始人手 head + pure action 的 3 个任务抽取命令；这里新增 robot 版本的 6 个 LeRobot 数据集抽取命令：

- 默认广角 robot replay head + pure replay action/wrist：`local/h2o_<TASK>_pure_repaint`
- AnyGrasp robot repaint head + planner action/wrist：`local/h2o_<TASK>_anygrasp_repaint`

运行顺序：

1. 先确认对应完整 LeRobot cache 已经生成。默认广角 robot replay 对应 L10.2；AnyGrasp robot 对应 L9 的 HDF5 处理完成后，再用 `convert_aloha_data_to_lerobot_R1.py` 转成 `local/h2o_<TASK>_anygrasp_repaint`。
2. 运行下面的 6 条 `subset_lerobot_episodes.py` 命令，生成 `_25ep` 子集。
3. 进入 `/home/zaijia001/.cache/huggingface/lerobot/local` 后分别 zip。
4. 先用 `rclone --dry-run` 检查，再去掉 `--dry-run` 真正上传。

#### L11.1.1 默认广角 robot replay：3 个任务各抽 25 episode

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_pick_diverse_bottles_pure_repaint --output-repo-id local/h2o_pick_diverse_bottles_pure_repaint_25ep --episodes '0-24' --overwrite

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_place_bread_basket_pure_repaint --output-repo-id local/h2o_place_bread_basket_pure_repaint_25ep --episodes '0-24' --overwrite

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_stack_cups_pure_repaint --output-repo-id local/h2o_stack_cups_pure_repaint_25ep --episodes '0-24' --overwrite
```

打包和上传检查：

```bash
cd /home/zaijia001/.cache/huggingface/lerobot/local && zip -r robot_replay_3task_25ep.zip h2o_pick_diverse_bottles_pure_repaint_25ep h2o_place_bread_basket_pure_repaint_25ep h2o_stack_cups_pure_repaint_25ep
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_replay_3task_25ep.zip gdrive:piper/multi/3task/robot_replay -P --drive-chunk-size 64M --transfers 4 --dry-run
```

#### L11.1.2 AnyGrasp robot：3 个任务各抽 25 episode

前提：下面三个源 repo 需要已经存在：

```text
/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_anygrasp_repaint
/home/zaijia001/.cache/huggingface/lerobot/local/h2o_place_bread_basket_anygrasp_repaint
/home/zaijia001/.cache/huggingface/lerobot/local/h2o_stack_cups_anygrasp_repaint
```

抽取命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_pick_diverse_bottles_anygrasp_repaint --output-repo-id local/h2o_pick_diverse_bottles_anygrasp_repaint_25ep --episodes '0-24' --overwrite

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_place_bread_basket_anygrasp_repaint --output-repo-id local/h2o_place_bread_basket_anygrasp_repaint_25ep --episodes '0-24' --overwrite

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_stack_cups_anygrasp_repaint --output-repo-id local/h2o_stack_cups_anygrasp_repaint_25ep --episodes '0-24' --overwrite
```

打包和上传检查：

```bash
cd /home/zaijia001/.cache/huggingface/lerobot/local && zip -r robot_anygrasp_3task_25ep.zip h2o_pick_diverse_bottles_anygrasp_repaint_25ep h2o_place_bread_basket_anygrasp_repaint_25ep h2o_stack_cups_anygrasp_repaint_25ep
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_anygrasp_3task_25ep.zip gdrive:piper/multi/3task/robot_anygrasp -P --drive-chunk-size 64M --transfers 4 --dry-run
```

#### L11.1.3 L10.5 后续：新三任务 human head + D435 action/wrist 抽 25 episode

用途：这是 L5.2 -> L10.5 后的专用抽取步骤。源 repo 是：

```text
local/h2o_handover_bottle_human_head_pure_d435_action
local/h2o_pnp_bread_human_head_pure_d435_action
local/h2o_pnp_tray_human_head_pure_d435_action
```

不要用 `local/h2o_<TASK>_pure_repaint` 抽 L10.5 的结果；`pure_repaint` 是 L6/L6.1 -> L10.4 的 robot replay 数据，不是原始人手 head + D435 action/wrist 数据。

坏数据排除说明：

```text
handover_bottle 原始 bad id：0,7,12,29
pnp_bread 原始 bad id：0,1,2,3,4,5,6,22,70
pnp_tray 当前未在这里额外排除
```

直接 `--episodes '0-24'` 会按 LeRobot episode index 抽取，不会自动理解原始 id。当前检查结果里：

```text
handover_bottle 的 0-24 会包含原始 id 7 和 12
pnp_bread 的 0-24 会包含原始 id 22
```

所以下面命令先读取 processed data 的 `episode_*/instructions.json` 里的 `source_episode_id`，跳过上述 bad 原始 id，再取前 25 个可用 LeRobot episode index。输出仍会重新编号成 `0..24`。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in handover_bottle pnp_bread pnp_tray; do case "$TASK" in handover_bottle) BAD_IDS="0,7,12,29" ;; pnp_bread) BAD_IDS="0,1,2,3,4,5,6,22,70" ;; pnp_tray) BAD_IDS="" ;; esac; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_human_head_pure_d435_action-120; SOURCE=local/h2o_${TASK}_human_head_pure_d435_action; [[ -d "$DATASET" ]] || { echo "[skip] missing processed DATASET=$DATASET; run L5.2 first"; continue; }; [[ -d "/home/zaijia001/.cache/huggingface/lerobot/${SOURCE}" ]] || { echo "[skip] missing LeRobot SOURCE=${SOURCE}; run L10.5 first"; continue; }; EPISODES=$(python3 - "$DATASET" "$BAD_IDS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bad = {int(x) for x in sys.argv[2].split(",") if x.strip()}
keep = []

for p in sorted(root.glob("episode_*/instructions.json"), key=lambda p: int(p.parent.name.split("_")[-1])):
    ep = int(p.parent.name.split("_")[-1])
    row = json.loads(p.read_text())
    src = int(row.get("source_episode_id", ep))
    if src in bad:
        continue
    keep.append(ep)
    if len(keep) >= 25:
        break

if len(keep) < 25:
    raise SystemExit(f"only {len(keep)} usable episodes after excluding {sorted(bad)}")
print(",".join(map(str, keep)))
PY
); echo "===== subset ${TASK}: episodes=${EPISODES} ====="; uv run python scripts/subset_lerobot_episodes.py --source "$SOURCE" --output-repo-id local/h2o_${TASK}_human_head_pure_d435_action_25ep --episodes "$EPISODES" --overwrite; done
```

当前检查得到的安全抽取 episode index 是：

```text
handover_bottle: 0,1,2,3,4,5,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26
pnp_bread: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25
pnp_tray: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
```

打包和上传检查：

```bash
cd /home/zaijia001/.cache/huggingface/lerobot/local && zip -r human_d435_action_3task_25ep.zip h2o_handover_bottle_human_head_pure_d435_action_25ep h2o_pnp_bread_human_head_pure_d435_action_25ep h2o_pnp_tray_human_head_pure_d435_action_25ep
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/human_d435_action_3task_25ep.zip gdrive:piper/multi/3task/human_d435_action -P --drive-chunk-size 64M --transfers 4 --dry-run
```

检查 3 个输出是否都是 25 episode：

```bash
for TASK in handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_${TASK}_human_head_pure_d435_action_25ep; echo "===== h2o_${TASK}_human_head_pure_d435_action_25ep ====="; python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
if not root.exists():
    print("missing")
    raise SystemExit
info = json.load(open(root / "meta/info.json"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
print("first episodes:", (root / "meta/episodes.jsonl").read_text().splitlines()[:3])
PY
done
```

检查 6 个输出是否都是 25 episode：

```bash
for D in h2o_pick_diverse_bottles_pure_repaint_25ep h2o_place_bread_basket_pure_repaint_25ep h2o_stack_cups_pure_repaint_25ep h2o_pick_diverse_bottles_anygrasp_repaint_25ep h2o_place_bread_basket_anygrasp_repaint_25ep h2o_stack_cups_anygrasp_repaint_25ep; do ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/${D}; echo "===== ${D} ====="; python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
if not root.exists():
    print("missing")
    raise SystemExit
info = json.load(open(root / "meta/info.json"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
print("first episodes:", (root / "meta/episodes.jsonl").read_text().splitlines()[:3])
PY
done
```

### L11.2 六个 H2O task：robot replay / AnyGrasp LeRobot cache 抽取 25 episode

用途：覆盖 FoundationPose 章节里的 6 个 H2O task：

```text
pick_diverse_bottles
place_bread_basket
stack_cups
handover_bottle
pnp_bread
pnp_tray
```

运行顺序：

1. 先确认每个完整 LeRobot cache 已存在，例如 `/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_pure_repaint` 和 `/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_anygrasp_repaint`。
2. 运行 L11.2.1 抽取 6 个 robot replay `_25ep`。
3. 运行 L11.2.2 抽取 6 个 AnyGrasp robot `_25ep`。
4. 如果处理 D435 robot replay，先确认 L10.6 已生成 `local/h2o_<TASK>_pure_d435_visible_reinit`，再运行 L11.2.4。
5. 运行 L11.2.3 检查默认广角和 AnyGrasp 的 12 个输出；D435 用 L11.2.4 自带检查。
6. 分别 zip，再先用 `rclone --dry-run` 检查上传路径；确认无误后去掉 `--dry-run`。

#### L11.2.1 六任务默认广角 robot replay：各抽 25 episode

注意：这条命令只适用于 L6/L6.1 -> L10.4 生成的 `local/h2o_<TASK>_pure_repaint`。如果你要处理 L10.5 的新三任务 human head + D435 action/wrist，使用 L11.1.3。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do uv run python scripts/subset_lerobot_episodes.py --source local/h2o_${TASK}_pure_repaint --output-repo-id local/h2o_${TASK}_pure_repaint_25ep --episodes '0-24' --overwrite; done
```

打包和上传检查：

```bash
cd /home/zaijia001/.cache/huggingface/lerobot/local && zip -r robot_replay_6task_25ep.zip h2o_pick_diverse_bottles_pure_repaint_25ep h2o_place_bread_basket_pure_repaint_25ep h2o_stack_cups_pure_repaint_25ep h2o_handover_bottle_pure_repaint_25ep h2o_pnp_bread_pure_repaint_25ep h2o_pnp_tray_pure_repaint_25ep
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_replay_6task_25ep.zip gdrive:piper/multi/6task/robot_replay -P --drive-chunk-size 64M --transfers 4 --dry-run
```

#### L11.2.2 六任务 AnyGrasp robot：各抽 25 episode

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do uv run python scripts/subset_lerobot_episodes.py --source local/h2o_${TASK}_anygrasp_repaint --output-repo-id local/h2o_${TASK}_anygrasp_repaint_25ep --episodes '0-24' --overwrite; done
```

打包和上传检查：

```bash
cd /home/zaijia001/.cache/huggingface/lerobot/local && zip -r robot_anygrasp_6task_25ep.zip h2o_pick_diverse_bottles_anygrasp_repaint_25ep h2o_place_bread_basket_anygrasp_repaint_25ep h2o_stack_cups_anygrasp_repaint_25ep h2o_handover_bottle_anygrasp_repaint_25ep h2o_pnp_bread_anygrasp_repaint_25ep h2o_pnp_tray_anygrasp_repaint_25ep
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_anygrasp_6task_25ep.zip gdrive:piper/multi/6task/robot_anygrasp -P --drive-chunk-size 64M --transfers 4 --dry-run
```

#### L11.2.3 检查 12 个六任务输出

```bash
for MODE in pure_repaint anygrasp_repaint; do for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_${TASK}_${MODE}_25ep; echo "===== h2o_${TASK}_${MODE}_25ep ====="; python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
if not root.exists():
    print("missing")
    raise SystemExit
info = json.load(open(root / "meta/info.json"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
print("first episodes:", (root / "meta/episodes.jsonl").read_text().splitlines()[:3])
PY
done; done
```

#### L11.2.4 六任务 D435 robot replay：各抽 25 episode

用途：从 L10.6 生成的 `local/h2o_<TASK>_pure_d435_visible_reinit` 中抽取 `_25ep`。这条链路对应 D435 robot replay，不要和 L11.2.1 的默认广角 `pure_repaint` 混用。

前提：

```text
I1/I1.1 Stage-1 BG 已完成
I3.4/I3.5 D435 visible-reinit repaint 已完成
L8.2 processed HDF5 已完成
L10.6 LeRobot cache 已完成
```

抽取命令：

和 L11.1.3 一样，这里不要盲目使用 `--episodes '0-24'`。D435 robot replay 的 processed HDF5 里也有 `episode_*/instructions.json/source_episode_id`，所以可以按原始 id 对齐并排除 bad id，再补足 25 个 LeRobot episode index。当前明确排除：

```text
handover_bottle 原始 bad id：0,7,12,29
pnp_bread 原始 bad id：0,1,2,3,4,5,6,22,70
pnp_tray 当前未在这里额外排除
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in handover_bottle) BAD_IDS="0,7,12,29" ;; pnp_bread) BAD_IDS="0,1,2,3,4,5,6,22,70" ;; *) BAD_IDS="" ;; esac; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_pure_d435_visible_reinit-120; SOURCE=local/h2o_${TASK}_pure_d435_visible_reinit; [[ -d "$DATASET" ]] || { echo "[skip] missing processed DATASET=$DATASET; run L8.2 first"; continue; }; [[ -d "/home/zaijia001/.cache/huggingface/lerobot/${SOURCE}" ]] || { echo "[skip] missing LeRobot SOURCE=${SOURCE}; run L10.6 first"; continue; }; EPISODES=$(python3 - "$DATASET" "$BAD_IDS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bad = {int(x) for x in sys.argv[2].split(",") if x.strip()}
keep = []

for p in sorted(root.glob("episode_*/instructions.json"), key=lambda p: int(p.parent.name.split("_")[-1])):
    ep = int(p.parent.name.split("_")[-1])
    row = json.loads(p.read_text())
    src = int(row.get("source_episode_id", ep))
    if src in bad:
        continue
    keep.append(ep)
    if len(keep) >= 25:
        break

if len(keep) < 25:
    raise SystemExit(f"only {len(keep)} usable episodes after excluding {sorted(bad)}")
print(",".join(map(str, keep)))
PY
); echo "===== subset D435 ${TASK}: episodes=${EPISODES} ====="; uv run python scripts/subset_lerobot_episodes.py --source "$SOURCE" --output-repo-id local/h2o_${TASK}_pure_d435_visible_reinit_25ep --episodes "$EPISODES" --overwrite; done
```

按当前 processed data 检查，安全抽取 episode index 是：

```text
pick_diverse_bottles: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
place_bread_basket: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
stack_cups: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
handover_bottle: 0,1,2,3,4,5,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26
pnp_bread: 0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25
pnp_tray: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
```

打包和上传检查：

```bash
cd /home/zaijia001/.cache/huggingface/lerobot/local && zip -r robot_d435_visible_reinit_6task_25ep.zip h2o_pick_diverse_bottles_pure_d435_visible_reinit_25ep h2o_place_bread_basket_pure_d435_visible_reinit_25ep h2o_stack_cups_pure_d435_visible_reinit_25ep h2o_handover_bottle_pure_d435_visible_reinit_25ep h2o_pnp_bread_pure_d435_visible_reinit_25ep h2o_pnp_tray_pure_d435_visible_reinit_25ep
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_d435_visible_reinit_6task_25ep.zip gdrive:piper/multi/6task/robot_d435_visible_reinit -P --drive-chunk-size 64M --transfers 4 --dry-run
```

检查 6 个 D435 输出是否都是 25 episode：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_${TASK}_pure_d435_visible_reinit_25ep; echo "===== h2o_${TASK}_pure_d435_visible_reinit_25ep ====="; python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
if not root.exists():
    print("missing")
    raise SystemExit
info = json.load(open(root / "meta/info.json"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
print("first episodes:", (root / "meta/episodes.jsonl").read_text().splitlines()[:3])
PY
done
```

#### L11.2.5 六任务 L16 whitebg repaint：各抽 25 episode

用途：从 L10.7 生成的 `local/h2o_<TASK>_l16_whitebg_repaint` 中抽取 `_25ep`。这条链路对应 L16 planner-style repaint，不要和 L11.2.4 的 D435 pure replay 混用。

前提：

```text
I3.6/I3.6.1 L16 whitebg repaint 已完成
L9.2 processed HDF5 已完成
L10.7 LeRobot cache 已完成
```

抽取命令和 L11.2.4 一样按 processed HDF5 的 `instructions.json/source_episode_id` 对齐原始 id，并排除已知 bad id：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in handover_bottle) BAD_IDS="0,7,12,29" ;; pnp_bread) BAD_IDS="0,1,2,3,4,5,6,22,70" ;; *) BAD_IDS="" ;; esac; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_l16_whitebg_repaint-120; SOURCE=local/h2o_${TASK}_l16_whitebg_repaint; [[ -d "$DATASET" ]] || { echo "[skip] missing processed DATASET=$DATASET; run L9.2 first"; continue; }; [[ -d "/home/zaijia001/.cache/huggingface/lerobot/${SOURCE}" ]] || { echo "[skip] missing LeRobot SOURCE=${SOURCE}; run L10.7 first"; continue; }; EPISODES=$(python3 - "$DATASET" "$BAD_IDS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bad = {int(x) for x in sys.argv[2].split(",") if x.strip()}
keep = []

for p in sorted(root.glob("episode_*/instructions.json"), key=lambda p: int(p.parent.name.split("_")[-1])):
    ep = int(p.parent.name.split("_")[-1])
    row = json.loads(p.read_text())
    src = int(row.get("source_episode_id", ep))
    if src in bad:
        continue
    keep.append(ep)
    if len(keep) >= 25:
        break

if len(keep) < 25:
    raise SystemExit(f"only {len(keep)} usable episodes after excluding {sorted(bad)}")
print(",".join(map(str, keep)))
PY
); echo "===== subset L16 ${TASK}: episodes=${EPISODES} ====="; uv run python scripts/subset_lerobot_episodes.py --source "$SOURCE" --output-repo-id local/h2o_${TASK}_l16_whitebg_repaint_25ep --episodes "$EPISODES" --overwrite; done
```

如果 `stack_cups` 最终采用 B 方案，额外抽取 B 数据集：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && TASK=stack_cups; DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_${TASK}_l16_whitebg_b_points_negative-120; SOURCE=local/h2o_${TASK}_l16_whitebg_b_points_negative; [[ -d "$DATASET" ]] || { echo "[skip] missing processed DATASET=$DATASET; run L9.2 B first"; exit 0; }; [[ -d "/home/zaijia001/.cache/huggingface/lerobot/${SOURCE}" ]] || { echo "[skip] missing LeRobot SOURCE=${SOURCE}; run L10.7 B first"; exit 0; }; EPISODES=$(python3 - "$DATASET" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
keep = []
for p in sorted(root.glob("episode_*/instructions.json"), key=lambda p: int(p.parent.name.split("_")[-1])):
    ep = int(p.parent.name.split("_")[-1])
    keep.append(ep)
    if len(keep) >= 25:
        break
if len(keep) < 25:
    raise SystemExit(f"only {len(keep)} usable episodes")
print(",".join(map(str, keep)))
PY
); echo "===== subset L16 B stack_cups: episodes=${EPISODES} ====="; uv run python scripts/subset_lerobot_episodes.py --source "$SOURCE" --output-repo-id local/h2o_${TASK}_l16_whitebg_b_points_negative_25ep --episodes "$EPISODES" --overwrite
```

打包和上传检查：

```bash
cd /home/zaijia001/.cache/huggingface/lerobot/local && zip -r robot_l16_whitebg_repaint_6task_25ep.zip h2o_pick_diverse_bottles_l16_whitebg_repaint_25ep h2o_place_bread_basket_l16_whitebg_repaint_25ep h2o_stack_cups_l16_whitebg_repaint_25ep h2o_handover_bottle_l16_whitebg_repaint_25ep h2o_pnp_bread_l16_whitebg_repaint_25ep h2o_pnp_tray_l16_whitebg_repaint_25ep
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_l16_whitebg_repaint_6task_25ep.zip gdrive:piper/multi/6task/robot_l16_whitebg_repaint -P --drive-chunk-size 64M --transfers 4 --dry-run

cd /home/zaijia001/.cache/huggingface/lerobot/local && zip -r robot_l16_whitebg_b_points_negative_stack_cups_25ep.zip h2o_stack_cups_l16_whitebg_b_points_negative_25ep
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_l16_whitebg_b_points_negative_stack_cups_25ep.zip gdrive:piper/multi/6task/robot_l16_whitebg_b_points_negative_stack_cups -P --drive-chunk-size 64M --transfers 4 --dry-run
```

检查：

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_${TASK}_l16_whitebg_repaint_25ep; echo "===== h2o_${TASK}_l16_whitebg_repaint_25ep ====="; python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
if not root.exists():
    print("missing")
    raise SystemExit
info = json.load(open(root / "meta/info.json"))
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
print("first episodes:", (root / "meta/episodes.jsonl").read_text().splitlines()[:3])
PY
done
```


#### L11.2.6 L16 ours：按 P4 review JSON 一键转换、抽取、打包

用途：用 P4 选出来的 `ours_review_selection.json` 作为过滤表，生成这次的正式 `ours` 数据集。命名规则是用 `ours` 代替之前的 `reinit`：

```text
processed HDF5: /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_<TASK>_ours-120
LeRobot:       /home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_ours
25ep subset:   /home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_ours_25ep
zip:           /home/zaijia001/.cache/huggingface/lerobot/local/robot_ours_<TASK_GROUP>_25ep.zip
rclone dst:    gdrive:piper/multi/<TASK_GROUP>/robot_ours
```

脚本位置：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/run_l16_ours_selected_pipeline.sh
```

当前先处理已有五个任务，`stack_cups` 后面 review 完再加入：

```bash
TASKS="pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray" \
TASK_GROUP=5task \
DRY_RUN=1 \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_ours_selected_pipeline.sh
```

六任务全部选完后：

```bash
TASKS="pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray stack_cups" \
TASK_GROUP=6task \
DRY_RUN=1 \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_ours_selected_pipeline.sh
```

分阶段运行：

```bash
# 只做 L9.2: final repaint + pose_debug + wrist -> processed HDF5
STEPS="process" TASKS="pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray" bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_ours_selected_pipeline.sh

# 只做 L10.7: processed HDF5 -> LeRobot
STEPS="lerobot" TASKS="pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray" bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_ours_selected_pipeline.sh

# 只做 L11.2.5 风格的 subset/zip dry-run
STEPS="subset zip" TASKS="pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray" TASK_GROUP=5task DRY_RUN=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_ours_selected_pipeline.sh
```

正式上传时把 `DRY_RUN=1` 改成 `DRY_RUN=0`。如果某个任务 accept 少于 25 条，脚本会按实际 accept 数量生成 subset 并打印 warning；正式六任务建议每个任务先在 P4 里选满 25 条。

### L11.3 task prompt 设置位置：先改 processed episode 的 `instructions.json`

注意：当前 `examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --task "..."` 对已经写好的 episode prompt 不会生效。该脚本实际读取每个 processed episode 目录里的：

```text
/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/<DATASET>/episode_*/instructions.json
```

因此 task prompt 的推荐设置顺序是：

1. 在 L5/L6/L8/L9 这类 HDF5 生成命令里设置 `INSTRUCTION="..."`，让 `process_repainted_headcam_with_wrist.py` 或 `process_repainted_planner_outputs.py` 写出正确 `instructions.json`。
2. 如果 processed data 已经生成，但还没转 LeRobot，先批量替换 `episode_*/instructions.json`，再运行 L10 的 LeRobot 转换命令。
3. 如果 LeRobot cache 已经生成，只想快速修正已生成 cache，用 L12 直接替换 `meta/tasks.jsonl` 和 `meta/episodes.jsonl`。

示例：在 LeRobot 转换前，批量把某个 processed dataset 的 prompt 改成新文本：

```bash
DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_pick_diverse_bottles_human_head_pure_action-120; PROMPT='pick up one bottle with one arm, and pick up another bottle with the other arm.'; python3 - "$DATASET" "$PROMPT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
prompt = sys.argv[2]
count = 0
for p in sorted(root.glob("episode_*/instructions.json")):
    data = json.loads(p.read_text())
    data["instructions"] = [prompt]
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    count += 1
print("updated instructions.json:", count)
print("prompt:", prompt)
PY
```

检查：

```bash
DATASET=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_pick_diverse_bottles_human_head_pure_action-120; find "$DATASET" -name instructions.json | sort | head -n 3 | xargs -r -I{} sh -c 'echo ===== {}; sed -n "1,20p" {}'
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

### L13. AnyGrasp Piper keyframe planner：pick_diverse_bottles id0-id10（旧版，保留历史）

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

注意：本小节命令含 `--keyframes 38 78`，只适用于当时 id0 的临时实验，不应作为批量命令使用。新命令见文末 `L15`，会通过 `--reuse_preview_frame_mode annotated_json_keyframes` 读取每个 id 手动标注的两个关键帧。

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

### L15. Piper AnyGrasp id0-id10：按手动标注关键帧、双臂同阶段同步执行

用途：批量运行 `pick_diverse_bottles` 的 `id0-id10`。本命令不写 `--keyframes 38 78`；关键帧来自每个 id 对应 preview summary 里的 `frame_selection.annotated_keyframes[:2]`。代码侧 `--dual_stage_require_all_plans 1` 表示每个 stage 必须左右臂都规划成功才会一起执行，避免某一只手 IK 失败时另一只手先单独移动。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && GPU=2; TASK=pick_diverse_bottles; OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_v3/4/${TASK}; for ID in $(seq 0 10); do ANY=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output/foundation_input_${ID}; REPLAY=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay/foundation_input_${ID}; HAND=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; PREVIEW=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/${TASK}/foundation_input_${ID}/summary.json; OUT=${OUT_ROOT}/foundation_input_${ID}; [[ -d "$ANY" ]] || { echo "[skip] missing anygrasp $ANY"; continue; }; [[ -d "$REPLAY" ]] || { echo "[skip] missing replay $REPLAY"; continue; }; [[ -f "$HAND" ]] || { echo "[skip] missing hand $HAND"; continue; }; [[ -f "$PREVIEW" ]] || { echo "[skip] missing preview $PREVIEW"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py --anygrasp_dir "$ANY" --replay_dir "$REPLAY" --hand_npz "$HAND" --output_dir "$OUT" --reuse_preview_summary_json "$PREVIEW" --reuse_preview_frame_mode annotated_json_keyframes --reuse_preview_candidate_group orientation --reuse_preview_top_rank 1 --arm auto --execute_both_arms 1 --dual_stage_require_all_plans 1 --planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik --urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.02 --urdfik_max_position_threshold_m 0.02 --urdfik_max_rotation_threshold_rad 0.12 --candidate_selection_mode planner --left_target_object left_bottle --right_target_object right_bottle --candidate_target_local_x_offset_m -0.05 --approach_offset_m 0.12 --reach_error_pose_source tcp --replan_until_reached 1 --replan_until_reached_max_attempts 3 --save_debug_preview 1 --save_debug_execution_preview 1 --save_pose_debug 1 --debug_visualize_targets 1 --debug_visualize_ik_waypoints 1 --reach_pos_tol_m 0.03 --reach_rot_tol_deg 180 --settle_steps 100 --joint_target_wait_steps 100 --joint_command_scene_steps 4 --execute_interp_steps 40 --hold_frames_after_stage 8 --pure_scene_output 1 --overlay_text 0 --head_only 0 --third_person_view 1 --vscode_compatible_video 1 --lighting_mode front_no_shadow --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --enable_viewer 0 --viewer_wait_at_end 0 --object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj; done
```

为什么 `--settle_steps 100 --joint_target_wait_steps 100` 仍可能不到位：

- 它们只是在已有 joint trajectory 执行完以后继续推进仿真/等待关节收敛，不能把 IK 失败的目标变成可达目标。
- 如果日志里有 `Failed to converge` 或某个 arm 的 plan status 不是 `Success`，等待再久也不会生成可执行轨迹。
- 如果 `reach_rot_tol_deg` 很小，而候选朝向和当前 TCP 差到 160 度左右，位置即使靠近也会被判定 `reached=0`。位置优先调试先用 `--reach_rot_tol_deg 180`，之后再收紧。
- 视频里想看到阶段结束后停留，需要 `--hold_frames_after_stage`；`settle_steps` 本身不等价于多写 100 帧视频。

### L15.1 Piper AnyGrasp id0-id10：viewer 可视化版，对齐旧 V7 执行节奏

用途：和 L15 一样按每个 id 的手动标注关键帧执行，但打开 viewer，并把执行节奏改得更接近旧 V7：`execute_interp_steps=24`、`joint_command_scene_steps=10`、`settle_steps=30`、`joint_target_wait_steps=25`。同时启用 `--require_keyframe1_reached_before_action 1`，如果第一关键帧 grasp 未到位，就不进入第二关键帧 action。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && GPU=2; TASK=pick_diverse_bottles; OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_v5_viewer/${TASK}; for ID in $(seq 0 10); do ANY=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output/foundation_input_${ID}; REPLAY=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay/foundation_input_${ID}; HAND=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; PREVIEW=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/${TASK}/foundation_input_${ID}/summary.json; OUT=${OUT_ROOT}/foundation_input_${ID}; [[ -d "$ANY" ]] || { echo "[skip] missing anygrasp $ANY"; continue; }; [[ -d "$REPLAY" ]] || { echo "[skip] missing replay $REPLAY"; continue; }; [[ -f "$HAND" ]] || { echo "[skip] missing hand $HAND"; continue; }; [[ -f "$PREVIEW" ]] || { echo "[skip] missing preview $PREVIEW"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py --anygrasp_dir "$ANY" --replay_dir "$REPLAY" --hand_npz "$HAND" --output_dir "$OUT" --reuse_preview_summary_json "$PREVIEW" --reuse_preview_frame_mode annotated_json_keyframes --reuse_preview_candidate_group orientation --reuse_preview_top_rank 1 --arm auto --execute_both_arms 1 --dual_stage_require_all_plans 1 --require_keyframe1_reached_before_action 1 --planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik --urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.01 --urdfik_max_position_threshold_m 0.02 --urdfik_max_rotation_threshold_rad 0.12 --candidate_selection_mode planner --left_target_object left_bottle --right_target_object right_bottle --candidate_target_local_x_offset_m -0.05 --approach_offset_m 0.12 --reach_error_pose_source tcp --replan_until_reached 1 --replan_until_reached_max_attempts 1 --save_debug_preview 1 --save_debug_execution_preview 0 --save_pose_debug 1 --debug_visualize_targets 0 --debug_visualize_ik_waypoints 1 --reach_pos_tol_m 0.03 --reach_rot_tol_deg 180 --enable_grasp_action_object_collision 1 --grasp_action_object_collision_start_stage pregrasp --execution_object_collision_mode convex --execution_object_visual_scale_override left_bottle=0.8 --execution_object_collision_scale_override left_bottle=0.8 --execution_object_visual_scale_override right_bottle=0.8 --execution_object_collision_scale_override right_bottle=0.8 --gripper_contact_monitor_mode all_robot_links --execute_interp_steps 24 --joint_command_scene_steps 10 --settle_steps 30 --joint_target_wait_steps 25 --joint_target_wait_tol_rad 0.01 --hold_frames_after_stage 8 --pure_scene_output 1 --overlay_text 0 --head_only 0 --third_person_view 1 --vscode_compatible_video 1 --lighting_mode front_no_shadow --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --enable_viewer 1 --viewer_wait_at_end 0 --viewer_show_camera_frustums 0 --object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj; done
```

和旧 V7 的关系：

- 旧 V7 用的是 `cup/bottle` 和 R1 数据路径；这里是 Piper H2O `left_bottle/right_bottle`。
- 旧 V7 的执行节奏是 `execute_interp_steps=24`、每 waypoint `joint_command_scene_steps=10`、`settle_steps=30`、`joint_target_wait_steps=25`；L15.1 已按这个节奏设置。
- 旧 V7 的实际期望是“第一关键帧到位后再进入第二关键帧”。L15.1 用 `--dual_stage_require_all_plans 1` 和 `--require_keyframe1_reached_before_action 1` 把这个行为显式化：任一 arm 规划失败则该 stage 不执行，第一关键帧未 reached 则不进入第二关键帧。

### L15.2 Piper AnyGrasp 最终推荐命令：无 viewer 批跑版 + viewer 单条调试版

用途：这组命令是 L15/L15.1 的整理版。无 viewer 版本用于稳定批跑 id0-id10；viewer 版本用于单条交互检查。注意 SAPIEN viewer 需要能看到驱动 VNC/X display 的 GPU，所以 viewer 版本不要设置 `CUDA_VISIBLE_DEVICES=2`，而是先 `unset CUDA_VISIBLE_DEVICES`。

无 viewer 批跑版：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && GPU=2; TASK=pick_diverse_bottles; OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_v6_noviewer/${TASK}; for ID in $(seq 0 10); do ANY=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output/foundation_input_${ID}; REPLAY=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay/foundation_input_${ID}; HAND=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; PREVIEW=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/${TASK}/foundation_input_${ID}/summary.json; OUT=${OUT_ROOT}/foundation_input_${ID}; [[ -d "$ANY" ]] || { echo "[skip] missing anygrasp $ANY"; continue; }; [[ -d "$REPLAY" ]] || { echo "[skip] missing replay $REPLAY"; continue; }; [[ -f "$HAND" ]] || { echo "[skip] missing hand $HAND"; continue; }; [[ -f "$PREVIEW" ]] || { echo "[skip] missing preview $PREVIEW"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py --anygrasp_dir "$ANY" --replay_dir "$REPLAY" --hand_npz "$HAND" --output_dir "$OUT" --reuse_preview_summary_json "$PREVIEW" --reuse_preview_frame_mode annotated_json_keyframes --reuse_preview_candidate_group orientation --reuse_preview_top_rank 1 --arm auto --execute_both_arms 1 --dual_stage_require_all_plans 1 --require_keyframe1_reached_before_action 1 --planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik --urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.01 --urdfik_max_position_threshold_m 0.02 --urdfik_max_rotation_threshold_rad 0.12 --candidate_selection_mode planner --left_target_object left_bottle --right_target_object right_bottle --candidate_target_local_x_offset_m -0.05 --approach_offset_m 0.12 --reach_error_pose_source tcp --replan_until_reached 1 --replan_until_reached_max_attempts 1 --save_debug_preview 1 --save_debug_execution_preview 0 --save_pose_debug 1 --debug_visualize_targets 0 --debug_visualize_ik_waypoints 1 --reach_pos_tol_m 0.03 --reach_rot_tol_deg 180 --enable_grasp_action_object_collision 1 --grasp_action_object_collision_start_stage pregrasp --execution_object_collision_mode convex --execution_object_visual_scale_override left_bottle=0.8 --execution_object_collision_scale_override left_bottle=0.8 --execution_object_visual_scale_override right_bottle=0.8 --execution_object_collision_scale_override right_bottle=0.8 --gripper_contact_monitor_mode all_robot_links --execute_interp_steps 24 --joint_command_scene_steps 10 --settle_steps 30 --joint_target_wait_steps 25 --joint_target_wait_tol_rad 0.01 --hold_frames_after_stage 8 --pure_scene_output 1 --overlay_text 0 --head_only 0 --third_person_view 1 --vscode_compatible_video 1 --lighting_mode front_no_shadow --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --enable_viewer 0 --viewer_wait_at_end 0 --viewer_show_camera_frustums 0 --object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj; done
```

viewer 单条调试版：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}"; TASK=pick_diverse_bottles; ID=0; ANY=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output/foundation_input_${ID}; REPLAY=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay/foundation_input_${ID}; HAND=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; PREVIEW=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/${TASK}/foundation_input_${ID}/summary.json; OUT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_v6_viewer/${TASK}/foundation_input_${ID}; conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py --anygrasp_dir "$ANY" --replay_dir "$REPLAY" --hand_npz "$HAND" --output_dir "$OUT" --reuse_preview_summary_json "$PREVIEW" --reuse_preview_frame_mode annotated_json_keyframes --reuse_preview_candidate_group orientation --reuse_preview_top_rank 1 --arm auto --execute_both_arms 1 --dual_stage_require_all_plans 1 --require_keyframe1_reached_before_action 1 --planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik --urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.01 --urdfik_max_position_threshold_m 0.02 --urdfik_max_rotation_threshold_rad 0.12 --candidate_selection_mode planner --left_target_object left_bottle --right_target_object right_bottle --candidate_target_local_x_offset_m -0.05 --approach_offset_m 0.12 --reach_error_pose_source tcp --replan_until_reached 1 --replan_until_reached_max_attempts 1 --save_debug_preview 1 --save_debug_execution_preview 0 --save_pose_debug 1 --debug_visualize_targets 1 --debug_visualize_ik_waypoints 1 --reach_pos_tol_m 0.03 --reach_rot_tol_deg 180 --enable_grasp_action_object_collision 1 --grasp_action_object_collision_start_stage pregrasp --execution_object_collision_mode convex --execution_object_visual_scale_override left_bottle=0.8 --execution_object_collision_scale_override left_bottle=0.8 --execution_object_visual_scale_override right_bottle=0.8 --execution_object_collision_scale_override right_bottle=0.8 --gripper_contact_monitor_mode all_robot_links --execute_interp_steps 24 --joint_command_scene_steps 10 --settle_steps 30 --joint_target_wait_steps 25 --joint_target_wait_tol_rad 0.01 --hold_frames_after_stage 8 --pure_scene_output 1 --overlay_text 0 --head_only 0 --third_person_view 1 --vscode_compatible_video 1 --lighting_mode front_no_shadow --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02 --viewer_show_camera_frustums 0 --object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj
```

viewer 如果仍然不弹窗，先跑最小探针：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py
```

### L15.3 Piper AnyGrasp D435：使用 J1.1/J1.2 的 D435 summary 和 D435 replay

用途：重新尝试 D435 链路时使用这一条，不要用 L15/L15.2 的默认广角路径。本命令有三个强约束：

- `REPLAY` 指向 `/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435/foundation_input_${ID}`
- `PREVIEW` 指向 `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}/foundation_input_${ID}/summary.json`
- 渲染相机使用 D435 color 内参：`--image_width 640 --image_height 480 --fovy_deg 42.499880046655484`

如果某个 id 没有 D435 `summary.json`，命令会直接 skip；这表示应该先回到 J1.1/J1.2 重新生成候选，而不是 fallback 到默认广角 `anygrasp_h2o_preview`。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && GPU=2; TASK=pick_diverse_bottles; IDS=($(seq 0 10)); OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1/${TASK}; case "$TASK" in pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle; MESH_ARGS=(--object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj) ;; pnp_tray) LEFT_OBJ=left_dark_red_cup; RIGHT_OBJ=right_bottle; MESH_ARGS=(--object_mesh_override left_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj) ;; *) echo "[error] add object mapping for TASK=$TASK"; exit 1 ;; esac; for ID in "${IDS[@]}"; do ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output; [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam; ANY=${ANY_ROOT}/foundation_input_${ID}; REPLAY=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435/foundation_input_${ID}; HAND=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; PREVIEW=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}/foundation_input_${ID}/summary.json; OUT=${OUT_ROOT}/foundation_input_${ID}; [[ -d "$ANY" ]] || { echo "[skip] id=${ID} missing anygrasp $ANY"; continue; }; [[ -d "$REPLAY" ]] || { echo "[skip] id=${ID} missing D435 replay $REPLAY"; continue; }; [[ -f "$HAND" ]] || { echo "[skip] id=${ID} missing hand $HAND"; continue; }; [[ -f "$PREVIEW" ]] || { echo "[skip] id=${ID} missing D435 preview summary $PREVIEW; run J1.1/J1.2 first"; continue; }; echo "[run-d435] task=${TASK} id=${ID} preview=${PREVIEW} replay=${REPLAY}"; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py --anygrasp_dir "$ANY" --replay_dir "$REPLAY" --hand_npz "$HAND" --output_dir "$OUT" --reuse_preview_summary_json "$PREVIEW" --reuse_preview_frame_mode annotated_json_keyframes --reuse_preview_candidate_group orientation --reuse_preview_top_rank 1 --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --arm auto --execute_both_arms 1 --dual_stage_require_all_plans 1 --require_keyframe1_reached_before_action 1 --planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik --urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.01 --urdfik_max_position_threshold_m 0.02 --urdfik_max_rotation_threshold_rad 0.12 --candidate_selection_mode planner --left_target_object "$LEFT_OBJ" --right_target_object "$RIGHT_OBJ" --candidate_target_local_x_offset_m -0.05 --approach_offset_m 0.12 --reach_error_pose_source tcp --replan_until_reached 1 --replan_until_reached_max_attempts 1 --save_debug_preview 1 --save_debug_execution_preview 0 --save_pose_debug 1 --debug_visualize_targets 0 --debug_visualize_ik_waypoints 1 --reach_pos_tol_m 0.03 --reach_rot_tol_deg 180 --enable_grasp_action_object_collision 1 --grasp_action_object_collision_start_stage pregrasp --execution_object_collision_mode convex --execution_object_visual_scale_override ${LEFT_OBJ}=0.8 --execution_object_collision_scale_override ${LEFT_OBJ}=0.8 --execution_object_visual_scale_override ${RIGHT_OBJ}=0.8 --execution_object_collision_scale_override ${RIGHT_OBJ}=0.8 --gripper_contact_monitor_mode all_robot_links --execute_interp_steps 24 --joint_command_scene_steps 10 --settle_steps 30 --joint_target_wait_steps 25 --joint_target_wait_tol_rad 0.01 --hold_frames_after_stage 8 --pure_scene_output 1 --overlay_text 0 --head_only 0 --third_person_view 1 --vscode_compatible_video 1 --lighting_mode front_no_shadow --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --enable_viewer 0 --viewer_wait_at_end 0 --viewer_show_camera_frustums 0 "${MESH_ARGS[@]}"; done
```

检查 D435 planner 输出：

```bash
TASK=pick_diverse_bottles; for ID in $(seq 0 10); do OUT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1/${TASK}/foundation_input_${ID}; echo "===== id=${ID} ====="; [[ -f "$OUT/plan_summary.json" ]] && jq '{success:.execution_success, failed:.execution_failed, failed_stage_records:.failed_stage_records, preview:.reuse_preview_summary_json}' "$OUT/plan_summary.json" || echo "missing $OUT/plan_summary.json"; ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/head_cam_plan.mp4" 2>/dev/null || true; done
```

单条 viewer 调试版把上面命令改成 `ID=18` 这类已确认有 D435 summary 的 id，并替换末尾 viewer 参数：

```bash
--enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02 --viewer_show_camera_frustums 0
```

### L15.4 Piper AnyGrasp D435：六任务从 summary 到执行

用途：这是 J1.1/J1.2 之后的六任务 D435 planner 入口。执行链是：

```text
C1.2 生成 foundation_replay_d435
J0.1 检查 AnyGrasp grasps + D435 replay + HaMeR 是否齐全
J1.1/J1.2 生成 anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json
L15.4 使用 D435 summary + D435 replay 执行 Piper AnyGrasp planner
```

这条命令只跑已经存在 D435 summary 的 id；如果某个任务 summary 数量为 0，先回到 J1.1 重新生成。不要在 zsh 里直接粘贴含 `mapfile` 的旧版长命令；`mapfile` 是 bash 内建。现在使用脚本入口，zsh 中也用 `bash ...` 调用。

先 dry-run 检查 6 个任务会跑哪些 id，不真正执行 planner：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run
```

6 个任务各跑 1 个 id 做 smoke test：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 3
```

只跑某个任务的前 1 个 summary：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --tasks pick_diverse_bottles
```

六任务全量执行：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2
```

六个任务各跑前 5 个 D435 summary，单个 id 出错也继续后面的任务：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error
```

六个任务分别跑前 5 个 D435 summary：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pick_diverse_bottles
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks place_bread_basket
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks stack_cups
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks handover_bottle
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pnp_bread
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pnp_tray
```

输出检查，不依赖 `jq`：

```bash
python3 - <<'PY'
import json
from pathlib import Path
tasks = ["pick_diverse_bottles", "place_bread_basket", "stack_cups", "handover_bottle", "pnp_bread", "pnp_tray"]
root = Path("/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1")
for task in tasks:
    print(f"===== {task} =====")
    for summary in sorted((root / task).glob("foundation_input_*/plan_summary.json"))[:20]:
        data = json.load(open(summary))
        print(summary.parent.name, "success=", data.get("execution_success"), "failed=", data.get("execution_failed"), "preview=", data.get("reuse_preview_summary_json"))
PY
```

### L15.5 Piper AnyGrasp D435：stack_cups id0 viewer 单条调试

用途：先用 `stack_cups/foundation_input_0` 交互确认 D435 summary、per-arm keyframes、rank preview 和执行目标。这个 id 的人工关键帧是：

```text
right: 51, 106
left: 139, 195
```

planner 的 `rank_previews` 应包含 4 张图：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1/stack_cups/foundation_input_0/rank_previews/keyframe_000051_rank_1.png
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1/stack_cups/foundation_input_0/rank_previews/keyframe_000106_rank_1.png
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1/stack_cups/foundation_input_0/rank_previews/keyframe_000139_rank_1.png
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1/stack_cups/foundation_input_0/rank_previews/keyframe_000195_rank_1.png
```

对应的 J1.1 D435 原始 preview 图已经复制到：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1/stack_cups/foundation_input_0/preview_compare_d435/
```

注意：planner `rank_previews` 是 SAPIEN 里渲染的 3D 夹爪模型；J1.1 preview 是 raw D435 图片上的候选投影。两者视觉样式不同。`--approach_offset_m 0.12` 只影响 pregrasp，不影响 rank preview；rank preview 里的目标位置包含 `--candidate_target_local_x_offset_m -0.05` 的 5 cm TCP 补偿。

先测 viewer 环境：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py
```

如果这里打印 `DISPLAY=` 为空并报 `Renderer does not support display`，说明当前 shell 没有连到图形会话；需要在本机图形终端运行，或先正确设置 X11/Wayland forwarding。这个报错和 AnyGrasp/planner 参数无关。

viewer 单条调试命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; TASK=stack_cups; ID=0; ANY=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output/foundation_input_${ID}; REPLAY=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435/foundation_input_${ID}; HAND=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; PREVIEW=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}/foundation_input_${ID}/summary.json; OUT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_viewer/${TASK}/foundation_input_${ID}; conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py --anygrasp_dir "$ANY" --replay_dir "$REPLAY" --hand_npz "$HAND" --output_dir "$OUT" --reuse_preview_summary_json "$PREVIEW" --reuse_preview_frame_mode annotated_json_keyframes --reuse_preview_candidate_group orientation --reuse_preview_top_rank 1 --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --arm auto --execute_both_arms 1 --dual_stage_require_all_plans 1 --require_keyframe1_reached_before_action 1 --planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik --urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.01 --urdfik_max_position_threshold_m 0.02 --urdfik_max_rotation_threshold_rad 0.12 --candidate_selection_mode planner --left_target_object left_light_pink_cup --right_target_object right_dark_red_cup --candidate_target_local_x_offset_m -0.05 --approach_offset_m 0.12 --reach_error_pose_source tcp --replan_until_reached 1 --replan_until_reached_max_attempts 1 --save_debug_preview 1 --save_debug_execution_preview 0 --save_pose_debug 1 --debug_visualize_targets 1 --debug_visualize_ik_waypoints 1 --reach_pos_tol_m 0.03 --reach_rot_tol_deg 180 --enable_grasp_action_object_collision 1 --grasp_action_object_collision_start_stage pregrasp --execution_object_collision_mode convex --execution_object_visual_scale_override left_light_pink_cup=0.8 --execution_object_collision_scale_override left_light_pink_cup=0.8 --execution_object_visual_scale_override right_dark_red_cup=0.8 --execution_object_collision_scale_override right_dark_red_cup=0.8 --gripper_contact_monitor_mode all_robot_links --execute_interp_steps 24 --joint_command_scene_steps 10 --settle_steps 30 --joint_target_wait_steps 25 --joint_target_wait_tol_rad 0.01 --hold_frames_after_stage 8 --pure_scene_output 1 --overlay_text 0 --head_only 0 --third_person_view 1 --vscode_compatible_video 1 --lighting_mode front_no_shadow --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02 --viewer_show_camera_frustums 0 --object_mesh_override left_light_pink_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj --object_mesh_override right_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj
```

### L15.6 Piper AnyGrasp D435：六任务 5 episode viewer/no-viewer 与第一关键帧 debug

本节使用统一脚本：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh
```

它会从每个任务的 `anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json` 读取 D435 候选和人工关键帧，不再手写 `--keyframes 38 78`。脚本内部已支持：

```text
--viewer                         打开 SAPIEN viewer，默认输出到 anygrasp_plan_keyframes_piper_d435_viewer
--output_root PATH               覆盖输出根目录
--debug_stop_after_keyframe1      只执行 init -> pregrasp -> grasp，不关爪、不进入第二关键帧
--continue_on_error              单个 id 失败后继续后续 id/task
--trajectory_mode MODE            cartesian_interp_ik 或 joint_interp
--cartesian_auto_step_m M         cartesian_interp_ik 的自动插值步长，默认 0.01
--joint_interp_waypoints N        joint_interp 的关节插值点数，默认 40
--replan_attempts N               每个 stage 的重规划次数，默认 1
--allow_partial_dual_stage        允许单臂 plan 成功时单臂执行，仅用于诊断，不推荐做最终数据
```

脚本默认同时传：

```text
--require_keyframe1_reached_before_close 1
--require_keyframe1_reached_before_action 1
```

这表示第一关键帧 grasp 未 reached 时，不关夹爪，也不进入第二关键帧。这样 viewer 里不会再出现“stage 没执行到位但马上闭合夹爪”的误导画面。

无 viewer：六任务各跑前 5 个 D435 summary：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/1
```

viewer：六任务各跑前 5 个 D435 summary。注意必须在图形终端或正确 X11/Wayland forwarding 环境中运行；如果 `DISPLAY=` 为空，先不要跑 viewer：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_viewer
```

viewer 中如果只是想确认“机械臂是否会动”，建议先用更接近旧 R1/V7 观感的关节插值模式。它只要求最终点 IK 成功，不会像 `cartesian_interp_ik` 那样要求每 1 cm 中间 waypoint 都 IK 成功：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --trajectory_mode joint_interp --joint_interp_waypoints 40 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_viewer_joint_interp
```

如果仍然使用 `cartesian_interp_ik`，但想减少中间 waypoint 失败，可以先把自动步长从 1 cm 放宽到 3 cm，并允许最多 3 次重规划：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --cartesian_auto_step_m 0.03 --replan_attempts 3 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_cart003
```

第一关键帧 debug：六任务各跑前 5 个，只从初始位置规划/执行到第一关键帧的 pregrasp/grasp；不关爪，也不进入第二关键帧。这个命令用于判断问题是否已经发生在第一关键帧 IK/轨迹阶段：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --debug_stop_after_keyframe1 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_keyframe1_debug
```

如果只调一个任务，例如 `stack_cups`：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --continue_on_error --tasks stack_cups --debug_stop_after_keyframe1
```

结果检查：

```bash
python3 - <<'PY'
import json
from pathlib import Path
roots = [
    Path('/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1'),
    Path('/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_keyframe1_debug'),
]
for root in roots:
    print(f"===== {root} =====")
    for p in sorted(root.glob('*/*/plan_summary.json'))[:60]:
        d = json.load(open(p))
        print(p.parent.parent.name, p.parent.name, "success=", d.get("execution_success"), "failed=", d.get("execution_failed"), "debug_kf1=", d.get("debug_stop_after_keyframe1"), "video=", d.get("head_video"))
        for rec in d.get("failed_stage_records", []):
            print("  ", rec.get("arm"), rec.get("stage"), rec.get("status"), "p=", rec.get("pos_err_m"), "r=", rec.get("rot_err_deg"))
PY
```

`stack_cups id0` 当前定位结果：第一关键帧就没有执行到位，不是 `joint_target_wait_steps/settle_steps` 太小。`cartesian_interp_ik` 会把当前 TCP 到目标之间按 `--urdfik_cartesian_interp_auto_step_m 0.01` 插成很多 1 cm waypoint；只要中间 waypoint IK 失败，整个 plan 就是 `Fail`。实测第一关键帧：

```text
pregrasp: left failed during cartesian waypoint IK waypoint=13/23, right waypoint=28/48
grasp:    left failed during cartesian waypoint IK waypoint=16/28, right waypoint=25/45
```

因为 `--dual_stage_require_all_plans 1`，任意 arm 的 plan 失败都会跳过整个双臂 stage，所以视频里看起来“几乎没有执行 waypoint”。这是严格同步门控的结果，不是 wait steps 没生效。

参数含义：

```text
execute_interp_steps / joint_command_scene_steps / settle_steps / joint_target_wait_steps
```

这些都只在 plan 已经是 `Success` 后才影响执行过程。如果日志里已经出现 `[plan-fail]` 或 `[dual-plan] skip stage execution`，这些 step 设置再大也不会让机械臂动起来。此时要改的是 IK/轨迹模式，例如 `joint_interp`、更大的 `cartesian_auto_step_m`、更宽松的候选/姿态选择，或者重新选可达的 AnyGrasp candidate。

当前脚本已恢复 R1/V7 风格执行节奏：

```text
--execute_interp_steps 24
--joint_command_scene_steps 10
--settle_steps 30
--joint_target_wait_steps 25
```

不要把 `execute_interp_steps` 和 `joint_command_scene_steps` 同时设到几千。之前出现过实际命令为 `--execute_interp_steps 2400 --joint_command_scene_steps 1000`，这会让每个 stage 变成极长 physics stepping，viewer 看起来像卡在 waypoint。

如果要在终端确认 TCP/EE 是否真的移动，加 `--print_pose_every N`。例如先允许单臂成功时执行，用来验证命令链和 viewer/pose 更新是否正常：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/3-pdb
```

期望看到类似：

```text
[exec-pose] stage=pregrasp_try1 step=1/25 ... right:tcp=(+0.5611,-0.0440,+0.9314)
[exec-pose] stage=pregrasp_try1 step=25/25 ... right:tcp=(+0.2808,+0.2035,+1.1518)
[exec-pose] stage=grasp_try1 step=25/25 ... right:tcp=(+0.1905,+0.2235,+1.0345)
```

这说明执行链是会动的。若不开 `--allow_partial_dual_stage`，当前 `pick_diverse_bottles id0` 左臂 first-keyframe plan 失败，严格双臂同步会整体跳过 stage，因此 TCP 不变；这是同步门控预期行为。最终数据建议仍保持严格同步，`--allow_partial_dual_stage` 只用于诊断。

### L15.7 Piper AnyGrasp D435：当前关键帧执行逻辑、EE 到位判定与六任务分开运行

当前推荐使用 L15.6 脚本入口：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh
```

关键修正记录：

- D435/Piper 脚本默认 `--reach_error_pose_source ee`。原因是当前 AnyGrasp 目标经过 `_trans_from_gripper_to_endlink()` 后实际送入 IK 的是 Piper wrist/endlink 约定；如果用 `tcp` 做 reached 检查，会固定留下约 12 cm 的 TCP/EE 偏移，表现为机械臂看起来已经到夹爪位置附近但日志仍显示 `pos≈0.12m, rot≈180deg`。
- `target_pose_for_error(..., ee)` 已按 arm 使用 `world_pose_to_base_pose_for_arm/base_pose_to_world_pose_for_arm`，避免右臂目标被错误转换到左臂 base。
- partial 诊断模式下，如果一只 arm plan 失败，另一只 arm 执行时会持续 hold 失败 arm 的当前关节，避免失败臂在物理仿真中漂移。
- `plan-solution` 现在用 target joints 的 FK/TCP 评估，不再把 `target_pose_world` 直接回填成 planned pose。

当前关键帧执行逻辑：

1. J1.1/J1.2 先读取人工标注，生成 `anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`。
2. planner 只复用这个 D435 `summary.json`，不再手写 `--keyframes 38 78`；如果 summary 内有 `effective_keyframes_by_arm`，左右手分别使用各自的前两个 effective keyframes。
3. 第一关键帧执行顺序是 `pregrasp -> grasp`。`pregrasp` 是在 grasp 目标基础上按 `--approach_offset_m` 做后退；`grasp` 是 rank1 候选加 `--candidate_target_local_x_offset_m` 后的目标。
4. 如果 `--dual_stage_require_all_plans 1`，双臂 stage 要求左右臂 plan 都是 `Success` 才执行；任一 arm 失败则该 stage 两只手都不动。
5. 如果第一关键帧 grasp 没有 reached，脚本默认 `--require_keyframe1_reached_before_close 1 --require_keyframe1_reached_before_action 1`，所以不会闭合夹爪，也不会进入第二关键帧 action。
6. 第二关键帧 action 只有在第一关键帧 grasp reached 后才执行，避免第一关键帧没到位就开始第二关键帧 replay。

`pick_diverse_bottles id0` 的复查结论：

- 在旧 `tcp` reached 检查下，右臂最终 EE 已接近目标，但 TCP 相对 EE 多了约 12 cm 偏移，所以日志显示 right grasp `pos≈0.125m, rot≈179.9deg`。
- 切到 `ee` 并修正 per-arm base 后，右臂 grasp 到位误差为 `pos≈0.0057m`，说明 waypoint 执行链本身可以动并能到右臂目标。
- id0 仍整体失败的原因是左臂第一关键帧 IK/目标失败，严格双臂同步会阻断整个 stage。`--allow_partial_dual_stage` 只用于诊断时确认右臂可达，不建议作为最终数据设置。

六任务分别跑前 5 个，严格同步、无 viewer：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pick_diverse_bottles --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks place_bread_basket --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks stack_cups --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks handover_bottle --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pnp_bread --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pnp_tray --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee
```

六任务分别跑前 5 个，partial 诊断 + `joint_interp` + pose 打印：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/partial-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks place_bread_basket --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/partial-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks stack_cups --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/partial-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks handover_bottle --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/partial-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pnp_bread --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/partial-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pnp_tray --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/partial-ee
```

viewer 版本同样建议先用单任务和 partial 诊断确认显示环境与运动链路：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee
```

六任务分别跑前 5 个，viewer + partial 诊断 + 目标 gripper/axis 可视化：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --tasks place_bread_basket --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --tasks stack_cups --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --tasks handover_bottle --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --tasks pnp_bread --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --tasks pnp_tray --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee
```

`--visualize_targets` 会自动把 `pure_scene_output` 关掉，并传 `--debug_visualize_targets 1`，viewer 中会显示当前 stage 的目标 axis 和 active frame 的候选 gripper。不开这个开关时，脚本默认保留 clean video，目标可视化不会出现在主 viewer/head 视频里。

最佳候选对应关系会自动保存到输出目录：

```text
<OUT>/source_preview_compare/
  frame_000038_d435_orientation_rank.png
  frame_000038_d435_fused_rank.png
  frame_000038_legacy_orientation_rank.png
  frame_000038_legacy_fused_rank.png
  selected_candidate_mapping.json
```

结果快速检查：

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path('/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2')
for p in sorted(root.glob('*/*/foundation_input_*/plan_summary.json')):
    d = json.load(open(p))
    print(p, 'success=', d.get('execution_success'), 'reach_pose=', d.get('reach_error_pose_source'), 'failed=', d.get('failed_stage_records'))
PY
```

### L15.8 AnyGrasp 候选可视化与 planner 映射一致性排查：D435 preview、local-X offset、replay 指令

本轮排查结论：

- 用户截图中的 `source_frame=38 left_candidate=16 right_candidate=11` 对应的是 D435 preview：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/pick_diverse_bottles/foundation_input_0/summary.json
```

- 它不对应默认广角 preview：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview/pick_diverse_bottles/foundation_input_0/summary.json
```

默认广角 frame 38 的 rank1 candidate id 和 D435 不同。因此调 D435 planner 时必须复用 `anygrasp_h2o_preview_d435`，不能拿 `anygrasp_h2o_preview` 的图去找对应候选。

发现的真实不一致点：

- `render_anygrasp_ranked_preview.py` 之前在 summary/world target 中应用了 `--candidate_target_local_x_offset_m -0.05`，但是图片上的 grasp wireframe 仍然画原始 AnyGrasp `translation_cam/rotation_matrix`。
- planner 复用 summary 时使用的是已经带 local-X offset 的 `translation_world`，所以会出现“源 AnyGrasp 图片看着在一个位置，planner/rank_previews 里夹爪靠后一点”的现象。
- 现在已修正：preview 图片绘制也使用同一套 remap/post-rot/local-X offset 后的 camera-frame target pose，并在 summary 中同时写入：

```text
translation_cam              原始 AnyGrasp 相机坐标
visual_translation_cam       实际绘制、summary world target、planner 使用的相机坐标
translation_world            实际 planner target world 坐标
rotation_matrix              原始 AnyGrasp 旋转
visual_rotation_matrix       实际绘制/目标旋转
```

`pick_diverse_bottles id0 frame 38` 的验证值：

```text
left_orientation rank1 candidate=16
  raw_cam    = [-0.137713, -0.049211, 0.378072]
  visual_cam = [-0.149713, -0.000824, 0.374238]
  world      = [-0.064218, 0.029608, 0.893666]

right_orientation rank1 candidate=11
  raw_cam    = [0.141768, -0.098418, 0.359000]
  visual_cam = [0.137988, -0.112478, 0.311167]
  world      = [0.223036, 0.106104, 0.997832]
```

这说明 5cm offset 已经明确体现在 `visual_cam/world` 中。后续对比应看 `visual_translation_cam` 或 planner 输出的 `<OUT>/source_preview_compare/selected_candidate_mapping.json`。

D435 AnyGrasp 候选 preview 重新生成：六任务一起跑。这个命令会重新生成带修正后 wireframe 的 `anygrasp_h2o_preview_d435`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle ;; place_bread_basket) LEFT_OBJ=basket; RIGHT_OBJ=bread ;; stack_cups) LEFT_OBJ=left_light_pink_cup; RIGHT_OBJ=right_dark_red_cup ;; handover_bottle) LEFT_OBJ=right_bottle; RIGHT_OBJ=right_bottle ;; pnp_bread) LEFT_OBJ=left_bread; RIGHT_OBJ=right_bread ;; pnp_tray) LEFT_OBJ=left_dark_red_cup; RIGHT_OBJ=right_bottle ;; esac; ANN=/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json; ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output; [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam; REPLAY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435; HAND_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output; OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}; [[ -f "$ANN" ]] || { echo "[skip] task=${TASK} missing annotation $ANN"; continue; }; [[ -d "$ANY_ROOT" ]] || { echo "[skip] task=${TASK} missing ANY_ROOT=$ANY_ROOT"; continue; }; [[ -d "$REPLAY_ROOT" ]] || { echo "[skip] task=${TASK} missing REPLAY_ROOT=$REPLAY_ROOT"; continue; }; VIDEO_PREFIX=foundation_input CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh "$ANY_ROOT" "$REPLAY_ROOT" "$HAND_ROOT" "$OUT_ROOT" --hand_keyframes_json "$ANN" --left_target_object "$LEFT_OBJ" --right_target_object "$RIGHT_OBJ" --anygrasp_score_weight 0.25 --orientation_score_weight 0.75 --max_rotation_distance_deg 90 --candidate_target_local_x_offset_m -0.05 --draw_object_overlay 1 --draw_hand_reference 1 --debug_dump_object_distances 1 --top_k 20 --camera_cv_axis_mode legacy_r1; done
```

单 id 验证，不覆盖主结果：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && VIDEO_PREFIX=foundation_input CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/pick_diverse_bottles_output /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_replay_d435 /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_output /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_offsetfix_debug/pick_diverse_bottles --ids 0 --hand_keyframes_json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/pick_diverse_bottles/hand_keyframes_all.json --left_target_object left_bottle --right_target_object right_bottle --anygrasp_score_weight 0.25 --orientation_score_weight 0.75 --max_rotation_distance_deg 90 --candidate_target_local_x_offset_m -0.05 --draw_object_overlay 1 --draw_hand_reference 1 --debug_dump_object_distances 1 --top_k 20 --camera_cv_axis_mode legacy_r1
```

检查 frame 38 rank1 的 raw/visual/world 映射：

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_offsetfix_debug/pick_diverse_bottles/foundation_input_0/summary.json')
d = json.loads(p.read_text())
for e in d['frames']:
    if int(e['frame']) == 38:
        for key in ['left_orientation', 'right_orientation']:
            c = e['top_candidates'][key][0]
            print(key, 'idx=', c['candidate_idx'])
            print('  raw_cam=', [round(x, 6) for x in c['translation_cam']])
            print('  visual_cam=', [round(x, 6) for x in c['visual_translation_cam']])
            print('  world=', [round(x, 6) for x in c['translation_world']])
PY
```

D435 AnyGrasp planner/replay：严格同步、六任务前 5 个，无 viewer：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee-offsetfix
```

D435 AnyGrasp planner/replay：viewer + 目标 gripper/axis 可视化，用于确认 planner 实际目标：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee-offsetfix
```

输出对比路径：

```text
<OUT>/source_preview_compare/
  frame_000038_d435_orientation_rank.png
  frame_000038_d435_fused_rank.png
  frame_000038_legacy_orientation_rank.png
  frame_000038_legacy_fused_rank.png
  selected_candidate_mapping.json
```

`selected_candidate_mapping.json` 中的 `source_entry_translation_world` 应与 planner 的 `planner_target_pose_world_wxyz[:3]` 对齐；如果 `planner_raw_pose_world_wxyz[:3]` 和 target 相差约 5cm，这是 `--candidate_target_local_x_offset_m -0.05` 的预期效果。

### L15.9 D435 AnyGrasp no-offset：复制安全版三步运行指令

用途：把原始 no-offset AnyGrasp 候选 preview/summary 保存到主目录 `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435`。避免 zsh 手动换行导致 `OUT_ROOT=/ home/...`、`--candidate_target_local_x_offset_m: command not found` 或脚本 fallback 到默认 `replay_m_obj_pose_d_pour_blue_norobot`。下面 3 段按顺序运行；第一段使用 `bash <<'BASH'`，可以整体复制到 zsh。

#### 1. 重新生成六任务 D435 AnyGrasp preview/summary

```bash
bash <<'BASH'
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  case "$TASK" in
    pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle ;;
    place_bread_basket) LEFT_OBJ=basket; RIGHT_OBJ=bread ;;
    stack_cups) LEFT_OBJ=left_light_pink_cup; RIGHT_OBJ=right_dark_red_cup ;;
    handover_bottle) LEFT_OBJ=right_bottle; RIGHT_OBJ=right_bottle ;;
    pnp_bread) LEFT_OBJ=left_bread; RIGHT_OBJ=right_bread ;;
    pnp_tray) LEFT_OBJ=left_dark_red_cup; RIGHT_OBJ=right_bottle ;;
  esac

  ANN=/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json
  ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output
  [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam
  REPLAY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435
  HAND_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output
  OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}

  VIDEO_PREFIX=foundation_input CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh \
    "$ANY_ROOT" \
    "$REPLAY_ROOT" \
    "$HAND_ROOT" \
    "$OUT_ROOT" \
    --hand_keyframes_json "$ANN" \
    --left_target_object "$LEFT_OBJ" \
    --right_target_object "$RIGHT_OBJ" \
    --anygrasp_score_weight 0.25 \
    --orientation_score_weight 0.75 \
    --max_rotation_distance_deg 90 \
    --candidate_target_local_x_offset_m 0.0 \
    --draw_object_overlay 1 \
    --draw_hand_reference 1 \
    --debug_dump_object_distances 1 \
    --top_k 20 \
    --camera_cv_axis_mode legacy_r1
done
BASH
```

运行时必须看到每个 task 的 replay root 是 D435 路径：

```text
[run-anygrasp-preview-keyframes-batch] replay_root=/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435
```

如果仍然出现 `replay_m_obj_pose_d_pour_blue_norobot`，说明命令参数没有正确传进脚本。

#### 2. 无 viewer 跑六任务 D435 planner/replay

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh \
  --gpu 2 \
  --max_per_task 5 \
  --continue_on_error \
  --reach_error_pose_source ee \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee-offsetfix
```

#### 3. viewer 跑六任务 D435 planner/replay，并显示 gripper 目标

先检查 viewer 环境：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py
```

再运行 viewer 版本：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh \
  --max_per_task 5 \
  --continue_on_error \
  --viewer \
  --visualize_targets \
  --trajectory_mode joint_interp \
  --joint_interp_waypoints 40 \
  --allow_partial_dual_stage \
  --print_pose_every 5 \
  --reach_error_pose_source ee \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee-offsetfix
```

说明：

- 当前本节 preview 保留原始 no-offset AnyGrasp candidate：`--candidate_target_local_x_offset_m 0.0`。
- 如果后续 planner 仍设置 `--candidate_target_local_x_offset_m -0.05`，planner 会在执行阶段再沿 local +X 反方向退 5cm；因此主 preview 表示“原始 AnyGrasp 候选”，planner `rank_previews/source_preview_compare` 表示“执行 target”。
- gripper 局部轴约定：`local +X = gripper approach/forward axis = rotation_matrix[:, 0]`。

preview 图片说明：

```text
frame_XXXXXX_left_right_orientation_rank.png
  只按朝向相似度排序。orientation_score = max(0, 1 - aligned_rot / 180)。
  当前 downstream planner 默认使用 --reuse_preview_candidate_group orientation --reuse_preview_top_rank 1，
  因此 orientation rank1 是默认会送入 planner 的候选。

frame_XXXXXX_left_right_fused_rank.png
  按综合分数排序。fused_score = anygrasp_score * 0.25 + orientation_score * 0.75。
  用于参考 AnyGrasp 置信度和手部朝向的折中结果；当前默认 planner 不使用 fused。

frame_XXXXXX_left_right_planner_selected_orientation_rank1.png
  只画 downstream planner 当前默认会选择的候选：orientation rank1。
  这是最直接的“最终选择”可视化。
```

颜色说明：

```text
候选 AnyGrasp gripper wireframe:
  左手候选：蓝色系
  右手候选：橙色系

人手参考 gripper / hand reference:
  左手参考：绿色
  右手参考：紫色
```

因此你看到的绿色/紫色通常不是 AnyGrasp 最终候选，而是 HaMeR 人手参考夹爪，用来给 orientation 排序做对照。

### L15.10 D435 AnyGrasp offset -5cm 对比版：保存到单独目录做可视化对照

用途：如果想对比 planner 执行时常用的 5cm local-X 补偿，运行本节命令。它把 `--candidate_target_local_x_offset_m` 改为 `-0.05`，输出到单独目录，不覆盖主 D435 no-offset preview：

```text
当前 L15.9 主目录: --candidate_target_local_x_offset_m 0.0 -> 保留原始 AnyGrasp candidate 位置
本节 L15.10 对比目录: --candidate_target_local_x_offset_m -0.05 -> 沿 local +X 反方向退 5cm
```

该目录只用于判断 5cm local-X offset 是否是你看到的“夹爪靠后”的来源。

#### 1. 生成 offset -5cm D435 preview/summary 到对比目录

```bash
bash <<'BASH'
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  case "$TASK" in
    pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle ;;
    place_bread_basket) LEFT_OBJ=basket; RIGHT_OBJ=bread ;;
    stack_cups) LEFT_OBJ=left_light_pink_cup; RIGHT_OBJ=right_dark_red_cup ;;
    handover_bottle) LEFT_OBJ=right_bottle; RIGHT_OBJ=right_bottle ;;
    pnp_bread) LEFT_OBJ=left_bread; RIGHT_OBJ=right_bread ;;
    pnp_tray) LEFT_OBJ=left_dark_red_cup; RIGHT_OBJ=right_bottle ;;
  esac

  ANN=/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json
  ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output
  [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam
  REPLAY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435
  HAND_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output
  OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_offset_minus_5cm_compare/${TASK}

  VIDEO_PREFIX=foundation_input CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh \
    "$ANY_ROOT" \
    "$REPLAY_ROOT" \
    "$HAND_ROOT" \
    "$OUT_ROOT" \
    --hand_keyframes_json "$ANN" \
    --left_target_object "$LEFT_OBJ" \
    --right_target_object "$RIGHT_OBJ" \
    --anygrasp_score_weight 0.25 \
    --orientation_score_weight 0.75 \
    --max_rotation_distance_deg 90 \
    --candidate_target_local_x_offset_m -0.05 \
    --draw_object_overlay 1 \
    --draw_hand_reference 1 \
    --debug_dump_object_distances 1 \
    --top_k 20 \
    --camera_cv_axis_mode legacy_r1
done
BASH
```

#### 2. 对比同一 id/frame 的 no-offset 与 offset summary

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python - <<'PY'
import json
from pathlib import Path

task = "pick_diverse_bottles"
episode = "foundation_input_0"
frame = 38
paths = {
    "raw_no_offset_main": Path(f"/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/{task}/{episode}/summary.json"),
    "offset_minus_5cm_compare": Path(f"/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_offset_minus_5cm_compare/{task}/{episode}/summary.json"),
}
for label, path in paths.items():
    print(f"===== {label} =====")
    d = json.loads(path.read_text())
    for entry in d["frames"]:
        if int(entry["frame"]) != frame:
            continue
        for key in ["left_orientation", "right_orientation"]:
            c = entry["top_candidates"][key][0]
            print(key, "idx=", c["candidate_idx"])
            print("  translation_cam=", [round(x, 6) for x in c["translation_cam"]])
            print("  visual_translation_cam=", [round(x, 6) for x in c.get("visual_translation_cam", c["translation_cam"])])
            print("  translation_world=", [round(x, 6) for x in c["translation_world"]])
PY
```

#### 3. 对比图片路径

```text
raw/no-offset main:
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/frame_000038_left_right_orientation_rank.png

offset -5cm compare:
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_offset_minus_5cm_compare/<TASK>/foundation_input_<ID>/frame_000038_left_right_orientation_rank.png
```

注意：如果要让 preview 和 planner 的最终执行 target 完全一致，preview 和 planner 的 `--candidate_target_local_x_offset_m` 必须相同。当前你的要求是主 `anygrasp_h2o_preview_d435` 保存 no-offset 原始候选，所以它主要用于候选检查；planner 是否再加 `-0.05` 由后续执行命令控制。

### L15.11 D435 AnyGrasp viewer：执行 Cartesian IK 成功前缀，直到第一个不可达 waypoint

用途：当 `cartesian_interp_ik` 在中间 waypoint IK 失败时，默认严格逻辑会把整段 plan 标记为 `Fail`，双臂同步模式下可能直接跳过 stage，所以 viewer 里看起来“有目标/waypoint，但机器人不动”。本节新增诊断开关：

```text
--execute_partial_cartesian_plan
```

开启后，如果第 N 个 Cartesian waypoint IK 失败，但前面 1..N-1 已经求解成功，planner 会返回 `status=Partial`，并执行成功前缀到最后一个可达 waypoint。日志会打印：

```text
[plan-partial] ... failed_waypoint=N/M solved_prefix=K
[exec-pose] ...
```

注意：

- 该开关只对 `--trajectory_mode cartesian_interp_ik` 有意义。
- `joint_interp` 只求最终点 IK，然后做关节插值；没有 Cartesian waypoint 前缀，所以不会触发 partial cartesian 行为。
- `Partial` 会执行，但不会被认为 reached；第一关键帧没 reached 时，当前 close/action guard 仍会阻止关爪和进入第二关键帧。
- 这是调试可达边界的命令，不是最终推荐数据生成命令。

viewer 调试命令：六任务前 5 个，显示 target/waypoint，并执行可达 waypoint 前缀：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh \
  --max_per_task 5 \
  --continue_on_error \
  --viewer \
  --visualize_targets \
  --trajectory_mode cartesian_interp_ik \
  --cartesian_auto_step_m 0.03 \
  --execute_partial_cartesian_plan \
  --allow_partial_dual_stage \
  --print_pose_every 5 \
  --reach_error_pose_source ee \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-cartesian-partial-prefix
```

单任务调试版本，例如 `pick_diverse_bottles`：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh \
  --max_per_task 5 \
  --continue_on_error \
  --tasks pick_diverse_bottles \
  --viewer \
  --visualize_targets \
  --trajectory_mode cartesian_interp_ik \
  --cartesian_auto_step_m 0.03 \
  --execute_partial_cartesian_plan \
  --allow_partial_dual_stage \
  --print_pose_every 5 \
  --reach_error_pose_source ee \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-cartesian-partial-prefix-pdb
```

无 viewer 批量诊断：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh \
  --gpu 2 \
  --max_per_task 5 \
  --continue_on_error \
  --trajectory_mode cartesian_interp_ik \
  --cartesian_auto_step_m 0.03 \
  --execute_partial_cartesian_plan \
  --allow_partial_dual_stage \
  --print_pose_every 5 \
  --reach_error_pose_source ee \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/cartesian-partial-prefix
```

位置优先 / 朝向优先的说明：

- 当前 URDFIK 的成功判定同时约束位置和朝向：`--urdfik_position_threshold_m`、`--urdfik_rotation_threshold_rad`，并可放宽到 `--urdfik_max_position_threshold_m`、`--urdfik_max_rotation_threshold_rad`。
- 当前脚本中已经把执行 reached 判定放宽成位置优先调试：`--reach_rot_tol_deg 180`，这只影响“是否算 reached”，不改变 IK 求解本身。
- 如果想在 IK 求解阶段真正“位置优先”，需要让 IK solver 支持位置/姿态不同权重，或者在失败时用多级策略：先用完整姿态求解；失败后放宽 rotation threshold；再失败则只保留位置、用当前 EE 朝向或候选朝向附近采样。这个属于下一步 IK 策略修改，不是本节 partial-prefix 执行改动。

### L15.12 D435 AnyGrasp Piper 轴修正版：验证 preview gripper 与执行 gripper 的坐标系

用途：检查 viewer 中的执行夹爪朝向是否和 `anygrasp_h2o_preview_d435` 的 gripper wireframe 一致。

当前确认的坐标系关系：

```text
AnyGrasp / preview target:
  local +X = gripper approach / forward axis = rotation_matrix[:, 0]

Piper robot reported gripper pose:
  R_report = R_link6 @ global_trans_matrix @ delta_matrix

当前 Piper 配置:
  global_trans_matrix = diag(1, -1, -1)
  delta_matrix = I
```

因此送给 URDFIK 的 link6 目标朝向必须是：

```text
R_link6_target = R_preview_gripper @ inv(global_trans_matrix @ delta_matrix)
```

之前 Piper URDFIK 路径只反掉了 `delta_matrix`，没有反掉 `global_trans_matrix`；这会让 viewer 里真正渲染出来的 gripper 相对 preview gripper 绕 local +X 翻转约 180 度。现在 `render_hand_retarget_piper_dual_npz_urdfik.py` 已在 `_target_tcp_world_to_ee_base()` 中补上 `global_trans_matrix` 的逆变换。`plan_anygrasp_keyframes_r1.py` 的 `reach_error_pose_source=ee` 也保持 Piper dual 的可见 gripper 坐标系，不再把目标错误转换成 raw link6 坐标系。

如果只是检查“轴是否已经对齐”，先跑下面的 viewer 命令：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh \
  --max_per_task 5 \
  --continue_on_error \
  --viewer \
  --visualize_targets \
  --trajectory_mode cartesian_interp_ik \
  --cartesian_auto_step_m 0.03 \
  --execute_partial_cartesian_plan \
  --allow_partial_dual_stage \
  --print_pose_every 5 \
  --reach_error_pose_source ee \
  --ik_max_rotation_threshold_rad 3.14 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-cartesian-partial-prefix-axisfix
```

注意：如果不加 `--ik_max_rotation_threshold_rad 3.14`，当前默认仍是严格完整姿态 IK 的 `0.12rad`。实测 `pick_diverse_bottles id0` 会在第一个 Cartesian waypoint 失败，因此 viewer 看起来完全不动。这个现象说明 IK 姿态约束过紧或候选朝向不可达，不说明 step/settle 不够。

如果 viewer 中仍然“有 target/waypoint 但机器人完全不动”，用下面的位置优先诊断版。它把 IK 旋转成功阈值上限临时放宽到 `3.14rad`，用于确认失败是否主要来自朝向约束，而不是位置不可达：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh \
  --gpu 2 \
  --max_per_task 1 \
  --continue_on_error \
  --tasks pick_diverse_bottles \
  --trajectory_mode cartesian_interp_ik \
  --cartesian_auto_step_m 0.03 \
  --execute_partial_cartesian_plan \
  --allow_partial_dual_stage \
  --print_pose_every 5 \
  --visualize_targets \
  --reach_error_pose_source ee \
  --ik_max_rotation_threshold_rad 3.14 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/axis-fix-posfirst-smoke
```

实测 `pick_diverse_bottles id0`：

- 默认 `--ik_max_rotation_threshold_rad 0.12` 时，pregrasp/grasp 都在第一个 Cartesian waypoint IK 失败，机器人不会动；这不是 `settle_steps` 或 `joint_target_wait_steps` 不够。
- 放宽到 `3.14rad` 后，会输出连续 `[exec-pose]`，证明执行链和 waypoint 执行本身是通的。
- 但这只是诊断命令：它允许很大的朝向误差，所以不能作为最终数据生成质量标准。

当前 partial-prefix 逻辑还增加了一个小 fallback：如果第一个 Cartesian waypoint 就失败，会在当前 pose 到第一个失败 waypoint 之间再采样更短的小步，尝试执行最远的可解小步。若仍然完全不动，说明连这个短距离小步在当前完整姿态约束下也不可解。

如果要做真正的“位置优先 IK”，下一步应实现多级 IK fallback：

```text
1. full pose IK
2. relaxed rotation IK
3. position-only / keep-current-orientation IK
4. around preview orientation sampling
```

当前 L15.12 只提供第 2 步的命令级诊断入口。

viewer 批跑说明：

- 现在 wrapper 的 `--viewer` 默认 `--viewer_wait_at_end 0`，一个 id 结束后会自动进入下一个 id，不再必须关闭 SAPIEN 窗口。
- 如果想单个 id 结束后停在 viewer 中检查，显式加：

```bash
--viewer_wait_at_end 1
```

### L15.13 D435 AnyGrasp Piper viewer 轴检查：六任务 id0-10 单独命令

轴说明：

- `anygrasp_h2o_preview_d435/*orientation_rank.png` 里的 gripper wireframe 使用 `rotation_matrix[:, 0]` 作为 local +X。图上它是从掌根横杆指向两根手指指尖/缺口方向的轴。
- viewer 中 `--visualize_targets` 画的是坐标轴 actor，不是夹爪模型：红色 = local +X，绿色 = local +Y，蓝色 = local +Z。
- 因此判断 preview gripper 和 viewer target 是否一致时，优先比较“preview 两根手指从掌根到指尖的方向”和 viewer 红轴方向。
- Piper 真实夹爪 mesh 的 link 轴不一定等于 viewer target 轴。`robot_config_PiperPika_agx_dual_table_0515.json` 中 `global_trans_matrix=diag(1,-1,-1)`，会翻转 local Y/Z；所以绿/蓝轴看起来和 mesh 不一致时，不一定表示 local +X 前进轴错了。
- 当前默认 `--piper_apply_global_trans_to_ik 0`，保持和前面 direct replay 相同的 Piper URDFIK 约定。`--piper_apply_global_trans_to_ik 1` 只作为对照诊断，不作为默认。
- 如果红轴和 preview wireframe 的手指/缺口方向一致，但机器人执行后真实夹爪没对齐，这是 IK/执行结果错误；如果红轴本身就和 preview wireframe 不一致，才是 planner target 可视化或 candidate 映射错误。

注意：下面命令都用 `--id_start 0 --id_end 10`，不要再用 `--max_per_task 5`。`output_root` 必须保持一整行，不能被终端换行切成两个 token。

#### pick_diverse_bottles id0-10 viewer

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks pick_diverse_bottles --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10
```

#### place_bread_basket id0-10 viewer

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks place_bread_basket --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10
```

#### stack_cups id0-10 viewer

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks stack_cups --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10
```

#### handover_bottle id0-10 viewer

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks handover_bottle --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10
```

#### pnp_bread id0-10 viewer

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks pnp_bread --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10
```

#### pnp_tray id0-10 viewer

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks pnp_tray --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10
```

### L15.14 Viewer 目标夹爪帧显示修正：按左右手分别显示当前关键帧

问题现象：

- 运行 L15.13 这类 viewer 命令时，`--visualize_targets` 看到的 gripper/axis 目标像是后一关键帧，第一关键帧没有正确显示。
- 这不一定是 AnyGrasp candidate 本身错了；之前执行预览只保存了一个全局 `active_frame`，并且在双臂模式下用左手帧作为当前帧。若左右手的 effective keyframes 不同，或者阶段切换到 action 后残留了后一帧，viewer 中的候选和 target actor 会按错误帧显示。

修正内容：

- `plan_anygrasp_keyframes_r1.py` 的 `DebugExecutionState` 增加 `active_frame_by_arm`。
- pregrasp/grasp 阶段写入 `{left: left_keyframe1, right: right_keyframe1}`，action 阶段写入 `{left: left_keyframe2, right: right_keyframe2}`。
- `record_frame()` 和 `update_candidate_debug_visuals()` 现在按每只手自己的 active frame 更新 selected gripper、candidate gripper 和 debug execution preview。
- `pose_debug.jsonl` / `execution_metrics.jsonl` 现在也会记录 `active_frame_by_arm`，方便确认 viewer 当前显示的是哪一帧。

复查命令：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --tasks pick_diverse_bottles --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10
```

运行实际 viewer 时仍使用 L15.13 的六任务 id0-10 命令。检查方式：

```bash
jq -c '{stage,active_frame,active_frame_by_arm}' /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10/pick_diverse_bottles/foundation_input_0/pose_debug.jsonl | head -n 20
```

如果当前 shell 没有 `jq`，用这个 sed 版本：

```bash
head -n 20 /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10/pick_diverse_bottles/foundation_input_0/pose_debug.jsonl | sed -E 's/.*"active_frame": ([^,]+), "active_frame_by_arm": \{([^}]*)\}, "stage": "([^"]+)".*/stage=\3 active_frame=\1 active_frame_by_arm={\2}/'
```

预期：

- `stage=pregrasp` / `stage=grasp` 时，`active_frame_by_arm` 应该是第一关键帧。
- `stage=action` 时，`active_frame_by_arm` 才切到第二关键帧。
- viewer 红轴仍表示 target local +X；它应该和 D435 preview 中 gripper wireframe 从掌根到指尖/缺口方向一致。

### L15.15 Stack Cups id0：关闭碰撞、只显示当前 target 轴的规划调试

用途：单独排查 `stack_cups/foundation_input_0` 的 IK/执行问题，避免物体碰撞、候选 gripper 坐标轴、selected-keyframe 坐标轴和 IK waypoint marker 混在一起。

新增 wrapper 参数：

```text
--disable_execution_collisions
  将 planner 的 --enable_grasp_action_object_collision 设为 0；不再启用 grasp/action 物体碰撞和 contact-stop close 逻辑。

--target_axes_only
  自动打开 --visualize_targets，同时隐藏候选 gripper 轴、selected-keyframe 轴和 IK waypoint marker。
  该模式下 viewer 里应主要只剩左右手当前执行 target 的坐标轴。

--debug_candidate_top_k <N>
--debug_common_candidate_top_k <N>
--debug_visualize_selected_keyframe_axes <0|1>
--debug_visualize_ik_waypoints <0|1>
  细粒度控制调试 actor。
```

为什么之前会看到“四个坐标系”：

- `stack_cups id0` 的 D435 summary 是 per-arm keyframes：左手第一关键帧为 `139`，右手第一关键帧为 `51`，不是同一帧。
- viewer 中同时可能存在：
  - 当前执行 target 轴：左右手各一套，红/绿/蓝分别是 local X/Y/Z。
  - selected-keyframe 轴：按关键帧配色，第一关键帧是橙/黄系，第二关键帧是蓝/紫系。
  - candidate gripper 轴：候选夹爪上的红/绿/蓝轴。
  - IK waypoint / endpoint marker。
- 因此判断 gripper target 时，推荐先用 `--target_axes_only`，只看当前执行 target 轴。红轴仍是 target local +X；不要用其它候选轴混判。

stack_cups id0 viewer，关闭碰撞，只显示当前 target 轴：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks stack_cups --ids 0 --continue_on_error --viewer --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/stackcups-id0-nocollision-targetaxes
```

只调第一关键帧，关闭碰撞、无 viewer：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks stack_cups --ids 0 --continue_on_error --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 20 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --debug_stop_after_keyframe1 --output_root /tmp/stack_cups_id0_no_collision_target_axes_only
```

本机无 viewer 实测结论：

- wrapper 已确认 `collisions=0`，`plan_summary.json` 中 `enable_grasp_action_object_collision=0`。
- `pose_debug.jsonl` 显示 pregrasp 阶段为 `active_frame_by_arm={"left": 139, "right": 51}`，符合 stack_cups per-arm 关键帧。
- 关闭碰撞后仍未到位：pregrasp 末端误差约 left `0.386m/147deg`、right `0.337m/82deg`；grasp 仍失败或 partial。
- 因此 `stack_cups id0` 当前主要不是物体碰撞导致，而是 IK 解/轨迹执行后实际关节跟踪偏差很大。日志中能看到 plan target 误差小，但执行后 `joint_err` 仍较大，下一步应重点看控制器跟踪、关节插值执行时间/scene steps、以及候选姿态是否导致远距离绕行。

### L15.16 Direct Piper hand replay viewer：对照 AnyGrasp planner 的目标轴和执行误差

用途：回看“直接 replay 人手 gripper pose”的 Piper URDFIK 执行效果，和 AnyGrasp planner 的 target 轴/执行误差做对照。该命令不走 AnyGrasp candidate 选择，只读取 HaMeR/hand NPZ 中存好的 gripper pose。

要点：

- `--debug_visualize_targets 1`：viewer 中显示每一帧目标 gripper 坐标轴。
- `--debug_mode 1 --debug_post_execute 1`：每帧执行后打印 target 与实际 TCP/EE 的误差，用于看 plan 执行差距。
- `--save_world_targets 1`：保存目标 pose 和执行状态到 `world_targets_and_status.npz`。
- `--enable_viewer 1 --viewer_wait_at_end 1`：打开 SAPIEN viewer，执行结束后停住检查。
- 如果只想快速看关键帧附近，可把 `--frame_start/--frame_end` 改成更小范围，例如 `--frame_start 45 --frame_end 115`。

stack_cups id0 直接 replay viewer：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py \
  --input_npz /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_detections_0.npz \
  --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/direct_replay_debug_piper_d435/stack_cups/id0_viewer_axes \
  --image_width 640 \
  --image_height 480 \
  --fovy_deg 42.499880046655484 \
  --fps 5 \
  --frame_start 0 \
  --frame_end 220 \
  --max_frames 221 \
  --arms both \
  --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json \
  --camera_cv_axis_mode legacy_r1 \
  --require_stored_gripper_pose 1 \
  --pose_source gripper \
  --orientation_remap_label identity \
  --stored_orientation_post_rot_xyz_deg 0 0 0 \
  --target_local_forward_retreat_m 0.05 \
  --target_world_offset_xyz 0 0.1 0.1 \
  --execute_waypoint_scene_steps 5 \
  --execute_settle_scene_steps 20 \
  --urdfik_joint_interp_waypoints 10 \
  --debug_mode 1 \
  --debug_post_execute 1 \
  --debug_frame_limit -1 \
  --debug_visualize_targets 1 \
  --debug_target_axis_length 0.10 \
  --debug_visualize_cameras 0 \
  --save_world_targets 1 \
  --clean_output 0 \
  --overlay_text_enable 1 \
  --save_png_frames 0 \
  --lighting_mode front_no_shadow \
  --enable_viewer 1 \
  --viewer_frame_delay 0.02 \
  --viewer_wait_at_end 1
```

输出检查：

```bash
ls -lh /home/zaijia001/ssd/RoboTwin/code_painting/direct_replay_debug_piper_d435/stack_cups/id0_viewer_axes
python3 - <<'PY'
import numpy as np
p='/home/zaijia001/ssd/RoboTwin/code_painting/direct_replay_debug_piper_d435/stack_cups/id0_viewer_axes/world_targets_and_status.npz'
d=np.load(p, allow_pickle=True)
print(d.files)
for k in d.files:
    arr=d[k]
    print(k, getattr(arr, 'shape', None), getattr(arr, 'dtype', None))
PY
```

### L15.17 Direct replay 与 AnyGrasp 的 gripper 轴约定差异

当前确认的根因：

- direct Piper hand replay 中，目标 gripper pose 来自 HaMeR/hand NPZ 的 stored gripper frame。该路径的 `--target_local_forward_retreat_m` 明确沿 `local +Z` 后退，日志会打印：

```text
[target-local-retreat] along_local_plus_z_blue_m=...
```

因此 direct replay viewer 里蓝色轴（local +Z）就是手工 gripper frame 的 approach/forward 轴。

- AnyGrasp preview/planner 使用 AnyGrasp candidate 自己的 frame。`render_anygrasp_ranked_preview.py::draw_grasp_wireframe()` 中：

```text
x_axis = rotation_matrix[:, 0]
y_axis = rotation_matrix[:, 1]
left_tip/right_tip = left_base/right_base + x_axis * finger_len
```

所以 AnyGrasp 可视化里 local +X（红轴）是 wireframe 两根手指从掌根到指尖的 finger-depth 方向；local +Y 是开合宽度方向。它不是 direct replay 的 local +Z approach 约定。

这解释了你看到的现象：direct replay 中蓝轴看起来对齐真实机器人前进轴；AnyGrasp 中如果按红轴看，会和 direct replay 的“蓝轴前进”不一致。这不是 viewer 颜色画错，而是两套 gripper frame 约定不同。

当前 AnyGrasp planner 默认仍使用：

```text
--candidate_orientation_remap_label identity
--candidate_target_local_x_offset_m -0.05
--approach_offset_m 0.12
```

也就是说它把 AnyGrasp candidate 的 raw rotation 直接当执行 target。若要让 AnyGrasp candidate 的 local +X 对齐到 direct replay 的 local +Z approach 约定，需要做 candidate orientation remap。先用下面的对照命令单独跑 `stack_cups id0`：

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py \
  --anygrasp_dir /home/zaijia001/ssd/data/piper/hand/stack_cups/stack_cups_output/foundation_input_0 \
  --replay_dir /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_replay_d435/foundation_input_0 \
  --hand_npz /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_detections_0.npz \
  --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_axis_remap_debug/stack_cups/id0_swap_red_blue \
  --reuse_preview_summary_json /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/stack_cups/foundation_input_0/summary.json \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 \
  --arm auto --execute_both_arms 1 --dual_stage_require_all_plans 0 \
  --require_keyframe1_reached_before_close 1 --require_keyframe1_reached_before_action 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps -1 \
  --urdfik_cartesian_interp_auto_step_m 0.03 \
  --execute_partial_cartesian_plan 1 \
  --urdfik_max_position_threshold_m 0.02 \
  --urdfik_max_rotation_threshold_rad 3.14 \
  --piper_urdfik_apply_global_trans_to_ik 0 \
  --candidate_selection_mode planner \
  --candidate_orientation_remap_label swap_red_blue \
  --left_target_object left_light_pink_cup \
  --right_target_object right_dark_red_cup \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.12 \
  --reach_error_pose_source ee \
  --replan_until_reached 1 --replan_until_reached_max_attempts 1 \
  --save_debug_preview 1 --save_pose_debug 1 \
  --debug_visualize_targets 1 \
  --debug_candidate_top_k 0 \
  --debug_common_candidate_top_k 0 \
  --debug_visualize_selected_keyframe_axes 0 \
  --debug_visualize_ik_waypoints 0 \
  --reach_pos_tol_m 0.03 --reach_rot_tol_deg 180 \
  --enable_grasp_action_object_collision 0 \
  --execute_interp_steps 24 --joint_command_scene_steps 10 \
  --settle_steps 30 --joint_target_wait_steps 25 --joint_target_wait_tol_rad 0.01 \
  --print_execution_pose_every 5 \
  --hold_frames_after_stage 8 \
  --pure_scene_output 0 --overlay_text 0 --head_only 0 --third_person_view 1 \
  --vscode_compatible_video 1 \
  --lighting_mode front_no_shadow \
  --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json \
  --camera_cv_axis_mode legacy_r1 \
  --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 \
  --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 \
  --enable_viewer 0 --viewer_wait_at_end 0 --viewer_show_camera_frustums 0 \
  --object_mesh_override left_light_pink_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj \
  --object_mesh_override right_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj
```

解释：

- `swap_red_blue` 的作用是把 AnyGrasp 的 local +X 映射到执行 target 的 local +Z，作为 direct replay 轴约定的候选对齐方式。
- 这条命令先无 viewer 跑，输出 `pose_debug.jsonl`、`debug_execution_metrics.jsonl` 和视频，避免 viewer 干扰。
- 如果该 remap 后蓝轴与 direct replay 的蓝轴一致、执行误差下降，再把同样的 `--candidate_orientation_remap_label swap_red_blue` 加回六任务 wrapper。若方向相反，再测 `swap_red_blue_keep_green` 或显式的 `x_from_zp_y_from_yp_z_from_xm`。

### L15.18 AnyGrasp replay-axis 关键帧执行逻辑（新增，不替换旧命令）

用途：保留前面 L15.x 的旧 AnyGrasp local-X 执行命令不变，新增一套“按 direct replay 轴约定”的 AnyGrasp 关键帧执行入口。这个入口专门用于排查和对齐你在 direct replay viewer 里看到的蓝轴前进逻辑。

为什么需要新增一套逻辑：

- direct Piper hand replay 的 stored gripper pose 使用 replay 约定：`local +Z`（viewer 蓝轴）是 approach/forward 轴。`render_hand_retarget_*` 路径中的 `--target_local_forward_retreat_m` 就是沿 local +Z 做后退。
- AnyGrasp raw candidate 使用 AnyGrasp 自己的 candidate frame：在 `render_anygrasp_ranked_preview.py::draw_grasp_wireframe()` 中，`rotation_matrix[:, 0]` 是两根手指从掌根到指尖的 finger-depth 方向，`rotation_matrix[:, 1]` 是开合宽度方向。
- 因此旧 planner 的 `identity + --candidate_target_local_x_offset_m -0.05 + --approach_axis local_x` 是“AnyGrasp local-X 逻辑”，不等价于 direct replay 的“蓝轴 local +Z 逻辑”。
- 新 wrapper 使用 `swap_red_blue`，把 AnyGrasp 原始 local +X 映射成执行 target 的 local +Z，然后 target offset 和 pregrasp retreat 都沿 local +Z 做。这样 viewer 中蓝轴的语义与 direct replay 保持一致。

新增代码入口：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh
```

它等价于在旧六任务 wrapper 前固定追加：

```text
--candidate_orientation_remap_label swap_red_blue
--candidate_target_local_x_offset_m 0.0
--candidate_target_local_z_offset_m -0.05
--approach_axis local_z
--approach_offset_m 0.12
```

注意：`--candidate_target_local_z_offset_m -0.05` 是沿 remap 后 target local +Z 的 5cm TCP/夹爪补偿；`--approach_axis local_z` 让 pregrasp 后退也沿 replay 蓝轴逻辑执行。旧脚本默认仍是 local-X，不会被这个新增入口改掉。

六任务分别跑前 5 个，无 viewer，保存目标轴视频和 pose/debug 输出：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pick_diverse_bottles --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/no-viewer
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks place_bread_basket --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/no-viewer
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks stack_cups --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/no-viewer
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks handover_bottle --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/no-viewer
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pnp_bread --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/no-viewer
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pnp_tray --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/no-viewer
```

六任务分别跑前 5 个，viewer 版本：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks pick_diverse_bottles --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks place_bread_basket --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks stack_cups --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks handover_bottle --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks pnp_bread --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks pnp_tray --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer
```

小范围调试某个任务 id0-10：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --tasks stack_cups --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/stack_cups_id0_10_viewer
```

### L15.19 设计记录：在候选筛选阶段统一 AnyGrasp gripper 与 robot gripper 轴

状态：本节只记录分析和对照运行命令，暂不修改代码。L15.18 的旧 `swap_red_blue + local_z` 指令保留。

目标问题：

- direct replay 的 hand/gripper frame 中，`local +Z` 是 robot/replay 的 approach/forward 轴；viewer 坐标轴颜色仍是标准定义：红色 = local +X，绿色 = local +Y，蓝色 = local +Z。
- AnyGrasp 的 C 形 gripper 可视化中，C 平面内“掌根到指尖”的方向来自 AnyGrasp raw `rotation_matrix[:, 0]`，也就是 raw local +X；开合宽度是 raw local +Y；C 平面的侧向法线是 raw local +Z。
- rank preview 中“蓝色/橙色 gripper”不是坐标轴颜色，而是左右手/候选 actor 颜色：通常蓝色表示 left，橙色表示 right。因此“橙色 C gripper”不等于“橙色轴”。

如果要从筛选阶段就让 gripper 朝向和 robot 朝向一致，推荐的长期逻辑是：

1. 在 `render_anygrasp_ranked_preview.py` 生成候选和写 `summary.json` 时，引入一个 canonical robot/replay gripper frame。
2. 把 AnyGrasp raw candidate rotation 转成 robot/replay frame 后再参与筛选、保存 summary、输出 preview：

```text
robot/replay target local +Z = AnyGrasp raw local +X   # C gripper 指尖/前进方向
robot/replay target local +Y = AnyGrasp raw local +Y   # gripper 开合宽度
robot/replay target local +X = -AnyGrasp raw local +Z  # 保持右手系
```

3. hand reference 的 rotation 也必须在同一个 canonical frame 下计算 orientation distance。也就是说 `align_hand_rotation_to_candidate_convention()`、`build_candidates_for_arm()`、`build_score_ranked_candidates_for_arm()` 不能一边用 raw AnyGrasp frame、一边用 robot frame。
4. planner 复用 D435 preview summary 时，应直接读取 canonical robot/replay target pose；此时执行阶段使用：

```text
--candidate_orientation_remap_label identity
--candidate_target_local_x_offset_m 0.0
--candidate_target_local_z_offset_m -0.05
--approach_axis local_z
```

5. 同时要改 C 形 gripper 的可视化 actor 或 visual-only transform。因为当前 C gripper actor 的几何默认 local +X 是指尖方向；如果 target pose 改为 local +Z 是 robot 前进方向，而 actor 几何不改，就会出现“蓝轴对了但 C 形夹爪仍横着”的视觉错位。改完后应满足：

```text
viewer 蓝轴 = target local +Z = robot/replay approach
C 形 gripper 指尖方向 = viewer 蓝轴方向
```

因此，按这套长期逻辑修改后：

- 蓝色轴仍然表示 local +Z。
- rank preview 中橙色仍然只是右手 gripper/candidate 的颜色，不表示轴。
- 如果右手候选是橙色 C gripper，那么“橙色 C gripper 的指尖方向”应与 robot target 蓝色 local +Z 对齐。
- 也就是说，不是“橙色轴和蓝轴对齐”，而是“橙色 C gripper 这个物体的前进方向与蓝色 local +Z 轴对齐”。

与 L15.18 当前有问题/待确认指令的区别：

- L15.18 是执行阶段 remap：候选筛选和 preview summary 仍主要来自 AnyGrasp raw/C-gripper frame，然后 planner wrapper 再用 `swap_red_blue + local_z` 把 raw local +X 映射到 target local +Z。
- L15.18 因为筛选、summary、rank preview、planner target、C gripper actor 不是同一套 canonical robot frame，容易出现“robot 蓝轴”和“C gripper 可视化方向/侧面法线”的解释不一致。
- L15.19 设计的长期版本是筛选阶段统一：summary 里保存的候选、orientation ranking 使用的候选、planner 执行 target、viewer target 轴、C gripper actor 都使用同一套 robot/replay frame。这样才不会在执行阶段靠 remap 补救。

当前可运行对照命令：仍使用 L15.18 现有 wrapper，仅把输出放到新的 `viewer_gripper` 目录，用于和旧 `/viewer` 输出分开保存。注意：这条命令还不是“筛选阶段统一 frame”的实现，只是保留当前 replay-axis 路径的对照输出。

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks pick_diverse_bottles --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

如果后续实现筛选阶段统一 frame，建议新建独立 preview/output 根目录，避免覆盖现有 D435 summary：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_robot_frame/
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_robot_frame/
```

#### L15.19.1 已实现入口：robot-frame preview + viewer_gripper planner

本轮已把 L15.19 的长期逻辑做成显式新路径，旧 L15.18 不变。

新增代码入口：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh
/home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh
```

实现含义：

- preview 阶段新增 `--candidate_frame_mode robot_replay`。
- robot-frame summary 中保存的 candidate rotation 满足：

```text
target local +Z = AnyGrasp raw local +X
target local +Y = AnyGrasp raw local +Y
target local +X = -AnyGrasp raw local +Z
```

- preview 阶段新增 `--candidate_target_local_z_offset_m`，可视化和 summary target 均按 local +Z 表示 robot approach。
- planner viewer 的 C 形 gripper actor 新增 `--debug_gripper_actor_forward_axis local_z`，让 C 形 gripper 的指尖方向沿 target 蓝色 local +Z 画出来。
- planner 阶段使用 `identity + local_z`，不再需要执行阶段 `swap_red_blue`：

```text
--candidate_orientation_remap_label identity
--candidate_target_local_x_offset_m 0.0
--candidate_target_local_z_offset_m -0.05
--approach_axis local_z
--debug_gripper_actor_forward_axis local_z
```

第一步：生成 robot-frame D435 AnyGrasp keyframe preview summary。先建议跑单任务/少量 id 检查：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh --gpu 2 --tasks pick_diverse_bottles --ids 0
```

六任务生成 robot-frame preview summary：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh --gpu 2
```

输出根目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_robot_frame
```

补充：`run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh` 现在会在 planner 前自动检查并生成缺失的 robot-frame preview summary。也就是说可以直接运行下面 planner 命令；如果 `anygrasp_h2o_preview_d435_robot_frame/<TASK>/foundation_input_<ID>/summary.json` 不存在，wrapper 会按同样的 `--tasks`、`--ids`、`--id_start/--id_end`、`--max_per_task` 范围先补 summary，再进入 planner。若只想使用已有 summary，可加：

```bash
--skip_preview_generation
```

注意：`--max_per_task` 是按已有 D435 preview summary 的可用 id 排序取前 N 个，不一定等于 `0..N-1`。例如某些任务首个可用 id 可能是 `1` 或 `7`。

第二步：使用 robot-frame summary 运行 viewer_gripper planner。单任务前 5 个：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks pick_diverse_bottles --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

六任务分别跑前 5 个 viewer_gripper：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --max_per_task 100 --continue_on_error --viewer --tasks pick_diverse_bottles --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks place_bread_basket --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks stack_cups --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks handover_bottle --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --max_per_task 10 --continue_on_error --viewer --tasks pnp_bread --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks pnp_tray --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

和 L15.18 的关键区别（2026-05-28 勘误）：

- L15.18：planner 用 `swap_red_blue` 将 target +Z 映射到 raw +X（指尖方向）。但 C-gripper actor 默认 `forward_axis=local_x` 即 target +X = raw +Z（C 平面法线），导致 viewer 中蓝轴（target +Z）指向指尖、C-gripper 指向法线——两者互相垂直，蓝轴看起来和 C 平面法线"对齐"是因为 C-gripper 被画成了 90 度旋转。
- L15.19.1：`identity` remap 不变换坐标轴，target +Z = raw +Z = C 平面法线。对于 power grasp（抓杯子/瓶子），C 平面法线就是正确的机械臂接近方向。`--debug_gripper_actor_forward_axis local_z` 让 C-gripper 指尖也沿 target +Z 画，因此蓝轴 ∥ C-gripper 指尖 ∥ 接近方向，视觉完全一致。
- 核心洞察：**AnyGrasp 的 power grasp 前进轴本就是 raw +Z（C 平面法线），不是 raw +X（指尖方向）。** `identity` remap 正确保留了这一轴约定，不需要 preview 阶段预先"转换"坐标系。
- 旧描述"preview summary 生成时就写 robot/replay frame"有误导性：summary 实际存储的仍是 raw rotation（来自 `build_score_ranked_candidates_for_arm()`）。`--candidate_frame_mode robot_replay` 的实质作用是改变 orientation ranking 中的 hand reference 对齐方式（`align_hand_rotation_to_candidate_convention`），从而影响候选排序，而非改变存储的旋转矩阵。

---

## AnyGrasp Planner 执行流程

### 阶段说明

AnyGrasp planner 对每个关键帧执行以下阶段流水线：

```
pregrasp → grasp → close_gripper → action
```

**1. pregrasp（预抓取接近）**

从机械臂当前位姿出发，运动到一个"后退位姿"——即目标抓取位姿沿 gripper 前进轴后退 `approach_offset_m` 距离的位置。目的是先到达一个安全的接近位姿，避免直接冲向物体。

- 后退方向由 `--approach_axis` 决定：
  - `local_x`（默认）：沿 target local +X 后退
  - `local_z`（L15.18/L15.19.1 使用）：沿 target local +Z（蓝轴）后退
- 后退距离由 `--approach_offset_m` 设定（通常 0.12m）
- 使用 `cartesian_interp_ik` 做 Cartesian 空间插值+IK求解
- 如果 `plan_vs_target_dist < ik_max_position_threshold_m` 且 `rot < ik_max_rotation_threshold_rad`，认为到达

**2. grasp（抓取接近）**

从 pregrasp 到达位姿出发，沿 gripper 前进轴向前运动到目标抓取位姿。这是真正接近物体的阶段。

- 目标位姿 = AnyGrasp 候选变换到世界坐标 + `candidate_target_local_x/z_offset_m` 偏移
- 同样使用 `cartesian_interp_ik`
- 到达条件同 pregrasp

**3. close_gripper（闭合夹爪）**

grasp 到达后闭合夹爪。如果 `--require_keyframe1_reached_before_close 1`，则 grasp 必须成功到达才闭合，否则跳过并标记 `grasp_not_reached_before_close`。

**4. action（动作执行）**

仅对第二个关键帧（keyframe_2）执行。包含抓取后的动作（如提起、移动、放置等），使用 joint-space replay 方式。

### 关键参数速查

| 参数 | 作用 | L15.18 值 | L15.19.1 值 |
|---|---|---|---|
| `--candidate_orientation_remap_label` | AnyGrasp raw → target 旋转映射 | `swap_red_blue` | `identity` |
| `--candidate_target_local_z_offset_m` | 沿 target +Z 偏移抓取目标 | -0.05 | -0.05 |
| `--approach_axis` | pregrasp 后退轴 | `local_z` | `local_z` |
| `--approach_offset_m` | pregrasp 后退距离(m) | 0.12 | 0.12 |
| `--debug_gripper_actor_forward_axis` | C-gripper actor 指尖方向 | `local_x`(默认) | `local_z` |
| `--ik_max_rotation_threshold_rad` | IK 允许的最大旋转误差(rad) | 3.14 | 3.14 |

### 日志字段含义

执行时会打印类似以下诊断信息：

```
[plan-request] stage=pregrasp try=1
  left: dx=0.1594 dy=0.0894 dz=-0.1391 dist=0.2297 rot=154.29 fwd_rot=84.50 fwd_cm=-11.46 lat_cm=19.90 theory=forward
```

| 字段 | 含义 |
|---|---|
| `dx/dy/dz` | 当前 TCP 到目标位姿的世界坐标差(m) |
| `dist` | 当前 TCP 到目标位姿的欧氏距离(m) |
| `rot` | 当前姿态到目标姿态的旋转角度差(deg) |
| `fwd_rot` | 当前 local +X 与目标 local +X 的夹角(deg)。**注意**：诊断始终用 local +X 作为"前进"参考，不随 `--approach_axis` 改变 |
| `fwd_cm` | 当前 TCP 沿目标 local +X 方向到目标的有符号距离(cm)。正值=当前在目标前方，负值=当前在目标后方 |
| `lat_cm` | 当前 TCP 垂直于目标 local +X 方向的偏移距离(cm) |
| `theory` | 理论运动方向：`forward`=目标在前方，`backward`=目标在后方 |

```
[attempt] stage=grasp try=1
  left: dx=0.2766 dy=0.3283 dz=-0.1357 dist=0.4502 rot=103.06 ... reached=0
```

`reached=0` 表示该阶段未到达目标（距离或旋转超过阈值）。`reached=1` 表示到达。

```
[warn] arms=left,right arm=left obj=left_light_pink_cup pre=ok(p=0.018,r=179.8) gr=miss(p=0.450,r=103.1) act=miss(p=0.526,r=99.1)
```

每个阶段的简写结果：`pre`=pregrasp, `gr`=grasp, `act`=action。`ok`=到达, `miss`=未到达。`p`=最终位置误差(m), `r`=最终旋转误差(deg)。

### L15.18 vs L15.19.1 蓝轴方向分析

**AnyGrasp raw 坐标轴约定**（C形夹爪）：
- raw +X：指尖方向（C 平面内，掌根→指尖）
- raw +Y：开合宽度方向（C 平面内）
- raw +Z：C 平面法线（侧面方向）

**对于 power grasp（抓杯子/瓶子等），机械臂的接近方向是 C 平面法线（raw +Z），不是指尖方向。** 机械臂从侧面接近物体，夹爪包络物体。

**L15.18 蓝轴为什么指向侧面法线**：

`swap_red_blue` 矩阵将 target +Z 映射为 raw +X（指尖方向）。同时 C-gripper actor 默认 `forward_axis=local_x` 即 target +X = raw +Z（C 平面法线）。结果：
- 蓝轴 = target +Z = raw +X = 指尖方向
- C-gripper 指尖 = target +X = raw +Z = C 平面法线
- 蓝轴 ⊥ C-gripper 指尖（90度错位）

viewer 中看到蓝轴和 C 平面法线"对齐"是因为蓝轴指向指尖方向，而 C-gripper 被画成了 90 度旋转——视觉上蓝轴垂直于 C-gripper，所以看起来像是和法线对齐。

**L15.19.1 为什么正确**：

`identity` remap 保留 raw 坐标轴不变。`--debug_gripper_actor_forward_axis local_z` 让 C-gripper 指尖沿 target +Z 画：
- 蓝轴 = target +Z = raw +Z = C 平面法线 = 正确的接近方向
- C-gripper 指尖 = target +Z = raw +Z = C 平面法线
- 蓝轴 ∥ C-gripper 指尖 ✓

**L15.18 的执行正确性**：虽然蓝轴方向在 viewer 中看着不对，但 `--approach_axis local_z` 让 pregrasp 沿 target +Z（= raw +X = 指尖方向）后退。对于需要从指尖方向接近的 grasp 类型（如 precision grasp），这可能恰好是正确的。但对于 power grasp（从侧面接近），这就是错误的方向。

### 常见失败模式

**1. `rot≈180` 导致机械臂大范围摆动**

当目标姿态与当前姿态几乎反向（rot ~180 deg）时，IK 求解器配合 `ik_max_rotation_threshold_rad=3.14`（接受任意旋转）会找出一条需要大幅重定向的路径。Cartesian 插值在 180 度旋转时 TCP 会走过大弧线，导致机械臂"旋转一圈"。

根因：preview 的 orientation ranking 按人手参考姿态排序，不按机械臂可达姿态排序。某些对人手自然的姿态对机械臂当前构型可能正好相反。

**排查建议**：
- 降低 `--ik_max_rotation_threshold_rad`（如 1.57 = 90deg）筛选掉旋转过大的候选
- 检查 `rot` 和 `fwd_rot` 值，如果接近 180 需要特别关注

**2. pregrasp 到达但 grasp 失败**

pregrasp 位姿（沿前进轴后退）到达，但从 pregrasp 到 grasp 的向前运动失败。常见于目标位姿过于靠近机械臂工作空间边界，pregrasp 恰好能到达但 grasp 需要的手臂构型不可达。

#### L15.19.2 Robot-frame planner 指定 id viewer 命令

指定单个 id 时使用 `--ids <ID>`。robot-frame planner wrapper 会先检查并自动生成缺失的 robot-frame preview summary，再进入 planner；若只想使用已有 summary，可加 `--skip_preview_generation`。

`stack_cups id4` viewer 调试命令：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids 4 --continue_on_error --viewer --tasks stack_cups --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

六个 task 分别指定 id 的 viewer 模板。把每行里的 `--ids <ID>` 改成要看的 episode id；也可以写多个 id，例如 `--ids 0 4 8`：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks pick_diverse_bottles --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks place_bread_basket --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids 4 --continue_on_error --viewer --tasks stack_cups --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 1 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks handover_bottle --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks pnp_bread --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks pnp_tray --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

六任务同时指定同一组 id 的命令：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids 0 1 2 3 4 --continue_on_error --viewer --tasks pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/0_viewer_gripper
```

---

## L15.20 异步独立关键帧执行设计 [已实现]

### 0611 Human Replay / Mode M IK 连续性修复

`pick_diverse_bottles` 属于模式 A（两个全局关键帧），不需要本节后面的异步 L/R staged wrapper。用户在 `modelL1521-5` 中运行的 `run_plan_keyframes_human_replay_piper_d435.sh` 实际是 Mode M 同步双臂路径。

旧默认存在四个叠加问题：

- 中间层声明了 `urdfik_num_seeds=20`，但未传给底层，实际日志一直是 `num_seeds=1`。
- `cartesian_interp_ik` 允许执行 IK 失败前的 partial prefix，可能把机器人停在离目标很远的中间状态。
- seeded/unseeded 解只按末端 pose 误差选，容易切换腕/肘分支；运动会突然翻腕或跳远。
- 第一关键帧 grasp 未 reached 时，安全门控会跳过第二关键帧 action，所以视觉上像“只有第一关键帧”，并不是第二关键帧没有读取。

当前推荐使用 O.1 V2/V4 的思路：关节空间 cubic smoothstep、围绕当前关节做显式扰动 seed 搜索、按关节连续性选解；第二关键帧使用自身 xyz，但保留第一关键帧 grasp quaternion。

> **2026-06-17 URDF 修复**：`render_hand_retarget_piper_dual_npz_urdfik.py` 的 `PIPER_URDF` 已从 `piper.urdf` 修正为 `piper_pika_agx.urdf`（与 SAPIEN 仿真一致）。旧 URDF 的关节原点与仿真完全不同，修复前 IK 解在错误模型上求解、在正确模型上执行，会产生 50+cm 的位置误差。
>
> **`--target_retreat_m` 参数**（新增）：人手关键帧给出的是 TCP（夹爪尖端）位姿，但 IK 以 `link6`（腕关节）为目标，TCP = link6 + `gripper_bias`。设置 `--target_retreat_m 0.12` 让目标沿夹爪前进轴回退 12cm，使 link6 到达 TCP 后方、实际 TCP 到达人手位置。Piper 的 `gripper_bias=0.12`。如果想模仿 Foundation 的 `grasp_standoff=0.14`（含 2cm 安全间距），设 `0.14`。

ID 1 viewer 命令（`--target_retreat_m 0.12` = link6 回退到 TCP 后方 12cm）：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks pick_diverse_bottles \
  --trajectory_mode joint_interp \
  --joint_trajectory_interpolation cubic \
  --ik_num_seeds 1 \
  --ik_solution_selection joint_continuity \
  --ik_seed_perturbations 6 \
  --ik_seed_perturbation_scale 0.05 \
  --ik_max_joint_step_rad 0 \
  --execute_partial_cartesian_plan 0 \
  --apply_global_trans_to_ik 0 \
  --action_orientation_source grasp \
  --dual_stage_freeze_reached_arms_on_replan 1 \
  --reach_pos_tol_m 0.04 \
  --reach_rot_tol_deg 180 \
  --replan_until_reached_max_attempts 5 \
  --fail_on_execution_failure 1 \
  --target_retreat_m 0.14 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay_smooth
```

无 viewer 对比多个 ID：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0 1 2 --continue_on_error --tasks pick_diverse_bottles \
  --target_retreat_m 0.12 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay_smooth
```

#### L5 批量无 viewer 采集（head + wrist 视频，`--target_retreat_m 0.14` + O.1 wrist 校准）

与 L15.20 单 ID viewer 命令相同参数，增加 O.1 同款 wrist 相机校准。输出 `head_cam_plan.mp4` / `third_cam_plan.mp4` / `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4`。

**单 ID 先验证**（viewer 模式，确认 wrist 视角正确）：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks pick_diverse_bottles \
  --trajectory_mode joint_interp --joint_trajectory_interpolation cubic \
  --ik_num_seeds 1 --ik_solution_selection joint_continuity \
  --ik_seed_perturbations 6 --ik_seed_perturbation_scale 0.05 \
  --ik_max_joint_step_rad 0 --execute_partial_cartesian_plan 0 \
  --apply_global_trans_to_ik 0 --action_orientation_source grasp \
  --dual_stage_freeze_reached_arms_on_replan 1 \
  --reach_pos_tol_m 0.04 --reach_rot_tol_deg 180 \
  --replan_until_reached_max_attempts 5 --fail_on_execution_failure 1 \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m 0.145 --wrist_right_forward_offset_m 0.13 \
  --wrist_left_roll_deg -15 --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg 15 --wrist_right_pitch_deg 15 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274
```

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks pick_diverse_bottles \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.01 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg -15 --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274

```
**批量无 viewer**（ID 0-10 小批，输出到 `keyframe/L5_human_replay`）：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0 1 2 3 4 5 6 7 8 9 10 --continue_on_error --tasks pick_diverse_bottles \
  --trajectory_mode joint_interp --joint_trajectory_interpolation cubic \
  --ik_num_seeds 1 --ik_solution_selection joint_continuity \
  --ik_seed_perturbations 6 --ik_seed_perturbation_scale 0.05 \
  --ik_max_joint_step_rad 0 --execute_partial_cartesian_plan 0 \
  --apply_global_trans_to_ik 0 --action_orientation_source grasp \
  --dual_stage_freeze_reached_arms_on_replan 1 \
  --reach_pos_tol_m 0.04 --reach_rot_tol_deg 180 \
  --replan_until_reached_max_attempts 5 --fail_on_execution_failure 1 \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m 0.145 --wrist_right_forward_offset_m 0.13 \
  --wrist_left_roll_deg -15 --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg 15 --wrist_right_pitch_deg 15 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/L5_human_replay
```

**全量分段并行**（GPU 2: ID 0-50 / GPU 3: ID 51-101）：

```bash
# === GPU 2: ID 0-50 ===
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
for id in $(seq 0 50); do
  timeout 600s bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
    --gpu 2 --ids $id --continue_on_error --tasks pick_diverse_bottles \
    --trajectory_mode joint_interp --joint_trajectory_interpolation cubic \
    --ik_num_seeds 1 --ik_solution_selection joint_continuity \
    --ik_seed_perturbations 6 --ik_seed_perturbation_scale 0.05 \
    --ik_max_joint_step_rad 0 --execute_partial_cartesian_plan 0 \
    --apply_global_trans_to_ik 0 --action_orientation_source grasp \
    --dual_stage_freeze_reached_arms_on_replan 1 \
    --reach_pos_tol_m 0.04 --reach_rot_tol_deg 180 \
    --replan_until_reached_max_attempts 5 --fail_on_execution_failure 1 \
    --target_retreat_m 0.14 \
    --wrist_left_forward_offset_m 0.145 --wrist_right_forward_offset_m 0.13 \
    --wrist_left_roll_deg -15 --wrist_right_roll_deg -60 \
    --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
    --wrist_left_pitch_deg 15 --wrist_right_pitch_deg 15 \
    --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
    --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/L5_human_replay \
    || echo "FAIL L5 id=$id" >> /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/L5_failures.log
done
```

```bash
# === GPU 3: ID 51-101 ===
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
for id in $(seq 51 101); do
  timeout 600s bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
    --gpu 3 --ids $id --continue_on_error --tasks pick_diverse_bottles \
    --trajectory_mode joint_interp --joint_trajectory_interpolation cubic \
    --ik_num_seeds 1 --ik_solution_selection joint_continuity \
    --ik_seed_perturbations 6 --ik_seed_perturbation_scale 0.05 \
    --ik_max_joint_step_rad 0 --execute_partial_cartesian_plan 0 \
    --apply_global_trans_to_ik 0 --action_orientation_source grasp \
    --dual_stage_freeze_reached_arms_on_replan 1 \
    --reach_pos_tol_m 0.04 --reach_rot_tol_deg 180 \
    --replan_until_reached_max_attempts 5 --fail_on_execution_failure 1 \
    --target_retreat_m 0.14 \
    --wrist_left_forward_offset_m 0.145 --wrist_right_forward_offset_m 0.13 \
    --wrist_left_roll_deg -15 --wrist_right_roll_deg -60 \
    --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
    --wrist_left_pitch_deg 15 --wrist_right_pitch_deg 15 \
    --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
    --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/L5_human_replay \
    || echo "FAIL L5 id=$id" >> /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/L5_failures.log
done
```

> **Wrist 相机校准**：shell 脚本默认加载 `calibration_bundle_piper_new_table_0515.json`，将 wrist 相机定位到 D435 的真实安装位置（不再卡在 link6 原点 → 不再白屏）。O.1 同款的 `--wrist_*` 微调参数在 calibration bundle 基础上进一步调整。所有 wrist 参数通过 `render_hand_retarget_r1_npz.py` 的 `get_wrist_camera_pose` → `_apply_wrist_tuning` 施加，约定与 `envs/camera/camera.py` 一致。
>
> **输出目录结构**：`<output_root>/pick_diverse_bottles/foundation_input_<ID>/`，每个 ID 包含 `head_cam_plan.mp4`、`third_cam_plan.mp4`、`left_wrist_cam_plan.mp4`、`right_wrist_cam_plan.mp4`。

```bash

实测 ID 1、ID 2 完整执行 pregrasp/grasp/action；ID 0 的 action 也已执行，但最终左右位置误差约 4.39cm/6.41cm，超过 4cm 阈值。这说明旧问题不是 ID 1 标注独有，而是共享 IK 连续性和执行策略问题；修复后仍有 episode-specific 的第二目标可达性/跟踪误差。

当前没有实现“绕 local +Z 前进轴只能在某个 180 度区间内”的严格 roll 约束。Piper 配置的 `global_trans_matrix=diag(1,-1,-1)` 本身是绕 local X 的固定 180 度坐标变换；目标送入 IK 和 EE 回报的变换约定还不一致，因此 summary 中约 178-180 度的旋转误差暂时不能直接解释为真实错误 roll。`--apply_global_trans_to_ik 1` 已测试，会使本批目标明显更难求解，当前保持 0；后续应统一目标/回报坐标系，再把旋转阈值从 180 度收紧。

### 六任务关键帧结构分析

通过解析 `h2o_manual_review/<TASK>/hand_keyframes_all.json`，六个任务分为四种模式：

| 模式 | 任务 | 结构 | 帧顺序示例 |
|---|---|---|---|
| **A: 2全局** | pick_diverse_bottles, pnp_tray | G1→G2 | G38→G78 |
| **B: L2+R2 独立** | place_bread_basket, stack_cups, pnp_bread(部分) | L1→L2→R1→R2 或 R1→R2→L1→L2 | L34→L64→R103→R119 (place_bread_basket) |
| **C: L1+R1+G1 交接** | handover_bottle | R1→G(交接)→L1 | R39→G80→L103 |
| **D: G1+L1+R1 混合** | pnp_bread(部分) | G(共享帧)→L1→R1 | G26→L52→R64 |

**关键发现：**

- place_bread_basket: 左手先操作（L1→L2），右手后操作（R1→R2），时间上不重叠
- stack_cups: **右手先操作**（R1→R2→L1→L2），顺序相反
- handover_bottle: 右手拾取→共享帧交接→左手接收，`effective_keyframes_by_arm` 中双手共享 frame 80
- pnp_bread: 部分ID是 L2+R2 独立，部分是 G+L1+R1 混合

当前 planner 的 `annotated_json_keyframes` 模式取 `effective_keyframes[:2]`，对所有模式都按两个全局关键帧处理。对于模式 B/C/D，需要独立解析每个 arm 的关键帧并按异步顺序执行。

### 模式 B 异步执行设计（L=2+R=2 独立）

#### 执行流水线

```
初始化：双手在初始位姿（joint replay frame 0 或默认位姿）

Stage 1: L1  — 左手 pregrasp → grasp 到左关键帧1目标
         右手：joint replay 对应帧的人手轨迹，或保持初始位姿

Stage 2: R1  — 右手 pregrasp → grasp 到右关键帧1目标
         左手：保持 L1 grasp 位姿（joint hold）

Stage 3: L2  — 左手从 L1 位姿过渡到 L2 位姿（Cartesian IK 或 joint replay）
         右手：保持 R1 grasp 位姿

Stage 4: R2  — 右手从 R1 位姿过渡到 R2 位姿
         左手：保持 L2 grasp 位姿

结束：双手停留在最终位姿（L2 + R2）
```

#### 关键设计问题

**Q1: 右手在 Stage 1 应该做什么？**

选项：
- `hold_initial`：保持在初始 joint 位姿不动（最安全）
- `replay_human`：按人手 joint 轨迹 replay 对应帧（更接近原始运动，但可能和左手 IK 规划冲突）
- `replay_human_cartesian`：按人手 TCP 轨迹做 Cartesian replay

建议默认 `hold_initial`，避免双手交互区域的碰撞。

**Q2: L1→L2 过渡（Stage 3）怎么执行？**

选项：
- `ik_plan`：L1 grasp 位姿 → L2 grasp 位姿，用 Cartesian IK 插值
- `joint_replay`：从 L1 帧到 L2 帧的人手 joint 轨迹 replay

建议默认 `ik_plan`，因为目标是精确到达 L2 AnyGrasp 候选位姿。

**Q3: 右手从 L1 到达后到 R1 执行前，是否需要用碰撞规避？**

L1 到达后左手在目标位姿，R1 执行时右手移动。如果左右手目标位姿在空间上接近，R1 pregrasp→grasp 的路径需要避开左手当前位姿。使用 `--enable_grasp_action_object_collision 1` 并将对侧手臂 link 加入碰撞检测。

**Q4: L2/R2 最终位姿可能碰撞，如何避免？**

选项：
- L2/R2 执行前，加入 retreat（沿 gripper 前进轴后退）作为最终停留位姿
- 或者在 R2 到达后，双手各自 retreat 到安全位姿

建议：L2 到达后先不退，R2 到达后再决定是否 retreat。可以在 stage 4 之后增加可选的 `final_retreat` 阶段。

#### 实现方案

**方案：基于现有 `plan_anygrasp_keyframes_piper.py` 的 staged wrapper**

新增脚本 `/home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged.sh`：

工作流程：
1. 读取 `hand_keyframes_all.json` 中指定 ID 的 per-arm 关键帧
2. 从 robot_frame preview summary 读取每个关键帧的 top-1 AnyGrasp 候选
3. 按 L1→R1→L2→R2 顺序，每个阶段调用 planner 单 arm 模式
4. 非活跃 arm 在场景中保持当前 joint 位姿
5. 每个阶段输出独立的 video/pose debug 文件

核心参数设计：

```
--async_mode staged              # 启用异步分阶段执行
--async_stage_order L1,R1,L2,R2  # 阶段顺序（默认，可按帧时间重排）
--hold_arm_mode static           # 非活跃arm行为: static / replay / cartesian_replay
--transition_mode ik_plan        # L1→L2 过渡方式: ik_plan / joint_replay
--final_retreat_m 0.05           # 最终位姿沿前进轴后退距离（0=不退）
```

#### 已实现命令

> **实现说明**：脚本已部署。核心文件：
> - `code_painting/plan_anygrasp_keyframes_async_staged.py` — Python 异步编排引擎（模式检测 + preview patching + staged 调用 planner）
> - `code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged.sh` — 单任务 bash 入口
> - `code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged_six_tasks.sh` — 六任务批量入口
> - `code_painting/plan_anygrasp_keyframes_r1.py` — 新增 `--init_left_arm_joints`, `--init_right_arm_joints`, `--save_final_joints_json`, `--init_gripper_open_val` 参数（向后兼容）
>
> **工作原理**：读取 `hand_keyframes_all.json` 判断模式(A/B/C/D) → 解析 stage 顺序 → 每个 stage 生成 patched preview summary（只含当前 arm/keyframe）→ 调用现有 planner → `--save_final_joints_json` 保存关节状态 → 下一 stage 用 `--init_*_arm_joints` 载入。模式 A 直接委托给标准 planner。
>
> **重要**：`--execute_partial_cartesian_plan` 和 `--allow_partial_dual_stage` 都是 flag（无需值），不要带 `=1`。
>
> **已修复的 Bug（2026-06-01 测试总结）**：
> 1. `plan_anygrasp_keyframes_r1.py`: `get_left_arm_joints()` → `get_left_arm_real_jointState()` — 方法名不匹配导致 joint state 保存失败
> 2. `plan_anygrasp_keyframes_async_staged.py`: `float("inf")` 不是合法 JSON → 改为 `999.0`，避免 `async_staged_summary.json` 写入失败
> 3. Bash 脚本: `--execute_partial_cartesian_plan` 从需要值的参数改为 flag（`shift` 替代 `shift 2`）
> 4. Bash→Python: 新增 `--allow_partial_dual_stage` 转发，planner 侧映射为 `--dual_stage_require_all_plans 0`
> 5. 模式 A 委托后缺少 summary 输出 → 已补充
>
> **已验证通过的测试**：
> - Mode A (pick_diverse_bottles id=0): headless 运行成功，生成完整 video/pose_debug
> - Mode B (stack_cups id=0): 4 个 stage (L1→R1→L2→R2) 全部完成，joint state 正确传递

**Step 1: 预览生成（同 L15.19.1）**

```bash
# 生成 robot_frame D435 preview summary
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh --gpu 2
```

**Step 2: 单任务异步执行（viewer 模式）**

```bash
# 以 stack_cups 为例（帧顺序 R1→R2→L1→L2，执行顺序 L1→R1→L2→R2）
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged.sh \
  --gpu 2 \
  --tasks stack_cups \
  --ids 0 \
  --viewer \
  --continue_on_error \
  --visualize_targets \
  --disable_execution_collisions \
  --trajectory_mode cartesian_interp_ik \
  --cartesian_auto_step_m 0.03 \
  --execute_partial_cartesian_plan \
  --allow_partial_dual_stage \
  --reach_error_pose_source ee \
  --ik_max_rotation_threshold_rad 3.14 \
  --viewer_wait_at_end 0 \
  --async_stage_order L1,R1,L2,R2 \
  --hold_arm_mode static \
  --transition_mode ik_plan \
  --final_retreat_m 0.0 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/1B_viewer_gripper_async
```

**Step 3: 六任务批量异步执行**

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged_six_tasks.sh \
  --gpu 2 \
  --max_per_task 5 \
  --continue_on_error \
  --viewer \
  --visualize_targets \
  --disable_execution_collisions \
  --trajectory_mode cartesian_interp_ik \
  --cartesian_auto_step_m 0.03 \
  --execute_partial_cartesian_plan \
  --allow_partial_dual_stage \
  --reach_error_pose_source ee \
  --ik_max_rotation_threshold_rad 3.14 \
  --viewer_wait_at_end 0 \
  --async_stage_order default \
  --hold_arm_mode static \
  --transition_mode ik_plan \
  --final_retreat_m 0.0 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframeL1520B/viewer_gripper_async
```

**Step 4: 六任务各自指定 ID viewer 命令**

把每行里的 `--ids <ID>` 改成要看的 episode id；也可以写多个 id，例如 `--ids 0 4 8`：

```bash
# pick_diverse_bottles (模式 A: 2全局，直接委托标准 planner)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks pick_diverse_bottles --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/1B_viewer_gripper_async

# place_bread_basket (模式 B: L1->R1->L2->R2)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks place_bread_basket --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/1B_viewer_gripper_async

# stack_cups (模式 B: L1->R1->L2->R2, 注意原始时间顺序是 R1->R2->L1->L2 但执行顺序推荐 L1->R1->L2->R2)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks stack_cups --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 1 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/1B_viewer_gripper_async

# handover_bottle (模式 C: R1->G->L1)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks handover_bottle --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/1B_viewer_gripper_async

# pnp_bread (模式 B/D 混合，自动检测)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks pnp_bread --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/1B_viewer_gripper_async

# pnp_tray (模式 A: 2全局，直接委托标准 planner)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged.sh --gpu 2 --ids <ID> --continue_on_error --viewer --tasks pnp_tray --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/1B_viewer_gripper_async
```

**Step 5: 六任务无 viewer 批量执行**

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_async_staged_six_tasks.sh   --gpu 2 --max_per_task 5 --continue_on_error   --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/async_staged
```

### 模式 C 异步执行设计（L1+R1+G1 交接：handover_bottle）

handover_bottle 的执行逻辑本质不同：**右手拾取 → 双手交接 → 左手接收**。

```
Stage 1: R1 grasp (右手 pregrasp → grasp 关键帧 R[0])
         左手保持初始位姿

Stage 2: Handover approach (双手从当前位置 IK 到共享帧 G 的目标位姿)
         右手持有物体，左手准备接收

Stage 3: 交接执行 (右手打开夹爪，左手闭合夹爪)
         此时双手在 G 帧位姿

Stage 4: L1 final (左手 pregrasp → grasp 关键帧 L[0] 的后续位姿)
         右手可 retreat 或保持在 G 帧位姿
```

handover 的特殊性在于：
- 共享帧 G 是交接点，双手都在此帧有目标位姿
- effective_keyframes_by_arm: L=[L1, G], R=[R1, G]
- G 帧不是 grasp，而是 transfer pose
- 需要一个明确的 gripper open/close 序列

对于 handover 任务，建议先用手动标注的交接帧 + 现有 planner 做基础验证，异步逻辑稍后实现。

### 模式 A/D 说明

- **模式 A (2全局)**：当前 L15.19.1 逻辑已适用，无需异步
- **模式 D (G+L1+R1)**：pnp_bread 的部分 ID。共享帧 G 通常是双手初始接近物体的帧，然后左右手各自有一个独立关键帧。可以按 G→L1→R1 顺序（或 G→R1→L1，取决于帧时间）类似模式 B 处理

### 关于碰撞和 retreat 的建议

当前 L15.19.1 执行时，部分 ID 出现 `grasp_not_reached_before_close` 或大范围摆动（如 stack_cups id=4 的 180 度旋转）。异步模式下这些问题可能加剧，建议：

1. **降低 IK 旋转容忍度**：`--ik_max_rotation_threshold_rad 1.57`（90度）过滤掉旋转过大的候选
2. **增加 pregrasp retreat 距离**：`--approach_offset_m 0.15` 给每个阶段更多的接近空间
3. **最终 retreat**：Stage 4 结束后双手沿各自前进轴后退 `--final_retreat_m 0.05`，避免 L2/R2 位姿近距离碰撞
4. **碰撞检测**：在 transition 阶段（L1→L2, R1→R2），将对方 arm 的 link 加入碰撞检测以避免运动中碰撞

### 实现优先级

| 优先级 | 内容 | 状态 |
|---|---|---|
| P0 | 模式 B 的 staged wrapper 脚本（解析 per-arm 关键帧 + 分阶段调用 planner） | ✅ 已实现（`plan_anygrasp_keyframes_async_staged.py` 619行） |
| P1 | 模式 C handover 基础支持（G 帧交接 + 双手夹爪控制） | 🔧 基础框架已就位，stage 顺序已支持（R1→G→L1），实际 Gripper 交接逻辑待完善 |
| P2 | `--hold_arm_mode static/replay` 实现 | 🔧 static 模式已通过 `--init_*_arm_joints` 实现 |
| P3 | `--transition_mode ik_plan`（L1→L2 Cartesian IK 过渡） | 🔧 通过 `--arm left/right` 单臂模式 + planner 默认 Cartesian IK 可实现 |
| P4 | `--final_retreat_m` 实现 | 📋 参数已定义，planner 侧 `--candidate_target_local_z_offset_m` 可替代 |
| P5 | 碰撞检测集成 | 📋 使用现有 `--enable_grasp_action_object_collision` |

### 现有代码复用分析

实现模式 B 的 staged wrapper 可以大量复用现有代码：

- **关键帧解析**：复用 `preview_frame_selection_keyframes_for_arm()` 从 summary 读取 per-arm keyframes
- **候选读取**：复用 `load_reused_preview_summary()` 和 `preview_candidate_entry_to_pose()`
- **单 arm 规划**：复用 `plan_anygrasp_keyframes_r1.py` 的 `--arm left/right` 模式
- **Joint hold**：复用 `replay_piper_dual_h5.py` 的 joint replay，对非活跃 arm replay 固定帧
- **Cartesian 过渡**：复用 `urdfik_trajectory_mode=cartesian_interp_ik`

新增逻辑主要是：
1. 解析 per-arm 关键帧并映射到执行顺序
2. 分阶段调用 planner，每个阶段用上一个阶段的到达位姿作为起始位姿
3. 非活跃 arm 的 joint hold 逻辑
4. Stage 间的状态传递和 video 拼接


### L15.20 六任务 Human Replay viewer debug 命令（Mode M + wrist 校准）

以下命令使用 Human Replay（Mode M）路径，带 L16 实测 wrist 参数。每个任务单独一条 viewer debug 命令，方便逐个调试。

> **注意**：这些命令使用 `run_plan_keyframes_human_replay_piper_d435.sh`（Human Replay），与上方的 `run_plan_anygrasp_keyframes_piper_d435_async_staged.sh`（AnyGrasp async staged）是两套不同的 target 来源。Human Replay 直接使用人手 gripper pose，AnyGrasp async staged 使用 AnyGrasp 候选。

```bash
# pick_diverse_bottles (模式 A: 2全局关键帧, ID 1)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks pick_diverse_bottles \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L1520_human_replay_viewer

# place_bread_basket (模式 B: L2+R2, ID 0)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0 --viewer --tasks place_bread_basket \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L1520_human_replay_viewer

# stack_cups (模式 B: L2+R2, ID 0)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0 --viewer --tasks stack_cups \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L1520_human_replay_viewer

# handover_bottle (模式 C: L1+R1+G1, ID 1, 无 id=0)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks handover_bottle \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L1520_human_replay_viewer

# pnp_bread (模式 B/D 混合, ID 10)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 10 --viewer --tasks pnp_bread \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L1520_human_replay_viewer

# pnp_tray (模式 A: 2全局关键帧, ID 0)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0 --viewer --tasks pnp_tray \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L1520_human_replay_viewer
```

> 使用时需要先设置环境变量：
> ```bash
> source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
> export DISPLAY=:1.0
> xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }
> ```

---

## L15.21 AnyGrasp 候选筛选逻辑详解

### 筛选流水线

对每个关键帧，AnyGrasp 候选经过以下 5 步筛选，最终选出 planner 使用的 top-1 候选：

```
原始 AnyGrasp (~20 candidates/frame)
    │
    ▼
[1] 物体分区 (Object Partition)
    计算每个候选最近的物体，按物体名分组
    │
    ▼
[2] 物体过滤 (Object Filter)
    仅保留 nearest_object == target_object 的候选
    pick_diverse_bottles: left->left_bottle, right->right_bottle
    │
    ▼
[3] 方向过滤 (Orientation Filter)
    仅保留 rotation_distance_deg(hand_ref, candidate) < max_rotation_distance_deg 的候选
    默认 max_rotation_distance_deg = 90
    │
    ▼
[4] 分数排名 (Score Ranking)
    fused_score = anygrasp_weight * norm(anygrasp_score) + orientation_weight * orientation_score
    默认: anygrasp=0.25, orientation=0.75
    输出两组排名: orientation_rank, fused_rank
    │
    ▼
[5] Planner 读取
    --reuse_preview_candidate_group orientation (默认)
    --reuse_preview_top_rank 1 (默认)
    从 orientation_rank 中取 rank=1 的候选作为 planner 目标
```

### 具体数据示例 (pick_diverse_bottles id=1, frame 46)

```
原始 AnyGrasp: 20 candidates
  ├─ 物体分区: left_bottle=14, right_bottle=6
  │
  ├─ [left] 物体过滤(target=left_bottle): 14 -> 方向过滤(max_rot=90deg): 4
  │   rank1: idx=9  anygrasp=0.251  orient=0.770  rot_dist=41.4deg  obj_dist=0.144m
  │   rank2: idx=4  anygrasp=0.312  orient=0.696  rot_dist=54.7deg  obj_dist=0.208m
  │   rank3: idx=14 anygrasp=0.239  orient=0.686  rot_dist=56.5deg  obj_dist=0.201m
  │   rank4: idx=16 anygrasp=0.229  orient=0.576  rot_dist=76.3deg  obj_dist=0.212m
  │
  └─ [right] 物体过滤(target=right_bottle): 6 -> 方向过滤(max_rot=90deg): 1
       rank1: idx=8  anygrasp=0.266  orient=0.657  rot_dist=61.7deg  obj_dist=0.154m
       ⚠ 仅剩 1 个候选！如果该候选不可达，无 fallback

Planner 选择: left=idx9, right=idx8 (均为 orientation_rank=1)
```

### 方向分数 (orientation_score) 计算

```
orientation_score = max(0, 1 - rotation_distance_deg / max_rotation_distance_deg)
```

其中 max_rotation_distance_deg = 90 (默认)。
- rot_dist <= 0deg -> score = 1.0 (完美对齐)
- rot_dist = 45deg -> score = 0.5
- rot_dist >= 90deg -> score = 0 (被过滤)

rotation_distance_deg 是 hand reference rotation 和候选 rotation 之间的角度差。hand reference 的 rotation 会根据 candidate_frame_mode 做对齐：
- anygrasp_raw 模式: hand rotation remap 为 columns = [hand_Z, hand_Y, -hand_X]
- robot_replay 模式: hand rotation 保持原样 (identity 对齐)

注意: 两种模式下排名可能不同。例如 frame 46 left idx=9: old模式 raw_rot_dist=102.1deg vs rot_dist=41.4deg，robot模式 raw_rot_dist=rot_dist=41.4deg。

### 融合分数 (fused_score) 计算

```
fused_score = anygrasp_weight * anygrasp_score_norm + orientation_weight * orientation_score
```

其中 anygrasp_score_norm 是 AnyGrasp 原始分数在全部候选中的归一化值。
当前默认 anygrasp=0.25, orientation=0.75 —— 方向对齐主导排名。

### 可视化文件路径

对于每个关键帧，以下图片被生成 (以 pick_diverse_bottles id=1, frame 46 为例):

| 文件 | 内容 | 路径 |
|---|---|---|
| 原始 AnyGrasp 密集结果 | 所有 ~20 个候选叠加在 D435 replay 图片上 | /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/pick_diverse_bottles_output/foundation_input_1/vis/grasp_result_000046.png |
| D435 replay 原图 | 无候选叠加的纯渲染图 | /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_replay_d435/foundation_input_1/head_anygrasp_frames/color_000046.png |
| Orientation 排名预览 | 左右手分别展示方向排名前N候选 + hand reference + 物体投影 | .../anygrasp_h2o_preview_d435/pick_diverse_bottles/foundation_input_1/frame_000046_left_right_orientation_rank.png |
| Fused 排名预览 | 同上但用融合分数排名 | .../anygrasp_h2o_preview_d435/pick_diverse_bottles/foundation_input_1/frame_000046_left_right_fused_rank.png |
| Planner 最终选择 | 仅高亮 rank1 候选 + 目标坐标轴 | .../anygrasp_h2o_preview_d435/pick_diverse_bottles/foundation_input_1/frame_000046_left_right_planner_selected_orientation_rank1.png |

robot_frame 版本路径为 .../anygrasp_h2o_preview_d435_robot_frame/pick_diverse_bottles/foundation_input_1/...

### 四图拼接可视化命令

脚本已部署到 `/tmp/batch_montage.py`，支持单帧和批量两种模式。

**单帧拼接 (指定任务/ID/帧):**
```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n RoboTwin_bw python3 /tmp/montage_script.py pick_diverse_bottles 1 46
```
输出: `.../foundation_input_1/frame_000046_montage_4panel.png`

**批量拼接 (指定任务和ID范围):**
```bash
# 所有6个任务，ID 0-4 (pnp_bread 和 handover_bottle 的 ID 范围不同，见下方)
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n RoboTwin_bw python3 /tmp/batch_montage.py --ids 0 1 2 3 4

# pnp_bread ID 范围 (从10开始)
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n RoboTwin_bw python3 /tmp/batch_montage.py --tasks pnp_bread --ids 10 11 12 13 14

# handover_bottle ID 范围 (从1开始，无id=0)
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n RoboTwin_bw python3 /tmp/batch_montage.py --tasks handover_bottle --ids 1 2 3 4

# 仅处理 robot_frame preview (默认同时处理 old 和 robot_frame)
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n RoboTwin_bw python3 /tmp/batch_montage.py --preview_base robot_frame --ids 0 1 2 3 4

# 仅处理旧版 preview
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda run -n RoboTwin_bw python3 /tmp/batch_montage.py --preview_base old --ids 0 1 2 3 4
```

每个 montage 包含 4 列 (左→右):
1. 原始 AnyGrasp 密集结果 (所有~20个候选)
2. Orientation 排名预览 (方向分数排名，默认前20)
3. Fused 排名预览 (融合分数排名)
4. Planner 最终选择 (仅 rank1，带 coordinate axes)

各任务 ID 范围:
| 任务 | ID 范围 | 备注 |
|---|---|---|
| pick_diverse_bottles | 0-101 | id 连续 |
| place_bread_basket | 0-N | id 连续 |
| stack_cups | 0-N | id 连续 |
| handover_bottle | 1-N | **无 id=0** |
| pnp_bread | 10-N | **从 10 开始** |
| pnp_tray | 0-N | id 连续 |

### 当前策略的问题

**问题 1: 右手候选太少**

frame 46 右手仅剩 1 个候选通过过滤 (方向差=61.7deg)。如果该候选被 IK 判定不可达，planner 无 fallback。
部分 ID (如 id=101 frame 0) 甚至出现 right_after=0 —— 右手无任何候选通过过滤。

建议: 将 --max_rotation_distance_deg 从 90 提高到 120，或启用 --reuse_preview_top_rank 2 作为 fallback。

**问题 2: anygrasp_score 权重过低**

当前 anygrasp=0.25, orientation=0.75。AnyGrasp 自身的抓取质量分数对最终排名影响很小，方向对齐几乎决定了排名。

建议: 尝试 --anygrasp_score_weight 0.5 --orientation_score_weight 0.5。

**问题 3: 方向过滤以人手为参考，不以机械臂为参考**

候选按与人手姿态的相似度排序，而非按机械臂从初始构型的可达性排序。导致某些"对人手自然"的姿态对机械臂需要大幅度重定向 (如 stack_cups id=4 的 180deg 旋转)。

建议: 在 planner 阶段加入基于当前机械臂构型的二次过滤，拒绝 rot > 120deg 的候选。

### 当前应使用的指令

使用 L15.19.1 的 robot_frame 路径:

**生成 robot_frame preview summary:**
```bash
# 单任务测试 (推荐先跑少量 id)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh --gpu 2 --tasks pick_diverse_bottles --ids 1

# 六任务全量
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh --gpu 2
```

**运行 planner (viewer 模式):**
```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh \
  --gpu 2 --max_per_task 5 --continue_on_error --viewer \
  --tasks pick_diverse_bottles --visualize_targets --disable_execution_collisions \
  --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 \
  --execute_partial_cartesian_plan --allow_partial_dual_stage \
  --print_pose_every 5 --reach_error_pose_source ee \
  --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

**查看候选选择可视化:**
```bash
# 单独查看 (在远程服务器上，需要有 X11 forwarding 或 DISPLAY=:1.0)
eog /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_robot_frame/pick_diverse_bottles/foundation_input_1/frame_000046_left_right_orientation_rank.png

# 或拼接四图 (运行上面的 PYEOF2 块)
```

**查看原始 AnyGrasp 密集结果:**
```bash
eog /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/pick_diverse_bottles_output/foundation_input_1/vis/grasp_result_000046.png
```

#### codex
```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks pick_diverse_bottles \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274
```

## L16. Human Replay + Piper D435 wrist 实测参数（2026-06-18）

本节记录当前实测较好的 Mode M / Human Replay wrist 相机参数。批量 no-viewer 命令必须和 viewer debug 使用同一组 wrist 外参参数，以保证保存的视频和机器人 obs 与调试视角一致。

关键 wrist 参数：

| 参数 | left | right | 说明 |
|---|---:|---:|---|
| `forward_offset_m` | `-0.04` | `-0.01` | 沿相机前向的最终微调 |
| `roll_deg` | `14.635` | `-44.649` | 画面横轴扶正到接近夹爪开合轴 |
| `yaw_deg` | `0.182` | `0.840` | 消除原始标定中前向轴的微小 Y 分量 |
| `pitch_deg` | `-90` | `-90` | 俯视方向：保持当前成功视角，不要改成 `+90` |
| `lateral_offset_m` | `-0.0207` | `0.0274` | 将相机平移到 gripper 中线附近 |

### L16.1 pick_diverse_bottles

`pick_diverse_bottles` 当前有效 ID 为 `0-101`，共 102 条。

#### L16.1.1 Viewer debug（ID 1，带 wrist preview 和相机轴）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks pick_diverse_bottles \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274
```

#### L16.1.2 No-viewer 批量生成数据（ID 0-101）

不启用 viewer，不需要 `DISPLAY`；不会打开 wrist preview，也不会绘制 wrist/head 相机框线或相机 RGB 轴。wrist 外参参数与 L16.1.1 完全一致。

输出目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/pick_diverse_bottles/foundation_input_<ID>/
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0-101 --continue_on_error --tasks pick_diverse_bottles \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean
```
```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks pick_diverse_bottles \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.01 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274
```

### L16.2 place_bread_basket

`place_bread_basket` 当前有效 ID 为 `0-73`，hand_keyframes_all.json 中 53 条有效（status 非 reject/discard/bad）。

**模式 B (L2+R2 独立)**：左手先操作（L1→L2），右手后操作（R1→R2），时间上不重叠。

关键帧动作（per-arm 策略：左手用 place strategy，右手不用）：

```
=== 左手 (place_strategy=true, G2 Z+5cm, lower=2cm, retreat=10cm) ===
Stage L1 — pregrasp → grasp → close left gripper                    （左手抓起 bread）
Stage L2 — action_approach(+5cm) → action(G2+5cm Z)                 （夹爪不能像人手伸进篮子，G2 Z 提高 5cm）
           → action_lower(+2cm) → open_gripper(step30+hold)         （松手放面包）
           → retreat(G1_TCP+10cm)                                    （回退防止挡右手）

=== 右手 (place_strategy=false, R2=R1 TCP Z+5cm) ===
Stage R1 — pregrasp → grasp → close right gripper                   （右手抓起 basket）
Stage R2 — action(R2=R1_TCP_Z+5cm, 夹爪保持 close)                  （提起篮子 5cm，不松手）
```

左手 G2 Z 在代码中自动 +5cm。右手 R2 的目标 Z = R1 TCP Z + 5cm，夹爪保持闭合——纯粹把篮子提起来。

#### L16.2.1 Viewer debug（ID 0，带 wrist preview 和相机轴）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0 --viewer --tasks place_bread_basket \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274
```

#### L16.2.2 No-viewer 批量生成数据（ID 0-73）

不启用 viewer，不需要 `DISPLAY`；不会打开 wrist preview，也不会绘制 wrist/head 相机框线或相机 RGB 轴。wrist 外参参数与 L16.2.1 完全一致。

输出目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/place_bread_basket/foundation_input_<ID>/
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0-73 --continue_on_error --tasks place_bread_basket \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean
```


```bash
#debug
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0-10 --continue_on_error --tasks place_bread_basket \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_de_human_replay_clean
```

---

### L16.3 stack_cups

`stack_cups` 当前有效 ID 为 `0-46`，hand_keyframes_all.json 中 41 条有效（status 非 reject/discard/bad）。

**模式 B (L2+R2 独立)**：右手先操作（R1→R2），左手后操作（L1→L2），与 place_bread_basket 顺序相反。

关键帧动作：

```
Stage 1: R1 — pregrasp → grasp → close right gripper  （右手抓起一个杯子）
Stage 2: R2 — pregrasp → grasp → open right gripper   （右手将杯子叠放到目标上）
Stage 3: L1 — pregrasp → grasp → close left gripper   （左手抓起另一个杯子）
Stage 4: L2 — pregrasp → grasp → open left gripper    （左手将杯子叠放到目标上）
```

执行顺序 R1→R2→L1→L2，非活跃 arm 保持当前 joint 位姿。注意原始时间顺序是 R1→R2→L1→L2，与执行顺序一致。

#### L16.3.1 Viewer debug（ID 0，带 wrist preview 和相机轴）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks stack_cups \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274
```

#### L16.3.2 No-viewer 批量生成数据（ID 0-46）

不启用 viewer，不需要 `DISPLAY`；不会打开 wrist preview，也不会绘制 wrist/head 相机框线或相机 RGB 轴。wrist 外参参数与 L16.3.1 完全一致。

输出目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/stack_cups/foundation_input_<ID>/
```

```bash
# debug先
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0-10 --continue_on_error --tasks stack_cups \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_de_human_replay_clean
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0-46 --continue_on_error --tasks stack_cups \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean
```

---

### L16.4 handover_bottle

`handover_bottle` 当前有效 ID 为 `1-50`（**无 id=0**），hand_keyframes_all.json 中 50 条有效（status 非 reject/discard/bad）。

**模式 C (L1+R1+G1 交接)**：右手拾取 → 共享帧交接 → 左手接收。

关键帧动作（专用 handover 执行路径，不走 generic Mode B 逻辑）：

```
Stage 1: 右手 pregrasp → grasp R1 → close right gripper
         （右手单独到达抓取帧，抓起 bottle，物体附着右手 TCP）

Stage 2: 右手移动到 G（交接位置）
         （右手带着 bottle 到达 handover 帧，此时左手仍在初始位姿）

Stage 3: 左手 pregrasp → grasp G（bottle 始终挂在右手 TCP 上）
         （左手从 pregrasp 到达交接位置，夹爪保持 OPEN。
          右手 target 设为实际 TCP 位置，确保不漂移）

Stage 4: Handover 交接
         → 左手 close gripper（接住 bottle）
         → 右手 open gripper（松手，step_scene(30) + hold frames）
         → 物体切换到左手 TCP（从 actor 当前世界位姿计算偏移，非 object_states 初始位置）

Stage 5: 右手 retreat 到 R1 TCP Z + 5cm 安全高度
         （左手 target 设为实际 TCP 位置，确保不漂移）
```

关键帧结构：`right=[R1, G]`, `left=[G, L1]`（left 按帧号排序后 G 在前）。
G 帧是双手共享的交接点。Stage 3/5 使用 `execute_dual_stage_until_reached`，
非活跃手 target 设为其**实际当前 TCP**（非 ideal pose），避免 plan 产生微小位移导致漂移。

#### L16.4.1 Viewer debug（ID 1，带 wrist preview 和相机轴）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks handover_bottle \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274
```

#### L16.4.2 No-viewer 批量生成数据（ID 1-50）

不启用 viewer，不需要 `DISPLAY`；不会打开 wrist preview，也不会绘制 wrist/head 相机框线或相机 RGB 轴。wrist 外参参数与 L16.4.1 完全一致。

输出目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/handover_bottle/foundation_input_<ID>/
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1-50 --continue_on_error --tasks handover_bottle \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean
```

```bash
# debug
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1-10 --continue_on_error --tasks handover_bottle \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_de_human_replay_clean
```

---

### L16.5 pnp_bread

`pnp_bread` 当前有效 ID 为 `0-80`，hand_keyframes_all.json 中 74 条有效（status 非 reject/discard/bad）。

**模式 B/D 混合**：部分 ID 是 L2+R2 独立（模式 B），部分 ID 是 G+L1+R1 混合（模式 D）。

- **模式 B (L2+R2)**：L1→L2→R1→R2，左右手各自独立 pick & place
- **模式 D (G+L1+R1)**：共享帧 G → 左手关键帧 L1 → 右手关键帧 R1

关键帧动作（模式 B）：

```
Stage 1: L1 — pregrasp → grasp → close left gripper   （左手抓起面包）
Stage 2: L2 — pregrasp → grasp → open left gripper    （左手将面包放到盘子）
Stage 3: R1 — pregrasp → grasp → close right gripper  （右手抓起另一个面包）
Stage 4: R2 — pregrasp → grasp → open right gripper   （右手将面包放到盘子）
```

关键帧动作（模式 D）：

```
Stage 1: G  — pregrasp → grasp（双手共享接近物体的帧）
Stage 2: L1 — pregrasp → grasp → close left gripper   （左手抓起面包）
Stage 3: R1 — pregrasp → grasp → close right gripper  （右手抓起面包）
```

模式自动检测由 planner 根据 `hand_keyframes_all.json` 中的 keyframes 结构决定。

#### L16.5.1 Viewer debug（ID 10，带 wrist preview 和相机轴）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 10 --viewer --tasks pnp_bread \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274
```

#### L16.5.2 No-viewer 批量生成数据（ID 0-80）

不启用 viewer，不需要 `DISPLAY`；不会打开 wrist preview，也不会绘制 wrist/head 相机框线或相机 RGB 轴。wrist 外参参数与 L16.5.1 完全一致。

输出目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/pnp_bread/foundation_input_<ID>/
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 0 --ids 0-80 --continue_on_error --tasks pnp_bread \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean
```

```bash
# debug
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 1 --ids 0-10 --continue_on_error --tasks pnp_bread \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_de_human_replay_clean
```

---

### L16.6 pnp_tray

`pnp_tray` 当前有效 ID 为 `0-50`，hand_keyframes_all.json 中 51 条有效（status 非 reject/discard/bad）。

**模式 A (2 全局关键帧)**：G1→G2，与 pick_diverse_bottles 相同的模式。两个全局关键帧对双手同时有效。

关键帧动作：

```
全局关键帧 G1 — pregrasp → grasp → close gripper  （双手分别抓起 cup 和 bottle）
全局关键帧 G2 — pregrasp → grasp → open gripper   （双手将物体放到 tray 上）
```

> 说明：`hand_keyframes_all.json` 中 G1/G2 为全局 keyframes（非 per-arm），左右手共享同一帧号。Planner 读取后按标准 pregrasp/grasp/action 管线执行。

#### L16.6.1 Viewer debug（ID 0，带 wrist preview 和相机轴）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 2 --viewer --tasks pnp_tray \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274
```

#### L16.6.2 No-viewer 批量生成数据（ID 0-50）

不启用 viewer，不需要 `DISPLAY`；不会打开 wrist preview，也不会绘制 wrist/head 相机框线或相机 RGB 轴。wrist 外参参数与 L16.6.1 完全一致。

输出目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/pnp_tray/foundation_input_<ID>/
```

```bash
# debug
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 0 --ids 0-10 --continue_on_error --tasks pnp_tray \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_de_human_replay_clean
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0-50 --continue_on_error --tasks pnp_tray \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.01 \
  --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean
```

---

## M. 消融实验：Human Replay 人手关键帧目标 [已实现]

### 设计目的

隔离 AnyGrasp 候选选择的作用。直接使用人手 gripper pose（从 `hand_detections_<ID>.npz` 读取的世界空间位置+朝向）作为 IK 规划目标，**不使用 AnyGrasp 的任何候选**。仍复用同一套 pregrasp→grasp→action、双臂 reached 门控和 Piper URDF IK 执行框架；0611 推荐配置改用关节连续性优先的 `joint_interp + cubic`。

对比 L15.19.1（AnyGrasp robot_frame planner）和 Mode M，可以量化 AnyGrasp 抓取排序对执行成功率的贡献。

### 目标位姿来源

```
hand_detections_<ID>.npz
  ├─ {left,right}_gripper_position[frame]     → 人手相机空间位置
  ├─ {left,right}_gripper_rotation_matrix[frame] → 人手相机空间旋转矩阵
  └─ {left,right}_gripper_valid[frame]         → 有效性标记

multi_object_world_poses.npz
  └─ head_camera_pose_world_wxyz[frame]        → 相机外参

计算流程：
  1. 读取关键帧对应的相机空间人手位姿
  2. 通过相机外参 + camera_cv_axis_mode (legacy_r1) 转换到世界空间
  3. 生成 plan_summary.json（pose_world_wxyz = 人手世界位姿）
  4. Planner 读取 plan_summary.json，执行正常的 pregrasp/grasp/action 管线
```

### 与 L15.19.1 AnyGrasp planner 的区别

| | L15.19.1 AnyGrasp | Mode M (Human Replay) |
|---|---|---|
| 目标来源 | AnyGrasp 候选 (筛选+排名) | 人手 gripper pose (直接读取) |
| 朝向来源 | AnyGrasp 候选 (orientation ranking) | 人手旋转矩阵 (直接读取) |
| 物体信息 | 候选 nearest_object | task config 指定的 target_object |
| pregrasp 后退 | `--approach_offset_m` 沿 `--approach_axis` | 相同 |
| IK 规划 | Cartesian IK / joint interp | 相同 |
| 轴约定 | robot_frame identity (蓝=+Z=前进) | 相同 |

### 脚本与参数

**核心脚本：** `/home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_human_replay.py`

关键参数：
| 参数 | 作用 | 默认值 |
|---|---|---|
| `--hand_keyframes_json` | hand_keyframes_all.json 路径（读取关键帧） | 必需 |
| `--video_id` | episode ID | 必需 |
| `--approach_offset_m` | pregrasp 沿夹爪前进轴后退距离 | 0.12 |
| `--approach_axis` | 后退轴 | local_z (蓝轴) |
| `--camera_cv_axis_mode` | 相机轴映射 | legacy_r1 |

### 六任务 viewer 命令

Viewer 模式会在 bash wrapper 和 Python 中间层都移除 `CUDA_VISIBLE_DEVICES`，避免 SAPIEN 只能看到计算 GPU 而看不到驱动 VNC/X display 的 GPU。若日志仍出现 `Renderer does not support display`，先看 `[viewer] creating interactive viewer ...`：`CUDA_VISIBLE_DEVICES` 应为 `None/unset`，并且当前终端需要有可用的 `DISPLAY`（例如 `:1.0`）。

以下命令直接替换 L15.19.1 viewer 命令中的 planner 脚本路径，其余参数保持一致：

Mode M wrapper 已内置 L15.20 的 0611 连续性默认值：`joint_interp + cubic`、6 个扰动 seed、按 joint continuity 选解、action 保持 grasp 朝向、冻结已达标手、禁止 partial Cartesian prefix，并在执行失败时返回非零退出码。

```bash
# pick_diverse_bottles (模式 A: 2全局关键帧)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks pick_diverse_bottles \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay_smooth

# place_bread_basket (模式 B: L2+R2)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks place_bread_basket \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay_viewer

# stack_cups (模式 B: L2+R2)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids <ID> --viewer --tasks stack_cups \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay_viewer

# handover_bottle (模式 C: L1+R1+G1)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids <ID> --viewer --tasks handover_bottle \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay_viewer

# pnp_bread (模式 B/D 混合)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids <ID> --viewer --tasks pnp_bread \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay_viewer

# pnp_tray (模式 A: 2全局关键帧)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 0 --viewer --tasks pnp_tray \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/Mv_human_replay_viewer
```

### 六任务无 viewer 批量【anygrasp+最终】

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
    --gpu 2 --ids 0-10 --continue_on_error --tasks $TASK \
    --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/M_human_replay
done
```

### 预期结果

与 L15.19.1 AnyGrasp planner 对比：
- **如果 Mode M 成功率 ≈ AnyGrasp 成功率**：说明人手位置+朝向本身足够，AnyGrasp 的贡献有限
- **如果 Mode M 成功率 << AnyGrasp 成功率**：说明 AnyGrasp 的候选筛选和排名显著提升了抓取质量

---

## N. 消融实验：Foundation Pose 物体位置 + 人手朝向 [已实现]

### 设计目的

分离 AnyGrasp 的两个功能：(1) 物体定位（WHERE to grasp）和 (2) 抓取策略（HOW to grasp）。

本模式使用 **Foundation 模型的物体世界位置** 作为目标位置，叠加 **人手的朝向** 作为目标朝向，**不使用 AnyGrasp**。额外沿夹爪前进轴（蓝轴 = local +Z）后退一小段距离（默认 3cm），用于补偿物体中心到表面的偏移。

对比 Mode M（纯人手位姿）和 Mode N（物体位置+人手朝向），可以分离「定位」和「策略」的贡献。

### 目标位姿来源

```
multi_object_world_poses.npz
  └─ {object_name}__pose_world_wxyz[frame]     → 物体世界位置 (xyz)

hand_detections_<ID>.npz
  └─ {left,right}_gripper_rotation_matrix[frame] → 人手相机空间旋转
  └─ head_camera_pose_world_wxyz[frame]           → 相机外参

计算流程：
  1. 读取物体世界位置 obj_pos（Foundation 模型输出）
  2. 读取人手旋转矩阵，通过相机外参转换到世界空间 → rot_world
  3. 提取 rot_world 的第三列（local +Z）作为夹爪前进轴
  4. 沿前进轴后退 --foundation_pose_retreat_m（默认 3cm）：
     target_pos = obj_pos - retreat_m * approach_axis_world
  5. 目标位姿 = {hand_quat_world, target_pos}
  6. Planner 读取 plan_summary.json，执行正常的 pregrasp/grasp/action 管线
```

### 后退参数的物理含义

```
物体中心 ●──────────────● 物体表面（抓取点）
          ← retreat →

retreat = --foundation_pose_retreat_m (默认 0.03m = 3cm)
方向 = 夹爪前进轴 (蓝轴, local +Z) 的反方向

物体中心到表面的距离 ≈ 3cm（典型小物体半径）
```

`--approach_offset_m`（默认 0.12m）在此基础之上再做 pregrasp 后退：
```
最终 pregrasp 位置 = obj_pos - retreat_m * +Z - approach_offset_m * +Z
                   = 物体中心 - (3cm + 12cm) * 前进轴方向
```

### 与 Mode M 的区别

| | Mode M (Human Replay) | Mode N (Foundation Pose) |
|---|---|---|
| 位置来源 | 人手 gripper 世界位置 | Foundation 物体世界位置 |
| 朝向来源 | 人手旋转矩阵 | 人手旋转矩阵（相同） |
| 物体中心偏移 | 无 | `--foundation_pose_retreat_m` (3cm) |
| pregrasp 后退 | `--approach_offset_m` | 相同 |

### 脚本与参数

**核心脚本：** `/home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_foundation_pose.py`

关键参数：
| 参数 | 作用 | 默认值 |
|---|---|---|
| `--hand_keyframes_json` | hand_keyframes_all.json 路径 | 必需 |
| `--video_id` | episode ID | 必需 |
| `--foundation_pose_retreat_m` | 沿夹爪前进轴从物体中心后退到表面的距离 | **0.03 (3cm)** |
| `--approach_offset_m` | pregrasp 沿夹爪前进轴后退距离 | 0.12 |
| `--approach_axis` | 后退轴 | local_z (蓝轴) |

### 六任务 viewer 命令

Viewer 模式会在 bash wrapper 和 Python 中间层都移除 `CUDA_VISIBLE_DEVICES`，避免 SAPIEN 只能看到计算 GPU 而看不到驱动 VNC/X display 的 GPU。若日志仍出现 `Renderer does not support display`，先看 `[viewer] creating interactive viewer ...`：`CUDA_VISIBLE_DEVICES` 应为 `None/unset`，并且当前终端需要有可用的 `DISPLAY`（例如 `:1.0`）。

```bash
# pick_diverse_bottles (模式 A)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks pick_diverse_bottles \
  --foundation_pose_retreat_m 0.03 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/foundation_pose_viewer

# place_bread_basket (模式 B)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
  --gpu 2 --ids <ID> --viewer --tasks place_bread_basket \
  --foundation_pose_retreat_m 0.03 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/foundation_pose_viewer

# stack_cups (模式 B)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
  --gpu 2 --ids <ID> --viewer --tasks stack_cups \
  --foundation_pose_retreat_m 0.03 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/foundation_pose_viewer

# handover_bottle (模式 C)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
  --gpu 2 --ids <ID> --viewer --tasks handover_bottle \
  --foundation_pose_retreat_m 0.03 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/foundation_pose_viewer

# pnp_bread (模式 B/D)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
  --gpu 2 --ids <ID> --viewer --tasks pnp_bread \
  --foundation_pose_retreat_m 0.03 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/foundation_pose_viewer

# pnp_tray (模式 A)
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
  --gpu 2 --ids <ID> --viewer --tasks pnp_tray \
  --foundation_pose_retreat_m 0.03 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/foundation_pose_viewer
```

### 六任务无 viewer 批量

```bash
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
    --gpu 1 --ids 0 1 2 3 4 --continue_on_error --tasks $TASK \
    --foundation_pose_retreat_m 0.03 \
    --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay/keyframe/N_founposePhumanrot/foundation_pose
done

# 0611 / N-7
# 修复点：Foundation replay 的 pose_world_wxyz 实际顺序是 [x,y,z,qw,qx,qy,qz]。
# 旧 N-2/N-3 曾把 object pose[4:7] 当位置，实际取到 quaternion，导致 C 型夹爪飞到视野外。
# rank_previews/*.png 会叠加 Mode N 合成目标的 2D C 型夹爪、RGB 局部轴、目标 xyz 与 object->target 偏移。
# 如果 C 型夹爪在 head camera 视野外，图片边缘会显示 offscreen/behind_camera 标记，避免误判为没有生成。
# C 型夹爪左臂蓝色、右臂橙色；X=红、Y=绿、Z=蓝，其中 +Z 是 foundation_pose_retreat_m 使用的夹爪前进/后退轴。
# 轨迹插值说明：默认 urdfik_trajectory_mode=cartesian_interp_ik，会在“当前 TCP → 本 stage 目标 TCP”之间做位置线性插值和四元数 Slerp，
# 再逐个 waypoint 求 IK。若 IK 在中途切到另一组腕/肘解，末端看起来可能先朝下再扭回目标；这通常是 IK 解分支变化，不是关键帧目标本身在反向。
# N-7 距离设置：grasp target = object center 沿 local +Z 后退 0.10m；pregrasp = grasp 再后退 0.07m，因此 pregrasp 总 retreat 为 0.17m，pregrasp→grasp 前进 0.07m。
# N-7 借鉴 O.1.2：action 只使用第二关键帧 Foundation 物体 xyz，朝向和 retreat 方向保持第一关键帧 grasp 朝向，避免第二帧人手朝向带来额外 roll/IK 分支切换。
# N-7 同时启用 dual_stage_freeze_reached_arms_on_replan=1：某只手在 dual replan 中已达标后，后续 attempt 冻结该手，只补偿未达标手，避免“左手已到位又被下一次 replan 带走”。
# 校对结论：R1/AnyGrasp 的 candidate_keep_camera_up 按 local X 作为 forward；Mode N 当前 +Z 是前进轴，不能直接照搬。N 专用 foundation_pose_keep_top_axis_up 会绕 local +Z 做 180 度二选一，但 pick_diverse_bottles id=1 试验中 top_axis=y 变差，推荐先保持 0。
# id=1 对比：N-6 action 右臂约 38.9cm miss；N-7 仅保持 grasp 朝向后约 12.1cm miss；N-7 再冻结已达标手后 action 左/右约 2.78cm/2.07cm，双臂达标。
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
    --gpu 1 --ids 0 1 2 3 4 --continue_on_error --tasks $TASK \
    --foundation_pose_retreat_m 0.10 \
    --approach_offset_m 0.07 \
    --foundation_pose_action_orientation_source grasp \
    --dual_stage_freeze_reached_arms_on_replan 1 \
    --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-7_action_grasp_rot_freeze
done

# N-7 viewer 演示：显示每次规划目标的 target axis、top-1 C 型夹爪 actor、head/third camera 轴与 frustum
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --viewer_wait_at_end 1 --tasks pick_diverse_bottles \
  --debug_viewer_overlay \
  --foundation_pose_retreat_m 0.10 \
  --approach_offset_m 0.07 \
  --foundation_pose_action_orientation_source grasp \
  --dual_stage_freeze_reached_arms_on_replan 1 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-7_action_grasp_rot_freeze_debug_viewer
```

### retreat 参数调试

`--foundation_pose_retreat_m` 沿夹爪前进轴（蓝轴 local +Z）后退。较大的物体需要更大的 retreat：

```bash
# 大物体（如面包），增大 retreat 到 5cm
--foundation_pose_retreat_m 0.05

# 小物体（如杯子），默认 3cm
--foundation_pose_retreat_m 0.03

# 扁平物体，减小 retreat
--foundation_pose_retreat_m 0.01
```

此参数可以在 viewer 中验证：蓝色轴（local +Z）指向物体内部时，retreat 就是沿蓝轴反方向从物体中心退到表面的距离。

### 预期结果

消融实验三组对比：
| 实验 | 位置 | 朝向 | 预期 |
|---|---|---|---|
| L15.19.1 AnyGrasp | AnyGrasp 候选 | AnyGrasp 候选 | 基准（上界） |
| Mode M (Human Replay) | 人手位置 | 人手朝向 | 测试人手位姿是否足够 |
| **Mode N (Foundation Pose)** | **物体位置** | **人手朝向** | **测试物体定位+人手策略** |

- 如果 Mode N ≈ L15.19.1：Foundation 物体定位 + 人手朝向基本可替代 AnyGrasp
- 如果 Mode N << Mode M：人手位置信息比物体位置信息更关键
- 如果 Mode N > Mode M：物体定位比人手位置更准确（例如人手 tracking 噪声大时）

### 与 L15.19.1 原版 viewer 命令的兼容性

Mode M 和 Mode N 的 bash 脚本接受与 L15.19.1 相同的 viewer/gpu 参数：
```
--gpu --viewer --viewer_wait_at_end --continue_on_error --output_root
--trajectory_mode --cartesian_auto_step_m --ik_max_rotation_threshold_rad
--approach_offset_m
```

原有的六个 task viewer 模板（L15.19.2）**完全不受影响**，因为 Mode M/N 使用独立的脚本和输出目录。

如果最小探针可以打开 SAPIEN viewer：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py
```

但 Mode M/N viewer 失败，优先检查是否在旧代码里由 `plan_keyframes_human_replay.py` 或 `plan_keyframes_foundation_pose.py` 把 `CUDA_VISIBLE_DEVICES` 又设回了 `--gpu`。当前已修复：`--enable_viewer 1` 时 Python 调 planner 前会 `pop("CUDA_VISIBLE_DEVICES")`；非 viewer 模式仍按 `--gpu` 设置计算 GPU。

### 实现文件

| 文件 | 说明 |
|---|---|
| `code_painting/plan_keyframes_human_replay.py` | Mode M 核心：人手位姿目标生成 + planner 调用 |
| `code_painting/plan_keyframes_foundation_pose.py` | Mode N 核心：物体位置+人手朝向目标生成 + planner 调用 |
| `code_painting/run_plan_keyframes_human_replay_piper_d435.sh` | Mode M 单任务 bash 入口 |
| `code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh` | Mode N 单任务 bash 入口 |

技术要点：
- 通过 `--reuse_plan_summary_json` 复用 planner 的完整执行管线（pregrasp/grasp/action + IK）
- 目标位姿计算复用 `CV_TO_WORLD_CAMERA_PRESETS` 和 `camera_cv_axis_mode` 与 direct replay 相同
- plan_summary.json 的 `selected_candidates_by_executed_arm` 格式支持独立左右臂关键帧
- 不对 `plan_anygrasp_keyframes_r1.py` 做任何修改

---

## O.0 对比实验：Piper/Pika 数据生成与 IK 诊断命令

O.0 当前只保留 4 条有明确用途的命令。`pick_diverse_bottles_piper demo_clean_piper_calibrated` 已不再作为推荐命令保留，因为它会进入原始 `grasp_actor`，在 `tmux gen1-1/gen1-2` 中持续失败。

最新 `tmux gen1-1/gen1-2` 结论：

```text
seed 72-115: Objects is unstable / target_pose cannot be None for move action
final: Ctrl-C interrupted
```

含义：
- `Objects is unstable` 是原始瓶子随机摆放后的物理稳定性检查失败。
- `target_pose cannot be None for move action` 来自原始 `pick_diverse_bottles.py -> grasp_actor -> choose_grasp_pose` 没有为标定 Piper/Pika 找到可执行抓取目标。
- `No left camera link` / `No right camera link` 不是本次失败原因；当前 `piper_pika_agx.urdf` 没有 `left_camera`、`right_camera` 或 `camera` link，所以 O.0 配置保持 `collect_wrist_camera: false`，只保存 head 视角。

#### O.0-1 已跑通：无 viewer 生成 head-only 数据

用途：生成可保存的 O.0 对照数据。该命令走 `pick_diverse_bottles_piper_motion`，保留原始瓶子随机采样和稳定性检查，但不调用原始 IK；动作为标定 Piper/Pika 的关节空间 motion baseline。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_motion demo_clean_piper_motion 0
```

输出示例：

```text
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/data/episode0.hdf5
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/video/episode0.mp4
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/_traj_data/episode0.pkl
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/instructions/episode0.json
```

#### O.0-2 已跑通：带运动 viewer 检查 motion baseline

用途：在 SAPIEN viewer 中看 O.0 motion baseline 的后续运动。该命令现在调用 `view_pick_diverse_bottles_piper_motion.py`，每次都会重新找稳定 seed 并执行一次 `play_once()`，不会被 `collect_data.py` 的旧 `seed.txt` 进度短路；`collect_data: false`，不保存 hdf5。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_pick_diverse_bottles_piper_motion_viewer.sh
```
```bash
  source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_motion --task_config demo_clean_piper_motion_viewer --seed 0 --max_seed_tries 50
```

无窗口 smoke 验证：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && DISPLAY=:1.0 timeout 120s bash run_pick_diverse_bottles_piper_motion_viewer.sh --seed 0 --max_seed_tries 3 --hold 0
```

状态：2026-06-04 已验证通过；seed 0/1 因瓶子不稳定被跳过，seed 2 加载后完成 `play_once()`。

#### O.0-2 坐标轴识别指南（2026-06-05 更新 v2 — 仅 tip，中文色名）

viewer 中每个坐标轴（红=+X, 绿=+Y, 蓝=+Z）会有一个**原点色块**，颜色对应类别。
**所有 EE / 阶段目标轴均为夹爪尖端位置**（腕部 link6 FK + 10cm 前进偏移），不再显示腕部。

| 原点色 | 中文色名 | 名称前缀 | 含义 |
|---|---|---|---|
| ■ | 白色 | `bottle_center_*` | 瓶子几何中心 |
| ■ | 亮黄色 | `place_target_*` | 放置目标位姿 |
| ■ | 亮青色 | `ee_current_*` | 当前夹爪尖端 (腕部+10cm) |
| ■ | 浅蓝色 | `stage_pregrasp_*` | 预抓取阶段 目标 |
| ■ | 浅绿色 | `stage_grasp_lower_*` | 下降抓取阶段 目标 |
| ■ | 金黄色 | `stage_lift_*` | 抬升阶段 目标 |
| ■ | 橙红色 | `stage_move_out_*` | 移出阶段 目标 |

**查找方法**：
1. 先看原点色块颜色 → 确定类别（对照终端 AXIS_LEGEND）
2. 红 = 局部 +X, 绿 = 局部 +Y, **蓝 = 局部 +Z（夹爪前进方向）**
3. 瓶子中心和放置目标在 y>0 方向，阶段目标在 y<0 方向
4. 如需看腕部位置，可在代码中临时调小 `_forward_offset_pose` 的 distance 参数

终端也会打印完整 AXIS_LEGEND 图例。

#### O.0-2 Piper/Pika 基座与工作空间分析（2026-06-04 实测）

**Piper base 位置**（from `assets/embodiments/piper_pika_agx/config.yml` `robot_pose`）：
- 左臂 base: `(-0.30, -0.25, 0.75)` → **y = -0.25**
- 右臂 base: `(0.556, -0.272, 0.770)` → **y ≈ -0.27**

**Home 位姿 EE 位置**（FK 实测，seed=2）：
- 腕部 (link6): 左 `y≈-0.128`, 右 `y≈-0.145`
- 尖端 (+10cm fwd): 左 `y≈-0.033`, 右 `y≈-0.050`

**瓶子 y 范围**: `[0.03, 0.23]`
- 距 base: 左臂 0.28~0.48m，右臂 0.30~0.50m → **Piper 桌面臂展 ~0.6m，完全可行** ✓
- 距 home tip: 最近仅 ~0.06~0.08m → **home 位姿已接近瓶子** ✓

**阶段目标 EE 位置**（FK 实测）：
- 腕部都在 `y≈-0.40~-0.47` 范围
- 比 home 腕部 **后退了约 0.27~0.34m** ✗

**结论**：`ylim=[0.03, 0.23]` 在 Piper workspace 内，**瓶子可及**。问题不在瓶子范围，而在 `_motion_joint_targets()` 的关节偏移量将手臂向 **后方**（y 负方向）移动，而非向前接近瓶子。要真正完成抓取，需要重新设计关节目标，使 EE 从 home 位姿向 y 正方向（瓶子方向）伸展。

终端阶段日志（仅 tip，无 wrist）：

```text
[piper-motion][setup] bottle ranges left=x[-0.30,-0.18],y[0.03,0.23] right=x[0.30,0.46],y[0.03,0.23]
[piper-motion][target-axis] ee_current (tip +10cm) left_pos=[...] right_pos=[...]
[piper-motion][target-axis] stage_pregrasp (tip) left_pos=[...] right_pos=[...]
[piper-motion][target-axis] stage_grasp_lower (tip) left_pos=[...] right_pos=[...]
[piper-motion][target-axis] stage_lift (tip) left_pos=[...] right_pos=[...]
[piper-motion][target-axis] stage_move_out (tip) left_pos=[...] right_pos=[...]
[piper-motion][stage] pregrasp: planning joint interpolation
[piper-motion][stage] grasp_lower: planning joint interpolation
[piper-motion][stage] close_gripper: start/finished
[piper-motion][stage] lift: planning joint interpolation
[piper-motion][stage] move_out: planning joint interpolation
[piper-motion][stage] open_gripper: start/finished
```

说明：O.0 motion baseline 保留原始 ALOHA/AgileX 的瓶子 y 范围 `[0.03, 0.23]`，经实测该范围在 Piper workspace 内可行。当前阶段的 EE 目标 (y≈-0.40~-0.47) 比 home 位姿更靠后，原因是关节偏移量设计为向后运动 — 后续需要重新设计关节目标使 EE 向瓶子方向 (y 正方向) 移动。

#### O.0 根因分析：为什么阶段目标跑到了 base 后方？（2026-06-05）

**原始 `pick_diverse_bottles.py` 的逻辑（Cartesian 空间、IK 驱动）**：

1. `grasp_actor(bottle, arm_tag, pre_grasp_dis=0.08)` 读取瓶子的 **contact point 世界坐标**
2. `get_grasp_pose()` 基于瓶子位姿计算 Cartesian pregrasp 位姿：瓶子前方 -0.12-pre_dis 处（即 **瓶子后方偏移**）
3. `choose_grasp_pose()` 遍历候选 contact points，对每个调用 `robot.left_plan_path(pre_pose)` 做 **IK 可行性检查**
4. 选出 IK 能成功的最优候选 → 返回 (pregrasp_pose, grasp_pose)
5. 执行顺序：**pregrasp（瓶子后方）→ grasp（瓶子处）→ lift → place**

关键：**所有位置都是从瓶子实际坐标算出来的 Cartesian 位姿，依赖 IK 求解到关节角。**

**当前 `pick_diverse_bottles_piper_motion.py` 的逻辑（关节空间、硬编码偏移）**：

1. `_motion_joint_targets()` 返回的是 **硬编码关节偏移**：`home_left + [0.20, 0.08, -0.12, 0.00, -0.05, 0.10]`
2. 这些数字 **完全不读瓶子位置**，也不做 IK
3. `_script_joint_stage()` 在关节空间做线性插值

**对比 ALOHA vs Piper 的 home state**：

| | ALOHA/AgileX | Piper/Pika |
|---|---|---|
| home state | `[0, 0, 0, 0, 0, 0]` (全零) | `[0.0, 0.8, 1.2, 0.0, -0.4, 0.0]` |
| base y | -0.65 | -0.25 (左) / -0.27 (右) |
| dual_arm | True (单URDF双臂) | False (独立URDF) |
| EE 方向 | 前伸 (y+) | 前伸 (y+) |

**根因**：

1. **原始 IK 链路在 Piper 上失败**：`choose_grasp_pose` → `robot.left_plan_path` → `CuroboPlanner.plan_path` 对 Piper 返回失败，因为 Piper 的 link 长度、关节限位与 ALOHA 不同，Cartesian grasp 候选位姿对 Piper **不可达**。这就是 `target_pose cannot be None for move action` 的来源。

2. **关节偏移量是为 ALOHA 全零 home 调的**：同样的 `joint1+0.2, joint2+0.08, joint3-0.12` 在 ALOHA（home 全零）上产生向前伸展的 Cartesian 运动，但在 Piper（home=[0,0.8,1.2,0,-0.4,0]）上由于关节起始角度完全不同，**同样的增量导致完全不同的末端位置**——Piper 上这些偏移让手臂向后缩回了 ~0.4m。

3. **异构问题的本质**：不是照搬了原始逻辑——恰恰相反，**正是因为原始 IK 链路走不通，才改用了关节空间硬编码方案**。但这个关节偏移方案是从 ALOHA 的 home pose 出发设计的，对 Piper 的标定 home pose 不适用。

**正确方向**：
方案 A：修复 Piper 的 IK/抓取链路 — 标定 EE grasp convention、修正 choose_grasp_pose 的候选评估、确保 Curobo 能对 Piper 求解
- 方案 B：为 Piper home pose 重新手动设计关节目标 — 从 home tip (y≈-0.03) 向瓶子 (y=0.03~0.23) 方向伸展，而非后退

#### O.0 自定义 IK 分析与集成方案（2026-06-05）

**自定义 IK 代码位置**：
- /home/zaijia001/ssd/RoboTwin/agent-read/ik_analyze/ik.py — URDFInverseKinematics 类
- 使用 Curobo 的 IKSolver（纯 IK），配合 seed_config 从当前关节角出发求解

**现有 planner vs 自定义 IK 对比**：

| | CuroboPlanner (现有 plan_path) | URDFInverseKinematics (自定义) |
|---|---|---|
| 底层 | MotionGen.plan_single() | IKSolver.solve_batch() |
| 求解类型 | 轨迹优化 (trajectory opt) | 纯 IK (单点求解) |
| 碰撞检测 | 有 (table + self-collision) | 无 (self_collision_check=False) |
| Seed 方式 | start_joint_states 轨迹起点 | seed_config IK 搜索起点 |
| 输出 | 完整轨迹 (position+velocity) | 单个关节配置 |
| 速度 | 慢（优化问题） | 快（纯 IK） |
| 配置 URDF | piper/piper.urdf (臂 only) | 用户指定 |
| frame_bias | [0., 0., 0.] | N/A |

**现有 planner 在 Piper 上失败的原因**：
- MotionGen.plan_single() 做的是完整轨迹优化（碰撞+平滑+关节限位），不只是 IK
- Piper curobo.yml 引用的 URDF 是 piper/piper.urdf（臂 only），与 SAPIEN 仿真用的 piper_pika_agx/piper_pika_agx.urdf（臂+夹爪）不同
- frame_bias: [0., 0., 0.] 不做额外偏移，但碰撞 link 列表 (link7/link8) 与 piper_pika_agx URDF (gripper_base_link/gripper_left_link/gripper_right_link) 不匹配

**自定义 IK 能否接入当前控制链路？**

答案：**可以，但需要适配层**。

目标：恢复原始 pick_diverse_bottles.py 的 Cartesian 流程：



**需要修改的地方**：

1. **在 robot.py 添加 ik_check 方法**：
   - 实例化 URDFInverseKinematics(urdf_file=piper_pika_agx.urdf, base_link=base_link, ee_link=link6)
   - 用 solve_ik(target_pos, target_quat, current_joints=now_qpos) 做可行性检查
   - 返回 {status: Success/Fail}

2. **修改 choose_grasp_pose**（Piper 专用版）：
   - 把 plan_func(pre_pose) 替换为新的 ik_check 方法
   - 只在检查通过后，用 plan_path 生成实际轨迹

3. **关于平滑路径**：
   - 自定义 IKSolver 只输出单点，不含速度/加速度
   - 平滑路径需要 MotionGen：先用自定义 IK 确认目标可达 -> 再用 MotionGen.plan_single(start, goal) 生成轨迹
   - 或使用关节空间线性插值（当前 _joint_result 方案），适合无障碍物场景

**推荐的集成路径**：



**为什么不能直接替换**：
- plan_path 接口要求返回 {status, position, velocity} 完整轨迹
- 自定义 IK 只有 status + 单点 solution
- 需要包装层：IK 成功 -> solution 转 trajectory（插值或调 MotionGen）

**重要**：O.0-2 有两个 viewer 入口：
- view_pick_diverse_bottles_piper_scene.py -> 只看场景和坐标轴，不执行 play_once()
- view_pick_diverse_bottles_piper_motion.py -> 执行 play_once() 运动，然后保持窗口
  命令：python view_pick_diverse_bottles_piper_motion.py --task_name pick_diverse_bottles_piper_motion --task_config demo_clean_piper_motion_viewer --seed 0 --max_seed_tries 50

#### O.0-3 只看场景：纯 scene viewer，不执行动作

用途：只检查标定 Piper/Pika、桌面、瓶子随机摆放、目标坐标轴和 viewer 是否能打开。该命令不进入 `play_once`，不会执行运动，也不会保存数据；现在会通过 `skip_planner=True` 跳过 Curobo planner 初始化，避免 scene-only viewer 卡在 Curobo warmup。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50
```

无窗口 smoke 验证：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && DISPLAY=:1.0 timeout 90s python view_pick_diverse_bottles_piper_scene.py --seed 0 --max_seed_tries 3 --hold 0
```

状态：2026-06-04 已验证通过；seed 0/1 因瓶子不稳定被跳过，seed 2 加载稳定场景，添加坐标轴后渲染一帧退出。

如果要只看 O.0 motion baseline 的场景和所有阶段目标轴，但不执行动作，用这个命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_motion --task_config demo_clean_piper_motion_viewer --seed 0 --max_seed_tries 50
```

#### O.0-4 诊断用：原始 IK/规划链路，不作为当前采集命令

用途：验证“标定 Piper/Pika URDF + 原始 `pick_diverse_bottles.py` 的 IK/规划逻辑”。该命令使用 `pick_diverse_bottles_piper`，会进入原始 `grasp_actor/place_actor -> robot.left/right_plan_path -> CuroboPlanner.plan_path` 链路；embodiment 使用 `piper_pika_agx_ik_orig_tcp`。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && timeout 120s bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_ik_orig_tcp 0
```

状态：已确认能进入 `Embodiment Config: piper_pika_agx_ik_orig_tcp+piper_pika_agx_ik_orig_tcp` 和原始 task/IK 链路，但没有完成 episode。最新 `tmux gen1-1/gen1-2` 仍显示大量 `Objects is unstable` 和 `target_pose cannot be None for move action`。因此它只用于定位 IK/抓取候选问题，不是当前推荐的数据生成命令。

原始 IK 链路：

```text
pick_diverse_bottles.py
-> grasp_actor/place_actor
-> Action(move target_pose)
-> Base_Task.move
-> robot.left_plan_path / robot.right_plan_path
-> robot._trans_from_gripper_to_endlink
-> CuroboPlanner.plan_path
```

```bash
  # V1 场景 viewer
  source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v1 --seed 0 --max_seed_tries 50


  source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v1 --seed 0 --max_seed_tries 50


  # V1 运动执行
  python /tmp/test_ik_motion.py

  # 数据采集 (V1-V4)
  bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0
  bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v2 0
  bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v3 0
  bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v4 0
```



## O. 对比实验：第一帧 FoundationPose 直接策略抓取 pick_diverse_bottles

### 设计目的

Mode O 用于测试一个更强约束的 baseline：**不依赖手工关键帧、不使用人手朝向、不使用 AnyGrasp 候选筛选**，只读取第 0 帧 FoundationPose 的两个瓶子世界位置，然后按 `/home/zaijia001/ssd/RoboTwin/envs/pick_diverse_bottles.py` 的任务逻辑生成抓取与放置目标。

`pick_diverse_bottles.py` 的数据生成/执行逻辑要点：

- 两个 bottle 不是固定两个离散朝向；`rand_create_actor(... rotate_rand=True, rotate_lim=[0,1,0])` 表示创建时会随机旋转，主要限制在 y 轴相关随机。
- 左瓶位置采样在 `xlim=[-0.25,-0.05], ylim=[0.03,0.23]`，右瓶位置采样在 `xlim=[0.05,0.25], ylim=[0.03,0.23]`。
- 执行时固定左瓶用左臂、右瓶用右臂。
- 抓取阶段是 `grasp_actor(... pre_grasp_dis=0.08)`，随后 `move_by_displacement(... z=0.1)` 抬起，再 `place_actor` 到左/右目标位。

### Mode O 当前实现

新增代码入口：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/plan_first_frame_foundation_pick_diverse_bottles.py
/home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh
```

实现逻辑：

1. 读取 `foundation_replay_d435/foundation_input_<ID>/multi_object_world_poses.npz`。
2. 使用第 `--foundation_frame` 帧，默认 `0`。
3. 读取 `left_bottle__pose_world_wxyz` 与 `right_bottle__pose_world_wxyz` 的前三位作为世界位置。注意：该 npz 键名包含 `wxyz`，但实际 planner/replay 使用顺序是 `[x, y, z, qw, qx, qy, qz]`。
4. 固定策略式抓取：
   - 左臂 local `+Z` 从左向右接近，世界方向 `[+1,0,0]`。
   - 右臂 local `+Z` 从右向左接近，世界方向 `[-1,0,0]`。
   - 抓取目标从物体中心沿接近轴反方向退 `--grasp_surface_retreat_m`，默认 `0.03m`。
   - planner 的 pregrasp 再沿 local `+Z` 后退 `--approach_offset_m`，默认 `0.08m`，对齐原 env 的 `pre_grasp_dis=0.08`。
5. 为每只手生成两个 synthetic keyframe：
   - keyframe 0：第一帧 Foundation 设计的 grasp target。
   - keyframe 1：lift+move 后的放置 target。
6. 通过 `--reuse_plan_summary_json` 复用现有 Piper planner 的 pregrasp/grasp/close/action、IK、视频和 debug 输出逻辑。

### 夹爪朝向/轴约定检查

本轮检查结论：Piper/Pika 的 `robot_config_PiperPika_agx_dual_table_0515.json` 与原始 ALOHA-AgileX 配置在 `global_trans_matrix=[[1,0,0],[0,-1,0],[0,0,-1]]`、`delta_matrix=I`、`grasp_perfect_direction=["front_right","front_left"]` 这些高层配置上是一致的，问题不在这里。

真正需要注意的是 URDF 夹爪几何与 target frame 约定不同：

- ALOHA-AgileX 的 `fl/fr_link6` 后面，finger depth/掌根到指尖方向更自然地落在 link6 local `+X`，两根夹指开合沿 local `+/-Y`。
- Piper/Pika AGX 的 gripper finger prismatic joint 使用 local `Z/-Z` 作为开合方向，夹爪结构轴和 ALOHA-AgileX 不完全同构。
- 当前 Mode O 沿用已标定 Piper/replay 管线：保存给 planner 的 target frame 使用 local `+Z`（蓝轴）作为接近/前进轴。因此左臂 target local `+Z=[+1,0,0]`，右臂 target local `+Z=[-1,0,0]`。

所以：当前 Mode O 的定义与前面 Piper direct replay / robot-frame AnyGrasp 的执行约定一致，但不是“原始 ALOHA-AgileX local +X 指尖深度”约定。如果要做严格的 ALOHA-style 对比，需要在 Mode O 中额外加一个分支，把接近轴写到 target local `+X`，并让 planner 使用 `--approach_axis local_x`；否则就继续把 ALOHA/AnyGrasp raw local `+X` 显式映射到 Piper/replay target local `+Z`。

### Mode O 轴约定验证方法

新增静态可视化脚本，不跑 IK、不启动 SAPIEN viewer，只读取 FoundationPose 第 0 帧并画出同一个抓取点下的三套 target frame：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/visualize_mode_o_gripper_frame_conventions.py
```

推荐先跑：

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python /home/zaijia001/ssd/RoboTwin/code_painting/visualize_mode_o_gripper_frame_conventions.py --video_id 0 --foundation_frame 0 --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/mode_o_frame_convention_debug
```

输出：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/mode_o_frame_convention_debug/pick_diverse_bottles_id0_frame0_gripper_frame_conventions.png
/home/zaijia001/ssd/RoboTwin/code_painting/mode_o_frame_convention_debug/pick_diverse_bottles_id0_frame0_gripper_frame_conventions.json
```

图中红/绿/蓝分别是 local `+X/+Y/+Z`，黑色箭头是物理侧向接近方向。id0 实测数值：

```text
piper_local_z:       local Z 与物理接近方向夹角 0deg，planner_axis=local_z
aloha_local_x_y_up:  local X 与物理接近方向夹角 0deg，planner_axis=local_x
aloha_local_x_z_up:  local X 与物理接近方向夹角 0deg，planner_axis=local_x
```

如果要“试原来的 ALOHA-style 设置”但不先跑完整 IK，可以只写 summary：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --plan_only --target_frame_convention aloha_local_x_z_up --output_root /tmp/mode_o_aloha_local_x_plan_only
```

该命令会生成 `plan_summary_first_frame_foundation.json`，其中 `target_frame_convention=aloha_local_x_z_up`、`planner_approach_axis=local_x`，不会执行 planner。若确认可视化方向更合理，再去掉 `--plan_only` 并加 `--viewer --viewer_wait_at_end 1` 做 SAPIEN viewer 检查。

默认放置点沿用 env：

```text
left:  [-0.06, -0.105, 1.0]
right: [ 0.06, -0.105, 1.0]
```

如果想只做“物体中心高度 + 0.1m 抬升”，可加：

```text
--place_z_mode object_plus_lift --lift_m 0.10
```

### 推荐先跑单条 smoke

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --continue_on_error --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation_smoke
```

### Viewer 调试

如果日志出现：

```text
[viewer-warning] failed to create interactive viewer ... Renderer does not support display
```

先检查最小 viewer。SAPIEN viewer 需要能看到驱动当前 VNC/X display 的 GPU；如果设置了 `CUDA_VISIBLE_DEVICES=2`，可能把 display 对应 GPU mask 掉。Mode O wrapper 已修复为 viewer 模式下不再把 `CUDA_VISIBLE_DEVICES` 写回 Python/planner 子进程，正常日志应显示 `CUDA_VISIBLE_DEVICES=None` 或 `unset`。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py
```

如果这个最小探针仍报 `Renderer does not support display`，说明当前 shell 的 `DISPLAY`/Vulkan 图形会话不可用，需要切到能打开 SAPIEN viewer 的 VNC/图形终端，或修复 X11/Wayland forwarding；这时 Mode O 会自动 fallback 到 offscreen 并继续生成视频。

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 2 --viewer --viewer_wait_at_end 1 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation_viewer
```

### 批量命令

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0-10 --continue_on_error --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/3_first_frame_foundation
```

### 已执行情况

已完成静态验证：

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_first_frame_foundation_pick_diverse_bottles.py
bash -n /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --ids 0 --dry_run
```

已运行 `pick_diverse_bottles id0` 无 viewer smoke。输出目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation_smoke/pick_diverse_bottles/foundation_input_0
```

本次 id0 的第 0 帧目标：

```text
left_bottle  center=(-0.038, 0.096, 0.733), grasp=(-0.068, 0.096, 0.733), action=(-0.060, -0.105, 1.000)
right_bottle center=( 0.230, 0.114, 0.745), grasp=( 0.260, 0.114, 0.745), action=( 0.060, -0.105, 1.000)
```

执行结果：

- pregrasp：左右臂均 reached。
- grasp：右臂第一次达到位置容差附近，左臂未到位；后续 replanning 中右臂 Cartesian waypoint IK 失败。
- 因 `--require_keyframe1_reached_before_close 1`，双臂 grasp 未同时 reached 时没有关爪，也没有进入 action。
- 已生成 `head_cam_plan.mp4`、`third_cam_plan.mp4`、`plan_summary_first_frame_foundation.json`、`pose_debug.jsonl`。

### 需要确认的部分

- 固定抓取朝向是否接受：当前 left local `+Z=[+1,0,0]`、right local `+Z=[-1,0,0]`，即两臂从外侧水平夹取。
- `--grasp_surface_retreat_m=0.03` 是否适合当前瓶子 mesh；如果 grasp 偏深/偏浅，优先扫 `0.01/0.03/0.05`。
- action z 是否使用 env 目标 `1.0`，还是改成 `object_z + 0.1` 只做严格 lift。
- 是否允许在 Mode O 放宽 `require_keyframe1_reached_before_close` 做“即使一臂未严格 reached 也尝试 close/action”的对比；当前默认保守，不会在第一阶段失败时关爪。

  bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 1  --viewer --viewer_wait_at_end 1 --target_frame_convention aloha_local_x_z_up --output_root /tmp/mode_o_aloha_local_x_plan_only

---

## O.0-5 新增：Piper IK Cartesian 抓取命令（2026-06-05）

**代码位置**：
- IK planner: `envs/robot/piper_ik.py` (4 个变体 V1-V4)
- Task: `envs/pick_diverse_bottles_piper_ik.py`
- Configs: `task_config/demo_piper_ik_seq_v{1,2,3,4}.yml`
- Motion viewer: `view_pick_diverse_bottles_piper_ik_motion.py`

**关键修复**：IK solver 必须使用 `piper_pika_agx/piper_pika_agx.urdf`（与 SAPIEN 仿真一致），而非 curobo.yml 中的 `piper/piper.urdf`（关节原点不同，IK 全部失败）。

**gen1-1/gen1-2 问题说明**：两个 tmux 都因旧版 CuroboPlanner 初始化时 `SelfCollisionCost` 的 GPU 张量分配卡死。PiperIKPlanner 设置 `self_collision_check=False` 绕过了此问题。

### 版本对比

| 版本 | 策略 | 插值 | 碰撞检测 | 速度 | 成功率 |
|---|---|---|---|---|---|
| V1 | IKSolver seed | 线性 | 无 | 最快 | 中 |
| V2 | IKSolver seed | 三次样条 | 无 | 快 | 中 |
| V3 | IKSolver + MotionGen | 轨迹优化 | 有 | 慢 | 较高 |
| V4 | 多种子 IKSolver | 三次样条 | 无 | 中 | 最高 |

### V1 命令（推荐默认 — 最快最可靠）

```bash
# 运动 viewer（执行 play_once + 保持窗口）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50

# 场景 viewer（只看坐标轴 + 场景，不执行运动）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v1 --seed 0 --max_seed_tries 50

# 无窗口 smoke 验证
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && unset DISPLAY && SAPIEN_RT_DENOISER=none timeout 180s python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 10 --hold 0 --render_freq 0 --show_axes 0 --require_success 1

# 数据采集（生成 hdf5）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0

# 连续 10 episode（自动找 stable seed，episode 间延时 2s）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 10 --episode_delay 2.0 --seed 0 --max_seed_tries 50
```

### V2 命令（三次样条 — 更平滑）

```bash
# 运动 viewer
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v2 --seed 0 --max_seed_tries 50

# 场景 viewer
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v2 --seed 0 --max_seed_tries 50

# 无窗口 smoke 验证
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && unset DISPLAY && SAPIEN_RT_DENOISER=none timeout 180s python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v2 --seed 0 --max_seed_tries 10 --hold 0 --render_freq 0 --show_axes 0 --require_success 1

# 数据采集
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v2 0
```

### V3 命令（MotionGen — 碰撞感知轨迹优化）

```bash
# 运动 viewer
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v3 --seed 0 --max_seed_tries 50

# 场景 viewer
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v3 --seed 0 --max_seed_tries 50

# 无窗口 smoke 验证
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && unset DISPLAY && SAPIEN_RT_DENOISER=none timeout 180s python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v3 --seed 0 --max_seed_tries 10 --hold 0 --render_freq 0 --show_axes 0 --require_success 1

# 数据采集
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v3 1
```

### V4 命令（多种子 — 最高成功率）

```bash
# 运动 viewer
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v4 --seed 0 --max_seed_tries 50

# 场景 viewer
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v4 --seed 0 --max_seed_tries 50

# 无窗口 smoke 验证
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && unset DISPLAY && SAPIEN_RT_DENOISER=none timeout 180s python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v4 --seed 0 --max_seed_tries 10 --hold 0 --render_freq 0 --show_axes 0 --require_success 1

# 数据采集
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v4 2
```


### V1-V4 统一命令（2026-06-08 更新，6步流程 + approach方向修正）

```bash
# === V1 (推荐) ===
# 运动 viewer（6步：pregrasp→grasp→close→lift→place→open）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50

# 逐步确认模式
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --step_mode 1 --seed 0 --max_seed_tries 50

# 连续 10 episode
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 10 --episode_delay 2.0 --seed 0 --max_seed_tries 50

# 场景 viewer
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v1 --seed 0 --max_seed_tries 50

# 数据采集
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0

# === V2 ===
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v2 --seed 0 --max_seed_tries 50

# === V3 ===
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v3 --seed 0 --max_seed_tries 50

# === V4 ===
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v4 --seed 0 --max_seed_tries 50
```
状态：2026-06-05 V1 motion viewer 和 scene viewer 均已验证通过（seed=2，IK 求解成功，pregrasp/grasp/gripper 完整执行）。

---

## O.0-5 更新 (2026-06-08)：Piper 专用瓶子范围 & gen 状态

### 瓶子范围 & 放置目标（覆盖 ALOHA 原始值）

Piper 双臂基座比 ALOHA 间距更大（左 x≈-0.30，右 x≈0.556），因此覆盖了父类 pick_diverse_bottles 的 load_actors()：

| 参数 | ALOHA 原始值 | Piper IK 新值 |
|---|---|---|
| 左瓶 xlim | [-0.25, -0.05] | [-0.35, -0.18] |
| 右瓶 xlim | [0.05, 0.25] | [0.30, 0.50] |
| ylim | [0.03, 0.23] | [0.03, 0.23] (不变) |
| 左放置目标 | [-0.06, -0.105, 1.0] | [-0.28, -0.15, 1.0] |
| 右放置目标 | [0.06, -0.105, 1.0] | [0.48, -0.15, 1.0] |

改动原因：Piper 基座比 ALOHA 外移约 30cm，瓶子需要跟随外移；放置目标 y 从 -0.105 改为 -0.15（更靠近 Piper home tip y≈-0.03 方向）。

### gen tmux 状态 (2026-06-08)

- gen1-9: `view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1` 运行正常，play_once 完整执行
- gen2-10: 已结束（空 shell）
- gen1-1 / gen1-2: 已关闭（旧版 CuroboPlanner SelfCollisionCost GPU 卡死）

### 运动 viewer 快速入口

```bash
# V1 (推荐默认)
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50

# V2
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v2 --seed 0 --max_seed_tries 50

# V3
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v3 --seed 0 --max_seed_tries 50

# V4
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v4 --seed 0 --max_seed_tries 50
```

### 场景 viewer 快速入口

```bash
# V1
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v1 --seed 0 --max_seed_tries 50

# V2-V4 同理替换 demo_piper_ik_v{2,3,4}
```

---

## O.0-5 更新 (2026-06-08 #2)：连续多 episode + 右手范围修正

### 连续多 episode viewer

新增 `--num_episodes N` 参数，自动连续运行 N 个 seed，episode 间短暂保持 viewer：

```bash
# V1 连续 10 个 episode
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 10 --episode_delay 2.0 --seed 0 --max_seed_tries 50

# V2 连续 10 个 episode
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v2 --num_episodes 10 --episode_delay 2.0 --seed 0 --max_seed_tries 50

# V3 连续 10 个 episode
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v3 --num_episodes 10 --episode_delay 2.0 --seed 0 --max_seed_tries 50

# V4 连续 10 个 episode
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v4 --num_episodes 10 --episode_delay 2.0 --seed 0 --max_seed_tries 50
```

参数说明：
- `--num_episodes 10`：连续运行 10 个 episode（自动跳过 unstable seed）
- `--episode_delay 2.0`：每个 episode 完成后保持 viewer 2 秒再进入下一个
- 调小 `--episode_delay` 可加快连续运行速度
- 终端会打印每个 episode 的进度 `Episode N/M: seed=X FINISHED (total_ok=Y)`

### 右手范围修正 (2026-06-08)

右臂基座位于 x≈0.556，原右瓶范围 x=[0.30, 0.50] 全部在基座左侧，导致右臂需跨身体够取。
修正为 x=[0.38, 0.52]，使右瓶更靠近右臂自然 workspace。

| 参数 | 修正前 | 修正后 |
|---|---|---|
| 右瓶 xlim | [0.30, 0.50] | [0.38, 0.52] |
| 右放置目标 x | 0.48 | 0.52 |

### 机器人不运动修复 (2026-06-08)

根因：play_once 只调用了 set_arm_joints（设 PD 目标），未调用 scene.step()（物理步进）。
修复：轨迹循环中添加 scene.step()，关节才会实际运动到目标位置。

### V3 MotionGen 修复 (2026-06-08)

- 添加 `**kwargs` 支持工厂函数传参
- MotionGen 初始化异常捕获 → 降级为 IK + 三次样条插值

---

## O.0-5 更新 (2026-06-08 #3)：collect_data 修复 + 连续 episode 命令整理

### collect_data GPU 卡死修复

**错误**：`bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0` 时，
setup_demo → CuroboPlanner 初始化 → MotionGen.warmup() GPU 张量分配卡死。

**根因**：config YAML 缺少 `skip_planner: true`。父类 `pick_diverse_bottles.setup_demo()`
会初始化 CuroboPlanner（含 collision checking），在 Piper 上 GPU 卡死。

**修复**：`demo_piper_ik_v{1,2,3,4}.yml` 全部添加 `skip_planner: true`。
PiperIKPlanner 在 setup_demo 末尾独立初始化（含 `self_collision_check=False`），
不受 skip_planner 影响。

### 调试保存：save_all_episodes

在 config YAML 中添加 `save_all_episodes: true` 可强制保存失败 episode 的 hdf5，
方便回放分析 IK 失败原因：

```yaml
# demo_piper_ik_seq_v1.yml 末尾添加：
save_all_episodes: true
```

此时 `check_success()` 始终返回 True，collect_data 会保存所有 episode 数据。
终端也会打印详细的 bottle 位置 vs target 距离信息：
```
[piper-ik][check] success=False b1=(-0.261,0.119,0.740) t1=(-0.28,-0.15) b1_z_ok=True dist1=0.270 ...
```

### viewer 预渲染修复

在 `play_once()` 前增加 30 帧预渲染，确保 SAPIEN 窗口已打开可见，
不会错过运动过程。

### V1 连续 10 episode 命令（已整合到 V1 section）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 10 --episode_delay 2.0 --seed 0 --max_seed_tries 50
```

---

## O.0-5 更新 (2026-06-08 #4)：collect_data 保存修复 + viewer 预热

### gen2-10 问题分析

gen2-10 运行的 collect_data 是在代码更新前启动的，缺少 `save_all_episodes`。
需 Ctrl-C 终止后重跑。

### 为什么 check_success 返回 False

当前 IK task 的 play_once 只执行 pregrasp → grasp → gripper_close，
不包含 lift → place 步骤，也不包含抓取物理（夹爪关闭不会附着瓶子）。
因此瓶子始终保持原位附近，`check_success` 检查瓶子是否在目标位置会返回 False。

日志示例：
```
[piper-ik][check] success=False b1=(-0.289,0.086,0.865) t1=(-0.280,-0.150) dist1=0.236
```
瓶子 z 从 0.740 升到 0.865（被推高 12cm），但未到达目标 y=-0.15。

### 如何保存失败 episode 用于回放

1. config YAML 已添加 `save_all_episodes: true`
2. 重跑 collect_data：
```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0
```
3. hdf5 文件会保存到 `data/pick_diverse_bottles_piper_ik/demo_piper_ik_seq_v1/data/`
4. 轨迹 pickle 文件在 `_traj_data/` 目录

### viewer 预热优化

预渲染帧数从 30 增加到 90（约 1.5 秒），第 1 个 episode 加载最慢（Vulkan 初始化），
后续 episode 加载会快很多。

### 所有 V1 命令汇总

```bash
# 场景 viewer（只看坐标轴）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v1 --seed 0 --max_seed_tries 50

# 运动 viewer（单次）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50

# 运动 viewer（连续 10 episode）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 10 --episode_delay 2.0 --seed 0 --max_seed_tries 50

# 无窗口 smoke 验证
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && unset DISPLAY && SAPIEN_RT_DENOISER=none timeout 180s python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 10 --hold 0 --render_freq 0 --show_axes 0 --require_success 1

# 数据采集（含失败保存）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0
```

---

## O.0-5 更新 (2026-06-08 #5)：collect_data 全流程跑通 ✓

### 完整数据采集验证通过

```text
Phase 1: seed 5 → IK 求解 → play_once → save_traj_data (pickle) → check_success (save_all=True → 强制保存)
Phase 2: 回放 pickle → 保存帧 → merge_pkl_to_hdf5_video → episode0.hdf5 + episode0.mp4

输出：
  data/pick_diverse_bottles_piper_ik/demo_piper_ik_seq_v1/data/episode0.hdf5 (268KB)
  data/pick_diverse_bottles_piper_ik/demo_piper_ik_seq_v1/video/episode0.mp4  (15KB)
```

### 核心修复汇总

1. `folder_path` 初始化：setup_demo 中确保 folder_path 存在
2. Phase 1/2 兼容：play_once 在 Phase 1 将规划结果存入 left_joint_path，Phase 2 从中回放
3. 帧保存：Phase 2 回放时调用 _take_picture() → hdf5 merge
4. save_all_episodes：check_success 强制返回 True，失败 episode 也保存
5. skip_planner：所有 config YAML 添加此标志，避免 CuroboPlanner GPU 卡死
6. URDF 修正：IK solver 使用 piper_pika_agx.urdf（匹配 SAPIEN 仿真）
7. viewer 预热：90 帧预渲染确保运动过程可见
8. 右手范围修正：xlim [0.38, 0.52]，避免跨身体够取

---

## O.0-5 更新 (2026-06-08 #6)：viewer 智能等待 + 视频标注 + 多视角

### viewer 智能等待

不再使用固定帧数预热，改为轮询 `viewer.window` 直到窗口真正创建：
- 最多等待 600 帧（12 秒）
- 检测到窗口后再渲染 5 帧确保内容可见
- 日志：`viewer window ready after N frames`

### 视频文件名加 success/fail 标注

`merge_pkl_to_hdf5_video()` 重写：合并完成后将文件重命名：
- `episode0.mp4` → `episode0_fail.mp4` 或 `episode0_succ.mp4`
- `episode0.hdf5` → `episode0_fail.hdf5` 或 `episode0_succ.hdf5`

### 多视角采集

所有 config YAML 启用 `third_view: true` + `observer: true`：
- head_camera: 正常 D435 视角（320×240, 30fps）
- observer: 第三人称全局视角（位置由 SAPIEN 自动管理）
- 两路视频帧均存入 hdf5

### head camera 位置确认

当前 head camera 使用 Piper 标定 config 的位置：
```yaml
# piper_pika_agx/config.yml → static_camera_list
position: [-0.032, -0.45, 1.35]   # 桌面上方，y=-0.45, z=1.35
forward: [0, 0.6, -0.8]            # 向前下方俯瞰
left: [-1, 0, 0]
```
与 ALOHA config 位置相同（y=-0.45, z=1.35），居中俯瞰桌面。

### GPU 占用说明

collect_data 需要 GPU（Curobo IK 求解）。当前 GPU 0/1 均为 100% 占用（~77GB/98GB），
collect_data 会因 GPU 内存不足而卡死。等待 GPU 空闲后重跑即可。

### V1 命令汇总（最终版）

```bash
# 场景 viewer
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v1 --seed 0 --max_seed_tries 50

# 运动 viewer（单次）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50

# 运动 viewer（连续 10 episode）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 10 --episode_delay 2.0 --seed 0 --max_seed_tries 50

# 数据采集（episode_num=5，含 save_all + succ/fail 标注 + 多视角）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0
```

---

## O.0-5 更新 (2026-06-08 #7)：step_mode 交互确认 + third_view mp4

### 交互确认模式

新增 `--step_mode 1` 参数。每个动作（pregrasp → grasp → gripper）执行完毕后，
终端输出 `[step-mode] Step move_0 done. Press Enter to continue...`，
等待用户按回车才继续下一步。

```bash
# V1 逐步确认模式
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --step_mode 1 --seed 0 --max_seed_tries 50

# V1 逐步确认 + 连续 3 episode
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 3 --step_mode 1 --seed 0 --max_seed_tries 50
```

终端输出示例：
```
[piper-ik] Step 1.0: move / move
... (pregrasp 运动完成)
[piper-ik][step-mode] Step move_0 done. Press Enter to continue...
[piper-ik] Step 1.1: move / move
... (grasp 运动完成)
[piper-ik][step-mode] Step move_1 done. Press Enter to continue...
[piper-ik] Step 1.2: gripper / gripper
... (夹爪关闭完成)
[piper-ik][step-mode] Step gripper_2 done. Press Enter to continue...
```

### head camera 确认

Piper 和 ALOHA 使用完全相同的 head camera 位置：
```yaml
# piper_pika_agx/config.yml → static_camera_list
position: [-0.032, -0.45, 1.35]   # 桌面居中上方
forward: [0, 0.6, -0.8]            # 向前下方俯瞰
```
两个 embodiment config 的 head_camera 参数完全一致。

### third_view mp4 生成

`merge_pkl_to_hdf5_video()` 重写，同时生成两路 mp4：
- `episode0_fail_head.mp4`  — head_camera 视角（原有）
- `episode0_fail_third.mp4` — third_view / observer 全局视角（新增）
- `episode0_fail.hdf5`      — 完整数据

### V1 全部命令汇总

```bash
# 场景 viewer（只看坐标轴）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_ik --task_config demo_piper_ik_seq_v1 --seed 0 --max_seed_tries 50

# 运动 viewer（单次）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50

# 运动 viewer（逐步确认，每步按回车）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --step_mode 1 --seed 0 --max_seed_tries 50

# 运动 viewer（连续 10 episode）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 10 --episode_delay 2.0 --seed 0 --max_seed_tries 50

# 数据采集（多视角 + succ/fail 标注）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0
```

---

## O.0-5 更新 (2026-06-08 #8)：step_mode 非阻塞等待 + 新增 Piper 视角 camera

### step_mode 非阻塞修复

原 `input()` 阻塞主线程 → SAPIEN viewer 无法渲染 → 窗口在确认后才出现。

修复为 `select.select` 非阻塞轮询：每 50ms 检查 stdin 同时持续调 `_update_render()`，
viewer 在等待确认期间保持渲染。交互终端中正常使用：

```bash
# V1 逐步确认（viewer 持续渲染）
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --step_mode 1 --seed 0 --max_seed_tries 50
```

终端的 3 次回车对应 pregrasp → grasp → gripper 各一次确认，viewer 每个阶段保持可见。

### 新增 Piper workspace 视角 camera

`piper_pika_agx/config.yml` 新增两个 camera：

| camera | 位置 | 朝向 | 用途 |
|---|---|---|---|
| `head_camera` | (-0.032, -0.45, 1.35) | 向前下方 | 俯视全局 (原有) |
| `front_camera` | (0.12, 0.55, 1.05) | 向后方桌面 | 正面看 Piper 操作区 |
| `side_camera` | (-0.8, 0.0, 1.2) | 向右看 | 侧面看左臂操作 |

数据采集自动包含所有 camera 的 rgb 帧，`merge_pkl_to_hdf5_video` 为每个 camera 生成独立 mp4：
```
episode0_fail_head_camera.mp4
episode0_fail_front_camera.mp4
episode0_fail_side_camera.mp4
episode0_fail_third_view.mp4       (observer 全局视角)
episode0_fail.hdf5
```

### 数据采集命令（多视角）

```bash
# V1（8 路视频：head+front+side+third，各带 succ/fail 标注）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0
```

---

## O.0-5 更新 (2026-06-08 #9)：step_mode 渲染修复 + third_camera 对称视角

### step_mode viewer 不渲染 → 已修复

**根因**：`_update_render()` 只更新 scene 内部状态，不调用 `viewer.render()` 绘制到屏幕。
step_mode 等待循环中缺了 `viewer.render()`，导致窗口标签存在但内容为空白。

**修复**：等待循环中同时调用 `_update_render()` + `viewer.render()`，每 50ms 一帧。

### Camera 最终配置（4 路固定视角 + 1 路 observer）

`piper_pika_agx/config.yml` static_camera_list：

| camera | 位置 | 朝向 | 说明 |
|---|---|---|---|
| `head_camera` | (-0.032, -0.45, 1.35) | 向前下方 | 俯视桌面 (原有) |
| `front_camera` | (0.12, 0.55, 1.05) | 向后方桌面 | 正面看操作区 |
| `side_camera` | (-0.8, 0.0, 1.2) | 向右看 | 左侧视角 |
| `third_camera` | (-0.032, 0.45, 1.35) | 向后方桌面 | 与 head 关于桌面(y=0)对称 |
| `third_view` (observer) | SAPIEN 自由视角 | - | 全局第三人称 |

### 数据采集输出

```bash
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0

# 每个 episode 输出：
video/episode0_fail_head_camera.mp4
video/episode0_fail_front_camera.mp4
video/episode0_fail_side_camera.mp4
video/episode0_fail_third_camera.mp4
video/episode0_fail_third_view.mp4     (observer)
data/episode0_fail.hdf5
```

### step_mode 交互确认命令

```bash
# V1 逐步确认（每步按回车，viewer 持续可见）
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --step_mode 1 --seed 0 --max_seed_tries 50
```

---

## O.0-5 更新 (2026-06-08 #10)：前进轴方向修正

### pregrasp/grasp 方向修复

原逻辑错误：pregrasp 在瓶子上方（top-down），而非原始 ALOHA 的"沿前进轴后方→前方"。

**修复**：`_cartesian_grasp_actor` 改为与原始 `get_grasp_pose` 一致的逻辑：

1. 从 bottle contact matrix 计算抓取坐标系
2. approach_dir = 抓取坐标系局部 +X（前进方向）
3. 根据 approach_dir 指向 base 还是背离 base 确定偏移方向
4. pregrasp = contact + approach_dir × (±0.10) ← 机器人与瓶子之间
5. grasp = pregrasp ± approach_dir × 0.08 ← 向瓶子前进

**验证**（seed=3）：
```
左臂: base y=-0.25  bottle y=0.141
      pregrasp y=0.048 → grasp y=0.122 (向瓶子前进 +Y) ✓
右臂: base y=-0.27  bottle y=0.194
      pregrasp y=0.100 → grasp y=0.175 (向瓶子前进 +Y) ✓
FK 精度: 0.000m（pregrasp/grasp 完全准确）
```

### 6 步完整流程

```
Step 0 (pregrasp):      EE → 瓶子后方 10cm（沿前进轴，机器人与瓶子之间）
Step 1 (grasp):         EE → 向前 8cm 到瓶子
Step 2 (close_gripper): 夹爪关闭
Step 3 (lift):          EE 抬升 25cm
Step 4 (place):         EE → 放置目标
Step 5 (open_gripper):  夹爪打开
```

### 运行命令

```bash
# 完整 6 步运动 viewer
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50

# 逐步确认模式
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --step_mode 1 --seed 0

# 连续 10 episode
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --num_episodes 10 --episode_delay 2.0 --seed 0
```

---

## O.0-5 数据采集状态 (2026-06-09)

### 各版本采集结果

| 版本 | 状态 | episode 数 | 视频 | hdf5 |
|---|---|---|---|---|
| V1 | ✓ 完成 | 5 (seed 3,4,7,10,11) | head/front/side ×5 | ✓ |
| V2 | ✓ 完成 | 5 | head/front/side ×5 | ✓ |
| V3 | △ 搜索中 | 0 | - | - |
| V4 | ✓ 完成 | 5 | head/front/side ×5 | ✓ |

### 已知问题

1. **third_camera mp4 缺失**：已修复，`merge_pkl_to_hdf5_video` 添加 `third_camera` 到生成列表
2. **pre_grasp_dis 不一致**：Phase 1 用 0.08、Phase 2 用 0.12，已统一为 0.12
3. **指令 JSON 缺失**：`pick_diverse_bottles_piper_ik.json` 不存在，不影响数据采集（仅影响 instruction 生成）
4. **V3 不稳定**：种子成功率低（大量 "Objects is unstable"），MotionGen 偶发失败

### 采集的视频 vs viewer 差异

采集视频是 Phase 2 回放（need_plan=False），从 pickle 读取 Phase 1 记录的关节轨迹回放。
viewer 是 Phase 1 直接执行。两者运动应一致（轨迹来源相同）。
如有明显差异，检查 pre_grasp_dis 是否一致（现已统一为 0.12）。

### 数据采集命令（episode_num=5）

```bash
# V1
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0

# V2
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v2 0

# V3 (不稳定)
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v3 0

# V4
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v4 0
```

### 输出文件

```
data/pick_diverse_bottles_piper_ik/demo_piper_ik_seq_v1/
├── video/
│   ├── episode0_fail_head_camera.mp4
│   ├── episode0_fail_front_camera.mp4
│   ├── episode0_fail_side_camera.mp4
│   ├── episode0_fail_third_camera.mp4    (现已修复)
│   └── ...
├── data/
│   ├── episode0_fail.hdf5
│   └── ...
└── seed.txt
```

## O.0-5 更新（2026-06-10）：连续 IK、轨迹 v2、双新增视角与可用命令

### 当前逻辑

1. 动作顺序固定为 `pregrasp -> grasp -> close_gripper -> lift -> place -> open_gripper`。
2. `lift` 从 `grasp` 目标构造：保持抓取点 x/y 和姿态，仅将 z 增加 `lift_height`（默认 0.12 m）。
3. 四段 move 逐段规划和执行；下一段的 `last_qpos` 是上一段 IK 轨迹的末端关节状态，不再从 home 重算。
4. `close_gripper` 后测量瓶子功能点与实际末端的 x/y 偏移，并据此修正 place 的夹爪目标。任务目标是瓶子位置，不是夹爪位置。
5. 每段运动末尾持续命令最终 IK 关节状态 `move_settle_steps`，避免接触后 PD 尚未收敛就进入 lift/place。
6. pickle 使用 `piper_ik_cartesian` schema、版本 2、IK 版本、动作名、目标和非空有限值校验。旧 pickle 会被明确拒绝，因此必须使用新的 `demo_piper_ik_seq_v*` 输出目录。
7. `third_camera` 保留为右侧 side 视角；新增 `opposite_top_camera`，从机器人头部对向俯视桌面。采集会为所有 RGB camera 自动生成 MP4。

### Viewer（V1-V4）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50 --require_success 1
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v2 --seed 0 --max_seed_tries 50 --require_success 1
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v3 --seed 0 --max_seed_tries 50 --require_success 1
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v4 --seed 0 --max_seed_tries 50 --require_success 1
```

无显示器 smoke 验证示例：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
unset DISPLAY && SAPIEN_RT_DENOISER=none timeout 180s python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50 --hold 0 --render_freq 0 --show_axes 0 --require_success 1
```

### 数据采集（V1-V4）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v2 1
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v3 2
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v4 3
```

以上四个版本已用真实 `check_success()` 验证。V3 首选 MotionGen；若 MotionGen 初始化、优化或返回轨迹失败，会回退到同一有效 IK 终点的三次插值轨迹。`pick_diverse_bottles_piper_ik.json` 已临时从 `pick_diverse_bottles.json` 复制，instruction 生成可以正常运行。

---

## O.1 对比实验：Foundation 位姿 + 原始 OBJ + Piper IK Cartesian 抓取 [2026-06-11 修正并实测]

### 当前结论

O.1 不再使用 O.0 的随机位置和 `001_bottle` 替代模型，而是读取：

`foundation_input_<ID>/multi_object_world_poses.npz`

并使用其中的左右物体位置和原始 OBJ：

- 左侧：`cola/cola.obj`
- 右侧：`bottle/bottle.obj`
- 默认使用直立朝向；`foundation_use_orientation: true` 才使用 FoundationPose 四元数。
- 视觉网格是原始 OBJ；默认 `support_proxy` 只在 OBJ 底部放置薄圆柱支撑碰撞体，避免夹爪在 pregrasp/grasp 时先把细瓶推倒。
- 抓取基准点是 OBJ bounds 的局部几何中心变换到世界坐标，不是 actor 原点。
- V1-V4 沿用 O.0 的顺序轨迹逻辑：每段从上一段末端关节状态继续，lift 保持 grasp 的 x/y 和姿态，只增加 z。

### 原错误与根因

1. **Viewer 读取错配置**：旧 viewer 固定加载 `demo_piper_ik_seq_v*`，因此 O.1 缺少 `foundation_input_dir`。现在可自动推断，也推荐显式传 `--task_config`。
2. **只记录 OBJ 路径但仍加载 `001_bottle`**：旧实现并没有真正使用 Foundation OBJ。现在 visual actor 直接来自 NPZ 指定的 OBJ。
3. **把 Foundation pose 的 p 当作瓶身中心**：这些 OBJ 的原点接近底部。input 0 的 actor z 约为 0.733/0.745 m，但实际几何中心约为 0.864/0.841 m；旧抓取 z 因而错误。
4. **完整瓶身碰撞会在抓取前推倒物体**：cola/bottle 直径约 0.066 m，当前 Pika 张开夹爪和完整 `cylinder_proxy`/`exact_convex` 在接近路径上会提前碰撞。默认改为底部 `support_proxy`：保留桌面支撑，不让瓶身阻挡接近。
5. **旧 close 的“成功”包含无条件瞬移**：旧实现在闭合前执行 `actor.set_pose(settled_pose)`，因此即使 pregrasp/grasp 已把瓶子撞倒，close 也会把它塞回夹爪。现在已删除这一 pose reset；先真正执行 close，再检查物体相对稳定位姿的位移/旋转、与 link6 的距离、是否位于两指之间以及到两指连线的径向距离。任一条件失败都拒绝建立 drive，整个 episode 失败；通过后也在物体**当前 pose**建立 drive，不会瞬移。
6. **V3 采集卡在相机帧**：Phase 2 只回放已保存关节轨迹，却重复初始化两套 MotionGen，占用 GPU 并拖慢多相机渲染。现在 `need_plan=false` 时跳过 planner 初始化。
7. **语言描述路径错误**：旧代码生成不存在的 `001_bottle/basefoundation_*.json`。O.1 现在写入直接语义文本，并补充了任务 prompt。
8. **批量命令污染配置和输出**：旧命令用 `sed -i` 修改同一个 YAML，使多个 Foundation ID 共用 config 名、seed 和轨迹目录。现在使用 `collect_foundation_piper_ik.sh` 为版本、ID、frame 生成独立配置和输出目录。

### 场景和轨迹逻辑

```text
NPZ pose + OBJ path
  -> trimesh 解析 bounds
  -> 原始 OBJ visual
  -> 默认底部 support_proxy collision
  -> 按旋转后最低点补 table clearance
  -> OBJ 几何中心作为 grasp target
  -> pregrasp
  -> grasp
  -> close
  -> 无瞬移 grasp-state 门控
  -> 通过后才在当前物体 pose 建立 grasp-assist
  -> lift（grasp x/y 不变，只增加 z）
  -> place（使用闭合后实测物体/末端偏移修正）
  -> open + 释放 grasp-assist
```

viewer 执行 Phase 1 的实时规划。采集先执行同一 Phase 1 并保存带 schema/version/action names/Foundation source 的 pickle，再在 Phase 2 重新建立相同场景、严格校验输入目录、frame、mesh 几何、碰撞模式后逐点回放。因此 viewer 与采集的动作目标和关节轨迹逻辑一致；采集额外保存 RGB、HDF5、视频和语言指令。

### 关键配置

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `foundation_input_dir` | `foundation_input_0` | NPZ 目录 |
| `foundation_mode` | `o1` | `o1` / `o1.1` / `o1.2` |
| `foundation_frame` | `0` | 仅 O.1 直接使用；O.1.1/O.1.2 由标注第一关键帧覆盖 |
| `foundation_annotation_json` | `hand_keyframes_all.json` | O.1.1/O.1.2 每 episode 的两个全局关键帧 |
| `foundation_hand_targets_root` | `human_replay/h2_pure_d435/pick_diverse_bottles` | O.1.2 第二关键帧 EE 位置来源；见 O.1.2.7 修复 |
| `foundation_use_orientation` | `false` | 默认保持直立；true 使用检测朝向 |
| `foundation_table_clearance` | `0.002` | 网格最低点到桌面的间隙 |
| `foundation_collision_mode` | `support_proxy` | 默认仅底部支撑；可选 `cylinder_proxy` / `exact_convex` 做严格碰撞试验 |
| `foundation_grasp_standoff` | `0.105` | gripper base / EE 目标相对 OBJ 中心的后退距离；比旧值 `0.085` 多 2cm，更偏指尖抓取 |
| `foundation_pregrasp_distance` | `0.12` | pregrasp 在 grasp 后方的距离；碰撞调试先调该值 |
| `foundation_grasp_assist` | `true` | 状态门控通过后的抓取约束，不再修改物体 pose |
| `foundation_grasp_max_displacement` | `0.025` | close 后相对稳定中心最大位移 |
| `foundation_grasp_max_rotation_deg` | `15` | close 后相对稳定姿态最大转动 |
| `foundation_grasp_require_contact` | `false` | `support_proxy` 默认用几何夹持门控；完整碰撞试验可设 true 要求两指接触 |
| `ik_version` | `v1-v4` | V1 线性、V2 三次、V3 MotionGen+回退、V4 多种子 |

### 环境

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
cd /home/zaijia001/ssd/RoboTwin
```

### O.1 / O.1.1 / O.1.2 设定

- **O.1**：按 `foundation_frame` 读取物体位置；动作为 `pregrasp -> grasp -> close -> lift -> place -> open`。
- **O.1.1**：从 `hand_keyframes_all.json` 读取 `hand_vis_<ID>.mp4` 的前两个标注关键帧，用第一关键帧的 Foundation OBJ pose 建场；后续动作与 O.1 相同。
- **O.1.2**：用第一关键帧建场并执行 `pregrasp -> grasp -> close`；然后从 `id_<ID>/world_targets_and_status.npz` 取第二关键帧左右 EE 的 xyz，以单个 `action` 取代 lift/place。`action` 的四元数继续使用 grasp 姿态，不读第二关键帧的朝向，且不自动 open。

input 0 当前标注为第一关键帧 `38`、第二关键帧 `78`。标注少于两帧、被标记 reject/bad、关键帧在 Foundation NPZ 越界，或第二帧在 `selected_indices` 中不是唯一匹配时，代码会直接报错。

### V1-V4 Viewer

有窗口：

```bash
python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --task_config demo_piper_ik_foundation_v1 \
  --ik_version v1 --foundation_id 0 --foundation_frame 0 --foundation_mode o1 \
  --seed 0 --max_seed_tries 1 --require_success 1
```

O.1.1/O.1.2 只需切换 mode：

```bash
python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --task_config demo_piper_ik_foundation_v1 \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.1 \
  --seed 0 --max_seed_tries 1 --require_success 1

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --task_config demo_piper_ik_foundation_v1 \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --seed 0 --max_seed_tries 1 --require_success 1 --wrist_preview 1
```

viewer 会创建 SAPIEN 窗口；`--wrist_preview 1` 另外创建一个 OpenCV 左右腕 RGB 拼接窗口。两者都需要可用的 X11/Vulkan display；远程桌面通常显式加 `DISPLAY=:1.0`。`unset DISPLAY` 只适用于下面的离屏采集。`--foundation_id` / `--foundation_frame` / `--foundation_mode` 只覆盖本次 viewer 的内存配置，不修改 YAML。

### V1-V4 数据采集

推荐使用 `collect_foundation_piper_ik.sh` 批量入口：

```text
bash collect_foundation_piper_ik.sh <v1|v2|v3|v4> <foundation_id> [foundation_frame] [gpu_id] [o1|o1.1|o1.2] [run_tag]
```

参数说明：
| 位置 | 参数 | 说明 |
|---|---|---|
| 1 | `v1` / `v2` / `v3` / `v4` | IK 版本：V1 线性、V2 三次、V3 MotionGen、V4 多种子 |
| 2 | foundation_id | `foundation_input_<ID>` 的 ID（0-101 可选，需 NPZ 和标注存在） |
| 3 | foundation_frame | NPZ 帧号（O.1 直接使用；O.1.1/O.1.2 由标注关键帧覆盖，可填 0） |
| 4 | gpu_id | GPU 编号（0-3） |
| 5 | `o1` / `o1.1` / `o1.2` | 模式：O.1=直接帧 / O.1.1=标注第一关键帧 / O.1.2=标注关键帧+人手 action |
| 6 | run_tag | 可选输出标签，只允许字母、数字、`_`、`-`；O.1.2.1 wrist 建议使用 `wrist_o121_verified_0615` |

脚本自动生成独立 config 和输出目录，不会互相覆盖，并强制 `episode_num: 1`，避免 V1 基础配置误让每个 ID 连续采 10 个 episode。

**输出目录规则**：
- O.1: `demo_piper_ik_foundation_v<N>_id<ID>_frame<FRAME>`
- O.1.1 / O.1.2: `demo_piper_ik_foundation_v<N>_o1_<1|2>_id<ID>`
- 有 run tag：在上述名字末尾追加 `_<run_tag>`，例如 `demo_piper_ik_foundation_v1_o1_2_id0_wrist0515`

```text
data/pick_diverse_bottles_piper_ik_foundation/<config_name>/
  ├── data/episode0_succ.hdf5
  ├── video/episode0_succ_left_camera.mp4        # 左腕 D435（0515 标定）
  ├── video/episode0_succ_right_camera.mp4       # 右腕 D435（0515 标定）
  ├── video/episode0_succ_head_camera.mp4        # 头部 D435
  ├── video/episode0_succ_front_camera.mp4        # 前向固定视角
  ├── video/episode0_succ_side_camera.mp4         # 侧向固定视角
  ├── video/episode0_succ_third_camera.mp4        # 第三视角
  ├── video/episode0_succ_opposite_top_camera.mp4 # 对向俯视
  ├── video/episode0_succ_third_view.mp4          # observer 视角
  ├── _traj_data/episode0.pkl
  └── instructions/episode0.json
```

**关于 wrist 相机**：四份 Foundation 配置现在使用 `collect_wrist_camera: true`。左右相机分别读取 `calibration_bundle_piper_new_table_0515.json` 中互不相同的 `left_gripper_T_camera` / `right_gripper_T_camera`，再通过 `wrist_camera_pose_reference: urdf_end_link` 跟随仿真 `link6`。`wrist_camera_simulation_adapter: piper_pika_agx` 处理基础 TCP/CAD 平移，`wrist_camera_tuning` 再做仿真专用的逐侧前移和画面 roll 校正；0515 原始标定 JSON 不被改写。当前值为左 `forward_offset_m=0.125, image_roll_deg=-15`，右 `0.11, -60`。`legacy_r1` 把 OpenCV optical frame 转成 SAPIEN render frame。

0515 检查结论：没有证据表明 0515 求解突然失效；左右平移分别约 `-14.93 cm` / `-13.50 cm`，两侧 roll 差异也真实写在外参中。问题在于实机 TCP、官方 Piper/Pika 固定连接、AGX 转换后的 DAE 坐标和 RoboTwin `link6` 不是同一个父帧。训练数据还要求左右画面统一朝上，因此可以保留实机外参，同时在仿真末端增加明确、可覆盖的 image-roll tuning。当前 wrist 渲染仍使用 `D435` 的 `320x240, fovy=37°`；仓库没有 D405 内参项，D405/D435 差异会改变 FOV，但不会造成 60 度 roll 或相机落进外壳。Phase 2 会自动输出两路 MP4 并写入 HDF5 observations。

旧的无 tag 目录若已有 `episode0_succ.hdf5` 会被断点逻辑跳过，不会自动补生成 wrist 视频。启用 wrist 后应使用新的 run tag 重新采集，并保留旧数据不动。

---

#### O.1.2 全量批量采集（V1-V4 × 0-120）

以下四段分别完整粘贴到四个 tmux pane；每段都自行设置 `RUN_TAG`，不要只复制 `FAIL_LOG/for` 部分：

```bash
# V1 / GPU 0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
RUN_TAG=wrist_o121_verified_0615; mkdir -p data/tmp
FAIL_LOG="$PWD/data/tmp/o12_v1_${RUN_TAG}_failures.log"; : > "$FAIL_LOG"
for id in $(seq 0 120); do timeout 600s bash collect_foundation_piper_ik.sh v1 "$id" 0 0 o1.2 "$RUN_TAG" || echo "FAIL v1 id=$id status=$?" | tee -a "$FAIL_LOG"; done
```

```bash
# V2 / GPU 1
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
RUN_TAG=wrist_o121_verified_0615; mkdir -p data/tmp
FAIL_LOG="$PWD/data/tmp/o12_v2_${RUN_TAG}_failures.log"; : > "$FAIL_LOG"
for id in $(seq 0 120); do timeout 600s bash collect_foundation_piper_ik.sh v2 "$id" 0 1 o1.2 "$RUN_TAG" || echo "FAIL v2 id=$id status=$?" | tee -a "$FAIL_LOG"; done
```

```bash
# V3 / GPU 2
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
RUN_TAG=wrist_o121_verified_0615; mkdir -p data/tmp
FAIL_LOG="$PWD/data/tmp/o12_v3_${RUN_TAG}_failures.log"; : > "$FAIL_LOG"
for id in $(seq 0 120); do timeout 600s bash collect_foundation_piper_ik.sh v3 "$id" 0 2 o1.2 "$RUN_TAG" || echo "FAIL v3 id=$id status=$?" | tee -a "$FAIL_LOG"; done
```

```bash
# V4 / GPU 3
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
RUN_TAG=wrist_o121_verified_0615; mkdir -p data/tmp
FAIL_LOG="$PWD/data/tmp/o12_v4_${RUN_TAG}_failures.log"; : > "$FAIL_LOG"
for id in $(seq 0 120); do timeout 600s bash collect_foundation_piper_ik.sh v4 "$id" 0 3 o1.2 "$RUN_TAG" || echo "FAIL v4 id=$id status=$?" | tee -a "$FAIL_LOG"; done
```

4 个版本分别在 tmux 窗口并行运行。当前实际 Foundation 输入是 ID 0-101；102-120 会因缺少 NPZ 快速返回非零并写入 failure log。每个 ID 最多尝试 `max_seed_tries: 3` 个 seed，确定性失败不会再无限循环；外层 `timeout 600s` 处理 GPU 拥塞或底层渲染无响应。每个成功 episode 会生成 Phase 1 轨迹、Phase 2 replay、8 路视频和 HDF5。

#### 按 Foundation ID 汇总为 episode ID

每个 `foundation_input_<ID>` 使用独立输出目录，目录内部始终是 `episode0_*`。下面的索引脚本把源 ID 映射成目标 `episode<ID>_*`；默认创建相对软链接，不复制大视频，也不修改源数据：

```bash
python script/index_foundation_piper_ik_videos.py \
  --version v4 --mode o1.2 --run-tag wrist_o121_verified_0615 \
  --output-video-dir \
  /home/zaijia001/ssd/RoboTwin/data/pick_diverse_bottles_piper_ik/demo_piper_ik_foundation_v4_o1_2_wrist_o121_verified_0615/video \
  --method symlink --dry-run

# dry-run 确认后，去掉 --dry-run 正式建立 episode<ID> 链接和 manifest
```

如果要把当前已有的无 tag V4/O.1.2 子目录写入 `demo_piper_ik_v4_3/video`，省略 `--run-tag` 并先 dry-run。实测当前可索引 ID 0-8；ID 9 是失败目录，没有成功视频。目标目录已有 `episode0` 至 `episode4`，默认会报 conflict 并停止：

```bash
python script/index_foundation_piper_ik_videos.py \
  --version v4 --mode o1.2 \
  --output-video-dir \
  /home/zaijia001/ssd/RoboTwin/data/pick_diverse_bottles_piper_ik/demo_piper_ik_v4_3/video \
  --method symlink --dry-run
```

确认必须替换旧 episode 时，去掉 `--dry-run` 并加 `--replace-episode`。该参数会删除目标目录中同 ID 的旧 `episode<ID>_*.mp4`，但不会删除 Foundation 源目录。默认推荐写到新的聚合目录。

2026-06-11 复查 tmux `gen2-10`、`genikv2-11`、`genikv3-12`、`genikv4-13`：pane 都已回到 shell，不是仍在采集。历史输出有 `Killed` 和 Ctrl-C；同时 V1 原配置为每 ID 10 episodes，失败 seed 又无限重试，因此视觉上很像卡住。V4 ID 9 在多个 seed 中右侧 grasp 旋转约 `25.6°`，稳定超过 `15°` 门限；这是同一关键帧/几何条件的确定性失败，单纯继续换 seed 无法解决。

**进度监控** — 查看已完成的 succ/fail 数量：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

for ver in v1 v2 v3 v4; do
  succ=$(find data/pick_diverse_bottles_piper_ik_foundation -name "episode*_succ.hdf5" -path "*${ver}_o1_2_id*" | wc -l)
  fail=$(find data/pick_diverse_bottles_piper_ik_foundation -name "episode*_fail.hdf5" -path "*${ver}_o1_2_id*" | wc -l)
  echo "${ver}: ${succ} succ, ${fail} fail"
done
```

**旧版兼容**（不推荐 — 固定 ID=0，不通过脚本入口）：

```bash
bash collect_data.sh pick_diverse_bottles_piper_ik_foundation demo_piper_ik_foundation_v1 0
```

### input 0 实测结果（2026-06-11）

- O.1、O.1.1、O.1.2 的 V1 viewer 均通过真实 `check_success()`。
- O.1/O.1.1/O.1.2 的 V1 均完成 Phase 1 保存、Phase 2 validated replay、HDF5、MP4 和 instruction 生成。
- O.1.2 另外在 V2/V3/V4 完成了同样的完整采集链路；四版本的 action EE 误差约为 1.8-4.3cm，两个物体均移动超过 10cm。
- V3 的 MotionGen 在该场景返回 optimization failure，按设计回退到同一 IK 终点的插值轨迹；这不是采集失败。
- 稳定后的 OBJ 中心：左约 `(-0.038, 0.096, 0.864)`，右约 `(0.230, 0.114, 0.841)`。
- 采集包含原 side 视角以及新增对向俯视桌面的 `opposite_top_camera`。
- 使用 `urdf_end_link + piper_pika_agx + wrist_camera_tuning` 后，V4/O.1.2 ID 0 完整两阶段采集成功：左右腕视频均为 38 帧、320x240；抽帧确认外壳不再遮挡，右腕画面按训练视角扶正。
- prompt：`description/task_instruction/pick_diverse_bottles_piper_ik_foundation.json`。
- 轨迹会拒绝旧格式、空路径以及 mode/ID/keyframes/action source/pregrasp/抓取门控/mesh/collision 不匹配的 pickle。
- 默认 `support_proxy + require_contact=false` 是“防提前碰撞 + 几何夹持状态门控 + drive”，不是纯接触物理。如需严格接触试验，改为 `cylinder_proxy` 或 `exact_convex`，并设 `foundation_grasp_require_contact: true`；当前 input 0 预期会暴露 pregrasp/grasp 碰撞问题，而不是被瞬移掩盖。

### 代码位置

- `envs/pick_diverse_bottles_piper_ik_foundation.py`
- `envs/pick_diverse_bottles_piper_ik.py`
- `view_pick_diverse_bottles_piper_ik_motion.py`
- `collect_foundation_piper_ik.sh`
- `script/index_foundation_piper_ik_videos.py`
- `code_painting/build_piper_calibration_bundle.py`
- `task_config/demo_piper_ik_foundation_v1.yml` 至 `v4.yml`

## O.1.2.1 新增：Pika wrist 相机帧、外壳遮挡与逐侧微调（2026-06-15）

### tmux 结论

- `gen1-9`、`gen2-10`、`genikv2-11`、`genikv3-12`、`genikv4-13` 当前都已回到 shell，检查时没有任务卡住。
- `gen2/genik*` 曾只粘贴 `FAIL_LOG=...${RUN_TAG}...` 和循环，没有先执行 `RUN_TAG=...`，因此出现 `o12_v4__failures.log`，随后写入无 tag 的 `demo_piper_ik_foundation_v4_o1_2_id<ID>`。
- 无 tag 目录曾被 `planner_gripper`、`urdf_end_link` 和不同 adapter 配置重复使用，不能据此比较新旧 wrist。以后必须使用上面的自包含命令和新 tag。

### 坐标与 URDF 结论

1. 官方 `pika_gripper_description.urdf` 只有 `gripper_base_link`、左右夹指和两个 prismatic joint，不包含相机 link、相机支架或 D405/D435 模块。
2. 官方 Piper+Pika URDF 的 `joint6_to_gripper_base` 带 `rpy="0 -1.57 0"`；AGX `pika2_gripper.urdf` 把类似轴变换烘焙到 DAE/关节布局，而当前合并 URDF 使用 identity `link6 -> gripper_base_link`。因此“模型外观看起来正确”不能证明 hand-eye 父帧正确。
3. 上一次反向来自把 0515 optical 外参接到 `planner_gripper`：该帧相对 raw `link6` 带 `diag(1,-1,-1)`，即绕 X 轴 180 度，左右/上下轴随之翻转。当前使用 raw `urdf_end_link`，不再叠加该旋转。
4. 相机靠后不是单纯的真实连接件嵌套。0515 外参本身给出左/右 TCP 到相机 X 平移约 `-0.149/-0.135 m`，而仿真缺少真实 TCP、相机支架和镜头中心 link；直接组合会把虚拟镜头留在 Pika 外壳内或后方。
5. 左右 roll 差异不是 IK 朝向规划造成的。0515 外参已包含不同的绕光轴安装角；真实安装可以如此，但训练视频需要统一画面朝上，因此仿真允许最后一层 image-roll 校正。
6. `agx_arm_sim` 有独立 RealSense D405 xacro，也有 D435 示例，但它们没有被原始 Pika gripper URDF 自动挂载。当前 RoboTwin wrist 使用 `D435` 渲染参数；要严格复现实机 D405，还缺实测内参、镜头 optical center 到 Pika/TCP 的 CAD 变换和左右安装照片/尺寸。

官方参考：

- https://github.com/agilexrobotics/pika_ros/tree/master/src/pika_gripper_description
- https://github.com/agilexrobotics/agx_arm_urdf/tree/main/piper/urdf
- https://github.com/agilexrobotics/agx_arm_sim/blob/master/agx_arm_description/urdf/pika_gripper_description.urdf
- https://github.com/agilexrobotics/agx_arm_sim/blob/master/realsense2_description/urdf/_d405.urdf.xacro

### 当前实现与 viewer

基础配置保持 0515 JSON 不变，额外使用：

```yaml
wrist_camera_tuning:
  left: {forward_offset_m: 0.125, image_roll_deg: -15.0}
  right: {forward_offset_m: 0.11, image_roll_deg: -60.0}
```

`forward_offset_m` 沿每台 SAPIEN 相机自身 `+X` 光轴移动，避免左右外参不同却硬套父帧 XYZ；`image_roll_deg` 只绕光轴调整画面，不改变抓取器或 IK。左侧需要比右侧多前移约 1.5 cm，与两份 0515 X 平移之差一致。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

# V1；V2-V4 只改 --ik_version
python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --wrist_preview 1 --hold 1

# 临时试调，不修改 YAML。正数/负数以终端打印的 tuning 为准。
python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.125 --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 --wrist_right_roll_deg -60
```

### 验证

- V4 / ID 0 / O.1.2 使用 tag `wrist_o121_verified_0615_smoke` 完成 Phase 1、validated Phase 2、HDF5、instructions 和 8 路 MP4。
- 左右 wrist 都是 38 帧、320x240；抽帧确认 Pika 外壳退出画面，右侧瓶身从约 60 度斜置恢复接近竖直。
- `DISPLAY=:1.0` 的 V4 viewer 使用最终默认 tuning 和 `--wrist_preview 1` 完整运行，`physical_success=True`。
- 该修正只改变观测相机 pose，不改变 pregrasp、grasp、close、action 轨迹或成功判定。

### O.1.2.1 补充：结论边界、参数方向与 viewer 录制

这里不是“只能微调、无法得出结论”。目前结论分为两类：

- **已确定的根因**：矩阵拼接公式没有问题，但旧代码把不等价的父坐标系当成了同一个 frame。完整链条应为 `world_T_link6 @ link6_T_real_tcp @ real_tcp_T_camera @ optical_T_render`。0515 提供 `real_tcp_T_camera`；当前 URDF 没有真实 TCP、相机支架和镜头 optical-center link，因此 `link6_T_real_tcp` 不能省略。
- **尚未由 CAD/测量确定的量**：`link6_T_real_tcp` 的精确 6D 外参。当前逐侧 tuning 是对这段缺失外参的经验补偿，已验证能消除外壳遮挡和画面 roll，但不能称为新的物理手眼标定。
- **得到物理结论的方法**：确认实机 `/urdf_end_pose_orient` 对应 link6、Pika base 还是 fingertip TCP；测量左右镜头 optical center 到该 frame 的 XYZ/RPY，或把真实支架和 camera link 加入 URDF。完成后 tuning 应回到零，或只保留可选的训练画面 roll normalization。

| 参数 | 调什么 | 正值效果 | 负值效果 |
|---|---|---|---|
| `--wrist_left_forward_offset_m` | 左相机沿自身光轴的位置，单位米 | 沿正在看的方向前移，物体变大；越过外壳前缘后遮挡减少 | 后退，视野更宽，但更容易落回外壳 |
| `--wrist_right_forward_offset_m` | 右相机沿自身光轴的位置 | 同上，仅右侧 | 同上，仅右侧 |
| `--wrist_left_roll_deg` | 左画面绕光轴旋转 | 当前 render frame 中顺时针 | 当前 render frame 中逆时针 |
| `--wrist_right_roll_deg` | 右画面绕光轴旋转 | 当前 render frame 中顺时针 | 当前 render frame 中逆时针 |
| `--wrist_debug_record 1` | 同步保存 viewer wrist 帧 | 启用三路 MP4 和 JSON | `0` 不录制 |
| `--wrist_debug_tag TAG` | 本次对比实验名 | 建独立目录；拒绝覆盖非空目录 | 不适用 |
| `--wrist_debug_fps 30` | MP4 回放帧率 | 只改变回放速度 | 不允许小于等于零 |

`forward_offset_m` 不是父坐标系 X/Y/Z，而是每台相机自己的 optical forward。因此左右姿态即使不同，正值都表示沿当前视线向前。roll 正负号是当前 SAPIEN render frame 的实测图像结果。

录制命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --wrist_preview 1 --hold 0 \
  --wrist_left_forward_offset_m 0.125 --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 --wrist_right_roll_deg -60 \
  --wrist_debug_record 1 \
  --wrist_debug_tag left125_right110_roll_m15_m60
```

输出到 `data/wrist_camera_debug/<TAG>/`：

- `wrist_debug_left.mp4`
- `wrist_debug_right.mp4`
- `wrist_debug_mosaic.mp4`
- `wrist_debug_config.json`

JSON 记录 camera type、reference、adapter、axis mode、左右 tuning、task/config、IK 版本、Foundation ID/mode 和 seed。录制不依赖桌面窗口，可以不加 `--wrist_preview 1`。当前每次命令只允许录一个 episode，避免多个 episode 覆盖同一 tag。

验证：V1/O.1.2 ID 0 viewer `physical_success=True`；最终保留的无窗口左、右和拼接三路 MP4 都是 511 帧、30 FPS，分辨率分别为 320x240、320x240、640x240，JSON 与命令参数一致。

### O.1.2.1 补充：VS Code 兼容视频与无 viewer 正式采集

旧 debug recorder 使用 OpenCV `mp4v`（MPEG-4 Part 2）。文件本身完整，但 VS Code/Chromium 经常不提供该 codec，所以会表现为“MP4 无法打开”。现在 recorder 已改为与正式数据兼容的 `H.264/avc1 + yuv420p`，并增加 `faststart`；`data/wrist_camera_debug` 下已有旧视频也已原地转码，文件名不变。

#### 1. 无窗口、只录 wrist debug 视频

这条命令不打开 SAPIEN viewer，不生成 HDF5，适合快速比较参数。必须为每组参数换一个 tag：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
unset DISPLAY

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --render_freq 0 --show_axes 0 --hold 0 --wrist_preview 0 \
  --wrist_left_forward_offset_m 0.125 \
  --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_debug_record 1 \
  --wrist_debug_tag left125_right110_roll_m15_m60_headless
```

输出为 `data/wrist_camera_debug/<TAG>/` 下的左右、拼接 H.264 MP4 和 JSON。

#### 2. 无 viewer、使用指定 wrist 参数跑原正式采集链路

这条命令使用原来的 `collect_foundation_piper_ik.sh`，完整运行 Phase 1、validated Phase 2，并生成 HDF5、instruction 和 8 路 H.264 视频。四个环境变量必须全部提供；run tag 必须随参数变化，禁止复用旧目录：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
unset DISPLAY

WRIST_LEFT_FORWARD_OFFSET_M=0.125 \
WRIST_RIGHT_FORWARD_OFFSET_M=0.11 \
WRIST_LEFT_ROLL_DEG=-15 \
WRIST_RIGHT_ROLL_DEG=-60 \
timeout 600s bash collect_foundation_piper_ik.sh \
  v1 0 0 0 o1.2 wrist_left125_right110_roll_m15_m60
```

参数顺序仍是：`版本 Foundation_ID frame GPU mode run_tag`。V2/V3/V4 分别改为 `v2 ... GPU=1`、`v3 ... GPU=2`、`v4 ... GPU=3`。基础 YAML 的 `render_freq: 0` 和采集器的 Phase 2 都不会打开 viewer。

正式输出示例：

```text
data/pick_diverse_bottles_piper_ik_foundation/
  demo_piper_ik_foundation_v1_o1_2_id0_wrist_left125_right110_roll_m15_m60/
    video/episode0_succ_left_camera.mp4
    video/episode0_succ_right_camera.mp4
    data/episode0_succ.hdf5
    instructions/episode0.json
```

实测 `wrist_collect_override_h264_0615`：生成配置包含左 `0.125/-15`、右 `0.11/-60`；完整两阶段采集成功，左右 wrist 均为 38 帧 `H.264/avc1/yuv420p`。

若需要手动修复其他历史 `mp4v` 文件：

```bash
ffmpeg -y -i OLD.mp4 -an \
  -c:v libx264 -crf 23 -pix_fmt yuv420p -movflags +faststart NEW.mp4
```

#### 3. 有 viewer：显示 wrist/head 相机框线并同步预览（可选录像）

`gen1` 当前可用图形显示为 `:1.0`。如果前面运行过 `unset DISPLAY`，必须使用 `export DISPLAY=:1.0` 恢复；`set DISPLAY` 不会设置或导出环境变量。先用 `xdpyinfo` 检查连接，再运行 viewer：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.125 \
  --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_debug_record 0 \
  --max_seed_tries 1 --require_success 1
```

- `--show_camera_frustums 1`：在 SAPIEN 主 viewer 中打开橙色相机视锥框线。日志会校验并列出 `left_camera`、`right_camera`、`head_camera`；左右 wrist 框线随夹爪运动，head 框线固定。SAPIEN 的这个开关会同时显示场景内其他相机框线。
- `--wrist_preview 1`：额外打开 OpenCV 左右 wrist RGB 拼接窗口。这与主 viewer 中的三维框线不是同一功能。
- `--hold 1`：动作完成后保持最终 viewer，关闭 SAPIEN 窗口或在终端按 `Ctrl-C` 才退出。需要自动结束时改为 `--hold 0 --episode_delay 2`。
- 默认 `--wrist_debug_record 0` 不录像。需要录像时改为 `--wrist_debug_record 1 --wrist_debug_tag "o121_v1_viewer_$(date +%Y%m%d_%H%M%S)"`；时间戳可避免覆盖已有非空目录。

V2/V3/V4 只改 `--ik_version v2/v3/v4`；该 viewer 命令本身不使用批采集 GPU 位置参数。

`gen1` 历史结果并非全部失败：

1. `left125_right110_roll_m15_m60_headless` 和 `left125_right110_roll_m15_m60` 两次在建场景前失败，原因是 debug tag 目录已存在且非空。
2. 改为 `left125_right110_roll_m15_m60_2` 后 viewer 完整执行，`physical_success=True`，保存 560 帧 wrist debug 视频。
3. 后续执行 `unset DISPLAY` 后，又用 `set DISPLAY` 尝试恢复，这是无效命令；因此两次 O.1 viewer 都报 `Create window failed: Renderer does not support display`。恢复方法是上面的 `export DISPLAY=:1.0`。

2026-06-15 实测上述框线功能：V1/O.1.2 ID 0 的 viewer 在 `DISPLAY=:1.0` 成功打开，日志确认左右 wrist 和 head 三个 camera frustum 均存在，最终 `physical_success=True`。

### O.1.2.1 补充：真正实时的机器人运动 Viewer（2026-06-16）

此前不是 SAPIEN 加载失败。Piper IK 自定义轨迹执行循环每一步只调用 `_update_render()`：这会更新 wrist camera 图像和 OpenCV 拼接窗口，但没有调用 `viewer.render()`。所以 wrist 画面实时变化，而 SAPIEN 主窗口只在动作结束后的 hold 阶段才绘制最终状态。现在 move、末端 settle 和 gripper settle 三类循环都已恢复实时 `viewer.render()`，并打印 `live SAPIEN motion frames=<N>`；若启用 viewer 却没有任何运动帧，命令会直接报错。

#### 1. 只看实时机器人运动（推荐先用这条确认）

该模式只有 SAPIEN 主窗口，不打开 wrist RGB 拼接窗口。主窗口实时显示机器人运动，并显示左右 wrist、head 以及场景其他相机的橙色视锥框线：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 0 --hold 1 \
  --wrist_left_forward_offset_m 0.125 \
  --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 \
  --wrist_right_yaw_deg 0.840 \
  --max_seed_tries 1 --require_success 1
```

#### 2. 同时看实时机器人运动和左右 wrist RGB

这条只比模式 1 多打开 `--wrist_preview 1`。运动期间同时存在：

- `SAPIEN`：实时机器人、物体、坐标轴和相机视锥。
- `RoboTwin wrist cameras`：实时左右 wrist RGB 拼接。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.125 \
  --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 \
  --wrist_right_yaw_deg 0.840 \
  --max_seed_tries 1 --require_success 1
```

这两条都不录像，因此不需要 `TAG`、`--wrist_debug_record` 或 `--wrist_debug_tag`。`--wrist_left_yaw_deg 0.182` 和 `--wrist_right_yaw_deg 0.840` 是把相机 forward 的开合轴 `Y` 分量压到 0 的小 yaw；如果只想使用旧默认相机朝向，可以删掉这两行。若需要动作结束后自动退出，把 `--hold 1` 改为 `--hold 0 --episode_delay 0`。

验证结果：模式 1 在运动执行期间检测到 1920x1080 的 `SAPIEN` 窗口；模式 2 同时检测到 `SAPIEN` 和 640x299 的 `RoboTwin wrist cameras`。两种模式均实时绘制 510 个 SAPIEN 运动帧，V1/O.1.2 ID 0 均为 `physical_success=True`。

### O.1.2.1 补充：用相机前向轴校准 wrist 外参的方法（2026-06-16）

可以用“相机前向轴是否落在夹爪开合方向的垂直平面内”来检查 wrist 外参，但必须先区分两个坐标约定：

- SAPIEN camera 的前向轴是 render local `+X`，代码中 `forward_offset_m` 也是沿这个轴移动。
- Pika CAD / `envs/robot/robot.py` 的 TCP 物理前向是 gripper local `+X`：`get_left/right_tcp_pose()` 明确从 wrist/endlink 沿 local `+X` 推 `gripper_bias=0.12`。
- 旧 debug 轴图例把蓝色 local `+Z` 标成“夹爪前进方向”，这是 IK/debug 目标姿态约定，不等价于 Pika CAD 的物理手指长度方向。
- Pika 手指开合方向是 gripper local `Y`；URDF 中两指关节和 mesh bounds 都支持这个判断。

复算命令会在末尾输出可直接加入 viewer 的 yaw 参数，例如当前输出为 `--wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840`。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
python script/diagnose_piper_wrist_camera_axes.py
```

当前 0515 bundle + `piper_pika_agx` adapter + `legacy_r1` 的结果：

| side | camera forward in gripper frame | 对开合轴 Y 的平面误差 | 到 Pika 物理 +X | 到旧 debug +Z | 若只消除 Y 分量，绕 gripper +Z |
|---|---:|---:|---:|---:|---:|
| left | `[0.999974, -0.003184, 0.006511]` | `-0.182 deg` | `0.415 deg` | `89.627 deg` | `+0.182 deg` |
| right | `[0.999622, -0.014664, 0.023248]` | `-0.840 deg` | `1.575 deg` | `88.668 deg` | `+0.840 deg` |

结论：

- 如果以 Pika CAD / Robot TCP 的物理 `+X` 为夹爪前向，当前 wrist camera 前向轴已经基本正确：左约 `0.4 deg`，右约 `1.6 deg`；它也基本落在“垂直于开合方向 Y”的平面内。此时没有证据支持再做 90 度大旋转。
- 如果把旧 debug 蓝色 `+Z` 当成物理前向，会计算出约 `-89 deg` 绕 gripper `Y` 的大旋转需求；这更像坐标约定混淆，而不是 wrist 外参错误。不要直接按这个量改相机。
- `image_roll_deg` 只绕相机自己的前向轴旋转画面，不能改变相机前向轴；它适合调画面水平/倾斜，不适合把 `+X` 前向改成 `+Z` 前向。
- 真正可作为外参微调的是“开合平面误差”：若希望相机 forward 的 `Y` 分量严格为 0，可对左/右分别增加一个很小的绕 gripper `+Z` yaw（当前估计左 `+0.18 deg`、右 `+0.84 deg`），现在可用 `--wrist_left_yaw_deg` / `--wrist_right_yaw_deg` 应用。这个量远小于目前画面 roll 调参，不应作为主要视觉偏差来源。

建议校准流程：

1. 先用 `script/diagnose_piper_wrist_camera_axes.py` 检查 forward 与开合轴 Y 的点积，确认相机是否在 `X-Z` 平面内。
2. 若平面误差超过约 `2-3 deg`，再考虑引入“绕 gripper +Z 的 per-side yaw 微调”。目前不建议改默认值。
3. 画面水平、左右旋转感继续用 `--wrist_left_roll_deg` / `--wrist_right_roll_deg` 调，因为这是绕 optical forward 的图像 roll。
4. 若你坚持用 debug `+Z` 作为物理前向，应先统一 Robot TCP、IK target、URDF/Pika CAD 的前向定义，再改 camera adapter；否则会把原本物理上接近正确的 wrist 相机旋转错。

### O.1.2.1 补充：wrist 到 tip 距离、俯视角和右手偏心检查（2026-06-16）

当前命令里的距离参数有两个口径，别混用：

- `gripper_bias=0.12`：Robot/Pika TCP 约定，表示 nominal tip/TCP 在 `link6/gripper_base` local `+X` 方向约 `12cm`。这个不是 wrist camera 调参。
- `--wrist_left_forward_offset_m 0.125`、`--wrist_right_forward_offset_m 0.11`：从 0515 标定 + `piper_pika_agx` adapter 得到的相机位姿，再沿每台相机自己的 SAPIEN render `+X` 光轴前移。你说“wrist 到 tip 距离增加 2cm”如果指这个调参，应改成左 `0.145`、右 `0.13`。

按当前 viewer 命令（含 yaw 左 `0.182`、右 `0.840`）计算，假设 nominal tip 在 gripper frame `[0.12, 0, 0]`：

| side | forward offset | camera center in gripper frame | camera 到 tip 欧氏距离 | 沿 camera forward 到 tip | camera Y 偏心 |
|---|---:|---:|---:|---:|---:|
| left | `0.125m` | `[0.0507, 0.0207, 0.0944]` | `0.119m` | `0.0687m` | `+2.07cm` |
| right | `0.110m` | `[0.0500, -0.0274, 0.0920]` | `0.119m` | `0.0679m` | `-2.74cm` |

如果把 `forward_offset_m` 各加 `2cm`，相机沿当前视线更靠近 tip：沿 forward 到 tip 约变成左 `4.87cm`、右 `4.79cm`。所以它会让画面更“往前/更近”，但不会让相机低头看夹爪。

可直接试的 `+2cm` 实时 viewer 命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py   --task_name pick_diverse_bottles_piper_ik_foundation   --ik_version v1 --foundation_id 0 --foundation_mode o1.2   --render_freq 1 --show_axes 1 --show_camera_frustums 1   --wrist_preview 1 --hold 1   --wrist_left_forward_offset_m 0.145   --wrist_right_forward_offset_m 0.13   --wrist_left_roll_deg -15   --wrist_right_roll_deg -60   --wrist_left_yaw_deg 0.182   --wrist_right_yaw_deg 0.840   --max_seed_tries 1 --require_success 1
```

关于“相机是否俯视夹爪”：当前 0515 原始标定和现在的 yaw 后结果都不是明显俯视，而是几乎沿 Pika 物理 `+X` 手指长度方向平视/平行：

| side | 原始 forward 到物理 `+X` | yaw 后 forward 到物理 `+X` | 当前 forward 指向 nominal tip 的夹角 | 若要直接看 tip，约需绕 gripper `Y` pitch |
|---|---:|---:|---:|---:|
| left | `0.415deg` | `0.373deg` | `54.7deg` | `~54.1deg` |
| right | `1.575deg` | `1.332deg` | `55.2deg` | `~54.0deg` |

这解释了你看到的现象：当前相机前向轴确实和夹爪前向/开合垂直平面基本共面，但它更像与手指长度方向近似平行，不是“略微俯看两个夹爪”。因此：

- `forward_offset_m +2cm` 只能改变相机沿视线的位置，不能解决“看不到夹爪”的俯视角问题。
- `parent_yaw_deg` 只把相机 forward 的 `Y` 分量压到 0，解决“是否偏出开合垂直平面”的小误差；它也不能让相机低头。
- 如果要让 wrist 画面明显看到夹爪，需要新增/使用绕 gripper local `Y` 的 parent pitch 调参，而不是继续堆 roll/yaw。按 nominal tip 精确瞄准会需要约 `54deg`，这太大；实际建议先小步试 `10deg/15deg/20deg` 这种“略微俯视”量级。
- 右手“偏右”的感觉有几何来源：右相机中心 `Y=-2.74cm`，左相机 `Y=+2.07cm`，右侧偏心更大约 `0.67cm`；如果要严格居中，需要 lateral `Y` offset 或重新建相机安装外参，不是 yaw/roll 能完全解决。



### O.1.2.2 更正：gripper 抓取规划距离 +2cm（2026-06-16）

你刚才说的“增加距离”不是 wrist camera 到 tip 的视觉外参，而是 Foundation O.1/O.1.2 里 gripper 规划目标相对物体中心的后退距离。这里对应参数是：

- `foundation_grasp_standoff`：grasp 阶段 gripper base / EE 目标相对瓶子中心沿 approach 方向的后退距离。
- 旧默认 `0.085m` 会让瓶子看起来更靠近夹爪根部。
- 新默认已改成 `0.105m`，也就是 gripper 规划目标离瓶子中心再远 `2cm`，让瓶子更靠近夹爪指尖/剪刀口区域。
- `foundation_pregrasp_distance` 仍是从 grasp 目标继续后退到 pregrasp 的距离，默认 `0.12m`；它控制靠近路径的起点，不是最终夹住深度。
- `--wrist_*_forward_offset_m` 只调相机位置，不改变 gripper 抓取深度。

#### 实时 viewer 验证指令

默认 YAML 已经是 `0.105m`；下面显式写出参数，方便 debug 时确认终端会打印 `grasp_standoff=0.105m`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --foundation_grasp_standoff_m 0.105 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.125 \
  --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 \
  --wrist_right_yaw_deg 0.840 \
  --max_seed_tries 1 --require_success 1
```

#### 采集指令临时覆盖

默认配置已改，不设置环境变量也会用 `0.105m`。如果要临时比较 `0.085/0.105/0.115`，用 `FOUNDATION_GRASP_STANDOFF_M`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

FOUNDATION_GRASP_STANDOFF_M=0.105 \
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.2 standoff105
```

判断方法：看 SAPIEN 里 `plan_grasp_*` 坐标轴和瓶子的相对位置。`foundation_grasp_standoff` 增大时，gripper base 会离瓶子更远；如果 gripper 前向/TCP 方向是对的，瓶子会从根部区域移动到更靠近指尖闭合区域。若仍然根部碰撞，下一步应继续调 standoff 或检查 gripper/TCP 前向，而不是调 wrist camera。


### O.1.2.3 补充：0515 原始 wrist 标定朝向、俯视角和右手 Y 偏心（2026-06-16）

这次只讨论 wrist camera 外参，不再混入 gripper 抓取深度。前一次 `foundation_grasp_standoff=0.105` 已经解决“夹爪根部抓取”问题；下面是相机是否俯视夹爪、是否共面、右手是否偏右。

#### 1. 原始标定是否已经处理“共面/俯视”

0515 bundle 中 wrist 的记录是：

- `parent_frame=urdf_end_pose_orient_tcp`
- `camera_frame=opencv_color_optical`
- `axis_conversion=render_camera = raw_optical @ legacy_r1.T`
- `piper_pika_agx` adapter 只有平移：`+0.075m X`、`+0.050m Z`，不改变朝向。

也就是说：原始标定 + 现有 adapter/axis conversion **有处理相机前向轴和 gripper 前向轴的坐标约定**，但它处理出来的是“接近 gripper `+X` 平视”，不是“明显俯视夹爪”。

角度表如下。`plane_err_y` 表示 camera forward 是否偏出“夹爪前向 +X 与上下 +Z 组成的平面”；越接近 0 越共面。`pitch_xz` 是 forward 在 X-Z 平面内相对 gripper `+X` 的俯仰角；当前只有约 `0-1deg`，所以不是俯视。

| side | 阶段 | forward in gripper frame | plane_err_y | 到 gripper +X | pitch_xz | yaw_xy | 到 nominal tip 视线夹角 |
|---|---|---:|---:|---:|---:|---:|---:|
| left | 0515 raw + legacy_r1，未加 adapter | `[0.999974,-0.003184,0.006511]` | `-0.182deg` | `0.415deg` | `+0.373deg` | `-0.182deg` | `10.427deg` |
| left | adapter 后，当前基础外参 | `[0.999974,-0.003184,0.006511]` | `-0.182deg` | `0.415deg` | `+0.373deg` | `-0.182deg` | `26.588deg` |
| left | 当前 viewer yaw/offset/roll 后 | `[0.999979,-0.000007,0.006511]` | `~0deg` | `0.373deg` | `+0.373deg` | `~0deg` | `54.722deg` |
| right | 0515 raw + legacy_r1，未加 adapter | `[0.999622,-0.014664,0.023248]` | `-0.840deg` | `1.575deg` | `+1.332deg` | `-0.840deg` | `12.233deg` |
| right | adapter 后，当前基础外参 | `[0.999622,-0.014664,0.023248]` | `-0.840deg` | `1.575deg` | `+1.332deg` | `-0.840deg` | `28.975deg` |
| right | 当前 viewer yaw/offset/roll 后 | `[0.999730,-0.000007,0.023248]` | `~0deg` | `1.332deg` | `+1.332deg` | `~0deg` | `55.154deg` |

结论：

- 原始标定确实和 gripper 前向轴基本共面，右手偏出也只有 `0.84deg`，不是“大角度不共面”。
- 原始标定没有提供明显俯视夹爪的角度；forward 基本平行于 gripper `+X`。
- 当前 `--wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840` 只是把 `Y` 分量归零，使它更共面；不会让相机低头。
- 当前 `--wrist_left_roll_deg -15 --wrist_right_roll_deg -60` 只旋转图像画面，不改变相机 forward。

#### 2. 原始/当前位置和右手偏心

按 nominal tip `[0.12,0,0]` 估算：

| side | 阶段 | camera center in gripper frame | camera Y | camera Z | 到 tip 欧氏距离 | 沿 forward 到 tip | 到 tip 横向误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| left | 0515 raw | `[-0.1493,0.0207,0.0436]` | `+2.07cm` | `4.36cm` | `27.36cm` | `26.91cm` | `4.95cm` |
| left | adapter 后 | `[-0.0743,0.0207,0.0936]` | `+2.07cm` | `9.36cm` | `21.67cm` | `19.38cm` | `9.70cm` |
| left | 当前 viewer | `[0.0507,0.0207,0.0944]` | `+2.07cm` | `9.44cm` | `11.90cm` | `6.87cm` | `9.71cm` |
| right | 0515 raw | `[-0.1350,-0.0274,0.0394]` | `-2.74cm` | `3.94cm` | `25.95cm` | `25.36cm` | `5.50cm` |
| right | adapter 后 | `[-0.0600,-0.0274,0.0894]` | `-2.74cm` | `8.94cm` | `20.29cm` | `17.75cm` | `9.83cm` |
| right | 当前 viewer | `[0.0500,-0.0274,0.0920]` | `-2.74cm` | `9.20cm` | `11.88cm` | `6.79cm` | `9.75cm` |

右手“偏右”的感觉有几何来源：右相机 `Y=-2.74cm`，左相机 `Y=+2.07cm`。如果只要求左右镜像对称，右手需要沿 gripper 父坐标系 `+Y` 平移约 `+0.0067m`，让它从 `-2.74cm` 变成 `-2.07cm`。如果要求右手直接回到中心线 `Y=0`，则需要 `+0.0274m`，这个量偏大，建议先不要一步到位。

#### 3. 后续角度/偏移怎么改

现在 viewer 已支持两个新调参：

- `--wrist_left_pitch_deg` / `--wrist_right_pitch_deg`：绕 gripper/link6 父坐标系 `+Y` 调俯仰。正值会让 camera forward 从近似 `+X` 向 nominal tip 方向下俯。
- `--wrist_left_lateral_offset_m` / `--wrist_right_lateral_offset_m`：沿 gripper/link6 父坐标系 `+Y` 平移相机中心。右手当前 `Y` 为负，想往中心/左侧修就用正值。

如果严格让相机 forward 看向 nominal tip，按当前外参大约需要：left `+54.1deg`、right `+54.0deg`。这个角度太大，不建议直接作为默认；它说明当前标定不是“俯看夹爪”的相机模型。实际调试建议先试小角度：`10deg`、`15deg`、`20deg`。

保守试调命令：给左右各加 `15deg` 下俯，同时把右手 Y 偏心先修到和左手镜像对称（`+0.0067m`）：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --foundation_grasp_standoff_m 0.105 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.125 \
  --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 \
  --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg 15 \
  --wrist_right_pitch_deg 15 \
  --wrist_right_lateral_offset_m 0.0067 \
  --max_seed_tries 1 --require_success 1
```

若画面里夹爪仍然太少：把 pitch 从 `15` 加到 `20`。若右手仍偏右：把 `--wrist_right_lateral_offset_m` 从 `0.0067` 小步加到 `0.010`、`0.015`。若右手过度偏左，则减小该值。不要用 `roll` 修 lateral 偏心；roll 只改变画面旋转。


### O.1.2.4 更正：wrist camera 的 gripper 中线修正与坐标轴定义（2026-06-16）

这节更正 O.1.2.3 里的一个容易混淆点：`+0.0067m` 是“右手调到和左手镜像对称”的保守修正，不是“把相机放到 gripper 中点”。如果目标是相机在 gripper 开合中线附近，应按 `Y=0` 修正。

#### 坐标轴定义

在当前 `piper_pika_agx` / 0515 wrist 标定诊断里采用的 gripper frame 约定是：

- `gripper +X`：物理前进轴，约等于 wrist 到 tip / 指尖方向。
- `gripper +Y`：夹爪开合方向，也是 wrist camera 的左右偏心方向。
- `gripper +Z`：与 `X-Y` 垂直的方向；旧 viewer/debug 里蓝轴曾被标成“夹爪前进方向”，这是旧 debug 标签约定，不应再拿来解释 wrist camera 物理前向。

因此：

- 调“相机是否靠 gripper 中点”，看 camera center 的 `Y`。
- 调“相机是否俯视夹爪”，看 camera forward 相对 `+X` 的 pitch。
- `+X` 不是开合方向；开合方向是 `+Y`。

#### 原始标定位置是否在 gripper 中线

不在。按 0515 bundle 原始标定位置：

| side | 0515 raw camera center | Y 偏心 | 若要到 gripper 中线 Y=0 |
|---|---:|---:|---:|
| left | `[-0.1493, +0.0207, 0.0436]` | `+2.07cm` | `--wrist_left_lateral_offset_m -0.0207` |
| right | `[-0.1350, -0.0274, 0.0394]` | `-2.74cm` | `--wrist_right_lateral_offset_m +0.0274` |

`piper_pika_agx` adapter 只加 `X=+0.075m`、`Z=+0.050m`，不改 `Y`；`forward_offset_m` 沿相机光轴前移，在当前 yaw 后也几乎不改 `Y`。所以当前 viewer 下仍然是：left `Y=+2.07cm`、right `Y=-2.74cm`。

这说明：原始标定的相机光轴方向基本和 gripper 前进轴共面，但相机中心并没有落在 gripper 开合中线 `Y=0`。如果真实硬件相机确实安装在 gripper 中点附近，则需要 lateral offset 或重新修 wrist 外参位置。

#### 两种不同修正目标

| 目标 | left lateral | right lateral | 含义 |
|---|---:|---:|---|
| 回到当前标定/不修 Y | 不传 | 不传 | 使用 0515 标定 + adapter + 当前 forward/yaw/roll，保留 left `+2.07cm`、right `-2.74cm` |
| 只让左右镜像更对称 | 不传 | `+0.0067` | right 从 `-2.74cm` 调到约 `-2.07cm`，和 left `+2.07cm` 镜像 |
| 相机放到 gripper 中线 | `-0.0207` | `+0.0274` | left/right 都调到 `Y≈0`，更符合“中点附近 camera”的假设 |

#### 中线 + 俯视试调 viewer 命令

如果你的目标是“相机在 gripper 中点附近，并且能略微看到两个夹爪”，建议先用下面这条，而不是只调右手 `+0.0067`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --foundation_grasp_standoff_m 0.105 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.125 \
  --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 \
  --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg 15 \
  --wrist_right_pitch_deg 15 \
  --wrist_left_lateral_offset_m -0.0207 \
  --wrist_right_lateral_offset_m 0.0274 \
  --max_seed_tries 1 --require_success 1
```


#### 已验证推荐版本：`o1.2_verified_grasp_wrist_v2`

这组是当前你实测效果好的 viewer 参数。它同时包含抓取深度和 wrist 相机外参修正：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `--foundation_grasp_standoff_m` | `0.14` | gripper base / EE grasp 目标离物体中心更远，物体更靠近夹爪指尖/剪刀口，减少根部抓取感 |
| `--wrist_left_forward_offset_m` | `0.145` | 左 wrist 相机沿自身光轴前移；比 `0.125` 多 `2cm` |
| `--wrist_right_forward_offset_m` | `0.13` | 右 wrist 相机沿自身光轴前移；比 `0.11` 多 `2cm` |
| `--wrist_left_pitch_deg` / `--wrist_right_pitch_deg` | `15` | 绕 gripper/link6 父坐标系 `+Y` 下俯，让 wrist 画面能看到更多夹爪 |
| `--wrist_left_lateral_offset_m` | `-0.0207` | 左相机移到 gripper 开合中线 `Y≈0` |
| `--wrist_right_lateral_offset_m` | `0.0274` | 右相机移到 gripper 开合中线 `Y≈0` |
| `--wrist_left/right_yaw_deg` | `0.182 / 0.840` | 消除相机 forward 的开合轴 `Y` 分量，使光轴回到 gripper `X-Z` 平面 |
| `--wrist_left/right_roll_deg` | `-15 / -60` | 只修正画面水平/旋转，不改变 forward |
| `--foundation_capture_radial_tolerance_m` | `0.08` | 适配 `standoff=0.14` 后的几何门控；默认 `0.065` 会偏严格 |
| `--foundation_grasp_assist_max_distance_m` | `0.16` | 适配 `standoff=0.14` 后的 EE-物体中心距离；默认 `0.14` 会贴边 |

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --foundation_grasp_standoff_m 0.14 \
  --foundation_capture_radial_tolerance_m 0.08 \
  --foundation_grasp_assist_max_distance_m 0.16 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.145 \
  --wrist_right_forward_offset_m 0.13 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 \
  --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg 15 \
  --wrist_right_pitch_deg 15 \
  --wrist_left_lateral_offset_m -0.0207 \
  --wrist_right_lateral_offset_m 0.0274 \
  --max_seed_tries 1 --require_success 1
```

如果你想看“纯标定位置”对照，不要传 `--wrist_left_lateral_offset_m` / `--wrist_right_lateral_offset_m`，也不要传 `--wrist_left_pitch_deg` / `--wrist_right_pitch_deg`。保留 yaw/roll/forward 只是当前模拟调参；完全回到 YAML 默认则连命令行的 wrist override 都可以省略。

验证：带中线修正 `left=-0.0207`、`right=+0.0274`、左右 pitch `15deg` 的 V1/O.1.2 headless 最小运行完成，日志确认 `parent_lateral_offset_m` 进入 camera tuning，且 `physical_success=True`。

> **注意**：human_replay/plan_keyframes 管线（modelL 系列实验）的 IK URDF 修复见 [L 节](#l-ik-urdf-修正--tcplink6-偏移修复human_replay-管线-2026-06-17)。


### O.1.2.5 抓取真实性与 OBJ/Gripper 碰撞 debug（2026-06-16）

当前 O.1.2 的默认成功路径不是“完全纯物理抓取”。它的逻辑是：

1. `pregrasp/grasp` 由 IK 移到目标位置。
2. `close_gripper` 后先做 grasp-state validation：检查物体位移/旋转是否没有被撞飞、EE 到物体中心距离、物体中心到两指连线的径向距离、两指接触列表。
3. validation 通过后，如果 `foundation_grasp_assist=true`，才在 **当前物体 pose** 上建立 gripper-object drive。它不会把物体瞬移到夹爪中心，但 drive 仍然是辅助约束，不是纯接触摩擦抓取。

因此要 debug “更真实的抓取”，建议分三档：

| 档位 | 目的 | 关键参数 | 预期 |
|---|---|---|---|
| A. 已验证成功档 | 保持你当前看到的好效果 | `support_proxy + grasp_assist=1 + require_contact=0 + radial=0.08 + max_dist=0.16` | 稳定生成动作和 wrist 画面 |
| B. 接触门控档 | 检查两指是否真的碰到 OBJ collision | `cylinder_proxy` 或 `exact_convex`，`grasp_require_contact=1` | 若 contacts 为空会 fail-fast，暴露假抓取 |
| C. 纯物理观察档 | 关闭 drive，看物体是否靠接触/摩擦被带走 | `grasp_assist=0`，通常 `require_success=0` | 物体可能掉落或不跟随，这是有效信息 |

#### 新增 viewer debug 参数

这些参数已经可以直接从 viewer 命令行覆盖，不需要改 YAML：

| 参数 | 含义 |
|---|---|
| `--foundation_collision_mode support_proxy|cylinder_proxy|exact_convex` | OBJ 碰撞模型。`support_proxy` 稳定但只保底部支撑；`cylinder_proxy` 更适合瓶子接触调试；`exact_convex` 更接近 mesh，但可能更慢或更不稳定 |
| `--foundation_collision_radius_padding_m` | 给 proxy collision 半径加 padding，正值更容易接触，负值更严格 |
| `--foundation_grasp_assist 0|1` | 是否在 validation 后创建 object-gripper drive；`0` 是纯物理观察档 |
| `--foundation_grasp_require_contact 0|1` | validation 是否要求两指都接触物体 collision |
| `--foundation_capture_radial_tolerance_m` | 物体中心到两指连线的最大径向距离；越小越严格 |
| `--foundation_grasp_assist_max_distance_m` | EE 到物体中心的最大距离；越小越严格 |

#### A. 当前已验证成功档（推荐日常 viewer）

这就是你刚测试效果好的版本。注意：`standoff=0.14` 会让默认几何门控略严格；headless 验证中默认 `radial_tolerance=0.065` 会因 left `radial=0.071m` 失败，所以这里显式设置 `0.08/0.16`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --foundation_grasp_standoff_m 0.14 \
  --foundation_capture_radial_tolerance_m 0.08 \
  --foundation_grasp_assist_max_distance_m 0.16 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.145 \
  --wrist_right_forward_offset_m 0.13 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 \
  --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg 15 \
  --wrist_right_pitch_deg 15 \
  --wrist_left_lateral_offset_m -0.0207 \
  --wrist_right_lateral_offset_m 0.0274 \
  --max_seed_tries 1 --require_success 1
```

#### B. 接触门控档：要求两指接触

先用 `cylinder_proxy`，因为瓶子类 OBJ 比较适合圆柱近似；`require_contact=1` 会让没有两指接触的抓取直接失败。失败时看终端里的：

- `contacts=[]`：没有真实接触，说明只是几何门控/assist 成功。
- `radial=...` 大：物体中心不在两指之间。
- `projection=...` 超出 `[0,1]` 附近：物体不在两指连线段内。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --foundation_grasp_standoff_m 0.14 \
  --foundation_collision_mode cylinder_proxy \
  --foundation_grasp_require_contact 1 \
  --foundation_capture_radial_tolerance_m 0.05 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.145 \
  --wrist_right_forward_offset_m 0.13 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 \
  --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg 15 \
  --wrist_right_pitch_deg 15 \
  --wrist_left_lateral_offset_m -0.0207 \
  --wrist_right_lateral_offset_m 0.0274 \
  --max_seed_tries 1 --require_success 0
```

#### C. 纯物理观察档：关闭 grasp-assist

这条用来回答“如果没有 drive，OBJ 是否真的被夹爪带走”。这里 `--require_success 0` 是故意的，因为纯物理下物体不跟随也不是程序失败，而是说明当前碰撞/摩擦/闭合深度还不够真实。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --foundation_grasp_standoff_m 0.14 \
  --foundation_collision_mode cylinder_proxy \
  --foundation_grasp_assist 0 \
  --foundation_grasp_require_contact 0 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.145 \
  --wrist_right_forward_offset_m 0.13 \
  --wrist_left_roll_deg -15 \
  --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 \
  --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg 15 \
  --wrist_right_pitch_deg 15 \
  --wrist_left_lateral_offset_m -0.0207 \
  --wrist_right_lateral_offset_m 0.0274 \
  --max_seed_tries 1 --require_success 0
```

如果 C 档物体不跟随，按这个顺序 debug：

1. 先看 B 档 `contacts` 是否为空；为空就不是摩擦问题，是 collision/抓取深度没真正接触。
2. 若有单指接触，调 `foundation_grasp_standoff_m` 或 `foundation_grasp_lateral_offset`，让物体中心落在两指之间。
3. 若两指有接触但 lift 掉落，再调 OBJ collision/friction/质量，或检查 Pika finger collision 是否太薄/位置不准。
4. `exact_convex` 可作为最后对照：更接近 OBJ，但可能把 mesh 分解/凸包误差和性能问题引入 debug。

#### AB/C 现象结论（2026-06-16）

你观察到的“AB 看起来没效果，C 一开始把瓶子碰倒”基本符合现在的实现逻辑：

| 档位 | 现象 | 原因 |
|---|---|---|
| A | 看起来稳定，但不是真正两指物理夹住 | `support_proxy` 只给 OBJ 底部一个很薄的支撑碰撞，不提供完整侧面碰撞；`grasp_assist=1` 在 validation 通过后用 drive 带着物体走 |
| B | 如果仍用 support proxy，效果也不明显 | `require_contact=1` 只有在 OBJ 有侧面 collision 时才有意义；support proxy 下 fingers 常常接触不到物体侧面，`contacts=[]` |
| C | `cylinder_proxy + grasp_assist=0` 容易在 pregrasp/grasp 或 close 前把瓶子碰倒 | 完整圆柱侧面 collision 暴露了真实碰撞：当前抓取路径/目标深度会先顶到物体，纯物理下没有 drive 帮忙，所以物体会倒或不跟随 |

所以当前可稳定采集的数据应使用 A 档：`support_proxy + assist + validation gate`。它适合生成干净轨迹和 calibrated head/wrist 视频；如果要追求严格真实抓取，需要另开接触调试线：用 `cylinder_proxy/exact_convex`，先让 pregrasp/grasp 不碰倒，再要求 contacts，最后再关闭 assist。

### O.1.2.6 稳定采集脚本：无 viewer，保存 head + wrist 视频（2026-06-16）

新增脚本：

```bash
/home/zaijia001/ssd/RoboTwin/collect_foundation_piper_ik_verified.sh
```

它固定使用当前验证过的无碰撞/稳定采集参数：

- `foundation_mode=o1.2`
- `foundation_collision_mode=support_proxy`
- `foundation_grasp_assist=true`
- `foundation_grasp_require_contact=false`
- `foundation_grasp_standoff=0.14`（pick_diverse_bottles）
- `foundation_capture_radial_tolerance=0.08`
- `foundation_grasp_assist_max_distance=0.16`
- `collect_head_camera=true`
- `collect_wrist_camera=true`
- wrist tuning：`left forward=0.145, roll=-15, yaw=0.182, pitch=15, lateral=-0.0207`；`right forward=0.13, roll=-60, yaw=0.840, pitch=15, lateral=0.0274`

单个 ID dry-run，只生成临时 config，不采集：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

DRY_RUN=1 bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v1 0 0 verified_v2
```

单个 ID 正式采集：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v1 0 0 verified_v2
```

V1-V4 批量采集：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

for id in $(seq 0 120); do
  bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v1 $id 0 verified_v2
done

for id in $(seq 0 120); do
  bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v2 $id 1 verified_v2
done

for id in $(seq 0 120); do
  bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v3 $id 2 verified_v2
done

for id in $(seq 0 120); do
  bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v4 $id 3 verified_v2
done
```

输出位于：

```text
/home/zaijia001/ssd/RoboTwin/data/pick_diverse_bottles_piper_ik_foundation/<generated_task_config>/
```

其中视频由原 collect_data 流程生成，包含 head cam 和左右 wrist cam；不打开 SAPIEN viewer，也不打开 wrist preview 窗口。

### O.1.2.7 修复：hand targets 源从 model L 切换到 h2_pure_d435（2026-06-17）

#### 问题诊断

O.1.2 的第二关键帧 EE xyz 来源原先默认为：

```text
code_painting/human_replay/pick_diverse_bottles/id_<ID>/world_targets_and_status.npz
```

该目录由旧 `run_piper_hamer_axes_replay_batch.sh`（model L）以 `ID_FILTER=0-10` 生成，**仅覆盖 ID 0-10**。ID 11+ 缺少此文件，导致 `_load_keyframe_action_positions()` 抛出 `FileNotFoundError`，seed search 失败，只能生成空的 `seed.txt`。

而 `h2_pure_d435` 目录下的同结构 NPZ 覆盖全部 ID（0-101+），且来自当前 HaMeR → retarget → pure replay 标准管线，EE 位置与当前标定一致。

#### 代码修改

`envs/pick_diverse_bottles_piper_ik_foundation.py` 默认值已更正：

```diff
-    FOUNDATION_DEFAULT_HAND_TARGETS_ROOT = "code_painting/human_replay/pick_diverse_bottles"
-    FOUNDATION_DEFAULT_HAND_TARGETS_PATTERN = "id_{episode}/world_targets_and_status.npz"
+    FOUNDATION_DEFAULT_HAND_TARGETS_ROOT = "code_painting/human_replay/h2_pure_d435/pick_diverse_bottles"
+    FOUNDATION_DEFAULT_HAND_TARGETS_PATTERN = "id{episode}_d435_z005/world_targets_and_status.npz"
```

这与 `pnp_tray_piper_ik_foundation.py` 已经使用 `h2_pure_d435` 的做法一致。

#### 数据状态回顾

当前 v1/v4 采集状态（model L 时期）：

| 版本 | complete (有完整数据) | traj_only (有轨迹，缺视频/HDF5) | seed_only (仅空 seed) |
|------|----------------------|-------------------------------|----------------------|
| v1 | 0,1,2,3,4,5,6,7,10 | 8 | 9, 11-101 |
| v4 | 1,2,3,4,5,6,7,8,10 | 0 | 9, 11-101 |

**traj_only 恢复**（id8 v1 / id0 v4）：同目录下已有 `_traj_data/episode0.pkl`，直接 rerun 相同命令，collect_data 会从 Data Collection 阶段补视频和 HDF5。

**seed_only 恢复**：
- id9：model L 目录下有 `id_9/world_targets_and_status.npz`，但 seed search 失败；切换源后重新 seed search + 采集
- id11+：model L 目录下完全缺文件；切换源后全新 seed search + 采集

#### 恢复命令（先只补缺失 ID）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

# === v1: 补 id8 (traj_only → 重跑补视频) + id9 (seed_only → 重新 seed search) ===
for id in 8 9; do
  bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v1 $id 0 verified_v2
done

# === v4: 补 id0 (traj_only → 重跑补视频) + id9 (seed_only → 重新 seed search) ===
for id in 0 9; do
  bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v4 $id 3 verified_v2
done
```

#### 补齐 id11+ 全部视频（v1 + v4）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

# v1: id11-101（之前只有空 seed）
for id in $(seq 11 101); do
  bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v1 $id 0 verified_v2
done

# v4: id11-101（之前只有空 seed）
for id in $(seq 11 101); do
  bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v4 $id 3 verified_v2
done
```

如果 v2/v3 也有同样问题（`foundation_hand_targets_root` 也是 model L），同样适用：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

for id in $(seq 0 101); do
  bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v2 $id 1 verified_v2
done

for id in $(seq 0 101); do
  bash collect_foundation_piper_ik_verified.sh pick_diverse_bottles v3 $id 2 verified_v2
done
```

### O.2 新增：pnp_tray Foundation + Piper IK pick-and-place（2026-06-16）

O.2 复用 O.1.2 的 Foundation/关键帧逻辑，但任务对象和动作语义不同：

| 项 | O.1.2 pick_diverse_bottles | O.2 pnp_tray |
|---|---|---|
| env | `pick_diverse_bottles_piper_ik_foundation` | `pnp_tray_piper_ik_foundation` |
| 左手对象 | `left_bottle` | `left_dark_red_cup` |
| 右手对象 | `right_bottle` | `right_bottle` |
| Foundation 输入 | `data/piper/hand/pick_diverse_bottles/foundation_replay_d435/foundation_input_<ID>` | `data/piper/hand/pnp_tray/foundation_replay_d435/foundation_input_<ID>` |
| 关键帧标注 | `code_painting/h2o_manual_review/pick_diverse_bottles/hand_keyframes_all.json` | `code_painting/h2o_manual_review/pnp_tray/hand_keyframes_all.json` |
| 第二关键帧目标 | `code_painting/human_replay/h2_pure_d435/pick_diverse_bottles/id<ID>_d435_z005/world_targets_and_status.npz` 的 EE xyz（见 O.1.2.7 修复） | Foundation NPZ 中第二关键帧 OBJ center |
| 动作 | pregrasp -> grasp -> close -> second-keyframe action | pregrasp -> grasp -> close -> object-keyframe action -> open gripper |

注意：pnp_tray 不能直接沿用 pick_diverse 的 `foundation_grasp_standoff=0.14`。ID0 验证中，`0.14` 会在 close 前把左手 cup 推偏，validation 失败；`0.105` 成功。因此 O.2 config 和采集脚本对 pnp_tray 固定使用 `0.105`。

#### O.2 修正说明（2026-06-17）

gen1 中看到 close 后夹爪继续往前很远，根因是旧 O.2 逻辑沿用了 O.1.2 的“第二关键帧 EE target”。ID0 中第二关键帧 EE target 是：

- left EE `[0.0236, 0.2666, 0.9180]`
- right EE `[0.1932, 0.2651, 0.9335]`

但 Foundation 第二关键帧物体中心是：

- left cup center `[0.0592, 0.1799, 0.8343]`
- right bottle center `[0.1846, 0.1740, 0.8780]`

所以旧逻辑会把 gripper target 推到 `Y≈0.266`，看起来比物体第二关键帧位置更靠前。现在 O.2 默认使用 `foundation_action_target_source=object_keyframe`：先取第二关键帧 OBJ center，再加上当前 grasp 时 gripper 相对物体中心的偏移，得到 action gripper target。ID0 新验证 action target 为：

- left gripper `[0.0592, 0.0749, 0.8343]`
- right gripper `[0.1846, 0.0690, 0.8780]`

这样 object center 会跟随到第二关键帧附近；headless 验证 object error 约 left `4.2cm`、right `3.3cm`，`physical_success=True`。

关于 open：gen1 里第一次 O.2 完整运行已经执行 `open_gripper`；第二次是在 `open_gripper` settle 过程中按了 `Ctrl-C`，所以看起来像没完全打开。

#### O.2 viewer 验证命令

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pnp_tray_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --foundation_grasp_standoff_m 0.105 \
  --foundation_action_target_source object_keyframe \
  --foundation_capture_radial_tolerance_m 0.08 \
  --foundation_grasp_assist_max_distance_m 0.16 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --max_seed_tries 1 --require_success 1
```

#### O.2 headless 最小验证命令

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pnp_tray_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --foundation_grasp_standoff_m 0.105 \
  --foundation_action_target_source object_keyframe \
  --foundation_capture_radial_tolerance_m 0.08 \
  --foundation_grasp_assist_max_distance_m 0.16 \
  --render_freq 0 --show_axes 0 --wrist_preview 0 --hold 0 \
  --max_seed_tries 1 --require_success 1
```

已验证结果：V1 / ID0 / O.2 headless 成功，日志中 `open_after_action=True`，并执行了 `open_gripper`；最终 `physical_success=True`。

#### O.2 可选避障试验：抬高 pregrasp waypoint

这不是完整动态物体碰撞规划；它是一个不改变默认无避障命令的 waypoint 版本：在 `pregrasp` 前插入 `approach_clearance`，位置等于 pregrasp 的 z 再抬高指定米数。用途是避免从 home 横向扫过物体高度。

ID0 验证结论：

- `--foundation_pregrasp_clearance_m 0.10`：失败，左杯旋转约 `16.3deg`，略超 `15deg` 门限。
- `--foundation_pregrasp_clearance_m 0.06`：成功，`physical_success=True`。

viewer/headless 可这样试，不覆盖默认命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pnp_tray_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --foundation_grasp_standoff_m 0.105 \
  --foundation_action_target_source object_keyframe \
  --foundation_pregrasp_clearance_m 0.06 \
  --foundation_capture_radial_tolerance_m 0.08 \
  --foundation_grasp_assist_max_distance_m 0.16 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --max_seed_tries 1 --require_success 1
```

正式采集的避障试验版本使用环境变量生成独立 tag：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

FOUNDATION_PREGRASP_CLEARANCE_M=0.06 \
bash collect_foundation_piper_ik_verified.sh pnp_tray v1 0 0 o2_pregrasp_clearance006
```

#### O.2 采集命令

单个 ID：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

bash collect_foundation_piper_ik_verified.sh pnp_tray v1 0 0 o2_verified_v1
```

V1-V4 批量：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin

for id in $(seq 0 50); do
  bash collect_foundation_piper_ik_verified.sh pnp_tray v1 $id 0 o2_verified_v1
done

for id in $(seq 0 50); do
  bash collect_foundation_piper_ik_verified.sh pnp_tray v2 $id 1 o2_verified_v1
done

for id in $(seq 0 50); do
  bash collect_foundation_piper_ik_verified.sh pnp_tray v3 $id 2 o2_verified_v1
done

for id in $(seq 0 50); do
  bash collect_foundation_piper_ik_verified.sh pnp_tray v4 $id 3 o2_verified_v1
done
```

输出位于：

```text
/home/zaijia001/ssd/RoboTwin/data/pnp_tray_piper_ik_foundation/<generated_task_config>/
```


## L. IK URDF 修正 + `--target_retreat_m` 参数（human_replay 管线，2026-06-17）

本修复来自分析 modelL1521-5 tmux 窗口的运行错误：pregrasp 阶段正常通过，但 grasp 阶段 IK plan 声称 `pos_err=0.67cm`，实际执行后 `dist=57cm`。

### 影响范围

以下管线使用 `render_hand_retarget_piper_dual_npz_urdfik.py` / `plan_anygrasp_keyframes_r1.py` 的 IK 求解器：

- **plan_keyframes_human_replay**（`run_plan_keyframes_human_replay_piper_d435.sh` → `plan_keyframes_human_replay.py` → `plan_anygrasp_keyframes_piper.py`）
- **plan_keyframes_foundation_pose**（`plan_keyframes_foundation_pose.py` 同一条 IK 路径）
- **render_hand_retarget_piper_dual_npz_urdfik_main.py**（直接 replay 脚本，使用同一 `urdfik.py` + URDF）

Foundation 管线（`view_pick_diverse_bottles_piper_ik_motion.py` → `piper_ik.py`）**不受影响**——它已经有独立的 `piper_pika_agx.urdf` 覆盖逻辑。

### 修复 1：URDF 不匹配（已修复）

`PIPER_URDF` 在 [render_hand_retarget_piper_dual_npz_urdfik.py](code_painting/render_hand_retarget_piper_dual_npz_urdfik.py) L26-30 从 `piper/piper.urdf` 改为 `piper_pika_agx/piper_pika_agx.urdf`。两个 URDF 的 DH 参数完全不同：

| Joint | `piper.urdf` (旧) | `piper_pika_agx.urdf` (新) |
|---|---|---|
| joint3 xyz | `0.28358, 0.028726, 0` | `0.28503, 0, 0` |
| joint3 rpy | `0, 0, 0.10095` | `0, 0, -1.7938` |
| joint4 xyz | `-0.24221, 0.068514, 0` | `-0.02198, -0.25075, 0` |

### 修复 2：`--target_retreat_m` 参数（新增，替代硬编码 TCP→link6 偏移）

人手关键帧给出的是 **TCP**（夹爪尖端）世界位姿，但 IK 以 `link6`（腕关节）为目标。TCP = link6 + `gripper_bias`（Piper 为 0.12m）。

> **为什么不在代码里硬编码 `-gripper_bias`？** 因为不同机器人的 `gripper_bias` 不同，而且 O 节的 Foundation 管线通过 `--foundation_grasp_standoff_m` 已经证明了命令行参数更灵活。

`--target_retreat_m` 在 `build_plan_summary` 中施加：将人手 TCP 位姿沿夹爪 local Z（前进轴）回退指定距离，使得 link6 目标 = TCP - retreat。设置 `--target_retreat_m 0.12`（=gripper_bias）时 link6 到达人手 TCP 后方 12cm，实际 TCP 恰好到达人手位置。

| `--target_retreat_m` | link6 位置 | 实际 TCP 位置 | 等效 Foundation |
|---|---|---|---|
| `0`（默认，旧行为） | 人手 TCP 处 | TCP + 12cm（穿透物体） | grasp_standoff=0 |
| `0.12`（=gripper_bias） | TCP 后方 12cm | 人手 TCP 处（精准） | grasp_standoff=0.12 |
| `0.14`（+2cm 安全间距） | TCP 后方 14cm | TCP 后方 2cm | grasp_standoff=0.14 |

### 已修改的文件

| 文件 | 行 | 改动 |
|---|---|---|
| `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py` | L26-30 | `PIPER_URDF` → `piper_pika_agx.urdf` |
| `code_painting/plan_keyframes_human_replay.py` | L207-210 | 新增 `--target_retreat_m` 参数 |
| 同上 | L226-234 | `build_plan_summary` 中施加 retreat |
| 同上 | L253 | retreat 写入 plan_summary 元数据 |
| `code_painting/run_plan_keyframes_human_replay_piper_d435.sh` | L40-53 | 新增 `TARGET_RETREAT_M` + wrist tuning 默认值 |
| 同上 | L72-82,108-117,242-253 | 参数解析与传递（retreat + wrist tuning） |
| `code_painting/render_hand_retarget_r1_npz.py` | L1037-1073 | `get_wrist_camera_pose` 支持 `_wrist_camera_tuning`，新增 `_apply_wrist_tuning` |
| `code_painting/plan_anygrasp_keyframes_r1.py` | L330-342 | 新增 `--piper_calibration_bundle` + wrist tuning CLI 参数 |
| 同上 | L806-823 | `build_renderer` 设置 `_wrist_camera_tuning` |
| 同上 | L5191-5192 | 调用 `base.apply_piper_calibration_bundle(args)` |
| `code_painting/plan_keyframes_human_replay.py` | L364 | 传递 `--piper_calibration_bundle` 到 planner |
| 同上 | L441-453 | 新增 calibration bundle + wrist tuning 参数 |
| `code_painting/run_plan_keyframes_human_replay_piper_d435.sh` | L41 | 默认 `PIPER_CALIBRATION_BUNDLE` 指向 0515 bundle |
| `assets/embodiments/piper_pika_agx/curobo.yml` | L10 | `urdf_path` → `piper_pika_agx.urdf` |

### modelL1521-5 viewer 测试命令（带 O.1 wrist 校准 + retreat）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin
export DISPLAY=:1.0
xdpyinfo >/dev/null || { echo "DISPLAY=:1.0 不可用"; exit 1; }

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --tasks pick_diverse_bottles \
  --trajectory_mode joint_interp --joint_trajectory_interpolation cubic \
  --ik_num_seeds 1 --ik_solution_selection joint_continuity \
  --ik_seed_perturbations 6 --ik_seed_perturbation_scale 0.05 \
  --ik_max_joint_step_rad 0 --execute_partial_cartesian_plan 0 \
  --apply_global_trans_to_ik 0 --action_orientation_source grasp \
  --dual_stage_freeze_reached_arms_on_replan 1 \
  --reach_pos_tol_m 0.04 --reach_rot_tol_deg 180 \
  --replan_until_reached_max_attempts 5 --fail_on_execution_failure 1 \
  --target_retreat_m 0.14 \
  --wrist_left_forward_offset_m 0.145 --wrist_right_forward_offset_m 0.13 \
  --wrist_left_roll_deg -15 --wrist_right_roll_deg -60 \
  --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
  --wrist_left_pitch_deg 15 --wrist_right_pitch_deg 15 \
  --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274
```

> 批量无 viewer 命令（含 wrist 校准）见 [L15.20 → L5 批量采集](#l5-批量无-viewer-采集head--wrist-视频--target_retreat_m-014--o1-wrist-校准)。

## P. 可视化：L16 HaMeR / Foundation / repaint 拼接对比

### P1. L16 七面板拼接脚本（自动两行）

脚本位置：

```bash
/home/zaijia001/ssd/RoboTwin/code_painting/make_l16_repaint_montage.py
```

默认面板顺序：

1. `HaMeR gripper`：`/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_output/hand_vis_gripper_<ID>.mp4`
2. `Foundation object`：优先读取 `/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435/foundation_input_<ID>/head_cam_replay.mp4`，不存在时回退到 `foundation_replay`
3. `L16 robot plan`：`/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/<TASK>/foundation_input_<ID>/head_cam_plan.mp4`
4. `L16 left wrist`（存在才加入）：`.../foundation_input_<ID>/left_wrist_cam_plan.mp4`
5. `L16 right wrist`（存在才加入）：`.../foundation_input_<ID>/right_wrist_cam_plan.mp4`
6. `Stage1 inpaint`（可选，存在才加入）：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4`
7. `Final repaint`（可选，存在才加入）：`/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_visible_reinit/e0_robot_object/<TASK>/id_<ID>_l16/final_repainted.mp4`

默认 `--max_columns 4`，面板超过 4 个会自动用两行 `xstack` 拼接；如果只想强制横向拼接，可以加 `--layout hstack`。

输出位置：

```bash
/home/zaijia001/ssd/RoboTwin/code_painting/l16_repaint_montage/<TASK>/id_<ID>/compare_hamer_foundation_l16_repaint_<TASK>_id<ID>.mp4
```

### P2. 已测试 id0 命令（新 tmux）

`handover_bottle` 没有 L16 id0，`pnp_bread` 的 L16 从 id7 开始；因此 id0 debug 推荐先用 `pick_diverse_bottles`。

```bash
tmux new-session -d -s l16_vis_id0 'cd /home/zaijia001/ssd/RoboTwin && python3 /home/zaijia001/ssd/RoboTwin/code_painting/make_l16_repaint_montage.py --task pick_diverse_bottles --id 0 --overwrite'
```

检查输出：

```bash
ls -lh /home/zaijia001/ssd/RoboTwin/code_painting/l16_repaint_montage/pick_diverse_bottles/id_0
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,r_frame_rate,duration -of default=noprint_wrappers=1 /home/zaijia001/ssd/RoboTwin/code_painting/l16_repaint_montage/pick_diverse_bottles/id_0/compare_hamer_foundation_l16_repaint_pick_diverse_bottles_id0.mp4
```

当前已用 `pick_diverse_bottles id0` 做 1 秒 smoke 验证：七面板默认按 4 列自动折成两行，输出 `1704x640`、`5fps`。如果某个 id 缺少左右 wrist 或 optional 视频，实际面板数和输出宽高会按存在的视频自动调整。

### P3. 批处理命令

跑 id0-id4 中 L16 从 0 开始的任务：

```bash
cd /home/zaijia001/ssd/RoboTwin && python3 /home/zaijia001/ssd/RoboTwin/code_painting/make_l16_repaint_montage.py \
  --tasks pick_diverse_bottles place_bread_basket stack_cups pnp_tray \
  --ids 0-4 \
  --overwrite
```

补 `handover_bottle` 的前 5 个有效 L16 id：

```bash
cd /home/zaijia001/ssd/RoboTwin && python3 /home/zaijia001/ssd/RoboTwin/code_painting/make_l16_repaint_montage.py \
  --task handover_bottle \
  --ids 1-5 \
  --overwrite
```

补 `pnp_bread` 的前 5 个有效 L16 id：

```bash
cd /home/zaijia001/ssd/RoboTwin && python3 /home/zaijia001/ssd/RoboTwin/code_painting/make_l16_repaint_montage.py \
  --task pnp_bread \
  --ids 7-11 \
  --overwrite
```

如果只想看前三联（HaMeR / Foundation / L16 head，不加入 wrist 与当前 inpainting/repainting 结果），加：

```bash
--include_wrist 0 --include_optional 0
```

如果只想保留 head + wrist、但不加入 Stage1/final，使用：

```bash
--include_optional 0
```


### P4. ours 可视化选择：P montage -> JSON

用途：先把 P 中的 HaMeR / Foundation / L16 head / L16 left wrist / L16 right wrist / Stage1 / final repaint 七面板可视化整合出来，再用交互窗口逐条选择可用 episode。输出 JSON 兼容 L9.2 的 `--review-json`，后续 `ours` 数据集只会整合你标记为 `y` 的 id。

底层脚本：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/review_l16_ours_montages.py
```

任务级标注脚本：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pick_diverse_bottles.sh
/home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_place_bread_basket.sh
/home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_handover_bottle.sh
/home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pnp_bread.sh
/home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pnp_tray.sh
/home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_stack_cups.sh
```

建议按任务单独筛选，每个任务目标先卡 `25` 条。窗口底部和启动终端都会显示当前任务 `accepted/25`、`remaining`、`maybe`、`reject`、`unreviewed` 和 `total`。任务级脚本默认 `TARGET_COUNT=25`、`OVERWRITE_MONTAGE=1`，第一次升级到七面板时会重生成 montage；如果后面只想继续标注、不要重生成视频，使用 `OVERWRITE_MONTAGE=0 bash ...`。

`pick_diverse_bottles`：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pick_diverse_bottles.sh
```

`place_bread_basket`：

```bash
OVERWRITE_MONTAGE=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_place_bread_basket.sh
```

`handover_bottle`：

```bash
OVERWRITE_MONTAGE=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_handover_bottle.sh
```

`pnp_bread`：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pnp_bread.sh
```

`pnp_tray`：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pnp_tray.sh
```

`stack_cups` 的 B 方案完成后单独筛选。底层脚本会优先使用 `e0_robot_object_b_points_negative`，如果该目录存在：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_stack_cups.sh
```

常用环境变量：

```bash
TARGET_COUNT=25 OVERWRITE_MONTAGE=0 INITIAL_SPEED=1.5 bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pick_diverse_bottles.sh
```

如果只想先看一小段 id，可以把参数直接追加到任务脚本后面；已有 JSON 标记会保留：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pick_diverse_bottles.sh --ids 0-4
```

交互按键：

```text
y: accept 当前 id 并跳到下一条
n: reject 当前 id 并跳到下一条
m: maybe，默认 strict 转换不会纳入
u: 清除当前标记
space: 暂停/继续
.: 下一条
,: 上一条
+ 或 =: 加速播放
- 或 _: 减速播放
] / [: 兼容调速键；如果窗口/播放器把它们解释成逐帧，请改用 +/-
r: 从头播放当前视频
s: 保存 JSON
q/Esc: 保存并退出
```

输出位置：

```text
Montage:
/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review/montages/<TASK>/id_<ID>/compare_hamer_foundation_l16_repaint_<TASK>_id<ID>.mp4

每个任务的选择 JSON:
/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review/selections/<TASK>/ours_review_selection.json

合并 JSON:
/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review/selections/ours_review_selection_all.json
```

### P5. L16 whitebg / stack B 方案 mask 反选 debug

用途：检查 I3.6/I3.6.2 的 white-background Stage-2 是否正确。注意 `run_l16_whitebg_repaint_task.sh` 使用了 `--invert_mask`，所以输出目录中的 `mask_head_cam_plan/*.jpg` 已经是**反选后的 foreground alpha**，也就是用于把 L16 robot/object 像素贴回 Stage-1 背景的 mask；它不是原始白背景 mask。`w_box_head_cam_plan.mp4` 仍可能显示白背景检测框，因此不能只用它判断机械臂/杯子有没有被保留。

脚本位置：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/make_l16_whitebg_mask_debug.py
```

已生成的 stack id0 debug：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/l16_whitebg_mask_debug/stack_cups/id_0/whitebg_invert_debug_stack_cups_id0.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/l16_whitebg_mask_debug/stack_cups/id_0/whitebg_invert_debug_stack_cups_id0.json
```

重新生成 stack id0 前 120 帧 debug：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && \
python /home/zaijia001/ssd/RoboTwin/code_painting/make_l16_whitebg_mask_debug.py --task stack_cups --id 0 --max_frames 120
```

换其它 id：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && \
python /home/zaijia001/ssd/RoboTwin/code_painting/make_l16_whitebg_mask_debug.py --task stack_cups --id 10 --max_frames 120
```

debug 视频面板：

```text
1 L16 head source
2 Stage1 BG
3 saved alpha overlay      # 保存出的反选 foreground alpha 叠加图
4 saved alpha binary       # 保存出的反选 foreground alpha 二值图
5 inverse bg check         # 反过来看背景区域
6 final repaint
```

判断方式：如果第 3/4 面板主要覆盖机械臂和目标杯子、final repaint 正常贴回，说明 Stage-2 反选逻辑是对的；如果第 3/4 面板大面积覆盖白色背景，说明 `--invert_mask` 或后处理方向有问题。
