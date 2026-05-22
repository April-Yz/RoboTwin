# SSD 指令库（中文）

> 约定：每条都采用 **一行注释 + 一行命令**。可直接复制到终端执行。


# 新的人手数据
rclone copy  gdrive:piper/human/stack_cups/ /home/zaijia001/ssd/data/piper/hand -P --dry-run
rclone copy  gdrive:piper/human/pick_diverse_bottles/pick_diverse_bottles-human-101.zip  /home/zaijia001/ssd/data/piper/hand -P --dry-run
rclone copy  gdrive:piper/human/place_bread_basket/human_place_bread_basket.zip  /home/zaijia001/ssd/data/piper/hand/place_bread_basket -P --dry-run
rclone copy  gdrive:piper/human/pnp_bread/pnp_bread-7.zip /home/zaijia001/ssd/data/piper/hand/pnp_bread/origin -P --dry-run

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

### B2. 运行 FoundationPose（pear + star fruit）

# 跑单个序列（video_id=0），同时检测 pear 和 star fruit（杨桃）
source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=3 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_input --output_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis --video_ids 0 --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj --object "star fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45

# 跑全部序列（默认会跳过已存在 poses.npz 的项）
source /home/zaijia001/FoundationPose/source_foundationpose_env.sh && cd /home/zaijia001/FoundationPose && CUDA_VISIBLE_DEVICES=3 python run_realr1_dino_sam_batch.py --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_input --output_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj --object "star fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj" --save_video 1 --save_mesh_overlay_video 1 --save_bbox_overlay_video 1 --mesh_overlay_alpha 0.45

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

# pick_diverse_bottles：FoundationPose 双物体 replay，使用 Piper 0515 head/base 标定；先用 --ids 0 单条验证
CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_vis/obs_vis /home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_replay 5 --ids 0 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object "right bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj" --object "left bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj"



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
CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json TARGET_LOCAL_FORWARD_RETREAT_M=0.05 GPU=2 FPS=5 MAX_FRAMES=300 ARMS=both KEEP_ONLY_ZED_THIRD=1 ID_FILTER=0 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_hamer_axes_replay_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2 /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_retreat_blue_z

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
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && GPU=3; FPS=5; MAX_FRAMES=300; RETREAT=0.05; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 0 10); do IN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz; OUT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK}/id${ID}_z005; [[ -f "$IN" ]] || { echo "[skip] missing $IN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz "$IN" --output_dir "$OUT" --fps ${FPS} --max_frames ${MAX_FRAMES} --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m ${RETREAT} --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 0 --debug_frame_limit -1 --debug_visualize_targets 0 --debug_visualize_cameras 0 --clean_output 1 --overlay_text_enable 0 --save_png_frames 0 --lighting_mode front_no_shadow; rm -f "$OUT"/zed_depth.mp4 "$OUT"/left_wrist_replay.mp4 "$OUT"/right_wrist_replay.mp4 "$OUT"/smooth_*.mp4; rm -rf "$OUT"/frames; for V in zed_replay third_replay; do [[ -f "$OUT/${V}.mp4" ]] || continue; ffmpeg -y -i "$OUT/${V}.mp4" -an -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT/${V}.tmp.mp4" && mv "$OUT/${V}.tmp.mp4" "$OUT/${V}.mp4"; done; done; done
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
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/place_bread_basket/harmer_output/hand_detections_${ID}.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/place_bread_basket/id${ID}_z005 --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 1 --debug_frame_limit -1 --save_png_frames 1 --object_replay_input_dir /home/zaijia001/ssd/data/piper/hand/place_bread_basket/foundation_vis/obs_vis/foundation_input_${ID} --object basket=/home/zaijia001/ssd/data/R1/hand/obj_mesh/basket/basket.obj --object bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_y/bread_y.obj --object_missing_frame_policy hide --lighting_mode front_no_shadow; done

# viewer 版本：建议先把 `seq 0 10` 改成单个 ID（例如 `seq 0 0`），再在 python 参数末尾追加这三个参数；无显示环境不要开
# --enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02

#### E2.3 stack_cups：手 + right_dark_red_cup/left_light_pink_cup

# 非 viewer 版本：批量跑 id0-id10，默认后退 5cm
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=2 conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_detections_${ID}.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay/h2o/stack_cups/id${ID}_z005 --fps 5 --max_frames 300 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 0 --debug_post_execute 1 --debug_frame_limit -1 --save_png_frames 1 --object_replay_input_dir /home/zaijia001/ssd/data/piper/hand/stack_cups/foundation_vis/obs_vis/foundation_input_${ID} --object right_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj --object left_light_pink_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj --object_missing_frame_policy hide --lighting_mode front_no_shadow; done

# viewer 版本：建议先把 `seq 0 10` 改成单个 ID（例如 `seq 0 0`），再在 python 参数末尾追加这三个参数；无显示环境不要开
# --enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02

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
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; DUMMY_ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/pick_diverse_bottles/id0_z005/zed_replay.mp4; [[ -f "$DUMMY_ROBOT" ]] || DUMMY_ROBOT=$(find /home/zaijia001/ssd/RoboTwin/code_painting/human_replay /home/zaijia001/ssd/RoboTwin/code_painting/human_object_replay -path '*id*' -name zed_replay.mp4 2>/dev/null | sort | head -n 1); [[ -f "$DUMMY_ROBOT" ]] || { echo "[error] no robot_video found; run E0 or E2.0 first"; exit 1; }; echo "[stage1] dummy robot_video=$DUMMY_ROBOT"; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 0 10); do HUMAN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input/rgb_${ID}.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; [[ -f "$HUMAN" ]] || { echo "[skip] task=${TASK} id=${ID} missing HUMAN=$HUMAN"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --human_video "$HUMAN" --robot_video "$DUMMY_ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --human_save_debug_artifacts 0 --robot_save_removed_video 0 --robot_save_mask_artifacts 0 --robot_save_debug_videos 0 --robot_save_composite_video 0; done; done
```

输出检查：

```bash
find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1 -path '*/stage1_human_inpaint/removed_w_mask_*.mp4' | sort
```

### I2. 三个任务：把 E0 pure replay 贴回 I1 背景

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=3; FPS=5; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 0 10); do BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/${TASK}/id${ID}_z005/zed_replay.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/${TASK}/id_${ID}; [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG under ${BG_ROOT}/stage1_human_inpaint; run I1 first"; continue; }; [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing pure ROBOT=$ROBOT; run E0 for this id first"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1; done; done
```

输出检查：

```bash
find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot -type f \( -name '*target_with_original*.mp4' -o -name '*repaint*.mp4' -o -name '*.mp4' \) | sort | head -n 80
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

### K2. 三个任务：把 AnyGrasp replay 贴回 I1 背景

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate inpainting-sam2-r1 && cd /home/zaijia001/ssd/inpainting_sam2_robot && GPU=0; FPS=5; for TASK in pick_diverse_bottles place_bread_basket stack_cups; do for ID in $(seq 0 10); do BG_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1/${TASK}/id_${ID}; BG=${BG_ROOT}/human_hand_bg.mp4; [[ -f "$BG" ]] || BG=$(find "${BG_ROOT}/stage1_human_inpaint" -maxdepth 1 -type f -name 'removed_w_mask_*.mp4' 2>/dev/null | sort | head -n 1); ROBOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/${TASK}/foundation_input_${ID}/head_cam_plan.mp4; OUT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/${TASK}/id_${ID}; [[ -f "$BG" ]] || { echo "[skip] task=${TASK} id=${ID} missing BG under ${BG_ROOT}/stage1_human_inpaint"; continue; }; [[ -f "$ROBOT" ]] || { echo "[skip] task=${TASK} id=${ID} missing ROBOT=$ROBOT"; continue; }; CUDA_VISIBLE_DEVICES=${GPU} python run_human_robot_inpaint_repaint.py --stage1_bg_video "$BG" --robot_video "$ROBOT" --output_dir "$OUT" --coords_type key_in --point_coords 10 80 --point_labels 1 --human_dilate_kernel_size 100 --robot_dilate_kernel_size 0 --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." --robot_box_threshold 0.20 --robot_text_threshold 0.20 --robot_max_mask_area_ratio 1.0 --robot_erode_kernel_size 3 --robot_composite_erode_kernel_size 1 --robot_blend_alpha_sigma 1.0 --robot_exclude_bottom_ratio 0.14 --mask_idx 2 --fps ${FPS} --device cuda --reuse_stage1; done; done
```

输出检查：

```bash
find /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp -type f \( -name '*target_with_original*.mp4' -o -name '*repaint*.mp4' -o -name '*.mp4' \) | sort | head -n 80
```
