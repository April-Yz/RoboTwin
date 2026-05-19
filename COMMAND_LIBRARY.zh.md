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

# 回放 FoundationPose 结果（Piper 标定 head cam 版本，无机器人，且保存姿态debug；建议先跑单个id）
CUDA_VISIBLE_DEVICES=3 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_pnp_star_pear_norobot 5 --ids 0 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj --object star_fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj
批处理
CUDA_VISIBLE_DEVICES=3 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_pnp_star_pear_norobot 5 --lighting_mode front_no_shadow --hide_robot 1 --save_head_depth 1 --save_anygrasp_frames 1 --save_pose_debug 1 --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json --camera_cv_axis_mode legacy_r1 --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj --object star_fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj

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
