source /data1/zjyang/anaconda3/etc/profile.d/conda.sh
# bash script/run_painting_v1_table.sh 3 "pour" 0 0 5 > log/1103_pour_0_0_2mode3max=5maddtable1.json
# bash script/1-0run_painting_v1_only_obj.sh 3 "pour" 0 0 5
# bash script/1-1run_painting_v1_only_obj.sh 1 "pour" 2 2 5
# 用 hdf5_aloha.py 生成的json render + inpainting
#调整代码，帮我检查导入的这两个模型是否在按照json的数据执行（model1可能需要修改，我看到结果好像他被固定在了一个位置）。2.把手部动作的执行取消，机械臂不用移动，在系统的初始位置
# python -m code_painting.2-1render_from_preprocessed_grasp_R1 --mode 2 --test_orientation

# 后退基座，拍不到手臂，减少干扰
train_gpu=${1:-"0"}
# task=${1}
task_name=${2:-"basic_pick_place"} #"clean_cups" # "declutter_desk" #"basic_pick_place" # "assemble_disassemble_furniture_bench_lamp"
start_idx=${3:-"0"}
end_idx=${4:-"1"}
fps=${5:-"5"}
lg=${6:-"0"}
rg=${7:-"3"}
printf 'train_gpu = %s\n' "$train_gpu"
printf 'task = %s\n' "$task_name"
printf 'start_idx = %s\n' "$start_idx"
printf 'end_idx = %s\n' "$end_idx"
printf 'fps = %s\n' "$fps"
printf 'lg = %s\n' "$lg"
printf 'rg = %s\n' "$rg"
# cd ../
for video_id in $(seq ${start_idx} ${end_idx})
# for video_id in {7..7}
do
    conda activate RoboTwin
    printf 'video_id = %s\n' "$video_id"
    CUDA_VISIBLE_DEVICES=${train_gpu}  python -m code_painting.2-1render_from_preprocessed_grasp_R1 \
            --mode 2 \
            --task ${task_name} \
            --id ${video_id} \
            --fps 5 \
            --pure 0 \
            --lg ${lg} \
            --rg ${rg} \
            --test_orientation \
            --session pour_${video_id}_left_bottle_right_cup #"pour_0_right_cup" \
            # --session2 "pour_0_left_bottle"
            # --session "session_1101190816" \
            # --session2 "session_1029165750"
            # --model_path "/data1/zjyang/program/OnePoseviaGen/temp_local/session_1029165750/model/mid_files/scaled_mesh.obj" \
            # --poses_path "/data1/zjyang/program/OnePoseviaGen/temp_local/session_1029165750/pose_result/poses.json"
            # /double_model/double_model.obj session_1029163454 session_1029165750 | bottle session_1101161225 session_1101190816


    printf '处理Inpaint-Anything: video_idx=%s\n' "$video_id"
    printf '处理Inpaint-Anything: video_path=/data1/zjyang/program/third/RoboTwin/code_painting/%s/%s_%s_%sfps.mp4\n' "$task_name" "$task_name" "$video_id" "$fps"
    cd ../Inpaint-Anything/
    conda activate inpainting


    printf '处理Inpaint-Anything_human: video_idx=%s\n' "$video_id"
    # CUDA_VISIBLE_DEVICES=${train_gpu} python remove_anything_video_sam2.py \
    #     --input_video /data1/zjyang/program/egodex/egodex_stored/${task_name}/${video_id}.mp4 \
    #     --coords_type key_in \
    #     --point_coords 10 80 \
    #     --point_labels 1 \
    #     --dilate_kernel_size 100 \
    #     --output_dir ./results/${task_name}/ \
    #     --sam_model_type "vit_h" \
    #     --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
    #     --lama_config lama/configs/prediction/default.yaml \
    #     --lama_ckpt ./pretrained_models/big-lama \
    #     --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
    #     --mask_idx 2 \
    #     --vi_ckpt ./pretrained_models/sttn.pth \
    #     --fps 15

    # printf '处理Inpaint-Anything_robot: video_idx=%s\n' "$video_id"
    # CUDA_VISIBLE_DEVICES=${train_gpu}  python remove_anything_video_sam2_robot.py \
    #     --input_video /data1/zjyang/program/third/RoboTwin/code_painting/${task_name}/${task_name}_${video_id}_lg${lg}_rg${rg}_${fps}fps.mp4 \
    #     --target_video /data1/zjyang/program/third/Inpaint-Anything/results/${task_name}/removed_w_mask_${video_id}.mp4 \
    #     --coords_type key_in \
    #     --point_coords 10 80 \
    #     --point_labels 1 \
    #     --dilate_kernel_size 0 \
    #     --output_dir ./results/${task_name} \
    #     --sam_model_type "vit_h" \
    #     --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
    #     --lama_config lama/configs/prediction/default.yaml \
    #     --lama_ckpt ./pretrained_models/big-lama \
    #     --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
    #     --mask_idx 2 \
    #     --vi_ckpt ./pretrained_models/sttn.pth \
    #     --fps 15    
    cd ../RoboTwin/
done



# # training data
# xvfb-run -a python nerf_dataset_generator_bimanual.py --tasks=${task} \
#                             --save_path="../../../data_nerf/train_data" \
#                             --image_size=256x256 \
#                             --episodes_per_task=100  
#                             # --all_variations=True
#                             # --processes=1 \      
#                             # --renderer=opengl \                     
# cd ..