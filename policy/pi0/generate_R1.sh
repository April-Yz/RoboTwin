data_dir=${1}
repo_id=${2}
use_wrist=${3:-false}  # 第三个参数，默认为false

if [ "$use_wrist" = "true" ]; then
    uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw_dir $data_dir --repo_id $repo_id --use-wrist
else
    uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --raw_dir $data_dir --repo_id $repo_id
fi
