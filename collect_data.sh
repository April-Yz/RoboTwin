#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}
export SAPIEN_RT_DENOISER=${SAPIEN_RT_DENOISER:-none}

# File lock to prevent SAPIEN renderer deadlock when two collect_data
# instances run on different GPUs simultaneously.
LOCKFILE="/tmp/robottwin_collect_data.lock"
(
  flock -x 9 || exit 1
  PYTHONWARNINGS=ignore::UserWarning \
  python script/collect_data.py $task_name $task_config
  RET=$?
  rm -rf data/${task_name}/${task_config}/.cache
  exit $RET
) 9>"$LOCKFILE"
