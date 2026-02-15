import os
import h5py
import glob
import numpy as np
import argparse
# 在process_data_R1.py 前检查数据
"""
r1_data_20260202_171414.h5               | OK         | ONLY_RIGHT      | 682   | 
r1_data_20260202_172011.h5               | OK         | ONLY_RIGHT      | 468   | 
"""

def check_h5_file(file_path):
    """
    检查单个 HDF5 文件的结构完整性
    """
    report = {
        "status": "OK",
        "missing_keys": [],
        "shape_mismatch": [],
        "wrist_cam_status": "BOTH", # BOTH, LEFT, RIGHT, NONE
        "steps": 0
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 1. 检查关键路径是否存在 (根据 process_data_R1.py 的 load_hdf5 函数)
            required_keys = [
                "/obs/arm_left/eef_pos",
                "/obs/arm_left/eef_euler",
                "/obs/gripper_left/joint_pos",
                "/obs/arm_right/eef_pos",
                "/obs/camera_head/rgb", # 头部相机通常是必须的
                "/action/arm_left/eef_pos", 
                "/action/gripper_left/commanded_pos"
            ]
            
            for key in required_keys:
                if key not in f:
                    report["missing_keys"].append(key)
                    report["status"] = "CORRUPTED"
            
            # 如果关键 key 缺失，直接返回
            if report["status"] == "CORRUPTED":
                return report

            # 2. 检查手腕相机情况
            has_left_cam = "/obs/camera_left/rgb" in f
            has_right_cam = "/obs/camera_right/rgb" in f
            
            if has_left_cam and has_right_cam:
                report["wrist_cam_status"] = "BOTH_PRESENT"
            elif has_left_cam:
                report["wrist_cam_status"] = "ONLY_LEFT"
            elif has_right_cam:
                report["wrist_cam_status"] = "ONLY_RIGHT"
            else:
                report["wrist_cam_status"] = "NO_WRIST_CAM"

            # 3. 检查数据长度对齐
            # 获取观测长度
            obs_len = f["/obs/arm_left/eef_pos"].shape[0]
            action_len = f["/action/arm_left/eef_pos"].shape[0]
            report["steps"] = obs_len

            # 你的转换代码假设 obs 和 action 长度一致，或者允许差1
            # 原代码: state_list = state_all[:-1], actions = action_all[1:]
            # 这意味着 obs 和 action 的原始长度必须相等，或者非常接近
            if abs(obs_len - action_len) > 1:
                report["shape_mismatch"].append(f"Obs len {obs_len} != Action len {action_len}")
                report["status"] = "MISMATCH"

            # 4. 检查图像数据长度是否与状态一致
            head_cam_len = f["/obs/camera_head/rgb"].shape[0]
            if head_cam_len != obs_len:
                report["shape_mismatch"].append(f"Head Cam len {head_cam_len} != Obs len {obs_len}")
                report["status"] = "MISMATCH"

    except OSError:
        report["status"] = "BROKEN_FILE" # 文件损坏无法打开
    except Exception as e:
        report["status"] = f"ERROR: {str(e)}"
        
    return report

def main():
    # 默认路径
    default_dir = "/projects/zaijia001/R1/h5/pour/selected"
    
    # 获取所有 h5 文件
    files = sorted(glob.glob(os.path.join(default_dir, "*.h5")))
    
    if not files:
        print(f"❌ 在路径 {default_dir} 下没有找到 .h5 文件。")
        return

    print(f"🔎 开始检查路径: {default_dir}")
    print(f"📦 共发现 {len(files)} 个文件\n")
    print(f"{'文件名':<40} | {'状态':<10} | {'手腕相机':<15} | {'步数':<5} | {'备注'}")
    print("-" * 100)

    error_count = 0
    no_wrist_count = 0

    for file_path in files:
        file_name = os.path.basename(file_path)
        result = check_h5_file(file_path)
        
        status = result["status"]
        wrist = result["wrist_cam_status"]
        steps = result["steps"]
        notes = ", ".join(result["missing_keys"] + result["shape_mismatch"])
        
        # 颜色输出 (终端高亮)
        color_start = ""
        color_end = "\033[0m"
        
        if status == "OK":
            color_start = "\033[92m" # Green
        elif status == "BROKEN_FILE":
            color_start = "\033[91m" # Red
            error_count += 1
        else:
            color_start = "\033[93m" # Yellow
            error_count += 1
            
        if wrist == "NO_WRIST_CAM":
            no_wrist_count += 1

        print(f"{color_start}{file_name:<40} | {status:<10} | {wrist:<15} | {steps:<5} | {notes}{color_end}")

    print("-" * 100)
    print(f"检查完成。总文件: {len(files)}")
    if error_count > 0:
        print(f"❌ 发现 {error_count} 个文件有问题！请检查上方红色/黄色标记。")
    else:
        print(f"✅ 所有文件结构基本正常。")
    
    if no_wrist_count > 0:
        print(f"⚠️ 注意: 有 {no_wrist_count} 个文件缺少手腕相机数据。")
        print("   如果在运行 process_data_R1.py 时不加 --no-wrist 参数，这些文件可能会导致处理出的数据为空或报错。")

if __name__ == "__main__":
    main()