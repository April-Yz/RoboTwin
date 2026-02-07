#!/usr/bin/env python3
"""
ROS bag to MP4 Converter (Using pure-python 'rosbags' library)
Native support for LZ4/ZSTD compression. No ROS environment needed.

python3 vis_cam_final.py /projects/zaijia001/R1/pick/ --batch --output /projects/zaijia001/R1/pick_vis/

# lz4压缩版

    python3 vis_cam_final.py /projects/zaijia001/R1/pour/pour_0201 --batch --output /projects/zaijia001/R1/pour_vis/pour_0201/
    python3 vis_cam_final.py /projects/zaijia001/R1/pour/pour_0202 --batch --output /projects/zaijia001/R1/pour_vis/pour_0202/
    python3 vis_cam_final.py /projects/zaijia001/R1/pour/pour_0203 --batch --output /projects/zaijia001/R1/pour_vis/pour_0203/
    python3 vis_cam_final.py /projects/zaijia001/R1/pour/pour_0203_1 --batch --output /projects/zaijia001/R1/pour_vis/pour_0203_1/
    python3 vis_cam_final.py /projects/zaijia001/R1/pick/ --batch --output /projects/zaijia001/R1/pick_vis/
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# 尝试导入 rosbags
try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    print("错误: 缺少必要库。请运行: pip install rosbags")
    exit(1)

# --- 配置参数 ---
TARGET_TOPIC = '/hdas/camera_head/rgb/image_rect_color/compressed'
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
TARGET_FPS = 15 

def bag_to_mp4(bag_path, output_path):
    bag_path = Path(bag_path)
    if output_path.suffix.lower() != '.mp4':
        output_path = output_path.with_suffix('.mp4')

    print(f"\n>>> Processing: {bag_path.name}")
    print(f"    Output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取 ROS1 的类型定义
    typestore = get_typestore(Stores.ROS1_NOETIC)

    writer = None
    success_frames = 0
    
    try:
        # 使用 rosbags Reader 打开 (自动处理压缩)
        with Reader(bag_path) as reader:
            # 筛选我们需要的 topic 连接
            connections = [x for x in reader.connections if x.topic == TARGET_TOPIC]
            
            if not connections:
                print(f"    Warning: Topic {TARGET_TOPIC} not found. Skipping.")
                return False

            total_msgs = sum(1 for _ in reader.messages(connections=connections))
            if total_msgs == 0:
                print("    Warning: Topic found but contains no messages.")
                return False
                
            print(f"    [1/1] Converting images to MP4 ({total_msgs} frames)...")

            # 重新建立生成器进行读取
            msg_gen = reader.messages(connections=connections)
            
            # 初始化视频写入器 (使用第一帧来确定尺寸，或者强制 Resize)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, TARGET_FPS, (TARGET_WIDTH, TARGET_HEIGHT))

            with tqdm(total=total_msgs, leave=False, unit='frame') as pbar:
                for connection, timestamp, rawdata in msg_gen:
                    try:
                        # 反序列化数据
                        msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                        
                        # 提取图像数据 (CompressedImage format)
                        # msg.data 是 uint8 数组
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if img_bgr is not None:
                            # Resize
                            if img_bgr.shape[1] != TARGET_WIDTH or img_bgr.shape[0] != TARGET_HEIGHT:
                                img_bgr = cv2.resize(img_bgr, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
                            
                            writer.write(img_bgr)
                            success_frames += 1
                        
                    except Exception as e:
                        # pass
                        print(f"Debug Error: {e}")
                    
                    pbar.update(1)

    except Exception as e:
        print(f"    Error reading bag: {e}")
        return False
    finally:
        if writer:
            writer.release()
        # 清理空文件
        if success_frames == 0 and output_path.exists():
            output_path.unlink()
            return False

    print(f"    Done! Saved to {output_path.name}")
    return True

# --- 批量处理逻辑 ---
def batch_process(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bag_files = sorted(list(input_path.glob("*.bag")))
    # 排除掉之前生成的 fixed_ 文件 (如果还没删的话)
    bag_files = [f for f in bag_files if not f.name.startswith('fixed_')]

    print(f"Found {len(bag_files)} bag files in {input_path}")
    
    success_count = 0
    for i, bag_file in enumerate(bag_files):
        mp4_name = bag_file.with_suffix('.mp4').name
        mp4_path = output_path / mp4_name
        
        print(f"\n[{i+1}/{len(bag_files)}] Starting conversion...")
        if mp4_path.exists():
            print(f"    Skipping {mp4_name} (Already exists)")
            success_count += 1
            continue
            
        if bag_to_mp4(bag_file, mp4_path):
            success_count += 1
            
    print(f"\nBatch processing complete! Processed {success_count}/{len(bag_files)} files.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ROS bag to MP4 using pure python rosbags')
    parser.add_argument('input', type=str, help='Input directory or file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output directory')
    parser.add_argument('--batch', '-b', action='store_true', help='Enable batch processing')
    
    args = parser.parse_args()
    input_path = Path(args.input)
    is_dir = input_path.is_dir() or args.batch
    
    if is_dir:
        batch_process(args.input, args.output)
    else:
        output_file = Path(args.output)
        if output_file.suffix == '' or output_file.is_dir():
            output_file.mkdir(parents=True, exist_ok=True)
            output_file = output_file / input_path.with_suffix('.mp4').name
        bag_to_mp4(input_path, output_file)