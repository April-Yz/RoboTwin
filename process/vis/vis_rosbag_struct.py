#!/usr/bin/env python3
"""
rosbag info clone (Pure Python - rosbags version v4)
Fix: Uses 'reader.topics' API for reliable message counting.
"""

import argparse
import datetime
from pathlib import Path

try:
    from rosbags.rosbag1 import Reader
except ImportError:
    print("错误: 未找到 rosbags 库。请运行: pip install rosbags")
    exit(1)

def format_size(size_bytes):
    if size_bytes == 0: return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(0)
    p = 1024
    s = float(size_bytes)
    while s >= p and i < len(size_name) - 1:
        s /= p
        i += 1
    return f"{s:.1f} {size_name[i]}"

def format_duration(duration_ns):
    if duration_ns is None: return "Unknown"
    seconds = duration_ns / 1e9
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):d}:{int(m):02d}:{s:05.2f}s ({int(seconds)}s)"

def format_time(timestamp_ns):
    if timestamp_ns is None or timestamp_ns == 0:
        return "Unknown"
    dt = datetime.datetime.fromtimestamp(timestamp_ns / 1e9)
    return dt.strftime("%b %d %Y %H:%M:%S.%f")[:-3] + f" ({timestamp_ns/1e9:.2f})"

def get_bag_info(bag_path):
    bag_path = Path(bag_path)
    if not bag_path.exists():
        print(f"Error: File {bag_path} does not exist.")
        return

    print(f"{'path:':<15} {bag_path.absolute()}")
    
    try:
        with Reader(bag_path) as reader:
            # --- 基本信息 ---
            print(f"{'version:':<15} 2.0")
            
            duration_ns = 0
            if reader.end_time and reader.start_time:
                duration_ns = reader.end_time - reader.start_time
            
            print(f"{'duration:':<15} {format_duration(duration_ns)}")
            print(f"{'start:':<15} {format_time(reader.start_time)}")
            print(f"{'end:':<15} {format_time(reader.end_time)}")
            print(f"{'size:':<15} {format_size(bag_path.stat().st_size)}")
            print(f"{'messages:':<15} {reader.message_count}")
            
            # 检测压缩类型
            # reader.compression 属性通常返回 compression format (e.g., Compression.LZ4)
            comp_str = str(reader.compression).split('.')[-1] if hasattr(reader, 'compression') else "Unknown"
            print(f"{'compression:':<15} {comp_str}")

            # --- 统计 Topics (使用 reader.topics) ---
            # reader.topics 返回一个字典: {topic_name: TopicInfo(msgtype, msgcount, connections)}
            
            topic_stats = {}
            types_set = set()

            if hasattr(reader, 'topics'):
                # 优先使用 topics 属性 (新版 rosbags)
                for topic, info in reader.topics.items():
                    topic_stats[topic] = {
                        'type': info.msgtype,
                        'count': info.msgcount
                    }
                    types_set.add(info.msgtype)
            else:
                # 降级方案：如果没有 topics 属性，只列出 topic 和 type (count 设为 ?)
                print("\n[Warning] Older rosbags version detected. Message counts unavailable.")
                for conn in reader.connections:
                    types_set.add(conn.msgtype)
                    topic_stats[conn.topic] = {'type': conn.msgtype, 'count': '?'}

            # --- 打印 Types ---
            print(f"{'types:':<15}")
            for t in sorted(list(types_set)):
                print(f"{'':<15} {t}")

            # --- 打印 Topics ---
            print(f"{'topics:':<15}")
            sorted_topics = sorted(topic_stats.items())
            
            for topic, stats in sorted_topics:
                # 格式对齐
                print(f"{'':<15} {topic:<40} {stats['count']:>10} msgs : {stats['type']}")

    except Exception as e:
        print(f"\nError reading bag structure: {e}")
        # 调试信息: 打印 reader 的所有属性，方便排查
        # print(f"Debug: Available attributes: {dir(reader)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get info from ROS1 bag (LZ4/ZSTD supported)")
    parser.add_argument("bag_file", help="Path to the .bag file")
    args = parser.parse_args()

    get_bag_info(args.bag_file)