# import os
# import h5py
# import glob

# # 将此路径替换为你报错日志中正在处理的数据文件夹路径
# # 也就是 ./training_data/pour-50 里面的具体 episode 路径
# data_path = "/projects/zaijia001/RoboTwin/policy/pi0/training_data/pour-50" 

# files = glob.glob(os.path.join(data_path, "*.hdf5")) + glob.glob(os.path.join(data_path, "*.h5"))

# print(f"Checking {len(files)} files...")

# for f_path in files:
#     try:
#         with h5py.File(f_path, 'r') as f:
#             # 检查关键的 key 是否存在
#             if 'observations/images/cam_left_wrist' not in f:
#                 print(f"[MISSING DATA] File: {f_path} is missing 'cam_left_wrist'")
#     except Exception as e:
#         print(f"[CORRUPTED] File: {f_path} cannot be opened. Error: {e}")


import os
import h5py
import glob

#这是你 ls 命令显示的根目录
root_dir = "/projects/zaijia001/RoboTwin/policy/pi0/training_data/pour-50"

# 使用 recursive=True 递归查找所有子目录下的 .hdf5 文件
# 注意：**/*.hdf5 配合 recursive=True 可以穿透文件夹
print(f"正在扫描目录: {root_dir} ...")
files = glob.glob(os.path.join(root_dir, "**", "*.hdf5"), recursive=True)

print(f"找到 {len(files)} 个 HDF5 文件。开始检查...")

for f_path in files:
    try:
        with h5py.File(f_path, 'r') as f:
            # 检查报错的那个关键 key
            # 注意：不同数据的层级可能不同，这里检查报错日志里缺少的那个路径
            target_key = 'observations/images/cam_left_wrist'
            
            if target_key not in f:
                print(f"\n[❌ 发现缺损] 文件: {f_path}")
                print(f"   原因: 缺少键值 '{target_key}'")
                
                # 顺便打印一下它有什么，方便对比
                # print(f"   它只有这些键: {list(f['observations/images'].keys())}")
            else:
                # print(f"[OK] {os.path.basename(f_path)}") # 如果不想刷屏，这行可以注释掉
                pass

    except Exception as e:
        print(f"\n[⚠️ 文件损坏] 无法打开文件: {f_path}")
        print(f"   错误信息: {e}")

print("\n检查完成。")