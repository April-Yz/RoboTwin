# Piper IK Foundation OBJ 抓放

## 用途

使用 Foundation NPZ 中的位置和原始 OBJ，在 SAPIEN 中运行 Piper V1-V4 Cartesian 抓放、viewer 和两阶段数据采集。

## 前置条件

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
cd /home/zaijia001/ssd/RoboTwin
```

## Viewer

```bash
python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --task_config demo_piper_ik_foundation_v1 \
  --ik_version v1 --foundation_id 0 --foundation_frame 0 \
  --seed 0 --max_seed_tries 1 --require_success 1
```

无窗口验证时增加 `unset DISPLAY`、`--hold 0 --render_freq 0 --show_axes 0`。

## 数据采集

```bash
# 固定 input 0
bash collect_data.sh pick_diverse_bottles_piper_ik_foundation demo_piper_ik_foundation_v1 0

# 参数：IK 版本、foundation_input ID、frame、GPU ID
bash collect_foundation_piper_ik.sh v1 0 0 0
```

独立采集脚本按版本、ID 和 frame 生成独立 config 与输出目录。不能用 `sed -i` 修改基础 YAML 后复用同一个轨迹目录。

## 执行逻辑

- NPZ 提供左右位置、四元数和 OBJ 路径。
- OBJ bounds 的几何中心是抓取基准；actor 原点通常接近瓶底。
- visual 使用原始 OBJ；collision 默认使用 bounds 圆柱代理。
- 默认直立朝向。FoundationPose 朝向可选，但可能物理不稳定。
- 窄 OBJ 无法由当前 Pika 夹爪稳定纯接触抓取，默认启用距离门控 grasp drive，并在 open 时释放。
- Phase 1 顺序规划并保存；Phase 2 校验 Foundation source 和轨迹 schema 后回放，不初始化 planner。

## 已验证状态

2026-06-11 使用 `foundation_input_0/frame0`：V1-V4 viewer 和完整两阶段采集均成功。V3 MotionGen 优化失败后按设计回退到 IK 插值轨迹。

## 相关代码

- `envs/pick_diverse_bottles_piper_ik_foundation.py`
- `envs/pick_diverse_bottles_piper_ik.py`
- `view_pick_diverse_bottles_piper_ik_motion.py`
- `collect_foundation_piper_ik.sh`
- `task_config/demo_piper_ik_foundation_v1.yml` 至 `v4.yml`
