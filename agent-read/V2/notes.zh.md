# 说明

- lift 保持 grasp x/y 和姿态，仅增加 z。
- 每段从上一段轨迹末端关节状态开始。
- third 是右侧视角；opposite_top 是对向俯视。
- 旧 pickle 会被拒绝，不能与 v2 回放混用。

## Foundation O.1.1 / O.1.2

- O.1 不再在 close 前重置物体 pose。close 后状态门控检查物体是否仍稳定且位于双指夹持区域，通过后才在当前 pose 建立 drive。
- 默认底部 `support_proxy` 避免细瓶身在 pregrasp/grasp 时被张开夹爪推倒。
- O.1.1 使用第一标注关键帧建场；O.1.2 再用第二关键帧 EE xyz 的单一 action 代替 lift/place，朝向保留 grasp 设定。
