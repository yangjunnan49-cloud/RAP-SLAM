# 2026-11-07：历史同步与审计完成

本仓库历史已提交并同步所有者指定的新GitHub仓库及AutoDL文件存储；旧origin保留为legacy-origin。审计详情与最新实验纠正见[统一报告](../RAP-main/docs/maintenance/2026-11-07/source-and-research-audit-01/REPORT_CN.md)。已复现外部评估分离目录路径错误，尚未修复；训练采样问题见主仓库审计。历史科学停止和原始产物保全保持，不自动启动实验或训练。

---

# 2026-11-07：历史提交与审计

所有者已明确授权提交、同步现有历史，并审计源码、实验结果与创新策略。历史入库范围及原AGENTS见[本次记录](docs/versioning/2026-11-07/README.md)。本轮允许Git新提交与指定目标同步；原科学停止、来源缺口和原始产物保全继续有效。审计与头脑风暴不自动启动新训练或恢复旧实验。

---

# RAP-SLAM 当前约定

本仓库是 RAP 工作区中的下游系统，独立管理；先读 [RAP-main 核心上下文](../RAP-main/AGENTS.md)、[当前状态](../RAP-main/docs/CURRENT_STATE.md)和[项目约定](../RAP-main/docs/CONVENTIONS.md)。本文件不重复历史指标或建立另一套研究状态。

更新：2026-11-06，Asia/Shanghai。A8：文件、环境与有界模型I/O验收通过；完整训练/恢复链仍有保留项。实际记录由RAP-main/docs/migration/AUTODL.md和统一状态页承担；原科学停止与历史缺口保持。

- 保留原用户未提交修改、独立仓库边界、算法参数、RNG、精度作用域和输入/输出语义；当前代码改动需按 A3 做行为保持检查。
- `RAPBackend.register(a,b)` 映射 b→a；保持与 pose graph 的约定。原始/校正 LiDAR、KITTI 相机位姿和 POSS LiDAR/world 不混用。
- 数值精度以当前 rap_precision.py 的作用域及已冻结源码为准；不按父级旧历史描述恢复全局 TF32 设置。
- 当前离线入口从 RAP-main/scripts/run.py 生成计划、--check 检查路径、--execute 显式执行；路径优先级 CLI > JSON > 环境 > 布局。scripts/run_pipeline.py 参数已拆到 rap_slam/cli.py，runtime_paths.py 负责布局默认值与预检，新增 --rap-ckpt/--spinnet-ckpt。55 个原非路径参数及数值主体保持；[A4 环境说明](../RAP-main/docs/environment/README.md)登记依赖/编译来源与训练缺口；没有宣称可完整训练或目标环境可用。
- 数据、模型和历史原件未移动/删除；本地包与原路径清单见 [A5 资产入口](../RAP-main/docs/assets/README.md)。
- data/、output*/、根运行日志和失败产物均有证据/来源角色，不能因 Git ignored/untracked 或文件名看似临时就删除。新运行使用新目录，保护旧结果与 seal。
- A4 仅短暂启动 WSL 做元数据采集与显式 CPU 依赖/小型算子验证，已恢复停止；没有初始化 CUDA、构造模型、加载数据/权重、训练、安装/软件包下载、Git 历史修改或对外发送。PGC1 未读部分准确率保持未解释读取；具体停止与来源限制见核心上下文。
- 代码/注释英文，当前维护文档中文，历史原文保留。每阶段更新核心状态及本仓库相关入口，避免累积相互竞争的“当前状态”。
