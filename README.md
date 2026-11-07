# RAP-SLAM

RAP 工作区的离线 SLAM 系统仓库，与同级 RAP-main 模型仓库独立管理。当前任务为迁移前工程整理，研究 UNRESOLVED、论文 PAUSED；服务器将用于继续研究与训练，全部实验历史随迁并常驻服务器。

先读 [本仓库约定](AGENTS.md)、[工作区当前状态](../RAP-main/docs/CURRENT_STATE.md)、[共用约定](../RAP-main/docs/CONVENTIONS.md)和[迁移计划](../RAP-main/docs/migration/PLAN.md)。

## 代码定位

现有系统流程：数据加载 → 里程计 → 关键帧 → 回环候选与 RAP 验证 → 位姿图 → 评测/输出。以下为源码入口定位，不是本轮新运行结果：

| 文件/目录 | 职责 |
| --- | --- |
| [scripts/run_pipeline.py](scripts/run_pipeline.py) | 离线管线执行入口 |
| [rap_slam/cli.py](rap_slam/cli.py) / [runtime_paths.py](rap_slam/runtime_paths.py) | 标准库参数解析、布局默认值和路径预检 |
| [rap_slam/rap_backend.py](rap_slam/rap_backend.py) | RAP 模型适配与配准，register(a,b) 为 b→a |
| [rap_slam/rap_precision.py](rap_slam/rap_precision.py) | 配准数值精度作用域 |
| [rap_slam/real_dataset.py](rap_slam/real_dataset.py) | KITTI 格式输入、帧范围和坐标约定 |
| [rap_slam/odometry.py](rap_slam/odometry.py) | 里程计包装 |
| [rap_slam/loop_closure.py](rap_slam/loop_closure.py) | 回环候选与验证 |
| [rap_slam/pose_graph.py](rap_slam/pose_graph.py) | 位姿图求解接口 |
| [rap_slam/evaluate.py](rap_slam/evaluate.py) | 评测与可视化输出 |
| [ros2](ros2) | 可选实时集成/可视化组件，依赖待 A4 单独核对 |

## 运行与迁移状态

[统一启动器与路径配置](../RAP-main/docs/code/RUNNING.md)支持 plan/check/execute；从任意目录传入明确配置，默认输出到新 RAP-runs 树。直接入口保留 argparse 参数并新增 --rap-ckpt/--spinnet-ckpt；默认根按源码布局或环境解析，--help 和路径预检无需数值依赖。坐标校准、0.205° 修正、帧范围与 55 个非路径参数及数值主体保持原合同。A3 只完成静态保持和标准库测试，模型/目标环境的实际 I/O 待 A4。

[requirements.txt](requirements.txt)只是现有 SLAM 补充依赖清单。[A4 环境入口](../RAP-main/docs/environment/README.md)提供分层依赖、搭建顺序、CPU 检查与训练缺口。现有环境的 CPU 依赖检查通过；目标重建/GPU/完整训练恢复尚未验收，不把当前包清单当作已验证锁文件。

[A5 数据/模型/历史资产](../RAP-main/docs/assets/README.md)登记本地覆盖与原字节基础包；[A7本地交付](../RAP-main/docs/migration/DELIVERY.md)已完成项目层全量恢复/字节与相对链接检查；真实目标权限/环境及服务器验收仍待闭合。data/、output*/ 和根运行日志保留原路径。A1 已建立逐文件归属和校验，尚未搬迁或删除；新运行不能覆盖这些历史产物。

## 实验与历史

- [实验总索引](../RAP-main/docs/experiments/README.md)：研究、维护和旧输出按类查阅；状态未知不等于成功。
- [当前研究边界](../RAP-main/docs/CURRENT_STATE.md)：E0 工程正面记录保留，PGC1/SSC1 停止及 official MAC 未运行等限制继续有效。
- [保全缺口](../RAP-main/docs/experiments/GAPS.md)：五个早期源码哈希和一份旧 AGENTS 中间版本未恢复，不能用现代码冒充。
- [本 README 整理前的完整原件](../RAP-main/docs/maintenance/2026-11-03/context-index-01/before/rap-slam/README.md)：保存原 7 月起的结果、图表说明、路线图及命令。原文的“当前/下一步”只代表当时状态，指标和来源限制应结合各次协议/报告。

组会与论文材料由[组会/报告总入口](../RAP-main/docs/meetings/README.md)进入；A6 已核对成品身份、图表字节关联和便携制作准备，版本/参考PDF/新导出保留项另列，论文仍暂停。本轮没有重新运行这些实验。
