# RAP-SLAM

以 **RAP**（Register Any Point，基于流匹配的多路点云全局配准模型）为配准引擎的**离线 / 非实时 SLAM 管线**。与 sibling 目录 `RAP-main`（上游 RAP 仓库，只读使用、不做任何修改）分离管理。

```
传感器数据 → [前端] 预处理 → KISS-ICP 里程计 → 关键帧筛选
          → [回环] Scan Context 全局检索 + 邻近候选 → RAP 两路配准几何验证
          → [后端] GTSAM 位姿图优化 → evo 评估 / 拼接地图
```

## 当前状态：N 路联合配准后端首轮对照完成 ✅（路线图第 4 项）

### RAP N-way 联合全局配准 vs GTSAM PGO（POSS，2026-07-20）

`rap_backend.register_multi()` 把一批关键帧一次性送入 RAP 联合配准（所有部件落在同一坐标系，约束天然环一致），三个批次对照：

| 批次 | 内容 | N-way 布局（oracle SE(3) 对 GT 拟合残差） | 回环相对位姿误差 | Path A：N-way 约束 + PGO | Path B：直接锚定（无优化） |
|---|---|---|---|---|---|
| B1 | 5 个回环关键帧 ×4000 点 | **rmse 0.145 m** | 0.17–0.53 m / ≤1.1° | **1.736 m**（最优） | 8.37 m |
| B2 | 7 个（B1+时序邻居）×4000 点 | rmse 0.153 m | 0.12–0.50 m / ≤1.7° | 1.781 m | 8.65 m |
| B3 | 全部 82 关键帧 ×1500 点（12.3 万点，推理 38.6 s） | rmse 55.0 m（全局翘曲） | 0.21–0.49 m / ≤1.8° | 2.313 m | 98.2 m |

基线：里程计 8.257 m，pairwise-PGO 1.821 m。

**结论（首轮）**：
1. **小规模回环批次的 N 路联合配准精度极高**——B1 的 5 个关键帧位姿本身中位误差仅 0.35 m，布局残差 0.145 m；
2. **N 路约束喂给 PGO（Path A）优于独立两两配准约束**：1.736 vs 1.821 m（−4.7%），环一致性的价值得到验证；
3. **一次性联合配准扩展到 82 帧时全局布局翘曲**（局部相对几何仍保持 0.2–0.5 m 精度）——全局一致性仍需 PGO 负责；
4. Path B（直接锚定、无优化）失败原因有二：稀疏批次只修正批次关键帧、轨迹内部保持漂移（B1/B2 ≈ 里程计）；稠密批次受布局翘曲影响（B3）。
5. RAP 采样是随机的，单次运行间存在 ±0.05 m 量级波动（B1 Path A 两次运行 1.79 / 1.74 m），后续需多种子评估。

**定位**：RAP N 路目前最适合做"批次一致的回环约束工厂"喂给 PGO，而非替代 PGO；下一步是分层方案（回环簇内 N 路联合、簇间 PGO 缝合）。

实验脚本：`scripts/eval_nway_backend.py`（产物 `output_nway/`，含 `accepted_loops.json`、`batch_*.npz` 缓存、`summary_nway.json`；`--reuse-cache` 可跳过重复推理）。

### Scan Context 检索验证（POSS 序列，2026-07-20）

- **检索精度**（82 个关键帧，GT 距离 < 3 m 且间隔 ≥ 200 帧为真回环）：SC 距离阈值 0.13–0.30 区间内 **precision = recall = 1.0**；真对 SC 距离 0.06 / 0.106，最近假对 0.347，间隔 3 倍以上
- **抗漂移**：邻近检索半径收紧到 ≤ 10 m 时 recall 归零（回环处漂移 13.3 m）；SC 不依赖里程计，照常全部命中
- **端到端对照**（`--loop-dist 5`）：proximity-only 0 候选、APE 停在 8.26 m；`--loop-method both` 下 SC 单独捞出 2 对真回环，RAP 2/2 验收（对 GT 误差 0.15 / 0.34 m），PGO 后 **APE 1.90 m**
- **开销**：82 帧描述子构建 0.2 s，654 对匹配 0.6 s，相对 RAP 验证可忽略
- 评估脚本：`scripts/eval_loop_retrieval.py`（复用 `output_poss/odom_poses.txt`，无需重跑里程计）；对照产物 `output_poss_prox5/`、`output_poss_sc5/`

### 真实序列结果（POSS 校园数据集，帧 3850–5268）

- **数据**：北大 POSS odometry 数据集（Velodyne-32e，KITTI 同格式，CC BY-NC-SA 3.0），1419 帧真实扫描，路径 413 m，每帧约 6.2 万点，首尾回环（GT 间距 0.08 m）
- **里程计**（KISS-ICP，deskew 关闭）：APE RMSE 8.26 m（约 2% 漂移）
- **回环**：候选 4 对全部命中真实回环，**RAP 验证精度 0.15–0.47 m / 0.0°**（对 GT），重叠率 0.75–0.88
- **PGO 优化后**：**APE RMSE 1.82 m（↓ 78%）**，max 3.80 m；GTSAM 3 次迭代收敛
- **耗时**：全程 67.9 s（RTX 5070；里程计 40.5 s，RAP 验证 4 次共 9.2 s，每次约 0.5–1 s）

| 指标 | 纯里程计 | PGO 优化后 | 变化 |
|---|---|---|---|
| APE RMSE | 8.26 m | **1.82 m** | ↓ 78.0% |
| APE max | 13.32 m | 3.80 m | ↓ 71.4% |

结果产物：`output_poss/trajectory_plot.png`、`output_poss/summary.json`、`output_poss/merged_map_*.ply`

### 模拟序列基线（KITTI-00 整图模拟闭环）

- 101 帧模拟扫描，APE RMSE 5.53 m → 3.67 m（↓ 33.7%），产物在 `output/`。用于管线逻辑冒烟测试。

### 真实序列踩过的两个关键坑（已修复）

1. **GT 坐标系约定**：POSS 的位姿**已经是** LiDAR/world 系（轨迹水平、z-up），不能再套 KITTI 的相机系变换——否则轨迹被旋转成立面，评估全错。`real_dataset.load_gt_poses(apply_calib=...)` 控制，KITTI 用 True，POSS 用 False。**KITTI 转换方向已于 2026-07-22 修正**：正确公式是 `T_velo = Tr⁻¹ @ T_cam @ Tr`（此前误用 `Tr @ T @ Tr⁻¹`，GT 世界系被转约 90°，KITTI 04"里程计 APE"高达 310.7 m，修复后 2.96 m；该 bug 因一直无 KITTI 雷达数据而潜伏未暴露）。
2. **回环验收逻辑自相矛盾**：原逻辑要求 RAP 结果与（已漂移的）里程计一致（< 5 m）才接受——但回环存在的意义就是修正漂移，真实回环在漂移下必然"不一致"（实测漂移 13.3 m，RAP 对 GT 误差仅 0.15–0.47 m）。现改为以 RAP 内在几何质量（重叠率）验收，一致性仅作诊断日志。

## 数据获取（POSS，国内服务器速度快）

位姿/标定小文件直接下载；雷达数据从 24 GB 官方 zip 用 HTTP Range 按需抽取单段（约 1.4 GB / 4 分钟）：

```bash
# 位姿 + 标定（约 3 MB）
curl -O http://www.poss.pku.edu.cn/OpenDataResource/poss_odomotry/data_odometry_poses.zip
curl -O http://www.poss.pku.edu.cn/OpenDataResource/poss_odomotry/data_odometry_calib.zip
# 雷达帧段（断点续传，已下载的自动跳过）
python scripts/download_poss_segment.py --start 3850 --end 5268 --threads 4
# 第二段（POSS-B，帧 6750–9500，~2.8 GB；GT 重访分析选出的回环富集段，
# 分析依据见 RAP-main/AGENTS.md 2026-08-01 附录）
python scripts/download_poss_segment.py --start 6750 --end 9500 --threads 4
```

## 运行方法（WSL2）

```bash
wsl -d Ubuntu-24.04
source /root/miniconda3/bin/activate rap-inference
cd /mnt/e/githouse/RAP-SLAM

# 真实序列（POSS）
python scripts/run_pipeline.py --source real --output ./output_poss --loop-sep 200

# 模拟序列（默认）
python scripts/run_pipeline.py --output ./output
```

真实序列参数：`--real-root`（默认 data/poss）、`--real-start/--real-end`（帧段）、`--real-stride`（抽帧）、`--real-apply-calib`（KITTI 约定时启用）、`--odom-max-range`。

回环检索参数：`--loop-method {proximity,sc,both}`（默认 both）、`--sc-dist-thresh`（默认 0.25，POSS 上 0.13–0.30 均 P=R=1.0）、`--sc-rings/--sc-sectors`（20×60）、`--sc-max-radius`（80 m）、`--sc-shortlist`（ring-key 粗筛数，默认 10）。

## 目录结构

```
RAP-SLAM/
├── README.md
├── requirements.txt          # kiss-icp / gtsam / evo（装入 rap-inference conda 环境）
├── rap_slam/
│   ├── sequence.py           # 模拟扫描序列（真实整图 + 虚拟轨迹）
│   ├── real_dataset.py       # KITTI 格式真实序列加载（KITTI odometry / POSS）
│   ├── odometry.py           # KISS-ICP 逐帧里程计（Open3D ICP 兜底）
│   ├── keyframes.py          # 按位移/旋转阈值筛选关键帧
│   ├── scan_context.py       # Scan Context 全局位置识别（ring-key 粗筛 + 列移匹配）
│   ├── rap_backend.py        # RAP 模型进程内封装（register 两路 / register_multi N 路联合配准）
│   ├── loop_closure.py       # 邻近/SC 候选合并 + RAP 几何验证（重叠率验收）
│   ├── pose_graph.py         # GTSAM 位姿图优化
│   └── evaluate.py           # evo APE/RPE + 轨迹图 + 拼接地图
├── scripts/
│   ├── run_pipeline.py       # 唯一入口：六阶段一键运行
│   ├── eval_loop_retrieval.py    # 回环检索 precision/recall 评估（对照 GT）
│   ├── eval_nway_backend.py  # RAP N 路联合配准后端 vs PGO 对照实验
│   ├── download_poss_segment.py  # POSS 雷达段 Range 抽取下载器
│   ├── test_rap_backend.py   # RAP 封装单元测试
│   └── _*.py / sweep_*.py    # 开发期调试/参数扫描脚本（非正式产物）
├── data/
│   ├── poss/dataset/...      # POSS 位姿、标定、1419 帧 velodyne
│   └── kitti/dataset/...     # KITTI 位姿 + 标定（全 22 序列）；velodyne 扫描
│                             #   在 E:\githouse\Datasets\data_odometry_velodyne
│                             #   （全 22 序列完整已验证 2026-08-01，poses/calib/times 已组装，可直接作 --real-root）
├── output/                   # 模拟序列结果
├── output_poss/              # 真实序列结果（proximity 检索）
├── output_poss_sc/           # 真实序列结果（both：SC + 邻近合并，复现 1.82 m）
├── output_poss_prox5/        # 抗漂移对照：proximity-only @5 m（0 回环，8.26 m）
├── output_poss_sc5/          # 抗漂移实验：both @5 m（SC 捞出 2 回环，1.90 m）
└── output_nway/              # N 路后端实验（accepted_loops.json、batch_*.npz、summary_nway.json）
```

## RAP 集成方式（rap_backend.py）

- 通过 `sys.path` 只读导入 `RAP-main`，Hydra 组合 `RAP_inference` 配置（`rap_12`，5 步 Euler），加载 `weights/rap_model_12.ckpt`，**模型全程只加载一次**。
- 预处理完全复刻 `demo.py`：体素降采样 → 统计去离群 → miniSpinNet 32-D 特征 → 与 `PointCloudDataset._transform()` 一致的归一化。
- `register(points_a, points_b)` 返回 `T_rel = T_a⁻¹ @ T_b`（b 在 a 坐标系下的位姿），与 GTSAM `BetweenFactor` 约定一致。

## 已知限制

- ~~回环候选依赖里程计邻近检索~~ → 已由 Scan Context 全局检索解决（2026-07-20）；KITTI seq06 / 多序列下的 SC 阈值泛化仍待验证。
- KISS-ICP 未开 deskew（POSS bin 无点级时间戳）；户外 32 线数据上 2% 漂移，开 deskew 或换 LIO 可进一步降低。
- ~~RAP 验证为两路配准~~ → N 路联合配准已接入（2026-07-20）：小批次（5–7 帧）环一致约束优于 pairwise-PGO；但一次性 82 帧联合配准全局布局翘曲，尚不能替代 PGO，待分层方案。
- PGO 中回环边的信息矩阵为固定值，未按 RAP 重叠率/拟合残差自适应加权。
- 真实验证已覆盖 POSS + KITTI 00/02/05/06（2026-08-08：验收门与 SC 检索跨数据集泛化，累计零假接受）；MCD/Livox 仍未做。
- ~~密集同址冗余回环边使 PGO 略差于纯里程计（KITTI-06 live 2.777→3.221 m）~~ → 已修复（2026-08-08）：PGO 前贪心 (i,j) 空间 NMS（`nms_loop_edges`，保留 Chebyshev-w 单元内最高 overlap 边；**live 默认 w=10 开、离线默认关**——live 检索不设上限故密集簇是常态，离线有候选预算）。证据 `scripts/eval_loop_edge_nms.py`：KITTI-06 3.221→2.573 m、KITTI-02 182→38 边 26.423→26.018 m、POSS 不变；live 复测 KITTI-06 PGO **2.581 m < 里程计 2.777 m**（NMS 114→25 边），POSS 离线金标准 1.821 m 逐位不变。

## 论文对比表（P0 收尾，2026-08-22）

- `scripts/eval_paper_metrics.py`（P0）：对全部已存轨迹产物统一计算论文口径指标——evo aligned ATE RMSE（SE(3) Umeyama）+ KITTI 官方 100–800 m 分段相对漂移 ter/rer。产物 `output_paper_metrics/paper_metrics.{json,md}`（被 gitignore，可重生成）。
- `scripts/rerun_pgo_saved.py`：从已存 live 产物（TUM 里程计 + keyframes + accepted_loops）用**当前后端**重跑 PGO。KITTI-02 现行后端权威数字：unaligned APE **26.018 m** / aligned **4.035 m**（旧后端 27.387 m，精确命中 `eval_loop_edge_nms` 预测）；POSS / KITTI-06 逐位复现已存数字（回归通过）。
- `data/literature/kitti_baselines.json`（**已入版本库**，gitignore 例外 `/data/*` + `!/data/literature/`）：15 个文献方法的 KITTI 逐序列 ter/rer，全部转录自论文表格并记录来源与约定（MULLS arXiv:2102.03771 Table II、CT-ICP arXiv:2109.12979 Table I、PSF-LO arXiv:2010.13355 Table II、KISS-ICP arXiv:2209.15397 Table II）。约定坑：MULLS/CT-ICP 施加 0.2°/0.205° 内参角校正而我们没有；CT-ICP 的 AVG 是全段平均；KISS-ICP 论文只报 split 均值（0.50%）。
- `scripts/make_comparison_table.py` → `output_paper_metrics/comparison_table.{md,json}`（CPU-only 可随时重跑）：三张表——ter 逐序列（含 00–06 均值列）、rer 逐序列、aligned ATE（含回环数注记）。
- **结论要点**：ter 是局部指标，回环价值体现在 aligned ATE——00: 3.94→**2.26 m**（40/40 真回环）、02: 7.95→**4.03 m**（182 边）、05: 1.87→**1.13 m**、06: 0.89→**0.87 m**、POSS-A 2.56→**0.67 m**、POSS-B 1.92→**1.04 m**。我们 00–06 ter 均值 1.13% 弱于文献均值（前端是 POSS 调参的 KISS-ICP，非论文配置 0.50%；01 高速公路拉高均值）——论文叙事应主打回环验证精度与全局一致性。

### P1 模块级 verifier-swap（2026-08-22，`scripts/eval_verifier_swap.py`，CPU-only）

同一前端（KISS-ICP）、同一检索（**live 系统实际提出的候选集**，从节点日志/已存产物解析）、同一后端（relfloor+Huber+NMS w=10），只换验证器四臂：`noverif`（识别即闭环，恒等 T_rel 全接受）/ `icp_odom`（双尺度 point-to-point ICP 5.0→1.0 m，里程计初始化）/ `icp_scyaw`（SC 偏航提示初始化）/ `rap`（已存结果）。产物 `output_verifier_swap/`（swap_table.{md,json} + 逐候选诊断 JSON + 分臂 TUM）。

- **KITTI-02**（256 候选 = 32≤3m + 93×3–10m + 70×10–30m + 61>30m 真假）：noverif 全接受 → **地图报废 265 m**；icp_odom 与 rap 都**零/近乎零真假日接受**（>30 m：0 vs 1）——双指标门跨引擎可迁移；**RAP 边精度 ~2× 于 ICP**（中位 2.03 vs 3.58 m），aligned ATE 4.035 vs 5.005 m、ter 1.237 vs 1.342 全胜；RAP 多救回 11 条同街约束（ICP 从里程计初始化对不齐的 11–37 m 对，ICP 接受集 ⊂ RAP 接受集）；ICP unaligned 24.423 略低于 RAP 26.018——因为它更保守、少收了 11 条含噪同街边，属"少收边"而非"收得准"。
- **KITTI-06**（114 全真候选，短误差易回环）：ICP 精修略细（0.07 vs 0.08 m，APE 2.513 vs 2.581）——短初始误差下 ICP 在收敛域内；**SC 偏航提示重算后反而害 ICP 丢 4 条真回环**（SC 位移在重复结构上混叠出 318–342° 假提示），与 RC-2 结论互证。
- **POSS**（2 全真候选）：rap 1.757 m 仍最优，icp 1.774 紧咬（13 m 初始误差下 dR 小，ICP 两级收敛成功）；noverif 2.270 m——Huber+relfloor 兜住了恒等边的部分伤害。
- 回归校验：三数据集 rap 臂逐位复现 `rerun_pgo_saved.py` 基准（26.018 / 2.581 / 1.757）。

### P2 标定生成式不确定性（2026-08-23，`scripts/eval_calibrated_weighting.py`，CPU-only）

在 KITTI-02 全部 182 条已存边 + POSS 18 条边上拟合条件误差模型 p(边误差 | overlap, mean_dist, 多生成离散度)，并对 KITTI 00/05/06 做零调参迁移。产物 `output_calibrated/`（edge_dataset.json 378 边 + fit_report.{md,json}）。

- **拟合结果**：σ_t = 2.835·mean_dist^0.032 m（**近似常数 2.8 m**）、σ_r = 1.174·mean_dist^0.305 deg。特征信号量（Pearson vs ln GT 误差）：**mean_dist r=0.022、overlap r=0.044（≈零）**，多生成离散度 disp_t r=0.377（唯一有信号，但需额外采样趟数且 RC-3 已证 PGO 无增益）。
- **标定对比**（fit 集 1σ 平移覆盖率，目标 68%）：fixed/relfloor **2%（灾难性过自信）**、adaptive 28%、calibrated 86%、离散度模型 76%（同子集上 mean_dist 模型 85%）。
- **零调参迁移**：平移覆盖率 00/05/06 全部 100%（安全但保守——误差水平由候选难度决定而非传感器，形状平坦可迁移、绝对水平不可直接迁移）。
- **PGO 黄金回归**（calibrated vs relfloor，unaligned/aligned）：七黄金全部 ±5% 以内；aligned 在 00/02/06live/posslive 上更好（02: 3.78 vs 4.04 m），poss_live 1.674 vs 1.757 m。按预先声明的准则（全胜才换默认）**relfloor 保持默认**；`loop_weighting="calibrated"` + `calib_params` 已接入 `pose_graph.py`（默认不变）。
- **论文价值**：量化了"PGO 实践里固定 σ=0.2 m 的过自信程度"（2% 覆盖率）；证明 RAP 过门后边误差与其几何诊断近似无关→近常数标定 σ 就是正确模型；离散度是唯一有信号的轴但工程上不划算。属"标定 + 性质"型贡献，不必主张 PGO 精度增益。

## 路线图（RAP-CORE，2026-08-01 重构）

**核心目标：以 RAP 为核心的 SLAM（RAP-CORE）**——每个环节要么由 RAP 完成，要么围绕 RAP 的能力与输出设计，而不是把 RAP 当作可替换的验证插件。

- **RC-1 ✅ 2026-08-01 —— 后端消费 RAP 逐边诊断**：PGO 回环边信息加权 = floor-relative 自适应 σ（`relfloor`，scale = clip(mean_dist/1.0 m, 1, 3)）+ Huber(k=1) 鲁棒核（`pose_graph.py` 默认，经 `rap_server.py` / 节点 `pgo_*` 参数 / `run_pipeline.py` 全线接通）。依据 `scripts/eval_pgo_robust.py`（182 条 KITTI-02 + 2 条 POSS 已存边）：KITTI-02 APE 27.387→**26.423 m**，POSS 逐位不变 **1.757 m**；50% 边被 5 m σ 污染时鲁棒核保持 ~25 m（固定加权爆到 80 m+）。诊断→误差相关性弱（r≈0.25），故只做"只松不紧"的 floor-relative 放宽；绝对自适应（`--loop-weighting adaptive`）在 KITTI 上最优（25.688 m）但 POSS 回退 1.961 m，保留为调参项。
- **RC-2 ✅ 2026-08-01（证伪）—— SC yaw-hint → RAP 预旋转**：零收益——RAP 在实践中偏航不变（`scripts/eval_yaw_prehint*.py`），保持仅日志诊断，不接入。
- **RC-3 ✅ 2026-08-08 —— 多生成置信度**：`register_consistent(k)` 多次生成一致性（`rap_backend.py`；离线 `--multigen-k`，live 节点 `multigen_k`/`min_overlap_min`/`max_disp_r` 参数，默认 k=1 关闭）。依据 `scripts/eval_multigen_confidence.py`（195 对 × 4 代）：**验收门加固价值大**——跨代最小 overlap 干净分离真/假回环（真 0.661–0.979 vs 假 0.031–0.156，对比单代假回环曾达 0.64、边际仅 0.06），并暴露 90° 双稳态偏航翻转；**PGO σ 价值微弱**（离散度 vs 真实误差 ρ=+0.165 < mean_dist 的 +0.245，APE 增益 ≤0.1 m）；best-of-k 估计 KITTI −0.45 m 但 POSS 金标准 1.757→1.827 退化，未作默认。黄金回归：multigen k=3 POSS 4/4 / KITTI-06 12/12 全接受，默认 k=1 路径 POSS 1.821 m 逐位复现。产品：`output_multigen/`。
- **RC-4 ✅ 2026-08-01（评估：无收益，保持 pairwise）—— 分层 N 路后端**：`scripts/eval_hier_nway.py`（182 边 → 12 簇，簇内 `register_multi` 联合配准 + 簇间 PGO 缝合）。KITTI-02 分层 26.80–26.93 m vs pairwise 26.423 m（3 种子），POSS 1.763/1.808 vs 1.757（种子噪声内）；逐边精度 N 路略优（中位 2.33 vs 2.45 m）但 APE 不动——残差在无重访长段，环侧工程无法触及。结论：保持 pairwise 边；多种子方差已量化（±0.05–0.13 m）。已知约束：82 帧一次性配准全局翘曲（55 m）；≤16 帧 × ≤4000 点内布局不翘曲。
- **RC-5 ✅ 2026-08-08（证伪）—— 稀疏里程计精化**：`scripts/eval_sparse_odom_refine.py`（434 个连续关键帧对：无重访段 0–900/2000–3400 共 412 + 对照 22）。Q1：关键帧短基线上 RAP 边比 KISS-ICP 边更噪（stretch 中位 0.105 m / 0.44° vs 0.072 m / 0.09°）——"以独立噪声换相关漂移"在数值上不成立；Q2/Q3：PGO 随替换数单调变差（stride 1/2/4 = 72.98 / 51.05 / 40.20 m vs 基线 26.423 m；对照臂 26.84 m ≈ 基线）。结论：里程计边保持不动；长段残差只能靠前端升级（deskew/LIO）或更密重访。产物 `output_odom_refine/`。

已完成（保留备查）：阶段一离线管线 ✅；POSS 真实序列 8.26→1.82 m ✅；Scan Context 全局检索 ✅（+2026-08-01 流式召回重构 `b39d116`，KITTI-02 0→182 边）；N 路首轮对照 ✅（→RC-4）；阶段二 ROS 2 Step 0–3 ✅（剩 rosbag 录制、每次后端改动后 GUI 抽查）；KITTI velodyne 全 22 序列已到位 ✅（2026-08-01 合并验证，seq 06 冒烟 PGO APE 2.441 m、12/12 回环全真）；RC-2 SC 偏航预旋转 ✅（证伪：RAP 本身偏航不变，无需接入）；多数据集泛化 ✅（2026-08-08：KITTI 00 离线 12/12 全真 PGO −6.9%、05 离线 12/12 全真 PGO −34%、06 live 114/114 全真零假提案、里程计 2.777 m = 离线；发现密集同址冗余边问题，见已知限制）；回环边 NMS ✅（2026-08-08：live 默认 w=10，KITTI-06 live PGO 2.581 m < 里程计 2.777 m，POSS 金标准逐位不变）。

