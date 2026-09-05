# 7DofSRSKinematics

中文 | [English](README_EN.md)

基于 NumPy 的 7 自由度球肩—旋转肘—球腕（S-R-S）机械臂几何解析运动学库，仓库内提供可直接使用的 KUKA LBR iiwa 7 R800 模型。

## 功能

- 7 自由度 S-R-S 机械臂解析正运动学与逆运动学。
- 完整枚举肩、肘、腕的 8 种离散构型分支。
- 支持连续臂角冗余参数与指定构型精确求解。
- 支持周期关节限位映射和基于 seed 权重的最近解选择。
- NumPy 批量臂角搜索与解析几何雅可比，支持固定 TCP 的零空间运动。
- 提供输入、可达性和数值结果校验。
- 提供 Viser 交互式 FK/IK 仿真与性能监控面板。
- 核心仅依赖 NumPy，可视化依赖保持可选。
- R800 视觉资产使用带橙/灰面颜色的 GLB，碰撞资产保留来源 STL。

## 安装

```bash
python -m pip install -e .
python -m pip install -e '.[test,visualization]'  # 开发和仿真环境
```

## 快速开始

```python
import numpy as np
from kuka_iiwa_solver import KUKAiiwaSolver

solver = KUKAiiwaSolver()
seed = np.zeros(7)
target = solver.get_fk(np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]))
success, joints = solver.get_ik(target, seed, num_samples=73)
```

默认 `search_mode="continuous"` 会优先保持 seed 的 S/E/W 构型和真实臂角，适合实时控制。需要完整枚举全部构型和臂角样本时使用：

```python
success, joints = solver.get_ik(
    target, seed, num_samples=73, search_mode="global"
)
```

`get_configuration(q)` 返回肩、肘、腕构型符号和连续臂角。将结果传入 `solve_configuration(target, configuration, seed)` 可以在指定分支和臂角上求解。

`solve_configurations(target, configurations, seed)` 可以为同一个目标和 seed 批量求解多个精确构型。返回 `(success, joints)`，形状分别为 `(N,)` 和 `(N, 7)`，保持输入顺序，无效行置零，使用前必须检查对应的成功标志。每个肘分支只计算一次几何系数，臂角与 FK 校验批量执行；适合采样自运动流形。

连续查询同一 TCP 时，单构型求解、批量构型求解和 NumPy IK 搜索复用最近目标的几何系数，最多保留两个肘分支。缓存按局部目标、DH 参数和连杆长度的实际内容匹配，不会将相邻目标近似为同一个目标；种子、限位、权重和可行性校验仍在每次调用中生效。固定 TCP 自运动通常受益，持续变化的目标会重建系数并承担少量缓存检查开销。

## 接口与数值约定

- `get_ik` 和 `solve_configuration` 无解时返回 `(False, None)`；位姿必须是有限的刚体齐次变换，关节数组形状必须为 `(7,)`，非法输入会抛出异常。
- `continuous` 优先返回 seed 构型附近的可行解，局部搜索失败后再全局采样；`global` 在 `[-π, π)` 上均匀采样全部八种分支，按 `||(q - seed) * weight||₂` 选择样本中的最近解。有限采样不保证找到连续最优解或非常窄的可行区间。
- 单次调用的 `num_samples` 不再修改求解器默认值；默认采样数通过 `set_iteration_params(num_samples=...)` 设置，至少为 2。兼容旧参数名 `num_sample`，其他未知 IK 参数会报错。
- 所有返回的 IK 解（包括 seed 快速路径）都满足关节限位和 `set_elbow_up(True)` 的正肘分支约束。周期限位映射逐关节计算，不枚举周期值的笛卡尔积。
- `set_tcp(T)` 设置法兰到 TCP 的变换，参与 FK、IK 和雅可比计算。`get_fk` 与 `get_all_fk_mat` 均返回世界坐标系位姿；后者依次包含七个 DH 连杆坐标系和 TCP。
- `get_fk(q, index=-1)` 对 `(7,)` 输入返回 `(4, 4)`，对 `(N, 7)` 输入返回 `(N, 4, 4)`；`index=-1` 为 TCP，`0..6` 为指定 DH 连杆。批量调用支持空批次和非连续数组，仅计算所需连杆之前的变换，每次调用独立持有输出。批量构型 IK 的末端残差校验使用此接口，避免分配整条连杆历史。
- `get_all_fk_mat(q)` 对 `(7,)` 输入保留八个矩阵的列表返回值；对 `(N, 7)` 输入返回 `(N, 8, 4, 4)` 数组。批量路径向量化计算 DH 变换，支持空批次、非连续数组，每次调用独立持有输出。
- `get_jacobian(q)` 返回世界坐标系下的 `6×7` 解析雅可比，前三行为 TCP 线速度，后三行为角速度。保留 `step` 参数以兼容旧调用，但不再进行有限差分。
- 肩或腕奇异处联合求解两个外侧关节，在限位内最小化与 seed 的加权距离；肩腕中心重合时臂角无定义，`get_arm_angle` 抛出 `ValueError`。库导入时不修改应用的全局日志配置。

## 与 Warp 后端对照

两个仓库可以分别安装；wheel 包含各自的 URDF 与视觉、碰撞资源。修改求解公式时，请同时运行本仓库测试与 Warp 仓库的 `test/test_backend_consistency.py`。后者在两个仓库并列放置时自动运行交叉对照，也可通过 `SRS_NUMPY_REPO` 指定本仓库路径。

固定 TCP 扫描、移动 TCP 轨迹和 584 点流形批量求解可单独计时；脚本先预热，再报告多次重复的中位数，同时校验所有结果的 TCP 残差：

```bash
python tools/benchmark_srs.py --frames 180 --repeats 15 --output benchmark.json
```

此基准只需要核心依赖，`--output` 写入一个新 JSON 文件。精确构型求解在 Warp 仓库中也使用 CPU 几何路径；该脚本不用于衡量 GPU 全局批量 IK 的吞吐量。

```bash
python tools/compare_backends.py --warp-repo ../7DofSRSKinematicsWarp --samples 40
```

此工具需要 Warp 仓库的依赖，使用相同目标、限位和臂角采样，对照 NumPy 标量、NumPy 批量回退及 Warp 内核的 FK、解析雅可比、全局 IK 与残差；编译预热不计入耗时。默认在 CPU 上运行，可用 `--device cuda` 测量 Warp CUDA。

## Viser 仿真

SRS 实验室包含五个可切换示例：

```bash
python examples/viser_srs_lab.py --demo arm-angle --port 8081
python examples/viser_srs_lab.py --demo branches --port 8082
python examples/viser_srs_lab.py --demo jacobian --port 8083
python examples/viser_trajectory_demo.py --port 8084
python examples/viser_manifold_demo.py --port 8085
```

安装 `.[visualization]` 后也可使用 `srs-viser-lab` 命令；五个模式在同一个页面内通过 Experiment 切换。两个仓库的可视化安装均固定使用已验证的 Viser 1.0.24。

| 模式 | 可观察的行为 |
|:--|:--|
| `arm-angle` | 固定 TCP 扫描臂角，显示肩 S、肘 E、腕 W、肩腕轴和理论肘部圆；受限位排除的臂角保留最后一个可行姿态并提示。 |
| `branches` | 同时展示八种 S/E/W 构型、可行状态和最小关节余量；每个机器人仅为展示而平移，其局部目标与臂角相同。 |
| `jacobian` | 显示线速度椭球与关节居中偏好速度；播放时将居中速度投影到完整雅可比零空间，每步重新精确求解以保持 TCP。 |
| `trajectory` | 直线、圆和 8 字 TCP 路径跟踪。灰线为参考路径，绿线为 FK 实际轨迹；保持末端姿态和构型，显示误差与最大关节步长。 |
| `manifold` | 固定 TCP 下八条自运动分支在 J1/J3/J5 关节空间中的投影，显示可行样本比例和当前关节点；无解样本与关节跳变处断开。 |

轨迹模式的 Cartesian path 选择路径，Path plane 选择世界 XY/XZ/YZ 或 TCP 局部 XY 平面，Path size 设置尺寸，Joint speed bound 限制每步关节变化。TCP 平面以起点姿态为准。Play 开始或暂停，重新选择路径、平面或尺寸从当前 TCP 重新起步；逼近约束时会减少路径推进量，无法继续时停在最后可行姿态。频率是参考值，减速时实际沿路径运动的频率会降低。右侧放大六倍的位移图使用所选平面的坐标轴；暂停后可拖动 TCP 控制器。

流形模式中，每条分支按臂角采样，Samples / branch 控制分辨率；Projection axes 可切换 J1/J3/J5、肩部 J1/J2/J3、腕部 J5/J6/J7 或 J2/J4/J6 投影。臂角滑块和 Play 驱动左侧机器人及右侧关节点；固定曲线在目标、采样数或投影改变时更新。投影坐标是弧度，绘图缩放为 0.22；投影曲线相交不代表七维关节配置相同，采样图也不保证发现所有狭窄可行区间。

所有模式的自动播放均遵守 Joint speed bound，以每帧实际关节差值除以时间计算，不对关节差值取模。大幅度臂角扫描和低速轨迹会自动减少参考相位推进量；最多尝试 12 个逐次减半的步长，仍无法前进时暂停并保留最后可行姿态与相位。流形播放遇到不可行区间时不会继续扫描到另一端再跳过去。手动修改关节、构型和臂角用于选择姿态，不受播放速度限制；调整扫描幅度会以当前臂角为新中心。

手动暂停与恢复保留扫描相位、中心及运动方向；修改姿态、构型、臂角、扫描幅度或实验模式才开始新的扫描。

命令行也可设置 `--path`、`--plane`、`--path-size`、`--preset`、`--frequency`、`--joint-speed` 和 `--amplitude`。以下示例可直接打开工具坐标系轨迹，或不安装 Viser 就导出可复现的数据：

```bash
python examples/viser_trajectory_demo.py --plane tool-xy --path circle --path-size 0.015 --autoplay
python examples/viser_trajectory_demo.py --plane world-xz --path circle --path-size 0.015 --frames 300 --fps 30 --export-csv trajectory.csv
python examples/viser_manifold_demo.py --frames 300 --fps 30 --export-csv self_motion.csv
```

CSV 记录初始姿态和每个实际接受的运动步，包括模拟时间、7 个关节角、世界坐标系中的实际/目标 TCP 位置及旋转矩阵、位置/姿态误差、关节限位裕量、峰值关节速度、相位、臂角和 S/E/W 分支。角度以弧度、位置以米表示，旋转矩阵按行展开。`--fps` 为 20–240，导出使用固定模拟步长 `1/fps`，与计算耗时无关。五种模式均可导出当前选中机器人的运动；提前暂停时只保留已接受的步，JSON 报告包含原因且退出码为 2。已有 CSV 不会被覆盖；`--validate-only` 与 `--export-csv` 互斥。

拖动 TCP 控制器可以选择新目标，关节滑块和 Pose preset 可以重设姿态。Play 播放臂角扫描或关节居中运动；`--autoplay` 自动播放，`--no-mesh` 仅显示几何骨架。默认只监听 `127.0.0.1`。无客户端操作且暂停时不重复求解或刷新场景。

同一控件的连续拖动样本在入队时合并；不同控件、重置与模式切换之间保留操作顺序，避免合并事件改变 TCP 目标。连杆位姿在场景与诊断面板间复用，自运动复用上一帧的雅可比；八构型的 IK 与连杆位姿按目标和臂角缓存。自动停止时立即刷新最终状态。

橙色肘部圆表示几何轨迹，整圆并不一定满足关节限位。线速度椭球表示单位关节速度范数下的 TCP 平移速度，绘图缩放为 0.25，**不约束姿态速度**；实际自运动使用完整 `6×7` 雅可比。居中是保持当前目标与分支的局部下降示例，不保证全局最优。

五个实验的无界面数值验证只需要核心依赖；检查限位、位置与旋转残差、居中目标下降、流形样本与轨迹跟踪，失败时返回非零退出码：

```bash
python examples/viser_srs_lab.py --validate-only --validation-frames 40
```

原 FK/IK 性能监控示例仍可运行：

```bash
python examples/viser_demo.py --port 8080
```

Viser 默认使用连续快速模式；可通过 `--search-mode global` 对比完整全局搜索。

访问 `http://localhost:8080`。关节滑块驱动 FK；拖动橙色 TCP 控制器执行 IK；“Null-space motion”面板可以播放保持 TCP 不动的臂角自运动。状态面板会显示当前构型、臂角、位姿残差、求解耗时、场景更新 FPS，以及启动基准测试的 median/p95 延迟。

无界面性能测试只依赖 NumPy，并按指定搜索模式运行：

```bash
python examples/viser_demo.py --validate-only --validation-samples 100
python examples/viser_demo.py --validate-only --validation-samples 100 --search-mode global
```

测试使用固定随机种子和扰动后的关节 seed，报告成功数、median/p95 耗时与位姿残差；出现求解失败时返回非零退出码。

零空间与固定 TCP 臂角扫描：

```bash
python examples/null_space_demo.py --samples 25 --range-deg 35
```

## 测试

```bash
pytest -q
```

## 算法与参考文献

解析分解首先确定腕部中心，然后求解肩—肘—腕三角形；通过臂角 ψ 旋转参考肩平面，最后分别分解肩部和腕部球形关节。全局搜索为每个肘分支复用几何系数，批量计算臂角候选，将满足关节限位的周期等价解按与 seed 的加权距离排序，再验证返回解的 FK 残差。

- Shimizu 等，《Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to Redundancy Resolution》，IEEE T-RO，2008，[DOI](https://doi.org/10.1109/TRO.2008.2003266)。
- Faria 等，《Position-based kinematics for 7-DoF serial manipulators with global configuration control, joint limit, and singularity avoidance》，MMT，2018，[DOI](https://doi.org/10.1016/j.mechmachtheory.2017.10.025)。

开发规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

机器人 mesh 来源、再生成方法和 BSD-3-Clause 资产许可证见 [urdf/README.md](urdf/README.md) 与 [urdf/ASSET_LICENSE](urdf/ASSET_LICENSE)。
