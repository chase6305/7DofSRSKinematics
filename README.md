# 7DofSRSKinematics

中文 | [English](README_EN.md)

基于 NumPy 的 7 自由度球肩—旋转肘—球腕（S-R-S）机械臂几何解析运动学库，仓库内提供可直接使用的 KUKA LBR iiwa 7 R800 模型。

## 功能

- 7 自由度 S-R-S 机械臂解析正运动学与逆运动学。
- 完整枚举肩、肘、腕的 8 种离散构型分支。
- 支持连续臂角冗余参数与指定构型精确求解。
- 支持周期关节限位映射和基于 seed 权重的最近解选择。
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

## Viser 仿真

```bash
python examples/viser_demo.py --port 8080
```

Viser 默认使用连续快速模式；可通过 `--search-mode global` 对比完整全局搜索。

访问 `http://localhost:8080`。关节滑块驱动 FK；拖动橙色 TCP 控制器执行 IK；“Null-space motion”面板可以播放保持 TCP 不动的臂角自运动。状态面板会显示当前构型、臂角、位姿残差、求解耗时、场景更新 FPS，以及启动基准测试的 median/p95 延迟。

无界面性能测试：

```bash
python examples/viser_demo.py --validate-only --validation-samples 100
```

零空间与固定 TCP 臂角扫描：

```bash
python examples/null_space_demo.py --samples 25 --range-deg 35
```

## 测试

```bash
pytest -q
```

## 算法与参考文献

解析分解首先确定腕部中心，然后求解肩—肘—腕三角形；通过臂角 ψ 旋转参考肩平面，最后分别分解肩部和腕部球形关节。`get_ik` 会搜索全部 8 个离散分支，并将满足关节限位的周期等价解按照与 seed 的加权距离排序。

- Shimizu 等，《Analytical Inverse Kinematic Computation for 7-DOF Redundant Manipulators With Joint Limits and Its Application to Redundancy Resolution》，IEEE T-RO，2008，[DOI](https://doi.org/10.1109/TRO.2008.2003266)。
- Faria 等，《Position-based kinematics for 7-DoF serial manipulators with global configuration control, joint limit, and singularity avoidance》，MMT，2018，[DOI](https://doi.org/10.1016/j.mechmachtheory.2017.10.025)。

开发规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

机器人 mesh 来源、再生成方法和 BSD-3-Clause 资产许可证见 [urdf/README.md](urdf/README.md) 与 [urdf/ASSET_LICENSE](urdf/ASSET_LICENSE)。
