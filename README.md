# 7DofSRSKinematics

中文 | [English](README_EN.md) · [Warp 后端](https://github.com/chase6305/7DofSRSKinematicsWarp)

基于 NumPy 的 7 自由度 S-R-S 机械臂解析运动学，内置 KUKA iiwa 7 R800。

![固定 TCP 自运动与 8 字轨迹跟踪实录](docs/media/demo.gif)

固定 TCP 自运动 → 8 字轨迹跟踪，10 秒 NumPy 实录。

- 解析 FK / IK、八种肩肘腕构型与连续臂角控制。
- 批量 FK、精确构型求解、解析雅可比与零空间运动。
- 核心仅依赖 NumPy；交互演示和机器人资产随安装包提供。

## 运行演示

```bash
python -m pip install -e '.[visualization]'
srs-viser-lab --autoplay
```

打开 **http://127.0.0.1:8081**。通过 Experiment 切换实验，Play 暂停/继续；
拖动 TCP 控制器可选择新目标。右侧显示实际位姿误差和关节限位余量。

| 实验 | 看点 |
|:--|:--|
| Arm angle | TCP 固定，肘部沿圆周自运动 |
| Eight branches | 八种 S/E/W 构型的机器人模型并列展示 |
| Jacobian | 速度椭球与固定 TCP 的关节居中 |
| TCP trajectory | 直线、圆、8 字路径与实际轨迹对照 |
| Self-motion manifold | 固定 TCP 的关节空间分支曲线 |

示例：`srs-viser-lab --demo trajectory --autoplay`。
[更多运行方式、截图与 CSV 导出 →](docs/demos.md)

## Python API

只使用求解器：`python -m pip install -e .`。

```python
import numpy as np
from kuka_iiwa_solver import KUKAiiwaSolver

solver = KUKAiiwaSolver()
target = solver.get_fk(np.array([0.3, -0.5, 0.7, 1.0, -0.4, 0.8, -0.2]))
ok, joints = solver.get_ik(target, np.zeros(7), num_samples=73)
if ok:
    print(joints)  # (7,), radians
```

默认连续搜索优先保持 seed 附近的构型；`search_mode="global"` 枚举臂角样本。
有限采样不保证找到所有可行解。[批量接口与数值约定 →](docs/api.md)

## 文档与验证

- [交互实验与录制](docs/demos.md) · [API](docs/api.md)
- [测试、基准与后端对照](docs/development.md) · [贡献指南](CONTRIBUTING.md)
- [R800 资产、PBR 材质与 iiwa14 参考分析](docs/assets.md) · [资产许可证](urdf/ASSET_LICENSE)

```bash
python -m pip install -e '.[test]'
python -m pytest -q
srs-viser-lab --validate-only
```
