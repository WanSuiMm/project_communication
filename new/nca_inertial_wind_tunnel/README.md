# Inertial NCA：推导 + 最小 wind-tunnel

**当前状态：本机四臂 seed 0 已完成各 800 次更新及全部评估；当前 Inertial RD 配方得到负向初筛结果。**

2026-10-01 接入与运行说明见 [LOCAL_INTEGRATION.md](LOCAL_INTEGRATION.md)。
先读[公开结果](../../evidence/inertial_seed0/RESULTS.md)与
[完整曲线摘要](../../evidence/inertial_seed0/summary.json)。原压缩包及其外部 CPU
检查记录保留在本地，不随仓库发布；公开证据来自这次实际执行的独立运行。

候选：

\[
V_{t+1}=B V_t+\eta R_\theta(H_t;X)-D\mathcal LH_t,\qquad H_{t+1}=H_t+V_{t+1}.
\]

研究目标不是用 NCA 拟合 PDE，而是判断 PDE 启发的离散参数化是否改善任务相关的持续空间计算。

## 先读

- `THEORY.md`：完整推导。包括精确 Schur 稳定域、有限波动尺度、Turing 条件、梯度乘积反例、局部修复条件与椭圆平衡的实际含义。
- `WIND_TUNNEL.md`：单一非 PDE 任务、四个对照、固定初筛配置与结果解释。
- [公开 RESULTS.md](../../evidence/inertial_seed0/RESULTS.md)：实际学习结果与尚未验证的命题。
- `SOURCES.md`：核对过的原始文献。特别注意 Momentum ResNet、GraphCON、RD-NCA；两行递推本身不应被宣称为新贡献。

## 运行

需要现有 Python + NumPy + PyTorch 环境。正式实验使用用户自己的 CUDA PyTorch 环境；不需要 Docker，也不需要下载数据。

```bash
python math_checks.py --out ../../runs/NEW_CHECKS/linear_checks.json
python -m unittest test_core -v
python run_wind_tunnel.py --smoke --device cpu --out ../../runs/NEW_SMOKE
```

正式第一轮（固定配置，不扫参）：

```bash
python run_wind_tunnel.py --device cuda --arms all --seed 0 --steps 800 --minutes 25 --out ../../runs/NEW_SCREEN
```

第一轮有差异后，只复核最相关的两臂：

```bash
python run_wind_tunnel.py --device cuda --arms momentum_nca inertial_rd --seed 1 --steps 800 --out ../../runs/NEW_SEED1
python run_wind_tunnel.py --device cuda --arms momentum_nca inertial_rd --seed 2 --steps 800 --out ../../runs/NEW_SEED2
```

默认 eager 模式。`--compile-cell` 可选，但必须同样用于所有对照；首次编译不计入 warm 推理延迟，训练日志单独注明包括第一次编译。提供可选开关不表示已验证编译收益。

## 文件

| 文件 | 内容 |
|---|---|
| `cells.py` | 四个可训练 cellular update；相同局部输入与无通量边界 |
| `tasks.py` | 带种子的区域身份任务；BFS 目标；损伤与证据切换 |
| `run_wind_tunnel.py` | 训练、尺寸外推、持续计算、条件修复、反事实、梯度和计时 |
| `math_checks.py` | 不训练神经网络的精确线性核验 |
| `test_core.py` | 六个实现正确性测试 |
| `../../evidence/inertial_seed0/linear_checks.json` | 本机线性核验、反例和谱时间表 |
| `../../evidence/inertial_seed0/summary.json` | 四臂参数/状态数量、全部精度曲线与时间指标 |
| `../../evidence/inertial_seed0/validation.json` | 八项测试和每臂三次更新的 GPU 冒烟摘要；不是效能比较 |

默认模型：

| Arm | 参数 | 每格 persistent 标量 | MLP 隐宽 |
|---|---:|---:|---:|
| State-matched NCA | 4,993 | 32 | 48 |
| Momentum NCA | 4,689 | 32 | 88 |
| RD NCA | 4,721 | 16 | 128 |
| Inertial RD | 4,737 | 32 | 128 |

宽度是 8 的倍数；参数近似匹配而非严格相等。最重要的 Momentum NCA 与 Inertial RD 有相同的状态量、接近的参数量，但操作和激活内存仍需实测。

## 输出边界

本机实验输出由必填 `--out` 指定，目录必须尚不存在。每个 arm/seed 独立 JSON 和 `.pt`。JSON 包含完整配置、初始/最终系数、训练曲线、资格覆盖率、修复与切换曲线、GPU/CPU 标识。

所有“修复”主指标依赖损伤前合格样本；无合格样本则为 null。本轮候选在每个尺寸均为
0/16 合格，因此修复主指标无法评价。指标中的 `distance` 是任务图距离，不能拿来替代底层网格因果距离。

代码使用普通 BPTT，没有实现可逆反传、稀疏更新或隐藏的全局 PDE 求解器。没有把 CPU 冒烟时间、谱根衰减或振荡振幅当成 GPU 加速/实际语义传输证据。
