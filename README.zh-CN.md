# usct-benchlab

[English README](README.md)

`usct-benchlab` 是一个轻量级 Python 基准测试包，用于超声计算机断层成像
（ultrasound computed tomography, USCT）重建算法的统一评估。它提供统一的
输入/输出格式、数据集转换工具、经典重建算法、FWI 结果适配器、常用指标、
预览图和 benchmark 汇总，方便对不同算法进行可复现的横向比较。

## USCT 是什么？

**USCT 本质上是一个 PDE 约束反问题。** 换句话说，换能器发射声波，声压场
在人体组织或仿体中按照声学波动方程传播，接收阵列记录时间信号；反问题的
目标是从这些接收信号中恢复介质的空间声学参数。

本仓库目前主要关注声速图 $c(x)$ 的重建。相关的物理参数还包括密度
$\rho(x)$ 和衰减 $\alpha(x)$。所有数据都会被转换成统一的 `USCTCase`
格式，所有算法输出都会保存为统一的 `ReconstructionResult`。

## 数学形式

USCT 应被理解为 PDE 驱动的反问题，而不是普通的图像重建任务。声源换能器
激发声压场，声场在未知介质中传播，接收器测量这些传播后的信号，再由这些
测量反推介质参数。

$$
\frac{1}{c(x)^2}\partial_{tt}p_s(t,x)-\Delta p_s(t,x)=q_s(t,x).
$$

在频域中，对应的 Helmholtz 形式常写为

$$
\left(\Delta+\omega^2m(x)\right)\hat p_s(\omega,x)=-\hat q_s(\omega,x).
$$

其中 $p_s$ 是声源 $s$ 对应的声压，$q_s$ 是发射源，$c(x)$ 是声速，$m(x)$
是平方慢度：

$$
m(x)=\frac{1}{c(x)^2}.
$$

本仓库中的多数声速重建方法估计的是声速图 $c(x)$，或者慢度图

$$
u(x)=\frac{1}{c(x)}.
$$

接收器 $r$ 通过测量算子观测传播后的声压场：

$$
d_{sr}(t)=\mathcal M_r p_s(t,\cdot)+\eta_{sr}(t).
$$

不同算法的核心区别在于保留了多少波动物理。FWI 在优化中保留声学 PDE 或
Helmholtz 求解，并匹配波形或复数频域压力。travel-time baseline 会先把数据
降维为到时特征，再反演射线或 eikonal 近似；它们更快、更稳定，但舍弃了相位、
幅度、衍射以及大量有限频物理。

直射线 travel-time 模型使用参考声速 $c_0$ 和固定路径 $\gamma_{sr}$：

$$
\Delta t_{sr}\approx\int_{\gamma_{sr}}\delta u(x)d\ell.
$$

慢度扰动为

$$
\delta u(x)=\frac{1}{c(x)}-\frac{1}{c_0}.
$$

像素离散化后得到

$$
A\delta u \approx b.
$$

CGLS、SIRT 和 SART 求解的都是类似下面的代数射线系统：

$$
\min_{\delta u}\|W(A\delta u-b)\|_2^2+\lambda^2\|L\delta u\|_2^2.
$$

Bent-ray 方法保留高频 travel-time 模型，路径会随当前介质变化：

$$
|\nabla T_s(x)|=u(x).
$$

接收器 travel time 近似为

$$
t_{sr}\approx T_s(r).
$$

理想化的非线性 travel-time 目标可以写为

$$
\min_c\sum_{s,r}\left|t_{sr}^{\mathrm{obs}}-T_s(r;c)\right|^2+\lambda R(c).
$$

FWI 直接使用波形或频域压力数据：

$$
\min_c
\frac{1}{2}\sum_{\omega,s,r}
\left|
\hat p_s(\omega,r;c)-\hat p_{sr}^{\mathrm{obs}}(\omega)
\right|^2
+\lambda R(c).
$$

其中 $\hat p_s(\omega,r;c)$ 不是任意图像算子，而是候选声速下由声学 PDE 或
Helmholtz solver 预测出来的压力。

| 方法 | 建模假设 | 优化目标 | 适用场景 |
| --- | --- | --- | --- |
| CGLS | 参考介质中的固定直射线；到时差在线性慢度扰动上近似。 | 对 $A\delta u\approx b$ 做加权正则化最小二乘 Krylov 求解。 | 快速、可复现的声速 baseline 和回归测试。 |
| SIRT | 与 CGLS 相同的直射线代数模型，但用同步归一化残差反投影更新。 | 通过 relaxation 和 smoothing 迭代降低 $A\delta u\approx b$ 的加权残差。 | 更重视稳定性的迭代 baseline。 |
| SART | 相同直射线模型，用发射器或射线子集做有序更新。 | 子集 row-action 更新。 | 早期收敛更快，但对排序和 relaxation 更敏感。 |
| Bent-ray | 高频 travel time 满足 eikonal 近似；射线路径随声速或慢度变化。 | 基于 $T_s(r;c)$ 的正则化非线性 travel-time mismatch。 | 无法使用完整波形反演时的折射感知 surrogate 对比。 |
| FWI | 完整声学波或 Helmholtz 传播；数据是波形或复数压力。 | 对声源、接收器和频率上的 PDE-constrained waveform mismatch 做优化。 | 有外部 k-Wave/FWI artifact 或外部 FWI 命令时的高保真汇报。 |

`bent_ray_gn` 是一个正则化的 bent-ray 风格 travel-time baseline，不是完整外部
eikonal solver。`rwave_adapter` 是 ray-Born-inspired adapter baseline，并不
声称完整复现外部 complex rWave solver。FWI 路线在本仓库中作为高保真外部
k-Wave/FWI 结果的适配器。更详细的数学说明见
[docs/math_formulation.md](docs/math_formulation.md)。

## 支持的算法

| 算法 | 注册命令 | 数学模型 | 输入要求 | 典型用途 | 配置文件 |
| --- | --- | --- | --- | --- | --- |
| CGLS | `straight_cgls` | 直射线加权最小二乘 | 带环形几何和 travel-time 测量的 `USCTCase` | 快速声速 baseline | `configs/algorithms/cgls.yaml` |
| SIRT | `straight_sirt` | 同步迭代射线层析 | 带环形几何和 travel-time 测量的 `USCTCase` | 稳健的迭代声速 baseline | `configs/algorithms/sirt.yaml` |
| SART | `straight_sart` | 有序/子集代数射线更新 | 带环形几何和 travel-time 测量的 `USCTCase` | 有序更新直射线 baseline | `configs/algorithms/sart.yaml` |
| Attenuation SIRT | `attenuation_sirt` | 直射线 log-amplitude 层析 | 带 log-amplitude 测量的 `USCTCase` | 衰减成像 baseline | `configs/algorithms/attenuation.yaml` |
| Bent-ray | `bent_ray_gn` | 正则化 bent-ray 风格 travel-time baseline | 带 travel-time 测量的 `USCTCase` | 折射风格对比方法 | `configs/algorithms/bent_ray.yaml` |
| rWave adapter | `rwave_adapter` | ray-Born-inspired adapter baseline | 带 travel-time 测量的 `USCTCase` | 波动启发式对比方法 | `configs/algorithms/rwave.yaml` |
| FWI adapter | `fwi_kwave_adapter` | PDE 层面的 full-wave inversion adapter | `USCTCase` 加外部 k-Wave/FWI 结果或命令路径 | 高保真 FWI 结果汇报 | `configs/algorithms/fwi_kwave.yaml` |
| Tiny FWI sanity | `fwi_tiny` | 小型 waveform-inversion sanity model | 小尺寸合成声速样本 | 本地 FWI 管线 sanity check | `configs/algorithms/fwi_tiny.yaml` |

更多算法说明见 [docs/algorithms.md](docs/algorithms.md)。

## 安装

使用 conda：

```bash
conda create -n usctbench python=3.10 -y
conda activate usctbench
pip install -e ".[dev,viz]"
```

或使用 pip：

```bash
pip install -r requirements.txt
pip install -e .
```

检查安装：

```bash
usct --help
usct list-algorithms
pytest -q
```

如果只想快速跑通一个端到端示例，并且把生成文件都写到 `/tmp`，可以运行：

```bash
bash examples/synthetic_quickstart.sh
```

## 环境变量和工作区布局

建议用环境变量管理数据和输出，避免把数据、运行结果或外部工程提交到 Git：

```bash
export USCT_WORKSPACE=/path/to/usct-benchlab
export USCT_DATA_ROOT=$USCT_WORKSPACE/data/openbreastus
export USCT_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
export USCT_NBP_ZIP_PATH=/path/to/NBPslices2D.zip
export USCT_KWAVE_FWI_RESULT_PATH=/path/to/fwi_result.mat
export USCT_KWAVE_ROOT=/path/to/external/USCT_kwave
export USCT_KWAVE_PYTHON_BIN=/path/to/python
```

推荐工作区结构：

```text
<workspace>/
  code/          # 本仓库
  data/          # 本地数据集和转换后的 case
  runs/          # benchmark 输出
  external/      # 可选外部工程
  checkpoints/   # 本地权重或 checkpoint
```

`scripts/setup_workspace.sh` 可以创建这套目录和仓库内的轻量 symlink；它不会
把数据集复制进 Git。

## 准备数据

合成 demo：

```bash
usct data make-synthetic-smoke \
  --out "$USCT_WORKSPACE/data/synthetic_demo" \
  --shape 48 \
  --n-transducers 48
```

OpenBreastUS：

```bash
usct data inspect-openbreastus \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_RUN_ROOT/openbreastus_index.json"

usct data make-quality \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_WORKSPACE/data/openbreastus_demo" \
  --cases-per-density 1 \
  --converted-shape 256 \
  --n-transducers 128
```

NBPslice2D：

```bash
usct data inspect-nbpslice2d \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_RUN_ROOT/nbpslice2d_index.json"

usct data make-nbp-quality \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_WORKSPACE/data/nbpslice2d_demo" \
  --cases-per-type 1 \
  --converted-shape 256 \
  --n-transducers 128
```

完整流程见 [docs/usage.md](docs/usage.md) 和
[docs/datasets.md](docs/datasets.md)。

## 运行单个算法

CGLS：

```bash
usct run straight_cgls \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/cgls.yaml \
  --out runs/single_cgls
```

SIRT：

```bash
usct run straight_sirt \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sirt.yaml \
  --out runs/single_sirt
```

SART：

```bash
usct run straight_sart \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sart.yaml \
  --out runs/single_sart
```

Bent-ray：

```bash
usct run bent_ray_gn \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/bent_ray.yaml \
  --out runs/single_bent_ray
```

rWave adapter：

```bash
usct run rwave_adapter \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/rwave.yaml \
  --out runs/single_rwave
```

FWI adapter：

```bash
usct run fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/fwi_kwave.yaml \
  --out runs/single_fwi
```

如果 FWI adapter 需要读取已有重建结果，请设置
`USCT_KWAVE_FWI_RESULT_PATH`。可读取的 artifact 必须包含 `VEL_ESTIM`；
可选字段 `C_INTERP`、`VEL_ESTIM_ITER` 和 `LOSS_ITER` 会启用 ground-truth
指标和迭代选择。

## 运行 benchmark

demo benchmark 会读取下面这些可选 case glob：

```bash
export USCT_SYNTHETIC_CASE_GLOB="$USCT_WORKSPACE/data/synthetic_demo/cases/*.h5"
export USCT_NBP_CASE_GLOB="$USCT_WORKSPACE/data/nbpslice2d_demo/cases/*.h5"
export USCT_OPENBREASTUS_CASE_GLOB="$USCT_WORKSPACE/data/openbreastus_demo/cases/*.h5"
export USCT_KWAVE_FWI_CASE_GLOB="$USCT_WORKSPACE/data/fwi_kwave_demo/cases/*.h5"
```

运行 benchmark：

```bash
usct bench --suite configs/benchmarks/synthetic_demo.yaml
usct bench --suite configs/benchmarks/nbpslice2d_demo.yaml
usct bench --suite configs/benchmarks/openbreastus_demo.yaml
usct bench --suite configs/benchmarks/fwi_kwave_demo.yaml
```

## 输出文件

单算法运行会写出：

```text
runs/single_cgls/synthetic_circular_sos/result.h5
runs/single_cgls/synthetic_circular_sos/metrics.json
runs/single_cgls/synthetic_circular_sos/metadata.yaml
runs/single_cgls/synthetic_circular_sos/preview.png
```

benchmark 会写出：

```text
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/result.h5
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/metrics.json
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/metadata.yaml
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/preview.png
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/benchmark_summary.csv
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/benchmark_report.md
```

`metrics.json` 保存每个 case 的图像指标和数据一致性指标；
`metadata.yaml` 记录算法、配置路径、case id、运行时间、状态和测量来源。

## 示例结果

OpenBreastUS 四类样本对比：

![OpenBreastUS FWI and baseline comparison](docs/assets/openbreastus_readme_fwi_vs_surrogate.png)

NBPslice2D，2D Acoustic Numerical Breast Phantoms for USCT：

![NBPslice2D FWI and baseline comparison](docs/assets/nbpslice2d_readme_fwi_vs_surrogate.png)

不同算法使用的测量假设不同，结果解读应结合
[docs/algorithms.md](docs/algorithms.md) 和每个 case 的 metadata。

## 常见问题

- `algorithm not found`：运行 `usct list-algorithms`，检查注册命令名。
- 缺少 `.h5` 或 `.mat` 数据：确认数据转换命令已完成，并检查相关环境变量是否指向存在的路径。
- FWI 结果路径不存在：设置 `USCT_KWAVE_FWI_RESULT_PATH`，或修改
  `configs/algorithms/fwi_kwave.yaml` 指向需要读取的结果。
- 输出出现 NaN/Inf：查看 `failure_report.md`，检查 case 单位，并尝试降低迭代次数或 relaxation。
- glob 没有匹配到 case：打印展开后的 `USCT_*_CASE_GLOB`，确认转换后的 case 位于 `data/.../cases/`。
- 缺少 `matplotlib` 或 `scikit-image`：运行 `pip install -e ".[viz]"`。

## 开发

```bash
black src tests scripts
ruff check src tests scripts --fix
python -m compileall src tests
pytest -q
bash scripts/run_smoke.sh
python scripts/audit_release.py
```

更多 release 检查和仓库卫生规则见 [docs/development.md](docs/development.md)。

## 引用 / 数据集

如果在实验中使用了 OpenBreastUS、NBPslice2D、k-Wave 或
WaveformInversionUST，请引用相应数据集和外部工具。参考文献见
[docs/references.bib](docs/references.bib)。

## 许可证

本仓库使用 MIT License 发布。见 [LICENSE](LICENSE)。
