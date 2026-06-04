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

对声源 $s$，一个简单的无损声压模型可以写为

$$
\frac{1}{c(x)^2}\partial_{tt}p_s(t,x)-\Delta p_s(t,x)=q_s(t,x).
$$

更一般的模型可以包含密度和衰减：

$$
\begin{aligned}
\frac{1}{c(x)^2}\partial_{tt}p_s
-\nabla\cdot\left(\frac{1}{\rho(x)}\nabla p_s\right)
+\mathcal A_\alpha[p_s]
&= q_s.
\end{aligned}
$$

接收器 $r$ 通过测量算子观测声压场：

$$
d_{sr}(t)=\mathcal M_r p_s(t,\cdot)+\eta_{sr}(t).
$$

因此，USCT 反问题可以概括为：

$$
\text{recover } c(x),\rho(x),\alpha(x)
\quad
\text{from}
\quad
\{d_{sr}(t)\}_{s,r}.
$$

本仓库的主要重建目标是 $c(x)$。

直射线 travel-time 模型使用参考声速 $c_0$ 和射线路径 $\gamma_{sr}$：

$$
\Delta t_{sr}
\approx
\int_{\gamma_{sr}}
\left(\frac{1}{c(x)}-\frac{1}{c_0}\right)d\ell.
$$

离散化后得到

$$
A\delta s \approx b,
\qquad
\delta s = \frac{1}{c}-\frac{1}{c_0}.
$$

CGLS、SIRT 和 SART 求解的都是类似下面的代数射线系统：

$$
\min_{\delta s}
\|W(A\delta s-b)\|_2^2+\lambda^2R(\delta s).
$$

对折射修正的 bent-ray travel-time，可以用 eikonal 模型描述：

$$
|\nabla T_s(x)| = \frac{1}{c(x)},
\qquad
t_{sr}\approx T_s(r).
$$

`bent_ray_gn` 是一个正则化的 bent-ray 风格 travel-time baseline，
不是完整外部 eikonal solver 的复现。

弱散射或 ray-Born 模型可以示意为

$$
\delta \hat p_{sr}(\omega)
\approx
\int_\Omega
G_0(\omega,r,x)K_\omega(x)G_0(\omega,x,s)\delta m(x)\,dx.
$$

`rwave_adapter` 是一个 ray-Born-inspired adapter baseline，
并不声称完整复现外部 complex rWave solver。

FWI 直接使用波形或频域压力数据：

$$
\min_c
\frac{1}{2}\sum_{\omega,s,r}
\left|
\hat p_s(\omega,r;c)-\hat p_{sr}^{\mathrm{obs}}(\omega)
\right|^2
+\lambda R(c).
$$

FWI 路线在本仓库中作为高保真外部 k-Wave/FWI 结果的适配器。更详细的数学
说明见 [docs/math_formulation.md](docs/math_formulation.md)。

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

## 环境变量和工作区布局

建议用环境变量管理数据和输出，避免把数据、运行结果或外部工程提交到 Git：

```bash
export USCT_WORKSPACE=/path/to/usct-benchlab
export USCT_DATA_ROOT=$USCT_WORKSPACE/data/openbreastus
export USCT_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
export USCT_NBP_ZIP_PATH=/path/to/NBPslices2D.zip
export USCT_KWAVE_FWI_RESULT_PATH=/path/to/fwi_result.mat
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
`USCT_KWAVE_FWI_RESULT_PATH`。

## 运行 benchmark

```bash
usct bench --suite configs/benchmarks/synthetic_demo.yaml
usct bench --suite configs/benchmarks/nbpslice2d_demo.yaml
usct bench --suite configs/benchmarks/openbreastus_demo.yaml
usct bench --suite configs/benchmarks/fwi_kwave_demo.yaml
```

demo benchmark 会读取下面这些可选 case glob：

```bash
export USCT_SYNTHETIC_CASE_GLOB="$USCT_WORKSPACE/data/synthetic_demo/cases/*.h5"
export USCT_NBP_CASE_GLOB="$USCT_WORKSPACE/data/nbpslice2d_demo/cases/*.h5"
export USCT_OPENBREASTUS_CASE_GLOB="$USCT_WORKSPACE/data/openbreastus_demo/cases/*.h5"
export USCT_KWAVE_FWI_CASE_GLOB="$USCT_WORKSPACE/data/fwi_kwave_demo/cases/*.h5"
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
black src tests
ruff check src tests --fix
python -m compileall src tests
pytest -q
bash scripts/run_smoke.sh
bash scripts/audit_release.py
```

更多 release 检查和仓库卫生规则见 [docs/development.md](docs/development.md)。

## 引用 / 数据集

如果在实验中使用了 OpenBreastUS、NBPslice2D、k-Wave 或
WaveformInversionUST，请引用相应数据集和外部工具。参考文献见
[docs/references.bib](docs/references.bib)。
