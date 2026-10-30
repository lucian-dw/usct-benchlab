# usct-benchlab

[English](README.md)

**面向二维超声计算机断层成像（USCT）的数值重建与基准评测库。**

`usct-benchlab` 从走时与复数声压观测重建声速分布，通过统一的样本格式连接物理算子、迭代求解器、数据准备和结果评估。研究人员与反问题智能体（Agent）可以通过命令行或 Python API 调用同一套算法。

项目由两个职责互补的仓库协作完成：

| 仓库 | 职责 |
| --- | --- |
| **usct-benchlab** | 算法发现、射线与 Born 重建、统一数据处理、计算预算和批量基准评测 |
| [**WaveformInversionUST**](https://github.com/lucian-dw/WaveformInversionUST) | 独立运行的 MATLAB/CUDA 频域 FWI 运行时，通过 `fwi_wust` 接入 BenchLab |

首次使用可从[安装与快速开始](#安装与快速开始)入手；阅读源码可查看[仓库结构](#仓库结构)；自动化调用可查看 [Agent 接入](#agent-接入)。

## 支持算法

所有算法输出单位为 m/s 的声速图。直线射线方法估计慢度扰动 $\delta s=1/c-1/c_0$，其中 $c_0$ 为参考声速。Bent-ray 与 WUST 使用慢度 $s=1/c$，Born 使用平方慢度 $m=1/c^2$。

| 算法 ID | 输入观测 | 物理模型与求解方法 | 配置示例 |
| --- | --- | --- | --- |
| `straight_cgls` | 相对水参考的差分走时 | Siddon 直线投影；采用投影 CG 或 Huber IRLS 的有界最小二乘 | [cgls.yaml](configs/algorithms/cgls.yaml) |
| `straight_sirt` | 相对水参考的差分走时 | 直线射线模型；归一化同时迭代更新 | [sirt.yaml](configs/algorithms/sirt.yaml) |
| `straight_sart` | 相对水参考的差分走时 | 直线射线模型；有序子集更新，每次迭代完整遍历全部子集 | [sart.yaml](configs/algorithms/sart.yaml) |
| `bent_ray_gn` | 绝对首波走时或相对水参考的首波走时 | 快速行进程函模型；离散切线/伴随与有界 Gauss–Newton | [bent_ray.yaml](configs/algorithms/bent_ray.yaml) |
| `born` | 具有源校准信息的复数总声压 | 有限频 Born 模型；固定背景或逐步重新线性化 | [born.yaml](configs/algorithms/born.yaml) |
| `fwi_wust` | 复数总声压 | 通过 WUST 执行带频率延拓的 Helmholtz 全波形反演 | [fwi_wust.yaml](configs/algorithms/fwi_wust.yaml) |

Born 提供四种物理变体：

| 背景模型 | 固定背景 | 重新线性化的 Gauss–Newton |
| --- | --- | --- |
| Eikonal/WKB 相位与输运振幅 | `wkb_fixed` | `wkb_nonlinear` |
| 体积分 Full-Green 模型 | `full_green_fixed` | `full_green_nonlinear` |

WKB 采用连续模型下的 Born 灵敏度近似；Full-Green 求解离散体积分模型并使用对应导数。Born 的源校准信息来自 `source_spectrum`、独立的 `water_reference`，或显式选择的单位源模型。Full-Green 支持 SciPy CPU 与 CuPy GPU 执行。

各方法的目标函数、正则化和导数定义见[算法指南](docs/algorithms.md)与[数学表述](docs/math_formulation.md)。


## 重建表现

以下保留的对比图展示 OpenBreastUS 的 HET/FIB/FAT/EXD 和 NBPslice2D 的 A/B/C/D。
它们是已有实验的保存结果，不是当前提交重新运行的结果。默认 YAML 是可运行示例，
不是这些主图的完整复现配置。

### 匹配模型示例

直线方法使用直线 ToF，Bent 使用 Eikonal ToF，固定 Born 使用匹配的复数声压。
FWI 为单独保存的高频参考结果；这些列并非使用同一种观测模型。

![OpenBreastUS 匹配模型重建](docs/assets/reconstruction/openbreastus_matched.png)

![NBPslice2D 匹配模型重建](docs/assets/reconstruction/nbpslice2d_matched.png)

### k-Wave 采集示例

五种原生方法使用重新生成的 128 发射/128 接收物体与水参考采集，按模型分别使用
ToF 或复数声压。保存的 FWI 参考具有不同的采集历史、频率计划和验证策略；
这些图用于展示，不构成严格控制条件下的统一排名。

![OpenBreastUS k-Wave 重建](docs/assets/reconstruction/openbreastus_kwave.png)

![NBPslice2D k-Wave 重建](docs/assets/reconstruction/nbpslice2d_kwave.png)

[图像来源与解读](docs/readme_results.md) ·
[保存的指标与停止原因](docs/assets/reconstruction/metrics.csv)。
早期对比图亦保留：
[OpenBreastUS](docs/assets/openbreastus_readme_fwi_vs_surrogate.png)、
[NBPslice2D](docs/assets/nbpslice2d_readme_fwi_vs_surrogate.png)。

## 仓库结构

一次重建包含四个步骤：准备样本与配置、构造物理模型、求解图像、保存并评估结果。源码按这些职责组织。

| 位置 | 职责 | 关键入口 |
| --- | --- | --- |
| [src/usctbench/](src/usctbench/) | 公开 Python API 与命令行入口 | [__init__.py](src/usctbench/__init__.py)、[cli.py](src/usctbench/cli.py) |
| [core/](src/usctbench/core/) | 样本类型、算法目录、参数、预算与停止条件 | `schema.py`、`registry.py`、`algorithm_specs.py`、`config.py` |
| [data/](src/usctbench/data/) | 数据读取、轴与单位、声压频谱、走时及源校准 | `waveforms.py`、`arrival.py`、`pressure.py`、`calibration.py` |
| [algorithms/](src/usctbench/algorithms/) | 各算法的准备过程、模型变量与结果组装 | `straight.py`、`bent_ray.py`、`born.py` |
| [operators/](src/usctbench/operators/) | 物理正演、伴随、线性化与模型空间 | `straight_ray.py`、`eikonal.py`、`ray_born.py`、`volume_integral.py`、`model_space.py` |
| [solvers/](src/usctbench/solvers/) | 共享数值更新步骤 | `linear.py`、`projected_cg.py`、`nonlinear.py`、`row_action.py` |
| [algorithms/fwi/](src/usctbench/algorithms/fwi/) | WUST 参数映射、数据交换与进程调用 | `algorithm.py`、`io.py`、`runtime.py` |
| [evaluation/](src/usctbench/evaluation/)、[metrics.py](src/usctbench/metrics.py) | 观测划分、残差与图像指标 | `evaluation/data.py`、[metrics.py](src/usctbench/metrics.py) |
| [benchmark/](src/usctbench/benchmark/) | 执行编排、产物保存与批量报告 | `runner.py`、`artifacts.py`、`report.py` |
| [configs/](configs/)、[examples/](examples/)、[scripts/](scripts/) | 算法参考配置、评测套件与运行示例 | `configs/algorithms/`、`configs/benchmarks/` |
| [docs/](docs/)、[tests/](tests/) | 技术说明与数值/API 验证 | [文档索引](docs/README.md)、[测试索引](docs/physics_validation.md) |

`core/`、`operators/` 等模块路径相对于 `src/usctbench/`；`configs/` 等顶层路径相对于仓库根目录。

阅读算法时，从 `algorithms/` 的对应文件进入，沿正演调用阅读 `operators/`，再到 `solvers/` 查看共享迭代流程。接入上层系统时，从 `benchmark/runner.py` 和 `core/registry.py` 的算法目录开始。

## 安装与快速开始

使用 Linux 或 macOS，以及 **Python 3.12+**。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[viz]"
```

生成一个小型仿体，利用直线走时数据重建声速，并汇总运行结果：

```bash
usct data prepare --dataset synthetic --out data/synthetic_demo \
  --shape 32 32 --n-transducers 32 --reference-sound-speed-mps 1500

usct run straight_cgls \
  --case data/synthetic_demo/cases/synthetic_circular_sos.h5 \
  --config configs/algorithms/cgls.yaml \
  --out runs/cgls_demo

usct evaluate --run runs/cgls_demo
```

示例在 `runs/cgls_demo/` 中生成重建图像、运行记录、指标和预览图。仿体采用确定性构造，观测来自匹配的直线正演模型。Born 与 FWI 使用与各自物理模型对应的声压观测。

按所需功能选择安装选项：

| 安装方式 | 功能 |
| --- | --- |
| `python -m pip install -e .` | 核心算法与数值输出；运行时选择 `--no-preview` 或 `preview=False` |
| `python -m pip install -e ".[viz]"` | 图像预览与绘图 |
| `python -m pip install -e ".[performance]"` | Numba 快速行进加速 |
| `python -m pip install -e ".[cuda12]"` 或 `".[cuda11]"` | CuPy Full-Green Born；按本机 CUDA 主版本选择 |
| `python -m pip install -e ".[dev,viz]"` | 测试、格式化、静态检查与预览 |

## 公开接口

### 命令行

| 任务 | 命令 |
| --- | --- |
| 发现算法 | `usct list-algorithms --json` |
| 查询物理变体 | `usct describe-algorithm born --variant full_green_nonlinear` |
| 查看样本 | `usct inspect path/to/case.h5` |
| 重建单个样本 | `usct run <algorithm> --case <case.h5> --config <config.yaml> --out <directory>` |
| 汇总保存的结果 | `usct evaluate --run <directory>` |
| 运行评测套件 | `usct bench --suite <suite.yaml>` |

`usct describe-algorithm ... --case <case.h5>` 还会给出样本的可用频率与校准信息。数据准备与声压导入命令见[数据集指南](docs/datasets.md)和[数据转换指南](docs/data_conversion.md)。

### Python

```python
from usctbench import evaluate, run
from usctbench.data.synthetic import make_sound_speed_case

case = make_sound_speed_case(shape=(32, 32), n_transducers=32)
result = run(
    "straight_cgls",
    case,
    {
        "parameters": {"sound_speed_bounds_mps": [1400.0, 1700.0]},
        "run_controls": {"max_iterations": 20},
    },
    out="runs/python_demo",
)

print(result.status, result.metrics.get("stop_reason"))
print(result.sound_speed_mps.shape)
summary = evaluate("runs/python_demo")
```

`run` 接收 `USCTCase` 或 HDF5 样本路径，以及字典、类型化配置或 YAML 路径，返回 `ReconstructionResult`。`out` 指定本次运行的完整输出目录；runner 负责配置准备、重建、保存及所选图像评价。真值由评价阶段使用。

`evaluate(directory)` 汇总保存的运行记录，返回 `records`、`passed`、`summary_csv` 和 `report_md`。`passed` 汇总执行、数值结果可用性与后处理状态，图像指标描述重建精度。`suite(path)` 执行评测套件，并额外返回 `run_root` 和 `num_cases`。

配置按用途分组：

| 配置部分 | 用途 |
| --- | --- |
| `parameters` | 物理模型、初始化、正则化和更新参数 |
| `run_controls` | 迭代、时间与工作量预算；停止条件与检查点选择 |
| `evaluation` | 训练与验证观测划分 |
| `image_evaluation` | 图像评价区域与指标 |
| `deployment` | 运行时路径、设备和宿主执行设置 |

开发内存中的算法调用时，先执行一次 `prepare_config(...)`，再将准备好的配置传给 `get_algorithm(name).run(case, prepared)`。完整接口说明见[使用指南](docs/usage.md)。

## Agent 接入

Agent 调用按四个步骤组织：发现算法、查看样本能力、构造类型化请求、使用宿主提供的资源执行。

```python
from usctbench import describe, run
from usctbench.core.algorithm_specs import case_capabilities, make_agent_config
from usctbench.data.synthetic import make_sound_speed_case

case = make_sound_speed_case(shape=(16, 16), n_transducers=16)
specification = describe("straight_cgls")
observations = case_capabilities("straight_cgls", case)

request = make_agent_config(
    "straight_cgls",
    parameters={"sound_speed_bounds_mps": [1400.0, 1700.0]},
    run_controls={"max_iterations": 10},
)
result = run(
    "straight_cgls",
    case,
    request,
    out="runs/agent_demo",
    host_limits={"max_iterations": 8, "max_elapsed_s": 60},
)
```

`describe` 提供变体、观测要求、参数 schema、默认值和预算能力；`make_agent_config` 根据开放的科学参数构造请求，`trusted_parameters` 用于传入宿主已知的物理信息。宿主负责选择部署路径、设备与资源上限。有效预算取请求预算与 `host_limits` 的交集。

字段定义与变体选择方式见 [Agent API](docs/agent_algorithm_api.md)和[参数约定](docs/parameter_contract.md)。

## 样本、坐标与声压数据

`USCTCase` 包含图像网格、采集几何、观测数据、元数据以及可选评价标签。

| 物理量 | 数组布局 | 单位或约定 |
| --- | --- | --- |
| 声速与空间掩膜 | `[y, x]` | 声速单位为 m/s |
| 网格间距与原点 | `[dy, dx]`、`[y, x]` | 米；原点是像素 `(0, 0)` 的外侧边界 |
| TX/RX 位置 | `[element, 2]` | 列顺序为 `[y, x]`，单位为米 |
| 走时 | `[tx, rx]` | 秒 |
| 射线权重 | `[tx, rx]` | 无量纲 |
| 时域声压 | `[time, tx, rx]` | 声压单位随数据声明；`time_axis_s` 以秒计 |
| 频域声压 | `[frequency, tx, rx]` | 显式频率、声压单位与 Fourier 约定 |
| 准备后的源谱 | `[frequency, tx]` | 复数源系数 |

像素 $(i,j)$ 的中心坐标为 $(y_0+(i+\tfrac12)\Delta y,\ x_0+(j+\tfrac12)\Delta x)$。WUST 适配层将 BenchLab 的 `[y, x]` 位置转换为 WUST 协议的 `[x, y]` 坐标。

频域样本在 `metadata["pressure_contract"]` 中声明 `fourier_sign`、`real_pressure`、`pressure_type`、`data_units` 和 `spectrum_normalization`。Born 使用正号 DTFT 频谱，WUST 准备负号频谱。声明为实值时域声压的数据支持通过复共轭转换符号。频域与时域水参考分别存放在 `water_reference` 和 `water_reference_time`。

数据层支持合成仿体、OpenBreastUS 属性图、NBPslices2D 和 k-Wave/MATLAB 声压采集。属性图准备生成带标签的模型走时样本；声压导入保留采集的几何、时间和源信息。字段定义与文件示例见[算子约定](docs/operator_contract.md)和[数据转换指南](docs/data_conversion.md)。

## 评测套件与结果

完成快速开始的数据准备后，可运行项目提供的套件：

```bash
usct bench --suite configs/benchmarks/synthetic_demo.yaml
```

套件将样本与具名实验组织在一起。以下示例可保存为仓库根目录的 `suite.yaml`：

```yaml
name: tof_comparison
case_glob: data/synthetic_demo/cases/*.h5
outputs:
  root: runs
host_limits:
  max_iterations: 10
algorithms:
  - id: cgls_default
    name: straight_cgls
    config: configs/algorithms/cgls.yaml
  - id: sirt_default
    name: straight_sirt
    config: configs/algorithms/sirt.yaml
```

路径相对于套件文件解析。每个实验具有唯一 `id`，同一算法的不同配置可以独立保存和比较。输出目录为 `<output-root>/<run-id>/<experiment-id>/<case-id>/`。显式 `run_id` 决定运行目录名，默认值由套件名称和 UTC 时间戳组成。

| 产物 | 内容 |
| --- | --- |
| `result.h5` | 重建数组与数值元数据 |
| `run.json` | 实验身份、配置、执行阶段、工作量、耗时与产物状态 |
| `metrics.json` | 求解历史、观测域指标和所选图像指标 |
| `index.json` | 已发布数值结果的索引 |
| `preview.png` | 使用 `viz` 扩展生成的图像预览 |
| `benchmark_summary.csv`、`benchmark_report.md` | 由 `evaluate` 或 `suite` 生成的汇总表 |

阅读结果时，将 `status`、`stop_reason`、选中迭代和实际工作量一起查看。预算结束的运行可以返回已完成的检查点。观测划分用于验证与检查点选择，已有真值用于图像评价。残差比较采用相同的观测域、单位和采集定义。详细说明见[评价指南](docs/evaluation.md)与[结果生命周期](docs/result_lifecycle.md)。

## WUST 接入

安装 [WaveformInversionUST](https://github.com/lucian-dw/WaveformInversionUST) 与 MATLAB，并指定后端和运行时路径：

```bash
usct run fwi_wust \
  --case /absolute/frequency_case.h5 \
  --config configs/algorithms/fwi_wust.yaml \
  --runtime-root /absolute/WaveformInversionUST \
  --matlab-executable /path/to/matlab \
  --backend cpu \
  --out runs/fwi
```

根据输入网格设置声速边界、PML 宽度和频率调度。示例配置通过 `run_controls.max_elapsed_s: 3600` 提供运行时间预算。每次重建请求启动一个 MATLAB 进程；WUST 负责频率排序、Fourier 规范化、采样索引与逐源复系数拟合。

WUST GPU 后端使用 Parallel Computing Toolbox 和已编译的 CUDA MEX 求解器。原生 Full-Green Born 通过 `deployment={"green_device": 0}` 选择 CuPy 设备。这两个设置分别控制各自算法的执行路径。完整部署说明见 [FWI 接入指南](docs/fwi.md)。

## 添加新算法

沿现有分层将新方法接入从观测到声速结果的完整流程：

1. **写明数学定义。** 给出观测、未知量、单位、目标、约束、导数，以及恢复声速的映射。
2. **声明参数。** 在 `algorithms/parameters.py` 或算法专属包中添加参数模型。
3. **实现算子与更新。** 将物理映射放入 `operators/`，复用数学问题匹配的求解器，并在 `algorithms/` 中实现 `Algorithm.run(case, prepared)`。
4. **注册方法。** 在 `core/algorithm_specs.py` 添加算法说明，在 `core/registry.py` 添加 `AlgorithmEntry`，并在 `configs/algorithms/` 提供参考 YAML。
5. **验证行为。** 添加对应的独立数值参考和一个小型公开入口用例，记录运行命令与结果。

上述 Python 模块路径相对于 `src/usctbench/`，`configs/algorithms/` 位于仓库根目录。[AGENTS.md](AGENTS.md) 与[开发指南](docs/development.md)规定项目共享的编写与验证方式。

## 开发与文档

```bash
python -m pip install -e ".[dev,viz]"
black --check src tests scripts
ruff check src tests scripts
pytest -q
```

CPU 测试覆盖数值参考、伴随、导数、有界更新、数据转换和公开接口。MATLAB、GPU 与原生仿真的运行命令集中在[验证指南](docs/validation.md)。

| 阅读目的 | 文档 |
| --- | --- |
| 运行算法与评测套件 | [使用指南](docs/usage.md) |
| 理解目标函数、离散与正则化 | [数学表述](docs/math_formulation.md)、[算法指南](docs/algorithms.md) |
| 准备数据与源信息 | [数据集](docs/datasets.md)、[数据转换](docs/data_conversion.md) |
| 配置 Agent 参数与预算 | [Agent API](docs/agent_algorithm_api.md)、[参数约定](docs/parameter_contract.md) |
| 查看数值验证 | [物理验证索引](docs/physics_validation.md)、[验证命令](docs/validation.md) |
| 浏览全部技术文档 | [文档索引](docs/README.md) |

## 许可证与参考资料

BenchLab 采用 [MIT 许可证](LICENSE)。数据集与外部运行时遵循各自的许可证。科学方法与数据集引用集中在 [references.bib](docs/references.bib)。
