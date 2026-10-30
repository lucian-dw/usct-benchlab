# usct-benchlab

[English](README.md)

用于超声计算机断层成像（USCT）的 **声速反演算法库**。以统一的数据对象连接观测、
物理算子、迭代求解与评价，供研究人员和反问题 Agent 调用。验证范围是明确模型下的
数值正确性，不代表临床有效性。

## 模型与算法

USCT 是由声学传播方程约束的反问题，例如：

$$
c(x)^{-2}\partial_{tt}p_s-\Delta p_s=q_s,\qquad
d_{sr}=\mathcal M_r p_s+\eta_{sr}.
$$

不同方法利用观测中的不同信息：

| 命令 | 输入观测 | 模型与变量 | 配置 |
| --- | --- | --- | --- |
| straight_cgls | 水参考差分 ToF，秒 | 直线加权最小二乘，$\delta s=1/c-1/c_0$ | [cgls.yaml](configs/algorithms/cgls.yaml) |
| straight_sirt | 水参考差分 ToF，秒 | 同时归一化射线更新，$\delta s$ | [sirt.yaml](configs/algorithms/sirt.yaml) |
| straight_sart | 水参考差分 ToF，秒 | 有序子集归一化更新，$\delta s$ | [sart.yaml](configs/algorithms/sart.yaml) |
| bent_ray_gn | 首波到达时间或水参考 ToF | 原生 Eikonal/Gauss–Newton，$s=1/c$ | [bent_ray.yaml](configs/algorithms/bent_ray.yaml) |
| born | 复压力与源校准或独立水参考 | 原生 WKB / Full-Green Born，$m=1/c^2$ | [born.yaml](configs/algorithms/born.yaml) |
| fwi_wust | 约定明确的复数总压力 | 独立维护的 WUST Helmholtz FWI，$s=1/c$ | [fwi_wust.yaml](configs/algorithms/fwi_wust.yaml) |

**born** 是本仓库 Born 方法的统一入口，不声称完整复现上游
r-Wave。详见[数学形式化](docs/math_formulation.md)和[算法说明](docs/algorithms.md)。
YAML 是显式示例配置，不是经过标定的强弱档位，也不保证特定图像质量。

## 安装与首次运行

需要 Python 3.12+：

~~~bash
pip install -e ".[dev,viz]"
usct data prepare --dataset synthetic --out data/synthetic_demo \
  --shape 32 32 --n-transducers 32 --reference-sound-speed-mps 1500
usct run straight_cgls \
  --case data/synthetic_demo/cases/synthetic_circular_sos.h5 \
  --config configs/algorithms/cgls.yaml --out runs/cgls_demo
usct evaluate --run runs/cgls_demo
~~~

该确定性圆形样本没有随机采样，其 ToF 来自匹配的直线模型，只用于数值演示，不是
压力波场实验。数据集导入、压力导入和 FWI 部署见[使用说明](docs/usage.md)与
[快速示例](examples/README.md)。

## Python 与 Agent 接口

~~~python
from usctbench import describe, run
from usctbench.data.synthetic import make_sound_speed_case

spec = describe("straight_cgls")
result = run(
    "straight_cgls",
    make_sound_speed_case(shape=(16, 16), n_transducers=16),
    {"run_controls": {"max_iterations": 10}},
    out="runs/python_demo",
)
print(result.status, result.metrics.get("stop_reason"))
~~~

~~~bash
usct list-algorithms --json
usct describe-algorithm born --variant full_green_nonlinear
usct bench --suite configs/benchmarks/synthetic_demo.yaml
~~~

科学参数、执行预算和宿主部署分开传入。未知参数明确报错；Agent 可修改范围来自
静态 schema，高级求解选项、运行时路径不开放给默认自主 Agent。
详见 [Agent API](docs/agent_algorithm_api.md)。

公开 runner 每次只准备一次配置；直接调用底层算法需传
`prepare_config(name, config, case=case)` 的结果。Bent 与非线性 Born 共用
GN 主循环，但保留各自的可行 trial 定义；直线 CGLS 与固定背景 Born 共用有界投影 CG，
后者明确处理实模型、复观测。默认只记录接受迭代及实际工作量，详细试探记录由
`diagnostics: true` 显式开启。

## 输入与结果

- 属性图生成带明确来源的模型 ToF；不能称作已采集压力。
- 实测或仿真压力保留实际几何、采样时间、傅里叶约定与单位；到达时间特征显式提取。
- 重建入口不接收 GT，真值仅交给结果评价。
- 图像是 **[y,x]**，位置及间距在 BenchLab 内均是米制 **[y,x]**；
  WUST 协议入口只转换一次位置列为 **[x,y]**。

结果直接写到指定目录：**result.h5**、**run.json**、数值成功索引 **index.json**、
**metrics.json**、可选 **preview.png**。评价或绘图失败不会抹掉已保存的数值结果。
Suite 另外生成 **benchmark_summary.csv** 与 **benchmark_report.md**。

ToF 与复压力残差、不同采集来源不能混排成统一排名。GT 指标分别报告前景和全景；
用完预算、执行完频率计划不等于数值收敛。

## 文档与验证

[使用说明](docs/usage.md) · [数据](docs/datasets.md) ·
[算子约定](docs/operator_contract.md) · [评价与停止](docs/evaluation.md) ·
[FWI](docs/fwi.md) · [参数](docs/parameter_contract.md) ·
[开发](docs/development.md) · [历史研究和图像](docs/research_index.md)

默认 CI 只执行一次 CPU pytest 和格式检查。MATLAB CPU、CUDA 与两仓集成是显式手动
验收，历史研究图不作为默认配置的复现承诺。

## 许可与引用

[MIT](LICENSE)，第三方实现保留各自许可。
[参考文献](docs/references.bib)包含
[OpenBreastUS](https://ai4scientificimaging.org/OpenBreastUS/)、
[NBPslices2D](https://doi.org/10.13012/B2IDB-5648161_V1) 及
[维护版 WUST](https://github.com/lucian-dw/WaveformInversionUST)。
