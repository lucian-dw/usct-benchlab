# EIT 接口

`eitbench` 与 `usctbench` 位于同一仓库，采用相同的 registry / config /
operators / algorithms / solvers / benchmark 分层与 describe / run / evaluate 工作流。
EIT 使用独立的网格、电导率和边界电压类型，不复用声速字段。依赖使用仓库现有科学计算栈，CPU 运行。

## 当前清单

| 层 | 实现 |
|---|---|
| 正演 | 二维各向同性实电导率，`div(sigma grad u)=0`，P1 三角形有限元 |
| 边界变体 | `cem` 完整电极模型；`continuum` 连续边界模型 |
| 伴随 | 当前电导率处精确离散伴随，稀疏 LU 复用于全部注流右端与伴随 |
| 数据项 | 掩模内归一化电压最小二乘，`difference` 或 `absolute` |
| 正则化 | 相对电导率上的平滑各向同性 FEM TV |
| 约束 | 相对背景电导率的盒约束，默认 `[0.05, 5]` |
| 求解算法 | 注册一个 `lbfgsb_tv`，L-BFGS-B；两种边界变体、两种观测模式 |
| 校准 | 仅参考观测拟合均匀背景；CEM 同时拟合电极接触阻抗 |

这是附件中经典 EIT 算法的迁移。首版没有为 EIT 接入 DIP、INR 或 DPS。
原始来源、SHA256 与数据署名见 [eit_source/SOURCE.json](eit_source/SOURCE.json) 和
[第三方声明](eit_source/THIRD_PARTY_NOTICES.md)。附件中的说明作为来源材料，不作为工作区操作指令。

## 数学与单位

节点电导率 `sigma` 为 S/m，坐标和厚度 `h` 为 m，电流 A，电压 V。
CEM 的接触阻抗 `z_l` 为物理量 Ω m²：

```
u + z_l sigma ∂n u = U_l
h ∫electrode_l sigma ∂n u ds = I_l
```

电极间隙绝缘，总注流为零，`sum(U_l)=0` 固定规范。
连续边界模式使用电极上的分段边界片电流、绝缘间隙和电极平均电位观测，
以边界电位积分为零固定规范；另提供任意 P1 连续边界片通量的正演方法。
连续模型与 CEM 的零接触阻抗极限不同。
电导率到电压的映射非线性；伴随返回当前模型处 `J(sigma)^T q`，不是固定反投影。

优化变量 `m=sigma/background`。差分模式比较
`F(background*m)-F(background)` 与 `V_target-V_reference`；绝对模式比较
`F(background*m)` 与 `V_target`。两种模式都用参考数据做校准。
目标函数：

```
0.5 || masked_residual ||² / || masked_measurement ||²
+ alpha ∑triangle area * (sqrt(|∇m|² + tv_epsilon²) - tv_epsilon)
```

默认值沿用附件：`alpha=.03`、`tv_epsilon=.1`、`ftol=1e-8`、`gtol=1e-6`。
背景校准范围 `[.001,.3] S/m`，接触阻抗校准范围 `[1e-7,.1] Ω m²`。
校准 L-BFGS-B 使用 `ftol=1e-13, gtol=1e-8`；其迭代上限单独默认为 400。

## 数据准备

数据默认放在被 Git 忽略的 `data/eit/kit4/`，不随 wheel 分发。可从用户提供的
ZIP 导入，导入器只复制数据，不执行附件代码：

```bash
python -m eitbench.data.import_submission --archive /path/to/eit_submission.zip --out data/eit/kit4
```

也可以使用 `python -m eitbench.data.prepare_kit4 --archive /path/to/data_mat_files.zip
--photos /path/to/target_photos.zip --output data/eit/kit4` 从官方归档重建。
归档地址、校验和与 CC-BY-4.0 署名位于 manifest 和来源声明中。
数据源为 Hauptmann 等的 [Zenodo 1203914](https://doi.org/10.5281/zenodo.1203914)。
原 MAT 的电流采用 mA 到 A 的换算，保留原始 `sqrt(2)` 幅度。
绝对 S/m 标度依赖这一解释，未将电流替换为报告中的 2 mA。

原生 NPZ 和 EIT HDF5 包含相同的具名数组：
`nodes[N,2]`、`triangles[T,3]`、`boundary_edges[B,2]`、`electrode_ids[B]`、
`currents_A[L,P]`、`measurement_matrix[Q,L]`、`voltages_V[Q,P]`、
`reference_voltages_V[Q,P]`、布尔 `measurement_mask[Q,P]`、标量 `thickness_m`、
接触阻抗 `contact_impedance_ohm_m2`、标量 `background_conductivity_S_m`、标量 `case_id`。
索引必须是整数。`eitbench.load_case/save_case` 处理两种格式。
实测电压作为观测输入；照片不进入算法，不具有定量电导率真值。

## 调用

安装本仓库后有独立 `eit` 命令。也可使用 `python -m eitbench.cli`。
建议运行前设置 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。

```bash
eit list-algorithms --json
eit describe-algorithm lbfgsb_tv --variant cem
eit run lbfgsb_tv --case data/eit/kit4/example.npz \
  --config configs/eit/lbfgsb_tv_cem.yaml --out runs/eit_example --preview
eit evaluate --run runs/eit_example
eit suite --cases data/eit/kit4/phantom_*.npz \
  --config configs/eit/lbfgsb_tv_continuum.yaml --out runs/eit_continuum
```

`--preview` 需要可选 Pillow；不使用预览时无需 Pillow。
输出目录必须为空。每个 suite 案例独立校准、独立计费；不跨案例复用校准。

```python
import eitbench as eit

config = eit.prepare_config({
    "name": "lbfgsb_tv",
    "parameters": {"variant": "cem", "mode": "difference"},
    "run_controls": {"max_iterations": 500, "max_forward_calls": 1000},
    "deployment": {"device": "cpu"},
})
result = eit.run(case="data/eit/kit4/example.npz", config=config, out="runs/eit_api")
print(result.status, result.calibration, result.work)
```

## 预算、停止与结果

配置仅有 `name / parameters / run_controls / deployment` 四层。
支持 `max_iterations`（目标重建默认 500，可设零，只评估初始模型）、
`max_elapsed_s`、`max_forward_calls`、`max_adjoint_calls`。
全局时间及算子预算包含参考校准、均匀基线正演和目标反演。
一次全部注流右端的批量 FEM 求解计一次正演，一次对应批量伴随计一次伴随；
缓存命中不计费，开始执行但失败的求解计费。上层预算与原生预算取更严格值。
时间限制在工作单元之间协作检查，单次稀疏求解不支持中途抢占；硬超时由调用方子进程执行器负责。

`converged` 仅表示局部优化器满足投影梯度或目标函数容差；
`max_iterations` 明确表示达到迭代上限；`resource_exhausted` 表示预算耗尽；
`numerical_error` 表示数值异常或线搜索失败。参考校准有独立状态和轨迹，
校准迭代到限可继续重建，但校准预算耗尽或数值失败会停止整个运行。
校准状态不能作为目标重建收敛证据。

只保存完成损失和梯度计算且被优化器接受的迭代；预算耗尽时不保存线搜索试探点。
若校准阶段中断，或尚未完成任何目标迭代评估，只生成诊断，不生成重建结果或 index。
在已完成初始目标评估后中断，可以保存初始模型作为明确标记的有效检查点。

有有效结果时生成：`result.h5`、`run.json`、`metrics.json`、`index.json`，可选 `preview.png`。
HDF5 的主结果是 `conductivity_S_m[N]`，同时保存节点电位、预测/参考/基线电压、
测量掩模、网格、仅供展示的 `conductivity_image_S_m` 和 `domain_mask`。
图像外域补零，不能将外域当作低电导率。JSON 保存解析后配置、输入/代码哈希、
依赖版本、校准与目标优化轨迹、停止原因和工作量。
`evaluate` 直接读取已保存结果，计算电压一致性，不额外调用正演、不使用照片算 PSNR/SSIM。

## 验证

快速数值与协议测试：`python -m pytest tests/eit -q`。
与原附件代码做四组合逐项对照：

```bash
python examples/eit_verify_migration.py --original /path/to/extracted/eit \
  --out runs/eit_acceptance/migration.json
```

原附件的完整 `verify.py` 检查了残差、互易、注流守恒、规范、伴随/接触梯度、
连续模型的圆域解析解及合成反演；迁移测试进一步核对正演、TV、目标梯度与重建一致性。
本次验收证据保存在本地 `runs/eit_acceptance/`，数据与运行记录均不进入源码 wheel。
