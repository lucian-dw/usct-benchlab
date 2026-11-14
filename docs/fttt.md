# FTTT：DIP、INR、DPS 接入说明

`fttt` 是统一首波走时神经重建入口，包含 `dip`、`inr`、`dps` 三个变体。
物理计算复用原生 Eikonal 正演及其离散 Jacobian 伴随；网络、求解循环和 VP
扩散调度器来自固定版本 inv_framework。

扩散权重按数据集存放在 diffusion/weights/，正式数据集的后续训练引导到
Yang Song 官方 Score SDE。目录、登记状态与训练步骤见
[扩散模型入口](../diffusion/README.md)。

## 安装与调用

在仓库根目录执行：

```bash
python -m pip install -e '.[neural,viz]'
usct describe-algorithm fttt --variant dip
usct describe-algorithm fttt --variant inr
usct describe-algorithm fttt --variant dps
usct run fttt --case data/first_arrival_case.h5 --config configs/algorithms/fttt_dip.yaml --out runs/fttt_dip
usct run fttt --case data/first_arrival_case.h5 --config configs/algorithms/fttt_inr.yaml --out runs/fttt_inr
```

Python 使用标准 `run`，返回 `ReconstructionResult`：

```python
from usctbench import run

result = run("fttt", case, {
    "parameters": {"variant": "inr", "learning_rate": 0.0001, "seed": 0},
    "run_controls": {"max_iterations": 1000},
}, "runs/my_fttt")
```

未安装 PyTorch 时，传统算法与算法发现仍可使用。只有执行 FTTT 才导入
PyTorch。首版 DPS 使用提取的 TinyUNet 和 VPSchedule，不依赖 diffusers。

## 观测、变量与损失

- 每次重建一个病例；图像为 `[y,x]`，走时为 `[tx,rx]`，位置 m，
  走时 s，输出声速 m/s。
- 输入使用 `measurement.tof_s` 或 `delta_tof_s`；同时提供时优先差分，
  与现有 Bent 一致。差分参考须与 reference_sound_speed_mps 一致。
- 调用者须确认首波语义。字段名不证明物理含义，互相关和相位时延不能无说明
  地作为首波。独立 tof_first_arrival_s 需由数据准备显式选入上述输入字段。
- 网络变量 u 的物理映射为
  `s = 1/c_max + clip(u,0,1)*(1/c_min - 1/c_max)`。DIP/INR 已使用 sigmoid；
  DPS 先验在 u 图像上训练，越界估计使用同一分段可微截断。
- ROI 外固定为参考慢度；ROI 不由真值推断。
- DIP/INR 损失为
  `0.5*sum(w*(pred-obs)^2)/sum(w)/residual_scale_s^2`。
  DPS 保留上游残差范数引导，使用加权 RMS 残差除以 residual_scale_s 的梯度。
  默认尺度为 1 微秒，是固定数值设置，不从验证观测估计。
- 通道先索引再减法；无效值、零权重、验证通道及其互易排除通道不进入训练。
  标准 runner 在数值执行前剥离图像真值。
- backward 使用该次正演对应的 Jacobian 伴随。梯度局部依赖活跃行进模板，
  路径切换处不保证光滑；适配器支持一阶反向传播。

## 参数与运行控制

参数由 algorithms/fttt/parameters.py 定义，未知字段会被拒绝。

| 变体 | 主要参数与默认值 |
| --- | --- |
| 共用 | seed=0，eikonal_order=1，residual_scale_s=1e-6，sound_speed_bounds_mps=[1300,1700] |
| DIP | learning_rate=0.001，input_channels=8，base_channels=16 |
| INR | learning_rate=0.0001，hidden_features=64，hidden_layers=3 |
| DPS | sampling_steps=64，guidance_scale=0.1，eta=0，宿主提供 prior_manifest 或 checkpoint |

这些是参考设置，并非任意数据上的最优参数或生理范围。DIP 将内部画布补到
至少 16 且边长为 8 的倍数，再裁剪回原物理网格。INR 按 x、y 输入坐标，
按原 [y,x] 网格恢复输出。

- DIP/INR 一轮为一次 Adam 更新；DPS 一轮为一次反向采样更新。
- 默认 DIP/INR 1000 轮，DPS 默认预算为 sampling_steps。
- DPS 宿主迭代上限截断原日程，不把日程压缩成更少步。
- 监控所需的额外正演也计入预算；时间在工作边界及 Eikonal 源循环检查。
- 返回的检查点拥有匹配的模型与预测；首次正演前预算耗尽时返回参考场，
  并标记没有完整检查点。
- 达到迭代上限为 budget_exhausted，不表示收敛。DPS 残差可能非单调，
  因此拒绝 objective_rtol 平台停止；留出验证选取检查点是显式可选项。
- DPS 中间检查点表示截断到物理范围的采样状态，不等于中间干净估计。

deployment.neural_device 默认为 cpu，可显式指定 cuda 或 cuda:0，设备不可用
时报告错误。Eikonal 始终在 CPU，GPU 网络需与其传输数据。可用
OMP_NUM_THREADS/MKL_NUM_THREADS 控制 CPU 并行；CUDA 路径尚需目标硬件验收。

## DPS 权重与训练

DPS 必须提供训练过的兼容权重；缺失时在物理执行前失败。不会采用随机模型
代替预训练先验，也不会在重建过程中隐式训练或下载。

默认示例通过 prior_manifest 选择 synthetic_ellipses。OpenBreastUS 和
NBPslices2D 的目录及配置已预留为 untrained；缺少权重时明确引导到
score_sde_pytorch。外部原始 checkpoint 只能先登记为 registered，其
网络、score/epsilon 和 SDE 适配验证完成前不会标成 ready。

训练可复现的合成参考先验：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m usctbench.algorithms.fttt.train_prior \
  --output diffusion/weights/synthetic_ellipses/reference.pt --steps 5000 --shape 16 16 --seed 70
usct run fttt --case data/first_arrival_case.h5 --config configs/algorithms/fttt_dps.yaml --out runs/fttt_dps
```

生成器在 1500 m/s 背景中放置随机椭圆，目标为 1400–1600 m/s，使用 512 张
训练图与另一随机种子的 32 张验证图，归一化范围 [1300,1700] m/s。
固定验证噪声和时间步只用于报告 epsilon MSE，不参与梯度更新。
默认训练噪声日程为 1000 步线性 beta，推理取其中 64 步。纯噪声初始化
要求末端 alpha_bar <= 1e-3，加载器会检查。

自备数据为 NPZ，包含 train_speed_mps 和 validation_speed_mps，
形状均为 [image,ny,nx]，单位 m/s。应预先按独立病例划分并排除正式测试病例：

```bash
python -m usctbench.algorithms.fttt.train_prior --images my_training_images.npz \
  --shape 128 128 --steps 5000 --output checkpoints/my_prior.pt
```

工具检查范围、形状、跨集合重复图像并记录哈希；不能自动证明病例级独立性。
合成先验仅供接入验证，不代表真实乳腺组织分布。

checkpoint 格式 usct.fttt.prior.v1 包含 state_dict 和元数据：网络结构、
epsilon 预测类型、归一化、尺寸、训练日程、步数、数据来源和验证记录。
使用 weights_only=True 加载，校验结构、有限数值、尺寸和归一化范围，
记录 SHA256，不自动缩放不匹配的图像。权重及侧车记录位于 Git 忽略的
diffusion/weights/<dataset>/（也支持宿主直接指定其他路径），源码保留完整训练入口。
数据集 manifest 的 SHA256 也会校验，并写入运行来源记录。

## 复跑与验证

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python examples/fttt_compare.py \
  --checkpoint diffusion/weights/synthetic_ellipses/reference.pt --out runs/fttt_reference_final \
  --iterations 100 --seeds 0 1 2
pytest -q tests/operators/test_torch_eikonal.py tests/algorithms/test_fttt.py
```

比较脚本使用独立 seed=907 测试图、固定参数，运行 bent_ray_gn、DIP、INR、
DPS 和同种子的无引导扩散对照，保存标准结果、汇总 JSON 和可选对照图。
不同方法迭代单位不同，需要结合实际算子次数与耗时比较。
这是同物理模型生成的合成接入证据，不是实验成像验证。

测试覆盖正演值、独立前向图的伴随、链式梯度、ROI、掩模和留出防泄漏、
预算、RNG 恢复、无 PyTorch 的发现、权重校验和实际重建。两步训练 fixture
仅检查接口；参考权重训练质量由独立运行记录报告。

## 来源

上游固定于 peterw634/inv_framework 提交
b258fa90d284f0105259aeffbfd43814f4cd3085。原文件 blob ID、许可声明和本地
适配说明见 _vendor/inv_framework/SOURCE.json 与 NOTICE.md。
