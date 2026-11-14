# 按数据集管理扩散模型

这里是 USCT 的扩散先验入口。每个数据集有独立权重目录、manifest 和训练配置。
正式数据集需要权重时，按本指南到 Yang Song 的官方实现训练。

- 原始官方仓库：[yang-song/score_sde](https://github.com/yang-song/score_sde)（JAX）。
- 推荐与本项目衔接的 [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)。

## 目录

```text
diffusion/
  README.md
  weights/
    synthetic_ellipses/       # 可复跑的小型原生接口示例
      manifest.json
      reference.pt           # 本地生成，Git 忽略
    openbreastus/            # 待训练
      manifest.json
      training_config.py     # 后续按实际数据准备
      checkpoints/           # Score SDE 训练输出，Git 忽略
    nbpslice2d/               # 待训练
      manifest.json
      training_config.py
      checkpoints/
```

添加数据集时复制一个待训练 manifest，使用唯一 dataset_id。不要将自然图像
的 CIFAR/人脸预训练模型登记为声速先验，也不要将合成示例登记为真实数据集权重。

## 查找与调用

仓库根目录执行，未安装 PyTorch 也可以查看：

```bash
python -m usctbench.algorithms.fttt.weight_bank
python -m usctbench.algorithms.fttt.weight_bank --dataset openbreastus
usct run fttt --case data/my_case.h5 --config configs/algorithms/fttt_dps_openbreastus.yaml --out runs/my_dps
```

DPS 配置通过宿主提供的 prior_manifest 选择数据集权重；仍支持显式 checkpoint，
二者不能同时指定。未训练时会返回缺失权重及官方训练地址，不会回退到随机模型。
载入前检查 checkpoint 文件、SHA256、尺寸、归一化、网络和噪声日程。

manifest 状态含义：

| 状态 | 含义 |
| --- | --- |
| untrained | 预留位置，尚无权重 |
| registered | 已保存权重及哈希，尚未完成当前 DPS 运行适配 |
| ready | 当前原生 DPS 加载器已支持的权重格式 |

ready 只表示运行格式兼容，不代表重建质量达标。现有合成参考权重用于接口演示，
其 DPS 重建质量仍需改进；正式数据集需要各自训练和评价。

## 到官方 Score SDE 训练

1. 按独立病例划分训练、验证、测试集，准备单通道声速或慢度图。只用训练集
   确定需要学习的归一化统计；记录物理范围、单位、网格方向和尺寸。
2. 在独立训练环境获取 score_sde_pytorch，按它的 requirements.txt 安装依赖。
   其原有数据管线不直接识别 USCT NPZ/HDF5，需要在官方数据加载层接入自己的
   图像。将通道数、尺寸、centered 数据缩放和 VP/VE/subVP 设置写入训练配置。
3. 将训练配置保存为本数据集目录下的 training_config.py，记录上游 Git commit、
   数据清单、划分和预处理。准备完成后，在官方仓库执行：

```bash
# 以下路径替换为自己的绝对路径；config 必须已完成本数据集适配。
python main.py --config=/path/to/usct-benchlab/diffusion/weights/openbreastus/training_config.py \
  --mode=train --workdir=/path/to/usct-benchlab/diffusion/weights/openbreastus
```

4. 保留 checkpoints/、EMA 权重、配置和评估记录，将选定 checkpoint 的相对路径
   与 SHA256 写入 manifest，状态改为 registered。记录模型训练步数和训练数据来源。
5. 接入重建前验证模型结构、时间标签、EMA 加载方式、数据缩放及 score 输出。
   当前 DPS 使用 VP epsilon 预测器：对于匹配的 VP 高斯扰动，
   epsilon = -sigma(t) * score。若使用 centered 缩放，还需对应的坐标尺度转换。
   VE、subVP、连续时间模型不能只改文件名就使用当前离散 VP 调度器。

**外部 Score SDE checkpoint 可以在这里登记，但当前加载器不会冒充支持它。**
这次交付建立了按数据集保存、选择、缺失时引导训练的路径；NCSN++/DDPM++ 的
运行适配应在拿到实际配置与权重后验证，完成前保持 registered。默认原生示例
无需安装 Score SDE，已有 DIP/INR 不依赖扩散权重。

## 原生示例

synthetic_ellipses 是 16×16 合成慢度图上的 TinyUNet 参考权重，供接口演示，
不代表 OpenBreastUS 或 NBPslices2D。可重新生成到自己的输出位置：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m usctbench.algorithms.fttt.train_prior \
  --output diffusion/weights/synthetic_ellipses/reference.pt --steps 5000 --shape 16 16 --seed 70
```

训练器不会覆盖已有权重。新文件须更新 manifest 的 SHA256。
