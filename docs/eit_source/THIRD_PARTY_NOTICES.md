# 数据来源、引用与第三方依赖

## 公开实验数据

- Hauptmann, Andreas; Kolehmainen, Ville; Mach, Nguyet Minh; Savolainen, Tuomo; Seppänen, Aku; Siltanen, Samuli (2017). **2D electrical impedance tomography dataset**, version 1.0.0. Zenodo. [DOI: 10.5281/zenodo.1203914](https://doi.org/10.5281/zenodo.1203914).
- 实验说明：同一作者，**Open 2D Electrical Impedance Tomography data archive**，[arXiv:1704.01178](https://arxiv.org/abs/1704.01178)。使用数据发表成果时请引用原作者和报告。
- 授权：[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)。已核对 [Zenodo 官方元数据](https://zenodo.org/api/records/1203914) 的 `metadata.license.id = cc-by-4.0`。
- 随包 `data/source/*.mat` 和 `data/photos/*.jpg` 是从官方归档取出的未修改原文件；只选取十个目标及一个公共参考。NPZ 是本项目整理的派生格式，包含单位转换、协议转置、通道掩模和新增二维网格；具体处理见 README 与 manifest。照片未被用作定量真值或反演输入，预览图仅作缩放展示。
- 原始归档 MD5/SHA256、逐文件 SHA256、下载链接均列在 `data/manifest.json`。原 FIPS 下载地址已失效，重新整理脚本使用作者的 Zenodo 存档。

## 软件依赖

NumPy 1.26.4、SciPy 1.13.1、Pillow 11.3.0 通过公开发行包安装，分别遵循其自身许可证；本 ZIP 不包含这些软件的源码或二进制副本。有限元、边界模型、TV 与伴随代码为本项目编写；没有复制 EIDORS、pyEIT 或其他 EIT 仓库代码，没有第三方训练权重。

本包的经典模型与优化参考用于算法说明，不表示原论文作者认可本实现或其测试结论。
