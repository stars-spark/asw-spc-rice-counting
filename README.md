# 粘连米粒计数

这是一份数字图像处理课堂作业。目标是数出一张图中的米粒，并处理米粒相互接触后在二值图中合为一个连通域、从而发生漏数的问题。

方法一不依赖训练。程序先从当前图像中估计单粒米的面积、短轴和形状，再判定粘连块的粒数，使用距离变换与自适应标记分水岭分开粘连区域，按面积和形状复核，剔除硬币、木框边的亮条和阴影等异物，最后回到去噪前的阈值图，找回被开运算抹掉的细小米粒。尺度判据都相对当前图像标定的单粒米来定，不使用固定的像素门限。

方法二以 SAM 3 的点标注作为教师信号，训练一个 27 万参数的密度图网络。报告同时保留了教师伪标签和学生模型检查点，因此不下载 SAM 3 权重也能重新生成报告中的学生模型图。

## 报告结果

三组共 810 张测试图中，方法一的平均绝对误差分别为 D1 真实照片 2.75 粒、D2 合成粘连图 0.25 粒、D3 低分辨率图 0.50 粒。D2 的粘连率从 0% 升到 80% 时，直接数连通域的误差从 0.25 粒升到 38.88 粒，本文方法为 0.88 粒。

报告中的所有源文件、图、表、指标、原始数据压缩包、学生模型检查点和教师伪标签均已纳入仓库。克隆后无需重新跑实验就能编译报告；核心几何实验也可以由原始数据重新计算。

## 安装

```bash
git clone https://github.com/stars-spark/asw-spc-rice-counting.git
cd asw-spc-rice-counting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

XeLaTeX、BibTeX 与中文 `ctex` 宏包用于编译报告。没有 GPU 时可以完成几何方法、指标表和报告编译；SAM 3 的原始推理需要另外安装 `transformers` 并下载模型权重。

## 直接编译报告

报告引用的图片、数据表和指标已随仓库提交。执行下面的命令即可生成与仓库版本一致的报告 PDF。

```bash
make report
```

输出文件为 `report/米粒粘连导致漏数的问题与解决.pdf`。公开仓库不包含 `report/author.tex`，因此编译版会省略作者行，不影响正文、图表、参考文献和页码。

## 从数据重新计算核心实验

四个 Roboflow 原始导出包位于 `data/`。数据许可、来源与目录映射见 [data/README.md](data/README.md)。以下命令会解压数据、用固定随机种子生成 D2，并重算几何方法、五个基线、消融与鲁棒性实验。

```bash
make core
make tables
```

`results/cache/`、`results/synthetic/` 和 `results/renders/` 是可再生成的大型中间产物，刻意未纳入版本控制。报告真正需要的 `results/figures/`、`results/metrics/`、`results/labels/` 与 `results/student/` 则已提交。

若要重新生成报告中不需要运行 SAM 3 的插图，先完成数据准备，再运行下面的命令。它依赖仓库中缓存的学生模型与教师伪标签：

```bash
make figures
```

三张方法一与 SAM 3 的逐粒对比图和附录中的六张图需要实际运行 SAM 3，附录所用场景已写在 `src/visualize.py` 的 `APPENDIX_SCENES` 中：

```bash
make figures-sam3
```

## 重新运行 SAM 3 与学生模型

仓库保存了生成报告所需的 SAM 3 汇总指标、伪标签和学生模型权重，因而报告可离线复现。若希望从零重新运行教师模型与训练学生模型，需要安装额外依赖、准备可用 GPU，并先下载 Meta 的 SAM 3 权重：

```bash
pip install transformers
python -m src.teacher_sam --matrix
python -m src.pseudo
python -m src.student
python -m src.student --eval
```

这部分会改变与硬件、模型版本和随机数状态有关的耗时及部分中间结果；报告中的版本化指标用于固定可核对的实验记录。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `src/` | 几何方法、基线、评测、SAM 3 对照和学生模型 |
| `data/` | 原始数据压缩包与许可说明 |
| `results/figures/` | 报告实际引用的插图 |
| `results/metrics/` | 报告表格和图形使用的实验指标 |
| `results/labels/`、`results/student/` | 学生模型图所需的教师伪标签与检查点 |
| `report/` | 课程报告 LaTeX 源码 |

## 许可

本仓库代码采用 MIT 许可。数据集保持其原始 CC BY 4.0 许可与署名信息，详见 [data/README.md](data/README.md)。
