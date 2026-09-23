# 报告编译说明

主文件：`米粒粘连导致漏数的问题与解决.tex`，编译出的 PDF 与它同名

## 文件分工

| 文件 | 内容 |
|---|---|
| `米粒粘连导致漏数的问题与解决.tex` | 版式：字号、页眉页脚、浮动体参数、标题层次 |
| `body.tex` | 正文（引言到结论） |
| `appendix.tex` | 附录 A：更多计数结果图 |
| `abstract.tex` / `keywords.tex` | 摘要 / 关键词 |
| `common.tex` | 共用宏、图片搜索路径 |
| `refs.bib` | 参考文献（GB/T 7714 格式，由 gbt7714 宏包排版） |
| `tables/*.tex` | 表格片段，**由 `python3 -m src.report_tables` 自动生成，不要手改** |

插图在 `../results/figures/`，由 `python3 -m src.visualize` 生成。

## 编译

VSCode 装 LaTeX Workshop 后直接按 `Ctrl+Alt+B`，
首次或改动参考文献后请选 `xelatex -> bibtex -> xelatex x2`。

命令行：

    xelatex 米粒粘连导致漏数的问题与解决 && bibtex 米粒粘连导致漏数的问题与解决 && xelatex 米粒粘连导致漏数的问题与解决 && xelatex 米粒粘连导致漏数的问题与解决

## 改动提示

- 表格数字变了：重跑 `python3 -m src.report_tables`，不要手改 `tables/` 下的文件
- 插图要改：改 `src/visualize.py` 后重跑，不要直接编辑 PNG
- 浮动体只用 `cfigure`（通栏）或 `figure[htbp]`（单栏）。
  通栏浮动体在双栏下只能置于页顶或独占一页，图多文字少时会产生空白页；
  内容不宽的图请用单栏。

## GPU 版本

`src/gpu_pipeline.py` 是二值化与标定阶段的 GPU 实现，需要额外安装：

```bash
pip install --index-url https://pypi.org/simple cupy-cuda13x cucim-cu13
```

用法 `from src import gpu_pipeline; gpu_pipeline.count_rice_gpu(img)`。
注水分割仍在 CPU 上运行（cuCIM 无 GPU 实现，且只作用于很小的粘连块截图）。
图很小时（如 D3 的 224×224）GPU 反而更慢，此时应使用 CPU 版。

## 批量处理与线程设置

```bash
python -m src.batch photos/ --out counts.csv          # 每核一个进程
python -m src.batch photos/ --workers 8
```

`src/batch.py` 会先把每个进程的线程数锁成 1（需要 `threadpoolctl`）。
这一步比并行本身更重要：OpenBLAS 默认用 20 个线程去做标定里那些很小的矩阵运算，
单张图会占满 17 个核，反而更慢。锁定后单张快 1.6–1.8 倍，再按图并行才有加速
（D3 上 240 张图用 20 进程为 2.0 s，相对单进程 4.4 倍）。
`src.evaluate` 与 `src.robustness` 已自动应用该设置。
