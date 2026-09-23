# 数据

四个压缩包都是从 Roboflow Universe 直接导出的原始文件，未作改动，许可均为 CC BY 4.0。
解压后即可运行 `src/` 下的脚本。

| 压缩包 | 在本项目中的代号 | 内容 | 来源 |
|---|---|---|---|
| `Rice.v1i.coco.zip` | D1 | 真实照片，51 张，深色布上摊开的米粒，每张带一枚参照硬币 | [annotation-gnc0j/rice](https://universe.roboflow.com/annotation-gnc0j/rice-nqqbr) |
| `RICE.v3i.coco.zip` | D3 | 低分辨率 224×224 小图，去重后 719 张 | [ezekiel-se47x/RICE](https://universe.roboflow.com/ezekiel-se47x/rice-ljhi4) |
| `RICE.v2i.folder.zip` | D2 的米粒素材 | 单粒米照片，Karacadag 品种 | [thanaree/RICE](https://universe.roboflow.com/thanaree/rice-fgvkm) |
| `ricecount.v2i.coco.zip` | D4 | 带直尺的真实照片，只用于失效分析，不参与评测 | [asia-pacific-university-xxaah/rice count](https://universe.roboflow.com/asia-pacific-university-xxaah/rice-count) |

每个压缩包内的 `README.dataset.txt` 与 `README.roboflow.txt` 保留了上游的署名与许可信息，请勿删除。
`ricecount.v2i.coco.zip` 的上游文件名含空格。为使解压命令与代码目录名一致，仓库中已将空格去掉，内容未变。

## 解压

代码按 `data/extracted/<数据集名>/` 查找图片，解压时要指定目录名：

```bash
cd data
unzip -q Rice.v1i.coco.zip      -d extracted/rice.v1i.coco
unzip -q RICE.v3i.coco.zip      -d extracted/RICE.v3i.coco
unzip -q RICE.v2i.folder.zip    -d extracted/RICE.v2i.folder
unzip -q ricecount.v2i.coco.zip -d extracted/ricecount.v2i.coco
```

D2 是本项目自行合成的，不在这里，由 `python -m src.synth` 按固定随机种子生成。
