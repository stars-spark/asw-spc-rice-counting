.DEFAULT_GOAL := report

.PHONY: data synth core tables figures report clean-report

# 解压版本化的原始数据。解压结果被 .gitignore 排除，可随时删除后重建。
data:
	mkdir -p data/extracted/rice.v1i.coco data/extracted/RICE.v2i.folder data/extracted/RICE.v3i.coco data/extracted/ricecount.v2i.coco
	unzip -q -o data/Rice.v1i.coco.zip -d data/extracted/rice.v1i.coco
	unzip -q -o data/RICE.v2i.folder.zip -d data/extracted/RICE.v2i.folder
	unzip -q -o data/RICE.v3i.coco.zip -d data/extracted/RICE.v3i.coco
	unzip -q -o data/ricecount.v2i.coco.zip -d data/extracted/ricecount.v2i.coco

# D2 由固定随机种子生成，生成 40 张图及其真值清单。
synth: data
	python -m src.synth

# 重新计算不依赖 SAM 3 权重的核心实验结果。
core: synth
	python -m src.evaluate --b5-matrix
	python -m src.ablation
	python -m src.robustness

tables:
	python -m src.report_tables

# 依赖版本化的学生模型检查点和教师伪标签，不要求下载 SAM 3 权重。
figures: synth
	python -m src.visualize

# 报告所需图、指标和表格均被版本化，克隆后可直接执行此目标。
report:
	cd report && xelatex 米粒粘连导致漏数的问题与解决 && bibtex 米粒粘连导致漏数的问题与解决 && xelatex 米粒粘连导致漏数的问题与解决 && xelatex 米粒粘连导致漏数的问题与解决

clean-report:
	rm -f report/*.aux report/*.bbl report/*.blg report/*.log report/*.out report/*.toc report/*.synctex.gz report/米粒粘连导致漏数的问题与解决.pdf
