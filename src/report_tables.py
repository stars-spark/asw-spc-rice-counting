"""Emit LaTeX tabular fragments from the metrics CSVs.

The report inputs these fragments, so regenerating the experiments updates the numbers in
the document instead of leaving hand-copied values to go stale.
"""
import numpy as np
import pandas as pd

from src.io_utils import RESULTS_ROOT, PROJECT_ROOT, ensure_dir

METRICS_ROOT = RESULTS_ROOT / "metrics"
TABLE_ROOT = PROJECT_ROOT / "report" / "tables"

METHOD_LABELS = {
    "B1_components": r"B1 连通域计数",
    "B2_area": r"B2 面积估算",
    "B3_dist_watershed": r"B3 距离变换注水分割",
    "B4_erosion_watershed": r"B4 腐蚀标记注水分割 \cite{kurade2023}",
    "B5_concave_ellipse": r"B5 凹点+椭圆拟合 \cite{avzalov2025}",
    "Ours_ASW_SPC": r"\textbf{本文方法}",
    "SAM3_teacher": r"SAM 3 零样本",
}

METHOD_ORDER = list(METHOD_LABELS)


def _fmt(value, digits=2):
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}"


def _best(values, better="min"):
    """一组数里最好的那个值，缺失值不参与；用于决定哪一格加粗。"""
    valid = [v for v in values if v is not None and not pd.isna(v)]
    if not valid:
        return None
    return min(valid) if better == "min" else max(valid)


def _mark(text, value, best):
    """与最好值相同的那一格加粗。按显示出来的数字比较，四舍五入后并列的一起加粗。"""
    if best is None or value is None or pd.isna(value) or text == "--":
        return text
    return rf"\textbf{{{text}}}" if text == _fmt_like(best, text) else text


def _fmt_like(value, text):
    """把 value 按 text 的小数位数格式化，便于比较显示值。"""
    digits = len(text.split(".")[1]) if "." in text else 0
    return f"{value:.{digits}f}"


def _write(name, body):
    ensure_dir(TABLE_ROOT)
    path = TABLE_ROOT / name
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path}")
    return path


def comparison_table():
    summary = pd.read_csv(METRICS_ROOT / "summary.csv")
    summary = summary[summary.stratum == "all"]

    sam3_path = METRICS_ROOT / "sam3_summary.csv"
    if sam3_path.exists():
        summary = pd.concat([summary, pd.read_csv(sam3_path)], ignore_index=True)

    # B5's disc radius has no setting that serves every grain size, so quoting one
    # configuration would understate it. It is given the same treatment as SAM 3's prompt:
    # the best configuration on each set, which is the strongest case that can be made for
    # it. That those three cases are three different configurations is the point of its own
    # table, not something this one should hide by picking a single row.
    b5_path = METRICS_ROOT / "b5_matrix.csv"
    if b5_path.exists():
        matrix = pd.read_csv(b5_path)
        summary = summary[summary.method != "B5_concave_ellipse"]
        for dataset in ("d1", "d2", "d3"):
            column = f"{dataset}_MAE"
            if column not in matrix:
                continue
            best = matrix.loc[matrix[column].idxmin()]
            summary = pd.concat([summary, pd.DataFrame([{
                "dataset": dataset, "method": "B5_concave_ellipse", "stratum": "all",
                "MAE": best[column], "accuracy_%": np.nan, "worst": np.nan,
            }])], ignore_index=True)

    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "rrr" * 3 + "}",
        r"\toprule",
        r"& \multicolumn{3}{c}{D1 真实稀疏} & \multicolumn{3}{c}{D2 合成粘连} & "
        r"\multicolumn{3}{c}{D3 低分辨率} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"方法 & MAE & 准确率\% & 最差 & MAE & 准确率\% & 最差 "
        r"& MAE & 准确率\% & 最差 \\",
        r"\midrule",
    ]

    # 加粗只在六种几何方法之间比较，SAM 3 是另一类做法，单列在最后作参照
    contenders = [m for m in METHOD_ORDER if m != "SAM3_teacher"]
    columns = (("MAE", "min", 2), ("accuracy_%", "max", 2), ("worst", "min", 0))
    best = {}
    for dataset in ("d1", "d2", "d3"):
        block = summary[(summary.dataset == dataset) & summary.method.isin(contenders)]
        for column, better, _ in columns:
            best[(dataset, column)] = _best(list(block[column]), better)

    for method in METHOD_ORDER:
        if method == "SAM3_teacher":
            lines.append(r"\midrule")
        cells = [METHOD_LABELS[method]]
        for dataset in ("d1", "d2", "d3"):
            row = summary[(summary.dataset == dataset) & (summary.method == method)]
            if row.empty:
                cells += ["--"] * 3
                continue
            row = row.iloc[0]
            for column, _, digits in columns:
                text = _fmt(row[column], digits)
                if method != "SAM3_teacher":
                    text = _mark(text, row[column], best[(dataset, column)])
                cells.append(text)
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("comparison.tex", "\n".join(lines))


def touching_table():
    summary = pd.read_csv(METRICS_ROOT / "summary.csv")
    strata = summary[(summary.dataset == "d2") & (summary.stratum != "all")].copy()
    strata["level"] = strata.stratum.str.replace("touch=", "", regex=False).astype(float)
    pivot = strata.pivot_table(index="level", columns="method", values="MAE")

    short = {"B1_components": "B1", "B2_area": "B2", "B3_dist_watershed": "B3",
             "B4_erosion_watershed": "B4", "B5_concave_ellipse": "B5",
             "Ours_ASW_SPC": r"\textbf{本文}"}
    methods = [m for m in METHOD_ORDER if m in pivot.columns]
    lines = [
        # 七列数字挤在一起不好读，列距放宽到默认的两倍
        r"\setlength{\tabcolsep}{12pt}",
        r"\begin{tabular}{l" + "r" * len(methods) + "}",
        r"\toprule",
        "粘连率 & " + " & ".join(short.get(m, m) for m in methods) + r" \\",
        r"\midrule",
    ]
    for level, row in pivot.iterrows():
        top = _best([row[m] for m in methods])
        cells = [f"{level * 100:.0f}\\%"] + [_mark(_fmt(row[m]), row[m], top) for m in methods]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("touching.tex", "\n".join(lines))


# The rows the report discusses. The other ablations stay in ablation.csv; they probe
# intermediate designs that the final text no longer describes.
REPORTED_ABLATIONS = (
    "full method",
    "no self-calibration (A0 = dataset mean)",
    "no shape-prior correction (M5)",
    "no foreign-object rejection",
    "foreign rejection without the width test",
    "foreign test also requiring roundness",
    "touching gate: OR instead of AND",
    "seed pieces not merged",
    "seed pieces not merged, beta = 0.45",
    "gray channel only",
    "with 3x3 median filter",
)


def ablation_table():
    table = pd.read_csv(METRICS_ROOT / "ablation.csv")
    labels = {
        "full method": r"完整方法",
        "no self-calibration (A0 = dataset mean)": r"去掉尺度自标定，$A_0$ 取数据集均值",
        "no shape-prior correction (M5)": r"去掉形状先验校正",
        "no foreign-object rejection": r"去掉异物剔除",
        "foreign rejection without the width test": r"异物剔除去掉宽度判据",
        "foreign test also requiring roundness": r"异物判据附加“更圆”条件，即最初写法",
        "seed pieces not merged": r"种子碎块不合并",
        "seed pieces not merged, beta = 0.45": r"种子碎块不合并，$\beta$ 取 0.45，即修正前的写法",
        "no measurement-reliability gate": r"去掉可测量性门限",
        "reliability gate abstains instead of using width excess":
            r"门限触发时弃权（而非改用宽度过剩量）",
        "coarse-scale rule on area instead of width excess":
            r"粗尺度只按面积切分（无粘连证据）",
        "one-sided binarisation priors": r"二值化先验只保留下界",
        "no post-removal re-validation": r"移除基底后不复检、不回退",
        "touching gate: OR instead of AND": r"粘连判据改为“或”",
        "full candidate grid": r"完整候选网格（默认）",
        "gray channel only": r"仅灰度通道",
        "with 3x3 median filter": r"加 $3\times3$ 中值滤波",
    }

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"消融项 & D1 & D2 & D3 \\",
        r"\midrule",
    ]
    shown = table[table.ablation.isin(REPORTED_ABLATIONS)]
    # 每列加粗误差最大的那一项，也就是去掉之后损失最大的那一步
    worst = {d: _best(list(shown[d]) if d in shown else [], "max") for d in ("D1", "D2", "D3")}
    for _, row in table.iterrows():
        if row["ablation"] not in REPORTED_ABLATIONS:
            continue
        name = labels[row["ablation"]]
        cells = [name] + [_mark(_fmt(row.get(d)), row.get(d), worst[d]) for d in ("D1", "D2", "D3")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("ablation.tex", "\n".join(lines))


def beta_table():
    table = pd.read_csv(METRICS_ROOT / "ablation.csv")
    betas = table[table.ablation.str.startswith("beta")].copy()
    if betas.empty:
        return None
    betas["value"] = betas.ablation.str.replace("beta = ", "", regex=False).astype(float)

    lines = [
        r"\begin{tabular}{l" + "r" * len(betas) + "}",
        r"\toprule",
        r"$\beta$ & " + " & ".join(f"{v:.2f}" for v in betas["value"]) + r" \\",
        r"\midrule",
    ]
    for dataset in ("D1", "D2", "D3"):
        if dataset not in betas.columns:
            continue
        lines.append(dataset + " & " + " & ".join(_fmt(v) for v in betas[dataset]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("beta.tex", "\n".join(lines))


def sam3_table():
    path = METRICS_ROOT / "sam3_calibration_d1.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    pivot = table.pivot_table(index="prompt", columns="threshold", values="MAE")

    lines = [
        r"\begin{tabular}{l" + "r" * len(pivot.columns) + "}",
        r"\toprule",
        r"文本提示 & " + " & ".join(f"$\\tau={c:.2f}$" for c in pivot.columns) + r" \\",
        r"\midrule",
    ]
    top = _best(list(pivot.values.ravel()))
    for prompt, row in pivot.iterrows():
        lines.append(f"\\texttt{{{prompt}}} & " + " & ".join(_mark(_fmt(v), v, top) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("sam3.tex", "\n".join(lines))


def sam3_matrix_table():
    path = METRICS_ROOT / "sam3_prompt_matrix.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{D1} & \multicolumn{2}{c}{D2} & \multicolumn{2}{c}{D3} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"文本提示 & MAE & $\tau^\ast$ & MAE & $\tau^\ast$ & MAE & $\tau^\ast$ \\",
        r"\midrule",
    ]
    top = {key: _best(list(table[f"{key}_MAE"])) for key in ("d1", "d2", "d3")}
    for _, row in table.iterrows():
        cells = [f"\\texttt{{{row['prompt']}}}"]
        for key in ("d1", "d2", "d3"):
            value = row[f"{key}_MAE"]
            cells += [_mark(_fmt(value), value, top[key]), _fmt(row[f"{key}_thr"])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("sam3_matrix.tex", "\n".join(lines))


def b5_matrix_table():
    path = METRICS_ROOT / "b5_matrix.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)

    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"圆盘半径 $R$ & 椭圆校正 & D1 & D2 & D3 \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = [
            f"${row['radius_ratio']:.1f}\\,b_0$",
            "是" if row["ellipse_correction"] else "否",
            _fmt(row["d1_MAE"]), _fmt(row["d2_MAE"]), _fmt(row["d3_MAE"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("b5_matrix.tex", "\n".join(lines))


AXIS_LABELS = {
    "blur": ("高斯模糊", r"$\sigma$", "{:.2f}$\\,b_0$"),
    "noise": ("加性噪声", r"$\sigma$", "{:.0f}"),
    "contrast": ("对比度保留", "", r"{:.0f}\%"),
    "resolution": ("降采样倍率", "", r"{:.1f}$\times$"),
}


# A few representative levels per axis; the full sweep stays in robustness.csv.
REPORTED_LEVELS = {
    "blur": (0.0, 0.2, 0.35, 0.7),
    "noise": (0.0, 20.0, 45.0),
    "contrast": (1.0, 0.3, 0.08),
    "resolution": (1.0, 3.0, 6.0),
}


def robustness_table():
    """Mean, median and blow-up count per degradation level.

    The median and the blow-up count are in the table because the mean alone misdescribes
    what happens: on three of the four axes nothing happens at all, and on the fourth the
    mean is moved by a couple of scenes whose binarisation flips.
    """
    path = METRICS_ROOT / "robustness.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    ours = table[table.method == "Ours_ASW_SPC"]
    b1 = table[table.method == "B1_components"].set_index(["axis", "level"])

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"退化类型 & 强度 & 平均误差 & 中位误差 & 崩溃场景 & B1 平均误差 \\",
        r"\midrule",
    ]
    for axis, (label, _symbol, fmt) in AXIS_LABELS.items():
        block = ours[(ours.axis == axis)
                     & ours.level.round(3).isin(REPORTED_LEVELS[axis])].sort_values("level")
        for position, (_, row) in enumerate(block.iterrows()):
            name = label if position == 0 else ""
            reference = b1.loc[(axis, row["level"]), "MAE"] if (axis, row["level"]) in b1.index else float("nan")
            lines.append(" & ".join([
                name, fmt.format(row["level"] * 100 if axis == "contrast" else row["level"]), _fmt(row["MAE"]), _fmt(row["median_AE"]),
                # 出现崩溃的格子加粗，一眼能看出只有模糊会让方法失效
                (rf"\textbf{{{int(row['blowups'])}/{int(row['n'])}}}" if row["blowups"] > 0
                 else f"{int(row['blowups'])}/{int(row['n'])}"), _fmt(reference),
            ]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return _write("robustness.tex", "\n".join(lines))


def cost_table():
    """Time, memory and model size of the proposed method against SAM 3 on one machine."""
    path = METRICS_ROOT / "cost.csv"
    if not path.exists():
        return None
    rows = pd.read_csv(path).set_index("config")
    cols = ["ours_cpu", "ours_gpu", "student_gpu", "sam3_gpu", "sam3_cpu"]

    def secs(key, col):
        value = rows.loc[key, col]
        if pd.isna(value):
            return "--"
        if value < 0.1:        # the student is in milliseconds; two decimals would read 0.00
            return f"{value:.3f}"
        return f"{value:.2f}" if value < 10 else f"{value:.1f}"

    def cells(fn):
        """每行里数值最小的一格加粗，即这一项上最省的做法。"""
        texts = [fn(key) for key in cols]
        numbers = []
        for text in texts:
            try:
                numbers.append(float(text.replace(" M", "")))
            except ValueError:
                numbers.append(None)
        top = _best(numbers)
        return " & ".join(
            rf"\textbf{{{t}}}" if n is not None and top is not None and n == top else t
            for t, n in zip(texts, numbers))

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"项目 & 本文 CPU & 本文 GPU & 学生模型 GPU & SAM 3 GPU & SAM 3 CPU \\",
        r"\midrule",
        r"D1 每张耗时／s & " + cells(lambda k: secs(k, "d1_s")) + r" \\",
        r"D2 每张耗时／s & " + cells(lambda k: secs(k, "d2_s")) + r" \\",
        r"D3 每张耗时／s & " + cells(lambda k: secs(k, "d3_s")) + r" \\",
        r"\midrule",
        r"模型参数量 & " + cells(lambda k: "0" if k.startswith("ours")
                                   else (f"{rows.loc[k, 'params'] / 1e6:.2f} M"
                                         if rows.loc[k, "params"] < 1e7
                                         else f"{rows.loc[k, 'params'] / 1e6:.0f} M")) + r" \\",
        r"权重文件／MB & " + cells(lambda k: "0" if k.startswith("ours")
                                   else f"{rows.loc[k, 'weights_mb']:.0f}") + r" \\",
        r"模型加载／s & " + cells(lambda k: "--" if k.startswith("ours")
                                  else f"{rows.loc[k, 'load_s']:.1f}") + r" \\",
        r"峰值内存／GB & " + cells(lambda k: f"{rows.loc[k, 'rss_mb'] / 1024:.2f}") + r" \\",
        r"峰值显存／GB & " + cells(lambda k: f"{rows.loc[k, 'vram_mb'] / 1024:.2f}"
                                   if k.endswith("gpu") else "--") + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return _write("cost.tex", "\n".join(lines))


def _student_cells(block, methods):
    """一行误差。教师是参照，加粗只在其余三种做法之间比。"""
    errors = {k: (block[k] - block.truth).abs().mean() for k, _ in methods}
    top = _best([v for k, v in errors.items() if k != "sam3"])
    return " & ".join(_fmt(errors[k]) if k == "sam3" else _mark(_fmt(errors[k]), errors[k], top)
                      for k, _ in methods)


def student_table():
    """Student against the classical pipeline and the teacher, on the held-out images."""
    path = METRICS_ROOT / "student_test.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    methods = [("b1", "直接数连通域"), ("ours", "方法一"), ("student", "方法二 学生模型"),
               ("sam3", "SAM 3 教师")]
    methods = [(k, label) for k, label in methods if k in table.columns]

    lines = [r"\begin{tabular}{lr" + "r" * len(methods) + "}", r"\toprule",
             "数据集 & 张数 & " + " & ".join(label for _, label in methods) + r" \\",
             r"\midrule"]
    for dataset, label in (("d1", "D1 真实照片"), ("d2", "D2 合成粘连"), ("d3", "D3 低分辨率")):
        block = table[table.dataset == dataset]
        if block.empty:
            continue
        lines.append(f"{label} & {len(block)} & {_student_cells(block, methods)}" + r" \\")
    lines.append(r"\midrule")
    lines.append(r"合计 & " + str(len(table)) + " & " + _student_cells(table, methods) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return _write("student.tex", "\n".join(lines))


def main():
    comparison_table()
    b5_matrix_table()
    robustness_table()
    touching_table()
    ablation_table()
    beta_table()
    sam3_table()
    sam3_matrix_table()
    cost_table()
    student_table()


if __name__ == "__main__":
    main()
