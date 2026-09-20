"""Figures for the report.

Labels are Chinese, to match the document they sit in, which needs a CJK font registered
with matplotlib. Figure sizes are chosen close to the width they are finally printed at, so
that the page scales them by roughly one and the type keeps the size it was set in; a
figure drawn far wider than its printed size arrives on the page with unreadably small
labels however large the point size looked in the source.
"""
import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager

# 图上的字与正文保持一致：西文用 Latin Modern Roman，中文用方正书宋，
# 与 report/tpl_cjournal.tex 里 \setCJKmainfont 指定的是同一套。
# 找不到时按候选顺序回退，也可用环境变量 RICE_LATIN_FONT / RICE_CJK_FONT 指定文件。
_LATIN_FAMILIES = ("Latin Modern Roman", "CMU Serif", "TeX Gyre Termes", "DejaVu Serif")
_CJK_FAMILIES = ("FZShuSong-Z01", "Source Han Serif SC", "Noto Serif CJK SC",
                 "Songti SC", "SimSun", "AR PL UMing CN", "Noto Sans CJK SC")

# TeX 发行版的字体不在系统字体目录里，需要单独找一遍
_EXTRA_FONT_GLOBS = (
    "~/.local/share/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-regular.otf",
    "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf",
)


def _register_font_files():
    """把候选字体文件登记进 matplotlib，返回已登记的字族名。"""
    import glob
    import os
    registered = []
    paths = []
    for name in ("RICE_LATIN_FONT", "RICE_CJK_FONT"):
        if os.environ.get(name):
            paths.append(os.path.expanduser(os.environ[name]))
    for pattern in _EXTRA_FONT_GLOBS:
        paths.extend(sorted(glob.glob(os.path.expanduser(pattern))))
    for path in paths:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                registered.append(font_manager.FontProperties(fname=path).get_name())
            except Exception:
                pass
    return registered


def _available(families):
    """按给定顺序返回第一个系统里真正存在的字族。"""
    known = {f.name for f in font_manager.fontManager.ttflist}
    for family in families:
        if family in known:
            return family
    return None


def _use_report_fonts():
    """让图上的字与报告正文同字体，中西文各自回退。"""
    extra = _register_font_files()
    latin = _available(tuple(extra) + _LATIN_FAMILIES) or "DejaVu Serif"
    cjk = _available(_CJK_FAMILIES) or "DejaVu Sans"
    # 逐字回退要把字族直接列在 font.family 上，写进 font.serif 只会取其中第一个
    matplotlib.rcParams["font.family"] = [latin, cjk, "DejaVu Serif"]
    matplotlib.rcParams["font.serif"] = [latin, cjk, "DejaVu Serif"]
    # 数学符号用内置的 Computer Modern，与正文公式同源。
    # 不能设成 custom 再指定西文字体：那样含 $...$ 的行会整体走数学字体，
    # 其中的中文便无字可用。中文与公式分行写则互不影响。
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return latin, cjk


_use_report_fonts()
# Every figure is set at the full text width, i.e. shrunk from its drawn width of about
# 9 in to about 6 in, so text is drawn at 1.5 times the size it should read at on the page.
FONT_SCALE = 1.5
_BASE_FONTS = {"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
               "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
               "figure.titlesize": 13}


def _fonts(scale):
    """rcParams for type drawn at `scale` times the base sizes."""
    return {key: size * scale for key, size in _BASE_FONTS.items()}


matplotlib.rcParams.update(_fonts(FONT_SCALE))

# Single plots are drawn narrower than the image grids, so the page enlarges them less and
# their type needs less compensation; these scales bring the labels to about 9 pt printed.
PLOT_FONT_SCALE = 1.0
SCATTER_FONT_SCALE = 1.25
ABSTRACT_FONT_SCALE = 1.15

# Okabe-Ito：色觉障碍下仍可分辨的一组颜色，学术出版常用。
# 顺序固定，同一方法在所有图里用同一颜色——颜色跟着对象走，不跟着名次走。
SERIES_COLORS = {
    "baseline": "#0072B2",   # 直接数连通域
    "ours": "#E69F00",       # 方法一
    "student": "#009E73",    # 方法二：学生模型
    "sam3": "#D55E00",       # SAM 3
}
INK = "#333333"
GRID = "#CCCCCC"

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import label2rgb
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries

from src import calibrate, counter, io_utils, preprocess, segment
from src.io_utils import RESULTS_ROOT, ensure_dir

# 图上一律写中文名，代号放在括号里：读者第一次看到图时，数据集还没有被正文定义过。
DATASET_CN = {"d1": "真实照片", "d2": "合成粘连图", "d3": "低分辨率图", "d4": "带直尺照片"}


def dataset_label(name, with_code=True):
    cn = DATASET_CN.get(name, name.upper())
    return f"{cn}（{name.upper()}）" if with_code else cn


METHOD_CN = {
    "B1_components": "B1 连通域计数",
    "B2_area": "B2 面积估算",
    "B3_dist_watershed": "B3 距离变换注水分割",
    "B4_erosion_watershed": "B4 腐蚀标记注水分割",
    "B5_concave_ellipse": "B5 凹点+椭圆拟合",
    "Ours_ASW_SPC": "本文方法",
}

FIG_ROOT = RESULTS_ROOT / "figures"
METRICS_ROOT = RESULTS_ROOT / "metrics"
DPI = 150


def _show(ax, image, title, cmap=None, fontsize=None):
    """图像面板。字号默认跟随 rcParams，这样同一张图里各面板的字一样大。"""
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=fontsize or 9 * FONT_SCALE)
    ax.axis("off")


def pipeline_figure(image_bgr, out_name="pipeline.png"):
    """Original -> channel -> mask -> components -> distance -> seeds -> result."""
    pre = preprocess.preprocess(image_bgr)
    calib = pre["calib"]
    mask = pre["mask"]
    total, _, debug = counter.count_rice(image_bgr, pre=pre, return_debug=True)

    dist = segment.distance_transform(mask)
    seeds = np.zeros(mask.shape, dtype=np.int32)
    for cluster in debug["clusters"]:
        r0, c0, r1, c1 = cluster["bbox"]
        seeds[r0:r1, c0:c1] = np.maximum(seeds[r0:r1, c0:c1], cluster["markers"])

    result = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).copy()
    for component in calib["components"]:
        if component["area"] < counter.SPECK_RATIO * calib["a0"]:
            continue
        colour = (255, 60, 60) if segment.is_foreign_object(component, calib) else (60, 255, 60)
        outline = find_boundaries(calib["labels"] == component["label"], mode="outer")
        result[outline] = colour

    fig, axes = plt.subplots(2, 4, figsize=(9.5, 5.2))
    _show(axes[0, 0], cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), "(a) 输入图像")
    _show(axes[0, 1], pre["channel"], "(b) 选出的通道", "gray")
    _show(axes[0, 2], mask, "(c) 二值图", "gray")
    _show(axes[0, 3], label2rgb(calib["labels"], bg_label=0), f"(d) 连通域 {calib['n_components']} 个")
    _show(axes[1, 0], dist, "(e) 距离变换", "magma")

    # Seeds are a handful of pixels each; dilate them in proportion to the image width so
    # they stay visible once the panel is shrunk to a quarter of the page.
    size = max(5, mask.shape[1] // 40) | 1
    grown = cv2.dilate((seeds > 0).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    seed_view = np.dstack([mask // 3] * 3)
    seed_view[grown > 0] = (255, 40, 40)
    _show(axes[1, 1], seed_view,
          "(f) 种子（红）")
    _show(axes[1, 2], result, f"(g) 计数 {total} 粒")

    axes[1, 3].axis("off")
    axes[1, 3].text(
        0.02, 0.95,
        "自标定结果：\n"
        f"$A_0$ = {calib['a0']:.0f} px\n"
        f"$b_0$ = {calib['minor0']:.1f} px\n"
        f"$\\rho_0$ = {calib['solidity0']:.2f}\n"
        f"$r_0$ = {calib['axis_ratio0']:.2f}\n\n"
        f"孤立米粒 {debug['n_isolated']} 粒\n"
        f"粘连块 {debug['n_clusters']} 个",
        va="top", fontsize=10 * FONT_SCALE,
    )

    fig.tight_layout()
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def cluster_detail_figure(image_bgr, out_name="cluster_detail.png", n_clusters=2):
    """Per-cluster view: mask, distance map, seeds, watershed, corrected count."""
    pre = preprocess.preprocess(image_bgr)
    _, _, debug = counter.count_rice(image_bgr, pre=pre, return_debug=True)
    clusters = sorted(debug["clusters"], key=lambda c: -c["component"]["area"])[:n_clusters]
    if not clusters:
        return None

    fig, axes = plt.subplots(len(clusters), 4, figsize=(9, 2.4 * len(clusters)), squeeze=False)
    for row, cluster in enumerate(clusters):
        ratio = cluster["component"]["area"] / pre["calib"]["a0"]
        _show(axes[row][0], cluster["mask"], f"粘连块，约 {ratio:.1f} 粒的面积", "gray")
        _show(axes[row][1], cluster["dist"], "距离变换", "magma")
        grown = cv2.dilate((cluster["markers"] > 0).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        seed_view = np.dstack([cluster["mask"] // 3] * 3)
        seed_view[grown > 0] = (255, 40, 40)
        _show(axes[row][2], seed_view, f"种子（红）：{cluster['markers'].max()} 个")
        _show(axes[row][3], label2rgb(cluster["merged"], bg_label=0),
              f"校正后：{cluster['count']} 粒")

    fig.tight_layout()
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def calibration_figure(image_bgr, out_name="calibration.png"):
    with plt.rc_context(_fonts(PLOT_FONT_SCALE)):
        return _calibration_figure(image_bgr, out_name)


def _calibration_figure(image_bgr, out_name="calibration.png"):
    """Log-area density with the estimated A0 and its multiples."""
    pre = preprocess.preprocess(image_bgr)
    calib = pre["calib"]
    areas = np.array([c["area"] for c in calib["components"]
                      if c["area"] >= calibrate.NOISE_FLOOR_PX])
    _, debug = calibrate.estimate_single_area(areas)
    if debug is None:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(np.log(areas), bins=40, density=True, alpha=0.35, color="steelblue", label="面积直方图")
    ax.plot(debug["grid"], debug["density"], color="darkblue", lw=2, label="核密度估计")
    for k in (1, 2, 3):
        ax.axvline(np.log(k * calib["a0"]), color="crimson" if k == 1 else "gray",
                   ls="--" if k == 1 else ":",
                   label=f"$A_0$ = {calib['a0']:.0f} px" if k == 1 else f"{k}$A_0$")
    # 同一行里既有中文又有 $...$ 时，整行会走数学字体，中文便成了方框，
    # 所以这类标签一律写成普通文字。
    ax.set_xlabel("连通域面积的对数 ln S")
    ax.set_ylabel("概率密度")
    ax.legend()

    fig.tight_layout()
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def error_curve_figure(out_name="error_vs_touching.png"):
    with plt.rc_context(_fonts(PLOT_FONT_SCALE)):
        return _error_curve_figure(out_name)


def _error_curve_figure(out_name="error_vs_touching.png"):
    """MAE against the synthetic touching level, per method."""
    per_image = pd.read_csv(METRICS_ROOT / "per_image.csv")
    d2 = per_image[(per_image.dataset == "d2") & per_image.touch_prob.notna()].copy()
    d2["abs_err"] = d2["error"].abs()
    table = d2.groupby(["touch_prob", "method"])["abs_err"].mean().unstack()

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for method in table.columns:
        if method.startswith("Ours"):
            style = dict(lw=2.8, marker="o", zorder=5)
        elif method == "B4_erosion_watershed":
            # B4 reproduces B1 exactly on this set, so a solid line would hide B1 entirely.
            style = dict(lw=1.8, marker="s", ls="--", alpha=0.9)
        else:
            style = dict(lw=1.4, marker="s", alpha=0.8)
        ax.plot(table.index * 100, table[method], label=METHOD_CN.get(method, method), **style)
    ax.set_xlabel("粘连率／%")
    ax.set_ylabel("平均绝对误差／粒")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


ROBUSTNESS_AXES = {
    "blur": ("高斯模糊", "σ／粒宽"),
    "noise": ("加性噪声", "σ／灰阶"),
    "contrast": ("对比度压缩", "保留的对比度"),
    "resolution": ("分辨率下降", "降采样倍率"),
}


def robustness_figure(out_name="robustness.png"):
    """Counting error against four controlled degradations.

    Mean and median are drawn together because they separate two different things: how the
    typical scene behaves, and whether any scene has collapsed. A log scale is needed on the
    error axis for the same reason - a collapsed binarisation lands two orders of magnitude
    away from an ordinary error, and a linear axis would render every other curve flat.
    """
    table = pd.read_csv(METRICS_ROOT / "robustness.csv")
    ours = table[table.method == "Ours_ASW_SPC"]
    reference = table[table.method == "B1_components"]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6))
    for ax, (axis, (title, xlabel)) in zip(axes.ravel(), ROBUSTNESS_AXES.items()):
        block = ours[ours.axis == axis].sort_values("level")
        base = reference[reference.axis == axis].sort_values("level")

        ax.plot(block["level"], block["MAE"], lw=2.5, marker="o", color="crimson",
                label="本文方法，平均")
        ax.plot(block["level"], block["median_AE"], lw=2.5, marker="o", ls="--",
                color="steelblue", label="本文方法，中位")
        ax.plot(base["level"], base["MAE"], lw=1.4, marker="s", alpha=0.7, color="gray",
                label="B1 连通域计数，平均")

        collapsed = block[block["blowups"] > 0]
        if not collapsed.empty:
            ax.scatter(collapsed["level"], collapsed["MAE"], s=150, facecolors="none",
                       edgecolors="crimson", lw=1.8, zorder=5)
            # One label for the whole run rather than one per point: the markers already
            # say which levels collapsed, and per-point labels overlap each other here.
            counts = ", ".join(f"{int(r['blowups'])}" for _, r in collapsed.iterrows())
            ax.annotate(f"圈出档位的崩溃场景数：{counts}（共 {int(block['n'].iloc[0])} 个）",
                        xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                        fontsize=9 * FONT_SCALE, color="crimson")

        ax.set_yscale("log")
        ax.margins(y=0.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("绝对计数误差／粒")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9 * FONT_SCALE)

    fig.tight_layout()
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def render_comparison_figure(prompt="white seed", threshold=0.40, picks=None,
                             datasets=("d1", "d2", "d3")):
    """One scene per dataset, counted by this method and by SAM 3, drawn side by side.

    Instances are outlined rather than filled so the grain underneath stays visible, and
    neighbouring instances are given different colours, so a reader can check the count by
    eye rather than take the reported number on trust. The counts go in the subplot titles
    instead of a banner drawn into the image, because they need to be in Chinese.

    One file per dataset rather than one stacked figure: the report sets every figure to the
    full text width, and a three-row stack at that width is taller than the text block, so it
    could only ever be placed on a page of its own and would truncate the page before it. Two
    panels side by side are half as tall at the same panel size, so each fits beside text.
    """
    from src import render, synth
    from src.teacher_sam import Sam3Teacher

    loaders = {"d1": io_utils.load_d1, "d2": synth.load_d2, "d3": io_utils.load_d3}
    picks = picks or {"d1": 0, "d2": None, "d3": 0}

    teacher = Sam3Teacher()
    ensure_dir(FIG_ROOT)
    written = []

    for name in datasets:
        samples = loaders[name]()
        index = picks.get(name)
        if index is None:  # for the synthetic set, take the most heavily touching scene
            sample = max(samples, key=lambda s: s.get("touch_prob") or 0)
        else:
            sample = samples[index]

        image = io_utils.imread(sample["path"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))

        labels, info, total, _ = counter.label_image(image)
        ours = render.draw(image, labels, info, banner_lines=None)
        _show(axes[0], cv2.cvtColor(ours, cv2.COLOR_BGR2RGB),
              f"{dataset_label(name)}　本文方法：{total} 粒（真值 {sample['gt_count']} 粒）")

        result = teacher.segment(image, prompt=prompt, threshold=threshold)
        scores = np.asarray(result["scores"].float().cpu())
        keep = scores >= threshold
        masks = np.asarray(result["masks"].float().cpu())[keep]
        sam_labels = render.masks_to_labels(masks, image.shape[:2])
        sam_info = {i: {"kind": "grain", "count": 1}
                    for i in range(1, int(sam_labels.max()) + 1)}
        sam = render.draw(image, sam_labels, sam_info, banner_lines=None)
        _show(axes[1], cv2.cvtColor(sam, cv2.COLOR_BGR2RGB),
              f"{dataset_label(name)}　SAM 3：{int(keep.sum())} 粒（真值 {sample['gt_count']} 粒）")

        fig.tight_layout()
        out_path = FIG_ROOT / f"render_{name}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written


def scatter_figure(out_name="pred_vs_true.png", datasets=("d1", "d2", "d3")):
    with plt.rc_context(_fonts(SCATTER_FONT_SCALE)):
        return _scatter_figure(out_name, datasets)


def _scatter_figure(out_name="pred_vs_true.png", datasets=("d1", "d2", "d3")):
    """Predicted against true counts for the proposed method."""
    per_image = pd.read_csv(METRICS_ROOT / "per_image.csv")
    ours = per_image[per_image.method == "Ours_ASW_SPC"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3.4))
    for ax, name in zip(np.atleast_1d(axes), datasets):
        block = ours[ours.dataset == name]
        truth, pred = block["gt"], block["pred"]
        lo, hi = truth.min() * 0.9, truth.max() * 1.1
        ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1)
        ax.scatter(truth, pred, s=14, alpha=0.6, color="steelblue")
        ax.set_xlabel("真值／粒")
        ax.set_ylabel("预测值／粒")
        ax.set_title(f"{dataset_label(name)}：MAE {block['error'].abs().mean():.2f} 粒")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def scope_failure_figure(out_name="scope_failure.png"):
    """Two photographs the method gets wrong, with the foreground its selector chose.

    The point is not that the counts are wrong but why. The selection score looks for a
    field of many similar, elongated, convex, high-contrast objects, and in these images the
    graduations of the ruler lying along the edge of the frame satisfy that description
    better than the rice does: they are evenly spaced, identical in size, sharply bounded
    and strongly contrasted, while the grains themselves are pale and vary in size. The
    fraction of the chosen foreground that falls in the outer border of the frame is printed
    with each mask rather than asserted, since that is where the ruler lies.
    """
    from src import io_utils as io

    samples = io.load_d4()
    picks = []
    multi = next((s for s in samples if s["file_name"].startswith("Test1")), None)
    if multi:
        picks.append(multi)
    for sample in samples[:30]:
        if len(picks) >= 2:
            break
        if multi and sample["file_name"] == multi["file_name"]:
            continue
        try:
            pre = preprocess.preprocess(io.imread(sample["path"]))
        except ValueError:
            continue
        if pre["calib"]["a0"] < 60:
            picks.append(sample)

    # One row: stacked, the two examples came to 13.8 cm on the page and no longer fitted
    # beside the text that introduces them.
    fig, axes = plt.subplots(1, 2 * len(picks), figsize=(9.6, 2.9), squeeze=False)
    for index, sample in enumerate(picks):
        row, left = 0, 2 * index
        image = io.imread(sample["path"])
        _show(axes[row][left], cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
              f"原图（真值 {sample['gt_count']} 粒）")
        try:
            pre = preprocess.preprocess(image)
            calib, mask = pre["calib"], pre["mask"] > 0
            height, width = mask.shape
            border = np.zeros_like(mask)
            margin_h, margin_w = int(0.18 * height), int(0.18 * width)
            border[:margin_h], border[-margin_h:] = True, True
            border[:, :margin_w], border[:, -margin_w:] = True, True
            share = mask[border].sum() / max(mask.sum(), 1)
            _show(axes[row][left + 1], pre["mask"],
                  f"选出的前景：{share * 100:.0f}% 在直尺处", "gray")
        except ValueError:
            axes[row][left + 1].axis("off")
            axes[row][left + 1].text(0.5, 0.5, "无可用二值化候选", ha="center", va="center")

    fig.tight_layout()
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name



def appendix_renders_figure(dataset, n_scenes=2, prompt="white seed", threshold=0.40,
                            teacher=None):
    """Scenes from one dataset, counted by this method and by SAM 3, one file per scene.

    Scenes are taken evenly across the set rather than chosen, so the appendix shows what the
    methods usually do rather than what they do at their best. Counts go in the panel titles
    because the drawing itself carries no text. One file per scene, named ``_a``, ``_b``, ...,
    so each can be placed next to the text that discusses it.
    """
    from src import render, synth
    from src import io_utils as io

    loaders = {"d1": io.load_d1, "d2": synth.load_d2, "d3": io.load_d3}
    samples = loaders[dataset]()
    step = max(len(samples) // n_scenes, 1)
    picked = [samples[i * step] for i in range(n_scenes) if i * step < len(samples)]

    if teacher is None:
        from src.teacher_sam import Sam3Teacher
        teacher = Sam3Teacher()

    ensure_dir(FIG_ROOT)
    written = []
    for index, sample in enumerate(picked):
        image = io.imread(sample["path"])
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))

        labels, info, total, _ = counter.label_image(image)
        _show(axes[0], cv2.cvtColor(render.draw(image, labels, info, None), cv2.COLOR_BGR2RGB),
              f"本文方法：{total} 粒（真值 {sample['gt_count']} 粒）")

        result = teacher.segment(image, prompt=prompt, threshold=threshold)
        keep = np.asarray(result["scores"].float().cpu()) >= threshold
        masks = np.asarray(result["masks"].float().cpu())[keep]
        sam_labels = render.masks_to_labels(masks, image.shape[:2])
        sam_info = {i: {"kind": "grain", "count": 1}
                    for i in range(1, int(sam_labels.max()) + 1)}
        _show(axes[1],
              cv2.cvtColor(render.draw(image, sam_labels, sam_info, None), cv2.COLOR_BGR2RGB),
              f"SAM 3：{int(keep.sum())} 粒（真值 {sample['gt_count']} 粒）")

        fig.tight_layout()
        out_path = FIG_ROOT / f"appendix_{dataset}_{chr(ord('a') + index)}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
    return written


def abstract_figure(out_name="graphical_abstract.png"):
    """首页图文摘要：上排两个场景的逐粒结果，下排两幅定量图。

    题目问的是"数出多少粒"和"解决连体计数不准"，这几幅各答一问：
    上排说明程序把哪些区域算作了几粒，左下说明粘连加重时误差并未跟着涨，
    右下把四种做法放在同一批图上比较。

    两幅定量图共用一套配色，同一种方法在哪里都是同一个颜色；
    字号统一由 rcParams 给出，不在各面板单独设置。
    """
    from src import counter, render, synth

    real = io_utils.load_d1()[0]
    dense = max(synth.load_d2(), key=lambda s: s.get("touch_prob") or 0)

    with plt.rc_context(_fonts(ABSTRACT_FONT_SCALE)):
        fig = plt.figure(figsize=(9.8, 8.4))
        grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.2, wspace=0.22)

        for column, (sample, label) in enumerate((
                (real, "真实照片"),
                (dense, f"合成图，粘连率 {int((dense.get('touch_prob') or 0) * 100)}%"))):
            image = io_utils.imread(sample["path"])
            labels, info, total, _ = counter.label_image(image)
            drawn = render.draw(image, labels, info, banner_lines=None)
            _show(fig.add_subplot(grid[0, column]),
                  cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB),
                  f"{label}：数出 {total} 粒（真值 {sample['gt_count']} 粒）")

        _abstract_curve(fig.add_subplot(grid[1, 0]))
        _abstract_bars(fig.add_subplot(grid[1, 1]))

    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def _tidy(ax):
    """去掉上右边框、让网格退到背景里：坐标系是背景，数据才是主角。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, length=3)
    ax.grid(color=GRID, alpha=0.6, lw=0.6)
    ax.set_axisbelow(True)


def _abstract_curve(ax):
    """误差随粘连率的变化。只留两条线，首页要的是对比清楚。"""
    per_image = pd.read_csv(METRICS_ROOT / "per_image.csv")
    d2 = per_image[(per_image.dataset == "d2") & per_image.touch_prob.notna()].copy()
    d2["abs_err"] = d2["error"].abs()
    curve = d2.groupby(["touch_prob", "method"])["abs_err"].mean().unstack()

    for method, label, colour, style in (
            ("B1_components", "直接数连通域", SERIES_COLORS["baseline"], "--"),
            ("Ours_ASW_SPC", "方法一", SERIES_COLORS["ours"], "-")):
        if method in curve:
            ax.plot(curve.index * 100, curve[method], label=label, color=colour,
                    ls=style, lw=2.0, marker="o", markersize=5.5)
    ax.set_xlabel("粘连率／%", color=INK)
    ax.set_ylabel("平均绝对误差／粒", color=INK)
    ax.set_title("误差随粘连程度的变化", color=INK)
    _tidy(ax)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)


def student_bars_figure(out_name="student_bars.png"):
    """把图文摘要里的柱状图单独出一张，供仓库的 README 使用。"""
    with plt.rc_context(_fonts(PLOT_FONT_SCALE)):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        _abstract_bars(ax)
        fig.tight_layout()
        ensure_dir(FIG_ROOT)
        fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    return FIG_ROOT / out_name


def _abstract_bars(ax):
    """四种做法在同一批测试图上的误差。

    同一批是关键：学生模型只在留出集上测过，把它的数字与另外几种在全集上的
    数字并排，比较就不成立了，所以这里四者用的都是那 162 张测试图。
    """
    table = pd.read_csv(METRICS_ROOT / "student_test.csv")
    series = [(k, label, SERIES_COLORS[key]) for k, label, key in (
        ("b1", "直接数连通域", "baseline"), ("ours", "方法一", "ours"),
        ("student", "方法二：学生模型", "student"), ("sam3", "SAM 3", "sam3"))
        if k in table.columns]
    datasets = [d for d in ("d1", "d2", "d3") if (table.dataset == d).any()]

    width = 0.78 / len(series)
    tallest = (0.0, None)
    for index, (key, label, colour) in enumerate(series):
        values = [float((table[table.dataset == d][key]
                         - table[table.dataset == d].truth).abs().mean()) for d in datasets]
        offsets = [i + index * width - 0.39 + width / 2 for i in range(len(datasets))]
        # 0.92 的宽度留出柱间缝隙，相邻柱子不会糊在一起
        ax.bar(offsets, values, width * 0.92, label=label, color=colour)
        for x, value in zip(offsets, values):
            if value > tallest[0]:
                tallest = (value, (x, value))

    # 只标注最高的一根：它是这幅图要说的事（不处理粘连时误差有多大），
    # 其余数值在表 9 中列全，不必每根柱子都写数字。
    if tallest[1]:
        x, value = tallest[1]
        ax.annotate(f"{value:.1f} 粒", xy=(x, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK,
                    fontsize=plt.rcParams["xtick.labelsize"])

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([dataset_label(d).replace("（", "\n（") for d in datasets])
    ax.set_ylabel("平均绝对误差／粒", color=INK)
    ax.set_title("四种做法在同一批测试图上的误差", color=INK)
    ax.set_ylim(0, tallest[0] * 1.12 if tallest[0] else 1)
    _tidy(ax)
    ax.grid(axis="x", visible=False)
    # 四条中文图例放在图内总会压到柱子或标注，改为置于图下横排两列
    ax.legend(frameon=False, labelcolor=INK, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.20), columnspacing=1.4,
              handlelength=1.2, handletextpad=0.5)


# --------------------------------------------------- 与正文对照的局部说明图
#
# 这一组图的用途与流程图、结果图不同：正文里凡是用文字描述一种现象或一条判据的
# 地方，就在旁边给出把该现象放大出来的实图，让读者不必只凭文字想象。

def _crop_around(image, box, pad_ratio=0.45):
    """按外接框裁一块带留白的局部，返回裁剪图与裁剪框。"""
    r0, c0, r1, c1 = box
    pad = int(max(r1 - r0, c1 - c0) * pad_ratio) + 4
    top, left = max(r0 - pad, 0), max(c0 - pad, 0)
    bottom = min(r1 + pad, image.shape[0])
    right = min(c1 + pad, image.shape[1])
    return image[top:bottom, left:right], (top, left, bottom, right)


def touching_problem_figure(out_name="problem_zoom.png"):
    """引言用图：把'两粒挨在一起就只被计为一粒'这件事放大给读者看。"""
    from src import synth
    sample = max(synth.load_d2(), key=lambda s: s.get("touch_prob") or 0)
    image = io_utils.imread(sample["path"])
    pre = preprocess.preprocess(image)
    calib = pre["calib"]

    # 找一个正好由两三粒粘成的连通域，太大的块反而看不清接缝
    target = min((c for c in calib["components"]
                  if 1.8 * calib["a0"] <= c["area"] <= 3.2 * calib["a0"]),
                 key=lambda c: abs(c["area"] - 2.2 * calib["a0"]), default=None)
    if target is None:
        return None

    photo, box = _crop_around(image, target["bbox"])
    labels = calib["labels"][box[0]:box[2], box[1]:box[3]]
    binary = (labels > 0).astype(np.uint8) * 255
    tinted = np.dstack([binary] * 3)
    tinted[labels == target["label"]] = (255, 120, 60)   # 整块同色，说明它是一个连通域

    with plt.rc_context(_fonts(ABSTRACT_FONT_SCALE)):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
        _show(axes[0], cv2.cvtColor(photo, cv2.COLOR_BGR2RGB), "原图局部，这里有 2 粒米")
        _show(axes[1], binary, "二值化之后", "gray")
        _show(axes[2], tinted, "两粒连成一个连通域，只被计为 1 粒")
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def binarisation_failure_figure(out_name="binarisation_failure.png"):
    """3.2 用图：前景取亮还是取暗，都会多出一块巨大的伪前景。

    两种极性各画一张，并把最大连通域涂成橙色，读者一眼能看出多出来的是什么。
    """
    sample = io_utils.load_d1()[0]
    image = io_utils.imread(sample["path"])

    panels = []
    for polarity in ("bright", "dark"):
        cand = next(c for c in preprocess.candidate_masks(image, channel_names=("gray",))
                    if c["method"] == "otsu" and c["polarity"] == polarity)
        binary = (cand["mask"] > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1 if count > 1 else 0
        share = stats[biggest, cv2.CC_STAT_AREA] / binary.size if biggest else 0.0

        view = np.dstack([binary * 255] * 3)
        if biggest:
            view[labels == biggest] = (255, 140, 60)
        name = "取亮的一类作前景" if polarity == "bright" else "取暗的一类作前景"
        panels.append((view, f"{name}\n{count - 1} 个连通域，最大块占 {share * 100:.0f}%"))

    with plt.rc_context(_fonts(ABSTRACT_FONT_SCALE)):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.0))
        _show(axes[0], cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
              "原图\n白米、深色衬布、浅色桌面")
        for ax, (view, title) in zip(axes[1:], panels):
            _show(ax, view, title)
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def touching_criterion_figure(out_name="touching_criterion.png"):
    """3.4 用图：单粒与粘连块在凸包和最大内切圆上的差别，一眼可辨。"""
    from skimage.morphology import convex_hull_image
    from src import synth
    sample = max(synth.load_d2(), key=lambda s: s.get("touch_prob") or 0)
    image = io_utils.imread(sample["path"])
    pre = preprocess.preprocess(image)
    calib = pre["calib"]
    band = calibrate.SINGLE_BAND

    single = min((c for c in calib["components"]
                  if band[0] * calib["a0"] <= c["area"] <= band[1] * calib["a0"]),
                 key=lambda c: abs(c["area"] - calib["a0"]), default=None)
    # 取凹口最深的那一块，凸实度接近 1 的粘连块说明不了问题
    cluster = min((c for c in calib["components"]
                   if 1.8 * calib["a0"] <= c["area"] <= 3.2 * calib["a0"]),
                  key=lambda c: c["solidity"], default=None)
    if single is None or cluster is None:
        return None

    with plt.rc_context(_fonts(ABSTRACT_FONT_SCALE)):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
        for ax, component, name in ((axes[0], single, "单粒"), (axes[1], cluster, "两粒粘连")):
            piece = (calib["labels"] == component["label"])
            sub, _ = _crop_around(piece.astype(np.uint8), component["bbox"], pad_ratio=0.3)
            hull = convex_hull_image(sub > 0)
            canvas = np.zeros(sub.shape + (3,), np.uint8)
            canvas[hull] = (250, 226, 180)          # 凸包，浅色
            canvas[sub > 0] = (60, 90, 160)         # 区域本身
            dist = cv2.distanceTransform((sub > 0).astype(np.uint8), cv2.DIST_L2, 5)
            radius = float(dist.max())
            cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
            ax.imshow(canvas)
            circle = plt.Circle((cx, cy), radius, fill=False, color="#D55E00", lw=1.8)
            ax.add_patch(circle)
            ax.set_title(f"{name}\n凸实度 {component['solidity']:.2f}，"
                         f"最大内切圆直径 {2 * radius:.0f} 像素",
                         fontsize=plt.rcParams["axes.titlesize"])
            ax.axis("off")
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def seed_depth_figure(out_name="seed_depth.png"):
    """3.5 用图：一粒内部是浅坑、两粒之间是深谷，所以该看深度而不是距离。"""
    from src import synth
    sample = max(synth.load_d2(), key=lambda s: s.get("touch_prob") or 0)
    image = io_utils.imread(sample["path"])
    pre = preprocess.preprocess(image)
    calib = pre["calib"]

    # 挑一个确实被切成两半的粘连块，只出一个种子的块说明不了"深谷"这件事
    h = segment.BETA * calib["minor0"] / 2.0
    chosen = None
    for component in sorted((c for c in calib["components"]
                             if 1.6 * calib["a0"] <= c["area"] <= 3.2 * calib["a0"]),
                            key=lambda c: c["solidity"]):
        piece = (calib["labels"] == component["label"]).astype(np.uint8)
        sub, _ = _crop_around(piece, component["bbox"], pad_ratio=0.25)
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        markers = segment.adaptive_markers(dist, calib["minor0"])
        if int(markers.max()) >= 2:
            chosen = (sub, dist, markers)
            break
    if chosen is None:
        return None
    sub, dist, markers = chosen

    centres = [tuple(np.mean(np.nonzero(markers == i), axis=1))
               for i in range(1, int(markers.max()) + 1)]
    peaks = sorted(centres, key=lambda rc: -dist[int(rc[0]), int(rc[1])])[:2]
    (r0, c0), (r1, c1) = peaks
    steps = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    rows = np.linspace(r0, r1, steps).astype(int)
    cols = np.linspace(c0, c1, steps).astype(int)
    profile = dist[rows, cols]

    with plt.rc_context(_fonts(ABSTRACT_FONT_SCALE)):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                 gridspec_kw={"width_ratios": [1, 1.5]})
        axes[0].imshow(dist, cmap="magma")
        axes[0].plot([c0, c1], [r0, r1], color="#00A0B0", lw=1.8, ls="--")
        axes[0].scatter([c0, c1], [r0, r1], s=26, color="#00A0B0")
        axes[0].set_title("粘连块的距离变换\n虚线连接两个种子，即右图剖面的取法",
                          fontsize=plt.rcParams["axes.titlesize"])
        axes[0].axis("off")

        ax = axes[1]
        ax.plot(profile, color=SERIES_COLORS["ours"], lw=2.0)
        ax.axhline(h, color=SERIES_COLORS["baseline"], ls="--", lw=1.6,
                   label=f"h-maxima 的深度门限 h = {h:.1f}")
        ax.set_xlabel("自一个种子到另一个种子的位置／像素")
        ax.set_ylabel("到背景的距离／像素")
        ax.set_title("两粒之间是一道深谷\n谷底低于门限，两个峰才会被分开",
                     fontsize=plt.rcParams["axes.titlesize"])
        _tidy(ax)
        ax.legend(frameon=False, labelcolor=INK, loc="lower center")
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def b3_seed_figure(out_name="b3_seeds.png"):
    """4.3 用图：B3 的全局阈值被硬币抬高，米粒的种子被整片抹掉。"""
    from src import baselines
    sample = io_utils.load_d1()[0]
    image = io_utils.imread(sample["path"])
    pre = preprocess.preprocess(image)
    mask, calib = pre["mask"], pre["calib"]

    dist = segment.distance_transform(mask)
    b3_seeds = (dist > 0.5 * dist.max())

    ours = np.zeros(mask.shape, bool)
    for component in calib["components"]:
        if component["area"] < counter.SPECK_RATIO * calib["a0"]:
            continue
        r0, c0, r1, c1 = component["bbox"]
        piece = (calib["labels"][r0:r1, c0:c1] == component["label"]).astype(np.uint8)
        local = cv2.distanceTransform(piece, cv2.DIST_L2, 5)
        markers = segment.adaptive_markers(local, calib["minor0"])
        ours[r0:r1, c0:c1] |= markers > 0

    size = max(5, mask.shape[1] // 40) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    views = []
    for seeds in (b3_seeds, ours):
        grown = cv2.dilate(seeds.astype(np.uint8), kernel)
        view = np.dstack([mask // 3] * 3)
        view[grown > 0] = (255, 60, 60)
        views.append(view)

    with plt.rc_context(_fonts(ABSTRACT_FONT_SCALE)):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
        _show(axes[0], views[0],
              f"B3 的全局阈值，只剩 {int(label(b3_seeds).max())} 个种子")
        _show(axes[1], views[1], f"本文的自适应种子，共 {int(label(ours).max())} 个")
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def main():
    from src import synth

    d1 = io_utils.load_d1()
    d2 = synth.load_d2()
    dense = max(d2, key=lambda s: s["touch_prob"])

    # The calibration figure needs a scene that still holds both single grains and clusters:
    # at the heaviest touching level most grains have merged, leaving too few components for
    # the area distribution to show the structure the figure is there to show. The scene
    # with the most components is the one that shows it.
    mixed = max(
        (s for s in d2 if 0.3 <= (s.get("touch_prob") or 0) <= 0.6),
        key=lambda s: len(preprocess.preprocess(io_utils.imread(s["path"]))["calib"]["components"]),
        default=dense,
    )

    outputs = [
        pipeline_figure(io_utils.imread(dense["path"]), "pipeline_synthetic.png"),
        pipeline_figure(io_utils.imread(d1[0]["path"]), "pipeline_real.png"),
        cluster_detail_figure(io_utils.imread(dense["path"]), "cluster_detail.png"),
        calibration_figure(io_utils.imread(mixed["path"]), "calibration.png"),
        error_curve_figure(),
        scatter_figure(),
        student_bars_figure(),
    ]
    for path in outputs:
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
