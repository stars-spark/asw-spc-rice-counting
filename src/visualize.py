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
    # 柱顶数值用粗体，Latin Modern 的粗体是单独一个文件
    "~/.local/share/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-bold.otf",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-bold.otf",
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

    # 最后一格与其它效果图同一种画法，每粒米涂一种颜色，切开的粘连块也分成几种颜色
    from src import render
    labels, info, _, _ = counter.label_image(image_bgr, pre=pre)
    result = cv2.cvtColor(render.draw(image_bgr, labels, info, banner_lines=None),
                          cv2.COLOR_BGR2RGB)

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


def _tidy(ax):
    """去掉上右边框、让网格退到背景里：坐标系是背景，数据才是主角。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, length=3)
    ax.grid(color=GRID, alpha=0.6, lw=0.6)
    ax.set_axisbelow(True)


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

    两种极性各画一张，并把最大连通域涂成蓝色，读者一眼能看出多出来的是什么。
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
            view[labels == biggest] = (0, 114, 178)
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


def terrain_figure(out_name="terrain_3d.png", elev=30, azim=-58, z_exaggeration=0.65):
    """把距离变换当成地形来看。

    注水分割真正淹没的曲面是距离变换的负值。米粒内部离背景远，取负之后陷成一个盆地；
    两粒贴合处离背景近，取负之后隆成一道埂。一粒米一个盆地，水自盆底的种子漫上来，
    在埂上相遇，相遇的地方就是切分线。3.1 节用文字描述的这件事，这张图直接画了出来。

    选的是一个真被切开的两粒粘连块，而不是按面积补数的那种：只有前者的地形上
    才既有两个盆地、又有一条切分线。
    """
    from src import synth

    scene = max(synth.load_d2(), key=lambda s: s["touch_prob"])
    image = io_utils.imread(scene["path"])
    pre = preprocess.preprocess(image)
    _, _, debug = counter.count_rice(image, pre=pre, return_debug=True)

    def parts_of(cluster):
        return len(np.unique(np.asarray(cluster["merged"]))) - 1

    cut = [c for c in debug["clusters"]
           if parts_of(c) == c["count"] == int(np.asarray(c["markers"]).max()) >= 2]
    cluster = max(cut or debug["clusters"], key=lambda c: c["component"]["area"])

    pad = 8
    dist = np.pad(np.asarray(cluster["dist"], dtype=np.float32), pad)
    mask = np.pad(np.asarray(cluster["mask"]) > 0, pad)
    merged = np.pad(np.asarray(cluster["merged"]), pad)
    markers = np.pad(np.asarray(cluster["markers"]), pad)
    panels = [mask.astype(np.float32), None, label2rgb(merged, bg_label=0)]
    titles = ["(a) 粘连块的二值图", "(b) 取距离变换的负值当作地形",
              f"(c) 切开后为 {cluster['count']} 粒"]

    # 三维面板先单独渲染成图片，再与另两格一起按普通图像排版。
    # 地形是这张图的主角，占右侧一大格；二值图与切分结果上下叠在左侧，
    # (a) 与 (b) 顶边对齐、标题落在同一条线上。三维图按最终显示尺寸渲染，
    # 刻度与图例的字号因此与报告里其它插图一致。
    side_aspect = mask.shape[1] / mask.shape[0]
    total, gap, title_gap, top_space = 9.0, 0.25, 0.42, 0.40
    size = (5.6, 4.0)
    for _ in range(3):
        terrain = _render_terrain(dist, mask, merged, markers, elev, azim,
                                  z_exaggeration, size)
        mid_aspect = terrain.shape[1] / terrain.shape[0]
        # 右格宽 wr、高 wr/mid_aspect；左侧两格各高 (右格高 - title_gap)/2
        wr = (total - gap + side_aspect * title_gap / 2) / (1 + side_aspect / (2 * mid_aspect))
        rendered = terrain.shape[1] / DPI
        if abs(wr / rendered - 1) < 0.03:
            break
        size = (size[0] * wr / rendered, size[1] * wr / rendered)
    panels[1] = terrain
    hr = wr / mid_aspect
    hl = (hr - title_gap) / 2
    wl = hl * side_aspect

    fig = plt.figure(figsize=(total, hr + top_space))
    fw, fh = total, hr + top_space
    boxes = [(0, hr - hl, wl, hl), (total - wr, 0, wr, hr), (0, 0, wl, hl)]
    for index, ((x, y, w, h), image, title) in enumerate(zip(boxes, panels, titles)):
        ax = fig.add_axes([x / fw, y / fh, w / fw, h / fh])
        _show(ax, image, title, "gray" if index == 0 else None)

    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    _trim_white(FIG_ROOT / out_name)
    return FIG_ROOT / out_name


def _render_terrain(dist, mask, merged, markers, elev, azim, z_exaggeration, size):
    """把 -D 画成三维地形，返回裁掉白边后的 RGB 数组。"""
    import io

    from matplotlib.lines import Line2D
    from PIL import Image

    rows, cols = np.mgrid[0:dist.shape[0], 0:dist.shape[1]]
    depth = dist.max()
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    # 关掉自动深度排序，否则插在盆底的种子杆会被曲面整片挡住
    ax.computed_zorder = False
    ax.plot_surface(cols, rows, -dist, rstride=1, cstride=1, linewidth=0, antialiased=True,
                    facecolors=plt.cm.magma(dist / max(depth, 1e-6)), shade=False, zorder=1)

    # 种子画成从盆底竖到地面的一根杆，否则俯视时会被盆壁挡住
    # 一个种子常占好几个像素，每个种子只取其中最深的一点画一根杆
    seeds = []
    for value in np.unique(markers[markers > 0]):
        spots = np.argwhere(markers == value)
        seeds.append(spots[np.argmax(dist[spots[:, 0], spots[:, 1]])])
    seeds = np.array(seeds).reshape(-1, 2)
    for row, col in seeds:
        ax.plot([col, col], [row, row], [-dist[row, col], 0.6],
                color=SERIES_COLORS["sam3"], lw=1.0, zorder=4)
    ax.scatter(seeds[:, 1], seeds[:, 0], np.full(len(seeds), 0.6),
               color=SERIES_COLORS["sam3"], s=26, depthshade=False, zorder=5)
    # 只要两块之间的那道分界，不要粘连块自身的外轮廓
    inner = find_boundaries(merged, mode="thick") & mask
    inner &= ~find_boundaries(mask.astype(np.int32), mode="thick")
    line = np.argwhere(inner)
    if len(line):
        ax.scatter(line[:, 1], line[:, 0], -dist[line[:, 0], line[:, 1]] + 0.6,
                   color=SERIES_COLORS["baseline"], s=5, depthshade=False, zorder=3)

    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((dist.shape[1], dist.shape[0], z_exaggeration * max(dist.shape)))
    ax.set_zlim(-depth * 1.12, depth * 0.10)
    # 画出三条坐标轴与三个参考面，读者能看出地形的尺度：平面方向是像素位置，
    # 竖直方向是离背景的距离，米粒最深处约比地面低 {depth} 个像素
    step = 20 if max(dist.shape) > 60 else 10
    ax.set_xticks(np.arange(0, dist.shape[1], step))
    ax.set_yticks(np.arange(0, dist.shape[0], step))
    ax.set_zticks([0, -round(depth / 2), -round(depth)])
    ax.tick_params(axis="both", pad=0, labelsize=6.5 * FONT_SCALE, colors=INK)
    ax.set_xlabel("x / 像素", labelpad=2, fontsize=7.5 * FONT_SCALE)
    ax.set_ylabel("y / 像素", labelpad=2, fontsize=7.5 * FONT_SCALE)
    ax.set_zlabel("深度 / 像素", labelpad=2, fontsize=7.5 * FONT_SCALE)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
        axis.pane.set_edgecolor(GRID)
        axis.line.set_color(INK)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.5)
    ax.grid(True)
    handles = [Line2D([], [], color=SERIES_COLORS["sam3"], lw=1.2, marker="o", markersize=4,
                      label="种子"),
               Line2D([], [], color=SERIES_COLORS["baseline"], lw=0, marker="o", markersize=3,
                      label="切分线")]
    # 图例横排放在三维框的正上方，不压住地形
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=2,
              frameon=False, fontsize=7.5 * FONT_SCALE, handlelength=1.2,
              columnspacing=1.2, borderaxespad=0)

    buffer = io.BytesIO()
    # 坐标轴标签会伸出画布，按实际内容裁切而不是按画布
    fig.savefig(buffer, dpi=DPI, format="png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    buffer.seek(0)
    image = np.asarray(Image.open(buffer).convert("RGB"))
    ink = np.argwhere((image < 245).any(axis=2))
    (top, left), (bottom, right) = ink.min(axis=0), ink.max(axis=0)
    return image[max(top - 4, 0):bottom + 5, max(left - 4, 0):right + 5]


def _trim_white(path, margin=8):
    """三维坐标框即使隐藏也会占位，存图后把四周多余的白边裁掉。"""
    from PIL import Image
    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image)
    ink = np.argwhere((pixels < 245).any(axis=2))
    if len(ink):
        (top, left), (bottom, right) = ink.min(axis=0), ink.max(axis=0)
        image.crop((max(left - margin, 0), max(top - margin, 0),
                    min(right + margin, image.width), min(bottom + margin, image.height))
                   ).save(path)


# ------------------------------------------------------------------ 方法二用图
# 这三张图与前面的插图共用 _show、_crop_around 与同一套配色，
# 密度图一律用 viridis，点标注一律用下面这三种颜色，读者在三张图之间不必重新适应。

STUDENT_PIPELINE_IMAGE = "WIN_20240127_13_47_18_Pro_jpg.rf.0166d2906ccb16ad5d7fc3b284356c7c.jpg"
STUDENT_WIN_IMAGE = "WIN_20240126_11_50_40_Pro_jpg.rf.6955042005db49274dd07b952ea0d5ec.jpg"
TEACHER_LABEL_IMAGE = "IMG_5974_JPG.rf.01d37e6f9e5fa5d90b047c1eb803cc7a.jpg"

POINT_COLORS = {"matched": "#009E73", "missed": "#D55E00", "extra": "#0072B2"}
DENSITY_CMAP = "viridis"
MASK_CACHE = RESULTS_ROOT / "labels" / "fig_sam3_masks.npz"


def _student_model():
    """载入训练好的学生模型。torch 只在用到时导入，没有显卡的机器照样能跑其余插图。"""
    import torch

    from src.student import MODEL_ROOT, UNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(MODEL_ROOT / "student_sam3.pt", map_location=device)
    model = UNet(width=checkpoint["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, device


def _student_density(model, image_bgr, device):
    """学生模型输出的密度图，其积分即为粒数。"""
    import torch
    import torch.nn.functional as F

    from src.student import DENSITY_SCALE
    with torch.no_grad():
        x = np.ascontiguousarray(image_bgr[:, :, ::-1]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
        pad_h, pad_w = (-x.shape[-2]) % 4, (-x.shape[-1]) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        out = model(x)[0, 0].detach().cpu().numpy() / DENSITY_SCALE
    return out[:image_bgr.shape[0], :image_bgr.shape[1]]


def _sam3_masks(sample):
    """取一张图的 SAM 3 掩膜。算过一次就存下来，插图不必每次都启动教师模型。"""
    key = sample["file_name"]
    cache = {}
    if MASK_CACHE.exists():
        with np.load(MASK_CACHE) as data:
            cache = {k: data[k] for k in data.files}
    if key in cache:
        return cache[key]

    from src import pseudo
    from src.teacher_sam import Sam3Teacher
    teacher = Sam3Teacher()
    result = teacher.segment(io_utils.imread(sample["path"]),
                             prompt=pseudo.PROMPT, threshold=pseudo.THRESHOLD)
    scores = np.asarray(result["scores"].float().cpu())
    masks = np.asarray(result["masks"].float().cpu())[scores >= pseudo.THRESHOLD]
    cache[key] = masks.astype(np.uint8)
    ensure_dir(MASK_CACHE.parent)
    np.savez_compressed(MASK_CACHE, **cache)
    return cache[key]


def _match_points(truth, pred):
    """把教师给的点与人工标注配对，返回配对上的、漏掉的与多出的三组下标。

    判据与 pseudo.compare_to_human 相同，容差取人工标注最近邻间距中位数的一半。
    """
    if len(truth) == 0 or len(pred) == 0:
        return [], list(range(len(truth))), list(range(len(pred)))
    gaps = np.linalg.norm(truth[:, None, :] - truth[None, :, :], axis=2)
    np.fill_diagonal(gaps, np.inf)
    limit = float(np.median(gaps.min(axis=1)) / 2.0) if len(truth) > 1 else 10.0

    distance = np.linalg.norm(truth[:, None, :] - pred[None, :, :], axis=2)
    used, matched, missed = set(), [], []
    for i in np.argsort(distance.min(axis=1)):
        order = np.argsort(distance[i])
        hit = next((int(j) for j in order if j not in used and distance[i, j] <= limit), None)
        if hit is None:
            missed.append(int(i))
        else:
            used.add(hit)
            matched.append((int(i), hit))
    extra = [j for j in range(len(pred)) if j not in used]
    return matched, missed, extra


def _scatter_points(ax, points, colour, size=14, marker="o"):
    if len(points):
        ax.scatter(points[:, 1], points[:, 0], s=size, marker=marker,
                   facecolors="none", edgecolors=colour, linewidths=1.1)


def student_pipeline_figure(out_name="student_pipeline.png", file_name=None):
    """方法二的四步：教师的掩膜、掩膜质心、由点摊成的密度图、学生的预测。"""
    from src import pseudo
    from src.student import density_map

    samples = {s["file_name"]: s for s in io_utils.load_d3()}
    name = file_name or STUDENT_PIPELINE_IMAGE
    sample = samples[name]
    image = io_utils.imread(sample["path"])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    points = np.asarray(pseudo.load("d3_sam3")[name], dtype=np.float32)

    try:
        masks = _sam3_masks(sample)
    except Exception:
        masks = None

    target = density_map(image.shape, points)
    model, device = _student_model()
    predicted = _student_density(model, image, device)

    fig, axes = plt.subplots(1, 5, figsize=(9, 2.7))
    _show(axes[0], rgb, "(a) 输入图像")

    if masks is not None and len(masks):
        overlay = label2rgb(render_labels_from_masks(masks, image.shape[:2]),
                            image=rgb, bg_label=0, alpha=0.45)
        _show(axes[1], overlay, f"(b) SAM 3 的掩膜\n{len(masks)} 个")
    else:
        _show(axes[1], rgb, "(b) SAM 3 的掩膜")

    _show(axes[2], rgb, f"(c) 取每个掩膜的质心\n得到 {len(points)} 个点")
    _scatter_points(axes[2], points, POINT_COLORS["matched"])
    _show(axes[3], target, f"(d) 摊成密度图\n积分 {target.sum():.1f}", DENSITY_CMAP)
    _show(axes[4], predicted, f"(e) 学生的预测\n积分 {predicted.sum():.1f}", DENSITY_CMAP)

    fig.tight_layout()
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def render_labels_from_masks(masks, shape):
    from src import render
    return render.masks_to_labels(masks.astype(np.float32), shape)


def teacher_labels_figure(out_name="teacher_labels.png", file_name=None):
    """教师的点与人工标注的对照，右侧放大一处漏检。"""
    from src import pseudo

    name = file_name or TEACHER_LABEL_IMAGE
    human = pseudo.load("d1_human")
    teacher = pseudo.load("d1_sam3")
    if name not in human:
        name = sorted(human)[0]
    sample = {s["file_name"]: s for s in io_utils.load_d1()}[name]
    image = cv2.cvtColor(io_utils.imread(sample["path"]), cv2.COLOR_BGR2RGB)

    truth = np.asarray(human[name], dtype=np.float32)
    pred = np.asarray(teacher[name], dtype=np.float32)
    matched, missed, extra = _match_points(truth, pred)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    _show(axes[0], image,
          f"人工标注 {len(truth)} 粒，SAM 3 给出 {len(pred)} 粒\n"
          f"配对上 {len(matched)} 粒，漏 {len(missed)} 粒，多 {len(extra)} 点")
    _scatter_points(axes[0], truth[[i for i, _ in matched]], POINT_COLORS["matched"], size=10)
    _scatter_points(axes[0], truth[missed], POINT_COLORS["missed"], size=40)
    _scatter_points(axes[0], pred[extra], POINT_COLORS["extra"], size=40, marker="s")

    focus = truth[missed[0]] if missed else (truth[extra[0]] if extra else truth[0])
    half = max(image.shape[0], image.shape[1]) // 12
    box = (int(focus[0] - half), int(focus[1] - half),
           int(focus[0] + half), int(focus[1] + half))
    crop, (top, left, bottom, right) = _crop_around(image, box, pad_ratio=0.0)
    _show(axes[1], crop, "局部放大\n橙圈为漏掉的米粒，蓝方块为多给的点")
    inside = lambda pts: np.array([p for p in pts
                                   if top <= p[0] < bottom and left <= p[1] < right],
                                  dtype=np.float32).reshape(-1, 2) - [top, left]
    _scatter_points(axes[1], inside(truth[[i for i, _ in matched]]),
                    POINT_COLORS["matched"], size=60)
    _scatter_points(axes[1], inside(truth[missed]), POINT_COLORS["missed"], size=90)
    _scatter_points(axes[1], inside(pred[extra]), POINT_COLORS["extra"], size=90, marker="s")

    fig.tight_layout()
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


def student_vs_geometric_figure(out_name="student_vs_ours.png", file_name=None):
    """同一张低分辨率图上，方法一把木框亮边算成了米，学生模型没有。"""
    from src import render

    samples = {s["file_name"]: s for s in io_utils.load_d3()}
    name = file_name or STUDENT_WIN_IMAGE
    sample = samples[name]
    image = io_utils.imread(sample["path"])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    labels, info, total, _ = counter.label_image(image)
    drawn = cv2.cvtColor(render.draw(image, labels, info, banner_lines=None),
                         cv2.COLOR_BGR2RGB)
    model, device = _student_model()
    predicted = _student_density(model, image, device)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
    _show(axes[0], rgb, f"输入图像，真值 {sample['gt_count']} 粒")
    _show(axes[1], drawn, f"方法一数出 {total} 粒")
    _show(axes[2], predicted,
          f"方法二的密度图，积分 {predicted.sum():.1f} 粒", DENSITY_CMAP)

    fig.tight_layout()
    ensure_dir(FIG_ROOT)
    fig.savefig(FIG_ROOT / out_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return FIG_ROOT / out_name


# ------------------------------------------------------------------ 数据图
# 这一组都按 plotstyle 的规矩画：画布即版心宽、字号即印刷字号、输出 PDF 矢量图，
# 系列的颜色与标记整项取自 plotstyle.SERIES，保存前查宽度与重叠，不合格不写文件。

def _data_fig(height, **kw):
    from src import plotstyle as ps
    old = ps.apply()
    fig, axes = plt.subplots(figsize=(ps.TEXT_WIDTH_IN, height), **kw)
    fig._rc_before = old
    return fig, axes


def _save_data_fig(fig, out_name):
    """保存后把 rcParams 还原，同一进程里接着画的图像类插图不受影响。"""
    from src import plotstyle as ps
    ensure_dir(FIG_ROOT)
    try:
        return ps.save_checked(fig, FIG_ROOT / out_name)
    finally:
        plt.close(fig)
        matplotlib.rcParams.update(getattr(fig, "_rc_before", {}))


def calibration_figure(image_bgr, out_name="calibration.pdf"):
    """对数面积的分布，标出 A0 及其 2、3 倍。"""
    from src import plotstyle as ps
    pre = preprocess.preprocess(image_bgr)
    calib = pre["calib"]
    areas = np.array([c["area"] for c in calib["components"]
                      if c["area"] >= calibrate.NOISE_FLOOR_PX])
    _, debug = calibrate.estimate_single_area(areas)
    if debug is None:
        return None

    fig, ax = _data_fig(2.35)
    base = ps.SERIES["B1_components"]["color"]
    ax.hist(np.log(areas), bins=40, density=True, facecolor=ps.light(base, 0.7),
            edgecolor=base, linewidth=0.5, label="面积直方图")
    ax.plot(debug["grid"], debug["density"], color=base, lw=1.6, label="核密度估计")
    ps.headroom(ax, 0.28)
    top = ax.get_ylim()[1]
    for k in (1, 2, 3):
        x = np.log(k * calib["a0"])
        colour = ps.SERIES["ours"]["color"] if k == 1 else "#7F7F7F"
        # 竖线只画到标注带下方，不穿过顶部的文字
        ax.vlines(x, 0, top * 0.80, colors=colour, linestyles="-" if k == 1 else ":",
                  lw=1.4 if k == 1 else 1.1)
        text = f"$A_0$ = {calib['a0']:.0f} px" if k == 1 else f"{k}$A_0$"
        ax.annotate(text, (x, top * 0.82), ha="center", va="bottom", color=colour,
                    fontsize=8)
    ax.set_xlabel("连通域面积的对数 ln S")
    ax.set_ylabel("概率密度")
    ax.grid(axis="x", visible=False)
    fig.legend(loc="outside upper center", ncol=2)
    return _save_data_fig(fig, out_name)


def _touching_table():
    per_image = pd.read_csv(METRICS_ROOT / "per_image.csv")
    d2 = per_image[(per_image.dataset == "d2") & per_image.touch_prob.notna()].copy()
    d2["abs_err"] = d2["error"].abs()
    return d2.groupby(["touch_prob", "method"])["abs_err"].mean().unstack()


# 误差曲线与散点图里本文方法用深蓝。B1 原本占着深蓝，在这两张图里改用绿色
CURVE_COLOURS = {"Ours_ASW_SPC": "#0072B2", "B1_components": "#009E73"}


def error_curve_figure(out_name="error_vs_touching.pdf"):
    """各方法的误差随粘连率的变化。只给本文方法标数值，六条线全标会挤成一团。"""
    from src import plotstyle as ps
    table = _touching_table()
    order = ["B1_components", "B4_erosion_watershed", "B5_concave_ellipse",
             "B3_dist_watershed", "B2_area", "Ours_ASW_SPC"]
    fig, ax = _data_fig(2.9)
    x = table.index * 100
    for method in order:
        if method not in table:
            continue
        extra = dict(lw=2.0, zorder=5, markersize=5.5) if method == "Ours_ASW_SPC" else {}
        if method == "B4_erosion_watershed":
            # B4 在这组数据上与 B1 逐档相同，换细线与实心标记，两条线叠着也都看得见
            extra = dict(lw=1.0, markerfacecolor=ps.SERIES[method]["color"], markersize=3.5)
        if method in CURVE_COLOURS:
            extra["color"] = CURVE_COLOURS[method]
        ax.plot(x, table[method], **ps.line_style(method, **extra))
    ours = table["Ours_ASW_SPC"]
    ps.label_points(ax, x, ours, fmt="{:.2f}", dy=-5, fontsize=7,
                    color=CURVE_COLOURS["Ours_ASW_SPC"])
    ax.set_ylim(-4.5, None)
    ps.headroom(ax, 0.04)
    ax.set_xticks(x)
    ax.set_xlabel("粘连率／%")
    ax.set_ylabel("平均绝对误差／粒 ↓")
    # 图例按 B1 到 B5、本文方法的顺序逐行读；fig.legend 按列填，所以先把次序换好
    handles, labels = ax.get_legend_handles_labels()
    wanted = [ps.SERIES[m]["label"] for m in
              ("B1_components", "B4_erosion_watershed", "B2_area", "B5_concave_ellipse",
               "B3_dist_watershed", "Ours_ASW_SPC")]
    pairs = sorted(zip(handles, labels), key=lambda hl: wanted.index(hl[1]))
    fig.legend(*zip(*pairs), loc="outside upper center", ncol=3)
    return _save_data_fig(fig, out_name)


def scatter_figure(out_name="pred_vs_true.pdf", datasets=("d1", "d2", "d3")):
    """本文方法逐张的预测值与真值，画成热力图。格子颜色是落在该格的图片张数，虚线为两者相等。

    D3 有 719 张图，计数都是小整数，散点会大片重合，看不出哪里密；热力图把张数显出来。
    三幅子图共用一条对数色标，D1、D2 张数少，格子多为 1 到 3 张，仍能与 D3 放在一起比。
    """
    from src import plotstyle as ps
    per_image = pd.read_csv(METRICS_ROOT / "per_image.csv")
    ours = per_image[per_image.method == "Ours_ASW_SPC"]
    # 每个数据集的格宽。D3 计数是小整数，一格一粒；D1、D2 量程大，几粒并一格
    bin_width = {"d1": 5, "d2": 4, "d3": 1}
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", plt.get_cmap("Blues")(np.linspace(0.22, 1.0, 256)))
    cmap.set_bad("white")

    fig, axes = _data_fig(2.5, ncols=len(datasets))
    grids = []
    for ax, name in zip(np.atleast_1d(axes), datasets):
        block = ours[ours.dataset == name]
        truth, pred = block["gt"].to_numpy(), block["pred"].to_numpy()
        w = bin_width.get(name, 1)
        lo = np.floor(min(truth.min(), pred.min()) / w) * w - w / 2
        hi = np.ceil(max(truth.max(), pred.max()) / w) * w + w / 2
        edges = np.arange(lo, hi + w, w)
        counts, _, _ = np.histogram2d(truth, pred, bins=[edges, edges])
        grids.append((ax, edges, np.ma.masked_equal(counts.T, 0)))
        ax.set_title(f"{dataset_label(name)}\nMAE {block['error'].abs().mean():.2f}",
                     linespacing=1.3)
    top = max(g.max() for _, _, g in grids)
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=top)
    for ax, edges, grid in grids:
        mesh = ax.pcolormesh(edges, edges, grid, cmap=cmap, norm=norm,
                             edgecolors="white", linewidth=0.3)
        ax.plot(edges[[0, -1]], edges[[0, -1]], color="#555555", ls="--", lw=0.8,
                label="_y=x")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(edges[0], edges[-1])
        ax.set_aspect("equal")
        ax.grid(False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, prune="lower"))
        ax.set_xlabel("真值／粒")
    np.atleast_1d(axes)[0].set_ylabel("预测值／粒")
    bar = fig.colorbar(mesh, ax=list(np.atleast_1d(axes)), shrink=0.62, aspect=16, pad=0.02)
    bar.set_label("图片张数")
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if t <= top]
    bar.set_ticks(ticks, labels=[str(t) for t in ticks])
    bar.minorticks_off()
    fig.get_layout_engine().set(wspace=0.06)
    return _save_data_fig(fig, out_name)


def _abstract_curve(ax):
    """误差随粘连率的变化。只留两条线，首页要的是对比清楚。"""
    from src import plotstyle as ps
    curve = _touching_table()
    x = curve.index * 100
    for method, key in (("B1_components", "b1"), ("Ours_ASW_SPC", "ours")):
        if method in curve:
            extra = dict(lw=1.8, markersize=5) if key == "ours" else {}
            ax.plot(x, curve[method], **ps.line_style(key, **extra))
            if key == "b1":
                # 粘连率为 0 时两种做法数值相同，只在方法一那条线上标一次
                for xi, yi in list(zip(x, curve[method]))[1:]:
                    ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(-4, 3),
                                textcoords="offset points", ha="right", va="bottom",
                                fontsize=6.5, color=ps.SERIES[key]["color"])
            else:
                ps.label_points(ax, x, curve[method], fmt="{:.1f}", dy=-5, fontsize=6.5,
                                color=ps.SERIES[key]["color"])
    ax.set_ylim(-6, None)
    ps.headroom(ax, 0.12)
    ax.set_xticks(x)
    ax.set_xlabel("粘连率／%")
    ax.set_ylabel("平均绝对误差／粒 ↓")
    ax.set_title("误差随粘连程度的变化")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=GRID,
              framealpha=1)


def _abstract_bars(ax):
    """四种做法在同一批测试图上的误差。

    同一批是关键：学生模型只在留出集上测过，把它的数字与另外几种在全集上的
    数字并排，比较就不成立了，所以这里四者用的都是那 162 张测试图。
    """
    from src import plotstyle as ps
    table = pd.read_csv(METRICS_ROOT / "student_test.csv")
    keys = [k for k in ("b1", "ours", "student", "sam3") if k in table.columns]
    datasets = [d for d in ("d1", "d2", "d3") if (table.dataset == d).any()]
    width = 0.8 / len(keys)
    for index, key in enumerate(keys):
        values = [float((table[table.dataset == d][key]
                         - table[table.dataset == d].truth).abs().mean()) for d in datasets]
        xs = [i - 0.4 + width * (index + 0.5) for i in range(len(datasets))]
        ax.bar(xs, values, width * 0.9, **ps.bar_style(key))
        ps.label_points(ax, xs, values, fmt="{:.1f}", bold=True, fontsize=6)
    ps.headroom(ax, 0.10)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DATASET_CN[d] for d in datasets])
    ax.set_ylabel("平均绝对误差／粒 ↓")
    ax.set_title("四种做法在同一批测试图上的误差")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=1, frameon=True, facecolor="white",
              edgecolor=GRID, framealpha=1, fontsize=7, handlelength=1.6)


def student_bars_figure(out_name="student_bars.pdf"):
    """把图文摘要里的柱状图单独出一张，供仓库的 README 使用。"""
    fig, ax = _data_fig(2.6)
    _abstract_bars(ax)
    return _save_data_fig(fig, out_name)


def abstract_figure(out_name="graphical_abstract.pdf"):
    """首页图文摘要：上排两个场景的逐粒结果，下排两幅定量图。

    题目问的是"数出多少粒"和"解决连体计数不准"，这几幅各答一问：
    上排说明程序把哪些区域算作了几粒，左下说明粘连加重时误差并未跟着涨，
    右下把四种做法放在同一批图上比较。
    """
    from src import counter, plotstyle as ps, render, synth

    real = io_utils.load_d1()[0]
    dense = max(synth.load_d2(), key=lambda s: s.get("touch_prob") or 0)

    old = ps.apply()
    fig = plt.figure(figsize=(ps.TEXT_WIDTH_IN, 5.5))
    fig._rc_before = old
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    for column, (sample, label) in enumerate((
            (real, "真实照片"),
            (dense, f"合成图，粘连率 {int((dense.get('touch_prob') or 0) * 100)}%"))):
        image = io_utils.imread(sample["path"])
        labels, info, total, _ = counter.label_image(image)
        drawn = render.draw(image, labels, info, banner_lines=None)
        ax = fig.add_subplot(grid[0, column])
        _show(ax, cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB),
              f"{label}，数出 {total} 粒，真值 {sample['gt_count']} 粒",
              fontsize=plt.rcParams["axes.titlesize"])
    _abstract_curve(fig.add_subplot(grid[1, 0]))
    _abstract_bars(fig.add_subplot(grid[1, 1]))
    return _save_data_fig(fig, out_name)


def seed_depth_figure(out_name="seed_depth.pdf"):
    """3.5 用图：两粒之间的谷比低峰矮出 h 以上，所以切开；看的是相对深度，不是绝对距离。

    剖面沿块内的山脊走，而不是两种子间的直线。直线会穿出块外，剖面在背景上掉到 0，
    那是块的轮廓，不是注水时水面真正要漫过的鞍点。
    """
    from skimage.graph import route_through_array
    from src import plotstyle as ps, synth
    sample = max(synth.load_d2(), key=lambda s: s.get("touch_prob") or 0)
    image = io_utils.imread(sample["path"])
    pre = preprocess.preprocess(image)
    calib = pre["calib"]

    # 挑一个恰好出两个种子的两粒块，取谷最深的那个，差别最直观
    h = segment.BETA * calib["minor0"] / 2.0
    best = None
    for component in calib["components"]:
        if not 1.6 * calib["a0"] <= component["area"] <= 2.4 * calib["a0"]:
            continue
        piece = (calib["labels"] == component["label"]).astype(np.uint8)
        sub, _ = _crop_around(piece, component["bbox"], pad_ratio=0.25)
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        markers = segment.adaptive_markers(dist, calib["minor0"])
        if int(markers.max()) != 2:
            continue
        seeds = []
        for i in (1, 2):
            rr, cc = np.nonzero(markers == i)
            k = int(np.argmax(dist[rr, cc]))
            seeds.append((int(rr[k]), int(cc[k])))
        # 代价随距离指数下降，最短路就贴着山脊走，路上的最低点即鞍点
        cost = np.where(sub > 0, np.exp(-dist), 1e6)
        path, _ = route_through_array(cost, seeds[0], seeds[1], fully_connected=True,
                                      geometric=True)
        path = np.array(path)
        profile = dist[path[:, 0], path[:, 1]]
        depth = min(profile[0], profile[-1]) - profile.min()
        if best is None or depth > best[0]:
            best = (depth, dist, path, profile)
    if best is None:
        return None
    depth, dist, path, profile = best

    blue, red, grey = "#0072B2", "#D55E00", "#7F7F7F"
    steps = np.hypot(*np.diff(path, axis=0).T)
    xs = np.concatenate([[0.0], np.cumsum(steps)])
    low_peak = min(profile[0], profile[-1])
    saddle = int(np.argmin(profile))

    fig, axes = _data_fig(2.45, ncols=2, gridspec_kw={"width_ratios": [1, 1.7]})
    axes[0].imshow(dist, cmap="magma")
    axes[0].plot(path[:, 1], path[:, 0], color="#56B4E9", lw=1.4, label="_ridge")
    axes[0].scatter(path[[0, -1], 1], path[[0, -1], 0], s=22, color="white",
                    edgecolors=blue, linewidths=1.0, zorder=3)
    axes[0].scatter([path[saddle, 1]], [path[saddle, 0]], s=18, marker="v", color=red,
                    zorder=3)
    axes[0].set_title("距离变换，蓝线为沿山脊的剖面")
    axes[0].axis("off")

    ax = axes[1]
    ax.fill_between(xs, 0, profile, color=blue, alpha=0.12, lw=0)
    ax.plot(xs, profile, color=blue, lw=1.8, label="沿山脊的剖面")
    ax.axhline(low_peak, color=grey, ls=":", lw=1.0, label="较低的峰")
    ax.axhline(low_peak - h, color=red, ls="--", lw=1.1,
               label=f"较低的峰减 h，h = {h:.1f}")
    ax.plot(xs[[0, -1]], profile[[0, -1]], ls="none", marker="o", ms=4.5,
            markerfacecolor="white", markeredgecolor=blue, markeredgewidth=1.2, label="_peaks")
    ax.plot(xs[saddle], profile[saddle], ls="none", marker="v", ms=5, color=red,
            label="_saddle")
    # 双箭头量出谷深，放在鞍点右侧，不压剖面线
    arrow_x = xs[saddle] + 0.06 * xs[-1]
    ax.annotate("", (arrow_x, profile[saddle]), (arrow_x, low_peak),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    ax.annotate(f"谷深 {depth:.1f}", (arrow_x, (low_peak + profile[saddle]) / 2),
                xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                fontsize=7.5, color=INK)
    for index in (0, len(profile) - 1):
        ax.annotate(f"{profile[index]:.1f}", (xs[index], profile[index]), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=blue)
    ax.annotate(f"{profile[saddle]:.1f}", (xs[saddle], profile[saddle]), xytext=(0, -6),
                textcoords="offset points", ha="center", va="top", fontsize=7, color=red)
    ax.set_xlim(-0.04 * xs[-1], 1.04 * xs[-1])
    ax.set_ylim(0, None)
    ps.headroom(ax, 0.42)
    ax.set_xlabel("沿剖面自一个种子到另一个种子／像素")
    ax.set_ylabel("到背景的距离／像素")
    ax.set_title("谷比低峰矮出 h 以上，判为两粒")
    ax.legend(loc="upper center", ncol=3, frameon=True, facecolor="white", edgecolor=GRID,
              framealpha=1, handlelength=1.8, columnspacing=1.0)
    return _save_data_fig(fig, out_name)

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
        calibration_figure(io_utils.imread(mixed["path"])),
        error_curve_figure(),
        scatter_figure(),
        student_bars_figure(),
        teacher_labels_figure(),
        student_pipeline_figure(),
        student_vs_geometric_figure(),
        terrain_figure(),
    ]
    for path in outputs:
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
