"""数据图的统一画法：按版心实际宽度作图、系列的颜色与标记绑定、保存前查重叠。

报告里的插图一律设为版心宽，所以作图时画布就定成版心宽，字号用印刷后的真实点数，
进版不缩放，图内的字与图题、正文的比例始终一致。原先是 9 英寸画布再缩到 6 英寸，
字号要靠一个 1.5 倍的系数去凑，换一种排版就会走样。

只管坐标轴类的数据图。照片、掩膜这类图像面板仍由 visualize._show 画。
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

# 版心宽 15.4 cm，即 LaTeX 日志里的 \textwidth = 438.17 pt
TEXT_WIDTH_PT = 438.17
TEXT_WIDTH_IN = TEXT_WIDTH_PT / 72

INK = "#333333"
GRID = "#D9D9D9"

# 印刷后的真实字号。正文五号 10.5 pt、图题小五 9 pt，图内的字略小于图题
RC = {
    "pdf.fonttype": 42, "ps.fonttype": 42,     # TrueType，不出 Type 3
    "savefig.bbox": None,                       # 宽度由 figsize 定死，不按内容裁
    "figure.constrained_layout.use": True,
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "axes.titlecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "hatch.linewidth": 0.5,
}

# 同一个方法在所有图里是同一种颜色、同一种标记、同一种线型与填充图案。
# 颜色取 Okabe-Ito 色盲友好色板；颜色之外再用标记与图案区分，灰度打印也分得开。
SERIES = {
    "B1_components": dict(label="B1 连通域计数", color="#0072B2", marker="s",
                          ls="--", hatch="///"),
    "B2_area": dict(label="B2 面积估算", color="#56B4E9", marker="v", ls=":", hatch=".."),
    "B3_dist_watershed": dict(label="B3 距离变换注水分割", color="#CC79A7", marker="^",
                              ls="-.", hatch="\\\\\\"),
    "B4_erosion_watershed": dict(label="B4 腐蚀标记注水分割", color="#000000", marker="X",
                                 ls=(0, (5, 2, 1, 2)), hatch="xx"),
    "B5_concave_ellipse": dict(label="B5 凹点+椭圆拟合", color="#7F7F7F", marker="D",
                               ls=(0, (1, 1.5)), hatch="--"),
    "ours": dict(label="方法一", color="#E69F00", marker="o", ls="-", hatch=""),
    "student": dict(label="方法二 学生模型", color="#009E73", marker="P", ls="-",
                    hatch="\\\\\\"),
    "sam3": dict(label="SAM 3", color="#D55E00", marker="*", ls="-", hatch=".."),
}
SERIES["Ours_ASW_SPC"] = dict(SERIES["ours"], label="本文方法")
SERIES["b1"] = dict(SERIES["B1_components"], label="直接数连通域")


def _check_series_table():
    """标记与颜色在基础系列之间两两不同，否则同一张图里可能分不出两条线。"""
    base = [v for k, v in SERIES.items() if k not in ("Ours_ASW_SPC", "b1")]
    colors = [v["color"] for v in base]
    markers = [v["marker"] for v in base]
    assert len(set(colors)) == len(colors), "系列颜色有重复"
    assert len(set(markers)) == len(markers), "系列标记有重复"


_check_series_table()


def apply():
    """启用本模块的样式。返回旧值，便于在同一进程里画别的图时恢复。"""
    old = {k: matplotlib.rcParams[k] for k in RC}
    matplotlib.rcParams.update(RC)
    return old


def light(color, amount=0.55):
    """把颜色往白色调淡，用作柱子的浅色填充。"""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def line_style(key, **extra):
    s = SERIES[key]
    style = dict(color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"],
                 markerfacecolor="white", markeredgewidth=1.0)
    style.update(extra)
    return style


def bar_style(key, **extra):
    s = SERIES[key]
    style = dict(facecolor=light(s["color"]), edgecolor=s["color"], hatch=s["hatch"],
                 linewidth=0.8, label=s["label"])
    style.update(extra)
    return style


def label_points(ax, xs, ys, fmt="{:.2f}", bold=False, dy=3.5, **kw):
    """在数据点或柱顶上方标数值。偏移用点数，与量程和对数轴无关。"""
    size = kw.pop("fontsize", 7)
    colour = kw.pop("color", INK)
    va = "bottom" if dy >= 0 else "top"
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), (x, y), xytext=(0, dy), textcoords="offset points",
                    ha="center", va=va, fontsize=size,
                    fontweight="bold" if bold else "normal", color=colour, **kw)


def headroom(ax, frac=0.18):
    """给顶部留白，免得数值标签顶到轴框或图例。对数轴按乘法留。"""
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        ax.set_ylim(lo, hi * (1 + 10 * frac))
    else:
        ax.set_ylim(lo, hi + frac * (hi - lo))


# ------------------------------------------------------------------ 重叠检查

def _texts(fig):
    """真正会画出来的文字。

    刻度标签要单独处理：findobj 会把关掉坐标轴的图像面板、以及视野外备用的刻度标签
    也一并找出来，它们并不绘制，拿来查重叠只会误报。所以刻度标签只取各轴
    当前要画的那几个，其余文字照常取。
    """
    tick_ids = set()
    drawn_ticks = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_ids.update((id(tick.label1), id(tick.label2)))
            if not (ax.get_visible() and ax.axison and axis.get_visible()):
                continue
            for tick in axis._update_ticks():
                for label in (tick.label1, tick.label2):
                    if label.get_visible() and label.get_text().strip():
                        drawn_ticks.append(label)
    out = []
    for artist in fig.findobj(Text):
        if id(artist) in tick_ids:
            continue
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        if artist.axes is not None and not artist.axes.get_visible():
            continue
        out.append(artist)
    return out + drawn_ticks


def _legend_members(fig):
    members = set()
    for leg in fig.findobj(matplotlib.legend.Legend):
        members.update(id(a) for a in leg.findobj())
    return members


def _gridlines(fig):
    ids = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            ids.update(id(l) for l in axis.get_gridlines())
            ids.update(id(t.tick1line) for t in axis.get_major_ticks())
            ids.update(id(t.tick2line) for t in axis.get_major_ticks())
    return ids


def _dense_points(line, renderer, step=1.5):
    """沿线段按像素间距插值取点。只取顶点会漏掉'直线穿过扁长文字框'这种情况。"""
    xy = line.get_transform().transform(np.column_stack(line.get_data()))
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if len(xy) < 2:
        return xy
    parts = [xy[:1]]
    for a, b in zip(xy[:-1], xy[1:]):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        t = np.linspace(0, 1, n + 1)[1:, None]
        parts.append(a + (b - a) * t)
    return np.vstack(parts)


def _inside(points, box, pad=0.5):
    if len(points) == 0:
        return False
    x0, y0, x1, y1 = box.x0 + pad, box.y0 + pad, box.x1 - pad, box.y1 - pad
    return bool(np.any((points[:, 0] > x0) & (points[:, 0] < x1)
                       & (points[:, 1] > y0) & (points[:, 1] < y1)))


def check_overlap(fig):
    """返回 [(类型, 说明), ...]，空表示没有文字压线、压柱、相挤，图例也没压住数据。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    in_legend = _legend_members(fig)
    skip_lines = _gridlines(fig)
    hits = []

    texts = [t for t in _texts(fig) if id(t) not in in_legend]
    boxes = [(t, t.get_window_extent(renderer)) for t in texts]

    for i, (a, box_a) in enumerate(boxes):
        for b, box_b in boxes[i + 1:]:
            if box_a.overlaps(box_b) and box_a.width > 0 and box_b.width > 0:
                inter = (min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)) * \
                        (min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0))
                if inter > 1.0:
                    hits.append(("文字相挤", f"「{a.get_text()}」与「{b.get_text()}」"))

    data_lines, bars, scatters = [], [], []
    for ax in fig.axes:
        data_lines += [l for l in ax.get_lines()
                       if l.get_visible() and id(l) not in skip_lines and id(l) not in in_legend]
        bars += [p for p in ax.patches if isinstance(p, Rectangle) and p.get_visible()
                 and p.get_height() != 0 and id(p) not in in_legend]
        scatters += [c for c in ax.collections if isinstance(c, PathCollection)
                     and id(c) not in in_legend]

    line_pts = [(l, _dense_points(l, renderer)) for l in data_lines]
    bar_boxes = [(p, p.get_window_extent(renderer)) for p in bars]
    scatter_pts = [(c, c.get_offset_transform().transform(c.get_offsets())) for c in scatters]

    for t, box in boxes:
        for line, pts in line_pts:
            if _inside(pts, box):
                hits.append(("文字压线", f"「{t.get_text()}」压在线 {line.get_label()} 上"))
        for patch, pbox in bar_boxes:
            if box.overlaps(pbox) and min(box.x1, pbox.x1) - max(box.x0, pbox.x0) > 1 \
                    and min(box.y1, pbox.y1) - max(box.y0, pbox.y0) > 1:
                hits.append(("文字压柱", f"「{t.get_text()}」压在柱子上"))
        for coll, pts in scatter_pts:
            if _inside(pts, box, pad=1.0):
                hits.append(("文字压点", f"「{t.get_text()}」压在散点上"))

    for leg in fig.findobj(matplotlib.legend.Legend):
        lbox = leg.get_window_extent(renderer)
        for line, pts in line_pts:
            if _inside(pts, lbox):
                hits.append(("图例压数据", f"图例压住线 {line.get_label()}"))
        for patch, pbox in bar_boxes:
            if lbox.overlaps(pbox):
                hits.append(("图例压数据", "图例压住柱子"))
        for coll, pts in scatter_pts:
            if _inside(pts, lbox):
                hits.append(("图例压数据", "图例压住散点"))
        for t, box in boxes:
            if lbox.overlaps(box):
                hits.append(("图例压字", f"图例压住「{t.get_text()}」"))
    return hits


def save_checked(fig, path, width_in=TEXT_WIDTH_IN, png_copy=True, dpi=200):
    """先量宽度、查重叠，都通过才写文件。PDF 给报告用，另存一份 PNG 给 README。"""
    got = fig.get_size_inches()[0] * 72
    want = width_in * 72
    if got > want + 1.5:
        raise AssertionError(f"实际宽 {got:.1f} pt 超过版心 {want:.1f} pt，未保存")
    hits = check_overlap(fig)
    if hits:
        lines = "\n".join(f"  {kind}：{what}" for kind, what in hits)
        raise AssertionError(f"{path} 有重叠，未保存：\n{lines}")
    fig.savefig(path)
    if png_copy:
        fig.savefig(str(path).rsplit(".", 1)[0] + ".png", dpi=dpi)
    plt.close(fig)
    return path
