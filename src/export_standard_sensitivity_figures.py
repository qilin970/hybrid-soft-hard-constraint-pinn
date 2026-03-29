from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize_col(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def find_column(df: pd.DataFrame, aliases: List[str], required: bool = True) -> str | None:
    norm_map = {normalize_col(c): c for c in df.columns}
    for a in aliases:
        key = normalize_col(a)
        if key in norm_map:
            return norm_map[key]
    if required:
        raise KeyError(f"Cannot find required column from aliases: {aliases}")
    return None


def resolve_data_file(run_dir: Path) -> Path:
    preferred = run_dir / "summary" / "confirmed_sensitivity_grid.xlsx"
    if preferred.exists():
        return preferred

    parent = run_dir.parent
    all_confirmed = list(parent.rglob("confirmed_sensitivity_grid.xlsx"))
    if all_confirmed:
        all_confirmed.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return all_confirmed[0]

    all_partial = list(parent.rglob("confirmed_sensitivity_grid_partial.xlsx"))
    if all_partial:
        all_partial.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return all_partial[0]

    raise FileNotFoundError("No confirmed_sensitivity_grid.xlsx or partial grid found.")


def choose_best_point(df: pd.DataFrame, col_min_pcd: str, col_reg: str, col_mono: str, col_lm: str, col_ls: str) -> pd.Series:
    # Rule:
    # 1) max min_rule_pcd_test
    # 2) min regression loss
    # 3) min monotonic loss
    # 4) smaller lambda_m
    # 5) smaller lambda_s
    srt = df.sort_values(
        by=[col_min_pcd, col_reg, col_mono, col_lm, col_ls],
        ascending=[False, True, True, True, True],
    )
    return srt.iloc[0]


def apply_paper_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif", "serif"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.titlesize": 11.3,
            "axes.titleweight": "regular",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.2,
            "axes.linewidth": 1.02,
            "grid.alpha": 0.15,
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
        }
    )


def get_curve_styles(ls_values: List[float]) -> Dict[float, Tuple[str, str]]:
    # Fixed reference-style palette and markers for key lambda_s values.
    fixed_styles = [
        (0.0, "#7F7F7F", "s"),
        (0.001, "#FF5A4F", "o"),
        (0.01, "#2F95FF", "^"),
        (0.1, "#22C06F", "D"),
    ]
    fallback_palette = ["#4C566A", "#B55D60", "#5A86B6", "#5C9A7A", "#8C7AA9", "#8A8D52"]
    fallback_markers = ["s", "o", "^", "D", "v", "P"]
    styles: Dict[float, Tuple[str, str]] = {}
    tol = 1e-10
    for i, ls in enumerate(ls_values):
        lsf = float(ls)
        selected: Tuple[str, str] | None = None
        for k, color, marker in fixed_styles:
            if abs(lsf - k) <= tol:
                selected = (color, marker)
                break
        if selected is None:
            selected = (fallback_palette[i % len(fallback_palette)], fallback_markers[i % len(fallback_markers)])
        styles[lsf] = selected
    return styles


def style_axis(ax):
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(1.02)
        sp.set_color("#1a1a1a")
    ax.tick_params(axis="both", which="both", direction="in", length=3.8, width=0.82, colors="#222222")


def style_legend(ax):
    leg = ax.legend(loc="best", frameon=True, framealpha=0.90, borderpad=0.28, labelspacing=0.20, handletextpad=0.45)
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("0.75")
        frame.set_linewidth(0.70)


def set_lambda_m_xlim(ax, x_vals: np.ndarray):
    x = np.asarray(x_vals, dtype=float)
    if x.size == 0:
        return
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    left = -25.0 if x_min >= 0 else x_min - 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    right = x_max + 0.04 * (x_max - left if x_max > left else 1.0)
    ax.set_xlim(left, right)


def add_low_lambda_inset(
    *,
    ax,
    df: pd.DataFrame,
    col_lm: str,
    col_ls: str,
    col_y: str,
    best_row: pd.Series,
    inset_rect: Tuple[float, float, float, float],
    inset_xlim: Tuple[float, float],
    style_map: Dict[float, Tuple[str, str]],
    use_symlog: bool,
    draw_zero_line: bool,
    star_size: float,
    star_facecolor: str,
    star_edgecolor: str,
    star_edgewidth: float,
    star_alpha: float,
):
    axins = ax.inset_axes(list(inset_rect))
    axins.set_facecolor("white")
    ls_values = sorted(df[col_ls].astype(float).unique())
    x_all = df[col_lm].to_numpy(float)
    y_all = df[col_y].to_numpy(float)

    for ls in ls_values:
        sub = df[np.isclose(df[col_ls].astype(float), ls)].sort_values(col_lm)
        color, marker = style_map[float(ls)]
        axins.plot(
            sub[col_lm].to_numpy(float),
            sub[col_y].to_numpy(float),
            marker=marker,
            markersize=4.0,
            linewidth=1.25,
            color=color,
            markerfacecolor=color,
            markeredgecolor="#4a4a4a",
            markeredgewidth=0.6,
        )

    x0, x1 = inset_xlim
    axins.set_xlim(float(x0), float(x1))
    region = (x_all >= x0) & (x_all <= x1) & np.isfinite(y_all)
    y_reg = y_all[region]
    if y_reg.size == 0:
        y_reg = y_all[np.isfinite(y_all)]

    if use_symlog:
        pos = y_reg[y_reg > 0]
        linthresh = float(np.percentile(pos, 20)) if pos.size else 1e-9
        linthresh = max(linthresh, 1e-10)
        axins.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
        if pos.size:
            ymax = float(np.nanmax(pos)) * 1.25
            ymin_pos = float(np.nanmin(pos)) * 0.85
            ymin = -max(linthresh, ymin_pos * 0.5)
            axins.set_ylim(ymin, ymax)
    else:
        if y_reg.size:
            y_min = float(np.nanmin(y_reg))
            y_max = float(np.nanmax(y_reg))
            span = max(y_max - y_min, 1e-12)
            axins.set_ylim(y_min - 0.12 * span, y_max + 0.15 * span)

    if draw_zero_line:
        axins.axhline(0.0, color="#777777", lw=0.65, alpha=0.8)

    x_best = float(best_row[col_lm])
    y_best = float(best_row[col_y])
    if x0 <= x_best <= x1:
        axins.scatter(
            [x_best],
            [y_best],
            marker="*",
            s=star_size,
            color=star_facecolor,
            edgecolors=star_edgecolor,
            linewidths=star_edgewidth,
            alpha=star_alpha,
            zorder=5,
        )

    for sp in axins.spines.values():
        sp.set_linewidth(0.8)
        sp.set_edgecolor("#444444")
    axins.tick_params(axis="both", which="both", direction="in", length=2.5, width=0.65, labelsize=7.2)
    axins.grid(True, axis="y", color="0.93", linewidth=0.35, alpha=0.6)
    return axins


def annotate_best_point(
    ax,
    best_row: pd.Series,
    *,
    col_lm: str,
    col_ls: str,
    col_y: str,
    dx: float,
    dy: float,
    connector_color: str = "0.2",
    connector_lw: float = 0.9,
    connector_shrinkA: float = 2,
    connector_shrinkB: float = 4,
    connector_style: str = "arc3,rad=0.08",
    connector_mutation_scale: float = 10,
    connector_alpha: float = 1.0,
    star_size: float = 156,
    star_facecolor: str = "#C96A5B",
    star_edgecolor: str = "#444444",
    star_edgewidth: float = 0.7,
    star_alpha: float = 0.82,
    box_edgecolor: str = "0.65",
    box_alpha: float = 0.90,
    box_linewidth: float = 0.8,
    textcoords: str = "offset points",
    enforce_manual_position: bool = False,
):
    x = float(best_row[col_lm])
    y = float(best_row[col_y])
    ls = float(best_row[col_ls])

    # Keep best marker visible but not overly dominant.
    ax.scatter(
        [x],
        [y],
        marker="*",
        s=star_size,
        color=star_facecolor,
        edgecolors=star_edgecolor,
        linewidths=star_edgewidth,
        alpha=star_alpha,
        zorder=7,
    )

    label = f"Best\n({ls:g}, {x:g})"
    ann = ax.annotate(
        label,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords=textcoords,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            color=connector_color,
            lw=connector_lw,
            shrinkA=connector_shrinkA,
            shrinkB=connector_shrinkB,
            connectionstyle=connector_style,
            mutation_scale=connector_mutation_scale,
            alpha=connector_alpha,
        ),
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor=box_edgecolor,
            alpha=box_alpha,
            linewidth=box_linewidth,
        ),
        fontsize=10.6,
        fontweight="semibold",
        zorder=8,
        annotation_clip=True,
    )

    if not enforce_manual_position:
        # Keep annotation inside axes bounds when offsets are close to borders.
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = ann.get_window_extent(renderer=renderer)
        ax_bbox = ax.get_window_extent(renderer=renderer)
        shift_x = 0.0
        shift_y = 0.0
        pad = 4.0
        if bbox.x1 > ax_bbox.x1 - pad:
            shift_x = (ax_bbox.x1 - pad) - bbox.x1
        if bbox.x0 < ax_bbox.x0 + pad:
            shift_x = max(shift_x, (ax_bbox.x0 + pad) - bbox.x0)
        if bbox.y1 > ax_bbox.y1 - pad:
            shift_y = (ax_bbox.y1 - pad) - bbox.y1
        if bbox.y0 < ax_bbox.y0 + pad:
            shift_y = max(shift_y, (ax_bbox.y0 + pad) - bbox.y0)
        if shift_x != 0.0 or shift_y != 0.0:
            ox, oy = ann.get_position()
            ann.set_position((ox + shift_x, oy + shift_y))


def plot_multiline(
    *,
    df: pd.DataFrame,
    col_lm: str,
    col_ls: str,
    col_y: str,
    y_label: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
    best_row: pd.Series,
    annotate_dx: float,
    annotate_dy: float,
    annotate_connector_color: str = "0.2",
    annotate_connector_lw: float = 0.9,
    annotate_connector_shrinkA: float = 2,
    annotate_connector_shrinkB: float = 4,
    annotate_connector_style: str = "arc3,rad=0.08",
    annotate_connector_mutation_scale: float = 10,
    annotate_connector_alpha: float = 1.0,
    annotate_star_size: float = 156,
    annotate_star_facecolor: str = "#C96A5B",
    annotate_star_edgecolor: str = "#444444",
    annotate_star_edgewidth: float = 0.7,
    annotate_star_alpha: float = 0.82,
    annotate_box_edgecolor: str = "0.65",
    annotate_box_alpha: float = 0.90,
    annotate_box_linewidth: float = 0.8,
    annotate_textcoords: str = "offset points",
    annotate_enforce_manual_position: bool = False,
    add_inset: bool = False,
    inset_rect: Tuple[float, float, float, float] = (0.56, 0.50, 0.38, 0.34),
    inset_xlim: Tuple[float, float] = (-5.0, 120.0),
    inset_use_symlog: bool = False,
    inset_draw_zero_line: bool = False,
    use_symlog: bool = False,
    draw_zero_line: bool = False,
):
    ls_values = sorted(df[col_ls].astype(float).unique())
    style_map = get_curve_styles(ls_values)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for i, ls in enumerate(ls_values):
        sub = df[np.isclose(df[col_ls].astype(float), ls)].sort_values(col_lm)
        color, marker = style_map[float(ls)]
        ax.plot(
            sub[col_lm].to_numpy(float),
            sub[col_y].to_numpy(float),
            marker=marker,
            markersize=6.4,
            linewidth=1.95,
            color=color,
            markerfacecolor=color,
            markeredgecolor="#4a4a4a",
            markeredgewidth=0.75,
            label=fr"$\lambda_S$={ls:g}",
        )

    if use_symlog:
        y = df[col_y].to_numpy(float)
        pos = y[np.isfinite(y) & (y > 0)]
        linthresh = float(np.percentile(pos, 25)) if len(pos) else 1e-8
        linthresh = max(linthresh, 1e-10)
        ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)

    if draw_zero_line:
        ax.axhline(0.0, color="#444444", lw=1.0, linestyle="-", alpha=0.75, zorder=1)

    annotate_best_point(
        ax,
        best_row,
        col_lm=col_lm,
        col_ls=col_ls,
        col_y=col_y,
        dx=annotate_dx,
        dy=annotate_dy,
        connector_color=annotate_connector_color,
        connector_lw=annotate_connector_lw,
        connector_shrinkA=annotate_connector_shrinkA,
        connector_shrinkB=annotate_connector_shrinkB,
        connector_style=annotate_connector_style,
        connector_mutation_scale=annotate_connector_mutation_scale,
        connector_alpha=annotate_connector_alpha,
        star_size=annotate_star_size,
        star_facecolor=annotate_star_facecolor,
        star_edgecolor=annotate_star_edgecolor,
        star_edgewidth=annotate_star_edgewidth,
        star_alpha=annotate_star_alpha,
        box_edgecolor=annotate_box_edgecolor,
        box_alpha=annotate_box_alpha,
        box_linewidth=annotate_box_linewidth,
        textcoords=annotate_textcoords,
        enforce_manual_position=annotate_enforce_manual_position,
    )
    set_lambda_m_xlim(ax, df[col_lm].to_numpy(float))
    ax.set_xlabel(r"$\lambda_M$")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    style_axis(ax)
    ax.grid(True, axis="y", color="0.92", linewidth=0.45, alpha=0.60)
    style_legend(ax)
    if add_inset:
        add_low_lambda_inset(
            ax=ax,
            df=df,
            col_lm=col_lm,
            col_ls=col_ls,
            col_y=col_y,
            best_row=best_row,
            inset_rect=inset_rect,
            inset_xlim=inset_xlim,
            style_map=style_map,
            use_symlog=inset_use_symlog,
            draw_zero_line=inset_draw_zero_line,
            star_size=annotate_star_size,
            star_facecolor=annotate_star_facecolor,
            star_edgecolor=annotate_star_edgecolor,
            star_edgewidth=annotate_star_edgewidth,
            star_alpha=annotate_star_alpha,
        )
    fig.subplots_adjust(top=0.90)
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, facecolor="white", transparent=False)
    fig.savefig(out_pdf, dpi=600, facecolor="white", transparent=False)
    plt.close(fig)


def plot_triptych(
    *,
    df: pd.DataFrame,
    col_lm: str,
    col_ls: str,
    col_reg: str,
    col_mono: str,
    col_pcd: str,
    best_row: pd.Series,
    out_png: Path,
    out_pdf: Path,
):
    ls_values = sorted(df[col_ls].astype(float).unique())
    style_map = get_curve_styles(ls_values)

    fig = plt.figure(figsize=(10.8, 8.35))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.06], hspace=0.46, wspace=0.28)
    ax_mono = fig.add_subplot(gs[0, 0])
    ax_pcd = fig.add_subplot(gs[0, 1])
    ax_reg = fig.add_subplot(gs[1, :])

    # Monotonic
    for ls in ls_values:
        sub = df[np.isclose(df[col_ls].astype(float), ls)].sort_values(col_lm)
        color, marker = style_map[float(ls)]
        ax_mono.plot(
            sub[col_lm].to_numpy(float),
            sub[col_mono].to_numpy(float),
            marker=marker,
            linewidth=1.9,
            markersize=6.4,
            color=color,
            markerfacecolor=color,
            markeredgecolor="#4a4a4a",
            markeredgewidth=0.75,
            label=fr"$\lambda_S$={ls:g}",
        )
    y = df[col_mono].to_numpy(float)
    pos = y[np.isfinite(y) & (y > 0)]
    linthresh = float(np.percentile(pos, 25)) if len(pos) else 1e-8
    linthresh = max(linthresh, 1e-10)
    ax_mono.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    ax_mono.axhline(0.0, color="#444444", lw=1.0, linestyle="-", alpha=0.75)
    annotate_best_point(
        ax_mono,
        best_row,
        col_lm=col_lm,
        col_ls=col_ls,
        col_y=col_mono,
        dx=0.34,
        dy=0.26,
        connector_color="#4A4A4A",
        connector_lw=0.75,
        connector_shrinkA=4,
        connector_shrinkB=4,
        connector_style="arc3,rad=0.10",
        connector_mutation_scale=9,
        connector_alpha=0.85,
        star_size=156,
        star_facecolor="#E76F5A",
        star_edgecolor="#4A4A4A",
        star_edgewidth=0.75,
        star_alpha=0.95,
        box_edgecolor="0.68",
        box_alpha=0.92,
        box_linewidth=0.75,
        textcoords="axes fraction",
        enforce_manual_position=True,
    )
    set_lambda_m_xlim(ax_mono, df[col_lm].to_numpy(float))
    ax_mono.set_xlabel(r"$\lambda_M$")
    ax_mono.set_ylabel(r"$E_M$")
    style_axis(ax_mono)
    ax_mono.grid(True, axis="y", color="0.92", linewidth=0.45, alpha=0.60)
    add_low_lambda_inset(
        ax=ax_mono,
        df=df,
        col_lm=col_lm,
        col_ls=col_ls,
        col_y=col_mono,
        best_row=best_row,
        inset_rect=(0.45, 0.54, 0.33, 0.32),
        inset_xlim=(-5.0, 120.0),
        style_map=style_map,
        use_symlog=True,
        draw_zero_line=True,
        star_size=156,
        star_facecolor="#E76F5A",
        star_edgecolor="#4A4A4A",
        star_edgewidth=0.75,
        star_alpha=0.95,
    )
    ax_mono.text(
        0.5,
        -0.25,
        r"(a) Effect of hyperparameters on monotonicity loss $E_M$",
        transform=ax_mono.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
    )

    # PCD
    for ls in ls_values:
        sub = df[np.isclose(df[col_ls].astype(float), ls)].sort_values(col_lm)
        color, marker = style_map[float(ls)]
        ax_pcd.plot(
            sub[col_lm].to_numpy(float),
            sub[col_pcd].to_numpy(float),
            marker=marker,
            linewidth=1.9,
            markersize=6.4,
            color=color,
            markerfacecolor=color,
            markeredgecolor="#4a4a4a",
            markeredgewidth=0.75,
            label=fr"$\lambda_S$={ls:g}",
        )
    annotate_best_point(
        ax_pcd,
        best_row,
        col_lm=col_lm,
        col_ls=col_ls,
        col_y=col_pcd,
        dx=95,
        dy=-38,
    )
    set_lambda_m_xlim(ax_pcd, df[col_lm].to_numpy(float))
    ax_pcd.set_xlabel(r"$\lambda_M$")
    ax_pcd.set_ylabel("Minimum rule-level PCD")
    style_axis(ax_pcd)
    ax_pcd.grid(True, axis="y", color="0.92", linewidth=0.45, alpha=0.60)
    ax_pcd.text(
        0.5,
        -0.25,
        "(b) Effect of hyperparameters on minimum rule-level PCD",
        transform=ax_pcd.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
    )

    # Regression
    for ls in ls_values:
        sub = df[np.isclose(df[col_ls].astype(float), ls)].sort_values(col_lm)
        color, marker = style_map[float(ls)]
        ax_reg.plot(
            sub[col_lm].to_numpy(float),
            sub[col_reg].to_numpy(float),
            marker=marker,
            linewidth=2.0,
            markersize=6.4,
            color=color,
            markerfacecolor=color,
            markeredgecolor="#4a4a4a",
            markeredgewidth=0.75,
            label=fr"$\lambda_S$={ls:g}",
        )
    annotate_best_point(
        ax_reg,
        best_row,
        col_lm=col_lm,
        col_ls=col_ls,
        col_y=col_reg,
        dx=255.0,
        dy=0.000205,
        connector_color="#444444",
        connector_lw=0.8,
        connector_shrinkA=4,
        connector_shrinkB=4,
        connector_style="arc3,rad=0.12",
        connector_mutation_scale=9,
        connector_alpha=0.9,
        star_size=156,
        star_facecolor="#E76F5A",
        star_edgecolor="#4A4A4A",
        star_edgewidth=0.75,
        star_alpha=0.95,
        box_edgecolor="0.68",
        box_alpha=0.92,
        box_linewidth=0.75,
        textcoords="data",
        enforce_manual_position=True,
    )
    set_lambda_m_xlim(ax_reg, df[col_lm].to_numpy(float))
    ax_reg.set_xlabel(r"$\lambda_M$")
    ax_reg.set_ylabel(r"$E_R$")
    style_axis(ax_reg)
    ax_reg.grid(True, axis="y", color="0.92", linewidth=0.45, alpha=0.60)
    ax_reg.text(
        0.5,
        -0.18,
        r"(c) Effect of hyperparameters on regression loss $E_R$",
        transform=ax_reg.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
    )

    # Unified legend on top
    handles, labels = ax_reg.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, len(labels)),
        frameon=True,
        bbox_to_anchor=(0.5, 0.985),
        borderpad=0.35,
        handletextpad=0.45,
        columnspacing=0.85,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("0.75")
    leg.get_frame().set_linewidth(0.70)
    leg.get_frame().set_alpha(0.90)
    fig.subplots_adjust(top=0.89, bottom=0.08)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])
    fig.savefig(out_png, dpi=600, facecolor="white", transparent=False)
    fig.savefig(out_pdf, dpi=600, facecolor="white", transparent=False)
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_run_dir = repo_root / "results" / "confirmed_sensitivity"
    default_out_dir = repo_root / "results" / "confirmed_sensitivity" / "standard_figures"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_run_dir,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
    )
    args = parser.parse_args()

    apply_paper_style()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data_file = resolve_data_file(args.run_dir)
    df = pd.read_excel(data_file)

    col_map = {
        "lambda_s": find_column(df, ["lambda_s", "lambdas", "ls"]),
        "lambda_m": find_column(df, ["lambda_m", "lambdam", "lm"]),
        "regression_loss": find_column(
            df,
            [
                "E_R_train_final",
                "final training regression loss",
                "regression_loss_final",
                "final_regression_loss",
                "e_r_train",
            ],
        ),
        "monotonic_loss": find_column(
            df,
            [
                "E_M_train_final",
                "final training monotonic loss",
                "monotonic_loss_final",
                "final_monotonic_loss",
                "e_m_train",
            ],
        ),
        "min_rule_pcd_test": find_column(df, ["min_rule_pcd_test", "min_rule_pcd", "min_pcd", "minimum_rule_pcd"]),
        "r2_mean_test": find_column(df, ["R2_mean_test", "r2_mean_test", "r2_test"], required=False),
    }

    # Keep only unique point rows
    df2 = df.copy()
    df2 = df2.drop_duplicates(subset=[col_map["lambda_s"], col_map["lambda_m"]], keep="last").copy()

    # best point selection
    best = choose_best_point(
        df2,
        col_min_pcd=col_map["min_rule_pcd_test"],
        col_reg=col_map["regression_loss"],
        col_mono=col_map["monotonic_loss"],
        col_lm=col_map["lambda_m"],
        col_ls=col_map["lambda_s"],
    )

    # standard single figures
    mono_png = out_dir / "monotonic_loss_standard.png"
    mono_pdf = out_dir / "monotonic_loss_standard.pdf"
    pcd_png = out_dir / "min_rule_pcd_standard.png"
    pcd_pdf = out_dir / "min_rule_pcd_standard.pdf"
    reg_png = out_dir / "regression_loss_standard.png"
    reg_pdf = out_dir / "regression_loss_standard.pdf"
    tri_png = out_dir / "sensitivity_standard_triptych.png"
    tri_pdf = out_dir / "sensitivity_standard_triptych.pdf"

    plot_multiline(
        df=df2,
        col_lm=col_map["lambda_m"],
        col_ls=col_map["lambda_s"],
        col_y=col_map["monotonic_loss"],
        y_label=r"$E_M$",
        title="Effect of Hyperparameters on Monotonicity Loss",
        out_png=mono_png,
        out_pdf=mono_pdf,
        best_row=best,
        annotate_dx=0.34,
        annotate_dy=0.26,
        annotate_connector_color="#4A4A4A",
        annotate_connector_lw=0.75,
        annotate_connector_shrinkA=4,
        annotate_connector_shrinkB=4,
        annotate_connector_style="arc3,rad=0.10",
        annotate_connector_mutation_scale=9,
        annotate_connector_alpha=0.85,
        annotate_star_size=156,
        annotate_star_facecolor="#E76F5A",
        annotate_star_edgecolor="#4A4A4A",
        annotate_star_edgewidth=0.75,
        annotate_star_alpha=0.95,
        annotate_box_edgecolor="0.68",
        annotate_box_alpha=0.92,
        annotate_box_linewidth=0.75,
        annotate_textcoords="axes fraction",
        annotate_enforce_manual_position=True,
        add_inset=True,
        inset_rect=(0.45, 0.54, 0.33, 0.32),
        inset_xlim=(-5.0, 120.0),
        inset_use_symlog=True,
        inset_draw_zero_line=True,
        use_symlog=True,
        draw_zero_line=True,
    )
    plot_multiline(
        df=df2,
        col_lm=col_map["lambda_m"],
        col_ls=col_map["lambda_s"],
        col_y=col_map["min_rule_pcd_test"],
        y_label="Minimum rule-level PCD",
        title="Effect of Hyperparameters on Minimum Rule-level PCD",
        out_png=pcd_png,
        out_pdf=pcd_pdf,
        best_row=best,
        annotate_dx=95,
        annotate_dy=-38,
        use_symlog=False,
        draw_zero_line=False,
    )
    plot_multiline(
        df=df2,
        col_lm=col_map["lambda_m"],
        col_ls=col_map["lambda_s"],
        col_y=col_map["regression_loss"],
        y_label=r"$E_R$",
        title="Effect of Hyperparameters on Regression Loss",
        out_png=reg_png,
        out_pdf=reg_pdf,
        best_row=best,
        annotate_dx=255.0,
        annotate_dy=0.000205,
        annotate_connector_color="#444444",
        annotate_connector_lw=0.8,
        annotate_connector_shrinkA=4,
        annotate_connector_shrinkB=4,
        annotate_connector_style="arc3,rad=0.12",
        annotate_connector_mutation_scale=9,
        annotate_connector_alpha=0.9,
        annotate_star_size=156,
        annotate_star_facecolor="#E76F5A",
        annotate_star_edgecolor="#4A4A4A",
        annotate_star_edgewidth=0.75,
        annotate_star_alpha=0.95,
        annotate_box_edgecolor="0.68",
        annotate_box_alpha=0.92,
        annotate_box_linewidth=0.75,
        annotate_textcoords="data",
        annotate_enforce_manual_position=True,
        add_inset=False,
        use_symlog=False,
        draw_zero_line=False,
    )
    plot_triptych(
        df=df2,
        col_lm=col_map["lambda_m"],
        col_ls=col_map["lambda_s"],
        col_reg=col_map["regression_loss"],
        col_mono=col_map["monotonic_loss"],
        col_pcd=col_map["min_rule_pcd_test"],
        best_row=best,
        out_png=tri_png,
        out_pdf=tri_pdf,
    )

    summary_path = out_dir / "summary_standard_figures.txt"
    lines = [
        f"data_file: {data_file}",
        f"data_points: {len(df2)}",
        "column_mapping:",
    ]
    for k, v in col_map.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "best_point_rule:",
        "1) max min_rule_pcd_test",
        "2) min regression loss",
        "3) min monotonic loss",
        "4) min lambda_m",
        "5) min lambda_s",
        "",
        f"best_point: lambda_s={float(best[col_map['lambda_s']]):g}, lambda_m={float(best[col_map['lambda_m']]):g}",
        f"best_regression_loss: {float(best[col_map['regression_loss']]):.12g}",
        f"best_monotonic_loss: {float(best[col_map['monotonic_loss']]):.12g}",
        f"best_min_rule_pcd_test: {float(best[col_map['min_rule_pcd_test']]):.12g}",
        "",
        "generated_files:",
        f"- {mono_png}",
        f"- {mono_pdf}",
        f"- {pcd_png}",
        f"- {pcd_pdf}",
        f"- {reg_png}",
        f"- {reg_pdf}",
        f"- {tri_png}",
        f"- {tri_pdf}",
        f"- {summary_path}",
        "",
        "annotation style updated",
        "annotation positions adjusted to avoid overlapping curves",
        "style updated to publication-like reference style",
        "white background applied",
        "muted academic color palette applied",
        "y-axis labels updated to E_M and E_R",
        "curve colors aligned to reference style",
        "best-point star restyled to publication-friendly dark red marker",
        "annotation positions refined for cleaner layout",
        "best-point star size reduced",
        "best-point star color softened for publication style",
        "curve colors brightened to cleaner publication palette",
        "regression-loss best annotation repositioned to avoid overlap",
        "curve colors updated to brighter teacher-style publication palette",
        "all lambda subscripts standardized to uppercase",
        "background forced to pure white",
        "regression-loss best annotation further repositioned",
        "left x-margin adjusted for less crowded low-lambda region",
        "regression-loss best connector refined",
        "monotonic-loss best label moved to lower-right of star",
        "annotation connector curves smoothed for publication style",
        "best labels manually repositioned according to user-marked target areas",
        "monotonic-loss best moved to upper-right blank region",
        "regression-loss best moved to lower-right blank region",
        "regression-loss best label repositioned again",
        "best connector arrows refined to smoother publication style",
        "best star reduced and slightly brightened",
        "inset zoom added for low-lambda_M region",
        "best labels moved closer to best points",
        "best connector lines shortened and softened",
        "monotonic inset shifted left to avoid legend overlap",
        "regression inset removed",
        "regression best label moved to lower-right of star with shorter connector",
        "best stars brightened for clearer publication emphasis",
        "best stars made more prominent",
        "regression-loss best label manually moved to user-specified lower-right area",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("Standard sensitivity figures exported successfully.")
    print(f"Data file: {data_file}")
    print(f"Output dir: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
