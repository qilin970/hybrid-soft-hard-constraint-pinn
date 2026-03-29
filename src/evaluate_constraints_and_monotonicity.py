"""Constraint and monotonicity figures for the fixed candidate final model.

This script only performs evaluation / plotting:
- boundary verification figures and summary tables
- senior-style finite-difference PCD tables on the real train/test sets
- 2D partial dependence contour plots on Temperature and ER

It does not retrain, retune, or modify the model.
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import FormatStrFormatter


OUTPUT_NAMES = ["N2", "H2", "CO", "CO2", "CH4"]


def load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.linewidth": 1.1,
            "savefig.dpi": 360,
        }
    )


def find_existing(path_or_none: str | None, fallback: Path) -> Path:
    if path_or_none:
        p = Path(path_or_none)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    if not fallback.exists():
        raise FileNotFoundError(fallback)
    return fallback


def make_model(base_mod, cfg, bundle, device: torch.device):
    model = base_mod.HardBoundaryANN(
        in_dim=bundle["x_tr_full"].shape[1],
        hidden_dim=int(cfg.hidden_dim),
        out_dim=len(OUTPUT_NAMES),
        c=bundle["c"],
        s=bundle["s"],
        activation=str(cfg.activation),
        distance_feature_idx=bundle["distance_feature_idx"],
    ).to(device)
    return model


@torch.no_grad()
def predict_array(model, x_np: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    x = torch.tensor(x_np, dtype=torch.float32)
    preds: List[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        yp = model(xb).cpu().numpy()
        preds.append(yp)
    return np.vstack(preds)


def compute_component_metrics(y_true: np.ndarray, y_pred: np.ndarray, split: str) -> pd.DataFrame:
    rows = []
    for i, comp in enumerate(OUTPUT_NAMES):
        yt = y_true[:, i] * 100.0
        yp = y_pred[:, i] * 100.0
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        ss_res = float(np.sum((yp - yt) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append({"split": split, "component": comp, "R2": r2, "RMSE": rmse})
    return pd.DataFrame(rows)


def build_boundary_sample_table(yhat: np.ndarray, split: str) -> pd.DataFrame:
    sum_frac = yhat.sum(axis=1)
    min_frac = yhat.min(axis=1)
    return pd.DataFrame(
        {
            "split": split,
            "sample_id": np.arange(len(yhat), dtype=int),
            "sum_pred_frac": sum_frac,
            "sum_pred_volpct": sum_frac * 100.0,
            "sum_abs_err": np.abs(sum_frac - 1.0),
            "min_component_frac": min_frac,
            "min_component_volpct": min_frac * 100.0,
        }
    )


def summarize_boundary(yhat: np.ndarray, split: str) -> Dict[str, float | str]:
    sum_frac = yhat.sum(axis=1)
    sum_abs = np.abs(sum_frac - 1.0)
    min_frac = yhat.min(axis=1)
    return {
        "split": split,
        "negative_ratio": float((yhat < 0.0).mean()),
        "min_component_min": float(min_frac.min()),
        "sum_mean": float(sum_frac.mean()),
        "sum_min": float(sum_frac.min()),
        "sum_max": float(sum_frac.max()),
        "sum_abs_err_mean": float(sum_abs.mean()),
        "sum_abs_err_max": float(sum_abs.max()),
    }


def _plot_dist_with_minmax(
    ax,
    data_list: List[np.ndarray],
    labels: List[str],
    ylabel: str,
    title: str,
    reference_y: float | None,
    value_fmt: str,
    color: str,
) -> None:
    rng = np.random.default_rng(20260320)
    positions = np.arange(1, len(data_list) + 1)

    for pos, label, values in zip(positions, labels, data_list):
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            s=18,
            alpha=0.55,
            color=color,
            edgecolors="none",
            zorder=2,
        )

        q1, med, q3 = np.percentile(values, [25, 50, 75])
        vmin = float(values.min())
        vmax = float(values.max())
        mean_v = float(values.mean())

        ax.vlines(pos, vmin, vmax, color="#3a3a3a", lw=1.4, zorder=3)
        ax.hlines([q1, q3], pos - 0.12, pos + 0.12, color="#3a3a3a", lw=1.4, zorder=3)
        ax.hlines(med, pos - 0.16, pos + 0.16, color="black", lw=2.0, zorder=4)
        ax.scatter([pos], [mean_v], marker="D", s=36, color="white", edgecolors="black", zorder=5)
        ax.scatter([pos], [vmin], marker="v", s=36, color="#7f1d1d", zorder=5)
        ax.scatter([pos], [vmax], marker="^", s=36, color="#14532d", zorder=5)

        ann = f"min={format(vmin, value_fmt)}\nmax={format(vmax, value_fmt)}"
        ax.annotate(
            ann,
            xy=(pos, vmax),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bfbfbf", alpha=0.9),
        )

    if reference_y is not None:
        ax.axhline(reference_y, color="black", lw=1.0, ls=(0, (4, 4)), alpha=0.85, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.25)


def plot_boundary_verification(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    fig.patch.set_facecolor("#f4f1ec")
    for ax in axes:
        ax.set_facecolor("#f9f7f4")

    _plot_dist_with_minmax(
        ax=axes[0],
        data_list=[train_df["sum_pred_volpct"].to_numpy(), test_df["sum_pred_volpct"].to_numpy()],
        labels=["Training Set", "Testing Set"],
        ylabel="Sum of predicted gas fractions (vol.%)",
        title="Sum-to-One Verification",
        reference_y=100.0,
        value_fmt=".6f",
        color="#1f77b4",
    )
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.6f"))

    _plot_dist_with_minmax(
        ax=axes[1],
        data_list=[train_df["min_component_volpct"].to_numpy(), test_df["min_component_volpct"].to_numpy()],
        labels=["Training Set", "Testing Set"],
        ylabel=r"Sample-wise minimum predicted fraction $\min_j(\hat{y}_j)$ (vol.%)",
        title="Non-Negativity Verification",
        reference_y=0.0,
        value_fmt=".3f",
        color="#d97706",
    )
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def senior_rule_violation_table(
    runner,
    base_mod,
    model,
    x_np: np.ndarray,
    x_cols: List[str],
    rules: List[Tuple[int, int, float]],
    delta_scaled: float,
    forward_kind: str,
    split_name: str,
    axis_label: str,
    device: torch.device,
) -> pd.DataFrame:
    x_base = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_base = runner.forward_by_kind(base_mod, model, x_base, forward_kind)

    rows = []
    for out_i, in_i, sign in rules:
        x_shift = x_base.clone()
        x_shift[:, in_i] = x_shift[:, in_i] + float(delta_scaled)
        y_shift = runner.forward_by_kind(base_mod, model, x_shift, forward_kind)
        diff = y_shift[:, out_i] - y_base[:, out_i]
        if sign > 0:
            violation = (diff < 0.0).float()
            expected = "increasing"
        else:
            violation = (diff > 0.0).float()
            expected = "decreasing"

        violation_rate = float(violation.mean().item())
        compliance_rate = 1.0 - violation_rate
        rows.append(
            {
                "Data set": split_name,
                "axis": axis_label,
                "component": OUTPUT_NAMES[out_i],
                "input_column": x_cols[in_i],
                "expected": expected,
                "delta_scaled": float(delta_scaled),
                "forward_kind": forward_kind,
                "violation_rate": violation_rate,
                "compliance_rate": compliance_rate,
            }
        )
    return pd.DataFrame(rows)


def build_senior_pcd_tables(
    runner,
    base_mod,
    model,
    bundle,
    delta_scaled: float,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_cols = bundle["x_cols"]
    x_idx = {c: i for i, c in enumerate(x_cols)}
    out_idx = {o: i for i, o in enumerate(OUTPUT_NAMES)}
    if base_mod.TEMP_COL not in x_idx or base_mod.ER_COL not in x_idx:
        raise KeyError(f"Missing monotonic columns: {base_mod.TEMP_COL}, {base_mod.ER_COL}")
    mono_rules = [
        (out_idx["H2"], x_idx[base_mod.ER_COL], -1.0),
        (out_idx["CO"], x_idx[base_mod.ER_COL], -1.0),
        (out_idx["CO2"], x_idx[base_mod.ER_COL], +1.0),
        (out_idx["CH4"], x_idx[base_mod.ER_COL], -1.0),
        (out_idx["N2"], x_idx[base_mod.ER_COL], +1.0),
        (out_idx["H2"], x_idx[base_mod.TEMP_COL], +1.0),
        (out_idx["CO"], x_idx[base_mod.TEMP_COL], +1.0),
        (out_idx["CO2"], x_idx[base_mod.TEMP_COL], -1.0),
    ]
    er_rules = [r for r in mono_rules if x_cols[r[1]] == base_mod.ER_COL]
    temp_rules = [r for r in mono_rules if x_cols[r[1]] == base_mod.TEMP_COL]

    detailed_parts = []
    split_map = {
        "Training Set": bundle["x_tr_full"],
        "Testing Set": bundle["x_test"],
    }
    for split_name, x_np in split_map.items():
        detailed_parts.append(
            senior_rule_violation_table(
                runner=runner,
                base_mod=base_mod,
                model=model,
                x_np=x_np,
                x_cols=x_cols,
                rules=er_rules,
                delta_scaled=delta_scaled,
                forward_kind="full",
                split_name=split_name,
                axis_label="ER",
                device=device,
            )
        )
        detailed_parts.append(
            senior_rule_violation_table(
                runner=runner,
                base_mod=base_mod,
                model=model,
                x_np=x_np,
                x_cols=x_cols,
                rules=temp_rules,
                delta_scaled=delta_scaled,
                forward_kind="full",
                split_name=split_name,
                axis_label="Temperature",
                device=device,
            )
        )

    detailed_df = pd.concat(detailed_parts, ignore_index=True)
    er_df = (
        detailed_df[detailed_df["axis"] == "ER"]
        .pivot(index="Data set", columns="component", values="compliance_rate")
        .reset_index()
    )
    temp_df = (
        detailed_df[detailed_df["axis"] == "Temperature"]
        .pivot(index="Data set", columns="component", values="compliance_rate")
        .reset_index()
    )

    er_df = er_df[["Data set", "N2", "H2", "CO", "CO2", "CH4"]]
    temp_df = temp_df[["Data set", "H2", "CO", "CO2"]]
    return er_df, temp_df, detailed_df


def build_monotonic_metric_comparison(
    best_payload: Dict,
    senior_detail_df: pd.DataFrame,
) -> pd.DataFrame:
    selected = best_payload.get("selected_record", {})
    training_scope = str(best_payload.get("mono_training_source", "real_samples"))
    rows = []
    if "E_M_pair_syn" in selected:
        rows.append(
            {
                "metric_family": "pairwise_synthetic_training_eval",
                "scope": "synthetic_pairs",
                "split": "global",
                "axis": "global",
                "component": "all",
                "metric_name": "E_M_pair_syn",
                "value": float(selected.get("E_M_pair_syn", float("nan"))),
            }
        )
    if "E_M_pair_rate_syn" in selected:
        rows.append(
            {
                "metric_family": "pairwise_synthetic_training_eval",
                "scope": "synthetic_pairs",
                "split": "global",
                "axis": "global",
                "component": "all",
                "metric_name": "E_M_pair_rate_syn",
                "value": float(selected.get("E_M_pair_rate_syn", float("nan"))),
            }
        )
    if "E_M_diff_train" in selected:
        rows.append(
            {
                "metric_family": "differential_monotonic_training_eval",
                "scope": training_scope,
                "split": "Training Set",
                "axis": "global",
                "component": "all",
                "metric_name": "E_M_diff_train",
                "value": float(selected.get("E_M_diff_train", float("nan"))),
            }
        )
    if "E_M_seniorfd_train" in selected:
        rows.append(
            {
                "metric_family": "differential_monotonic_training_eval",
                "scope": best_payload.get("mono_report_source", "real_train_test_sets"),
                "split": "Training Set",
                "axis": "global",
                "component": "all",
                "metric_name": "E_M_seniorfd_train",
                "value": float(selected.get("E_M_seniorfd_train", float("nan"))),
            }
        )
    if "E_M_seniorfd_test" in selected:
        rows.append(
            {
                "metric_family": "differential_monotonic_training_eval",
                "scope": best_payload.get("mono_report_source", "real_train_test_sets"),
                "split": "Testing Set",
                "axis": "global",
                "component": "all",
                "metric_name": "E_M_seniorfd_test",
                "value": float(selected.get("E_M_seniorfd_test", float("nan"))),
            }
        )
    for _, row in senior_detail_df.iterrows():
        rows.append(
            {
                "metric_family": "senior_finite_difference_report_eval",
                "scope": "real_samples",
                "split": row["Data set"],
                "axis": row["axis"],
                "component": row["component"],
                "metric_name": "violation_rate",
                "value": float(row["violation_rate"]),
            }
        )
        rows.append(
            {
                "metric_family": "senior_finite_difference_report_eval",
                "scope": "real_samples",
                "split": row["Data set"],
                "axis": row["axis"],
                "component": row["component"],
                "metric_name": "compliance_rate",
                "value": float(row["compliance_rate"]),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def compute_pda_surfaces(
    model,
    background_raw: np.ndarray,
    scaler,
    x_cols: List[str],
    temp_values: np.ndarray,
    er_values: np.ndarray,
    temp_col: str,
    er_col: str,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    temp_idx = x_cols.index(temp_col)
    er_idx = x_cols.index(er_col)

    temp_mesh, er_mesh = np.meshgrid(temp_values, er_values)
    temp_flat = temp_mesh.reshape(-1)
    er_flat = er_mesh.reshape(-1)
    n_grid = temp_flat.size

    accum = np.zeros((n_grid, len(OUTPUT_NAMES)), dtype=np.float64)
    model.eval()

    for bg_row in background_raw:
        x_grid = np.repeat(bg_row.reshape(1, -1), n_grid, axis=0)
        x_grid[:, temp_idx] = temp_flat
        x_grid[:, er_idx] = er_flat
        x_scaled = scaler.transform(x_grid)
        preds = predict_array(model, x_scaled, device=device, batch_size=batch_size)
        accum += preds

    mean_pred = accum / float(len(background_raw))
    surfaces = {
        comp: mean_pred[:, i].reshape(len(er_values), len(temp_values)) * 100.0
        for i, comp in enumerate(OUTPUT_NAMES)
    }
    surfaces["Temperature"] = temp_mesh
    surfaces["ER"] = er_mesh
    return surfaces


def save_pda_surface_tables(
    surfaces: Dict[str, np.ndarray],
    out_xlsx: Path,
    out_dir: Path,
) -> None:
    with pd.ExcelWriter(out_xlsx) as writer:
        for comp in ["H2", "CO", "CO2"]:
            df = pd.DataFrame(
                {
                    "Temperature": surfaces["Temperature"].reshape(-1),
                    "ER": surfaces["ER"].reshape(-1),
                    "predicted_volpct": surfaces[comp].reshape(-1),
                }
            )
            df.to_excel(writer, sheet_name=f"{comp}_surface", index=False)
            df.to_csv(out_dir / f"pda_surface_{comp}.csv", index=False)


def plot_single_pda(
    temp_mesh: np.ndarray,
    er_mesh: np.ndarray,
    z_volpct: np.ndarray,
    component: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    setup_plot_style()
    levels = np.linspace(float(z_volpct.min()), float(z_volpct.max()), 18)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    fig.patch.set_facecolor("#f4f1ec")
    ax.set_facecolor("#f9f7f4")
    cf = ax.contourf(temp_mesh, er_mesh, z_volpct, levels=levels, cmap="YlGnBu")
    ax.contour(temp_mesh, er_mesh, z_volpct, levels=levels[::2], colors="black", linewidths=0.6, alpha=0.55)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Predicted volume fraction (vol.%)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("ER")
    ax.set_title(f"2D-PDA of {component}")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_three_panel_pda(
    surfaces: Dict[str, np.ndarray],
    out_png: Path,
    out_pdf: Path,
) -> None:
    setup_plot_style()
    comps = ["H2", "CO", "CO2"]
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))
    fig.patch.set_facecolor("#f4f1ec")

    for ax, comp in zip(axes, comps):
        ax.set_facecolor("#f9f7f4")
        z = surfaces[comp]
        levels = np.linspace(float(z.min()), float(z.max()), 18)
        cf = ax.contourf(surfaces["Temperature"], surfaces["ER"], z, levels=levels, cmap="YlGnBu")
        ax.contour(surfaces["Temperature"], surfaces["ER"], z, levels=levels[::2], colors="black", linewidths=0.55, alpha=0.55)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.92)
        cbar.set_label("Predicted volume fraction (vol.%)")
        ax.set_title(comp)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("ER")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def assess_surface_trend(surface: np.ndarray, temp_sign: float, er_sign: float, tol: float = 1e-9) -> Dict[str, float | str]:
    diff_temp = np.diff(surface, axis=1)
    diff_er = np.diff(surface, axis=0)
    temp_ratio = float((temp_sign * diff_temp >= -tol).mean())
    er_ratio = float((er_sign * diff_er >= -tol).mean())
    worst_temp = float((temp_sign * diff_temp).min())
    worst_er = float((er_sign * diff_er).min())

    if temp_ratio >= 0.98 and er_ratio >= 0.98:
        note = "Overall trend is strongly consistent with the expected monotonic pattern."
    elif temp_ratio >= 0.90 and er_ratio >= 0.90:
        note = "Overall trend is consistent, with only mild local irregularities."
    else:
        note = "Trend is only partially consistent; visible local irregularities remain."

    return {
        "temp_compliance_ratio": temp_ratio,
        "er_compliance_ratio": er_ratio,
        "worst_signed_temp_step": worst_temp,
        "worst_signed_er_step": worst_er,
        "assessment": note,
    }


def build_monotonic_summary(
    best_payload: Dict,
    boundary_summary_df: pd.DataFrame,
    pcd_er_df: pd.DataFrame,
    pcd_t_df: pd.DataFrame,
    pcd_detail_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    pda_trends: Dict[str, Dict[str, float | str]],
) -> Dict:
    selected = best_payload.get("selected_record", {})
    training_metrics = {}
    for key in ["E_M_pair_syn", "E_M_pair_rate_syn", "E_M_diff_train", "E_M_seniorfd_train", "E_M_seniorfd_test"]:
        if key in selected:
            training_metrics[key] = float(selected[key])
    return {
        "boundary_summary": boundary_summary_df.to_dict(orient="records"),
        "pcd_report_method": {
            "style": "senior-style finite-difference evaluation only",
            "forward_kind": "full",
            "reported_value": "compliance_rate = 1 - violation_rate",
            "detail_records": pcd_detail_df.to_dict(orient="records"),
            "against_er_table": pcd_er_df.to_dict(orient="records"),
            "against_temperature_table": pcd_t_df.to_dict(orient="records"),
        },
        "training_monotonic_metrics": {
            **training_metrics,
            "mono_eval_metric": best_payload.get("mono_eval_metric", best_payload.get("mono_report_metric", "")),
            "mono_training_source": best_payload.get("mono_training_source", "unknown"),
            "mono_report_source": best_payload.get("mono_report_source", "real_train_test_sets"),
            "lambda_p_effect_note": (
                "Earlier two-point diagnostics already showed that lambda_p now changes the residual trajectory "
                "after restricting D(x) to the 8 continuous variables."
            ),
        },
        "metric_comparison_table": comparison_df.to_dict(orient="records"),
        "pda_surface_trend_assessment": pda_trends,
    }


def build_paper_pcd_tables(pcd_detail_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rule_order = {
        "ER->H2": 0,
        "ER->CO2": 1,
        "ER->CO": 2,
        "ER->CH4": 3,
        "ER->N2": 4,
        "T->H2": 5,
        "T->CO2": 6,
        "T->CO": 7,
        "T->CH4": 8,
        "T->N2": 9,
    }
    detail = pcd_detail_df.copy()
    detail["rule"] = detail.apply(lambda r: ("ER" if str(r["axis"]) == "ER" else "T") + f"->{r['component']}", axis=1)
    detail["rule_order"] = detail["rule"].map(rule_order).astype(float)
    detail = detail.rename(columns={"Data set": "split", "compliance_rate": "Pm", "violation_rate": "1_minus_Pm"})
    detail = detail[["split", "rule", "rule_order", "axis", "component", "expected", "Pm", "1_minus_Pm", "delta_scaled", "forward_kind"]]
    detail = detail.sort_values(["split", "rule_order", "rule"]).reset_index(drop=True)

    overall = (
        detail.groupby(["split"], as_index=False)
        .agg(Pm=("Pm", "mean"), rule_count=("Pm", "size"))
        .sort_values(["split"])
    )
    by_component = (
        detail.groupby(["split", "component"], as_index=False)
        .agg(Pm=("Pm", "mean"), rule_count=("Pm", "size"))
        .sort_values(["split", "component"])
    )
    return detail, overall, by_component


def write_default_pcd_aliases(out_dir: Path) -> Dict[str, str]:
    """Write default aliases from already-generated paper-style outputs only."""
    required = {
        "pcd_overall.xlsx": out_dir / "pcd_overall_paper.xlsx",
        "pcd_by_rule.xlsx": out_dir / "PCD_detailed_rules_paper.xlsx",
        "pcd_by_component.xlsx": out_dir / "pcd_by_component_paper.xlsx",
        "summary.txt": out_dir / "monotonicity_summary_paper.json",
    }
    missing = [k for k, v in required.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required paper-style PCD outputs: {missing}")

    alias_sources: Dict[str, str] = {}
    for alias_name in ["pcd_overall.xlsx", "pcd_by_rule.xlsx", "pcd_by_component.xlsx"]:
        src = required[alias_name]
        dst = out_dir / alias_name
        dst.write_bytes(src.read_bytes())
        alias_sources[alias_name] = src.name

    summary_src = required["summary.txt"]
    payload = json.loads(summary_src.read_text(encoding="utf-8"))
    lines = [
        "Default PCD Summary (Paper-Style Pm)",
        f"- Source JSON: {summary_src.name}",
        "- pcd_overall.xlsx source: pcd_overall_paper.xlsx",
        "- pcd_by_rule.xlsx source: PCD_detailed_rules_paper.xlsx",
        "- pcd_by_component.xlsx source: pcd_by_component_paper.xlsx",
    ]
    detail = payload.get("pcd_report_method", {}).get("detail_records", [])
    if isinstance(detail, list) and len(detail) > 0:
        df_detail = pd.DataFrame(detail)
        split_col = "split" if "split" in df_detail.columns else ("Data set" if "Data set" in df_detail.columns else None)
        pm_col = "Pm" if "Pm" in df_detail.columns else ("compliance_rate" if "compliance_rate" in df_detail.columns else None)
        if split_col is not None and pm_col is not None:
            lines.append("- Pm mean by split:")
            split_stats = df_detail.groupby(split_col, as_index=False)[pm_col].mean()
            for _, row in split_stats.iterrows():
                lines.append(f"  - {row[split_col]}: {float(row[pm_col]):.6f}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    alias_sources["summary.txt"] = summary_src.name

    (out_dir / "default_pcd_alias_sources.json").write_text(json.dumps(alias_sources, indent=2), encoding="utf-8")
    return alias_sources


def write_markdown_summary(
    out_path: Path,
    boundary_summary: pd.DataFrame,
    pcd_er: pd.DataFrame,
    pcd_t: pd.DataFrame,
    pda_trends: Dict[str, Dict[str, float | str]],
    best_payload: Dict,
) -> None:
    selected = best_payload.get("selected_record", {})
    def frame_to_md(df: pd.DataFrame) -> List[str]:
        cols = [str(c) for c in df.columns]
        rows = []
        rows.append("| " + " | ".join(cols) + " |")
        rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in df.iterrows():
            vals = []
            for c in df.columns:
                v = row[c]
                if isinstance(v, (float, np.floating)):
                    vals.append(f"{float(v):.6f}")
                else:
                    vals.append(str(v))
            rows.append("| " + " | ".join(vals) + " |")
        return rows

    lines = [
        "# Constraint and Monotonicity Summary",
        "",
        "## Boundary Constraints",
    ]
    for _, row in boundary_summary.iterrows():
        lines.append(
            f"- {row['split']}: negative_ratio={row['negative_ratio']:.6g}, "
            f"min_component_min={row['min_component_min']:.6g}, "
            f"sum_mean={row['sum_mean']:.9f}, sum_range=[{row['sum_min']:.9f}, {row['sum_max']:.9f}], "
            f"sum_abs_err_mean={row['sum_abs_err_mean']:.6g}, sum_abs_err_max={row['sum_abs_err_max']:.6g}"
        )

    lines += [
        "",
        "## Monotonic Compliance (Senior-Style PCD)",
        "",
        "- Evaluation uses finite-difference monotonic checking on the real training/testing sets.",
        "- Forward path uses the full model.",
        "- The reported table value is compliance_rate = 1 - violation_rate.",
        "",
        "### Against ER",
        *frame_to_md(pcd_er),
        "",
        "### Against Temperature",
        *frame_to_md(pcd_t),
        "",
        "## 2D-PDA Trend Check",
    ]
    for comp, stats in pda_trends.items():
        lines.append(
            f"- {comp}: Temp compliance={stats['temp_compliance_ratio']:.4f}, "
            f"ER compliance={stats['er_compliance_ratio']:.4f}. {stats['assessment']}"
        )

    lines += [
        "",
        "## Pairwise Monotonic Training/Evaluation Indicator",
    ]
    if "E_M_pair_syn" in selected:
        lines += [
            f"- E_M_pair_syn = {float(selected.get('E_M_pair_syn', float('nan'))):.10f}",
            f"- E_M_pair_rate_syn = {float(selected.get('E_M_pair_rate_syn', float('nan'))):.10f}",
            "- This pairwise synthetic metric is the training/evaluation indicator inside the current method, not the final paper-reported PCD table.",
        ]
    if "E_M_diff_train" in selected:
        lines += [
            f"- E_M_diff_train = {float(selected.get('E_M_diff_train', float('nan'))):.10f}",
            f"- E_M_seniorfd_train = {float(selected.get('E_M_seniorfd_train', float('nan'))):.10f}",
            f"- E_M_seniorfd_test = {float(selected.get('E_M_seniorfd_test', float('nan'))):.10f}",
            f"- In the senior-injection branch, E_M_diff_train is the training backprop metric on {best_payload.get('mono_training_source', 'the configured monotonic source')}, while the finite-difference PCD table is the paper reporting metric on {best_payload.get('mono_report_source', 'real train/test sets')}.",
        ]
    lines += [
        "- lambda_p trajectory note: earlier two-point diagnostics confirmed that lambda_p now changes the residual branch trajectory.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate boundary and monotonicity figures for the fixed candidate model.")
    parser.add_argument("--best-dir", type=str, required=True, help="Path to best_model directory.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--runner-py", type=str, default=None)
    parser.add_argument("--base-py", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--pda-grid-size", type=int, default=81)
    args = parser.parse_args()

    best_dir = Path(args.best_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    src_dir = Path(__file__).resolve().parent
    runner_py = find_existing(args.runner_py, src_dir / "fixedsplit_utils.py")
    base_py = find_existing(args.base_py, src_dir / "core_model.py")
    runner = load_module(runner_py, "runner_fixedsplit")
    base_mod = load_module(base_py, "base_hardsoft")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    best_cfg_path = best_dir / "best_config.json"
    model_path = best_dir / "best_model.pth"
    run_config_path = best_dir.parent / "run_config.xlsx"
    if not best_cfg_path.exists():
        raise FileNotFoundError(best_cfg_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not run_config_path.exists():
        raise FileNotFoundError(run_config_path)

    best_payload = json.loads(best_cfg_path.read_text(encoding="utf-8"))
    cfg = runner.load_run_config_json(best_cfg_path)
    run_meta = pd.read_excel(run_config_path).iloc[0].to_dict()
    default_data_xlsx = src_dir.parent / "data" / "experiment_data.xlsx"
    data_xlsx_from_meta = Path(str(run_meta.get("data", default_data_xlsx)))
    if not data_xlsx_from_meta.exists():
        data_xlsx_from_meta = default_data_xlsx
    data_xlsx = find_existing(args.data, data_xlsx_from_meta)
    rand_label_mat = Path(str(best_payload["split"]["rand_label_mat"]))
    if not rand_label_mat.exists():
        rand_label_mat = src_dir.parent / "data" / "Rand_lable.mat"

    bundle = runner.prepare_bundle_fixedsplit(
        base_mod=base_mod,
        data_xlsx=data_xlsx,
        rand_label_mat=rand_label_mat,
        rand_row=int(best_payload["split"]["rand_row"]),
        train_count=int(best_payload["split"]["train_count"]),
        val_ratio_in_train=float(run_meta.get("val_ratio_in_train", 0.1)),
        val_seed=int(run_meta.get("val_seed", 42)),
        syn_temp_min=float(best_payload["synthetic_grid"]["temp_min"]),
        syn_temp_max=float(best_payload["synthetic_grid"]["temp_max"]),
        syn_temp_points=int(best_payload["synthetic_grid"]["temp_points"]),
        syn_er_min=float(best_payload["synthetic_grid"]["er_min"]),
        syn_er_max=float(best_payload["synthetic_grid"]["er_max"]),
        syn_er_points=int(best_payload["synthetic_grid"]["er_points"]),
    )

    model = make_model(base_mod, cfg, bundle, device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    yhat_train = predict_array(model, bundle["x_tr_full"], device=device)
    yhat_test = predict_array(model, bundle["x_test"], device=device)

    comp_metrics = pd.concat(
        [
            compute_component_metrics(bundle["y_tr_full"], yhat_train, "train"),
            compute_component_metrics(bundle["y_test"], yhat_test, "test"),
        ],
        ignore_index=True,
    )
    comp_metrics.to_excel(out_dir / "component_metrics_train_test.xlsx", index=False)
    comp_metrics_full = pd.concat(
        [
            comp_metrics,
            comp_metrics.groupby(["split"], as_index=False).agg(R2=("R2", "mean"), RMSE=("RMSE", "mean")).assign(component="overall_mean"),
        ],
        ignore_index=True,
    )
    comp_metrics_full.to_excel(out_dir / "component_metrics_full.xlsx", index=False)

    with torch.no_grad():
        d_train = model.distance(torch.tensor(bundle["x_tr_full"], dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
        d_test = model.distance(torch.tensor(bundle["x_test"], dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
    distance_stats_df = pd.DataFrame(
        [
            {"split": "train", "D_mean": float(np.mean(d_train)), "D_min": float(np.min(d_train)), "D_max": float(np.max(d_train)), "n_samples": int(d_train.shape[0])},
            {"split": "test", "D_mean": float(np.mean(d_test)), "D_min": float(np.min(d_test)), "D_max": float(np.max(d_test)), "n_samples": int(d_test.shape[0])},
        ]
    )
    distance_stats_df.to_excel(out_dir / "distance_stats.xlsx", index=False)
    distance_payload = {
        "D_mean_train": float(np.mean(d_train)),
        "D_min_train": float(np.min(d_train)),
        "D_max_train": float(np.max(d_train)),
        "D_mean_test": float(np.mean(d_test)),
        "D_min_test": float(np.min(d_test)),
        "D_max_test": float(np.max(d_test)),
    }
    (out_dir / "distance_stats.json").write_text(json.dumps(distance_payload, indent=2), encoding="utf-8")

    train_boundary_df = build_boundary_sample_table(yhat_train, "Training Set")
    test_boundary_df = build_boundary_sample_table(yhat_test, "Testing Set")
    boundary_summary_df = pd.DataFrame(
        [
            summarize_boundary(yhat_train, "Training Set"),
            summarize_boundary(yhat_test, "Testing Set"),
        ]
    )

    with pd.ExcelWriter(out_dir / "boundary_verification_stats.xlsx") as writer:
        boundary_summary_df.to_excel(writer, sheet_name="summary", index=False)
        train_boundary_df.to_excel(writer, sheet_name="train_samples", index=False)
        test_boundary_df.to_excel(writer, sheet_name="test_samples", index=False)
    pd.concat([train_boundary_df, test_boundary_df], ignore_index=True).to_csv(
        out_dir / "boundary_verification_stats.csv",
        index=False,
    )

    plot_boundary_verification(
        train_df=train_boundary_df,
        test_df=test_boundary_df,
        out_png=out_dir / "boundary_verification_main.png",
        out_pdf=out_dir / "boundary_verification_main.pdf",
    )

    pcd_er_df, pcd_t_df, pcd_detail_df = build_senior_pcd_tables(
        runner=runner,
        base_mod=base_mod,
        model=model,
        bundle=bundle,
        delta_scaled=float(cfg.mono_delta_scaled),
        device=device,
    )
    pcd_er_df.to_excel(out_dir / "PCD_against_ER_train_test_senior.xlsx", index=False)
    pcd_t_df.to_excel(out_dir / "PCD_against_T_train_test_senior.xlsx", index=False)
    pcd_er_df.to_csv(out_dir / "PCD_against_ER_train_test_senior.csv", index=False)
    pcd_t_df.to_csv(out_dir / "PCD_against_T_train_test_senior.csv", index=False)
    pcd_detail_df.to_excel(out_dir / "PCD_detailed_rules_senior.xlsx", index=False)
    pcd_detail_df.to_csv(out_dir / "PCD_detailed_rules_senior.csv", index=False)
    pcd_detail_paper_df, pcd_overall_paper_df, pcd_by_component_paper_df = build_paper_pcd_tables(pcd_detail_df)
    pcd_detail_paper_df.to_excel(out_dir / "PCD_detailed_rules_paper.xlsx", index=False)
    pcd_overall_paper_df.to_excel(out_dir / "pcd_overall_paper.xlsx", index=False)
    pcd_by_component_paper_df.to_excel(out_dir / "pcd_by_component_paper.xlsx", index=False)
    pcd_overall_senior_df = (
        pcd_detail_df.groupby(["Data set"], as_index=False)
        .agg(compliance_rate=("compliance_rate", "mean"), violation_rate=("violation_rate", "mean"), active_rules=("component", "size"))
        .sort_values(["Data set"])
    )
    pcd_by_component_senior_df = (
        pcd_detail_df.groupby(["Data set", "component"], as_index=False)
        .agg(compliance_rate=("compliance_rate", "mean"), violation_rate=("violation_rate", "mean"), active_rules=("component", "size"))
        .sort_values(["Data set", "component"])
    )
    pcd_overall_senior_df.to_excel(out_dir / "pcd_overall_senior.xlsx", index=False)
    pcd_by_component_senior_df.to_excel(out_dir / "pcd_by_component_senior.xlsx", index=False)

    comparison_df = build_monotonic_metric_comparison(best_payload, pcd_detail_df)
    comparison_df.to_excel(out_dir / "monotonic_metrics_comparison_senior.xlsx", index=False)
    comparison_df.to_csv(out_dir / "monotonic_metrics_comparison_senior.csv", index=False)

    x_df, _ = base_mod.load_data(data_xlsx)
    x_raw_all = x_df.to_numpy(dtype=float)
    temp_values = np.linspace(
        float(best_payload["synthetic_grid"]["temp_min"]),
        float(best_payload["synthetic_grid"]["temp_max"]),
        int(args.pda_grid_size),
    )
    er_values = np.linspace(
        float(best_payload["synthetic_grid"]["er_min"]),
        float(best_payload["synthetic_grid"]["er_max"]),
        int(args.pda_grid_size),
    )
    surfaces = compute_pda_surfaces(
        model=model,
        background_raw=x_raw_all,
        scaler=bundle["scaler"],
        x_cols=bundle["x_cols"],
        temp_values=temp_values,
        er_values=er_values,
        temp_col=base_mod.TEMP_COL,
        er_col=base_mod.ER_COL,
        device=device,
    )

    save_pda_surface_tables(surfaces, out_dir / "pda_surface_data.xlsx", out_dir)
    for comp in ["H2", "CO", "CO2"]:
        plot_single_pda(
            temp_mesh=surfaces["Temperature"],
            er_mesh=surfaces["ER"],
            z_volpct=surfaces[comp],
            component=comp,
            out_png=out_dir / f"pda_contour_{comp}.png",
            out_pdf=out_dir / f"pda_contour_{comp}.pdf",
        )
    plot_three_panel_pda(
        surfaces=surfaces,
        out_png=out_dir / "pda_contours_3panel.png",
        out_pdf=out_dir / "pda_contours_3panel.pdf",
    )

    pda_trends = {
        "H2": assess_surface_trend(surfaces["H2"], temp_sign=+1.0, er_sign=-1.0),
        "CO": assess_surface_trend(surfaces["CO"], temp_sign=+1.0, er_sign=-1.0),
        "CO2": assess_surface_trend(surfaces["CO2"], temp_sign=-1.0, er_sign=+1.0),
    }
    mono_summary = build_monotonic_summary(
        best_payload=best_payload,
        boundary_summary_df=boundary_summary_df,
        pcd_er_df=pcd_er_df,
        pcd_t_df=pcd_t_df,
        pcd_detail_df=pcd_detail_df,
        comparison_df=comparison_df,
        pda_trends=pda_trends,
    )
    (out_dir / "monotonicity_summary_senior.json").write_text(json.dumps(mono_summary, indent=2), encoding="utf-8")
    mono_summary_paper = dict(mono_summary)
    mono_summary_paper["pcd_report_method"] = {
        "style": "paper-style Pm summary based on detailed finite-difference rule compliance",
        "reported_value": "Pm",
        "detail_records": pcd_detail_paper_df.to_dict(orient="records"),
        "overall_table": pcd_overall_paper_df.to_dict(orient="records"),
        "by_component_table": pcd_by_component_paper_df.to_dict(orient="records"),
        "source_detail_file": "PCD_detailed_rules_paper.xlsx",
    }
    (out_dir / "monotonicity_summary_paper.json").write_text(json.dumps(mono_summary_paper, indent=2), encoding="utf-8")
    alias_sources = write_default_pcd_aliases(out_dir)
    write_markdown_summary(
        out_path=out_dir / "constraint_and_monotonic_summary_senior.md",
        boundary_summary=boundary_summary_df,
        pcd_er=pcd_er_df,
        pcd_t=pcd_t_df,
        pda_trends=pda_trends,
        best_payload=best_payload,
    )

    run_info = {
        "best_dir": str(best_dir),
        "data_xlsx": str(data_xlsx),
        "device": str(device),
        "pda_grid_size": int(args.pda_grid_size),
        "background_samples_for_pda": int(len(x_raw_all)),
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print("Done.")
    print("Out dir:", out_dir)
    print("Boundary figure:", out_dir / "boundary_verification_main.png")
    print("PCD ER table:", out_dir / "PCD_against_ER_train_test_senior.xlsx")
    print("PCD T table:", out_dir / "PCD_against_T_train_test_senior.xlsx")
    if alias_sources:
        print("Default PCD alias sources:", alias_sources)
    print("PDA figure:", out_dir / "pda_contours_3panel.png")


if __name__ == "__main__":
    main()
