from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_ORDER = ["N2", "H2", "CO", "CO2", "CH4"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, rmse, mae


def get_components(df: pd.DataFrame) -> list[str]:
    comps = []
    for c in df.columns:
        if c.startswith("true_"):
            name = c.replace("true_", "", 1)
            if f"pred_{name}" in df.columns:
                comps.append(name)
    ordered = [c for c in TARGET_ORDER if c in comps]
    rest = [c for c in comps if c not in ordered]
    return ordered + rest


def plot_panel(df: pd.DataFrame, dataset: str, out_png: Path, out_pdf: Path) -> pd.DataFrame:
    comps = get_components(df)
    if len(comps) < 5:
        raise ValueError(f"Need 5 components, got {comps}")

    comps = comps[:5]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    rows = []

    for i, comp in enumerate(comps):
        ax = axes[i]
        yt = df[f"true_{comp}"].to_numpy(dtype=float)
        yp = df[f"pred_{comp}"].to_numpy(dtype=float)
        r2, rmse, mae = compute_metrics(yt, yp)
        lo = float(min(np.min(yt), np.min(yp)))
        hi = float(max(np.max(yt), np.max(yp)))
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        lo -= pad
        hi += pad

        ax.scatter(yt, yp, s=18, alpha=0.75)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(f"{dataset} - {comp}")
        ax.text(
            0.03,
            0.97,
            f"R2={r2:.4f}\nRMSE={rmse:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        rows.append(
            {
                "dataset": dataset.lower(),
                "component": comp,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
            }
        )

    axes[5].axis("off")
    fig.suptitle(f"Pred vs True ({dataset})", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    best_dir = args.best_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_xlsx = best_dir / "pred_train_best.xlsx"
    test_xlsx = best_dir / "pred_test_best.xlsx"
    if not train_xlsx.exists() or not test_xlsx.exists():
        raise FileNotFoundError("pred_train_best.xlsx or pred_test_best.xlsx not found in best-dir")

    df_train = pd.read_excel(train_xlsx)
    df_test = pd.read_excel(test_xlsx)

    met_train = plot_panel(
        df_train,
        "Train",
        out_dir / "scatter_train_5panel.png",
        out_dir / "scatter_train_5panel.pdf",
    )
    met_test = plot_panel(
        df_test,
        "Test",
        out_dir / "scatter_test_5panel.png",
        out_dir / "scatter_test_5panel.pdf",
    )

    metrics = pd.concat([met_train, met_test], ignore_index=True)
    metrics.to_excel(out_dir / "scatter_metrics.xlsx", index=False)


if __name__ == "__main__":
    main()
