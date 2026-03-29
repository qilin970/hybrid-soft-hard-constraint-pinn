from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_float_list(text: str) -> List[float]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    return [float(x) for x in vals]


def tag_value(v: float) -> str:
    return f"{v:g}".replace(".", "p")


def point_dir(grid_runs_dir: Path, lambda_s: float, lambda_m: float) -> Path:
    return grid_runs_dir / f"ls_{tag_value(lambda_s)}__lm_{tag_value(lambda_m)}"


def run_complete(run_dir: Path) -> bool:
    need = [
        run_dir / "best_model" / "best_model.pth",
        run_dir / "best_model" / "best_config.json",
        run_dir / "eval" / "PCD_detailed_rules_paper.xlsx",
        run_dir / "eval" / "pcd_overall_paper.xlsx",
    ]
    return all(p.exists() for p in need)


def parse_test_pcd(eval_dir: Path) -> Dict[str, float | int | str]:
    detail = pd.read_excel(eval_dir / "PCD_detailed_rules_paper.xlsx")
    overall = pd.read_excel(eval_dir / "pcd_overall_paper.xlsx")
    dtest = detail[detail["split"].astype(str).str.contains("Testing", case=False)].copy()
    otest = overall[overall["split"].astype(str).str.contains("Testing", case=False)].iloc[0]
    min_row = dtest.loc[dtest["Pm"].idxmin()]
    return {
        "overall_pcd_test": float(otest["Pm"]),
        "min_rule_pcd_test": float(dtest["Pm"].min()),
        "num_rule_pcd_eq_1_test": int((dtest["Pm"] >= 1.0 - 1e-12).sum()),
        "hardest_rule_test": str(min_row["rule"]),
    }


def extract_row_from_completed_run(run_dir: Path) -> Dict[str, float | int | str]:
    cfg = json.loads((run_dir / "best_model" / "best_config.json").read_text(encoding="utf-8"))
    rec = cfg.get("selected_record", {})
    pcd = parse_test_pcd(run_dir / "eval")
    ts = datetime.fromtimestamp((run_dir / "best_model" / "best_config.json").stat().st_mtime).isoformat(timespec="seconds")
    return {
        "lambda_m": float(rec.get("lambda_m")),
        "lambda_s": float(rec.get("lambda_s")),
        "E_R_train_final": float(rec.get("E_R_train_final")),
        "E_M_train_final": float(rec.get("E_M_train_final")),
        "overall_pcd_test": float(pcd["overall_pcd_test"]),
        "min_rule_pcd_test": float(pcd["min_rule_pcd_test"]),
        "num_rule_pcd_eq_1_test": int(pcd["num_rule_pcd_eq_1_test"]),
        "R2_mean_test": float(rec.get("R2_mean_test")),
        "RMSE_mean_test": float(rec.get("RMSE_mean_test")),
        "hardest_rule_test": str(pcd["hardest_rule_test"]),
        "run_dir": str(run_dir),
        "status": "reused",
        "timestamp": ts,
    }


def run_one_point(
    *,
    run_dir: Path,
    lambda_m: float,
    lambda_s: float,
    alpha: float,
    cfg: Dict[str, float | int | str],
    bundle: Dict,
    tryfix_mod,
    base_mod,
    joint_mod,
    device: torch.device,
    python_exe: Path,
    eval_py: Path,
    runner_py: Path,
    base_py: Path,
    data_xlsx: Path,
    split_meta: Dict,
    syn_meta: Dict,
    val_ratio_in_train: float,
    val_seed: int,
) -> Dict[str, float | int | str]:
    run_dir.mkdir(parents=True, exist_ok=False)
    best_dir = run_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=False)
    eval_dir = run_dir / "eval"

    rules = joint_mod.build_mu_rule_spec(base_mod, bundle["x_cols"])
    model = tryfix_mod.make_model(base_mod, cfg, bundle, device)
    train_loader, val_loader, syn_loader = tryfix_mod.make_loaders(bundle, int(cfg["batch_size"]), seed=20260328)

    hist1, val1 = tryfix_mod.train_stage1_part_only(
        model=model,
        base_mod=base_mod,
        joint_mod=joint_mod,
        rules=rules,
        train_loader=train_loader,
        val_loader=val_loader,
        syn_loader=syn_loader,
        x_syn_scaled=bundle["x_syn_scaled"],
        epochs=int(cfg["epochs_part"]),
        lr_part=float(cfg["lr_part"]),
        lambda_s=float(lambda_s),
        lambda_m=float(lambda_m),
        patience=int(cfg["patience"]),
        device=device,
    )
    hist2, val2 = tryfix_mod.train_stage2_res_only(
        model=model,
        base_mod=base_mod,
        joint_mod=joint_mod,
        rules=rules,
        train_loader=train_loader,
        val_loader=val_loader,
        syn_loader=syn_loader,
        x_syn_scaled=bundle["x_syn_scaled"],
        epochs=int(cfg["epochs_res"]),
        lr_res=float(cfg["lr_res"]),
        alpha=float(alpha),
        lambda_s=float(lambda_s),
        lambda_m=float(lambda_m),
        patience=int(cfg["patience"]),
        device=device,
    )

    torch.save(model.state_dict(), best_dir / "best_model.pth")
    with pd.ExcelWriter(best_dir / "training_histories.xlsx") as writer:
        hist1.to_excel(writer, sheet_name="stage1_part", index=False)
        hist2.to_excel(writer, sheet_name="stage2_res", index=False)

    yhat_train = base_mod.predict(model, bundle["x_tr_full"], device)
    yhat_test = base_mod.predict(model, bundle["x_test"], device)
    pred_train = pd.DataFrame({f"true_{c}": bundle["y_tr_full"][:, i] for i, c in enumerate(joint_mod.OUTPUT_NAMES)})
    pred_test = pd.DataFrame({f"true_{c}": bundle["y_test"][:, i] for i, c in enumerate(joint_mod.OUTPUT_NAMES)})
    for i, c in enumerate(joint_mod.OUTPUT_NAMES):
        pred_train[f"pred_{c}"] = yhat_train[:, i]
        pred_test[f"pred_{c}"] = yhat_test[:, i]
    pred_train.to_excel(best_dir / "pred_train_best.xlsx", index=False)
    pred_test.to_excel(best_dir / "pred_test_best.xlsx", index=False)

    met_train = base_mod.eval_metrics(bundle["y_tr_full"], yhat_train)
    met_test = base_mod.eval_metrics(bundle["y_test"], yhat_test)
    er_train_final = float(met_train["E_R"])
    em_train_final = float(
        joint_mod.calc_em_diff_on_synthetic_branch(
            model=model,
            x_syn_scaled=bundle["x_syn_scaled"],
            rules=rules,
            device=device,
            branch_kind="full",
            batch_size=2048,
        )
    )

    payload = {
        "best_cfg_id": 1,
        "run_cfg": {
            "hidden_dim": int(cfg["hidden_dim"]),
            "activation": str(cfg["activation"]),
            "alpha": float(alpha),
            "lr_part": float(cfg["lr_part"]),
            "lr_res": float(cfg["lr_res"]),
            "lr_joint": float(cfg["lr_joint"]),
            "batch_size": int(cfg["batch_size"]),
            "patience": int(cfg["patience"]),
            "epochs_part": int(cfg["epochs_part"]),
            "epochs_res": int(cfg["epochs_res"]),
            "epochs_joint": int(cfg["epochs_joint"]),
            "mono_delta_scaled": float(cfg["mono_delta_scaled"]),
        },
        "selected_record": {
            "lambda_m": float(lambda_m),
            "lambda_s": float(lambda_s),
            "alpha": float(alpha),
            "E_R_train_final": float(er_train_final),
            "E_M_train_final": float(em_train_final),
            "R2_mean_test": float(met_test["R2_mean"]),
            "RMSE_mean_test": float(met_test["RMSE_mean"]),
            "val_stage1_best": float(val1),
            "val_stage2_best": float(val2),
        },
        "split": {
            "rand_label_mat": str(split_meta["rand_label_mat"]),
            "rand_row": int(split_meta["rand_row"]),
            "train_count": int(split_meta["train_count"]),
            "test_count": int(split_meta["test_count"]),
        },
        "synthetic_grid": {
            "temp_min": float(syn_meta["temp_min"]),
            "temp_max": float(syn_meta["temp_max"]),
            "temp_points": int(syn_meta["temp_points"]),
            "er_min": float(syn_meta["er_min"]),
            "er_max": float(syn_meta["er_max"]),
            "er_points": int(syn_meta["er_points"]),
        },
        "method": "confirmed_sensitivity_analysis",
        "loss_train": "stage1: E_R_part + lambda_m*E_M_part + lambda_s*E_S; stage2: alpha*E_R_full + lambda_m*E_M_full + lambda_s*E_S",
    }
    (best_dir / "best_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "data": str(data_xlsx),
                "rand_label_mat": str(split_meta["rand_label_mat"]),
                "rand_row": int(split_meta["rand_row"]),
                "train_count": int(split_meta["train_count"]),
                "test_count": int(split_meta["test_count"]),
                "val_ratio_in_train": float(val_ratio_in_train),
                "val_seed": int(val_seed),
                "device": "cuda",
                "method": "confirmed_sensitivity_analysis",
                "lambda_m": float(lambda_m),
                "lambda_s": float(lambda_s),
                "alpha": float(alpha),
                "lr_part": float(cfg["lr_part"]),
                "lr_res": float(cfg["lr_res"]),
                "epochs_part": int(cfg["epochs_part"]),
                "epochs_res": int(cfg["epochs_res"]),
                "batch_size": int(cfg["batch_size"]),
                "patience": int(cfg["patience"]),
                "mono_delta_scaled": float(cfg["mono_delta_scaled"]),
            }
        ]
    ).to_excel(run_dir / "run_config.xlsx", index=False)

    cmd = [
        str(python_exe),
        str(eval_py),
        "--best-dir",
        str(best_dir),
        "--out-dir",
        str(eval_dir),
        "--runner-py",
        str(runner_py),
        "--base-py",
        str(base_py),
        "--data",
        str(data_xlsx),
        "--device",
        "cuda",
    ]
    subprocess.run(cmd, check=True)
    pcd = parse_test_pcd(eval_dir)
    now_ts = datetime.now().isoformat(timespec="seconds")

    return {
        "lambda_m": float(lambda_m),
        "lambda_s": float(lambda_s),
        "E_R_train_final": float(er_train_final),
        "E_M_train_final": float(em_train_final),
        "overall_pcd_test": float(pcd["overall_pcd_test"]),
        "min_rule_pcd_test": float(pcd["min_rule_pcd_test"]),
        "num_rule_pcd_eq_1_test": int(pcd["num_rule_pcd_eq_1_test"]),
        "R2_mean_test": float(met_test["R2_mean"]),
        "RMSE_mean_test": float(met_test["RMSE_mean"]),
        "hardest_rule_test": str(pcd["hardest_rule_test"]),
        "run_dir": str(run_dir),
        "status": "completed",
        "timestamp": now_ts,
    }


def plot_multils(
    *,
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
    log_if_small: bool = False,
) -> bool:
    if df.empty:
        return False
    ls_values = sorted(df["lambda_s"].unique().tolist())
    use_log = False
    if log_if_small:
        y = df[y_col].to_numpy(dtype=float)
        y_pos = y[np.isfinite(y) & (y > 0)]
        if y_pos.size >= 2:
            ratio = float(np.max(y_pos) / np.min(y_pos))
            if ratio >= 50:
                use_log = True

    plt.figure(figsize=(9, 6))
    for ls in ls_values:
        sub = df[df["lambda_s"] == ls].sort_values("lambda_m")
        plt.plot(
            sub["lambda_m"].to_numpy(dtype=float),
            sub[y_col].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            label=f"lambda_s={ls:g}",
        )
    if use_log:
        plt.yscale("log")
        title = f"{title} (log y)"
    plt.xlabel("lambda_m")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    return use_log


def normalize_rows(rows: List[Dict[str, float | int | str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # keep latest by timestamp for same point
    df["__key"] = df.apply(lambda r: f"{float(r['lambda_s']):.12g}|{float(r['lambda_m']):.12g}", axis=1)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="__key", keep="last").drop(columns="__key")
    return df


def save_snapshot(
    *,
    summary_dir: Path,
    rows: List[Dict[str, float | int | str]],
    target_pairs: List[Tuple[float, float]],
):
    df = normalize_rows(rows)
    target_set = {(float(ls), float(lm)) for ls, lm in target_pairs}
    if not df.empty:
        df = df[df.apply(lambda r: (float(r["lambda_s"]), float(r["lambda_m"])) in target_set, axis=1)]
        df = df.sort_values(["lambda_s", "lambda_m"]).reset_index(drop=True)
    partial_path = summary_dir / "confirmed_sensitivity_grid_partial.xlsx"
    df.to_excel(partial_path, index=False)

    expected = len(target_pairs)
    done = len(df)
    remaining = expected - done

    # sort tables on current available points
    if not df.empty:
        df.sort_values(["min_rule_pcd_test", "overall_pcd_test", "R2_mean_test", "RMSE_mean_test"], ascending=[False, False, False, True]).to_excel(
            summary_dir / "sorted_by_pcd.xlsx", index=False
        )
        df.sort_values(["R2_mean_test", "RMSE_mean_test", "min_rule_pcd_test"], ascending=[False, True, False]).to_excel(
            summary_dir / "sorted_by_r2.xlsx", index=False
        )

    # plots
    log_mono = plot_multils(
        df=df,
        y_col="E_M_train_final",
        y_label="Final training monotonic loss",
        title="Effect of Hyperparameters on Monotonicity Loss",
        out_png=summary_dir / "monotonic_loss_final_vs_lambda_m_multils.png",
        out_pdf=summary_dir / "monotonic_loss_final_vs_lambda_m_multils.pdf",
        log_if_small=True,
    )
    plot_multils(
        df=df,
        y_col="E_R_train_final",
        y_label="Final training regression loss",
        title="Effect of Hyperparameters on Regression Loss",
        out_png=summary_dir / "regression_loss_final_vs_lambda_m_multils.png",
        out_pdf=summary_dir / "regression_loss_final_vs_lambda_m_multils.pdf",
        log_if_small=False,
    )
    plot_multils(
        df=df,
        y_col="min_rule_pcd_test",
        y_label="min_rule_pcd_test",
        title="Effect of Hyperparameters on Minimum Rule-level PCD",
        out_png=summary_dir / "pcd_final_vs_lambda_m_multils.png",
        out_pdf=summary_dir / "pcd_final_vs_lambda_m_multils.pdf",
        log_if_small=False,
    )
    plot_multils(
        df=df,
        y_col="R2_mean_test",
        y_label="R2_mean_test",
        title="Effect of Hyperparameters on R²",
        out_png=summary_dir / "r2_final_vs_lambda_m_multils.png",
        out_pdf=summary_dir / "r2_final_vs_lambda_m_multils.pdf",
        log_if_small=False,
    )

    pending = sorted(list(target_set - {(float(r["lambda_s"]), float(r["lambda_m"])) for _, r in df.iterrows()}))
    ptxt = [
        f"updated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"completed_points: {done}",
        f"expected_points: {expected}",
        f"remaining_points: {remaining}",
        f"log_y_used_for_monotonic_plot: {log_mono}",
        "status: running" if remaining > 0 else "status: completed",
        "pending_points:",
    ] + [f"- lambda_s={ls:g}, lambda_m={lm:g}" for ls, lm in pending]
    (summary_dir / "progress_status.txt").write_text("\n".join(ptxt), encoding="utf-8")

    stxt = [
        f"updated_at: {datetime.now().isoformat(timespec='seconds')}",
        "method: same lambda_m/lambda_s in stage1 and stage2",
        f"completed_points: {done}/{expected}",
        "",
    ]
    if not df.empty:
        stxt.append("E_M_train_final trend by lambda_s (available points):")
        for ls in sorted(df["lambda_s"].unique().tolist()):
            sub = df[df["lambda_s"] == ls].sort_values("lambda_m")
            vals = ", ".join([f"{r.lambda_m:g}:{r.E_M_train_final:.3e}" for r in sub.itertuples()])
            stxt.append(f"- lambda_s={ls:g}: {vals}")
        stxt.append("")
        stxt.append("E_R_train_final trend by lambda_s (available points):")
        for ls in sorted(df["lambda_s"].unique().tolist()):
            sub = df[df["lambda_s"] == ls].sort_values("lambda_m")
            vals = ", ".join([f"{r.lambda_m:g}:{r.E_R_train_final:.6f}" for r in sub.itertuples()])
            stxt.append(f"- lambda_s={ls:g}: {vals}")
    (summary_dir / "summary.txt").write_text("\n".join(stxt), encoding="utf-8")

    if remaining == 0:
        df.to_excel(summary_dir / "confirmed_sensitivity_grid.xlsx", index=False)


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_results_root = repo_root / "results"

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-root", type=Path, default=default_results_root / "confirmed_sensitivity")
    parser.add_argument("--base-dir", type=Path, default=default_results_root / "final_main")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lambda-m-values", type=str, default="0,0.01,0.1,1,10,50,100,500,1000")
    parser.add_argument("--lambda-s-values", type=str, default="0,0.001,0.01,0.1")
    args = parser.parse_args()

    if args.device.lower() != "cuda":
        raise ValueError("This experiment requires --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable. Abort (no CPU fallback).")
    device = torch.device("cuda")

    lambda_m_values = parse_float_list(args.lambda_m_values)
    lambda_s_values = parse_float_list(args.lambda_s_values)
    target_pairs = list(itertools.product(lambda_s_values, lambda_m_values))

    result_root = default_results_root
    python_exe = Path(sys.executable)

    tryfix_py = script_dir / "train_two_stage_correction.py"
    runner_py = script_dir / "fixedsplit_utils.py"
    base_py = script_dir / "core_model.py"
    joint_py = script_dir / "joint_physics_branch.py"
    eval_py = script_dir / "evaluate_constraints_and_monotonicity.py"

    tryfix = load_module(tryfix_py, "tryfix_confirmed")
    runner = load_module(runner_py, "runner_confirmed")
    base_mod = load_module(base_py, "base_confirmed")
    joint_mod = load_module(joint_py, "joint_confirmed")

    base_cfg_path = args.base_dir / "best_model" / "best_config.json"
    run_cfg_path = args.base_dir / "run_config.xlsx"
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"base best_config not found: {base_cfg_path}")
    base_payload = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    run_cfg = pd.read_excel(run_cfg_path).iloc[0].to_dict() if run_cfg_path.exists() else {}

    data_xlsx = repo_root / "data" / "experiment_data.xlsx"
    rand_label_mat = Path(str(base_payload["split"]["rand_label_mat"]))
    if not rand_label_mat.exists():
        rand_label_mat = repo_root / "data" / "Rand_lable.mat"

    split_meta = {
        "rand_label_mat": rand_label_mat,
        "rand_row": int(base_payload["split"]["rand_row"]),
        "train_count": int(base_payload["split"]["train_count"]),
        "test_count": int(base_payload["split"]["test_count"]),
    }
    syn_meta = {
        "temp_min": float(base_payload["synthetic_grid"]["temp_min"]),
        "temp_max": float(base_payload["synthetic_grid"]["temp_max"]),
        "temp_points": int(base_payload["synthetic_grid"]["temp_points"]),
        "er_min": float(base_payload["synthetic_grid"]["er_min"]),
        "er_max": float(base_payload["synthetic_grid"]["er_max"]),
        "er_points": int(base_payload["synthetic_grid"]["er_points"]),
    }
    val_ratio_in_train = float(run_cfg.get("val_ratio_in_train", 0.1))
    val_seed = int(run_cfg.get("val_seed", 42))

    bundle = runner.prepare_bundle_fixedsplit(
        base_mod=base_mod,
        data_xlsx=data_xlsx,
        rand_label_mat=rand_label_mat,
        rand_row=int(split_meta["rand_row"]),
        train_count=int(split_meta["train_count"]),
        val_ratio_in_train=val_ratio_in_train,
        val_seed=val_seed,
        syn_temp_min=float(syn_meta["temp_min"]),
        syn_temp_max=float(syn_meta["temp_max"]),
        syn_temp_points=int(syn_meta["temp_points"]),
        syn_er_min=float(syn_meta["er_min"]),
        syn_er_max=float(syn_meta["er_max"]),
        syn_er_points=int(syn_meta["er_points"]),
    )

    cfg = {
        "hidden_dim": 384,
        "activation": "relu",
        "alpha": 0.05,
        "lr_part": 0.00075,
        "lr_res": 0.001,
        "lr_joint": 0.0002,
        "batch_size": 24,
        "patience": 120,
        "epochs_part": int(run_cfg.get("epochs_part", 700)),
        "epochs_res": 600,
        "epochs_joint": int(run_cfg.get("epochs_joint", 480)),
        "mono_delta_scaled": 0.1,
    }
    alpha = 0.05

    run_root = args.run_root if args.run_root is not None else (args.out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    grid_runs_dir = run_root / "grid_runs"
    summary_dir = run_root / "summary"
    if args.resume and run_root.exists():
        grid_runs_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
    else:
        grid_runs_dir.mkdir(parents=True, exist_ok=False)
        summary_dir.mkdir(parents=True, exist_ok=False)

    rows: List[Dict[str, float | int | str]] = []
    total = len(target_pairs)
    done_idx = 0
    for ls, lm in target_pairs:
        done_idx += 1
        one_run_dir = point_dir(grid_runs_dir, ls, lm)
        if args.resume and run_complete(one_run_dir):
            row = extract_row_from_completed_run(one_run_dir)
            rows.append(row)
            save_snapshot(summary_dir=summary_dir, rows=rows, target_pairs=target_pairs)
            print(f"[{done_idx}/{total}] SKIP completed lambda_s={ls:g}, lambda_m={lm:g}")
            continue

        if args.resume and one_run_dir.exists() and not run_complete(one_run_dir):
            shutil.rmtree(one_run_dir, ignore_errors=True)

        print(f"[{done_idx}/{total}] RUN lambda_s={ls:g}, lambda_m={lm:g}")
        row = run_one_point(
            run_dir=one_run_dir,
            lambda_m=float(lm),
            lambda_s=float(ls),
            alpha=float(alpha),
            cfg=cfg,
            bundle=bundle,
            tryfix_mod=tryfix,
            base_mod=base_mod,
            joint_mod=joint_mod,
            device=device,
            python_exe=python_exe,
            eval_py=eval_py,
            runner_py=runner_py,
            base_py=base_py,
            data_xlsx=data_xlsx,
            split_meta=split_meta,
            syn_meta=syn_meta,
            val_ratio_in_train=val_ratio_in_train,
            val_seed=val_seed,
        )
        rows.append(row)
        save_snapshot(summary_dir=summary_dir, rows=rows, target_pairs=target_pairs)
        print(
            f"[{done_idx}/{total}] DONE min_pcd={row['min_rule_pcd_test']:.6f}, "
            f"R2={row['R2_mean_test']:.6f}, ER_final={row['E_R_train_final']:.8f}, EM_final={row['E_M_train_final']:.8e}"
        )

    save_snapshot(summary_dir=summary_dir, rows=rows, target_pairs=target_pairs)
    print(f"run_root={run_root}")
    print(f"summary_dir={summary_dir}")
    print(f"total_target_points={total}")


if __name__ == "__main__":
    main()
