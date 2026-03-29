"""Two-stage correction experiment (GPU-only).

Stage 1 (part net only):
    y_part = softmax(z_part)
    L_part = E_R_part + lambda_m * E_M_part + lambda_s * E_S

Stage 2 (res net only, part frozen):
    y_full = softmax(z_part + D(x) * z_res)
    L_res = alpha * E_R_full + lambda_m * E_M_full + lambda_s * E_S

No architecture/loss-definition change. Uses existing eval script unchanged.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


def load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_loaders(bundle, batch_size: int, seed: int):
    g = torch.Generator()
    g.manual_seed(int(seed))
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(bundle["x_tr_sub"], dtype=torch.float32),
            torch.tensor(bundle["y_tr_sub"], dtype=torch.float32),
        ),
        batch_size=int(batch_size),
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(bundle["x_val"], dtype=torch.float32),
            torch.tensor(bundle["y_val"], dtype=torch.float32),
        ),
        batch_size=max(int(batch_size), 64),
        shuffle=False,
    )
    syn_loader = DataLoader(
        TensorDataset(torch.tensor(bundle["x_syn_scaled"], dtype=torch.float32)),
        batch_size=max(2048, int(batch_size) * 32),
        shuffle=True,
        generator=g,
        drop_last=False,
    )
    return train_loader, val_loader, syn_loader


def make_model(base_mod, cfg, bundle, device: torch.device):
    return base_mod.HardBoundaryANN(
        in_dim=len(bundle["x_cols"]),
        hidden_dim=int(cfg["hidden_dim"]),
        out_dim=5,
        c=bundle["c"],
        s=bundle["s"],
        activation=str(cfg["activation"]),
        distance_feature_idx=bundle["distance_feature_idx"],
    ).to(device)


def train_stage1_part_only(
    *,
    model,
    base_mod,
    joint_mod,
    rules,
    train_loader,
    val_loader,
    syn_loader,
    x_syn_scaled: np.ndarray,
    epochs: int,
    lr_part: float,
    lambda_s: float,
    lambda_m: float,
    patience: int,
    device: torch.device,
):
    for p in model.part_net.parameters():
        p.requires_grad = True
    for p in model.res_net.parameters():
        p.requires_grad = False

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(lr_part), weight_decay=0.0)
    best_val = float("inf")
    best_state = None
    wait = 0
    hist = []

    for ep in range(1, int(epochs) + 1):
        model.train()
        tr_er = tr_em = tr_es = tr_total = 0.0
        n_tr = 0
        real_iter = iter(train_loader)
        for syn_batch in syn_loader:
            try:
                xb, yb = next(real_iter)
            except StopIteration:
                real_iter = iter(train_loader)
                xb, yb = next(real_iter)
            xb = xb.to(device)
            yb = yb.to(device)
            x_syn = syn_batch[0].to(device)

            opt.zero_grad()
            y_part = joint_mod.forward_particular_only(model, xb)
            er = base_mod.regression_loss_er(y_part, yb)
            es = base_mod.structure_loss_l2_normalized(model, trainable_only=True)
            mu_batch = joint_mod.build_mu_matrix_for_synthetic(int(x_syn.size(0)), rules, x_syn.device)
            em = joint_mod.monotonic_loss_diff_branch(
                model=model,
                x_mono=x_syn,
                rules=rules,
                branch_kind="part",
                mu_matrix=mu_batch,
                create_graph=True,
            )
            loss = er + float(lambda_m) * em + float(lambda_s) * es
            loss.backward()
            opt.step()

            bs = int(xb.size(0))
            tr_er += float(er.item()) * bs
            tr_em += float(em.item()) * bs
            tr_es += float(es.item()) * bs
            tr_total += float(loss.item()) * bs
            n_tr += bs

        model.eval()
        va_er = va_total = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yp = joint_mod.forward_particular_only(model, xb)
                er = base_mod.regression_loss_er(yp, yb)
                bs = int(xb.size(0))
                va_er += float(er.item()) * bs
                n_va += bs
        em_val = float(
            joint_mod.calc_em_diff_on_synthetic_branch(
                model=model,
                x_syn_scaled=x_syn_scaled,
                rules=rules,
                device=device,
                branch_kind="part",
                batch_size=max(2048, 2048),
            )
        )
        with torch.no_grad():
            es_val = float(base_mod.structure_loss_l2_normalized(model, trainable_only=True).item())
        va_total = (va_er / max(n_va, 1)) + float(lambda_m) * em_val + float(lambda_s) * es_val

        rec = {
            "epoch": ep,
            "stage": "stage1_part_only",
            "lr": float(opt.param_groups[0]["lr"]),
            "train_E_R_part": tr_er / max(n_tr, 1),
            "train_E_M_part": tr_em / max(n_tr, 1),
            "train_E_S": tr_es / max(n_tr, 1),
            "train_total": tr_total / max(n_tr, 1),
            "val_E_R_part": va_er / max(n_va, 1),
            "val_E_M_part": em_val,
            "val_E_S": es_val,
            "val_total": va_total,
        }
        hist.append(rec)

        if va_total < best_val - 1e-12:
            best_val = va_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(hist), float(best_val)


def train_stage2_res_only(
    *,
    model,
    base_mod,
    joint_mod,
    rules,
    train_loader,
    val_loader,
    syn_loader,
    x_syn_scaled: np.ndarray,
    epochs: int,
    lr_res: float,
    alpha: float,
    lambda_s: float,
    lambda_m: float,
    patience: int,
    device: torch.device,
):
    for p in model.part_net.parameters():
        p.requires_grad = False
    for p in model.res_net.parameters():
        p.requires_grad = True

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(lr_res), weight_decay=0.0)
    best_val = float("inf")
    best_state = None
    wait = 0
    hist = []

    for ep in range(1, int(epochs) + 1):
        model.train()
        tr_er = tr_em = tr_es = tr_total = 0.0
        n_tr = 0
        real_iter = iter(train_loader)
        for syn_batch in syn_loader:
            try:
                xb, yb = next(real_iter)
            except StopIteration:
                real_iter = iter(train_loader)
                xb, yb = next(real_iter)
            xb = xb.to(device)
            yb = yb.to(device)
            x_syn = syn_batch[0].to(device)

            opt.zero_grad()
            y_full = joint_mod.forward_real_full(model, xb)
            er = base_mod.regression_loss_er(y_full, yb)
            es = base_mod.structure_loss_l2_normalized(model, trainable_only=True)
            mu_batch = joint_mod.build_mu_matrix_for_synthetic(int(x_syn.size(0)), rules, x_syn.device)
            em = joint_mod.monotonic_loss_diff_branch(
                model=model,
                x_mono=x_syn,
                rules=rules,
                branch_kind="full",
                mu_matrix=mu_batch,
                create_graph=True,
            )
            loss = float(alpha) * er + float(lambda_m) * em + float(lambda_s) * es
            loss.backward()
            opt.step()

            bs = int(xb.size(0))
            tr_er += float(er.item()) * bs
            tr_em += float(em.item()) * bs
            tr_es += float(es.item()) * bs
            tr_total += float(loss.item()) * bs
            n_tr += bs

        model.eval()
        va_er = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yp = joint_mod.forward_real_full(model, xb)
                er = base_mod.regression_loss_er(yp, yb)
                bs = int(xb.size(0))
                va_er += float(er.item()) * bs
                n_va += bs
        em_val = float(
            joint_mod.calc_em_diff_on_synthetic_branch(
                model=model,
                x_syn_scaled=x_syn_scaled,
                rules=rules,
                device=device,
                branch_kind="full",
                batch_size=max(2048, 2048),
            )
        )
        with torch.no_grad():
            es_val = float(base_mod.structure_loss_l2_normalized(model, trainable_only=True).item())
        va_total = float(alpha) * (va_er / max(n_va, 1)) + float(lambda_m) * em_val + float(lambda_s) * es_val

        rec = {
            "epoch": ep,
            "stage": "stage2_res_only",
            "lr": float(opt.param_groups[0]["lr"]),
            "train_E_R_full": tr_er / max(n_tr, 1),
            "train_E_M_full": tr_em / max(n_tr, 1),
            "train_E_S": tr_es / max(n_tr, 1),
            "train_total": tr_total / max(n_tr, 1),
            "val_E_R_full": va_er / max(n_va, 1),
            "val_E_M_full": em_val,
            "val_E_S": es_val,
            "val_total": va_total,
        }
        hist.append(rec)

        if va_total < best_val - 1e-12:
            best_val = va_total
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(hist), float(best_val)


def collect_method_metrics(method: str, run_dir: Path) -> Dict[str, object]:
    cfg = json.loads((run_dir / "best_model" / "best_config.json").read_text(encoding="utf-8"))
    rec = cfg.get("selected_record", {})
    detail = pd.read_excel(run_dir / "eval" / "PCD_detailed_rules_paper.xlsx")
    overall = pd.read_excel(run_dir / "eval" / "pcd_overall_paper.xlsx")
    dtest = detail[detail["split"].astype(str).str.contains("Testing", case=False)].copy()
    otest = overall[overall["split"].astype(str).str.contains("Testing", case=False)].iloc[0]
    min_row = dtest.loc[dtest["Pm"].idxmin()]
    er_co2 = float(dtest[dtest["rule"].astype(str) == "ER->CO2"]["Pm"].iloc[0])

    return {
        "method": method,
        "lambda_m": float(rec.get("lambda_m", float("nan"))),
        "lambda_s": float(rec.get("lambda_s", float("nan"))),
        "alpha": float(rec.get("alpha", float("nan"))),
        "lr_part": float(rec.get("lr_part", float("nan"))),
        "lr_res": float(rec.get("lr_res", float("nan"))),
        "R2_mean_test": float(rec.get("R2_mean_test", float("nan"))),
        "RMSE_mean_test": float(rec.get("RMSE_mean_test", float("nan"))),
        "overall_pcd_test": float(otest["Pm"]),
        "min_rule_pcd_test": float(dtest["Pm"].min()),
        "num_rule_pcd_eq_1_test": int((dtest["Pm"] >= 1.0 - 1e-12).sum()),
        "ER->CO2_test_pcd": er_co2,
        "hardest_rule": str(min_row["rule"]),
        "run_dir": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the reviewer-facing two-stage correction main pipeline.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--data", type=Path, default=None, help="Default: <repo>/data/experiment_data.xlsx")
    parser.add_argument("--rand-label-mat", type=Path, default=None, help="Default: <repo>/data/Rand_lable.mat")
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Default: <repo>/results/final_main")
    parser.add_argument("--out-root", type=Path, default=None, help="Default: <repo>/results/two_stage_correction_runs")
    parser.add_argument("--out-dir", type=Path, default=None, help="If set, write outputs exactly to this run dir.")
    parser.add_argument("--overwrite-out-dir", action="store_true", help="Allow removing --out-dir when it already exists.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    results_root = repo_root / "results"

    python_exe = Path(sys.executable)
    runner_py = script_dir / "fixedsplit_utils.py"
    base_py = script_dir / "core_model.py"
    joint_py = script_dir / "joint_physics_branch.py"
    eval_py = script_dir / "evaluate_constraints_and_monotonicity.py"

    baseline_main = args.baseline_dir if args.baseline_dir is not None else (results_root / "final_main")
    baseline_lm100 = results_root / "baseline_lm100"
    out_root = args.out_root if args.out_root is not None else (results_root / "two_stage_correction_runs")
    out_root.mkdir(parents=True, exist_ok=True)
    if args.out_dir is not None:
        run_dir = args.out_dir
        if run_dir.exists():
            if args.overwrite_out_dir:
                shutil.rmtree(run_dir)
            else:
                raise FileExistsError(f"Output directory exists: {run_dir}. Use --overwrite-out-dir to replace it.")
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=False)

    # Fixed hyper-params for this experiment.
    cfg = {
        "hidden_dim": 384,
        "activation": "relu",
        "alpha": 1e-5,
        "lr_part": 0.00075,
        "lr_res": 0.001,
        "lr_joint": 0.0002,  # kept for config compatibility
        "batch_size": 24,
        "patience": 120,
        "epochs_part": 700,
        "epochs_res": 600,
        "epochs_joint": 480,  # kept for config compatibility
        "mono_delta_scaled": 0.1,
    }
    lambda_s = 0.01
    lambda_m = 100.0
    alpha = 0.05
    device_arg = str(args.device).lower()
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(device_arg)

    runner = load_module(runner_py, "runner_tryfix")
    base_mod = load_module(base_py, "base_tryfix")
    joint_mod = load_module(joint_py, "joint_tryfix")

    # Use locally migrated best config as fixed template.
    payload_base = json.loads((baseline_main / "best_model" / "best_config.json").read_text(encoding="utf-8"))
    run_meta_xlsx = baseline_main / "run_config.xlsx"
    if run_meta_xlsx.exists():
        run_meta_base = pd.read_excel(run_meta_xlsx).iloc[0].to_dict()
        val_ratio_in_train = float(run_meta_base["val_ratio_in_train"])
        val_seed = int(run_meta_base["val_seed"])
    else:
        run_meta_base = {}
        val_ratio_in_train = 0.1
        val_seed = 42
    data_xlsx = args.data if args.data is not None else (repo_root / "data" / "experiment_data.xlsx")
    rand_label_local = args.rand_label_mat if args.rand_label_mat is not None else (repo_root / "data" / "Rand_lable.mat")
    rand_label_from_cfg = Path(str(payload_base["split"]["rand_label_mat"]))
    rand_label_mat = rand_label_from_cfg if rand_label_from_cfg.exists() else rand_label_local

    bundle = runner.prepare_bundle_fixedsplit(
        base_mod=base_mod,
        data_xlsx=data_xlsx,
        rand_label_mat=rand_label_mat,
        rand_row=int(payload_base["split"]["rand_row"]),
        train_count=int(payload_base["split"]["train_count"]),
        val_ratio_in_train=val_ratio_in_train,
        val_seed=val_seed,
        syn_temp_min=float(payload_base["synthetic_grid"]["temp_min"]),
        syn_temp_max=float(payload_base["synthetic_grid"]["temp_max"]),
        syn_temp_points=int(payload_base["synthetic_grid"]["temp_points"]),
        syn_er_min=float(payload_base["synthetic_grid"]["er_min"]),
        syn_er_max=float(payload_base["synthetic_grid"]["er_max"]),
        syn_er_points=int(payload_base["synthetic_grid"]["er_points"]),
    )

    # Write run metadata.
    pd.DataFrame(
        [
            {
                "data": str(data_xlsx),
                "rand_label_mat": str(rand_label_mat),
                "rand_row": int(payload_base["split"]["rand_row"]),
                "train_count": int(payload_base["split"]["train_count"]),
                "test_count": int(payload_base["split"]["test_count"]),
                "val_ratio_in_train": float(val_ratio_in_train),
                "val_seed": int(val_seed),
                "device": device_arg,
                "method": "two_stage_try_fix",
                "stage1_loss": "E_R_part + lambda_m*E_M_part + lambda_s*E_S",
                "stage2_loss": "alpha*E_R_full + lambda_m*E_M_full + lambda_s*E_S",
                "alpha": float(alpha),
                "lambda_s": float(lambda_s),
                "lambda_m": float(lambda_m),
                "hidden_dim": int(cfg["hidden_dim"]),
                "activation": str(cfg["activation"]),
                "lr_part": float(cfg["lr_part"]),
                "lr_res": float(cfg["lr_res"]),
                "batch_size": int(cfg["batch_size"]),
                "patience": int(cfg["patience"]),
                "epochs_part": int(cfg["epochs_part"]),
                "epochs_res": int(cfg["epochs_res"]),
                "mono_delta_scaled": float(cfg["mono_delta_scaled"]),
            }
        ]
    ).to_excel(run_dir / "run_config.xlsx", index=False)

    max_len = max(len(bundle["train_idx"]), len(bundle["test_idx"]))
    train_col = np.full(max_len, np.nan, dtype=float)
    test_col = np.full(max_len, np.nan, dtype=float)
    train_col[: len(bundle["train_idx"])] = bundle["train_idx"].astype(float)
    test_col[: len(bundle["test_idx"])] = bundle["test_idx"].astype(float)
    pd.DataFrame({"train_idx": train_col, "test_idx": test_col}).to_excel(run_dir / "fixed_split_indices.xlsx", index=False)

    model = make_model(base_mod, cfg, bundle, device)
    rules = joint_mod.build_mu_rule_spec(base_mod, bundle["x_cols"])
    train_loader, val_loader, syn_loader = make_loaders(bundle, int(cfg["batch_size"]), seed=20260327)

    hist1, val1 = train_stage1_part_only(
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

    hist2, val2 = train_stage2_res_only(
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

    best_dir = run_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=False)
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
    em_diff_train = float(
        joint_mod.calc_em_diff_on_synthetic_branch(
            model=model,
            x_syn_scaled=bundle["x_syn_scaled"],
            rules=rules,
            device=device,
            branch_kind="full",
            batch_size=max(2048, 2048),
        )
    )
    em_seniorfd_train = float(
        joint_mod.senior_mono_violation_rate(
            runner=runner,
            base_mod=base_mod,
            model=model,
            x_mono=torch.tensor(bundle["x_tr_full"], dtype=torch.float32, device=device),
            rules=rules,
            delta_scaled=float(cfg["mono_delta_scaled"]),
            forward_kind="full",
        ).item()
    )
    em_seniorfd_test = float(
        joint_mod.senior_mono_violation_rate(
            runner=runner,
            base_mod=base_mod,
            model=model,
            x_mono=torch.tensor(bundle["x_test"], dtype=torch.float32, device=device),
            rules=rules,
            delta_scaled=float(cfg["mono_delta_scaled"]),
            forward_kind="full",
        ).item()
    )

    selected = {
        "cfg_id": 1,
        "hidden_dim": int(cfg["hidden_dim"]),
        "activation": str(cfg["activation"]),
        "alpha": float(cfg["alpha"]),
        "lr_part": float(cfg["lr_part"]),
        "lr_res": float(cfg["lr_res"]),
        "lr_joint": float(cfg["lr_joint"]),
        "batch_size": int(cfg["batch_size"]),
        "patience": int(cfg["patience"]),
        "epochs_part": int(cfg["epochs_part"]),
        "epochs_res": int(cfg["epochs_res"]),
        "epochs_joint": int(cfg["epochs_joint"]),
        "mono_delta_scaled": float(cfg["mono_delta_scaled"]),
        "lambda_s": float(lambda_s),
        "lambda_m": float(lambda_m),
        "alpha_stage2_regression_anchor": float(alpha),
        "val_stage1_best": float(val1),
        "val_stage2_best": float(val2),
        "E_R_train": float(met_train["E_R"]),
        "E_R_test": float(met_test["E_R"]),
        "E_M_diff_train": float(em_diff_train),
        "E_M_seniorfd_train": float(em_seniorfd_train),
        "E_M_seniorfd_test": float(em_seniorfd_test),
        "R2_mean_test": float(met_test["R2_mean"]),
        "RMSE_mean_test": float(met_test["RMSE_mean"]),
        "negative_ratio_test": float(met_test["negative_ratio"]),
        "min_component_min_test": float(met_test["min_component_min"]),
        "sum_abs_err_mean_test": float(met_test["sum_abs_err_mean"]),
        "sum_abs_err_max_test": float(met_test["sum_abs_err_max"]),
    }
    selected.update({f"R2_{c}_test": float(met_test[f"R2_{c}"]) for c in joint_mod.OUTPUT_NAMES})
    selected.update({f"RMSE_{c}_test": float(met_test[f"RMSE_{c}"]) for c in joint_mod.OUTPUT_NAMES})

    payload = {
        "best_cfg_id": 1,
        "run_cfg": {
            "hidden_dim": int(cfg["hidden_dim"]),
            "activation": str(cfg["activation"]),
            "alpha": float(cfg["alpha"]),
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
        "selected_record": selected,
        "split": payload_base["split"],
        "synthetic_grid": payload_base["synthetic_grid"],
        "selection_metric": "single_run_try_fix",
        "label_normalization": "row-wise closure normalization in base_mod.load_data()",
        "real_forward": "y_real = softmax(z_part + D(x) * z_res)",
        "synthetic_forward": "y_syn = softmax(z_part + z_res)",
        "loss_train": "stage1: E_R_part + lambda_m*E_M_part + lambda_s*E_S; stage2: alpha*E_R_full + lambda_m*E_M_full + lambda_s*E_S",
        "mono_training_metric": "E_M_diff_train",
        "mono_diff_definition": "E_M = sum_{i,j} ReLU(-mu_ij * d y_hat[out_j] / d x[in_j]) / N_effective(mu_ij!=0)",
        "mono_rule_columns_fixed": payload_base.get("mono_rule_columns_fixed", []),
        "mu_default_row": payload_base.get("mu_default_row", []),
        "mono_report_metric": "senior finite-difference PCD compliance on real train/test sets",
        "mono_training_source": "synthetic_samples_physics_branch",
        "mono_report_source": "real_train_test_sets",
        "optimizer_weight_decay": 0.0,
        "two_stage_correction": {
            "enabled": True,
            "stage1_trainable": "part_net only",
            "stage2_trainable": "res_net only",
            "stage2_alpha": float(alpha),
        },
    }
    (best_dir / "best_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Call official eval script unchanged.
    eval_dir = run_dir / "eval"
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
        str(device_arg),
    ]
    subprocess.run(cmd, check=True)

    # Compare against baselines.
    rows = []
    if baseline_main.exists():
        rows.append(collect_method_metrics("baseline_part_down_joint_same", baseline_main))
    if baseline_lm100.exists():
        rows.append(collect_method_metrics("baseline_lm100", baseline_lm100))
    rows.append(collect_method_metrics("try_correction", run_dir))
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_excel(run_dir / "summary_compare.xlsx", index=False)

    cur = rows[-1]
    b_main = next((r for r in rows if r["method"] == "baseline_part_down_joint_same"), None)
    b_lm = next((r for r in rows if r["method"] == "baseline_lm100"), None)

    lines = [
        f"run_dir: {run_dir}",
        "method: two_stage_try_fix",
        "device: cuda",
        f"alpha: {alpha}",
        f"lambda_s: {lambda_s}",
        f"lambda_m: {lambda_m}",
        f"lr_part: {cfg['lr_part']}",
        f"lr_res: {cfg['lr_res']}",
        "",
    ]
    if b_main is not None:
        lines.extend(
            [
                "vs baseline_part_down_joint_same:",
                f"- R2_mean_test: {b_main['R2_mean_test']} -> {cur['R2_mean_test']}",
                f"- RMSE_mean_test: {b_main['RMSE_mean_test']} -> {cur['RMSE_mean_test']}",
                f"- overall_pcd_test: {b_main['overall_pcd_test']} -> {cur['overall_pcd_test']}",
                f"- min_rule_pcd_test: {b_main['min_rule_pcd_test']} -> {cur['min_rule_pcd_test']}",
                f"- num_rule_pcd_eq_1_test: {b_main['num_rule_pcd_eq_1_test']} -> {cur['num_rule_pcd_eq_1_test']}",
                f"- ER->CO2_test_pcd: {b_main['ER->CO2_test_pcd']} -> {cur['ER->CO2_test_pcd']}",
                f"- hardest_rule: {b_main['hardest_rule']} -> {cur['hardest_rule']}",
                "",
            ]
        )
    if b_lm is not None:
        lines.extend(
            [
                "vs baseline_lm100:",
                f"- R2_mean_test: {b_lm['R2_mean_test']} -> {cur['R2_mean_test']}",
                f"- RMSE_mean_test: {b_lm['RMSE_mean_test']} -> {cur['RMSE_mean_test']}",
                f"- overall_pcd_test: {b_lm['overall_pcd_test']} -> {cur['overall_pcd_test']}",
                f"- min_rule_pcd_test: {b_lm['min_rule_pcd_test']} -> {cur['min_rule_pcd_test']}",
                f"- num_rule_pcd_eq_1_test: {b_lm['num_rule_pcd_eq_1_test']} -> {cur['num_rule_pcd_eq_1_test']}",
                f"- ER->CO2_test_pcd: {b_lm['ER->CO2_test_pcd']} -> {cur['ER->CO2_test_pcd']}",
                f"- hardest_rule: {b_lm['hardest_rule']} -> {cur['hardest_rule']}",
                "",
            ]
        )
    if b_lm is not None and cur["overall_pcd_test"] > b_lm["overall_pcd_test"] and cur["R2_mean_test"] >= 0.90:
        lines.append("结论：尝试修正方案值得继续，PCD 提升且精度仍可接受。")
    elif b_lm is not None and cur["overall_pcd_test"] > b_lm["overall_pcd_test"]:
        lines.append("结论：PCD 有提升，但精度需继续权衡。")
    else:
        lines.append("结论：已完成当前 run 统计（本地基线目录缺失时仅输出当前结果）。")
    (run_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"run_dir={run_dir}")
    print(f"best_dir={best_dir}")
    print(f"eval_dir={eval_dir}")
    print(f"summary_compare={run_dir / 'summary_compare.xlsx'}")
    print(f"summary_txt={run_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
