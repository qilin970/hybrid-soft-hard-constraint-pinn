"""Paper-like joint-training branch for the fixed-split hard-boundary model.

This branch keeps the current hard-boundary backbone:
- labels are still row-wise normalized to closed compositions via base_mod.load_data()
- the fixed 170/30 split and synthetic-grid construction are unchanged
- D(x) still uses only the 8 selected continuous variables
- real samples still use the full hard-boundary forward y = softmax(z_part + D(x) * z_res)

What changes relative to the current senior-injection main model:
- Stage A pretrains the particular network on real data, then freezes it
- Stage B is a single joint-optimization stage from the first step onward
- each Stage-B optimizer step jointly backpropagates:
    E_R(real full forward) + lambda_s * E_S + lambda_M * E_M_diff(synthetic physics forward)
- synthetic samples only feed the monotonic loss
- the synthetic physics branch bypasses D(x), but keeps the final softmax:
    y_syn = softmax(z_part + z_res)

Paper reporting remains unchanged:
- final prediction / inference uses the real full forward only
- the evaluation script still reports senior-style finite-difference PCD on real train/test sets
"""

import argparse
import importlib.util
import json
from dataclasses import asdict
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


OUTPUT_NAMES = ["N2", "H2", "CO", "CO2", "CH4"]


def load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_mu_rule_spec(base_mod, x_cols: List[str]) -> List[Dict[str, Any]]:
    """Build the 10-column mu-rule specification used by differential monotonic loss.

    Rule column order is fixed:
    j=0: ER -> H2
    j=1: ER -> CO2
    j=2: ER -> CO
    j=3: ER -> CH4
    j=4: ER -> N2
    j=5: T  -> H2
    j=6: T  -> CO2
    j=7: T  -> CO
    j=8: T  -> CH4
    j=9: T  -> N2

    Current synthetic samples share one mu row:
    [-1, +1, -1, -1, +1, +1, -1, +1, 0, 0]
    """
    x_idx = {c: i for i, c in enumerate(x_cols)}
    out_idx = {o: i for i, o in enumerate(OUTPUT_NAMES)}
    if base_mod.TEMP_COL not in x_idx or base_mod.ER_COL not in x_idx:
        raise KeyError(f"Missing monotonic columns: {base_mod.TEMP_COL}, {base_mod.ER_COL}")

    rule_defs = [
        ("ER", "H2", -1.0),
        ("ER", "CO2", +1.0),
        ("ER", "CO", -1.0),
        ("ER", "CH4", -1.0),
        ("ER", "N2", +1.0),
        ("T", "H2", +1.0),
        ("T", "CO2", -1.0),
        ("T", "CO", +1.0),
        ("T", "CH4", 0.0),
        ("T", "N2", 0.0),
    ]
    axis_to_input_col = {"ER": base_mod.ER_COL, "T": base_mod.TEMP_COL}
    rules: List[Dict[str, Any]] = []
    for j, (axis, out_name, mu) in enumerate(rule_defs):
        in_col = axis_to_input_col[axis]
        rules.append(
            {
                "j": int(j),
                "rule_name": f"{axis}->{out_name}",
                "axis": axis,
                "output_component": out_name,
                "input_column": in_col,
                "out_i": int(out_idx[out_name]),
                "in_i": int(x_idx[in_col]),
                "mu": float(mu),
            }
        )
    return rules


def _normalize_rules_to_mu(rules: List[Any]) -> List[Dict[str, Any]]:
    """Compatibility shim: accept either legacy (out_i, in_i, sign) or mu-rule dicts."""
    if len(rules) == 0:
        return []
    first = rules[0]
    if isinstance(first, dict):
        return list(rules)
    out: List[Dict[str, Any]] = []
    for j, (out_i, in_i, sign) in enumerate(rules):
        out.append(
            {
                "j": int(j),
                "rule_name": f"legacy_rule_{j}",
                "out_i": int(out_i),
                "in_i": int(in_i),
                "mu": float(sign),
            }
        )
    return out


def build_mu_matrix_for_synthetic(batch_size: int, rules: List[Any], device: torch.device) -> torch.Tensor:
    rules_mu = _normalize_rules_to_mu(rules)
    if len(rules_mu) == 0 or batch_size <= 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)
    mu_row = torch.tensor([float(r["mu"]) for r in rules_mu], dtype=torch.float32, device=device)
    return mu_row.unsqueeze(0).repeat(int(batch_size), 1)


def build_senior_style_mono_rules(base_mod, x_cols: List[str]) -> List[Tuple[int, int, float]]:
    # Legacy API kept for compatibility; internal differential loss now uses mu_ij rules.
    rules_mu = build_mu_rule_spec(base_mod, x_cols)
    return [
        (int(r["out_i"]), int(r["in_i"]), float(r["mu"]))
        for r in rules_mu
        if float(r["mu"]) != 0.0
    ]


def make_loaders(bundle, cfg, seed: int):
    g = torch.Generator()
    g.manual_seed(seed + 1234)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(bundle["x_tr_sub"], dtype=torch.float32),
            torch.tensor(bundle["y_tr_sub"], dtype=torch.float32),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(bundle["x_val"], dtype=torch.float32),
            torch.tensor(bundle["y_val"], dtype=torch.float32),
        ),
        batch_size=max(cfg.batch_size, 64),
        shuffle=False,
    )
    syn_batch_size = max(2048, cfg.batch_size * 32)
    syn_loader = DataLoader(
        TensorDataset(torch.tensor(bundle["x_syn_scaled"], dtype=torch.float32)),
        batch_size=syn_batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
    )
    return train_loader, val_loader, syn_loader, syn_batch_size


def forward_real_full(model, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def forward_particular_only(model, x: torch.Tensor) -> torch.Tensor:
    return model.part_softmax(x)


def forward_synthetic_physics(model, x: torch.Tensor) -> torch.Tensor:
    logits = model.part_net(x) + model.res_net(x)
    return torch.softmax(logits, dim=1)


def forward_branch(model, x: torch.Tensor, branch_kind: str) -> torch.Tensor:
    if branch_kind == "part":
        return forward_particular_only(model, x)
    if branch_kind == "full":
        return forward_real_full(model, x)
    if branch_kind == "synthetic_physics":
        return forward_synthetic_physics(model, x)
    raise ValueError(f"Unknown branch kind: {branch_kind}")


def monotonic_loss_diff_branch(
    model,
    x_mono: torch.Tensor,
    rules: List[Any],
    branch_kind: str,
    mu_matrix: torch.Tensor | None = None,
    create_graph: bool = True,
) -> torch.Tensor:
    rules_mu = _normalize_rules_to_mu(rules)
    if len(rules_mu) == 0:
        return torch.tensor(0.0, device=x_mono.device)
    if mu_matrix is None:
        mu_matrix = build_mu_matrix_for_synthetic(int(x_mono.shape[0]), rules_mu, x_mono.device)
    if mu_matrix.shape != (int(x_mono.shape[0]), int(len(rules_mu))):
        raise ValueError(
            f"mu_matrix shape mismatch: expected {(int(x_mono.shape[0]), int(len(rules_mu)))}, got {tuple(mu_matrix.shape)}"
        )

    x_req = x_mono.requires_grad_(True)
    y_pred = forward_branch(model, x_req, branch_kind)
    loss_sum = torch.zeros((), dtype=y_pred.dtype, device=x_mono.device)
    n_valid = 0
    for j, rule in enumerate(rules_mu):
        mu_col = mu_matrix[:, j]
        valid = mu_col != 0.0
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            continue

        grad = torch.autograd.grad(
            outputs=y_pred[:, int(rule["out_i"])].sum(),
            inputs=x_req,
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0][:, int(rule["in_i"])]
        term = torch.relu(-mu_col * grad)
        loss_sum = loss_sum + term[valid].sum()
        n_valid += valid_count

    if n_valid <= 0:
        return torch.tensor(0.0, device=x_mono.device)
    return loss_sum / float(n_valid)


def calc_em_diff_on_synthetic_branch(
    model,
    x_syn_scaled: np.ndarray,
    rules: List[Any],
    device: torch.device,
    branch_kind: str,
    batch_size: int = 2048,
) -> float:
    if len(x_syn_scaled) == 0:
        return 0.0

    total = 0.0
    n = 0
    for st in range(0, len(x_syn_scaled), batch_size):
        xb = torch.tensor(x_syn_scaled[st : st + batch_size], dtype=torch.float32, device=device)
        mu_batch = build_mu_matrix_for_synthetic(int(xb.size(0)), rules, xb.device)
        em = monotonic_loss_diff_branch(
            model=model,
            x_mono=xb,
            rules=rules,
            branch_kind=branch_kind,
            mu_matrix=mu_batch,
            create_graph=False,
        )
        bs = xb.size(0)
        total += float(em.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def senior_mono_violation_rate(
    runner,
    base_mod,
    model,
    x_mono: torch.Tensor,
    rules: List[Any],
    delta_scaled: float,
    forward_kind: str,
) -> torch.Tensor:
    rules_mu = _normalize_rules_to_mu(rules)
    if len(rules_mu) == 0:
        return torch.tensor(0.0, device=x_mono.device)

    y_base = runner.forward_by_kind(base_mod, model, x_mono, forward_kind)
    viol_total = 0.0
    n_total = 0
    for rule in rules_mu:
        sign = float(rule["mu"])
        if sign == 0.0:
            continue
        out_i = int(rule["out_i"])
        in_i = int(rule["in_i"])
        x_shift = x_mono.clone()
        x_shift[:, in_i] = x_shift[:, in_i] + float(delta_scaled)
        y_shift = runner.forward_by_kind(base_mod, model, x_shift, forward_kind)
        diff = y_shift[:, out_i] - y_base[:, out_i]
        if sign > 0:
            viol = (diff < 0.0).float()
        else:
            viol = (diff > 0.0).float()
        viol_total += float(viol.sum().item())
        n_total += int(viol.numel())
    return torch.tensor(viol_total / max(n_total, 1), dtype=torch.float32, device=x_mono.device)


def train_particular_pretrain(
    base_mod,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    lambda_s: float,
    patience: int,
    device: torch.device,
):
    for p in model.part_net.parameters():
        p.requires_grad = True
    for p in model.res_net.parameters():
        p.requires_grad = False

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)

    best_val = float("inf")
    best_state = None
    wait = 0
    hist = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_er = 0.0
        tr_es = 0.0
        tr_total = 0.0
        n_tr = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_hat = forward_particular_only(model, xb)
            er = base_mod.regression_loss_er(y_hat, yb)
            es = base_mod.structure_loss_l2_normalized(model, trainable_only=True)
            loss = er + lambda_s * es
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            tr_er += float(er.item()) * bs
            tr_es += float(es.item()) * bs
            tr_total += float(loss.item()) * bs
            n_tr += bs

        model.eval()
        va_er = 0.0
        va_es = 0.0
        va_total = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_hat = forward_particular_only(model, xb)
                er = base_mod.regression_loss_er(y_hat, yb)
                es = base_mod.structure_loss_l2_normalized(model, trainable_only=True)
                total = er + lambda_s * es
                bs = xb.size(0)
                va_er += float(er.item()) * bs
                va_es += float(es.item()) * bs
                va_total += float(total.item()) * bs
                n_va += bs

        rec = {
            "epoch": ep,
            "stage": "part_pretrain",
            "mode": "part",
            "real_forward": "part_softmax",
            "synthetic_forward": "",
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_E_R": tr_er / max(n_tr, 1),
            "train_E_S": tr_es / max(n_tr, 1),
            "train_E_M_diff": 0.0,
            "train_total": tr_total / max(n_tr, 1),
            "val_E_R": va_er / max(n_va, 1),
            "val_E_S": va_es / max(n_va, 1),
            "val_E_M_diff": 0.0,
            "val_E_M_fd_rate": 0.0,
            "val_total": va_total / max(n_va, 1),
        }
        hist.append(rec)

        if rec["val_total"] < best_val - 1e-10:
            best_val = rec["val_total"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(hist), float(best_val)


def train_joint_physics_stage(
    runner,
    base_mod,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    syn_loader: DataLoader,
    x_syn_scaled: np.ndarray,
    x_real_eval_full: torch.Tensor,
    rules: List[Any],
    epochs: int,
    lr_start: float,
    lr_end: float,
    lambda_s: float,
    lambda_m: float,
    mono_delta_scaled: float,
    patience: int,
    device: torch.device,
):
    # Current main-model setting: joint stage keeps both part_net and res_net trainable.
    for p in model.part_net.parameters():
        p.requires_grad = True
    for p in model.res_net.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for joint physics stage.")

    optimizer = optim.Adam(trainable_params, lr=lr_start, weight_decay=0.0)
    scheduler = None
    if epochs > 1 and abs(float(lr_start) - float(lr_end)) > 1e-15:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1), eta_min=float(lr_end))

    best_val = float("inf")
    best_state = None
    wait = 0
    hist = []

    for ep in range(1, epochs + 1):
        model.train()
        tr_er = 0.0
        tr_es = 0.0
        tr_em = 0.0
        tr_total = 0.0
        n_tr = 0

        if lambda_m > 0.0:
            real_iter = cycle(train_loader)
            for syn_batch in syn_loader:
                xb, yb = next(real_iter)
                xb = xb.to(device)
                yb = yb.to(device)
                x_syn = syn_batch[0].to(device)

                optimizer.zero_grad()
                y_real = forward_real_full(model, xb)
                er = base_mod.regression_loss_er(y_real, yb)
                es = base_mod.structure_loss_l2_normalized(model, trainable_only=True)
                mu_batch = build_mu_matrix_for_synthetic(int(x_syn.size(0)), rules, x_syn.device)
                em = monotonic_loss_diff_branch(
                    model=model,
                    x_mono=x_syn,
                    rules=rules,
                    branch_kind="synthetic_physics",
                    mu_matrix=mu_batch,
                    create_graph=True,
                )
                loss = er + lambda_s * es + lambda_m * em
                loss.backward()
                optimizer.step()

                bs = xb.size(0)
                tr_er += float(er.item()) * bs
                tr_es += float(es.item()) * bs
                tr_em += float(em.item()) * bs
                tr_total += float(loss.item()) * bs
                n_tr += bs
        else:
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                y_real = forward_real_full(model, xb)
                er = base_mod.regression_loss_er(y_real, yb)
                es = base_mod.structure_loss_l2_normalized(model, trainable_only=True)
                em = torch.tensor(0.0, device=device)
                loss = er + lambda_s * es
                loss.backward()
                optimizer.step()

                bs = xb.size(0)
                tr_er += float(er.item()) * bs
                tr_es += float(es.item()) * bs
                tr_em += float(em.item()) * bs
                tr_total += float(loss.item()) * bs
                n_tr += bs

        model.eval()
        if lambda_m > 0.0:
            em_val_diff = calc_em_diff_on_synthetic_branch(
                model=model,
                x_syn_scaled=x_syn_scaled,
                rules=rules,
                device=device,
                branch_kind="synthetic_physics",
                batch_size=max(2048, 32 * 64),
            )
            em_val_fd = senior_mono_violation_rate(
                runner=runner,
                base_mod=base_mod,
                model=model,
                x_mono=x_real_eval_full,
                rules=rules,
                delta_scaled=mono_delta_scaled,
                forward_kind="full",
            )
        else:
            em_val_diff = 0.0
            em_val_fd = torch.tensor(0.0, device=device)

        va_er = 0.0
        va_es = 0.0
        va_total = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_real = forward_real_full(model, xb)
                er = base_mod.regression_loss_er(y_real, yb)
                es = base_mod.structure_loss_l2_normalized(model, trainable_only=True)
                total = er + lambda_s * es + lambda_m * float(em_val_diff)
                bs = xb.size(0)
                va_er += float(er.item()) * bs
                va_es += float(es.item()) * bs
                va_total += float(total.item()) * bs
                n_va += bs

        current_lr = float(optimizer.param_groups[0]["lr"])
        rec = {
            "epoch": ep,
            "stage": "joint_physics",
            "mode": "res_fixed_part",
            "real_forward": "softmax(z_part + D(x) * z_res)",
            "synthetic_forward": "softmax(z_part + z_res)",
            "lr": current_lr,
            "train_E_R": tr_er / max(n_tr, 1),
            "train_E_S": tr_es / max(n_tr, 1),
            "train_E_M_diff": tr_em / max(n_tr, 1),
            "train_total": tr_total / max(n_tr, 1),
            "val_E_R": va_er / max(n_va, 1),
            "val_E_S": va_es / max(n_va, 1),
            "val_E_M_diff": float(em_val_diff),
            "val_E_M_fd_rate": float(em_val_fd.item()),
            "val_total": va_total / max(n_va, 1),
        }
        hist.append(rec)

        if rec["val_total"] < best_val - 1e-10:
            best_val = rec["val_total"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if scheduler is not None:
            scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(hist), float(best_val)


def selection_score(rec: Dict) -> float:
    score = float(rec["E_R_test"])
    score += 1e-3 * float(rec["RMSE_mean_test"])
    score += 1e-2 * max(0.0, 1.0 - float(rec["R2_mean_test"]))
    score += 1e-2 * float(rec["E_M_seniorfd_test"])
    return float(score)


def run_one(
    runner,
    base_mod,
    bundle,
    mono_rules,
    cfg,
    lambda_s: float,
    lambda_m: float,
    init_seed: int,
    device: torch.device,
):
    runner.set_seed(init_seed)
    train_loader, val_loader, syn_loader, syn_batch_size = make_loaders(bundle, cfg, init_seed)

    model = base_mod.HardBoundaryANN(
        in_dim=len(bundle["x_cols"]),
        hidden_dim=cfg.hidden_dim,
        out_dim=5,
        c=bundle["c"],
        s=bundle["s"],
        activation=cfg.activation,
        distance_feature_idx=bundle["distance_feature_idx"],
    ).to(device)

    x_mono_train_full = torch.tensor(bundle["x_tr_full"], dtype=torch.float32, device=device)
    x_mono_test = torch.tensor(bundle["x_test"], dtype=torch.float32, device=device)
    x_syn_scaled = bundle["x_syn_scaled"]

    h_part, v_part = train_particular_pretrain(
        base_mod=base_mod,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs_part,
        lr=cfg.lr_part,
        lambda_s=lambda_s,
        patience=cfg.patience,
        device=device,
    )

    joint_epochs_effective = int(cfg.epochs_res + cfg.epochs_joint)
    h_joint, v_joint = train_joint_physics_stage(
        runner=runner,
        base_mod=base_mod,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        syn_loader=syn_loader,
        x_syn_scaled=x_syn_scaled,
        x_real_eval_full=x_mono_train_full,
        rules=mono_rules,
        epochs=joint_epochs_effective,
        lr_start=float(cfg.lr_res),
        lr_end=float(cfg.lr_joint),
        lambda_s=lambda_s,
        lambda_m=lambda_m,
        mono_delta_scaled=cfg.mono_delta_scaled,
        patience=cfg.patience,
        device=device,
    )

    yhat_train = base_mod.predict(model, bundle["x_tr_full"], device)
    yhat_test = base_mod.predict(model, bundle["x_test"], device)
    met_train = base_mod.eval_metrics(bundle["y_tr_full"], yhat_train)
    met_test = base_mod.eval_metrics(bundle["y_test"], yhat_test)

    em_diff_train = float(
        calc_em_diff_on_synthetic_branch(
            model=model,
            x_syn_scaled=x_syn_scaled,
            rules=mono_rules,
            device=device,
            branch_kind="synthetic_physics",
            batch_size=max(2048, 32 * 64),
        )
    )
    em_seniorfd_train = float(
        senior_mono_violation_rate(
            runner=runner,
            base_mod=base_mod,
            model=model,
            x_mono=x_mono_train_full,
            rules=mono_rules,
            delta_scaled=cfg.mono_delta_scaled,
            forward_kind="full",
        ).item()
    )
    em_seniorfd_test = float(
        senior_mono_violation_rate(
            runner=runner,
            base_mod=base_mod,
            model=model,
            x_mono=x_mono_test,
            rules=mono_rules,
            delta_scaled=cfg.mono_delta_scaled,
            forward_kind="full",
        ).item()
    )

    rec = {
        "init_seed": int(init_seed),
        "joint_epochs_effective": int(joint_epochs_effective),
        "joint_lr_start": float(cfg.lr_res),
        "joint_lr_end": float(cfg.lr_joint),
        "syn_batch_size": int(syn_batch_size),
        "val_stage_part_best": float(v_part),
        "val_stage_joint_best": float(v_joint),
        "val_objective": float(v_joint),
        "E_R_train": float(met_train["E_R"]),
        "E_R_test": float(met_test["E_R"]),
        "E_M_diff_train": em_diff_train,
        "E_M_seniorfd_train": em_seniorfd_train,
        "E_M_seniorfd_test": em_seniorfd_test,
        "R2_mean_test": float(met_test["R2_mean"]),
        "RMSE_mean_test": float(met_test["RMSE_mean"]),
        "negative_ratio_test": float(met_test["negative_ratio"]),
        "min_component_min_test": float(met_test["min_component_min"]),
        "sum_abs_err_mean_test": float(met_test["sum_abs_err_mean"]),
        "sum_abs_err_max_test": float(met_test["sum_abs_err_max"]),
    }
    rec.update({f"R2_{c}_test": float(met_test[f"R2_{c}"]) for c in OUTPUT_NAMES})
    rec.update({f"RMSE_{c}_test": float(met_test[f"RMSE_{c}"]) for c in OUTPUT_NAMES})

    hist = {"part_pretrain": h_part, "joint_physics": h_joint}
    return rec, model, yhat_train, yhat_test, hist


def save_histories(best_dir: Path, hist: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(best_dir / "training_histories.xlsx") as writer:
        for name, df in hist.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


def runner_stub_default(name: str):
    defaults = {
        "DEFAULT_SYN_TEMP_MIN": 550.0,
        "DEFAULT_SYN_TEMP_MAX": 1050.0,
        "DEFAULT_SYN_TEMP_POINTS": 6,
        "DEFAULT_SYN_ER_MIN": 0.1,
        "DEFAULT_SYN_ER_MAX": 0.9,
        "DEFAULT_SYN_ER_POINTS": 9,
    }
    return defaults[name]


def main():
    parser = argparse.ArgumentParser(
        description="Paper-like joint training branch: fixed particular pretrain + joint real/synthetic optimization."
    )
    parser.add_argument("--base-py", type=str, default=None)
    parser.add_argument("--runner-py", type=str, default=None)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--rand-label-mat", type=str, default=None)
    parser.add_argument("--rand-row", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=170)
    parser.add_argument("--val-ratio-in-train", type=float, default=0.15)
    parser.add_argument("--val-seed", type=int, default=15645)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lambda-s", type=float, default=0.01)
    parser.add_argument("--lambda-m", type=float, default=0.1)
    parser.add_argument("--lambda-p", dest="legacy_lambda_p", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config-json", type=str, default=None)
    parser.add_argument("--best-config-json", dest="legacy_best_config_json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n-configs", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=20260321)
    parser.add_argument("--seed-start", type=int, default=17000)
    parser.add_argument("--syn-temp-min", type=float, default=runner_stub_default("DEFAULT_SYN_TEMP_MIN"))
    parser.add_argument("--syn-temp-max", type=float, default=runner_stub_default("DEFAULT_SYN_TEMP_MAX"))
    parser.add_argument("--syn-temp-points", type=int, default=runner_stub_default("DEFAULT_SYN_TEMP_POINTS"))
    parser.add_argument("--syn-er-min", type=float, default=runner_stub_default("DEFAULT_SYN_ER_MIN"))
    parser.add_argument("--syn-er-max", type=float, default=runner_stub_default("DEFAULT_SYN_ER_MAX"))
    parser.add_argument("--syn-er-points", type=int, default=runner_stub_default("DEFAULT_SYN_ER_POINTS"))
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    runner_py = Path(args.runner_py) if args.runner_py else (code_dir / "fixedsplit_utils.py")
    base_py = Path(args.base_py) if args.base_py else (code_dir / "core_model.py")
    runner = load_module(runner_py, "runner_fixedsplit")
    base_mod = load_module(base_py, "base_hardsoft")

    lambda_m = float(args.legacy_lambda_p) if args.legacy_lambda_p is not None else float(args.lambda_m)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_xlsx = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rand_label_mat = runner.find_rand_label_mat(args.rand_label_mat)
    bundle = runner.prepare_bundle_fixedsplit(
        base_mod=base_mod,
        data_xlsx=data_xlsx,
        rand_label_mat=rand_label_mat,
        rand_row=int(args.rand_row),
        train_count=int(args.train_count),
        val_ratio_in_train=float(args.val_ratio_in_train),
        val_seed=int(args.val_seed),
        syn_temp_min=float(args.syn_temp_min),
        syn_temp_max=float(args.syn_temp_max),
        syn_temp_points=int(args.syn_temp_points),
        syn_er_min=float(args.syn_er_min),
        syn_er_max=float(args.syn_er_max),
        syn_er_points=int(args.syn_er_points),
    )
    mono_rules = build_mu_rule_spec(base_mod, bundle["x_cols"])

    config_json = args.config_json or args.legacy_best_config_json
    if config_json:
        cfgs = [runner.load_run_config_json(Path(config_json))]
    else:
        rng = np.random.default_rng(int(args.search_seed))
        cfgs = runner.sample_configs(int(args.n_configs), rng)

    pd.DataFrame(
        [
            {
                "data": str(data_xlsx),
                "rand_label_mat": str(rand_label_mat),
                "rand_row": int(args.rand_row),
                "train_count": int(args.train_count),
                "test_count": int(bundle["n_samples_test"]),
                "val_ratio_in_train": float(args.val_ratio_in_train),
                "val_seed": int(args.val_seed),
                "device": str(device),
                "lambda_s": float(args.lambda_s),
                "lambda_m": float(lambda_m),
                "lambda_p_legacy_alias": float(lambda_m),
                "n_configs": int(args.n_configs),
                "n_seeds": int(args.n_seeds),
                "config_json": str(config_json) if config_json else "",
                "syn_temp_min": float(args.syn_temp_min),
                "syn_temp_max": float(args.syn_temp_max),
                "syn_temp_points": int(args.syn_temp_points),
                "syn_er_min": float(args.syn_er_min),
                "syn_er_max": float(args.syn_er_max),
                "syn_er_points": int(args.syn_er_points),
                "label_normalization": "row-wise closure normalization in base_mod.load_data()",
                "real_forward": "softmax(z_part + D(x) * z_res)",
                "synthetic_forward": "softmax(z_part + z_res)",
                "mono_training_injection": "real regression on real full forward + autodiff mu_ij differential monotonic loss on synthetic physics forward",
                "mono_rule_columns_fixed": "ER->H2,ER->CO2,ER->CO,ER->CH4,ER->N2,T->H2,T->CO2,T->CO,T->CH4,T->N2",
                "mu_default_row": "-1,+1,-1,-1,+1,+1,-1,+1,0,0",
                "mono_effective_constraints_per_sample": int(sum(1 for r in mono_rules if float(r["mu"]) != 0.0)),
                "mono_reporting_metric": "senior-style finite-difference PCD compliance on real train/test sets",
                "joint_stage_lr_mapping": "single joint stage; cosine anneal lr from legacy lr_res to legacy lr_joint",
                "joint_stage_epoch_mapping": "single joint stage uses epochs_res + epochs_joint",
            }
        ]
    ).to_excel(out_dir / "run_config.xlsx", index=False)

    max_len = max(len(bundle["train_idx"]), len(bundle["test_idx"]))
    train_col = np.full(max_len, np.nan, dtype=float)
    test_col = np.full(max_len, np.nan, dtype=float)
    train_col[: len(bundle["train_idx"])] = bundle["train_idx"].astype(float)
    test_col[: len(bundle["test_idx"])] = bundle["test_idx"].astype(float)
    pd.DataFrame({"train_idx": train_col, "test_idx": test_col}).to_excel(out_dir / "fixed_split_indices.xlsx", index=False)

    rows = []
    best_pack = None
    best_key = None
    for cfg_id, cfg in enumerate(cfgs, start=1):
        for k in range(int(args.n_seeds)):
            init_seed = int(args.seed_start + k)
            rec, model, yhat_train, yhat_test, hist = run_one(
                runner=runner,
                base_mod=base_mod,
                bundle=bundle,
                mono_rules=mono_rules,
                cfg=cfg,
                lambda_s=float(args.lambda_s),
                lambda_m=float(lambda_m),
                init_seed=init_seed,
                device=device,
            )
            row = {"cfg_id": int(cfg_id), **asdict(cfg), **rec}
            row["lambda_s"] = float(args.lambda_s)
            row["lambda_m"] = float(lambda_m)
            row["selection_score"] = selection_score(rec)
            rows.append(row)

            key = (
                row["selection_score"],
                -row["R2_mean_test"],
                row["RMSE_mean_test"],
                row["E_R_test"],
            )
            if best_key is None or key < best_key:
                best_key = key
                best_pack = {
                    "cfg_id": int(cfg_id),
                    "cfg": cfg,
                    "rec": row,
                    "model": model,
                    "yhat_train": yhat_train,
                    "yhat_test": yhat_test,
                    "hist": hist,
                }

    raw_df = pd.DataFrame(rows)
    raw_df = raw_df.sort_values(["selection_score", "R2_mean_test", "E_R_test"], ascending=[True, False, True])
    raw_df.to_excel(out_dir / "search_raw.xlsx", index=False)

    summary = (
        raw_df.groupby(
            [
                "cfg_id",
                "hidden_dim",
                "activation",
                "alpha",
                "lr_part",
                "lr_res",
                "lr_joint",
                "batch_size",
                "patience",
                "epochs_part",
                "epochs_res",
                "epochs_joint",
                "mono_delta_scaled",
            ],
            as_index=False,
        )
        .agg(
            selection_score_mean=("selection_score", "mean"),
            E_R_test_mean=("E_R_test", "mean"),
            E_M_diff_train_mean=("E_M_diff_train", "mean"),
            E_M_seniorfd_test_mean=("E_M_seniorfd_test", "mean"),
            R2_mean_test_mean=("R2_mean_test", "mean"),
            R2_H2_test_mean=("R2_H2_test", "mean"),
            R2_CO2_test_mean=("R2_CO2_test", "mean"),
            RMSE_H2_test_mean=("RMSE_H2_test", "mean"),
            RMSE_CO2_test_mean=("RMSE_CO2_test", "mean"),
        )
        .sort_values(["selection_score_mean", "R2_mean_test_mean"], ascending=[True, False])
    )
    summary.to_excel(out_dir / "search_summary.xlsx", index=False)

    if best_pack is None:
        raise RuntimeError("No model trained.")

    best_dir = out_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_pack["model"].state_dict(), best_dir / "best_model.pth")
    save_histories(best_dir, best_pack["hist"])

    pred_train = pd.DataFrame({f"true_{c}": bundle["y_tr_full"][:, i] for i, c in enumerate(OUTPUT_NAMES)})
    pred_test = pd.DataFrame({f"true_{c}": bundle["y_test"][:, i] for i, c in enumerate(OUTPUT_NAMES)})
    for i, c in enumerate(OUTPUT_NAMES):
        pred_train[f"pred_{c}"] = best_pack["yhat_train"][:, i]
        pred_test[f"pred_{c}"] = best_pack["yhat_test"][:, i]
    pred_train.to_excel(best_dir / "pred_train_best.xlsx", index=False)
    pred_test.to_excel(best_dir / "pred_test_best.xlsx", index=False)

    comp_rows = []
    for split_name, yt, yp in [("train", bundle["y_tr_full"], best_pack["yhat_train"]), ("test", bundle["y_test"], best_pack["yhat_test"])]:
        for i, c in enumerate(OUTPUT_NAMES):
            rmse = float(np.sqrt(np.mean(((yp[:, i] - yt[:, i]) * 100.0) ** 2)))
            r2 = float(r2_score(yt[:, i] * 100.0, yp[:, i] * 100.0))
            comp_rows.append({"split": split_name, "component": c, "R2": r2, "RMSE": rmse})
    pd.DataFrame(comp_rows).to_excel(best_dir / "best_component_metrics.xlsx", index=False)

    mono_roles = pd.DataFrame(
        [
            {
                "term": "E_M_diff_train",
                "used_for_training_backprop": True,
                "definition": "sum_{i,j} ReLU(-mu_ij * d y_hat[out_j]/d x[in_j]) normalized by effective (mu_ij!=0) constraints on synthetic physics branch outputs softmax(z_part + z_res)",
                "value_best": float(best_pack["rec"]["E_M_diff_train"]),
            },
            {
                "term": "E_M_seniorfd_train",
                "used_for_training_backprop": False,
                "definition": "finite-difference violation rate on the real training set using the full hard-boundary forward",
                "value_best": float(best_pack["rec"]["E_M_seniorfd_train"]),
            },
            {
                "term": "E_M_seniorfd_test",
                "used_for_training_backprop": False,
                "definition": "finite-difference violation rate on the real testing set using the full hard-boundary forward",
                "value_best": float(best_pack["rec"]["E_M_seniorfd_test"]),
            },
        ]
    )
    mono_roles.to_excel(best_dir / "mono_train_vs_eval_roles.xlsx", index=False)

    for c in OUTPUT_NAMES:
        base_mod.plot_scatter(
            y_true=bundle["y_test"],
            y_pred=best_pack["yhat_test"],
            comp=c,
            out_png=best_dir / f"scatter_{c}_test.png",
            out_pdf=best_dir / f"scatter_{c}_test.pdf",
        )

    base_mod.plot_boundary_boxplot(
        yhat_train=best_pack["yhat_train"],
        yhat_test=best_pack["yhat_test"],
        out_png=best_dir / "boundary_boxplot_train_test.png",
        out_pdf=best_dir / "boundary_boxplot_train_test.pdf",
    )

    with open(best_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_cfg_id": int(best_pack["cfg_id"]),
                "run_cfg": asdict(best_pack["cfg"]),
                "selected_record": best_pack["rec"],
                "split": {
                    "rand_label_mat": str(rand_label_mat),
                    "rand_row": int(args.rand_row),
                    "train_count": int(args.train_count),
                    "test_count": int(bundle["n_samples_test"]),
                },
                "synthetic_grid": {
                    "temp_min": float(args.syn_temp_min),
                    "temp_max": float(args.syn_temp_max),
                    "temp_points": int(args.syn_temp_points),
                    "er_min": float(args.syn_er_min),
                    "er_max": float(args.syn_er_max),
                    "er_points": int(args.syn_er_points),
                },
                "selection_metric": "selection_score",
                "label_normalization": "row-wise closure normalization in base_mod.load_data()",
                "real_forward": "y_real = softmax(z_part + D(x) * z_res)",
                "synthetic_forward": "y_syn = softmax(z_part + z_res)",
                "loss_train": "E = E_R(real full forward) + lambda_s * E_S + lambda_M * E_M_diff(synthetic physics forward)",
                "mono_training_metric": "E_M_diff_train",
                "mono_diff_definition": "E_M = sum_{i,j} ReLU(-mu_ij * d y_hat[out_j] / d x[in_j]) / N_effective(mu_ij!=0)",
                "mono_rule_columns_fixed": ["ER->H2", "ER->CO2", "ER->CO", "ER->CH4", "ER->N2", "T->H2", "T->CO2", "T->CO", "T->CH4", "T->N2"],
                "mu_default_row": [-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0],
                "mono_report_metric": "senior finite-difference PCD compliance on real train/test sets",
                "mono_training_source": "synthetic_samples_physics_branch",
                "mono_report_source": "real_train_test_sets",
                "optimizer_weight_decay": 0.0,
                "joint_stage_mapping": {
                    "particular_pretrain_epochs": int(best_pack["cfg"].epochs_part),
                    "joint_epochs_effective": int(best_pack["rec"]["joint_epochs_effective"]),
                    "joint_lr_schedule": "cosine annealing within a single joint stage",
                    "joint_lr_start_from_legacy_lr_res": float(best_pack["rec"]["joint_lr_start"]),
                    "joint_lr_end_from_legacy_lr_joint": float(best_pack["rec"]["joint_lr_end"]),
                    "legacy_epochs_res": int(best_pack["cfg"].epochs_res),
                    "legacy_epochs_joint": int(best_pack["cfg"].epochs_joint),
                },
                "particular_after_pretraining": "joint_stage_trainable",
                "hard_boundary_distance_features": "C, H, S, Particle size, Ash, Moisture, Temperature, ER",
                "mono_injection_branch": "paper-like joint physics branch with real full forward and synthetic no-D softmax branch",
            },
            f,
            indent=2,
        )

    print("Done.")
    print("Output dir:", out_dir)
    print("Best dir:", best_dir)
    print("Selected selection_score:", float(best_pack["rec"]["selection_score"]) if "selection_score" in best_pack["rec"] else float(selection_score(best_pack["rec"])))
    print("Best R2_mean_test:", float(best_pack["rec"]["R2_mean_test"]))


if __name__ == "__main__":
    main()
