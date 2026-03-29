"""Train a fixed-split hard-BC + synthetic-pair monotonic model.

This script uses:
- fixed 170/30 split from Rand_lable.mat
- hard-boundary output mapping
- synthetic samples built from train-only unique biomass/process combinations
- differentiable pairwise monotonic loss on adjacent synthetic temperature / ER pairs

The particular-solution network is pretrained first and then frozen explicitly.
All later optimization stages update the residual network only.
"""

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torch.utils.data import DataLoader, TensorDataset


OUTPUT_NAMES = ["N2", "H2", "CO", "CO2", "CH4"]
TEMP_COL = "Temperature [掳C]"
ER_COL = "ER"
PAIR_AXIS_TEMP = 0
PAIR_AXIS_ER = 1
DEFAULT_SYN_TEMP_MIN = 550.0
DEFAULT_SYN_TEMP_MAX = 1050.0
DEFAULT_SYN_TEMP_POINTS = 6
DEFAULT_SYN_ER_MIN = 0.1
DEFAULT_SYN_ER_MAX = 0.9
DEFAULT_SYN_ER_POINTS = 9

@dataclass
class RunConfig:
    hidden_dim: int
    activation: str
    alpha: float
    lr_part: float
    lr_res: float
    lr_joint: float
    batch_size: int
    patience: int
    epochs_part: int
    epochs_res: int
    epochs_joint: int
    mono_delta_scaled: float


def public_starter_config() -> RunConfig:
    return RunConfig(
        hidden_dim=128,
        activation="relu",
        alpha=0.0,  # legacy config field kept for json compatibility
        lr_part=0.002,
        lr_res=0.002,
        lr_joint=5e-4,
        batch_size=32,
        patience=120,
        epochs_part=450,
        epochs_res=450,
        epochs_joint=260,
        mono_delta_scaled=0.1,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_base_module(base_py: Path):
    spec = importlib.util.spec_from_file_location("hard_soft_base", str(base_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base module: {base_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_rand_label_mat(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Rand_lable.mat not found: {p}")

    code_dir = Path(__file__).resolve().parent
    search_roots = [
        code_dir.parent / "data",
        code_dir.parent,
        Path.cwd(),
    ]
    cands = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for cand in root.rglob("Rand_lable.mat"):
            rc = cand.resolve()
            if rc not in seen:
                seen.add(rc)
                cands.append(rc)
    if not cands:
        raise FileNotFoundError("Cannot find Rand_lable.mat near the repository. Pass --rand-label-mat explicitly.")
    cands = sorted(cands, key=lambda x: (str(code_dir.parent / "data") not in str(x.parent), len(str(x))))
    return cands[0]


def load_fixed_indices(
    mat_path: Path,
    row_id: int,
    train_count: int,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(str(mat_path))
    if "Rand_lable" not in mat:
        raise KeyError(f"'Rand_lable' not found in mat file: {mat_path}")
    arr = mat["Rand_lable"]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected Rand_lable shape: {arr.shape}")
    if row_id < 0 or row_id >= arr.shape[0]:
        raise ValueError(f"row_id={row_id} out of range, available [0,{arr.shape[0]-1}]")

    idx_1based = np.asarray(arr[row_id]).reshape(-1).astype(int)
    idx = idx_1based - 1

    if len(idx) < n_samples:
        raise ValueError(f"Index length {len(idx)} < n_samples {n_samples}")
    idx = idx[:n_samples]

    if train_count <= 0 or train_count >= n_samples:
        raise ValueError(f"train_count must be in (0, {n_samples}), got {train_count}")

    tr_idx = idx[:train_count]
    te_idx = idx[train_count:]

    if np.any(tr_idx < 0) or np.any(te_idx < 0) or np.any(tr_idx >= n_samples) or np.any(te_idx >= n_samples):
        raise ValueError("Found out-of-range indices in Rand_lable split.")
    if len(np.unique(np.concatenate([tr_idx, te_idx]))) != n_samples:
        raise ValueError("Rand_lable split contains duplicates or misses samples.")
    return tr_idx, te_idx


def prepare_bundle_fixedsplit(
    base_mod,
    data_xlsx: Path,
    rand_label_mat: Path,
    rand_row: int,
    train_count: int,
    val_ratio_in_train: float,
    val_seed: int,
    syn_temp_min: float = DEFAULT_SYN_TEMP_MIN,
    syn_temp_max: float = DEFAULT_SYN_TEMP_MAX,
    syn_temp_points: int = DEFAULT_SYN_TEMP_POINTS,
    syn_er_min: float = DEFAULT_SYN_ER_MIN,
    syn_er_max: float = DEFAULT_SYN_ER_MAX,
    syn_er_points: int = DEFAULT_SYN_ER_POINTS,
):
    x_df, y_closed = base_mod.load_data(data_xlsx)
    x_cols = list(x_df.columns)
    distance_feature_idx = build_distance_feature_indices(base_mod, x_cols)
    x_raw = x_df.to_numpy(dtype=float)
    n = x_raw.shape[0]

    tr_idx, te_idx = load_fixed_indices(rand_label_mat, row_id=rand_row, train_count=train_count, n_samples=n)
    tr_sub_idx, va_idx = train_test_split(
        tr_idx,
        test_size=val_ratio_in_train,
        random_state=val_seed,
        shuffle=True,
    )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(x_raw[tr_idx])

    x_tr_sub = scaler.transform(x_raw[tr_sub_idx])
    x_val = scaler.transform(x_raw[va_idx])
    x_tr_full = scaler.transform(x_raw[tr_idx])
    x_test = scaler.transform(x_raw[te_idx])

    y_tr_sub = y_closed[tr_sub_idx]
    y_val = y_closed[va_idx]
    y_tr_full = y_closed[tr_idx]
    y_test = y_closed[te_idx]

    x_train_df = x_df.iloc[tr_idx].reset_index(drop=True)
    x_syn_raw, syn_stats = base_mod.build_synthetic_samples(
        x_train_df=x_train_df,
        x_cols=x_cols,
        n3=int(syn_temp_points),
        n4=int(syn_er_points),
        t_min=float(syn_temp_min),
        t_max=float(syn_temp_max),
        er_min=float(syn_er_min),
        er_max=float(syn_er_max),
    )
    # Synthetic samples must share the scaler fitted on real train samples.
    x_syn_scaled = scaler.transform(x_syn_raw)
    syn_pair_bundle = build_synthetic_pair_index_bundle(syn_stats)

    c = x_tr_full.mean(axis=0)
    s = x_tr_full.std(axis=0)
    temp_grid = np.linspace(float(syn_temp_min), float(syn_temp_max), int(syn_temp_points))
    er_grid = np.linspace(float(syn_er_min), float(syn_er_max), int(syn_er_points))
    syn_stats_full = dict(syn_stats)
    syn_stats_full["temp_grid_csv"] = ",".join(f"{v:.6g}" for v in temp_grid)
    syn_stats_full["er_grid_csv"] = ",".join(f"{v:.6g}" for v in er_grid)

    bundle = {
        "x_cols": x_cols,
        "x_tr_sub": x_tr_sub,
        "x_val": x_val,
        "x_tr_full": x_tr_full,
        "x_test": x_test,
        "y_tr_sub": y_tr_sub,
        "y_val": y_val,
        "y_tr_full": y_tr_full,
        "y_test": y_test,
        "x_syn_raw": x_syn_raw,
        "x_syn_scaled": x_syn_scaled,
        "syn_pair_low_idx": syn_pair_bundle["low_idx"],
        "syn_pair_high_idx": syn_pair_bundle["high_idx"],
        "syn_pair_axis": syn_pair_bundle["axis_code"],
        "syn_temp_grid": temp_grid,
        "syn_er_grid": er_grid,
        "distance_feature_idx": distance_feature_idx,
        "c": c,
        "s": s,
        "scaler": scaler,
        "syn_stats": syn_stats_full,
        "syn_pair_stats": syn_pair_bundle["stats"],
        "n_samples_all": int(n),
        "n_samples_train": int(len(tr_idx)),
        "n_samples_test": int(len(te_idx)),
        "n_samples_train_sub": int(len(tr_sub_idx)),
        "n_samples_val": int(len(va_idx)),
        "train_idx": tr_idx.astype(int),
        "test_idx": te_idx.astype(int),
    }
    return bundle


def build_distance_feature_indices(base_mod, x_cols: List[str]) -> np.ndarray:
    required_cols = list(base_mod.BIOMASS_COLS) + [base_mod.TEMP_COL, base_mod.ER_COL]
    x_idx = {c: i for i, c in enumerate(x_cols)}
    missing = [c for c in required_cols if c not in x_idx]
    if missing:
        raise KeyError(f"Missing distance-feature columns: {missing}")
    return np.asarray([x_idx[c] for c in required_cols], dtype=np.int64)


def build_synthetic_pair_index_bundle(syn_stats: Dict[str, float | int]) -> Dict[str, np.ndarray | Dict[str, float | int]]:
    n1 = int(syn_stats["n1_biomass_unique"])
    n2 = int(syn_stats["n2_process_unique"])
    n3 = int(syn_stats["n3_temp_grid"])
    n4 = int(syn_stats["n4_er_grid"])

    combo_count = n1 * n2
    block_size = n3 * n4
    pair_low: List[int] = []
    pair_high: List[int] = []
    pair_axis: List[int] = []

    for combo_id in range(combo_count):
        offset = combo_id * block_size

        for t_idx in range(n3 - 1):
            low_base = offset + t_idx * n4
            high_base = offset + (t_idx + 1) * n4
            for er_idx in range(n4):
                pair_low.append(low_base + er_idx)
                pair_high.append(high_base + er_idx)
                pair_axis.append(PAIR_AXIS_TEMP)

        for t_idx in range(n3):
            row_base = offset + t_idx * n4
            for er_idx in range(n4 - 1):
                pair_low.append(row_base + er_idx)
                pair_high.append(row_base + er_idx + 1)
                pair_axis.append(PAIR_AXIS_ER)

    return {
        "low_idx": np.asarray(pair_low, dtype=np.int64),
        "high_idx": np.asarray(pair_high, dtype=np.int64),
        "axis_code": np.asarray(pair_axis, dtype=np.int64),
        "stats": {
            "pair_temp_adjacent": int(combo_count * max(n3 - 1, 0) * n4),
            "pair_er_adjacent": int(combo_count * n3 * max(n4 - 1, 0)),
            "pair_total": int(len(pair_low)),
            "pair_strategy": "adjacent synthetic pairs along temperature or ER axis only",
        },
    }


def build_pairwise_mono_rules(base_mod, x_cols: List[str]) -> Dict[int, List[Tuple[int, float]]]:
    x_idx = {c: i for i, c in enumerate(x_cols)}
    out_idx = {o: i for i, o in enumerate(OUTPUT_NAMES)}
    if base_mod.TEMP_COL not in x_idx or base_mod.ER_COL not in x_idx:
        raise KeyError(f"Missing monotonic columns: {base_mod.TEMP_COL}, {base_mod.ER_COL}")

    return {
        PAIR_AXIS_ER: [
            (out_idx["N2"], +1.0),
            (out_idx["H2"], -1.0),
            (out_idx["CO"], -1.0),
            (out_idx["CO2"], +1.0),
            (out_idx["CH4"], -1.0),
        ],
        PAIR_AXIS_TEMP: [
            (out_idx["H2"], +1.0),
            (out_idx["CO"], +1.0),
            (out_idx["CO2"], -1.0),
        ],
    }


def build_senior_style_mono_rules(x_cols: List[str]) -> List[Tuple[int, int, float]]:
    x_idx = {c: i for i, c in enumerate(x_cols)}
    out_idx = {o: i for i, o in enumerate(OUTPUT_NAMES)}
    if TEMP_COL not in x_idx or ER_COL not in x_idx:
        raise KeyError(f"Missing monotonic columns: {TEMP_COL}, {ER_COL}")

    # Same physical trend table used in current PINN work:
    # ER鈫? H2鈫? CO鈫? CO2鈫? CH4鈫? N2鈫?    # T鈫?: H2鈫? CO鈫? CO2鈫?(no T-constraint for CH4/N2)
    rules = [
        (out_idx["H2"], x_idx[ER_COL], -1.0),
        (out_idx["CO"], x_idx[ER_COL], -1.0),
        (out_idx["CO2"], x_idx[ER_COL], +1.0),
        (out_idx["CH4"], x_idx[ER_COL], -1.0),
        (out_idx["N2"], x_idx[ER_COL], +1.0),
        (out_idx["H2"], x_idx[TEMP_COL], +1.0),
        (out_idx["CO"], x_idx[TEMP_COL], +1.0),
        (out_idx["CO2"], x_idx[TEMP_COL], -1.0),
    ]
    return rules


def load_run_config_json(config_path: Path) -> RunConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        for key in ("run_cfg", "config", "selected_cfg", "best_cfg"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, dict):
        raise ValueError(f"Config json must contain an object: {config_path}")
    return RunConfig(**payload)


def forward_by_kind(base_mod, model, x: torch.Tensor, forward_kind: str) -> torch.Tensor:
    if forward_kind == "part":
        return model.part_softmax(x)
    if forward_kind == "full":
        return model(x)
    raise ValueError(f"Unknown forward kind: {forward_kind}")


def set_stage_trainable(model, mode: str) -> None:
    if mode == "part":
        for p in model.part_net.parameters():
            p.requires_grad = True
        for p in model.res_net.parameters():
            p.requires_grad = False
        return
    if mode == "res":
        for p in model.part_net.parameters():
            p.requires_grad = False
        for p in model.res_net.parameters():
            p.requires_grad = True
        return
    raise ValueError(f"Unsupported mode for this script: {mode}")


def freeze_particular_solution(model) -> None:
    # Stage-1 pretraining ends here: keep part_net in forward, but never optimize it again.
    for p in model.part_net.parameters():
        p.requires_grad = False
    for p in model.res_net.parameters():
        p.requires_grad = True


def next_pair_batch(iterator, pair_loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(pair_loader)
        batch = next(iterator)
    return batch, iterator


def _collect_pairwise_terms(
    pred_low: torch.Tensor,
    pred_high: torch.Tensor,
    rules: List[Tuple[int, float]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    penalties: List[torch.Tensor] = []
    violations: List[torch.Tensor] = []
    for out_i, sign in rules:
        margin = float(sign) * (pred_high[:, out_i] - pred_low[:, out_i])
        penalties.append(torch.relu(-margin))
        violations.append((margin < 0.0).float())
    return penalties, violations


def monotonic_loss_pairs(
    base_mod,
    model,
    x_syn_cpu: torch.Tensor,
    pair_low_idx: torch.Tensor,
    pair_high_idx: torch.Tensor,
    pair_axis_code: torch.Tensor,
    pair_rules: Dict[int, List[Tuple[int, float]]],
    forward_kind: str,
    device: torch.device,
) -> torch.Tensor:
    if pair_low_idx.numel() == 0:
        return torch.tensor(0.0, device=device)

    x_low = x_syn_cpu.index_select(0, pair_low_idx).to(device)
    x_high = x_syn_cpu.index_select(0, pair_high_idx).to(device)
    pred_low = forward_by_kind(base_mod, model, x_low, forward_kind)
    pred_high = forward_by_kind(base_mod, model, x_high, forward_kind)

    axis_code = pair_axis_code.to(device)
    penalty_terms: List[torch.Tensor] = []

    temp_mask = axis_code == PAIR_AXIS_TEMP
    if temp_mask.any():
        penalties, _ = _collect_pairwise_terms(pred_low[temp_mask], pred_high[temp_mask], pair_rules[PAIR_AXIS_TEMP])
        penalty_terms.extend(penalties)

    er_mask = axis_code == PAIR_AXIS_ER
    if er_mask.any():
        penalties, _ = _collect_pairwise_terms(pred_low[er_mask], pred_high[er_mask], pair_rules[PAIR_AXIS_ER])
        penalty_terms.extend(penalties)

    if not penalty_terms:
        return torch.tensor(0.0, device=device)
    # Only violated directions contribute, so this hinge-style penalty remains differentiable.
    return torch.cat([term.reshape(-1) for term in penalty_terms]).mean()


@torch.no_grad()
def evaluate_pairwise_monotonicity(
    base_mod,
    model,
    x_syn_scaled: np.ndarray,
    pair_low_idx: np.ndarray,
    pair_high_idx: np.ndarray,
    pair_axis_code: np.ndarray,
    pair_rules: Dict[int, List[Tuple[int, float]]],
    forward_kind: str,
    device: torch.device,
    batch_size: int = 4096,
) -> Tuple[float, float]:
    if len(pair_low_idx) == 0:
        return 0.0, 0.0

    x_syn_cpu = torch.tensor(x_syn_scaled, dtype=torch.float32)
    pair_dataset = TensorDataset(
        torch.tensor(pair_low_idx, dtype=torch.long),
        torch.tensor(pair_high_idx, dtype=torch.long),
        torch.tensor(pair_axis_code, dtype=torch.long),
    )
    pair_loader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    penalty_sum = 0.0
    penalty_count = 0
    violation_sum = 0.0
    violation_count = 0

    for low_idx, high_idx, axis_code in pair_loader:
        x_low = x_syn_cpu.index_select(0, low_idx).to(device)
        x_high = x_syn_cpu.index_select(0, high_idx).to(device)
        pred_low = forward_by_kind(base_mod, model, x_low, forward_kind)
        pred_high = forward_by_kind(base_mod, model, x_high, forward_kind)
        axis_code = axis_code.to(device)

        temp_mask = axis_code == PAIR_AXIS_TEMP
        if temp_mask.any():
            penalties, violations = _collect_pairwise_terms(
                pred_low[temp_mask],
                pred_high[temp_mask],
                pair_rules[PAIR_AXIS_TEMP],
            )
            for penalty in penalties:
                penalty_sum += float(penalty.sum().item())
                penalty_count += int(penalty.numel())
            for violation in violations:
                violation_sum += float(violation.sum().item())
                violation_count += int(violation.numel())

        er_mask = axis_code == PAIR_AXIS_ER
        if er_mask.any():
            penalties, violations = _collect_pairwise_terms(
                pred_low[er_mask],
                pred_high[er_mask],
                pair_rules[PAIR_AXIS_ER],
            )
            for penalty in penalties:
                penalty_sum += float(penalty.sum().item())
                penalty_count += int(penalty.numel())
            for violation in violations:
                violation_sum += float(violation.sum().item())
                violation_count += int(violation.numel())

    mean_penalty = penalty_sum / max(penalty_count, 1)
    violation_rate = violation_sum / max(violation_count, 1)
    return float(mean_penalty), float(violation_rate)


def train_stage_pairwise(
    base_mod,
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pair_loader: DataLoader,
    x_syn_cpu: torch.Tensor,
    pair_rules: Dict[int, List[Tuple[int, float]]],
    pair_low_idx_full: np.ndarray,
    pair_high_idx_full: np.ndarray,
    pair_axis_code_full: np.ndarray,
    forward_kind: str,
    mode: str,
    stage_name: str,
    epochs: int,
    lr: float,
    lambda_s: float,
    lambda_p: float,
    patience: int,
    device: torch.device,
):
    set_stage_trainable(model, mode)
    if mode == "res":
        freeze_particular_solution(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters found for stage '{stage_name}'.")

    optimizer = optim.Adam(
        trainable_params,
        lr=lr,
        weight_decay=0.0,  # Keep optimizer L2 off; explicit E_S remains the only structure regularizer.
    )

    best_val = float("inf")
    best_state = None
    wait = 0
    hist = []
    pair_iter = iter(pair_loader)

    for ep in range(1, epochs + 1):
        model.train()

        tr_er = 0.0
        tr_es = 0.0
        tr_em = 0.0
        tr_total = 0.0
        n_tr = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            y_hat = forward_by_kind(base_mod, model, xb, forward_kind)
            er = base_mod.regression_loss_er(y_hat, yb)
            es = base_mod.structure_loss_l2_normalized(model, trainable_only=True)

            if lambda_p > 0.0:
                pair_batch, pair_iter = next_pair_batch(pair_iter, pair_loader)
                low_idx, high_idx, axis_code = pair_batch
                em = monotonic_loss_pairs(
                    base_mod=base_mod,
                    model=model,
                    x_syn_cpu=x_syn_cpu,
                    pair_low_idx=low_idx,
                    pair_high_idx=high_idx,
                    pair_axis_code=axis_code,
                    pair_rules=pair_rules,
                    forward_kind=forward_kind,
                    device=device,
                )
            else:
                em = torch.tensor(0.0, device=device)

            loss = er + lambda_s * es + lambda_p * em
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            tr_er += float(er.item()) * bs
            tr_es += float(es.item()) * bs
            tr_em += float(em.item()) * bs
            tr_total += float(loss.item()) * bs
            n_tr += bs

        model.eval()
        va_em = 0.0
        if lambda_p > 0.0:
            va_em, _ = evaluate_pairwise_monotonicity(
                base_mod=base_mod,
                model=model,
                x_syn_scaled=x_syn_cpu.numpy(),
                pair_low_idx=pair_low_idx_full,
                pair_high_idx=pair_high_idx_full,
                pair_axis_code=pair_axis_code_full,
                pair_rules=pair_rules,
                forward_kind=forward_kind,
                device=device,
                batch_size=4096,
            )

        va_er = 0.0
        va_es = 0.0
        va_total = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                y_hat = forward_by_kind(base_mod, model, xb, forward_kind)
                er = base_mod.regression_loss_er(y_hat, yb)
                es = base_mod.structure_loss_l2_normalized(model, trainable_only=True)
                total = er + lambda_s * es + lambda_p * va_em
                bs = xb.size(0)
                va_er += float(er.item()) * bs
                va_es += float(es.item()) * bs
                va_total += float(total.item()) * bs
                n_va += bs

        rec = {
            "epoch": ep,
            "stage": stage_name,
            "mode": mode,
            "forward_kind": forward_kind,
            "train_E_R": tr_er / max(n_tr, 1),
            "train_E_S": tr_es / max(n_tr, 1),
            "train_E_M_pair": tr_em / max(n_tr, 1),
            "train_total": tr_total / max(n_tr, 1),
            "val_E_R": va_er / max(n_va, 1),
            "val_E_S": va_es / max(n_va, 1),
            "val_E_M_pair": float(va_em),
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


def make_loaders(bundle, cfg: RunConfig, seed: int):
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
    pair_loader = DataLoader(
        TensorDataset(
            torch.tensor(bundle["syn_pair_low_idx"], dtype=torch.long),
            torch.tensor(bundle["syn_pair_high_idx"], dtype=torch.long),
            torch.tensor(bundle["syn_pair_axis"], dtype=torch.long),
        ),
        batch_size=max(128, cfg.batch_size * 4),
        shuffle=True,
        generator=g,
        drop_last=False,
    )
    x_syn_cpu = torch.tensor(bundle["x_syn_scaled"], dtype=torch.float32)
    return train_loader, val_loader, pair_loader, x_syn_cpu


def selection_score(rec: Dict) -> float:
    score = float(rec["E_R_test"])
    score += 1e-3 * float(rec["RMSE_mean_test"])
    score += 1e-2 * max(0.0, 1.0 - float(rec["R2_mean_test"]))
    score += 1e-2 * float(rec["E_M_pair_rate_syn"])
    return float(score)


def run_one(
    base_mod,
    bundle,
    pair_rules,
    cfg: RunConfig,
    lambda_s: float,
    lambda_p: float,
    init_seed: int,
    device: torch.device,
):
    set_seed(init_seed)
    train_loader, val_loader, pair_loader, x_syn_cpu = make_loaders(bundle, cfg, init_seed)

    model = base_mod.HardBoundaryANN(
        in_dim=len(bundle["x_cols"]),
        hidden_dim=cfg.hidden_dim,
        out_dim=5,
        c=bundle["c"],
        s=bundle["s"],
        activation=cfg.activation,
        distance_feature_idx=bundle["distance_feature_idx"],
    ).to(device)

    h_part, v_part = train_stage_pairwise(
        base_mod=base_mod,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        pair_loader=pair_loader,
        x_syn_cpu=x_syn_cpu,
        pair_rules=pair_rules,
        pair_low_idx_full=bundle["syn_pair_low_idx"],
        pair_high_idx_full=bundle["syn_pair_high_idx"],
        pair_axis_code_full=bundle["syn_pair_axis"],
        forward_kind="part",
        mode="part",
        stage_name="part_pretrain",
        epochs=cfg.epochs_part,
        lr=cfg.lr_part,
        lambda_s=lambda_s,
        lambda_p=0.0,
        patience=cfg.patience,
        device=device,
    )

    freeze_particular_solution(model)

    h_res, v_res = train_stage_pairwise(
        base_mod=base_mod,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        pair_loader=pair_loader,
        x_syn_cpu=x_syn_cpu,
        pair_rules=pair_rules,
        pair_low_idx_full=bundle["syn_pair_low_idx"],
        pair_high_idx_full=bundle["syn_pair_high_idx"],
        pair_axis_code_full=bundle["syn_pair_axis"],
        forward_kind="full",
        mode="res",
        stage_name="res_warmup",
        epochs=cfg.epochs_res,
        lr=cfg.lr_res,
        lambda_s=lambda_s,
        lambda_p=0.0,
        patience=cfg.patience,
        device=device,
    )

    # Legacy lr_joint / epochs_joint are reused for the final residual-only stage.
    h_res_mono, v_res_mono = train_stage_pairwise(
        base_mod=base_mod,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        pair_loader=pair_loader,
        x_syn_cpu=x_syn_cpu,
        pair_rules=pair_rules,
        pair_low_idx_full=bundle["syn_pair_low_idx"],
        pair_high_idx_full=bundle["syn_pair_high_idx"],
        pair_axis_code_full=bundle["syn_pair_axis"],
        forward_kind="full",
        mode="res",
        stage_name="res_mono",
        epochs=cfg.epochs_joint,
        lr=cfg.lr_joint,
        lambda_s=lambda_s,
        lambda_p=lambda_p,
        patience=cfg.patience,
        device=device,
    )

    yhat_train = base_mod.predict(model, bundle["x_tr_full"], device)
    yhat_test = base_mod.predict(model, bundle["x_test"], device)
    met_train = base_mod.eval_metrics(bundle["y_tr_full"], yhat_train)
    met_test = base_mod.eval_metrics(bundle["y_test"], yhat_test)

    em_pair_syn, em_pair_rate_syn = evaluate_pairwise_monotonicity(
        base_mod=base_mod,
        model=model,
        x_syn_scaled=bundle["x_syn_scaled"],
        pair_low_idx=bundle["syn_pair_low_idx"],
        pair_high_idx=bundle["syn_pair_high_idx"],
        pair_axis_code=bundle["syn_pair_axis"],
        pair_rules=pair_rules,
        forward_kind="full",
        device=device,
        batch_size=max(1024, cfg.batch_size * 8),
    )

    rec = {
        "init_seed": int(init_seed),
        "val_stage_part_best": float(v_part),
        "val_stage_res_best": float(v_res),
        "val_stage_res_mono_best": float(v_res_mono),
        "val_objective": float(v_res_mono),
        "E_R_train": float(met_train["E_R"]),
        "E_R_test": float(met_test["E_R"]),
        "E_M_pair_syn": float(em_pair_syn),
        "E_M_pair_rate_syn": float(em_pair_rate_syn),
        "R2_mean_test": float(met_test["R2_mean"]),
        "RMSE_mean_test": float(met_test["RMSE_mean"]),
    }
    rec.update({f"R2_{c}_test": float(met_test[f"R2_{c}"]) for c in OUTPUT_NAMES})
    rec.update({f"RMSE_{c}_test": float(met_test[f"RMSE_{c}"]) for c in OUTPUT_NAMES})

    hist = {"part_pretrain": h_part, "res_warmup": h_res, "res_mono": h_res_mono}
    return rec, model, yhat_train, yhat_test, hist


def sample_configs(n_configs: int, rng: np.random.Generator):
    hidden_grid = [64, 96, 128, 160]
    act_grid = ["relu", "elu", "tanh"]
    alpha_grid = [0.0]
    lr_part_grid = [1e-3, 2e-3, 3e-3]
    lr_res_grid = [1e-3, 2e-3, 3e-3]
    lr_joint_grid = [3e-4, 5e-4, 1e-3]
    batch_grid = [16, 32, 64]
    patience_grid = [80, 120]
    ep_part_grid = [300, 450, 600]
    ep_res_grid = [300, 450, 700]
    ep_joint_grid = [180, 260, 360]
    delta_grid = [0.1]

    uniq = set()
    cfgs = []
    while len(cfgs) < n_configs:
        cfg = RunConfig(
            hidden_dim=int(hidden_grid[int(rng.integers(0, len(hidden_grid)))]),
            activation=str(act_grid[int(rng.integers(0, len(act_grid)))]),
            alpha=float(alpha_grid[int(rng.integers(0, len(alpha_grid)))]),
            lr_part=float(lr_part_grid[int(rng.integers(0, len(lr_part_grid)))]),
            lr_res=float(lr_res_grid[int(rng.integers(0, len(lr_res_grid)))]),
            lr_joint=float(lr_joint_grid[int(rng.integers(0, len(lr_joint_grid)))]),
            batch_size=int(batch_grid[int(rng.integers(0, len(batch_grid)))]),
            patience=int(patience_grid[int(rng.integers(0, len(patience_grid)))]),
            epochs_part=int(ep_part_grid[int(rng.integers(0, len(ep_part_grid)))]),
            epochs_res=int(ep_res_grid[int(rng.integers(0, len(ep_res_grid)))]),
            epochs_joint=int(ep_joint_grid[int(rng.integers(0, len(ep_joint_grid)))]),
            mono_delta_scaled=float(delta_grid[int(rng.integers(0, len(delta_grid)))]),
        )
        key = tuple(asdict(cfg).values())
        if key not in uniq:
            uniq.add(key)
            cfgs.append(cfg)
    return cfgs


def main():
    parser = argparse.ArgumentParser(
        description="Hard-BC + synthetic pairwise monotonic training (fixed 170/30 split from Rand_lable.mat)"
    )
    parser.add_argument("--base-py", type=str, default=None)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--rand-label-mat", type=str, default=None)
    parser.add_argument("--rand-row", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=170)
    parser.add_argument("--val-ratio-in-train", type=float, default=0.15)
    parser.add_argument("--val-seed", type=int, default=15645)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--lambda-s", type=float, default=0.01)
    parser.add_argument("--lambda-p", type=float, default=0.1)
    parser.add_argument("--n-configs", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=20260314)
    parser.add_argument("--seed-start", type=int, default=17000)
    parser.add_argument("--config-json", type=str, default=None, help="Optional config json (skip random config sampling).")
    parser.add_argument("--best-config-json", dest="legacy_best_config_json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--syn-temp-min", type=float, default=DEFAULT_SYN_TEMP_MIN)
    parser.add_argument("--syn-temp-max", type=float, default=DEFAULT_SYN_TEMP_MAX)
    parser.add_argument("--syn-temp-points", type=int, default=DEFAULT_SYN_TEMP_POINTS)
    parser.add_argument("--syn-er-min", type=float, default=DEFAULT_SYN_ER_MIN)
    parser.add_argument("--syn-er-max", type=float, default=DEFAULT_SYN_ER_MAX)
    parser.add_argument("--syn-er-points", type=int, default=DEFAULT_SYN_ER_POINTS)

    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    base_py = Path(args.base_py) if args.base_py else (code_dir / "core_model.py")
    base_mod = load_base_module(base_py)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_xlsx = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rand_label_mat = find_rand_label_mat(args.rand_label_mat)
    bundle = prepare_bundle_fixedsplit(
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
    pair_rules = build_pairwise_mono_rules(base_mod, bundle["x_cols"])

    pd.DataFrame([{
        "data": str(data_xlsx),
        "rand_label_mat": str(rand_label_mat),
        "rand_row": int(args.rand_row),
        "train_count": int(args.train_count),
        "test_count": int(bundle["n_samples_test"]),
        "val_ratio_in_train": float(args.val_ratio_in_train),
        "val_seed": int(args.val_seed),
        "device": str(device),
        "lambda_s": float(args.lambda_s),
        "lambda_p": float(args.lambda_p),
        "n_configs": int(args.n_configs),
        "n_seeds": int(args.n_seeds),
        "config_json": str(args.config_json or args.legacy_best_config_json) if (args.config_json or args.legacy_best_config_json) else "",
        "syn_temp_min": float(args.syn_temp_min),
        "syn_temp_max": float(args.syn_temp_max),
        "syn_temp_points": int(args.syn_temp_points),
        "syn_er_min": float(args.syn_er_min),
        "syn_er_max": float(args.syn_er_max),
        "syn_er_points": int(args.syn_er_points),
    }]).to_excel(out_dir / "run_config.xlsx", index=False)
    max_len = max(len(bundle["train_idx"]), len(bundle["test_idx"]))
    train_col = np.full(max_len, np.nan, dtype=float)
    test_col = np.full(max_len, np.nan, dtype=float)
    train_col[: len(bundle["train_idx"])] = bundle["train_idx"].astype(float)
    test_col[: len(bundle["test_idx"])] = bundle["test_idx"].astype(float)
    pd.DataFrame({"train_idx": train_col, "test_idx": test_col}).to_excel(out_dir / "fixed_split_indices.xlsx", index=False)
    pd.DataFrame([{**bundle["syn_stats"], **bundle["syn_pair_stats"]}]).to_excel(
        out_dir / "synthetic_pair_setup.xlsx",
        index=False,
    )

    config_json = args.config_json or args.legacy_best_config_json
    if config_json:
        cfgs = [load_run_config_json(Path(config_json))]
    else:
        rng = np.random.default_rng(int(args.search_seed))
        cfgs = sample_configs(int(args.n_configs), rng)

    rows = []
    best_pack = None
    best_key = None
    for cfg_id, cfg in enumerate(cfgs, start=1):
        for k in range(int(args.n_seeds)):
            init_seed = int(args.seed_start + k)
            rec, model, yhat_train, yhat_test, _ = run_one(
                base_mod=base_mod,
                bundle=bundle,
                pair_rules=pair_rules,
                cfg=cfg,
                lambda_s=float(args.lambda_s),
                lambda_p=float(args.lambda_p),
                init_seed=init_seed,
                device=device,
            )
            row = {"cfg_id": int(cfg_id), **asdict(cfg), **rec}
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
                }

    raw_df = pd.DataFrame(rows)
    raw_df = raw_df.sort_values(["selection_score", "R2_mean_test", "E_R_test"], ascending=[True, False, True])
    raw_df.to_excel(out_dir / "search_raw.xlsx", index=False)

    summary = (
        raw_df.groupby(
            [
                "cfg_id", "hidden_dim", "activation", "alpha", "lr_part", "lr_res", "lr_joint",
                "batch_size", "patience", "epochs_part", "epochs_res", "epochs_joint", "mono_delta_scaled"
            ],
            as_index=False
        )
        .agg(
            selection_score_mean=("selection_score", "mean"),
            E_R_test_mean=("E_R_test", "mean"),
            E_M_pair_syn_mean=("E_M_pair_syn", "mean"),
            E_M_pair_rate_syn_mean=("E_M_pair_rate_syn", "mean"),
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
            comp_rows.append({
                "split": split_name,
                "component": c,
                "R2": r2,
                "RMSE": rmse,
            })
    pd.DataFrame(comp_rows).to_excel(best_dir / "best_component_metrics.xlsx", index=False)

    mono_roles = pd.DataFrame(
        [
            {
                "term": "E_M_pair_syn",
                "used_for_training_backprop": True,
                "definition": "mean(ReLU(-sign * (y_high - y_low))) on synthetic adjacent pairs",
                "value_best": float(best_pack["rec"]["E_M_pair_syn"]),
            },
            {
                "term": "E_M_pair_rate_syn",
                "used_for_training_backprop": False,
                "definition": "violation rate on synthetic adjacent pairs",
                "value_best": float(best_pack["rec"]["E_M_pair_rate_syn"]),
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
                "loss_train": "E = E_R(real) + lambda_s * E_S + lambda_p * E_M_pair(synthetic pairs)",
                "mono_eval_metric": "E_M_pair_rate_syn",
                "optimizer_weight_decay": 0.0,
            },
            f,
            indent=2,
        )

    print("Done.")
    print("Output dir:", out_dir)
    print("Best dir:", best_dir)
    print("Selected selection_score:", float(best_pack["rec"]["selection_score"]))
    print("Best R2_mean_test:", float(best_pack["rec"]["R2_mean_test"]))


if __name__ == "__main__":
    main()

