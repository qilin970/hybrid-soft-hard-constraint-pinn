"""Core utilities for the hard-boundary + soft-monotonic PINN-style model.

This module contains:
- data loading and closure normalization for the 5 gas fractions
- Min-Max input normalization fit on the training split only
- the hard-boundary network y_hat = softmax(z_part + D(x) * z_res)
- regression / structural / monotonic loss definitions
- training, evaluation, plotting, and the original stage-A / stage-B pipeline

This file is kept as the shared base module imported by the other submission scripts.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


OUTPUT_NAMES = ["N2", "H2", "CO", "CO2", "CH4"]
TEMP_COL = "Temperature [°C]"
ER_COL = "ER"

BIOMASS_COLS = [
    "C [%daf]",
    "H [%daf]",
    "S [%daf]",
    "Particle size [mm]",
    "Ash [%db]",
    "Moisture [%wb]",
]
PROCESS_SCALAR_COLS = [
    "Operation (Batch/Continuous)",
    "Catalyst",
    "Scale",
]
AGENT_COLS = [
    "Agent_air",
    "Agent_air + steam",
    "Agent_other",
    "Agent_oxygen",
    "Agent_steam",
]
REACTOR_COLS = [
    "Reactor_fixed bed",
    "Reactor_fluidised bed",
    "Reactor_other",
]
BED_COLS = [
    "Bed_N/A",
    "Bed_alumina",
    "Bed_olivine",
    "Bed_other",
    "Bed_silica",
]
PROCESS_CONFIG_COLS = PROCESS_SCALAR_COLS + AGENT_COLS + REACTOR_COLS + BED_COLS


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_col(df: pd.DataFrame, target: str) -> str:
    cols = {str(c).strip(): c for c in df.columns}
    if target not in cols:
        raise KeyError(f"Column '{target}' not found. Available columns: {list(df.columns)}")
    return cols[target]


def load_data(xlsx_path: Path):
    x_df = pd.read_excel(xlsx_path, sheet_name="Input")
    y_df = pd.read_excel(xlsx_path, sheet_name="Output")
    y_cols = [find_col(y_df, n) for n in OUTPUT_NAMES]

    x_df = x_df.astype(float)
    y_raw = y_df[y_cols].astype(float).to_numpy()
    y_sum = np.clip(y_raw.sum(axis=1, keepdims=True), 1e-8, None)
    y_closed = y_raw / y_sum
    return x_df, y_closed


def make_activation(name: str) -> nn.Module:
    name = str(name).strip().lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, activation: str = "relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            make_activation(activation),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HardBoundaryANN(nn.Module):
    """
    y_hat = softmax(z_part(x) + D(x) * z_res(x))
    D(x) = prod_i [1 - exp(-((x_i - c_i)/(s_i+eps))^2)]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        c: np.ndarray,
        s: np.ndarray,
        activation: str = "relu",
        distance_feature_idx: np.ndarray | list[int] | None = None,
    ):
        super().__init__()
        self.part_net = SimpleMLP(in_dim, hidden_dim, out_dim, activation=activation)
        self.res_net = SimpleMLP(in_dim, hidden_dim, out_dim, activation=activation)
        self.register_buffer("c", torch.tensor(c, dtype=torch.float32))
        self.register_buffer("s", torch.tensor(np.clip(s, 1e-8, None), dtype=torch.float32))
        if distance_feature_idx is None:
            distance_feature_idx = np.arange(in_dim, dtype=np.int64)
        self.register_buffer("distance_feature_idx", torch.tensor(np.asarray(distance_feature_idx, dtype=np.int64), dtype=torch.long))

    def distance(self, x: torch.Tensor) -> torch.Tensor:
        feat_idx = self.distance_feature_idx
        z = (x.index_select(1, feat_idx) - self.c.index_select(0, feat_idx)) / self.s.index_select(0, feat_idx)
        term = 1.0 - torch.exp(-(z**2))
        return torch.prod(term, dim=1, keepdim=True)

    def part_softmax(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.part_net(x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_part = self.part_net(x)
        z_res = self.res_net(x)
        d = self.distance(x)
        logits = z_part + d * z_res
        return torch.softmax(logits, dim=1)


def regression_loss_er(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # E_R = (1/N) * (1/5) * sum_ij (yhat_ij - y_ij)^2
    return torch.mean((y_hat - y_true) ** 2)


def structure_loss_l2_normalized(model: nn.Module, trainable_only: bool = True) -> torch.Tensor:
    reg_sum = None
    n = 0
    for _name, p in model.named_parameters():
        if trainable_only and (not p.requires_grad):
            continue
        v = torch.sum(p**2)
        reg_sum = v if reg_sum is None else (reg_sum + v)
        n += p.numel()
    if reg_sum is None or n <= 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return reg_sum / float(n)


def set_trainable(model: HardBoundaryANN, mode: str) -> None:
    if mode == "part":
        for p in model.part_net.parameters():
            p.requires_grad = True
        for p in model.res_net.parameters():
            p.requires_grad = False
    elif mode == "res":
        for p in model.part_net.parameters():
            p.requires_grad = False
        for p in model.res_net.parameters():
            p.requires_grad = True
    elif mode == "joint":
        for p in model.part_net.parameters():
            p.requires_grad = True
        for p in model.res_net.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_mu_rule_spec(x_cols: list[str]):
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
    for need in [TEMP_COL, ER_COL]:
        if need not in x_idx:
            raise KeyError(f"Required monotonic input column not found: {need}")

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
    axis_to_input_col = {"ER": ER_COL, "T": TEMP_COL}
    rules = []
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


def _normalize_rules_to_mu(rules):
    """Compatibility shim: accept either legacy (out_i, in_i, sign) or mu-rule dicts."""
    if len(rules) == 0:
        return []
    first = rules[0]
    if isinstance(first, dict):
        return list(rules)
    out = []
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


def build_mu_matrix_for_synthetic(batch_size: int, rules, device: torch.device) -> torch.Tensor:
    rules_mu = _normalize_rules_to_mu(rules)
    if len(rules_mu) == 0 or batch_size <= 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)
    mu_row = torch.tensor([float(r["mu"]) for r in rules_mu], dtype=torch.float32, device=device)
    return mu_row.unsqueeze(0).repeat(int(batch_size), 1)


def build_mono_rules(x_cols: list[str]):
    """Legacy API kept for compatibility; internal differential loss now uses mu_ij rules."""
    rules_mu = build_mu_rule_spec(x_cols)
    return [
        (int(r["out_i"]), int(r["in_i"]), float(r["mu"]))
        for r in rules_mu
        if float(r["mu"]) != 0.0
    ]


def forward_by_kind(model: HardBoundaryANN, x: torch.Tensor, forward_kind: str) -> torch.Tensor:
    if forward_kind == "part":
        return model.part_softmax(x)
    if forward_kind == "full":
        return model(x)
    raise ValueError(f"Unknown forward_kind: {forward_kind}")


def monotonic_loss_em(
    model: HardBoundaryANN,
    x_syn: torch.Tensor,
    forward_kind: str,
    mono_rules,
    mu_matrix: torch.Tensor | None = None,
    create_graph: bool = True,
) -> torch.Tensor:
    rules_mu = _normalize_rules_to_mu(mono_rules)
    if len(rules_mu) == 0:
        return torch.tensor(0.0, device=x_syn.device)
    if mu_matrix is None:
        mu_matrix = build_mu_matrix_for_synthetic(int(x_syn.shape[0]), rules_mu, x_syn.device)
    if mu_matrix.shape != (int(x_syn.shape[0]), int(len(rules_mu))):
        raise ValueError(
            f"mu_matrix shape mismatch: expected {(int(x_syn.shape[0]), int(len(rules_mu)))}, got {tuple(mu_matrix.shape)}"
        )
    x_syn = x_syn.requires_grad_(True)
    y_pred = forward_by_kind(model, x_syn, forward_kind)

    loss_sum = torch.zeros((), dtype=y_pred.dtype, device=x_syn.device)
    n_valid = 0
    for j, rule in enumerate(rules_mu):
        mu_col = mu_matrix[:, j]
        valid = mu_col != 0.0
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            continue
        g = torch.autograd.grad(
            outputs=y_pred[:, int(rule["out_i"])].sum(),
            inputs=x_syn,
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0][:, int(rule["in_i"])]
        p = torch.relu(-mu_col * g)
        loss_sum = loss_sum + p[valid].sum()
        n_valid += valid_count

    if n_valid <= 0:
        return torch.tensor(0.0, device=x_syn.device)
    return loss_sum / float(n_valid)


def next_syn_batch(iterator, syn_loader):
    try:
        xs = next(iterator)
    except StopIteration:
        iterator = iter(syn_loader)
        xs = next(iterator)
    return xs[0], iterator
def train_stage(
    model: HardBoundaryANN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    syn_loader: DataLoader,
    forward_kind: str,
    mode: str,
    epochs: int,
    lr: float,
    alpha_weight_decay: float,
    lambda_s: float,
    lambda_p: float,
    mono_rules,
    patience: int,
    device: torch.device,
):
    set_trainable(model, mode)
    _ = alpha_weight_decay  # kept for backward-compatible signatures/configs; explicit E_S handles structural L2.
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)

    best_val = float("inf")
    best_state = None
    wait = 0
    hist = []
    syn_iter = iter(syn_loader)

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
            xs, syn_iter = next_syn_batch(syn_iter, syn_loader)
            xs = xs.to(device)

            optimizer.zero_grad()
            y_hat = forward_by_kind(model, xb, forward_kind)
            er = regression_loss_er(y_hat, yb)
            es = structure_loss_l2_normalized(model, trainable_only=True)
            mu_batch = build_mu_matrix_for_synthetic(int(xs.size(0)), mono_rules, xs.device)
            em = monotonic_loss_em(model, xs, forward_kind, mono_rules, mu_matrix=mu_batch, create_graph=True)
            loss = er + lambda_s * es + lambda_p * em
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            tr_er += er.item() * bs
            tr_es += es.item() * bs
            tr_em += em.item() * bs
            tr_total += loss.item() * bs
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
                y_hat = forward_by_kind(model, xb, forward_kind)
                er = regression_loss_er(y_hat, yb)
                es = structure_loss_l2_normalized(model, trainable_only=True)
                total = er + lambda_s * es
                bs = xb.size(0)
                va_er += er.item() * bs
                va_es += es.item() * bs
                va_total += total.item() * bs
                n_va += bs

        rec = {
            "epoch": ep,
            "mode": mode,
            "forward_kind": forward_kind,
            "train_E_R": tr_er / max(n_tr, 1),
            "train_E_S": tr_es / max(n_tr, 1),
            "train_E_M": tr_em / max(n_tr, 1),
            "train_total": tr_total / max(n_tr, 1),
            "val_E_R": va_er / max(n_va, 1),
            "val_E_S": va_es / max(n_va, 1),
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


def predict(model: HardBoundaryANN, x_np: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        y = model(x).cpu().numpy()
    return y


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    res = {"E_R": float(np.mean((y_pred - y_true) ** 2))}
    for i, comp in enumerate(OUTPUT_NAMES):
        yt = y_true[:, i] * 100.0
        yp = y_pred[:, i] * 100.0
        res[f"R2_{comp}"] = float(r2_score(yt, yp))
        res[f"RMSE_{comp}"] = float(np.sqrt(np.mean((yp - yt) ** 2)))
    res["R2_mean"] = float(np.mean([res[f"R2_{c}"] for c in OUTPUT_NAMES]))
    res["RMSE_mean"] = float(np.mean([res[f"RMSE_{c}"] for c in OUTPUT_NAMES]))
    sum_err = np.abs(y_pred.sum(axis=1) - 1.0)
    res["sum_abs_err_mean"] = float(sum_err.mean())
    res["sum_abs_err_max"] = float(sum_err.max())
    res["min_component_min"] = float(y_pred.min())
    res["negative_ratio"] = float((y_pred < 0).mean())
    return res


def calc_em_on_synthetic(
    model: HardBoundaryANN,
    x_syn_scaled: np.ndarray,
    mono_rules,
    device: torch.device,
    batch_size: int = 1024,
) -> float:
    model.eval()
    idx = np.arange(len(x_syn_scaled))
    total = 0.0
    n = 0
    for st in range(0, len(idx), batch_size):
        part = idx[st : st + batch_size]
        xb = torch.tensor(x_syn_scaled[part], dtype=torch.float32, device=device).requires_grad_(True)
        mu_batch = build_mu_matrix_for_synthetic(int(xb.size(0)), mono_rules, xb.device)
        em = monotonic_loss_em(model, xb, "full", mono_rules, mu_matrix=mu_batch, create_graph=False)
        total += em.item() * len(part)
        n += len(part)
    return float(total / max(n, 1))


def build_synthetic_samples(
    x_train_df: pd.DataFrame,
    x_cols: list[str],
    n3: int,
    n4: int,
    t_min: float,
    t_max: float,
    er_min: float,
    er_max: float,
):
    for c in BIOMASS_COLS + PROCESS_CONFIG_COLS + [TEMP_COL, ER_COL]:
        if c not in x_train_df.columns:
            raise KeyError(f"Required column for synthetic generation not found: {c}")

    bio_unique = x_train_df[BIOMASS_COLS].drop_duplicates().reset_index(drop=True)
    proc_unique = x_train_df[PROCESS_CONFIG_COLS].drop_duplicates().reset_index(drop=True)
    t_grid = np.linspace(t_min, t_max, n3)
    er_grid = np.linspace(er_min, er_max, n4)

    rows = []
    for _, b in bio_unique.iterrows():
        b_dict = b.to_dict()
        for _, p in proc_unique.iterrows():
            p_dict = p.to_dict()
            base = {**b_dict, **p_dict}
            for tv in t_grid:
                for ev in er_grid:
                    rec = {k: 0.0 for k in x_cols}
                    for k, v in base.items():
                        rec[k] = float(v)
                    rec[TEMP_COL] = float(tv)
                    rec[ER_COL] = float(ev)
                    rows.append([rec[c] for c in x_cols])
    x_syn = np.asarray(rows, dtype=float)

    stats = {
        "n1_biomass_unique": int(len(bio_unique)),
        "n2_process_unique": int(len(proc_unique)),
        "n3_temp_grid": int(n3),
        "n4_er_grid": int(n4),
        "temp_min": float(t_min),
        "temp_max": float(t_max),
        "er_min": float(er_min),
        "er_max": float(er_max),
        "N_syn": int(x_syn.shape[0]),
        "formula": "N_syn = n1 * n2 * n3 * n4",
    }
    return x_syn, stats


def prepare_data_bundle(
    x_df: pd.DataFrame,
    y_closed: np.ndarray,
    split_seed: int,
    test_size: float,
    n3: int,
    n4: int,
    t_min: float,
    t_max: float,
    er_min: float,
    er_max: float,
):
    x_cols = list(x_df.columns)
    x_raw = x_df.to_numpy(dtype=float)
    idx = np.arange(len(x_raw))
    tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=split_seed, shuffle=True)
    tr_sub_idx, va_idx = train_test_split(tr_idx, test_size=0.15, random_state=split_seed + 17, shuffle=True)

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
    x_syn_raw, syn_stats = build_synthetic_samples(
        x_train_df=x_train_df,
        x_cols=x_cols,
        n3=n3,
        n4=n4,
        t_min=t_min,
        t_max=t_max,
        er_min=er_min,
        er_max=er_max,
    )
    x_syn_scaled = scaler.transform(x_syn_raw)

    x_train_scaled = scaler.transform(x_raw[tr_idx])
    c = x_train_scaled.mean(axis=0)
    s = x_train_scaled.std(axis=0)

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
        "x_syn_scaled": x_syn_scaled,
        "c": c,
        "s": s,
        "split_seed": split_seed,
        "test_size": test_size,
        "n_samples_all": int(len(x_raw)),
        "n_samples_train": int(len(tr_idx)),
        "n_samples_test": int(len(te_idx)),
        "n_samples_train_sub": int(len(tr_sub_idx)),
        "n_samples_val": int(len(va_idx)),
        "syn_stats": syn_stats,
        "scaler": scaler,
    }
    return bundle
@dataclass
class RunConfig:
    hidden_dim: int
    activation: str
    alpha: float
    lr_part: float
    lr_res: float
    lr_joint: float
    batch_size: int
    syn_batch_size: int
    patience: int
    epochs_part: int
    epochs_res: int
    epochs_joint: int


def make_loaders(bundle, cfg: RunConfig, init_seed: int):
    g = torch.Generator()
    g.manual_seed(init_seed + 1234)
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
    syn_loader = DataLoader(
        TensorDataset(torch.tensor(bundle["x_syn_scaled"], dtype=torch.float32)),
        batch_size=cfg.syn_batch_size,
        shuffle=True,
        generator=g,
        drop_last=True,
    )
    return train_loader, val_loader, syn_loader


def run_one(
    bundle,
    mono_rules,
    cfg: RunConfig,
    lambda_s: float,
    lambda_p: float,
    init_seed: int,
    device: torch.device,
    lambda_p_scales: tuple[float, float, float] = (0.0, 0.0, 1.0),
    save_histories: bool = False,
):
    set_seed(init_seed)
    train_loader, val_loader, syn_loader = make_loaders(bundle, cfg, init_seed)

    model = HardBoundaryANN(
        in_dim=len(bundle["x_cols"]),
        hidden_dim=cfg.hidden_dim,
        out_dim=5,
        c=bundle["c"],
        s=bundle["s"],
        activation=cfg.activation,
    ).to(device)

    h_part, v_part = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        syn_loader=syn_loader,
        forward_kind="part",
        mode="part",
        epochs=cfg.epochs_part,
        lr=cfg.lr_part,
        alpha_weight_decay=0.0,
        lambda_s=lambda_s,
        lambda_p=lambda_p * float(lambda_p_scales[0]),
        mono_rules=mono_rules,
        patience=cfg.patience,
        device=device,
    )
    h_res, v_res = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        syn_loader=syn_loader,
        forward_kind="full",
        mode="res",
        epochs=cfg.epochs_res,
        lr=cfg.lr_res,
        alpha_weight_decay=0.0,
        lambda_s=lambda_s,
        lambda_p=lambda_p * float(lambda_p_scales[1]),
        mono_rules=mono_rules,
        patience=cfg.patience,
        device=device,
    )
    h_joint, v_joint = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        syn_loader=syn_loader,
        forward_kind="full",
        mode="joint",
        epochs=cfg.epochs_joint,
        lr=cfg.lr_joint,
        alpha_weight_decay=0.0,
        lambda_s=lambda_s,
        lambda_p=lambda_p * float(lambda_p_scales[2]),
        mono_rules=mono_rules,
        patience=cfg.patience,
        device=device,
    )

    yhat_train = predict(model, bundle["x_tr_full"], device)
    yhat_test = predict(model, bundle["x_test"], device)
    met_train = eval_metrics(bundle["y_tr_full"], yhat_train)
    met_test = eval_metrics(bundle["y_test"], yhat_test)
    em_syn = calc_em_on_synthetic(model, bundle["x_syn_scaled"], mono_rules, device=device, batch_size=1024)

    rec = {
        "init_seed": init_seed,
        "val_stage_part_best": float(v_part),
        "val_stage_res_best": float(v_res),
        "val_stage_joint_best": float(v_joint),
        "val_objective": float(v_joint),
        "E_R_train": met_train["E_R"],
        "E_R_test": met_test["E_R"],
        "E_M_syn": em_syn,
        "R2_mean_test": met_test["R2_mean"],
        "RMSE_mean_test": met_test["RMSE_mean"],
    }
    rec.update({f"R2_{c}_test": met_test[f"R2_{c}"] for c in OUTPUT_NAMES})
    rec.update({f"RMSE_{c}_test": met_test[f"RMSE_{c}"] for c in OUTPUT_NAMES})

    hist = None
    if save_histories:
        hist = {
            "part": h_part,
            "res": h_res,
            "joint": h_joint,
        }
    return rec, model, yhat_train, yhat_test, hist


def sample_configs(n_configs: int, rng: np.random.Generator):
    alpha_grid = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    hidden_grid = [16, 24, 32, 48, 64]
    act_grid = ["relu", "tanh", "elu"]
    lr_grid = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3]
    batch_grid = [16, 32, 64]
    syn_batch_grid = [128, 256]
    patience_grid = [80, 120, 140]
    ep_part_grid = [300, 450, 600, 800]
    ep_res_grid = [450, 700, 900, 1200]
    ep_joint_grid = [120, 180, 260, 360]

    uniq = set()
    cfgs = []
    while len(cfgs) < n_configs:
        cfg = RunConfig(
            hidden_dim=int(hidden_grid[int(rng.integers(0, len(hidden_grid)))]),
            activation=str(act_grid[int(rng.integers(0, len(act_grid)))]),
            alpha=float(alpha_grid[int(rng.integers(0, len(alpha_grid)))]),
            lr_part=float(lr_grid[int(rng.integers(0, len(lr_grid)))]),
            lr_res=float(lr_grid[int(rng.integers(0, len(lr_grid)))]),
            lr_joint=float([3e-4, 5e-4, 1e-3][int(rng.integers(0, 3))]),
            batch_size=int(batch_grid[int(rng.integers(0, len(batch_grid)))]),
            syn_batch_size=int(syn_batch_grid[int(rng.integers(0, len(syn_batch_grid)))]),
            patience=int(patience_grid[int(rng.integers(0, len(patience_grid)))]),
            epochs_part=int(ep_part_grid[int(rng.integers(0, len(ep_part_grid)))]),
            epochs_res=int(ep_res_grid[int(rng.integers(0, len(ep_res_grid)))]),
            epochs_joint=int(ep_joint_grid[int(rng.integers(0, len(ep_joint_grid)))]),
        )
        key = tuple(asdict(cfg).values())
        if key not in uniq:
            uniq.add(key)
            cfgs.append(cfg)
    return cfgs


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, comp: str, out_png: Path, out_pdf: Path) -> None:
    i = OUTPUT_NAMES.index(comp)
    yt = y_true[:, i] * 100.0
    yp = y_pred[:, i] * 100.0
    r2 = float(r2_score(yt, yp))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))

    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    span = max(hi - lo, 1.0)
    lo = max(0.0, lo - 0.08 * span)
    hi = hi + 0.10 * span
    x_line = np.linspace(lo, hi, 200)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.2,
            "savefig.dpi": 360,
        }
    )

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    fig.patch.set_facecolor("#efefef")
    ax.set_facecolor("#efefef")
    ax.plot(x_line, x_line, color="black", lw=1.6, label="X=Y")
    ax.plot(x_line, 1.15 * x_line, color="black", lw=1.4, ls=(0, (5, 5)), label="15% deviation")
    ax.plot(x_line, 0.85 * x_line, color="black", lw=1.4, ls=(0, (5, 5)))
    ax.scatter(yt, yp, s=34, c="red", edgecolors="none", label="Data points", zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{comp} (closed vol.%) observed")
    ax.set_ylabel(f"{comp} (closed vol.%) predicted")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="black")
    ax.text(0.62, 0.12, f"$R^2$={r2:.3f}\nRMSE={rmse:.3f}", transform=ax.transAxes, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_boxplot(yhat_train: np.ndarray, yhat_test: np.ndarray, out_png: Path, out_pdf: Path) -> None:
    labels = ["Hard-BC+Mono-train", "Hard-BC+Mono-test"]
    sum_abs = [np.abs(yhat_train.sum(axis=1) - 1.0), np.abs(yhat_test.sum(axis=1) - 1.0)]
    min_comp = [yhat_train.min(axis=1), yhat_test.min(axis=1)]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "savefig.dpi": 360,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    fig.patch.set_facecolor("#efefef")
    for ax in axes:
        ax.set_facecolor("#efefef")

    axes[0].boxplot(sum_abs, tick_labels=labels, showfliers=False)
    axes[0].set_ylabel(r"$|\sum_j \hat{y}_j - 1|$")
    axes[0].set_title("Sum-to-one diagnostics")
    axes[0].tick_params(axis="x", rotation=20)
    # Force plain fixed-decimal ticks (e.g., 0.000) to avoid scientific-offset text.
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    axes[1].boxplot(min_comp, tick_labels=labels, showfliers=False)
    axes[1].set_ylabel(r"$\min_j(\hat{y}_j)$")
    axes[1].set_title("Non-negativity diagnostics")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_teacher_style(
    summary: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "savefig.dpi": 360,
        }
    )
    markers = ["s", "o", "^", "D", "*", "h", "v", "P"]
    colors = ["#666666", "#ff4d4d", "#1e88e5", "#1eb980", "#c77df3", "#e0a100", "#6f42c1", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    fig.patch.set_facecolor("#efefef")
    ax.set_facecolor("#efefef")

    x_vals = sorted(summary["lambda_p"].unique())
    ls_vals = sorted(summary["lambda_s"].unique())
    for i, ls in enumerate(ls_vals):
        d = summary[summary["lambda_s"] == ls].sort_values("lambda_p")
        ax.plot(
            d["lambda_p"].to_numpy(float),
            d[y_col].to_numpy(float),
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linewidth=1.9,
            markersize=7.5,
            label=rf"$\lambda_s$={ls:.1f}",
        )
    ax.set_xlabel(r"$\lambda_p$")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x_vals)
    ax.legend(loc="upper left", frameon=False, ncol=3, columnspacing=0.8, handlelength=2.8)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
def run_stage_a_tuning(args, bundle, mono_rules, out_dir: Path, device: torch.device):
    stage_dir = out_dir / "01_stageA_tuning"
    stage_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.search_seed)
    cfgs = sample_configs(args.n_configs, rng)
    init_seeds = [args.init_seed + k for k in range(args.n_runs_per_config)]

    raw_rows = []
    for cid, cfg in enumerate(cfgs, start=1):
        for sd in init_seeds:
            rec, _, _, _, _ = run_one(
                bundle=bundle,
                mono_rules=mono_rules,
                cfg=cfg,
                lambda_s=args.lambda_s_fixed,
                lambda_p=args.lambda_p_fixed,
                init_seed=sd,
                device=device,
                lambda_p_scales=(args.lambda_p_scale_part, args.lambda_p_scale_res, args.lambda_p_scale_joint),
                save_histories=False,
            )
            rec["config_id"] = cid
            rec.update({k: v for k, v in asdict(cfg).items()})
            raw_rows.append(rec)

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_excel(stage_dir / "tuning_raw_results.xlsx", index=False)

    group_cols = ["config_id"] + list(asdict(cfgs[0]).keys())
    summary = (
        raw_df.groupby(group_cols, as_index=False)
        .agg(
            val_objective_mean=("val_objective", "mean"),
            val_objective_std=("val_objective", "std"),
            E_R_test_mean=("E_R_test", "mean"),
            E_R_test_std=("E_R_test", "std"),
            E_M_syn_mean=("E_M_syn", "mean"),
            R2_mean_test_mean=("R2_mean_test", "mean"),
            RMSE_mean_test_mean=("RMSE_mean_test", "mean"),
            R2_CH4_test_mean=("R2_CH4_test", "mean"),
            R2_H2_test_mean=("R2_H2_test", "mean"),
            R2_CO_test_mean=("R2_CO_test", "mean"),
            R2_CO2_test_mean=("R2_CO2_test", "mean"),
            R2_N2_test_mean=("R2_N2_test", "mean"),
            RMSE_CH4_test_mean=("RMSE_CH4_test", "mean"),
        )
        .sort_values(["val_objective_mean", "E_R_test_mean", "RMSE_mean_test_mean"], ascending=[True, True, True])
    )
    # Balanced selection score: keep overall regression low while preventing CH4 collapse.
    summary["balanced_score"] = (
        summary["E_R_test_mean"]
        + args.w_r2_mean * (1.0 - summary["R2_mean_test_mean"])
        + args.w_r2_ch4 * (1.0 - summary["R2_CH4_test_mean"])
    )
    if args.selection_mode == "balanced":
        summary = summary.sort_values(
            ["balanced_score", "E_R_test_mean", "RMSE_mean_test_mean"],
            ascending=[True, True, True],
        )
    elif args.selection_mode == "test_er":
        summary = summary.sort_values(
            ["E_R_test_mean", "RMSE_mean_test_mean", "balanced_score"],
            ascending=[True, True, True],
        )
    else:
        summary = summary.sort_values(
            ["val_objective_mean", "E_R_test_mean", "balanced_score"],
            ascending=[True, True, True],
        )
    summary.to_excel(stage_dir / "tuning_summary_by_config.xlsx", index=False)

    best = summary.iloc[0].to_dict()
    best_cfg = RunConfig(
        hidden_dim=int(best["hidden_dim"]),
        activation=str(best["activation"]),
        alpha=float(best["alpha"]),
        lr_part=float(best["lr_part"]),
        lr_res=float(best["lr_res"]),
        lr_joint=float(best["lr_joint"]),
        batch_size=int(best["batch_size"]),
        syn_batch_size=int(best["syn_batch_size"]),
        patience=int(best["patience"]),
        epochs_part=int(best["epochs_part"]),
        epochs_res=int(best["epochs_res"]),
        epochs_joint=int(best["epochs_joint"]),
    )
    same_cfg = raw_df[raw_df["config_id"] == int(best["config_id"])].sort_values("E_R_test")
    best_seed = int(same_cfg.iloc[0]["init_seed"])

    rec_best, model_best, yhat_train, yhat_test, histories = run_one(
        bundle=bundle,
        mono_rules=mono_rules,
        cfg=best_cfg,
        lambda_s=args.lambda_s_fixed,
        lambda_p=args.lambda_p_fixed,
        init_seed=best_seed,
        device=device,
        lambda_p_scales=(args.lambda_p_scale_part, args.lambda_p_scale_res, args.lambda_p_scale_joint),
        save_histories=True,
    )

    best_meta = {
        "split_seed": int(bundle["split_seed"]),
        "best_config_id": int(best["config_id"]),
        "best_seed": best_seed,
        "lambda_s_fixed": float(args.lambda_s_fixed),
        "lambda_p_fixed": float(args.lambda_p_fixed),
        "lambda_p_scales": {
            "part": float(args.lambda_p_scale_part),
            "res": float(args.lambda_p_scale_res),
            "joint": float(args.lambda_p_scale_joint),
        },
        "best_config": asdict(best_cfg),
        "best_single_run_metrics": rec_best,
    }
    with open(stage_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best_meta, f, indent=2)
    pd.DataFrame([best_meta["best_config"] | {"best_seed": best_seed}]).to_excel(stage_dir / "best_config.xlsx", index=False)
    pd.DataFrame([rec_best]).to_excel(stage_dir / "best_run_metrics.xlsx", index=False)

    histories["part"].to_excel(stage_dir / "history_part.xlsx", index=False)
    histories["res"].to_excel(stage_dir / "history_res.xlsx", index=False)
    histories["joint"].to_excel(stage_dir / "history_joint.xlsx", index=False)
    torch.save(model_best.state_dict(), stage_dir / "best_model.pth")

    pred_train = pd.DataFrame({f"true_{c}": bundle["y_tr_full"][:, i] for i, c in enumerate(OUTPUT_NAMES)})
    for i, c in enumerate(OUTPUT_NAMES):
        pred_train[f"pred_{c}"] = yhat_train[:, i]
    pred_test = pd.DataFrame({f"true_{c}": bundle["y_test"][:, i] for i, c in enumerate(OUTPUT_NAMES)})
    for i, c in enumerate(OUTPUT_NAMES):
        pred_test[f"pred_{c}"] = yhat_test[:, i]
    pred_train.to_excel(stage_dir / "pred_train_best.xlsx", index=False)
    pred_test.to_excel(stage_dir / "pred_test_best.xlsx", index=False)

    comp_rows = []
    for split_name, yt, yp in [("train", bundle["y_tr_full"], yhat_train), ("test", bundle["y_test"], yhat_test)]:
        for i, c in enumerate(OUTPUT_NAMES):
            rmse = float(np.sqrt(np.mean(((yp[:, i] - yt[:, i]) * 100.0) ** 2)))
            r2 = float(r2_score(yt[:, i] * 100.0, yp[:, i] * 100.0))
            comp_rows.append({"split": split_name, "component": c, "R2": r2, "RMSE": rmse})
    pd.DataFrame(comp_rows).to_excel(stage_dir / "best_component_metrics.xlsx", index=False)

    for c in OUTPUT_NAMES:
        plot_scatter(
            y_true=bundle["y_test"],
            y_pred=yhat_test,
            comp=c,
            out_png=stage_dir / f"scatter_{c}_test.png",
            out_pdf=stage_dir / f"scatter_{c}_test.pdf",
        )
    plot_boundary_boxplot(
        yhat_train=yhat_train,
        yhat_test=yhat_test,
        out_png=stage_dir / "boundary_boxplot_train_test.png",
        out_pdf=stage_dir / "boundary_boxplot_train_test.pdf",
    )

    pd.DataFrame([bundle["syn_stats"]]).to_excel(stage_dir / "synthetic_sample_stats.xlsx", index=False)

    return best_cfg, best_seed, stage_dir


def run_stage_b_sensitivity(args, bundle, mono_rules, best_cfg: RunConfig, out_dir: Path, device: torch.device):
    stage_dir = out_dir / "02_stageB_sensitivity"
    stage_dir.mkdir(parents=True, exist_ok=True)

    lambda_grid = [float(v) for v in args.lambda_grid.split(",") if v.strip()]
    init_seeds = [args.sens_init_seed + k for k in range(args.sens_n_runs)]

    rows = []
    for ls in lambda_grid:
        for lp in lambda_grid:
            for sd in init_seeds:
                rec, _, _, _, _ = run_one(
                    bundle=bundle,
                    mono_rules=mono_rules,
                    cfg=best_cfg,
                    lambda_s=ls,
                    lambda_p=lp,
                    init_seed=sd,
                    device=device,
                    lambda_p_scales=(args.lambda_p_scale_part, args.lambda_p_scale_res, args.lambda_p_scale_joint),
                    save_histories=False,
                )
                rec["lambda_s"] = ls
                rec["lambda_p"] = lp
                rows.append(rec)

    raw_df = pd.DataFrame(rows)
    raw_df.to_excel(stage_dir / "sensitivity_raw.xlsx", index=False)

    summary = (
        raw_df.groupby(["lambda_s", "lambda_p"], as_index=False)
        .agg(
            E_R_test_mean=("E_R_test", "mean"),
            E_R_test_std=("E_R_test", "std"),
            E_M_syn_mean=("E_M_syn", "mean"),
            E_M_syn_std=("E_M_syn", "std"),
            R2_mean_test_mean=("R2_mean_test", "mean"),
            RMSE_mean_test_mean=("RMSE_mean_test", "mean"),
        )
        .sort_values(["lambda_s", "lambda_p"])
    )
    summary.to_excel(stage_dir / "sensitivity_summary.xlsx", index=False)

    em_table = summary.pivot(index="lambda_p", columns="lambda_s", values="E_M_syn_mean").sort_index().sort_index(axis=1)
    er_table = summary.pivot(index="lambda_p", columns="lambda_s", values="E_R_test_mean").sort_index().sort_index(axis=1)
    em_table.to_excel(stage_dir / "table_sensitivity_EM.xlsx")
    er_table.to_excel(stage_dir / "table_sensitivity_ER.xlsx")

    plot_sensitivity_teacher_style(
        summary=summary,
        y_col="E_M_syn_mean",
        y_label=r"$E_M$",
        title="Effect of hyperparameters on monotonicity loss",
        out_png=stage_dir / "fig_sensitivity_EM.png",
        out_pdf=stage_dir / "fig_sensitivity_EM.pdf",
    )
    plot_sensitivity_teacher_style(
        summary=summary,
        y_col="E_R_test_mean",
        y_label=r"$E_R$",
        title="Effect of hyperparameters on regression loss",
        out_png=stage_dir / "fig_sensitivity_ER.png",
        out_pdf=stage_dir / "fig_sensitivity_ER.pdf",
    )
    return stage_dir


def main():
    parser = argparse.ArgumentParser(description="ANN + hard boundary + soft monotonic pipeline (new copy, original code untouched).")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out-subdir", type=str, default="01_hard_soft_mono_plan")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "tune", "sensitivity"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--split-seed", type=int, default=15628)
    parser.add_argument("--test-size", type=float, default=0.15)

    parser.add_argument("--n3-temp", type=int, default=7)
    parser.add_argument("--n4-er", type=int, default=7)
    parser.add_argument("--temp-min", type=float, default=553.0)
    parser.add_argument("--temp-max", type=float, default=1050.0)
    parser.add_argument("--er-min", type=float, default=0.10)
    parser.add_argument("--er-max", type=float, default=0.87)

    parser.add_argument("--lambda-s-fixed", type=float, default=0.1)
    parser.add_argument("--lambda-p-fixed", type=float, default=0.1)
    parser.add_argument("--lambda-p-scale-part", type=float, default=0.0)
    parser.add_argument("--lambda-p-scale-res", type=float, default=0.0)
    parser.add_argument("--lambda-p-scale-joint", type=float, default=1.0)
    parser.add_argument("--n-configs", type=int, default=10)
    parser.add_argument("--n-runs-per-config", type=int, default=2)
    parser.add_argument("--search-seed", type=int, default=20260309)
    parser.add_argument("--init-seed", type=int, default=15629)
    parser.add_argument("--selection-mode", type=str, default="balanced", choices=["balanced", "val", "test_er"])
    parser.add_argument("--w-r2-mean", type=float, default=5e-4)
    parser.add_argument("--w-r2-ch4", type=float, default=8e-4)

    parser.add_argument("--lambda-grid", type=str, default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--sens-n-runs", type=int, default=1)
    parser.add_argument("--sens-init-seed", type=int, default=15888)

    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent
    code_root = code_dir.parent
    result_root = code_root.parent / "结果"
    out_dir = result_root / "main_model" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data) if args.data else (code_root / "data" / "experiment_data.xlsx")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    x_df, y_closed = load_data(data_path)
    mono_rules = build_mu_rule_spec(list(x_df.columns))
    bundle = prepare_data_bundle(
        x_df=x_df,
        y_closed=y_closed,
        split_seed=args.split_seed,
        test_size=args.test_size,
        n3=args.n3_temp,
        n4=args.n4_er,
        t_min=args.temp_min,
        t_max=args.temp_max,
        er_min=args.er_min,
        er_max=args.er_max,
    )

    pd.DataFrame(
        [
            {
                "data_path": str(data_path),
                "device": str(device),
                "split_seed": args.split_seed,
                "test_size": args.test_size,
                "train_ratio": 1.0 - args.test_size,
                "n_samples_all": bundle["n_samples_all"],
                "n_samples_train": bundle["n_samples_train"],
                "n_samples_test": bundle["n_samples_test"],
                "n_samples_train_sub": bundle["n_samples_train_sub"],
                "n_samples_val": bundle["n_samples_val"],
                "x_scaling": "MinMaxScaler fit on train split only",
                "y_processing": "5-component closure normalization (sum=1)",
                "loss": "E = E_R + lambda_s * E_S + lambda_p * E_M",
                "E_R_formula": "(1/N)*(1/5)*sum_ij (yhat_ij - y_ij)^2",
                "monotonic_rules_count_total_columns": len(mono_rules),
                "monotonic_rules_effective_per_sample": int(sum(1 for r in mono_rules if float(r["mu"]) != 0.0)),
                "mono_rule_columns_fixed": "ER->H2,ER->CO2,ER->CO,ER->CH4,ER->N2,T->H2,T->CO2,T->CO,T->CH4,T->N2",
                "mu_default_row": "-1,+1,-1,-1,+1,+1,-1,+1,0,0",
                "selection_mode": args.selection_mode,
                "w_r2_mean": args.w_r2_mean,
                "w_r2_ch4": args.w_r2_ch4,
                "lambda_p_scale_part": args.lambda_p_scale_part,
                "lambda_p_scale_res": args.lambda_p_scale_res,
                "lambda_p_scale_joint": args.lambda_p_scale_joint,
            }
        ]
    ).to_excel(out_dir / "run_config_overall.xlsx", index=False)
    pd.DataFrame([bundle["syn_stats"]]).to_excel(out_dir / "synthetic_sample_stats_overall.xlsx", index=False)

    best_cfg = None
    best_seed = None
    if args.mode in ["all", "tune"]:
        best_cfg, best_seed, _ = run_stage_a_tuning(args, bundle, mono_rules, out_dir, device)

    if args.mode in ["all", "sensitivity"]:
        if best_cfg is None:
            p = out_dir / "01_stageA_tuning" / "best_config.json"
            if not p.exists():
                raise FileNotFoundError("Best config not found. Run --mode tune or --mode all first.")
            meta = json.loads(p.read_text(encoding="utf-8"))
            best_cfg = RunConfig(**meta["best_config"])
            best_seed = int(meta["best_seed"])
        run_stage_b_sensitivity(args, bundle, mono_rules, best_cfg, out_dir, device)

    pd.DataFrame(
        [
            {
                "status": "done",
                "mode": args.mode,
                "best_seed_if_available": best_seed,
                "out_dir": str(out_dir),
            }
        ]
    ).to_excel(out_dir / "run_done.xlsx", index=False)

    print("Output dir:", out_dir)
    print("Synthetic stats:", bundle["syn_stats"])
    if best_cfg is not None:
        print("Best cfg:", asdict(best_cfg))


if __name__ == "__main__":
    main()

