"""
End-to-end baseline for Recover Hackathon:
- Loads train/val/test + metaData
- Adds seasonal features month_sin, month_cos
- Aggregates to one row per room instance (id)
- Masks labels for training to simulate missing ops
- Trains a small multi-label MLP (sigmoid + BCE)
- Tunes per-label thresholds on validation using hackathon score
- Predicts missing ops on test, zeroing already-observed ops
- Writes Kaggle-ready submission CSV
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Config & Repro
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_LABELS = 388
LABEL_COLUMNS = [str(i) for i in range(NUM_LABELS)]

ROOM_CATEGORIES = [
    "andre områder", "kjøkken", "stue", "gang", "soverom",
    "bad", "bod", "vaskerom", "wc", "kjeller", "garasje"
]
ROOM_INDEX = {r: i for i, r in enumerate(ROOM_CATEGORIES)}

# Toggle minimal extra features
USE_ROOM_ONEHOT = False  # set True if you want to include room one-hot (adds ~11 dims)

# -----------------------------
# Hackathon metric (standalone)
# -----------------------------
def get_room_scores(room_preds: List[int], room_targets: List[int]) -> Dict[str, float]:
    EMPTY_ROOM_REWARD = 1
    TRUE_POSITIVE_REWARD = 1
    FALSE_POSITIVE_PENALTY = 0.25
    FALSE_NEGATIVE_PENALTY = 0.5

    best_possible_score = 0.0
    dummy_score = 0.0
    score = 0.0

    room_targets_set = set(room_targets)
    room_preds_set = set(room_preds)

    if len(room_targets_set) == 0:
        best_possible_score += EMPTY_ROOM_REWARD
        dummy_score += EMPTY_ROOM_REWARD

        if len(room_preds_set) == 0:
            score += EMPTY_ROOM_REWARD
        else:
            score -= FALSE_POSITIVE_PENALTY * len(room_preds_set)

    else:
        best_possible_score += len(room_targets_set) * TRUE_POSITIVE_REWARD
        dummy_score -= len(room_targets_set) * FALSE_NEGATIVE_PENALTY

        score += TRUE_POSITIVE_REWARD * len(room_targets_set & room_preds_set)
        score -= FALSE_POSITIVE_PENALTY * len(room_preds_set - room_targets_set)
        score -= FALSE_NEGATIVE_PENALTY * len(room_targets_set - room_preds_set)

    if best_possible_score == dummy_score:
        normalized_score = -abs(score)
    else:
        normalized_score = (score - dummy_score) / (best_possible_score - dummy_score)

    return {
        "score": score,
        "dummy_score": dummy_score,
        "best_possible_score": best_possible_score,
        "normalized_score": normalized_score,
    }


def normalized_rooms_score(preds: List[List[int]], targets: List[List[int]]) -> float:
    best_possible_score = 0.0
    dummy_score = 0.0
    score = 0.0

    for room_targets, room_preds in zip(targets, preds, strict=True):
        room_scores = get_room_scores(room_preds, room_targets)
        score += room_scores["score"]
        dummy_score += room_scores["dummy_score"]
        best_possible_score += room_scores["best_possible_score"]

    if best_possible_score == dummy_score:
        return -abs(score)

    return (score - dummy_score) / (best_possible_score - dummy_score)


# -----------------------------
# Utils: rooms, features, masking
# -----------------------------
def map_room(room_name: str) -> str:
    if not isinstance(room_name, str):
        return "ukjent"
    s = room_name.lower()
    for r in ROOM_CATEGORIES:
        if r in s:
            return r
    return "ukjent"


def add_month_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - case_creation_month (int 1..12)
      - month_sin, month_cos (float32 in [-1,1])
    Requires column 'case_creation_month' from metaData.csv after join.
    """
    if "case_creation_month" not in df.columns:
        raise KeyError("case_creation_month missing after join. Ensure you add seasonality before pruning columns.")
    m = pd.to_numeric(df["case_creation_month"], errors="coerce").fillna(6).astype(int).clip(1, 12)
    angle = 2.0 * np.pi * (m - 1) / 12.0
    out = df.copy()
    out["case_creation_month"] = m
    out["month_sin"] = np.sin(angle).astype(np.float32)
    out["month_cos"] = np.cos(angle).astype(np.float32)
    return out


def multi_hot_from_codes(codes: List[int], num_labels: int = NUM_LABELS) -> np.ndarray:
    v = np.zeros(num_labels, dtype=np.float32)
    for c in codes:
        c_int = int(c)
        if 0 <= c_int < num_labels:
            v[c_int] = 1.0
    return v


def mask_codes_for_training(
    full_codes: List[int],
    min_hide: int = 1,
    max_frac: float = 0.5,
    rng: np.random.Generator | None = None,
) -> Tuple[List[int], List[int]]:
    """
    Split full set S into observed O and hidden H for supervision.
    Ensures H non-empty if len(S) > 1. If only 1 code, returns H=[] (no masking).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    S = sorted(set(int(x) for x in full_codes))
    if len(S) <= 1:
        return S, []  # cannot hide the only one
    max_hide = max(min_hide, int(math.floor(len(S) * max_frac)))
    max_hide = min(max_hide, len(S) - 1)  # keep at least one observed
    n_hide = int(rng.integers(low=min_hide, high=max_hide + 1))
    H = sorted(rng.choice(S, size=n_hide, replace=False).tolist())
    O = [x for x in S if x not in H]
    return O, H


def aggregate_per_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per room instance (project_id, room, id).
    - codes: list of operation codes in that room
    - room_category: normalized room string
    - month_sin, month_cos: from project-level metadata
    - case_creation_month: kept for inspection/debug
    """
    g = (
        df.groupby(["project_id", "room", "id"])
          .agg({
              "work_operation_cluster_code": list,
              "case_creation_month": "first",
              "month_sin": "first",
              "month_cos": "first",
          })
          .reset_index()
          .rename(columns={"work_operation_cluster_code": "codes"})
    )
    g["room_category"] = g["room"].apply(map_room)
    return g


def build_xy_from_agg(
    df_agg: pd.DataFrame,
    do_mask: bool,
    rng: np.random.Generator,
    use_room_onehot: bool = USE_ROOM_ONEHOT,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns:
        X: [N, D_in]
        Y: [N, NUM_LABELS]
        meta: df with ['id','project_id','room_category','observed_codes','hidden_codes','month_sin','month_cos']
    """
    X_list, Y_list, meta_rows = [], [], []

    for _, row in df_agg.iterrows():
        rid = int(row["id"])
        codes_full = [int(c) for c in row["codes"]]
        room_cat = row["room_category"]

        if do_mask:
            observed, hidden = mask_codes_for_training(codes_full, rng=rng)
        else:
            observed, hidden = codes_full, []

        # Base ops vector
        x_ops = multi_hot_from_codes(observed, NUM_LABELS)

        # Optional room one-hot
        if use_room_onehot:
            rc_vec = np.zeros(len(ROOM_CATEGORIES), dtype=np.float32)
            rc_idx = ROOM_INDEX.get(room_cat, None)
            if rc_idx is not None:
                rc_vec[rc_idx] = 1.0
            x_vec = np.concatenate([x_ops, rc_vec], axis=0)
        else:
            x_vec = x_ops

        # Add seasonality (always)
        ms = np.float32(row["month_sin"])
        mc = np.float32(row["month_cos"])
        x_vec = np.concatenate([x_vec, np.array([ms, mc], dtype=np.float32)], axis=0)

        # Targets are the hidden ones
        y_vec = multi_hot_from_codes(hidden, NUM_LABELS)

        X_list.append(x_vec)
        Y_list.append(y_vec)
        meta_rows.append({
            "id": rid,
            "project_id": int(row["project_id"]),
            "room_category": room_cat,
            "observed_codes": observed,
            "hidden_codes": hidden,
            "month_sin": float(ms),
            "month_cos": float(mc),
        })

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    meta = pd.DataFrame(meta_rows)
    return X, Y, meta


# -----------------------------
# Dataset & Model
# -----------------------------
class RoomsDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)  # logits
        )
    def forward(self, x):
        return self.net(x)


# -----------------------------
# Training / Evaluation helpers
# -----------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)

    return total_loss / max(1, total_n)


def preds_from_probs(
    probs: np.ndarray,
    thresholds: np.ndarray,
    observed_lists: List[List[int]]
) -> List[List[int]]:
    """
    Convert probability matrix -> index predictions using per-label thresholds,
    and remove any codes that are already observed for that room.
    """
    bin_mat = (probs >= thresholds[None, :]).astype(np.int32)
    preds = []
    for i in range(bin_mat.shape[0]):
        idx = bin_mat[i].nonzero()[0].tolist()
        obs = set(int(c) for c in observed_lists[i])
        idx = [j for j in idx if j not in obs]
        preds.append(idx)
    return preds


def score_with_thresholds(
    probs: np.ndarray,
    thresholds: np.ndarray,
    observed: List[List[int]],
    targets: List[List[int]]
) -> float:
    preds = preds_from_probs(probs, thresholds, observed)
    return normalized_rooms_score(preds, targets)


def tune_thresholds_coordinate_descent(
    probs: np.ndarray,
    observed: List[List[int]],
    targets: List[List[int]],
    init: float = 0.5,
    grid = np.linspace(0.2, 0.8, 7),
    max_passes: int = 2
) -> np.ndarray:
    """
    Greedy coordinate descent over labels with a small grid. Two passes is usually enough.
    """
    ths = np.full((probs.shape[1],), init, dtype=np.float32)
    base = score_with_thresholds(probs, ths, observed, targets)
    print(f"[tune] start score: {base:.4f}")

    for p in range(max_passes):
        improved_any = False
        for j in range(probs.shape[1]):
            best_t, best_s = ths[j], base
            for t in grid:
                if t == ths[j]:
                    continue
                ths_try = ths.copy()
                ths_try[j] = t
                s = score_with_thresholds(probs, ths_try, observed, targets)
                if s > best_s:
                    best_s, best_t = s, t
            if best_t != ths[j]:
                ths[j] = best_t
                base = best_s
                improved_any = True
        print(f"[tune] pass {p+1}: score {base:.4f}")
        if not improved_any:
            break
    return ths


# -----------------------------
# Main pipeline
# -----------------------------
def main(args):
    data_dir = args.data_dir
    out_csv = args.out

    print(f"Device: {DEVICE}")
    print(f"Data dir: {data_dir}")

    # 1) Load CSVs
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    meta  = pd.read_csv(os.path.join(data_dir, "metaData.csv"))

    # 2) Ensure join dtypes
    for df in (train, val, test, meta):
        df["project_id"] = df["project_id"].astype(int)

    # 3) Join with metadata (left)
    train_joined = train.merge(meta, on="project_id", how="left", suffixes=("_train", "_meta"))
    val_joined   = val.merge(meta,   on="project_id", how="left", suffixes=("_train", "_meta"))
    test_joined  = test.merge(meta,  on="project_id", how="left", suffixes=("_train", "_meta"))

    # 4) Add seasonal features BEFORE pruning
    train_joined = add_month_features(train_joined)
    val_joined   = add_month_features(val_joined)
    test_joined  = add_month_features(test_joined)

    # 5) Minimal prune (keep essentials, including month features)
    cols_keep = [
        "id", "project_id", "room", "work_operation_cluster_code",
        "case_creation_month", "month_sin", "month_cos"
    ]
    train_joined = train_joined[cols_keep].copy()
    val_joined   = val_joined[cols_keep].copy()
    test_joined  = test_joined[cols_keep].copy()

    # 6) Aggregate to one row per id
    train_agg = aggregate_per_id(train_joined)
    val_agg   = aggregate_per_id(val_joined)
    test_agg  = aggregate_per_id(test_joined)

    # 7) Build training/validation matrices (with masking)
    rng_train = np.random.default_rng(SEED)
    rng_val   = np.random.default_rng(SEED + 1)

    X_train, Y_train, meta_train = build_xy_from_agg(train_agg, do_mask=True, rng=rng_train)
    X_val,   Y_val,   meta_val   = build_xy_from_agg(val_agg,   do_mask=True, rng=rng_val)

    print("Train shapes:", X_train.shape, Y_train.shape)
    print("Val   shapes:", X_val.shape,   Y_val.shape)

    # 8) Datasets & Loaders
    train_ds = RoomsDataset(X_train, Y_train)
    val_ds   = RoomsDataset(X_val,   Y_val)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, drop_last=False)

    # 9) Model, loss, optimizer
    input_dim  = X_train.shape[1]
    output_dim = NUM_LABELS

    model = MLP(input_dim, output_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 10) Train
    epochs = 8
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer)
        va_loss = run_epoch(model, val_loader,   criterion, optimizer=None)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f}")
        best_val = min(best_val, va_loss)

    # 11) Collect validation probabilities for threshold tuning
    model.eval()
    with torch.no_grad():
        val_probs = []
        for xb, yb in val_loader:
            logits = model(xb.to(DEVICE))
            val_probs.append(torch.sigmoid(logits).cpu().numpy())
        probs_val = np.concatenate(val_probs, axis=0)

    targets_val  = meta_val["hidden_codes"].tolist()
    observed_val = meta_val["observed_codes"].tolist()
    room_cat_val = meta_val["room_category"].tolist()

    # 12) Tune per-label thresholds on validation
    thresholds_vec = tune_thresholds_coordinate_descent(
        probs=probs_val,
        observed=observed_val,
        targets=targets_val,
        init=0.5,
        grid=np.linspace(0.2, 0.8, 7),
        max_passes=2
    )
    tuned_val_score = score_with_thresholds(probs_val, thresholds_vec, observed_val, targets_val)
    print(f"[tune] validation score after tuning: {tuned_val_score:.4f}")

    # 13) Build test inputs (no masking), predict with tuned thresholds
    X_test_ops = []
    test_ids = []
    observed_by_id: Dict[int, List[int]] = {}

    for _, row in test_agg.iterrows():
        rid = int(row["id"])
        codes = [int(c) for c in row["codes"]]
        observed_by_id[rid] = sorted(set(codes))

        # base input from observed operations
        x_vec = multi_hot_from_codes(codes, NUM_LABELS)

        # optional room one-hot
        if USE_ROOM_ONEHOT:
            rc_vec = np.zeros(len(ROOM_CATEGORIES), dtype=np.float32)
            rc_idx = ROOM_INDEX.get(row["room_category"], None)
            if rc_idx is not None:
                rc_vec[rc_idx] = 1.0
            x_vec = np.concatenate([x_vec, rc_vec], axis=0)

        # append month_sin/month_cos
        ms = np.float32(row["month_sin"])
        mc = np.float32(row["month_cos"])
        x_vec = np.concatenate([x_vec, np.array([ms, mc], dtype=np.float32)], axis=0)

        X_test_ops.append(x_vec)
        test_ids.append(rid)

    X_test = np.stack(X_test_ops, axis=0).astype(np.float32)

    model.eval()
    with torch.no_grad():
        logits_test = model(torch.from_numpy(X_test).to(DEVICE))
        probs_test  = torch.sigmoid(logits_test).cpu().numpy()

    # Apply tuned thresholds
    ths = thresholds_vec
    pred_bin = (probs_test >= ths[None, :]).astype(np.int32)

    # Never predict already observed ops
    for i, rid in enumerate(test_ids):
        for c in observed_by_id[rid]:
            if 0 <= c < NUM_LABELS:
                pred_bin[i, c] = 0

    # 14) (Optional) Room-wise top-K guardrail from validation medians
    if args.use_room_topk:
        from collections import defaultdict
        room2counts = defaultdict(list)
        for rc, tgt in zip(room_cat_val, targets_val):
            room2counts[rc].append(len(tgt))
        room2k = {rc: max(1, int(np.median(cnts))) for rc, cnts in room2counts.items()}

        for i, rid in enumerate(test_ids):
            rc = test_agg.iloc[i]["room_category"]
            k = room2k.get(rc, args.default_topk)
            pos_idx = np.where(pred_bin[i] == 1)[0]
            if len(pos_idx) > k:
                keep = pos_idx[np.argsort(-probs_test[i, pos_idx])[:k]]
                drop_idx = pos_idx[~np.isin(pos_idx, keep)]
                pred_bin[i, drop_idx] = 0

    # 15) Build Kaggle-ready submission
    submission = pd.DataFrame(pred_bin, columns=LABEL_COLUMNS)
    submission.insert(0, "id", test_ids)
    for c in LABEL_COLUMNS:
        submission[c] = submission[c].astype(int)
    submission = submission.sort_values("id").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    submission.to_csv(out_csv, index=False)
    print(f"Saved submission to: {out_csv}")
    print(submission.head(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data", help="Folder containing train/val/test/metaData.csv")
    parser.add_argument("--out", type=str, default="./submission.csv", help="Output submission CSV path")
    parser.add_argument("--use-room-topk", action="store_true", help="Enable room-wise top-K clipping based on val medians")
    parser.add_argument("--default-topk", type=int, default=5, help="Fallback K if room category is unknown")
    args = parser.parse_args()
    main(args)
