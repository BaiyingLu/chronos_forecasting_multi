"""
Preprocess raw CGM CSVs (Bris / HUPA-UOM / OhioT1DM) for Chronos-2 covariate forecasting.

Two modes:
  - test : split sequences at any bg NaN. Used by zeroshot eval.
  - train: short bg NaN gaps (<= --max_gap) are linearly interpolated; long ones split sequences.
           Used for fine-tuning data prep.

Other columns are imputed per a built-in COLUMN_STRATEGY dict (see below).
Columns not in the dict are dropped (so unused indicator columns disappear).

Outputs cleaned CSVs (one per patient) plus a _preprocess_stats.json with per-patient details.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Per-column imputation strategy. Columns not listed here are dropped on output.
#   target              : split sequences on any NaN (test) / interpolate <=max_gap, split otherwise (train)
#   interpolate_or_split: linear interpolate <=max_gap; longer runs split sequences
#   ffill_bfill         : forward-fill within sequence; back-fill at sequence start
#   fill0               : event/state default of 0 (NaN means "no event")
COLUMN_STRATEGY = {
    "bg":             "target",
    "basal":          "ffill_bfill",
    "total_insulin":  "ffill_bfill",
    "bolus":          "fill0",
    "carbs":          "fill0",
    "steps":          "fill0",
    "iob":            "fill0",
    "cob":            "fill0",
    "smoothed_step":  "fill0",
}

# Always preserved untouched (not imputed, not strategy-driven).
META_COLUMNS = {"timestamp", "item_id"}


def _impute_within_sequence(df: pd.DataFrame, col: str, strategy: str, max_gap: int) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Apply a column's strategy within each sequence_id group. Returns df and a per-column stat dict."""
    n_nan_input = int(df[col].isna().sum())
    drop_mask = pd.Series(False, index=df.index)

    if strategy == "fill0":
        df[col] = df[col].fillna(0)

    elif strategy == "ffill_bfill":
        df[col] = df.groupby("sequence_id")[col].transform(lambda s: s.ffill().bfill())

    elif strategy == "interpolate_or_split":
        # Within each sequence, interpolate gaps up to max_gap. Anything still NaN -> drop row.
        df[col] = df.groupby("sequence_id")[col].transform(
            lambda s: s.interpolate(method="linear", limit=max_gap, limit_area="inside")
        )
        drop_mask = df[col].isna()

    elif strategy == "target":
        # 'target' is special; handled by the caller depending on mode. We should never reach here.
        raise RuntimeError("target strategy must be handled in apply_target_strategy()")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    n_dropped = int(drop_mask.sum())
    # n_filled / n_residual_nan are filled in by the caller after all drops are applied,
    # because rows dropped by other columns' masks also affect this column's residual count.
    return df, {
        "strategy": strategy,
        "n_nan_input": n_nan_input,
        "n_dropped": n_dropped,
    }, drop_mask


def _apply_target_strategy(df: pd.DataFrame, mode: str, max_gap: int) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Handle bg separately. In test mode, any bg NaN -> drop. In train, interpolate <=max_gap, drop the rest."""
    n_nan_input = int(df["bg"].isna().sum())

    if mode == "train":
        df["bg"] = df.groupby("sequence_id")["bg"].transform(
            lambda s: s.interpolate(method="linear", limit=max_gap, limit_area="inside")
        )
        drop_mask = df["bg"].isna()
        strategy_label = f"interpolate<={max_gap}_then_split"
    elif mode == "test":
        drop_mask = df["bg"].isna()
        strategy_label = "split_on_nan"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    n_dropped = int(drop_mask.sum())
    return df, {
        "strategy": strategy_label,
        "n_nan_input": n_nan_input,
        "n_dropped": n_dropped,
    }, drop_mask


def _assign_sequence_ids(df: pd.DataFrame, time_gap_threshold_min: int) -> pd.DataFrame:
    """Assign sequence_id based on timestamp gaps. Assumes df is sorted by timestamp."""
    threshold = pd.Timedelta(minutes=time_gap_threshold_min)
    diffs = df["timestamp"].diff()
    new_seq = (diffs > threshold) | diffs.isna()
    df["sequence_id"] = new_seq.cumsum().astype(int) - 1  # 0-indexed
    return df


def preprocess_patient(
    df: pd.DataFrame,
    patient_name: str,
    mode: str,
    max_gap: int,
    time_gap_threshold_min: int,
    min_seq_len: int,
) -> tuple[pd.DataFrame, dict]:
    raw_rows = len(df)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["item_id"] = patient_name

    # Drop columns not in strategy + not meta. Track what was dropped for the stats.
    keep_cols = [c for c in df.columns if c in COLUMN_STRATEGY or c in META_COLUMNS]
    dropped_cols = [c for c in df.columns if c not in keep_cols]
    df = df[keep_cols]

    # Initial sequence ids based on timestamp gaps.
    df = _assign_sequence_ids(df, time_gap_threshold_min)

    column_stats: dict = {}
    drop_masks = []

    # Apply non-target column strategies first (target depends on mode and gets handled last
    # so its drop rows are computed after other imputations have settled).
    for col, strat in COLUMN_STRATEGY.items():
        if col == "bg" or col not in df.columns:
            continue
        df, stat, dmask = _impute_within_sequence(df, col, strat, max_gap)
        column_stats[col] = stat
        drop_masks.append(dmask)

    # Target (bg) handling.
    df, bg_stat, bg_drop_mask = _apply_target_strategy(df, mode, max_gap)
    column_stats["bg"] = bg_stat
    drop_masks.append(bg_drop_mask)

    # Union all drop masks; drop those rows.
    if drop_masks:
        union_drop = drop_masks[0].copy()
        for m in drop_masks[1:]:
            union_drop |= m
        df = df.loc[~union_drop].reset_index(drop=True)

    # Re-fragment sequences because dropped rows may have created new time gaps.
    df = _assign_sequence_ids(df, time_gap_threshold_min)

    # Drop sequences shorter than min_seq_len.
    seq_lens = df.groupby("sequence_id").size()
    valid_seqs = seq_lens[seq_lens >= min_seq_len].index
    df = df[df["sequence_id"].isin(valid_seqs)].reset_index(drop=True)
    df = _assign_sequence_ids(df, time_gap_threshold_min)  # renumber 0..N-1 contiguously

    # Per-column residual NaN AFTER all drops + min-seq filter.
    # Strict check on bg only (target must be clean for evaluation). Non-target residuals are
    # allowed to flow through — downstream zeroshot must skip patients whose chosen covariates
    # have residual NaN (e.g. Bris patients with no basal data at all).
    for col, stat in column_stats.items():
        n_residual = int(df[col].isna().sum()) if col in df.columns else 0
        stat["n_residual_nan"] = n_residual
        stat["n_filled"] = stat["n_nan_input"] - stat["n_dropped"] - n_residual

    if column_stats["bg"]["n_residual_nan"] > 0:
        raise RuntimeError(f"{patient_name}: bg has residual NaN after preprocessing — invariant broken")

    residual_nan_columns = {c: s["n_residual_nan"] for c, s in column_stats.items() if s["n_residual_nan"] > 0}

    seq_lens_final = df.groupby("sequence_id").size().values if len(df) else np.array([])
    stats = {
        "raw_rows": raw_rows,
        "kept_rows": int(len(df)),
        "dropped_input_columns": dropped_cols,
        "residual_nan_columns": residual_nan_columns,  # cols with leftover NaN; downstream must filter
        "n_sequences": int(len(seq_lens_final)),
        "min_seq_len": int(seq_lens_final.min()) if len(seq_lens_final) else 0,
        "max_seq_len": int(seq_lens_final.max()) if len(seq_lens_final) else 0,
        "median_seq_len": float(np.median(seq_lens_final)) if len(seq_lens_final) else 0.0,
        "columns": column_stats,
    }
    return df, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name (used for output dir): bris, hupa-uom, ohiot1dm")
    parser.add_argument("--input_dir", required=True, help="Directory containing per-patient raw CSVs")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir. Defaults to input_data/<dataset>/<mode>/")
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--max_gap", type=int, default=6,
                        help="Max NaN run length (in steps) for interpolation. Longer runs split sequences. Default 6 (=30min @ 5min sampling).")
    parser.add_argument("--time_gap_threshold_min", type=int, default=60,
                        help="Minutes; timestamp gaps larger than this start a new sequence.")
    parser.add_argument("--min_seq_len", type=int, default=1,
                        help="Drop sequences shorter than this many rows. Default 1 (keep all). Set to context+horizon if you want to pre-filter.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / "input_data" / args.dataset / args.mode
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSVs found in {input_dir}")

    print(f"Preprocessing {len(csv_files)} patients from {input_dir}")
    print(f"  mode={args.mode}  max_gap={args.max_gap}  time_gap_threshold_min={args.time_gap_threshold_min}")
    print(f"  output -> {output_dir}")

    all_stats = {}
    total_raw, total_kept, total_seqs = 0, 0, 0

    for csv_path in csv_files:
        patient = csv_path.stem
        df_raw = pd.read_csv(csv_path)
        try:
            df_clean, stats = preprocess_patient(
                df_raw,
                patient_name=patient,
                mode=args.mode,
                max_gap=args.max_gap,
                time_gap_threshold_min=args.time_gap_threshold_min,
                min_seq_len=args.min_seq_len,
            )
        except Exception as e:
            print(f"  [{patient}] FAILED: {e}")
            all_stats[patient] = {"error": str(e)}
            continue

        out_path = output_dir / f"{patient}.csv"
        df_clean.to_csv(out_path, index=False)

        all_stats[patient] = stats
        total_raw += stats["raw_rows"]
        total_kept += stats["kept_rows"]
        total_seqs += stats["n_sequences"]
        warn = ""
        if stats["residual_nan_columns"]:
            warn = f"  ⚠ residual NaN: {stats['residual_nan_columns']}"
        print(f"  [{patient}] raw={stats['raw_rows']} kept={stats['kept_rows']} "
              f"seqs={stats['n_sequences']} (min/med/max={stats['min_seq_len']}/"
              f"{stats['median_seq_len']:.0f}/{stats['max_seq_len']}){warn}")

    report = {
        "config": {
            "dataset": args.dataset,
            "mode": args.mode,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "max_gap": args.max_gap,
            "time_gap_threshold_min": args.time_gap_threshold_min,
            "min_seq_len": args.min_seq_len,
            "column_strategy": COLUMN_STRATEGY,
        },
        "global": {
            "n_patients": len(csv_files),
            "total_raw_rows": total_raw,
            "total_kept_rows": total_kept,
            "total_sequences": total_seqs,
        },
        "patients": all_stats,
    }
    stats_path = output_dir / "_preprocess_stats.json"
    with open(stats_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote stats -> {stats_path}")
    print(f"Total: raw={total_raw} kept={total_kept} sequences={total_seqs}")


if __name__ == "__main__":
    main()
