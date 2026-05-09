import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from chronos import BaseChronosPipeline, Chronos2Pipeline

# Device selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2", device_map=device
)


def covariate_signature(covariates):
    """Return a slug used in output dir name. 'none' for univariate."""
    return "none" if not covariates else "_".join(sorted(covariates))


def rolling_window_forecast(df, pipeline, target_col, covariates, context_length,
                            prediction_length=18, step_size=1, verbose=False):
    """
    Rolling-window forecast over each sequence_id group in the preprocessed df.
    Covariates are passed as past-only (future_df=None) so the model sees their history but
    not their future values.
    """
    needed_cols = ["item_id", "timestamp", target_col] + list(covariates)

    all_predictions = []
    all_ground_truth = []
    all_timestamps = []
    all_sequence_ids = []

    for seq_id, seq_df in df.groupby("sequence_id"):
        seq_df = seq_df.reset_index(drop=True)
        seq_length = len(seq_df)
        max_start_idx = seq_length - context_length - prediction_length

        if max_start_idx < 0:
            if verbose:
                print(f"    Seq {seq_id}: too short ({seq_length} points), skipping")
            continue

        for start_idx in range(0, max_start_idx + 1, step_size):
            end_idx = start_idx + context_length
            pred_end_idx = end_idx + prediction_length

            context_window = seq_df.iloc[start_idx:end_idx][needed_cols].copy()
            ground_truth_window = seq_df.iloc[end_idx:pred_end_idx]

            try:
                pred_df = pipeline.predict_df(
                    context_window,
                    future_df=None,
                    prediction_length=prediction_length,
                    quantile_levels=[0.1, 0.5, 0.9],
                    target=target_col,
                )

                mask = pred_df["target_name"].astype(str) == target_col
                predictions = pred_df.loc[mask, "predictions"].values.astype(float)
                ground_truth = ground_truth_window[target_col].values

                if len(predictions) == 0:
                    if verbose:
                        print(f"    Empty predictions at window {start_idx} in seq {seq_id}")
                        print(f"    target_name values: {pred_df['target_name'].unique()}")
                    continue

                timestamps = ground_truth_window["timestamp"].values

                all_predictions.append(predictions)
                all_ground_truth.append(ground_truth)
                all_timestamps.append(timestamps)
                all_sequence_ids.append(int(seq_id))

            except Exception as e:
                print(f"    Error at window {start_idx} in seq {seq_id}: {e}")
                continue

    return all_predictions, all_ground_truth, all_timestamps, all_sequence_ids


def calculate_metrics(all_predictions, all_ground_truth, horizon_steps):
    """Calculate RMSE and MAE for different prediction horizons."""
    results = {}

    for horizon_name, horizon_step in horizon_steps.items():
        rmse_values = []
        mae_values = []

        for pred, gt in zip(all_predictions, all_ground_truth):
            if len(pred) >= horizon_step and len(gt) >= horizon_step:
                pred_value = pred[horizon_step - 1]
                gt_value = gt[horizon_step - 1]

                squared_error = (pred_value - gt_value) ** 2
                absolute_error = abs(pred_value - gt_value)

                rmse_values.append(squared_error)
                mae_values.append(absolute_error)

        if len(rmse_values) > 0:
            avg_rmse = np.sqrt(np.mean(rmse_values))
            avg_mae = np.mean(mae_values)
            results[horizon_name] = {
                "RMSE": avg_rmse,
                "MAE": avg_mae,
                "n_samples": len(rmse_values),
            }
        else:
            results[horizon_name] = {"RMSE": np.nan, "MAE": np.nan, "n_samples": 0}

    return results


def evaluate_single_patient(df, patient_name, pipeline, covariates, context_lengths,
                            target_col="bg", prediction_length=18, step_size=1):
    horizon_steps = {"15min": 3, "30min": 6, "60min": 12, "90min": 18}

    # Required cols must exist and be NaN-free in the preprocessed CSV.
    required = [target_col] + list(covariates)
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"  [{patient_name}] SKIP: columns not in CSV: {missing_cols}")
        return None
    nan_cols = {c: int(df[c].isna().sum()) for c in required if df[c].isna().any()}
    if nan_cols:
        print(f"  [{patient_name}] SKIP: residual NaN in selected columns: {nan_cols}")
        return None
    if "sequence_id" not in df.columns:
        print(f"  [{patient_name}] SKIP: no sequence_id column (data not preprocessed?)")
        return None

    try:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["sequence_id", "timestamp"]).reset_index(drop=True)
        df["item_id"] = patient_name

        n_seqs = df["sequence_id"].nunique()
        print(f"  Patient: {patient_name}")
        print(f"    Total points: {len(df)}, Sequences: {n_seqs}")

        patient_results = {}

        for context_length in context_lengths:
            all_predictions, all_ground_truth, all_timestamps, all_sequence_ids = rolling_window_forecast(
                df,
                pipeline,
                target_col,
                covariates,
                context_length,
                prediction_length,
                step_size,
                verbose=True,
            )

            if len(all_predictions) == 0:
                print(f"    Context {context_length}: NO valid predictions!")
                patient_results[context_length] = None
                continue

            results = calculate_metrics(all_predictions, all_ground_truth, horizon_steps)
            patient_results[context_length] = results

            print(f"    Context {context_length}: {len(all_predictions)} windows")

        return patient_results

    except Exception as e:
        print(f"  Error processing {patient_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_results_across_patients(all_patient_results, context_lengths):
    """Aggregate results across all patients."""
    horizons = ["15min", "30min", "60min", "90min"]
    summary_data = []

    for context_length in context_lengths:
        for horizon in horizons:
            rmse_values = []
            mae_values = []

            for patient_name, patient_results in all_patient_results.items():
                if patient_results is not None and context_length in patient_results:
                    if patient_results[context_length] is not None:
                        if horizon in patient_results[context_length]:
                            r = patient_results[context_length][horizon]
                            if not np.isnan(r["RMSE"]):
                                rmse_values.append(r["RMSE"])
                                mae_values.append(r["MAE"])

            if len(rmse_values) > 0:
                avg_rmse = np.mean(rmse_values)
                std_rmse = np.std(rmse_values)
                avg_mae = np.mean(mae_values)
                std_mae = np.std(mae_values)
                n_patients = len(rmse_values)
            else:
                avg_rmse = std_rmse = avg_mae = std_mae = np.nan
                n_patients = 0

            summary_data.append({
                "Context_Length": context_length,
                "Context_Hours": context_length * 5 / 60,
                "Horizon": horizon,
                "RMSE_Mean": avg_rmse,
                "RMSE_Std": std_rmse,
                "MAE_Mean": avg_mae,
                "MAE_Std": std_mae,
                "N_Patients": n_patients,
            })

    return pd.DataFrame(summary_data)


def save_detailed_results(all_patient_results, context_lengths, output_dir):
    """Save detailed per-patient results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    horizons = ["15min", "30min", "60min", "90min"]

    for context_length in context_lengths:
        patient_data = []

        for patient_name, patient_results in all_patient_results.items():
            if patient_results is not None and context_length in patient_results:
                if patient_results[context_length] is not None:
                    for horizon in horizons:
                        if horizon in patient_results[context_length]:
                            r = patient_results[context_length][horizon]
                            patient_data.append({
                                "Patient": patient_name,
                                "Context_Length": context_length,
                                "Horizon": horizon,
                                "RMSE": r["RMSE"],
                                "MAE": r["MAE"],
                                "N_Samples": r["n_samples"],
                            })

        if patient_data:
            detail_df = pd.DataFrame(patient_data)
            output_file = f"{output_dir}/context_{context_length}_detailed.csv"
            detail_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")


def main(args):
    base_dir = Path(__file__).resolve().parent / "input_data"

    datasets_to_run = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.covariates.strip().lower() in ("", "none"):
        covariates: list[str] = []
    else:
        covariates = [c.strip() for c in args.covariates.split(",") if c.strip()]
    cov_sig = covariate_signature(covariates)

    context_lengths = [144]

    print("Running zeroshot covariate eval")
    print(f"  run_name         : {args.run_name}")
    print(f"  datasets         : {datasets_to_run}")
    print(f"  covariates       : {covariates if covariates else '(univariate)'}")
    print(f"  output signature : {cov_sig}")
    print(f"  step_size        : {args.step_size}")
    print(f"  prediction_length: {args.prediction_length}")
    print(f"  context_lengths  : {context_lengths}")

    for dataset_name in datasets_to_run:
        data_dir = base_dir / dataset_name / "test"
        if not data_dir.is_dir():
            print(f"\n[SKIP] {dataset_name}: {data_dir} not found (run preprocess_data.py first)")
            continue

        csv_files = sorted([p for p in data_dir.glob("*.csv") if not p.name.startswith("_")])

        print("\n" + "=" * 80)
        print(f"PROCESSING DATASET: {dataset_name}  ({len(csv_files)} patients)")
        print("=" * 80)

        all_patient_results = {}

        for i, csv_path in enumerate(csv_files, 1):
            patient_name = csv_path.stem
            print(f"[{i}/{len(csv_files)}] {patient_name}")

            df = pd.read_csv(csv_path)

            patient_results = evaluate_single_patient(
                df=df,
                patient_name=patient_name,
                pipeline=pipeline,
                covariates=covariates,
                context_lengths=context_lengths,
                prediction_length=args.prediction_length,
                step_size=args.step_size,
            )

            all_patient_results[patient_name] = patient_results

        summary_df = aggregate_results_across_patients(all_patient_results, context_lengths)

        out_dir = Path(f"./results_covariate/{args.run_name}/{dataset_name}/{cov_sig}/step_{args.step_size}")
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_df.to_csv(out_dir / "summary.csv", index=False)

        save_detailed_results(
            all_patient_results,
            context_lengths,
            output_dir=str(out_dir / "patient_results"),
        )

        print(f"Saved results to {out_dir}")

    print("\nALL DATASETS COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--covariates", type=str, default="bolus,carbs",
                        help="Comma-separated covariate column names (must exist in preprocessed CSV). "
                             "Use '' or 'none' for univariate.")
    parser.add_argument("--datasets", type=str, default="ohiot1dm,bris,hupa-uom",
                        help="Comma-separated dataset names (subdirs of input_data/).")
    parser.add_argument("--run_name", type=str, default="zeroshot",
                        help="Output dir top-level under results_covariate/, e.g. 'zeroshot', "
                             "'finetune-lora-1e-4', 'zeroshot_v2'. Used to keep ablations separate "
                             "from fine-tuning runs.")
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--prediction_length", type=int, default=18)
    args = parser.parse_args()
    main(args)
