import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
import warnings
import os
import sys
import glob
import numpy as np
from typing import Dict, Any

# --- Suppress pandas PerformanceWarning ---
warnings.filterwarnings(
    "ignore", "DataFrame is highly fragmented", category=pd.errors.PerformanceWarning
)

# ======================================================================================
# --- DEBUG MODE TOGGLE ---
# This MUST match the setting in the training script.
# ======================================================================================
DEBUG_MODE = False

# ======================================================================================
# --- Configuration ---
# ======================================================================================
CHECKPOINT_DIR = "model_checkpoints"
LATEST_DATA_FILE = 'Prime 1.csv'
TIME_COLUMN = 'Datum'
SERIES_COLUMN = 'Naam'
TARGET_COLUMNS = [
    "OrderDag", "OrderUur_0", "OrderUur_1", "OrderUur_2", "OrderUur_3", "OrderUur_4",
    "OrderUur_5", "OrderUur_6", "OrderUur_7", "OrderUur_8", "OrderUur_9", "OrderUur_10",
    "OrderUur_11", "OrderUur_12", "OrderUur_13", "OrderUur_14", "OrderUur_15",
    "OrderUur_16", "OrderUur_17", "OrderUur_18", "OrderUur_19", "OrderUur_20",
    "OrderUur_21", "OrderUur_22", "OrderUur_23", "Picktime_0", "Picktime_1",
    "Picktime_2", "Picktime_3", "Picktime_4", "Picktime_5", "Picktime_6", "Picktime_7",
    "Picktime_8", "Picktime_9", "Picktime_10", "Picktime_11", "Picktime_12",
    "Picktime_13", "Picktime_14", "Picktime_15", "Picktime_16", "Picktime_17",
    "Picktime_18", "Picktime_19", "Picktime_20", "Picktime_21", "Picktime_22", "Picktime_23"
]
# --- ADDED: These constants must match the trainer configuration ---
PREDICTION_HORIZON = 7
LOOKBACK_WINDOW = 400 if not DEBUG_MODE else 10
MIN_ENCODER_LENGTH = LOOKBACK_WINDOW


# ======================================================================================
# --- Helper Functions (Unchanged) ---
# ======================================================================================
def find_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR, debug=False):
    prefix = "debug-model" if debug else "best-model"
    print(f"Searching for the latest '{prefix}' checkpoint in '{checkpoint_dir}'...")
    search_pattern = os.path.join(checkpoint_dir, f'*{prefix}*.ckpt')
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        print(f"--- ERROR: No checkpoints found with pattern: {search_pattern}")
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Found latest checkpoint: {latest_file}")
    return latest_file


def calculate_metrics(df, group_by_col=None):
    metrics_list = []
    df['actual'] = pd.to_numeric(df['actual'], errors='coerce')
    df['predicted'] = pd.to_numeric(df['predicted'], errors='coerce')
    df.dropna(subset=['actual', 'predicted'], inplace=True)
    if group_by_col:
        for group, group_df in df.groupby(group_by_col):
            if group_df.empty: continue
            mape_df = group_df[group_df['actual'] != 0]
            r2_val = r2_score(group_df['actual'], group_df['predicted']) if len(group_df) > 1 else np.nan
            metrics_list.append(
                {group_by_col: group, 'MAE': mean_absolute_error(group_df['actual'], group_df['predicted']),
                 'MAPE': mean_absolute_percentage_error(mape_df['actual'],
                                                        mape_df['predicted']) if not mape_df.empty else np.nan,
                 'R2': r2_val})
        return pd.DataFrame(metrics_list).set_index(group_by_col)
    else:
        mape_df = df[df['actual'] != 0]
        r2_val = r2_score(df['actual'], df['predicted']) if len(df) > 1 else np.nan
        return pd.DataFrame([{'MAE': mean_absolute_error(df['actual'], df['predicted']),
                              'MAPE': mean_absolute_percentage_error(mape_df['actual'], mape_df[
                                  'predicted']) if not mape_df.empty else np.nan, 'R2': r2_val}], index=['Overall'])


# ======================================================================================
# --- Main Evaluation Function ---
# ======================================================================================
def evaluate_model():
    print("\n--- Evaluation Script Started ---")
    latest_checkpoint_path = find_latest_checkpoint(debug=DEBUG_MODE)
    if not latest_checkpoint_path:
        sys.exit(1)

    try:
        print(f"1. Loading model from '{latest_checkpoint_path}'...")
        best_tft_model = TemporalFusionTransformer.load_from_checkpoint(latest_checkpoint_path)
        best_tft_model.eval()
        print("   Model loaded successfully.")

        print(f"2. Loading and preparing latest data from '{LATEST_DATA_FILE}'...")
        new_df = pd.read_csv(LATEST_DATA_FILE, sep=';', decimal=',')
        new_df[TIME_COLUMN] = pd.to_datetime(new_df[TIME_COLUMN])
        new_df[SERIES_COLUMN] = new_df[SERIES_COLUMN].astype(str).astype("category")
        min_date = new_df[TIME_COLUMN].min()
        new_df['time_idx'] = (new_df[TIME_COLUMN] - min_date).dt.days
        for col in TARGET_COLUMNS:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce').astype("float32")
        new_df[TARGET_COLUMNS] = new_df[TARGET_COLUMNS].fillna(0.0)

        print("3a. Reconstructing dataset from model parameters...")
        reference_dataset = TimeSeriesDataSet.from_parameters(
            best_tft_model.dataset_parameters,
            new_df,
        )
        print("   Reference dataset created.")

        print("3b. Identifying valid series for prediction...")
        series_encoder = reference_dataset.categorical_encoders[SERIES_COLUMN]
        valid_series = series_encoder.classes_
        print(f"   Model was trained on {len(valid_series)} unique series. Generating forecasts for these only.")

        prediction_input_df = new_df[new_df[SERIES_COLUMN].isin(valid_series)]
        prediction_input_df = prediction_input_df.groupby(SERIES_COLUMN).tail(LOOKBACK_WINDOW)

        print("3c. Generating forecast for the final 7-day period...")
        predictions = best_tft_model.predict(
            prediction_input_df,
            trainer_kwargs=dict(accelerator="cpu"),
            mode="raw",
            return_x=True
        )

        prediction_list = predictions.output['prediction']
        if predictions is None or not prediction_list or prediction_list[0].size(0) == 0:
            print("\n--- WARNING: No predictions were generated. ---")
            print(
                "This usually means the evaluation data does not contain any time series that are long enough for the model's lookback window.")
            sys.exit(0)

        print("\n4. Reshaping data and calculating metrics...")
        median_prediction_idx = 3
        predicted_values = prediction_list
        actuals_lookup_df = new_df.set_index([SERIES_COLUMN, 'time_idx'])
        results_list = []

        index_df = reference_dataset.x_to_index(predictions.x)
        names = index_df[SERIES_COLUMN]
        last_encoder_time_idx = index_df['time_idx']

        for i, target_name in enumerate(TARGET_COLUMNS):
            median_preds_for_target = predicted_values[i][:, :, median_prediction_idx].cpu().numpy()
            for sample_idx in range(len(names)):
                current_name = names.iloc[sample_idx]
                current_time_idx = last_encoder_time_idx.iloc[sample_idx]

                for day_idx in range(median_preds_for_target.shape[1]):
                    forecast_time_idx = current_time_idx + day_idx + 1
                    forecast_date = min_date + pd.to_timedelta(forecast_time_idx, unit='D')
                    try:
                        actual_value = actuals_lookup_df.loc[(current_name, forecast_time_idx), target_name]
                    except KeyError:
                        actual_value = np.nan
                    results_list.append(
                        {'forecast_date': forecast_date, 'target_variable': target_name, 'actual': actual_value,
                         'predicted': median_preds_for_target[sample_idx, day_idx], SERIES_COLUMN: current_name,
                         'prediction_day': day_idx + 1})

        results_df = pd.DataFrame(results_list)

        if results_df.empty:
            print("\n--- No valid results to calculate metrics. ---")
            sys.exit(0)

        # --- METRICS ENHANCEMENT ---
        # Create a new column to group target variables
        def get_target_group(target_name):
            if target_name.startswith('OrderUur'):
                return 'OrderUur'
            elif target_name.startswith('Picktime'):
                return 'Picktime'
            elif target_name == 'OrderDag':
                return 'OrderDag'
            return 'Other'

        results_df['target_group'] = results_df['target_variable'].apply(get_target_group)

        metrics_per_customer = calculate_metrics(results_df, group_by_col=SERIES_COLUMN)
        metrics_per_target_group = calculate_metrics(results_df, group_by_col='target_group')
        overall_metrics = calculate_metrics(results_df)

        print("\n--- METRICS PER TARGET GROUP ---\n", metrics_per_target_group.to_string())
        print("\n--- METRICS PER CUSTOMER ---\n", metrics_per_customer.to_string())
        print("\n--- OVERALL METRICS ---\n", overall_metrics.to_string())

        output_excel_path = "latest_forecast_by_customer.xlsx"
        print(f"\n5. Saving detailed forecast report to '{output_excel_path}'...")
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            overall_metrics.to_excel(writer, sheet_name="Overall_Metrics")
            metrics_per_customer.to_excel(writer, sheet_name="Metrics_Per_Customer")
            metrics_per_target_group.to_excel(writer, sheet_name="Metrics_Per_Target_Group")

            target_col_categorical = pd.CategoricalDtype(TARGET_COLUMNS, ordered=True)
            results_df['target_variable'] = results_df['target_variable'].astype(target_col_categorical)

            for customer_name, customer_df in results_df.groupby(SERIES_COLUMN):
                pivot_df = customer_df.pivot_table(index='forecast_date', columns='target_variable',
                                                   values=['actual', 'predicted'])
                pivot_df = pivot_df.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
                safe_sheet_name = "".join(x for x in str(customer_name) if x.isalnum())[:31]
                pivot_df.to_excel(writer, sheet_name=safe_sheet_name)
        print("   Excel report saved successfully.")

    except Exception as e:
        import traceback
        print("\n" + "=" * 50 + "\n    CRITICAL ERROR DURING EVALUATION\n" + "=" * 50)
        print(f"Error details: {e}\n\nTraceback:")
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Evaluation Script Finished Successfully ---")


if __name__ == "__main__":
    evaluate_model()
