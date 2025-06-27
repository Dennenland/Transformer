import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
import warnings
import os
import sys
import glob
import numpy as np
from tqdm import tqdm

# --- Import the centralized configuration ---
from config_loader import config
# --- ADDED: Import the feature enhancer ---
from date_deriv_feat_enhancer import add_cyclical_datetime_features


# --- Suppress pandas PerformanceWarning ---
warnings.filterwarnings(
    "ignore", "DataFrame is highly fragmented", category=pd.errors.PerformanceWarning
)


# ======================================================================================
# --- Helper Functions ---
# ======================================================================================
def find_latest_checkpoint(checkpoint_dir, debug=False):
    """Finds the most recent checkpoint file in the given directory."""
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
    """Calculates MAE, MAPE, and R2 score, with optional grouping."""
    metrics_list = []
    df['actual'] = pd.to_numeric(df['actual'], errors='coerce')
    df['predicted'] = pd.to_numeric(df['predicted'], errors='coerce')
    df.dropna(subset=['actual', 'predicted'], inplace=True)
    if group_by_col:
        for group, group_df in df.groupby(group_by_col, observed=True):
            if group_df.empty: continue
            mape_df = group_df[group_df['actual'] != 0]
            r2_val = r2_score(group_df['actual'], group_df['predicted']) if len(group_df) > 1 else np.nan
            metrics_list.append({
                group_by_col: group,
                'MAE': mean_absolute_error(group_df['actual'], group_df['predicted']),
                'MAPE': mean_absolute_percentage_error(mape_df['actual'],
                                                       mape_df['predicted']) if not mape_df.empty else np.nan,
                'R2': r2_val
            })
        return pd.DataFrame(metrics_list).set_index(group_by_col)
    else:
        mape_df = df[df['actual'] != 0]
        r2_val = r2_score(df['actual'], df['predicted']) if len(df) > 1 else np.nan
        return pd.DataFrame([{
            'MAE': mean_absolute_error(df['actual'], df['predicted']),
            'MAPE': mean_absolute_percentage_error(mape_df['actual'],
                                                   mape_df['predicted']) if not mape_df.empty else np.nan,
            'R2': r2_val
        }], index=['Overall'])


# ======================================================================================
# --- Main Evaluation Function ---
# ======================================================================================
def evaluate_model():
    """
    Main function to load the latest model and evaluate its performance
    using settings from the centralized config file.
    """
    print("\n--- Evaluation Script Started ---")

    # Unpack config sections for easier access
    cfg_data = config['data']
    cfg_model = config['model']
    cfg_eval = config['evaluation']
    cfg_train = config['training']  # Needed for batch_size
    is_debug = config['DEBUG_MODE']

    latest_checkpoint_path = find_latest_checkpoint(cfg_model['checkpoint_dir'], debug=is_debug)
    if not latest_checkpoint_path:
        sys.exit(1)

    try:
        print(f"1. Loading model from '{latest_checkpoint_path}'...")
        best_tft_model = TemporalFusionTransformer.load_from_checkpoint(latest_checkpoint_path)
        best_tft_model.eval()
        print("   Model loaded successfully.")

        print(f"2. Loading and preparing evaluation data from '{cfg_data['eval_file_path']}'...")
        new_df = pd.read_csv(cfg_data['eval_file_path'], sep=';', decimal=',')
        new_df[cfg_data['time_column']] = pd.to_datetime(new_df[cfg_data['time_column']])

        print("2a. Enhancing evaluation data with cyclical datetime features...")
        new_df = add_cyclical_datetime_features(new_df, datetime_col=cfg_data['time_column'])
        print("    Cyclical features added.")
        
        new_df[cfg_data['series_column']] = new_df[cfg_data['series_column']].astype(str)
        min_date = new_df[cfg_data['time_column']].min()
        new_df['time_idx'] = (new_df[cfg_data['time_column']] - min_date).dt.days
        for col in cfg_data['target_columns']:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce').astype("float32")
        new_df[cfg_data['target_columns']] = new_df[cfg_data['target_columns']].fillna(0.0)

        # --- FIX: Handle new groups by checking for sufficient history ---
        print("2b. Handling new groups in evaluation data...")
        encoder_length = best_tft_model.dataset_parameters['max_encoder_length']
        known_groups = list(best_tft_model.dataset_parameters["static_categoricals_encoders"][cfg_data['series_column']].classes_)
        
        all_eval_groups = new_df[cfg_data['series_column']].unique()
        new_groups = [group for group in all_eval_groups if group not in known_groups]
        
        valid_new_groups = []
        ignored_new_groups = []

        if new_groups:
            print(f"   Found new groups not in training data: {new_groups}. Checking for sufficient history...")
            for group in new_groups:
                group_history_length = len(new_df[new_df[cfg_data['series_column']] == group])
                if group_history_length >= encoder_length:
                    valid_new_groups.append(group)
                else:
                    ignored_new_groups.append(group)
            
            if valid_new_groups:
                print(f"   OK to proceed with new groups: {valid_new_groups}")
                # Add valid new groups to the model's known categories in-memory
                updated_known_groups = known_groups + valid_new_groups
                best_tft_model.dataset_parameters["static_categoricals_encoders"][cfg_data['series_column']].classes_ = updated_known_groups
            
            if ignored_new_groups:
                print(f"   WARNING: Ignoring new groups with insufficient history (need >= {encoder_length} records): {ignored_new_groups}")

        else:
            print("   No new groups found. All groups in evaluation data are known to the model.")

        # Filter the dataframe to only include known groups and valid new groups
        groups_to_process = known_groups + valid_new_groups
        evaluation_df = new_df[new_df[cfg_data['series_column']].isin(groups_to_process)].copy()
        evaluation_df[cfg_data['series_column']] = evaluation_df[cfg_data['series_column']].astype("category")
        
        # --- END OF FIX ---

        if is_debug:
            print("   Debug Mode: Evaluating model trained on a limited number of batches.")

        print(f"4. Generating forecast for the next {cfg_model['prediction_horizon']}-day period...")
        predictions_output, index_df = best_tft_model.predict(
            evaluation_df,  # Use the filtered dataframe
            mode="raw",
            return_index=True,
            trainer_kwargs=dict(accelerator=cfg_eval['accelerator']),
            batch_size=cfg_train['val_batch_size'] * 2
        )
        print("   Forecasts generated.")

        prediction_list = predictions_output['prediction']
        if not prediction_list or prediction_list[0].size(0) == 0:
            print("\n--- WARNING: No predictions were generated. ---")
            sys.exit(0)

        print("\n5. Reshaping data and calculating metrics...")
        median_prediction_idx = 3
        actuals_lookup_df = evaluation_df.set_index([cfg_data['series_column'], 'time_idx'])
        results_list = []

        names = index_df[cfg_data['series_column']]
        last_encoder_time_idx = index_df['time_idx']

        for i, target_name in enumerate(tqdm(cfg_data['target_columns'], desc="Processing Predictions")):
            median_preds_for_target = prediction_list[i][:, :, median_prediction_idx].cpu().numpy()
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
                    results_list.append({
                        'forecast_date': forecast_date,
                        'target_variable': target_name,
                        'actual': actual_value,
                        'predicted': median_preds_for_target[sample_idx, day_idx],
                        cfg_data['series_column']: current_name,
                        'prediction_day': day_idx + 1
                    })

        results_df = pd.DataFrame(results_list)

        if results_df.empty:
            print("\n--- No valid results to calculate metrics. ---")
            sys.exit(0)

        results_df['target_group'] = results_df['target_variable'].apply(
            lambda x: 'OrderUur' if x.startswith('OrderUur') else (
                'Picktime' if x.startswith('Picktime') else 'OrderDag')
        )

        metrics_per_customer = calculate_metrics(results_df, group_by_col=cfg_data['series_column'])
        metrics_per_target_group = calculate_metrics(results_df, group_by_col='target_group')
        metrics_per_prediction_day = calculate_metrics(results_df, group_by_col='prediction_day')
        overall_metrics = calculate_metrics(results_df)

        print("\n--- METRICS PER TARGET GROUP ---\n", metrics_per_target_group.to_string())
        print("\n--- METRICS PER CUSTOMER ---\n", metrics_per_customer.to_string())
        print("\n--- METRICS PER PREDICTION DAY ---\n", metrics_per_prediction_day.to_string())
        print("\n--- OVERALL METRICS ---\n", overall_metrics.to_string())

        output_excel_path = cfg_eval['output_report_name']
        print(f"\n6. Saving detailed forecast report to '{output_excel_path}'...")
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            overall_metrics.to_excel(writer, sheet_name="Overall_Metrics")
            metrics_per_customer.to_excel(writer, sheet_name="Metrics_Per_Customer")
            metrics_per_target_group.to_excel(writer, sheet_name="Metrics_Per_Target_Group")
            metrics_per_prediction_day.to_excel(writer, sheet_name="Metrics_Per_Prediction_Day")

            target_col_categorical = pd.CategoricalDtype(cfg_data['target_columns'], ordered=True)
            results_df['target_variable'] = results_df['target_variable'].astype(target_col_categorical)

            grouped_customers = results_df.groupby(cfg_data['series_column'], observed=True)
            for customer_name, customer_df in tqdm(grouped_customers, desc="Writing Customer Sheets"):
                pivot_df = customer_df.pivot_table(
                    index='forecast_date', columns='target_variable', values=['actual', 'predicted']
                )
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
