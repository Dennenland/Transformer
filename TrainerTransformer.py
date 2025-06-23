import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss
import warnings
import time
import psutil

# --- Import the centralized configuration ---
from config_loader import config

# --- Suppress pandas PerformanceWarning ---
warnings.filterwarnings(
    "ignore", "DataFrame is highly fragmented", category=pd.errors.PerformanceWarning
)

print(f"Available RAM: {psutil.virtual_memory().available / 1024 ** 3:.1f} GB")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
else:
    print("GPU not available.")


# ======================================================================================
# --- Main Training Function ---
# ======================================================================================
def train_forecasting_model():
    """
    Main function to orchestrate the model training process using settings
    from the centralized config file.
    """
    # Unpack config sections for easier access
    cfg_data = config['data']
    cfg_model = config['model']
    cfg_train = config['training']
    is_debug = config['DEBUG_MODE']

    torch.set_float32_matmul_precision(cfg_train['torch_precision'])
    pl.seed_everything(42, workers=True)
    start_time = time.time()
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print("--- Training Script Started ---")
    if is_debug:
        print("🐛 DEBUG MODE ENABLED: Running with minimal configuration.")

    print(f"1. Loading and preparing training data from '{cfg_data['train_file_path']}'...")
    df = pd.read_csv(cfg_data['train_file_path'], sep=';', decimal=',')
    df[cfg_data['time_column']] = pd.to_datetime(df[cfg_data['time_column']])
    df[cfg_data['series_column']] = df[cfg_data['series_column']].astype(str).astype("category")
    df['time_idx'] = (df[cfg_data['time_column']] - df[cfg_data['time_column']].min()).dt.days

    for col in cfg_data['target_columns']:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype("float32")
    df[cfg_data['target_columns']] = df[cfg_data['target_columns']].fillna(0.0)

    # --- REMOVED MANUAL DEBUG SLICING ---
    # The debug mode is now controlled by the trainer parameters (limit_train_batches)
    # which is a more robust approach.

    print(f"   Data loaded. Shape: {df.shape}")
    print("2. Defining the TimeSeriesDataSet...")

    training_data = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= x.time_idx.max() - cfg_model['prediction_horizon']],
        time_idx="time_idx",
        target=cfg_data['target_columns'],
        group_ids=[cfg_data['series_column']],
        max_encoder_length=cfg_model['lookback_window'],
        min_encoder_length=cfg_model['min_encoder_length'],
        max_prediction_length=cfg_model['prediction_horizon'],
        static_categoricals=[cfg_data['series_column']],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=cfg_data['target_columns'],
        target_normalizer=MultiNormalizer(
            [GroupNormalizer(groups=[cfg_data['series_column']]) for _ in cfg_data['target_columns']]
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    train_dataloader = training_data.to_dataloader(
        train=True,
        batch_size=cfg_train['batch_size'],
        num_workers=cfg_train['num_workers']
    )
    val_dataloader = TimeSeriesDataSet.from_dataset(
        training_data, df, predict=False, stop_randomization=True
    ).to_dataloader(
        train=False,
        batch_size=cfg_train['val_batch_size'],
        num_workers=cfg_train['num_workers']
    )

    print("3. Configuring the Trainer...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg_model['checkpoint_dir'],
        filename=f"{timestamp}-{'debug' if is_debug else 'best'}-model-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    early_stopping_callback = EarlyStopping(**cfg_train['early_stopping'])

    trainer_params = {
        "accelerator": cfg_train['accelerator'],
        "devices": cfg_train['devices'],
        "callbacks": [checkpoint_callback, early_stopping_callback]
    }
    if is_debug:
        trainer_params.update(cfg_train['debug_run_params'])
    else:
        # For full runs, max_epochs can be set to -1 for unlimited epochs (relying on early stopping)
        trainer_params["max_epochs"] = -1

    trainer = pl.Trainer(**trainer_params)

    print("4. Instantiating the TemporalFusionTransformer model...")
    tft_model = TemporalFusionTransformer.from_dataset(
        training_data,
        learning_rate=cfg_train['learning_rate'],
        optimizer=cfg_train['optimizer'],
        **cfg_model['hyperparameters'],
        output_size=[7] * len(cfg_data['target_columns']),
        loss=MultiLoss([QuantileLoss() for _ in range(len(cfg_data['target_columns']))]),
    )

    print(f"Starting {'debug' if is_debug else 'full'} model training run...")
    trainer.fit(tft_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    end_time = time.time()
    print(f"\n--- Training Script Finished --- (Total time: {end_time - start_time:.2f} seconds)")
    print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train_forecasting_model()
