import pandas as pd

import torch

import lightning.pytorch as pl

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

import warnings

import os

import time

import shutil

from typing import Dict, Any

import psutil

print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")

print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")



# --- Suppress pandas PerformanceWarning ---

warnings.filterwarnings(

    "ignore", "DataFrame is highly fragmented", category=pd.errors.PerformanceWarning

)



# ======================================================================================

# --- DEBUG MODE TOGGLE ---

# ======================================================================================

DEBUG_MODE = False



# ======================================================================================

# --- CONFIGURATION (Original, Unchanged) ---

# ======================================================================================

DATA_FILE = 'Prime 1.csv'

CHECKPOINT_DIR = "model_checkpoints"

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



PREDICTION_HORIZON = 7

LOOKBACK_WINDOW = 400 if not DEBUG_MODE else 10

MIN_ENCODER_LENGTH = LOOKBACK_WINDOW



MODEL_HPARAMS = {

    "hidden_size": 128 if not DEBUG_MODE else 16,

    "attention_head_size": 4 if not DEBUG_MODE else 1,

    "dropout": 0.1,

    "hidden_continuous_size": 32 if not DEBUG_MODE else 8,

}



TRAINING_HPARAMS = {"batch_size": 64 if not DEBUG_MODE else 4, "num_workers": 0}

VAL_BATCH_SIZE = 128 if not DEBUG_MODE else 4

LEARNING_RATE = 0.001

OPTIMIZER_NAME = "AdamW"

EARLY_STOPPING_HPARAMS = {"monitor": "val_loss", "patience": 10 if not DEBUG_MODE else 1, "mode": "min",

                          "verbose": True}

ACCELERATOR = "auto"

DEVICES = "auto"

TORCH_PRECISION = 'medium'

DEBUG_TRAINING_PARAMS = {"max_epochs": 1, "limit_train_batches": 1, "limit_val_batches": 1}





# ======================================================================================

# --- NO WRAPPER ---

# Reverting to the original, robust approach. The TFT model is trained directly.

# This prevents the "ValueError: Found array with 0 sample(s)".

# ======================================================================================



# ======================================================================================

# --- Main Training Function ---

# ======================================================================================

def train_forecasting_model():

    torch.set_float32_matmul_precision(TORCH_PRECISION)

    pl.seed_everything(42, workers=True)

    start_time = time.time()

    timestamp = time.strftime("%Y%m%d-%H%M%S")



    print("--- Training Script Started ---")

    if DEBUG_MODE: print("🐛 DEBUG MODE ENABLED: Running with minimal configuration.")



    print("1. Loading and preparing data...")

    df = pd.read_csv(DATA_FILE, sep=';', decimal=',')

    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])

    df[SERIES_COLUMN] = df[SERIES_COLUMN].astype(str).astype("category")

    df['time_idx'] = (df[TIME_COLUMN] - df[TIME_COLUMN].min()).dt.days



    for col in TARGET_COLUMNS:

        df[col] = pd.to_numeric(df[col], errors='coerce').astype("float32")

    df[TARGET_COLUMNS] = df[TARGET_COLUMNS].fillna(0.0)



    if DEBUG_MODE:

        print("   Debug: Subsetting data for faster processing...")

        max_time_idx = df['time_idx'].max()

        df = df[df['time_idx'] >= max_time_idx - 50]

        unique_series = df[SERIES_COLUMN].unique()[:2]

        df = df[df[SERIES_COLUMN].isin(unique_series)]



    print(f"   Data loaded. Shape: {df.shape}")

    print("2. Defining the TimeSeriesDataSet...")

    training_data = TimeSeriesDataSet(

        df[lambda x: x.time_idx <= x.time_idx.max() - PREDICTION_HORIZON],

        time_idx="time_idx", target=TARGET_COLUMNS, group_ids=[SERIES_COLUMN],

        max_encoder_length=LOOKBACK_WINDOW, min_encoder_length=MIN_ENCODER_LENGTH,

        max_prediction_length=PREDICTION_HORIZON, static_categoricals=[SERIES_COLUMN],

        time_varying_known_reals=["time_idx"], time_varying_unknown_reals=TARGET_COLUMNS,

        target_normalizer=MultiNormalizer([GroupNormalizer(groups=[SERIES_COLUMN]) for _ in TARGET_COLUMNS]),

        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,

        allow_missing_timesteps=True

    )

    train_dataloader = training_data.to_dataloader(train=True, **TRAINING_HPARAMS)

    val_dataloader = TimeSeriesDataSet.from_dataset(training_data, df, predict=False,

                                                    stop_randomization=True).to_dataloader(

        train=False, batch_size=VAL_BATCH_SIZE, num_workers=TRAINING_HPARAMS["num_workers"])



    print("3. Configuring the Trainer...")

    checkpoint_callback = ModelCheckpoint(

        dirpath=CHECKPOINT_DIR,

        filename=f"{timestamp}-{'debug' if DEBUG_MODE else 'best'}-model-{{epoch:02d}}-{{val_loss:.2f}}",

        save_top_k=1, verbose=True, monitor="val_loss", mode="min",

    )

    early_stopping_callback = EarlyStopping(**EARLY_STOPPING_HPARAMS)



    trainer_params = {"accelerator": ACCELERATOR, "devices": DEVICES,

                      "callbacks": [checkpoint_callback, early_stopping_callback]}

    if DEBUG_MODE:

        trainer_params.update(DEBUG_TRAINING_PARAMS)

    else:

        trainer_params["max_epochs"] = -1

    trainer = pl.Trainer(**trainer_params)



    print("4. Instantiating the TemporalFusionTransformer model directly...")

    # --- REVERTED TO ORIGINAL, WORKING LOGIC ---

    # The model is created directly from the real training_data.

    # Its hyperparameters, including dataset parameters, are automatically saved in the checkpoint.

    tft_model = TemporalFusionTransformer.from_dataset(

        training_data,

        learning_rate=LEARNING_RATE,

        optimizer=OPTIMIZER_NAME,

        **MODEL_HPARAMS,

        output_size=[7] * len(TARGET_COLUMNS),

        loss=MultiLoss([QuantileLoss() for _ in range(len(TARGET_COLUMNS))]),

    )



    print(f"Starting {'debug' if DEBUG_MODE else 'full'} model training run...")

    trainer.fit(tft_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)



    end_time = time.time()

    print(f"\n--- Training Script Finished --- (Total time: {end_time - start_time:.2f} seconds)")

    print(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")





if __name__ == "__main__":

    train_forecasting_model()
