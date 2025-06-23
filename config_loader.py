import yaml
from typing import Any, Dict
import os


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Loads the YAML configuration file and processes it to select the correct
    parameters based on the DEBUG_MODE flag.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        Dict[str, Any]: A dictionary containing the processed configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    is_debug = config.get('DEBUG_MODE', False)

    # --- Process Model Parameters ---
    model_cfg = config['model']
    model_cfg['lookback_window'] = model_cfg['lookback_window_debug'] if is_debug else model_cfg['lookback_window']
    model_cfg['min_encoder_length'] = model_cfg['lookback_window']  # Set min_encoder_length to match lookback

    if is_debug:
        model_cfg['hyperparameters'] = model_cfg['hyperparameters_debug']

    # Clean up debug-specific keys
    model_cfg.pop('lookback_window_debug', None)
    model_cfg.pop('hyperparameters_debug', None)

    # --- Process Training Parameters ---
    train_cfg = config['training']
    train_cfg['batch_size'] = train_cfg['batch_size_debug'] if is_debug else train_cfg['batch_size']
    train_cfg['val_batch_size'] = train_cfg['val_batch_size_debug'] if is_debug else train_cfg['val_batch_size']

    early_stop_cfg = train_cfg['early_stopping']
    early_stop_cfg['patience'] = early_stop_cfg['patience_debug'] if is_debug else early_stop_cfg['patience']
    early_stop_cfg.pop('patience_debug', None)

    # Clean up debug-specific keys
    train_cfg.pop('batch_size_debug', None)
    train_cfg.pop('val_batch_size_debug', None)

    return config


# Load the configuration once to be imported by other modules
try:
    config = load_config()
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure 'config.yaml' is in the same directory.")
    exit(1)

