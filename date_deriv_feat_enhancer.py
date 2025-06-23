"""
Datetime Features Library

This library provides functions to add cyclical datetime features (sin/cos transformations)
to datasets containing datetime columns. Cyclical encoding is useful for machine learning
models to understand the cyclical nature of time-based features.

Features supported:
- Day of week (0-6, where Monday=0)
- Day of month (1-31)
- Day of year (1-366)

Usage:
    import pandas as pd
    from datetime_features import add_cyclical_datetime_features
    
    df = pd.DataFrame({'datetime': ['08/03/2025 00:00:00', '09/03/2025 12:30:00']})
    df_with_features = add_cyclical_datetime_features(df, 'datetime')
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional


def add_cyclical_datetime_features(
    df: pd.DataFrame, 
    datetime_col: str, 
    features: Optional[List[str]] = None,
    drop_original: bool = False
) -> pd.DataFrame:
    """
    Add cyclical datetime features (sin/cos) to a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing datetime column
        datetime_col (str): Name of the datetime column
        features (List[str], optional): List of features to add. 
            Options: ['day_of_week', 'day_of_month', 'day_of_year']
            If None, adds all features.
        drop_original (bool): Whether to drop the original datetime column
        
    Returns:
        pd.DataFrame: DataFrame with added cyclical features
        
    Example:
        >>> df = pd.DataFrame({'datetime': ['08/03/2025 00:00:00', '09/03/2025 12:30:00']})
        >>> df_with_features = add_cyclical_datetime_features(df, 'datetime')
        >>> print(df_with_features.columns.tolist())
        ['datetime', 'day_of_week_sin', 'day_of_week_cos', 'day_of_month_sin', 
         'day_of_month_cos', 'day_of_year_sin', 'day_of_year_cos']
    """
    # Copy the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(result_df[datetime_col]):
        result_df[datetime_col] = pd.to_datetime(result_df[datetime_col])
    
    # Default to all features if none specified
    if features is None:
        features = ['day_of_week', 'day_of_month', 'day_of_year']
    
    # Add day of week features (0-6, Monday=0)
    if 'day_of_week' in features:
        day_of_week = result_df[datetime_col].dt.dayofweek
        result_df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        result_df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Add day of month features (1-31)
    if 'day_of_month' in features:
        day_of_month = result_df[datetime_col].dt.day
        # Normalize to 0-based for cyclical encoding
        result_df['day_of_month_sin'] = np.sin(2 * np.pi * (day_of_month - 1) / 31)
        result_df['day_of_month_cos'] = np.cos(2 * np.pi * (day_of_month - 1) / 31)
    
    # Add day of year features (1-366)
    if 'day_of_year' in features:
        day_of_year = result_df[datetime_col].dt.dayofyear
        # Normalize to 0-based for cyclical encoding
        result_df['day_of_year_sin'] = np.sin(2 * np.pi * (day_of_year - 1) / 366)
        result_df['day_of_year_cos'] = np.cos(2 * np.pi * (day_of_year - 1) / 366)
    
    # Drop original datetime column if requested
    if drop_original:
        result_df = result_df.drop(columns=[datetime_col])
    
    return result_df


def add_single_cyclical_feature(
    values: Union[pd.Series, np.ndarray], 
    max_value: int, 
    feature_name: str,
    zero_based: bool = True
) -> pd.DataFrame:
    """
    Add sin/cos features for a single cyclical variable.
    
    Args:
        values: Array-like of values to transform
        max_value: Maximum value in the cycle (e.g., 7 for days of week, 366 for days of year)
        feature_name: Base name for the features (e.g., 'day_of_week')
        zero_based: Whether the input values are 0-based (True) or 1-based (False)
        
    Returns:
        pd.DataFrame: DataFrame with sin and cos columns
        
    Example:
        >>> days = [0, 1, 2, 3, 4, 5, 6]  # Monday to Sunday
        >>> cyclical_features = add_single_cyclical_feature(days, 7, 'day_of_week')
    """
    if not zero_based:
        # Convert 1-based to 0-based
        values = np.array(values) - 1
    
    sin_values = np.sin(2 * np.pi * values / max_value)
    cos_values = np.cos(2 * np.pi * values / max_value)
    
    return pd.DataFrame({
        f'{feature_name}_sin': sin_values,
        f'{feature_name}_cos': cos_values
    })


def get_datetime_info(datetime_series: pd.Series) -> pd.DataFrame:
    """
    Extract basic datetime information from a pandas Series.
    
    Args:
        datetime_series: Pandas Series containing datetime values
        
    Returns:
        pd.DataFrame: DataFrame with extracted datetime components
        
    Example:
        >>> dates = pd.to_datetime(['08/03/2025 00:00:00', '09/03/2025 12:30:00'])
        >>> info = get_datetime_info(dates)
    """
    if not pd.api.types.is_datetime64_any_dtype(datetime_series):
        datetime_series = pd.to_datetime(datetime_series)
    
    return pd.DataFrame({
        'year': datetime_series.dt.year,
        'month': datetime_series.dt.month,
        'day': datetime_series.dt.day,
        'hour': datetime_series.dt.hour,
        'minute': datetime_series.dt.minute,
        'second': datetime_series.dt.second,
        'day_of_week': datetime_series.dt.dayofweek,  # Monday=0
        'day_of_year': datetime_series.dt.dayofyear,
        'week_of_year': datetime_series.dt.isocalendar().week,
        'is_weekend': datetime_series.dt.dayofweek.isin([5, 6])  # Saturday=5, Sunday=6
    })


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'datetime': [
            '08/03/2025 00:00:00',
            '09/03/2025 12:30:00',
            '15/03/2025 18:45:00',
            '01/01/2025 00:00:00',
            '31/12/2025 23:59:59'
        ],
        'value': [1, 2, 3, 4, 5]
    })
    
    print("Original data:")
    print(sample_data)
    print("\n" + "="*50 + "\n")
    
    # Add all cyclical features
    enhanced_data = add_cyclical_datetime_features(sample_data, 'datetime')
    print("Data with cyclical features:")
    print(enhanced_data)
    print(f"\nColumns: {enhanced_data.columns.tolist()}")
    print("\n" + "="*50 + "\n")
    
    # Add only specific features
    specific_features = add_cyclical_datetime_features(
        sample_data, 
        'datetime', 
        features=['day_of_week', 'day_of_year']
    )
    print("Data with specific features:")
    print(specific_features)
    print(f"\nColumns: {specific_features.columns.tolist()}")
    print("\n" + "="*50 + "\n")
    
    # Show datetime info
    datetime_info = get_datetime_info(sample_data['datetime'])
    print("Datetime information:")
    print(datetime_info)
