import pandas as pd
import os


def load_raw_data(filepath):
    """
    Load the raw CSV dataset from disk.
    """
    filepath = os.path.join("data", "GlobalWeatherRepository.csv")
    df = pd.read_csv(filepath)
    return df


def standardize_columns(df):
    """
    Standardize column names:
    - Remove leading/trailing spaces
    - Convert to lowercase
    - Replace spaces and hyphens with underscores

    This ensures consistency when referencing columns later.
    """
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def parse_datetime_local(df, datetime_col="last_updated"):
    """
    Convert the 'last_updated' column to datetime format.
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    # Drop rows where datetime conversion failed
    df = df.dropna(subset=[datetime_col])

    return df

def parse_datetime_unix(df, datetime_col="last_updated_epoch"):
    """
    Convert Unix epoch timestamp (seconds) into UTC datetime.
    """
    # Convert epoch (seconds) → UTC datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col], unit="s", utc=True)

    return df


def drop_duplicates(df):
    """
    Remove duplicate rows.
    """
    return df.drop_duplicates().reset_index(drop=True)


def create_timeseries_index(df, city_col="location_name", country_col="country", datetime_col="last_updated_epoch"):
    """
    Create a proper time-series multi-index:
    (city, country, datetime)
    """

    required_cols = [city_col, datetime_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found.")

    if country_col in df.columns:
        df = df.set_index([city_col, country_col, datetime_col])
    else:
        df = df.set_index([city_col, datetime_col])

    df = df.sort_index()

    return df