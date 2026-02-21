import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler



def missingness_summary(df):
    """
    Return a simple missingness table (count + percent) for each column.
    """
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df)) * 100

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct
    }).sort_values("missing_pct", ascending=False)

    return summary


def robust_zscore_outliers(series, z_thresh=3.5):
    """
    Robust z-score using median and MAD.
    Returns boolean mask where True indicates an outlier.
    """
    x = series.astype(float)

    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))

    # Avoid division by zero when MAD is 0
    if mad == 0 or np.isnan(mad):
        return pd.Series([False] * len(series), index=series.index)

    robust_z = 0.6745 * (x - median) / mad
    return np.abs(robust_z) > z_thresh

def flag_outliers(df, cols_to_check, z_thresh=3.5):
    """
    Add boolean outlier flags for selected columns.
    Example output columns:
      temp_c_outlier, precipitation_mm_outlier
    """
    df_out = df.copy()

    for col in cols_to_check:
        if col not in df_out.columns:
            continue
        flag_col = f"{col}_outlier"
        df_out[flag_col] = robust_zscore_outliers(df_out[col], z_thresh=z_thresh)

    return df_out

def apply_robust_scaler(df, cols_to_scale):
    """
     Fit a RobustScaler on selected columns.
    """
    df_out = df.copy()

    scaler = RobustScaler()
    scaler.fit(df[cols_to_scale])

    scaled = scaler.transform(df_out[cols_to_scale])
    for i, col in enumerate(cols_to_scale):
        df_out[f"{col}_scaled"] = scaled[:, i]

    return df_out


