# mysklearn/mypreprocessing.py

import pandas as pd
import numpy as np

from mysklearn.label_mapping import crime_mapping

# ==========================================================
#  Crime Data Preprocessing
# ==========================================================

def load_raw_crime_data(csv_path):
    """Load the LAPD crime dataset from a CSV file."""
    return pd.read_csv(csv_path, low_memory=False)


def extract_features(df, drop_other=True):
    """
    Select and clean the subset of features we want to use in classification.
    Adds a grouped 'Crime Category' label column using crime_mapping.

    Returns a cleaned dataframe of features + label.
    """
    # Keep only relevant columns (from the raw LAPD dataset)
    cols = [
        "Vict Age",
        "Vict Sex",
        "Vict Descent",
        "Premis Desc",
        "AREA NAME",
        "TIME OCC",
        "Crm Cd Desc",   # raw detailed crime description (will map to category)
    ]
    df = df[cols].copy()

    # Drop rows missing the label
    df = df.dropna(subset=["Crm Cd Desc"])

    # Fill missing categorical values
    cat_fill = {
        "Vict Sex": "Unknown",
        "Vict Descent": "Unknown",
        "Premis Desc": "Unknown",
        "AREA NAME": "Unknown"
    }
    df = df.fillna(cat_fill)

    # ============================
    # Victim age: clean + bin
    # ============================
    # Replace invalid ages with NaN
    df["Vict Age"] = df["Vict Age"].apply(
        lambda x: np.nan if pd.isna(x) or x < 0 or x > 120 else x
    )
    # Fill missing with median
    df["Vict Age"] = df["Vict Age"].fillna(df["Vict Age"].median())

    # Discretize victim age into bins
    df["Vict Age Group"] = pd.cut(
        df["Vict Age"],
        bins=[0, 12, 19, 35, 50, 65, 200],
        labels=["child", "teen", "young adult", "adult", "middle age", "senior"]
    )

    # ============================
    # Time of occurrence -> part of day
    # ============================
    def time_to_hour(t):
        # TIME OCC is like 30, 945, 1630 -> we zero-pad and grab the hour
        t_int = int(t)
        return int(str(t_int).zfill(4)[:2])

    df["Hour"] = df["TIME OCC"].apply(time_to_hour)

    df["Time Period"] = pd.cut(
        df["Hour"],
        bins=[0, 5, 12, 17, 21, 24],
        labels=["late night", "morning", "afternoon", "evening", "night"],
        right=False
    )

    # ============================
    # Crime Category (label)
    # ============================
    df["Crime Category"] = df["Crm Cd Desc"].map(crime_mapping).fillna("Other")

    if drop_other:
        # Drop rows where we mapped to "Other" to keep classes cleaner
        df = df[df["Crime Category"] != "Other"].copy()

    # ============================
    # Drop raw columns we no longer need
    # ============================
    df = df.drop(columns=["TIME OCC", "Vict Age", "Hour", "Crm Cd Desc"])

    # Final column order just for readability
    df = df[
        [
            "Vict Sex",
            "Vict Descent",
            "Premis Desc",
            "AREA NAME",
            "Vict Age Group",
            "Time Period",
            "Crime Category",
        ]
    ]

    return df


def build_X_y(df, label_col="Crime Category"):
    """
    Convert a cleaned dataframe into X (features) and y (labels)
    as plain Python lists, for use with mysklearn classifiers.
    """
    X_df = df.drop(columns=[label_col])
    y_series = df[label_col]

    X = X_df.values.tolist()
    y = y_series.tolist()
    return X, y
