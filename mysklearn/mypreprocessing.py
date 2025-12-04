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


def extract_features(df):
    """
    Select and clean the initial subset of features we want to use in classification.
    Returns a cleaned dataframe of features and the grouped target labels.
    """

    # Keep only relevant columns
    cols = ["Vict Age", "Vict Sex", "Vict Descent", "Premis Desc",
            "AREA NAME", "TIME OCC", "Crm Cd Desc"]
    df = df[cols].copy()

    # Drop rows missing the label
    df = df.dropna(subset=["Crm Cd Desc"])

    # Standardize label formatting (remove whitespace)
    df["Crm Cd Desc"] = df["Crm Cd Desc"].astype(str).str.strip()

    # Fill missing categorical values
    cat_fill = {"Vict Sex": "Unknown", "Vict Descent": "Unknown",
                "Premis Desc": "Unknown", "AREA NAME": "Unknown"}
    df = df.fillna(cat_fill)

    # Fix victim age (invalid or missing ages)
    df["Vict Age"] = df["Vict Age"].apply(lambda x: np.nan if x < 0 or x > 120 else x)
    df["Vict Age"] = df["Vict Age"].fillna(df["Vict Age"].median())

    # Discretize victim age into bins
    df["Vict Age Group"] = pd.cut(df["Vict Age"],
                                  bins=[0, 12, 19, 35, 50, 65, 200],
                                  labels=["child", "teen", "young adult", "adult",
                                          "middle age", "senior"])

    # Convert time of occurrence (e.g., 945 -> hour)
    df["Hour"] = df["TIME OCC"].apply(lambda t: int(str(int(t)).zfill(4)[:2]))
    df["Time Period"] = pd.cut(df["Hour"],
                               bins=[0, 5, 12, 17, 21, 24],
                               labels=["late night", "morning", "afternoon", "evening", "night"],
                               right=False)

    # Drop raw columns we replaced
    df = df.drop(columns=["TIME OCC", "Vict Age", "Hour"])

    # =======================================================
    # Apply crime grouping labels
    # =======================================================
    df["Crime Category"] = df["Crm Cd Desc"].map(crime_mapping)

    # Any missing mappings default to "Other"
    df["Crime Category"] = df["Crime Category"].fillna("Other")

    # Drop the original (unprocessed) label to avoid confusion
    df = df.drop(columns=["Crm Cd Desc"])

    return df
