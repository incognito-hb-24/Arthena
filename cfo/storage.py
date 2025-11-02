import os
import pandas as pd

DATA_DIR = os.path.join("data")
os.makedirs(DATA_DIR, exist_ok=True)

HISTORY_PATH = os.path.join(DATA_DIR, "user_history.csv")

COLS = ["Date", "Module", "Type", "Category", "Entity", "Amount (₹)", "Notes"]

def load_history():
    """Load existing history CSV or return empty DataFrame."""
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    else:
        return pd.DataFrame(columns=COLS)

def save_history(df: pd.DataFrame):
    """Save current session history."""
    df.to_csv(HISTORY_PATH, index=False)

def get_history_path():
    return HISTORY_PATH
