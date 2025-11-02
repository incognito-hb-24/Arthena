import pandas as pd
import numpy as np

def summarize_context(df: pd.DataFrame) -> str:
    """Summarize key financial metrics for the AI assistant."""
    if df.empty:
        return "No transactions yet."
    inc = df[df["Type"] == "Income"]["Amount (₹)"].sum()
    exp = abs(df[df["Type"] == "Expense"]["Amount (₹)"]).sum()
    inv = abs(df[df["Type"] == "Investment"]["Amount (₹)"]).sum()
    net = inc - exp - inv
    return f"Income ₹{inc:.2f}, Expenses ₹{exp:.2f}, Investments ₹{inv:.2f}, Net Savings ₹{net:.2f}"

def compute_summary(df: pd.DataFrame):
    """Return structured dictionary for dashboard display."""
    if df.empty:
        return {"Income": 0, "Expenses": 0, "Investments": 0, "Net": 0}
    inc = df[df["Type"] == "Income"]["Amount (₹)"].sum()
    exp = abs(df[df["Type"] == "Expense"]["Amount (₹)"]).sum()
    inv = abs(df[df["Type"] == "Investment"]["Amount (₹)"]).sum()
    return {
        "Income": round(inc, 2),
        "Expenses": round(exp, 2),
        "Investments": round(inv, 2),
        "Net": round(inc - exp - inv, 2)
    }

def plot_type_distribution(df: pd.DataFrame):
    """Generate a matplotlib figure for transaction type distribution."""
    import matplotlib.pyplot as plt
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data yet", ha='center', va='center')
        ax.axis("off")
        return fig
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Type"].value_counts()
    ax.bar(counts.index, counts.values, color="skyblue")
    ax.set_title("Transaction Types Distribution")
    ax.grid(axis="y", alpha=0.3)
    return fig
