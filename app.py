import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from cfo.logic import summarize_context, compute_summary, plot_type_distribution
from cfo.storage import load_history, save_history, get_history_path, COLS

# -------------------------
# Page & Session bootstrap
# -------------------------
st.set_page_config(page_title="Arthena — Personal CFO", page_icon="💸", layout="wide")

if "df" not in st.session_state:
    st.session_state.df = load_history()
if "chat" not in st.session_state:
    st.session_state.chat = []

df = st.session_state.df  # convenience alias

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Arthena")
st.sidebar.caption("Personal CFO — budget, track, and learn")

with st.sidebar.expander("Storage"):
    st.markdown(f"**History CSV:** `{get_history_path()}`")
    if st.button("💾 Save now"):
        save_history(df)
        st.success("Saved.")
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="arthena_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if st.button("🧹 Clear (session only)"):
        st.session_state.df = pd.DataFrame(columns=COLS)
        st.session_state.chat = []
        st.experimental_rerun()

with st.sidebar.expander("Sample data"):
    if st.button("↳ Load sample rows"):
        sample = pd.read_csv("data/sample_transactions.csv")
        st.session_state.df = pd.concat([st.session_state.df, sample], ignore_index=True)
        save_history(st.session_state.df)
        st.success("Sample data added.")
        st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Tip: set your Gemini API key in **Secrets** as `ARTHENA` or env var `GOOGLE_API_KEY`.")

# -------------------------
# Header
# -------------------------
st.title("Arthena — Personal CFO")
st.markdown("Track your **cashflow & wealth**, see clear insights, and ask **AI** for personalized tips — all from your own data.")

# -------------------------
# Add Transaction
# -------------------------
st.subheader("Add a transaction")

modules = ["Wealth", "Cashflow"]
wealth_types = ["Asset", "Liability", "Goal", "Insurance"]
cashflow_types = ["Income", "Expense", "Investment", "Payment"]

with st.form("add_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        module = st.selectbox("Module", modules, index=1)  # default Cashflow
    with c2:
        t_opts = wealth_types if module == "Wealth" else cashflow_types
        typ = st.selectbox("Type", t_opts)
    with c3:
        category = st.text_input("Category (e.g., Rent, SIP, Salary, Groceries)")

    c4, c5, c6 = st.columns([2, 1, 1])
    with c4:
        entity = st.text_input("Entity (e.g., HDFC MF, Landlord, Swiggy)")
    with c5:
        amount = st.number_input("Amount (₹)", value=0.0, step=100.0, format="%.2f")
    with c6:
        dt = st.date_input("Date", value=date.today())

    notes = st.text_input("Notes")
    submitted = st.form_submit_button("➕ Add Entry", use_container_width=True)

if submitted:
    if entity.strip() == "" or amount == 0:
        st.warning("Please fill **Entity** and a non-zero **Amount (₹)**.")
    else:
        new_row = {
            "Date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
            "Module": module,
            "Type": typ,
            "Category": category,
            "Entity": entity,
            "Amount (₹)": amount,
            "Notes": notes,
        }
        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_row])], ignore_index=True)
        save_history(st.session_state.df)
        st.success("Entry added.")
        df = st.session_state.df  # refresh alias

# -------------------------
# Insights Dashboard
# -------------------------
st.subheader("Insights dashboard")

if df.empty:
    st.info("No data yet. Add your first entry above.")
else:
    # Headline metrics
    summary = compute_summary(df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Income (₹)", f"{summary['Income']:,.0f}")
    m2.metric("Expenses (₹)", f"{summary['Expenses']:,.0f}")
    m3.metric("Investments (₹)", f"{summary['Investments']:,.0f}")
    m4.metric("Net Savings (₹)", f"{summary['Net']:,.0f}")

    # Charts
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Transaction type distribution**")
        fig = plot_type_distribution(df.copy())
        st.pyplot(fig, clear_figure=True)

    with c2:
        st.markdown("**Monthly cashflow**")
        plot_df = df.copy()
        plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")
        plot_df["YYYY-MM"] = plot_df["Date"].dt.to_period("M").astype(str)
        inflow = plot_df[plot_df["Type"].isin(["Income", "Investment"])].groupby("YYYY-MM")["Amount (₹)"].sum()
        outflow = plot_df[plot_df["Type"].isin(["Expense", "Payment"])]["Amount (₹)"].abs().groupby(plot_df["YYYY-MM"]).sum()
        idx = sorted(set(inflow.index).union(outflow.index))
        y_in = [inflow.get(i, 0) for i in idx]
        y_out = [outflow.get(i, 0) for i in idx]
        x = np.arange(len(idx))
        width = 0.4
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.bar(x - width/2, y_in, width, label="Inflows")
        ax2.bar(x + width/2, y_out, width, label="Outflows")
        ax2.set_xticks(x)
        ax2.set_xticklabels(idx, rotation=45, ha="right")
        ax2.set_ylabel("₹")
        ax2.set_title("Inflows vs Outflows by Month")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)
        st.pyplot(fig2, clear_figure=True)

    # Latest records
    st.markdown("**Recent entries**")
    st.dataframe(
        df.sort_values("Date", ascending=False).head(10),
        use_container_width=True,
        hide_index=True,
    )

# -------------------------
# AI Assistant (Gemini)
# -------------------------
st.subheader("Arthena AI — Ask for tips")

API_KEY = (
    st.secrets.get("ARTHENA", None)
    if hasattr(st, "secrets")
    else None
) or os.environ.get("ARTHENA") or os.environ.get("GOOGLE_API_KEY")

model_name_fallback = "gemini-1.5-flash"
model_obj = None
ai_ready = False

if API_KEY:
    try:
        import google.generativeai as genai

        genai.configure(api_key=API_KEY)

        # Try preferred name first, then fallback
        try_model_names = [
            "models/gemini-1.5-flash",
            "gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "gemini-1.5-flash-latest",
        ]
        chosen = None
        for m in try_model_names:
            try:
                model_obj = genai.GenerativeModel(m)
                chosen = m
                break
            except Exception:
                continue
        if chosen is None:
            model_obj = genai.GenerativeModel(model_name_fallback)
            chosen = model_name_fallback

        st.caption(f"Using model: `{chosen}`")
        ai_ready = True
    except Exception as e:
        st.error(f"Gemini init failed: {e}")
else:
    st.info("Add your Gemini key in **Secrets** (`ARTHENA`) or set env `ARTHENA`/`GOOGLE_API_KEY` to enable the chatbot.")

q = st.text_area("Ask a question (e.g., “How can I reduce expenses this month?”)")
ask = st.button("Ask Arthena", type="primary")

if ask:
    if not ai_ready:
        st.warning("AI is not configured yet.")
    elif df.empty and q.strip() == "":
        st.warning("Add some data and type a question.")
    else:
        try:
            context = summarize_context(df)
            history_str = "\n".join([f"User: {u}\nArthena: {a}" for u, a in st.session_state.chat[-3:]])
            prompt = (
                "You are Arthena — a personal finance coach. "
                "Use the user's data context to give simple, practical, India-friendly advice. "
                "Be concise (2–5 sentences) and avoid generic tips.\n\n"
                f"Recent chat:\n{history_str}\n\n"
                f"Data context: {context}\n\n"
                f"User question: {q}"
            )
            resp = model_obj.generate_content(prompt)
            ans = getattr(resp, "text", str(resp))
            st.session_state.chat.append((q, ans))
        except Exception as e:
            ans = f"Error: {e}"
            st.session_state.chat.append((q, ans))

# Chat transcript
if st.session_state.chat:
    st.markdown("**Conversation**")
    for u, a in st.session_state.chat[-8:]:
        st.markdown(f"**You:** {u}")
        st.markdown(f"**Arthena:** {a}")

