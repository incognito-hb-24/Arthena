# Arthena — Your Personal CFO (AI-Powered Finance Assistant)

Arthena is an intelligent, privacy-first **Personal CFO prototype** —  
a finance assistant that helps you **track spending, manage savings, and get personalized AI insights** powered by Google Gemini.

Built with **Streamlit**, **pandas**, and **Gemini AI**, Arthena turns simple user entries into smart financial dashboards and recommendations.


## Features
✅ Add and manage **Wealth** and **Cashflow** entries  
✅ Automatic tracking of **Income, Expenses, and Investments**  
✅ **Interactive dashboards** — metrics, charts, and monthly insights  
✅ **AI Chatbot (Gemini)** for personalized financial tips  
✅ Works entirely **offline** — your data stays in your CSV  
✅ Deployable on **Streamlit Cloud**, **Colab**, or **local system**

---

## Project Structure  <br>
Arthena/         <br>
│ app.py   <br>
│ requirements.txt     <br>
│ README.md   <br>
│        <br>
├─cfo/      <br>
│ ├─ init.py     <br>
│ ├─ logic.py # core calculations & plots     <br>
│ └─ storage.py # CSV load/save utilities   <br>
│      <br>
├─data/            <br>
│ └─ sample_transactions.csv         <br>
│        <br>
└─.streamlit/        <br>
└─ config.toml # (optional) Streamlit config        <br>


## Requirements
- Python **3.10+**
- Recommended: a virtual environment

**Install everything using:**
pip install -r requirements.txt



##  Local Run Guide
1️⃣ Create and Activate Virtual Environment

**Windows (PowerShell / CMD):**

python -m venv .venv   <br>
.venv\Scripts\activate

**macOS / Linux:**

python3 -m venv .venv    <br>
source .venv/bin/activate


2️⃣ Install Dependencies   <br>
pip install -r requirements.txt


3️⃣ (Optional) Set Gemini API Key   <br>
Arthena uses Google’s Gemini API for its chatbot.
You can set the API key in any of these ways:

🔹 Option A — Environment Variable       <br>
set ARTHENA=YOUR_API_KEY ** Windows**   <br>
export ARTHENA=YOUR_API_KEY  **macOS / Linux**      <br>

🔹 Option B — .streamlit/secrets.toml
Create a new file named .streamlit/secrets.toml and add:

[secrets]

ARTHENA = "YOUR_API_KEY"

Generate your Gemini API key here:

https://makersuite.google.com/app/apikey

4️⃣ Run Arthena
streamlit run app.py


## Streamlit Cloud Deployment (Free Hosting)

Push your repo to GitHub

**Visit** https://share.streamlit.io
 → Deploy an app

Select your repo and set the file path to app.py

In App → Settings → Secrets, add:

ARTHENA = "YOUR_GEMINI_API_KEY"


Click Restart — your Arthena app is now live!


## Google Colab Testing (optional)
Run Arthena inside Colab if you don’t have a local setup:

!pip -q install streamlit==1.39.0 google-generativeai==0.8.3 pandas==2.2.3 numpy==2.1.3 matplotlib==3.9.2
!git clone https://github.com/<your-username>/Arthena.git
%cd Arthena
import os; os.environ["ARTHENA"] = "YOUR_API_KEY"


# Start the app
import threading, subprocess, time
def run():
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"])
threading.Thread(target=run, daemon=True).start()
time.sleep(5)
print("Arthena is running — open the public URL from the output above.")


## Core Concepts
| Module           | Description                                        | Example                           |
| ---------------- | -------------------------------------------------- | --------------------------------- |
| **Wealth**       | Long-term assets, liabilities, goals, or insurance | FD, Loan, Health Plan             |
| **Cashflow**     | Day-to-day income or expense                       | Salary, Rent, Groceries           |
| **AI Assistant** | Uses your transaction data to generate tips        | "How can I save more this month?" |


## AI Assistant (Gemini)

Arthena’s chatbot uses your entered data to provide relevant financial insights.

Example prompts:
“How can I reduce expenses next month?”
“What’s my savings ratio?”
“Is my investment allocation healthy?”

To enable AI features, make sure your Gemini API key is active and valid.


## Troubleshooting
| Issue                   | Cause                   | Fix                                                      |
| ----------------------- | ----------------------- | -------------------------------------------------------- |
| **AI not responding**   | No API key configured   | Add your key in `.streamlit/secrets.toml` or environment |
| **Charts not showing**  | Missing matplotlib      | Reinstall via `pip install matplotlib`                   |
| **CSV not saving**      | Permission issue        | Check write access to `data/user_history.csv`            |
| **App fails to launch** | Wrong Streamlit version | Use `pip install streamlit==1.39.0`                      |




