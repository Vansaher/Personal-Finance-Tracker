# 💸 Personal Finance Tracker

A **data science–driven personal finance tracker** built in Python.  
This project extracts bank statements (PDF → CSV), cleans & categorizes transactions, performs exploratory data analysis (EDA), forecasts future spending with ARIMA, and visualizes everything through a **Streamlit dashboard**.

---
## 🔗 Link

https://personal-finance-tracker-vansaher.streamlit.app/


## 🚀 Features

- **Transaction Extraction**
  - Parse PDF bank statements into structured CSV/Excel.
  - Standardize descriptions & remove noise (`SALE DEBIT |`, `PRE-AUTH DEBIT |`, etc.).
  - Categorize expenses (Shopping, Food & Beverage, Transport, Bills, Rent/Housing, Online Purchase, etc.).

- **Data Cleaning**
  - Unified description & merchant normalization.
  - Expense vs Income classification.
  - Outlier flagging (big expenses ≥ 100).
  - Daily and monthly aggregates for analysis.

- **Exploratory Analysis**
  - Spending by category & vendor.
  - Biggest expenses (excl. Rent/Housing).
  - Monthly trends (Income, Expenses, Net).
  - Weekday average spending patterns.

- **Forecasting**
  - ARIMA/SES forecast of **next month’s expenses**.
  - Optional exclusion of fixed categories (like Rent/Housing).
  - Forecast visualized with confidence intervals.

- **Dashboard**
  - Built with **Streamlit** + **Plotly**.
  - KPI cards (Income, Expenses, Net, Savings%).
  - Category pie chart of discretionary spending.
  - Monthly bar + Net line chart.
  - Filterable transactions table.

---

## 🛠️ Tech Stack

- **Python** 3.10+
- **Pandas** – data manipulation  
- **Statsmodels** – ARIMA forecasting  
- **Plotly** – interactive charts  
- **Streamlit** – dashboard frontend  

---

## 📂 Project Structure
```
.
├── extraction/                 # Scripts to parse bank PDFs
│   └── extract_bank_statements.py
├── cleaning/                   # Scripts to clean and standardize transactions
│   └── clean_finance_excel.py
├── dashboard/                  # Streamlit app
│   └── app.py
├── data/                       # Example input/output (ignored in .gitignore)
│   ├── Statement_June2025.pdf
│   ├── transactions_clean.csv
│   └── daily_aggregate.csv
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```



---

## ⚡ Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/personal-finance-tracker.git
cd personal-finance-tracker
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Dashboard Preview
- Biggest Expenses (excluding rent)
- Spending Breakdown by Category
- Monthly Income vs Expense with Net overlay
- Forecast: Next Month Expense (ARIMA/SES)
- Weekday Average Spend (excluding rent)
- Detailed Transactions Table

