import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(
    page_title="Personal Finance Tracker",
    page_icon="💸",
    layout="wide",
)

st.title("💸 Personal Finance Tracker")

@st.cache_data
def load_csv(file_or_path):
    if isinstance(file_or_path, str):
        df = pd.read_csv(file_or_path, parse_dates=["date"])
    else:
        # UploadedFile
        df = pd.read_csv(file_or_path, parse_dates=["date"])
    return df

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    # Expected minimum columns: date, amount, type, category
    # Optional: clean_description / raw_description / big_expense / balance
    need = {"date", "amount"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Type (Income/Expense) if not present
    if "type" not in df.columns:
        df["type"] = np.where(df["amount"] > 0, "Income",
                              np.where(df["amount"] < 0, "Expense", "Neutral"))

    # Category if not present
    if "category" not in df.columns:
        df["category"] = np.where(df["type"] == "Income", "Income",
                           np.where(df["type"] == "Expense", "Uncategorized", "Neutral"))

    # Description column for tables
    if "clean_description" in df.columns:
        df["desc"] = df["clean_description"]
    elif "raw_description" in df.columns:
        df["desc"] = df["raw_description"]
    else:
        df["desc"] = ""

    # Weekday
    df["weekday"] = df["date"].dt.day_name()

    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

def kpi_card(label, value, help_text=None):
    col = st.column_config.NumberColumn(format="%.2f")
    st.metric(label, f"{value:,.2f}")
    if help_text:
        st.caption(help_text)

# ---------------------------
# SIDEBAR (file + filters)
# ---------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload cleaned CSV", type=["csv"])
default_hint = st.sidebar.text_input("...or type a CSV file path", value="")

df = None
try:
    if uploaded is not None:
        df = load_csv(uploaded)
    elif default_hint.strip():
        df = load_csv(default_hint.strip())
    else:
        st.info("Upload your cleaned CSV")
except Exception as e:
    st.error(f"Failed to load file: {e}")

if df is None:
    st.stop()

# Schema + baseline fields
try:
    df = coerce_schema(df)
except Exception as e:
    st.error(str(e))
    st.stop()

# Date range filter
min_d, max_d = df["date"].min().date(), df["date"].max().date()
dr = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(dr, (list, tuple)) and len(dr) == 2:
    start_d, end_d = dr
else:
    start_d, end_d = min_d, max_d

mask = (df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)
df = df.loc[mask].copy()

# ---------------------------
# KPIs
# ---------------------------
income_total = df.loc[df["type"] == "Income", "amount"].sum()
expense_total = df.loc[df["type"] == "Expense", "amount"].sum()  # negative
net_total = income_total + expense_total
savings_pct = (income_total and (net_total / abs(income_total) * 100)) if income_total != 0 else 0.0

k1, k2, k3, k4 = st.columns(4)
with k1: kpi_card("Income", income_total)
with k2: kpi_card("Expenses", abs(expense_total))
with k3: kpi_card("Net", net_total)
with k4: st.metric("% Saved of Income", f"{savings_pct:,.1f}%")

st.divider()

# ---------------------------
# Biggest expenses
# ---------------------------
st.subheader("💥 Biggest Expenses")

# Expenses only, exclude Rent/Housing
expense_df = df[(df["type"] == "Expense") & (df["category"] != "Rent/Housing")].copy()

# Absolute value for ranking
expense_df["amount_abs"] = expense_df["amount"].abs()

# Slider for top N
top_n = st.slider("Show top N expenses", min_value=5, max_value=30, value=10, step=1)

# Get top N
top_exp = expense_df.nlargest(top_n, "amount_abs")[["date","desc","category","amount_abs"]].copy()

# Format date (no time)
top_exp["date"] = pd.to_datetime(top_exp["date"]).dt.date

# Show table
st.dataframe(
    top_exp.rename(columns={
        "date": "Date",
        "desc": "Description",
        "category": "Category",
        "amount_abs": "Amount"
    })[["Date","Description","Category","Amount"]],
    use_container_width=True
)

st.divider()

# ---------------------------
# Spending Breakdown - Category pie (expenses only)
# ---------------------------
st.subheader("🥧 Spending Breakdown")

cat_spend = (
    df[(df["type"] == "Expense") & (df["category"] != "Rent/Housing")]
    .groupby("category")["amount"]
    .sum()
    .abs()
    .sort_values(ascending=False)
)

if not cat_spend.empty:
    fig_pie = px.pie(
        cat_spend.reset_index(),
        names="category",
        values="amount",
        title="Share of Spending by Category (excluding Rent)"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("No expenses (excluding Rent) in the selected period.")

st.divider()

# ---------------------------
# Monthly spending over time (Expenses only)
# ---------------------------
st.subheader("📈 Monthly Spending Over Time")
monthly = (
    df.assign(month=lambda x: x["date"].dt.to_period("M").astype(str))
      .groupby(["month","type"])["amount"].sum().unstack(fill_value=0).reset_index()
)
# Build expense series as positive
monthly["expense_pos"] = monthly.get("Expense", 0).abs()

fig_spend = px.line(
    monthly, x="month", y="expense_pos",
    markers=True, title="Monthly Expenses"
)
fig_spend.update_layout(yaxis_title="Amount", xaxis_title="Month")
st.plotly_chart(fig_spend, use_container_width=True)

# ---------------------------
# Forecast next month expense (robust: ARIMA with drift → SES → avg)
# ---------------------------
st.subheader("🔮 Forecast: Next Month Expense")

# OPTIONAL: exclude Rent/Housing from forecast (uncomment if desired)
# df_fc_src = df[(df["type"] == "Expense") & (df["category"] != "Rent/Housing")]
df_fc_src = df[df["type"] == "Expense"]

monthly_exp = (
    df_fc_src.set_index("date")
             .resample("M")["amount"]
             .sum()
             .abs()
             .astype(float)
)

forecast_note = st.empty()

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np
import plotly.graph_objects as go
import pandas as pd

def forecast_next(series: pd.Series, steps: int = 1):
    """Try ARIMA with/without drift, else SES, else 3-month avg."""
    series = series.asfreq("M").fillna(0)

    # (A) Try ARIMA grid with trend ('n' = no drift, 'c' = drift)
    best = {"aic": np.inf, "order": None, "trend": None, "res": None}
    if len(series) >= 4 and series.nunique() > 1:
        for p in range(0, 3):
            for d in [0, 1]:
                for q in range(0, 3):
                    for trend in ["n", "c"]:
                        try:
                            res = ARIMA(series, order=(p, d, q), trend=trend,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False).fit()
                            if res.aic < best["aic"]:
                                best = {"aic": res.aic, "order": (p, d, q), "trend": trend, "res": res}
                        except Exception:
                            continue
    if best["res"] is not None:
        fc = best["res"].get_forecast(steps=steps)
        yhat = fc.predicted_mean.values.astype(float)
        ci = fc.conf_int().values.astype(float)  # shape: (steps, 2)
        return yhat, ci, ("ARIMA", best["order"], best["trend"])

    # (B) Try Simple Exponential Smoothing
    try:
        ses = SimpleExpSmoothing(series, initialization_method="estimated").fit()
        yhat = ses.forecast(steps).values.astype(float)
        return yhat, None, ("SES", None, None)
    except Exception:
        pass

    # (C) Fallback: mean of last 3 months (repeat for horizon)
    last3 = series.tail(3)
    avg = float(last3.mean()) if len(last3) else 0.0
    return np.array([avg] * steps, None, ("AVG3", None, None))

if len(monthly_exp) >= 2:
    horizon = 1  # change to 3 if you want 3 months ahead
    yhat, ci, model_info = forecast_next(monthly_exp, steps=horizon)

    # KPI cards
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Next Month Expense (forecast)", f"{max(0.0, yhat[-1]):,.2f}")
    if ci is not None:
        lo, hi = max(0.0, ci[-1, 0]), max(0.0, ci[-1, 1])
        with c2: st.metric("Lower 95% CI", f"{lo:,.2f}")
        with c3: st.metric("Upper 95% CI", f"{hi:,.2f}")

    # Build figure: historical line + dashed forecast line
    hist = monthly_exp.copy()
    hist.index.name = "month_end"

    # Future index (next horizon months)
    last_period = hist.index.max()
    future_idx = pd.date_range(last_period + pd.offsets.MonthEnd(1), periods=horizon, freq="M")

    fig_fc = go.Figure()

    # Historical
    fig_fc.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode="lines+markers", name="Historical", line=dict(color="blue")
    ))

    # Forecast line (only future points)
    fig_fc.add_trace(go.Scatter(
        x=future_idx,
        y=yhat,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="red", dash="dash"),
        marker=dict(size=10, symbol="circle")
    ))

    # Confidence band (if available)
    if ci is not None:
        fig_fc.add_trace(go.Scatter(
            x=np.r_[future_idx, future_idx[::-1]],
            y=np.r_[ci[:, 1], ci[::-1, 0]],
            fill="toself", fillcolor="rgba(255,0,0,0.15)",
            line=dict(color="rgba(255,0,0,0)"),
            hoverinfo="skip",
            name="95% CI"
        ))

    fig_fc.update_layout(
        title="Monthly Expense with Forecast",
        xaxis_title="Month",
        yaxis_title="Expense"
    )

    st.plotly_chart(fig_fc, use_container_width=True)

    # Model info
    mname, order, trend = model_info
    if mname == "ARIMA":
        forecast_note.info(f"Model: ARIMA{order} with trend='{trend}' (fallbacks: SES → Avg of last 3)")
    else:
        forecast_note.info(f"Model: {mname} (fallbacks applied)")
else:
    st.warning("Not enough monthly data to forecast. Add more months of transactions.")

st.divider()

# ---------------------------
# Trends (Monthly net + Weekday averages)
# ---------------------------
st.subheader("📊 Trends")

# Monthly aggregation
monthly["Income"] = monthly.get("Income", 0.0)
monthly["Expense"] = monthly.get("Expense", 0.0).abs()  # make positive
monthly["Net"] = monthly["Income"] - monthly["Expense"]

# Bar chart for Income & Expense, line for Net
fig_net = px.bar(
    monthly, x="month", y=["Income","Expense"],
    title="Monthly Income vs Expense with Net Overlay",
    barmode="group"
)

# Add Net line
fig_net.add_scatter(
    x=monthly["month"],
    y=monthly["Net"],
    mode="lines+markers",
    name="Net",
    line=dict(color="black", width=2)
)

fig_net.update_layout(
    yaxis_title="Amount",
    xaxis_title="Month"
)

st.plotly_chart(fig_net, use_container_width=True)

# Weekday average spend per transaction (expenses only, exclude Rent)
wk = df[(df["type"] == "Expense") & (df["category"] != "Rent/Housing")].copy()

if not wk.empty:
    wk["amount_pos"] = wk["amount"].abs()
    wk["weekday"] = wk["date"].dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wk_avg = wk.groupby("weekday")["amount_pos"].mean().reindex(order)

    fig_wk = px.bar(
        wk_avg.reset_index(),
        x="weekday", y="amount_pos",
        title="Average Spend per Transaction by Weekday (Excl. Rent)"
    )
    fig_wk.update_layout(yaxis_title="Average Spend", xaxis_title="Weekday")
    st.plotly_chart(fig_wk, use_container_width=True)
else:
    st.info("No non-rent expense transactions in the selected period.")

st.divider()

# ---------------------------
# Raw transactions
# ---------------------------
st.subheader("🔎 Transactions")
show_cols = [c for c in ["date","desc","category","type","amount","balance"] if c in df.columns]
st.dataframe(df[show_cols].sort_values("date", ascending=False), use_container_width=True)

st.caption("Tip: Adjust the date range in the left sidebar to slice the dashboard.")
