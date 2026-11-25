"""
Last Mile Logistics Auditor - Veridi Logistics
Streamlit dashboard built on the Olist Brazilian E-Commerce dataset.

Reads the small summary tables in ./dashboard_data/ (created by section 8 of the notebook):
dash_main, dash_states, dash_days, dash_categories, dash_groups, dash_handoff (.csv) and brazil_states.geojson.
Every table keeps sums and counts, so percentages are recalculated correctly after any filter.

Run:  streamlit run app.py
"""
import inspect
import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

st.set_page_config(page_title="Last Mile Logistics Auditor", page_icon="🚚", layout="wide")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DATA = Path(__file__).parent / "dashboard_data"

STATUS_ORDER = ["On Time", "Late", "Super Late"]
STATUS_COLORS = {"On Time": "#2a9d8f", "Late": "#f4a261", "Super Late": "#d62828"}
REGION_ORDER = ["North", "Northeast", "Central-West", "Southeast", "South"]
REGION_COLORS = {"North": "#6a4c93", "Northeast": "#d62828", "Central-West": "#f4a261",
                 "Southeast": "#2a9d8f", "South": "#1d6fa5"}
CARD_COLORS = ["#29B5E8", "#FF9F36", "#D45B90", "#7D44CF"]      # blue, orange, pink, purple (as in the reference layout)
TIMEFRAMES = ["Monthly", "Quarterly"]                           # the tables are monthly, so nothing finer exists
REMOTE_KM = 1500            # "remote" = more than 1,500 km from Sao Paulo
GRID = "rgba(128,128,128,0.18)"

# columns that can be added up (late_orders / super_late_orders are created when loading)
SUM_COLS = ["orders", "late_orders", "super_late_orders", "review_sum", "review_count",
            "one_star", "promised_sum", "actual_sum"]

_chart_ids = itertools.count()   # unique key for every chart (Streamlit rejects identical un-keyed charts)

# ----------------------------------------------------------------------------
# Look & feel
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1450px;}
    @media (prefers-color-scheme: dark) {section[data-testid="stSidebar"] {background-color: #05070a;}}
    div[data-testid="stMetricValue"] {font-size: 2.1rem; font-weight: 600;}
    div[data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.85;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
    h3 {padding-top: 0.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _stretch_kwargs(func):
    """Full-width argument for the installed Streamlit version (it moved from use_container_width to width='stretch')."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params and isinstance(params["width"].default, str):      # new API
        return {} if params["width"].default == "stretch" else {"width": "stretch"}
    if "use_container_width" in params:                                      # old API
        return {"use_container_width": True}
    return {}


PLOTLY_KW = _stretch_kwargs(st.plotly_chart)
TABLE_KW = _stretch_kwargs(st.dataframe)


def show(fig, **kwargs):
    """Draw a Plotly chart full-width."""
    st.plotly_chart(fig, key=f"plot_{next(_chart_ids)}", **PLOTLY_KW, **kwargs)


def show_table(df, **kwargs):
    st.dataframe(df, hide_index=True, **TABLE_KW, **kwargs)


def style(fig, height=360, **layout):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=28, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      **layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def pct(x):
    return "–" if pd.isna(x) else f"{x:.1f}%"


def num(x, digits=2):
    return "–" if pd.isna(x) else f"{x:.{digits}f}"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
REQUIRED_FILES = ["dash_main.csv", "dash_states.csv", "dash_days.csv", "dash_categories.csv",
                  "dash_groups.csv", "dash_handoff.csv", "brazil_states.geojson"]
missing_files = [f for f in REQUIRED_FILES if not (DATA / f).exists()]
if missing_files:
    st.error("Dashboard data not found in `dashboard_data/`: " + ", ".join(missing_files) + ". "
             "Keep that folder next to `app.py` in the repository.")
    st.stop()


@st.cache_data(show_spinner="Loading data...")
def load_data():
    main = pd.read_csv(DATA / "dash_main.csv")
    states = pd.read_csv(DATA / "dash_states.csv")
    days = pd.read_csv(DATA / "dash_days.csv")
    cats = pd.read_csv(DATA / "dash_categories.csv")
    groups = pd.read_csv(DATA / "dash_groups.csv")
    handoff = pd.read_csv(DATA / "dash_handoff.csv")

    # Late = Late + Super Late
    main["late_orders"] = main["orders"].where(main["delivery_status"] != "On Time", 0)
    main["super_late_orders"] = main["orders"].where(main["delivery_status"] == "Super Late", 0)

    handoff["seller_missed"] = handoff["seller_missed"].astype(bool)
    handoff["late_orders"] = handoff["orders"].where(handoff["delivery_status"] != "On Time", 0)
    handoff["missed_orders"] = handoff["orders"].where(handoff["seller_missed"], 0)
    handoff["late_missed_orders"] = handoff["late_orders"].where(handoff["seller_missed"], 0)
    return main, states, days, cats, groups, handoff


@st.cache_resource(show_spinner=False)
def load_geojson():
    return json.loads((DATA / "brazil_states.geojson").read_text(encoding="utf-8"))


main_all, states, days_all, cats_all, groups_all, handoff_all = load_data()
geojson = load_geojson()

DATA_FIRST = pd.Timestamp(main_all["month"].min() + "-01")
DATA_LAST_END = pd.Timestamp(main_all["month"].max() + "-01") + pd.offsets.MonthEnd(0)


# ----------------------------------------------------------------------------
# Sidebar - settings (same layout as the reference: dates + time frame)
# ----------------------------------------------------------------------------
min_d = DATA_FIRST.date()
max_d = DATA_LAST_END.date()

with st.sidebar:
    st.markdown("## 🚚 Last Mile Logistics Auditor")
    st.caption("Veridi Logistics · delivery performance vs customer sentiment")
    st.markdown("### ⚙️ Settings")
    start_date = st.date_input("Start date", value=max(min_d, date(2017, 9, 1)), min_value=min_d, max_value=max_d)
    end_date = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d)
    freq = st.selectbox("Select time frame", TIMEFRAMES, index=0,
                        help="The summary tables are monthly, so Daily and Weekly are not available.")
    regions = st.multiselect("Region", REGION_ORDER, default=REGION_ORDER)
    st.caption("Orders are grouped by purchase month, so whole months are included. "
               "The delay-distribution chart only uses the region filter.")
    st.markdown("---")
    st.caption("Data: Olist Brazilian E-Commerce (Kaggle). Only delivered orders with a delivery date are analysed; "
               "canceled / unavailable / in-transit orders are excluded.")

if start_date is None or end_date is None or start_date > end_date:
    st.warning("Please choose a start date that is on or before the end date.")
    st.stop()
if not regions:
    st.warning("Please select at least one region.")
    st.stop()

start_month = start_date.strftime("%Y-%m")
end_month = end_date.strftime("%Y-%m")


def by_region(df):
    return df[df["region"].isin(regions)]


def in_window(df):
    return df[(df["month"] >= start_month) & (df["month"] <= end_month)]


base_main = by_region(main_all)                 # all months, region filter only  -> "All-Time"
sel_main = in_window(base_main)                 # selected months                 -> "Selected Duration"
base_h = by_region(handoff_all)
sel_h = in_window(base_h)
cats = in_window(by_region(cats_all))
groups = in_window(by_region(groups_all))
days = by_region(days_all)                      # no month column in this table

if sel_main["orders"].sum() == 0:
    st.warning("No delivered orders match the current filters. Try widening the date range or the region selection.")
    st.stop()


# ----------------------------------------------------------------------------
# Calculations
# ----------------------------------------------------------------------------
def add_ratios(d):
    d = d.copy()
    d["late_pct"] = d["late_orders"] / d["orders"] * 100
    d["super_late_pct"] = d["super_late_orders"] / d["orders"] * 100
    d["avg_review"] = d["review_sum"] / d["review_count"]
    d["one_star_pct"] = d["one_star"] / d["review_count"] * 100
    d["avg_promised"] = d["promised_sum"] / d["orders"]
    d["avg_actual"] = d["actual_sum"] / d["orders"]
    d["cushion"] = d["avg_promised"] - d["avg_actual"]
    return d


def summarize(df, by):
    """Add up the sums and counts for each group, then calculate the percentages."""
    return add_ratios(df.groupby(by)[SUM_COLS].sum().reset_index())


def add_period(df, timeframe):
    ts = pd.to_datetime(df["month"].astype(str) + "-01")
    if timeframe == "Quarterly":
        ts = ts.dt.to_period("Q").dt.start_time
    return df.assign(period=ts)


def trim_edges(t, frac=0.15):
    """Drop sparse first/last periods (e.g. the few orders of late 2016)."""
    if t.empty:
        return t
    ok = (t["orders"] >= t["orders"].median() * frac).to_numpy()
    if not ok.any():
        return t
    first = int(ok.argmax())
    last = len(ok) - 1 - int(ok[::-1].argmax())
    return t.iloc[first:last + 1]


def period_table(main_df, handoff_df, timeframe):
    """One row per month / quarter with every KPI, used for the trend charts and the card sparklines."""
    if main_df.empty:
        return pd.DataFrame()
    t = add_ratios(add_period(main_df, timeframe).groupby("period")[SUM_COLS].sum())
    if handoff_df.empty:
        t["seller_missed_pct"] = np.nan
    else:
        hh = add_period(handoff_df, timeframe).groupby("period")[["orders", "missed_orders"]].sum()
        t["seller_missed_pct"] = hh["missed_orders"] / hh["orders"] * 100
    if timeframe == "Quarterly":            # hide quarters the data does not fully cover (e.g. Jul-Aug 2018 only)
        quarter_end = pd.DatetimeIndex(t.index) + pd.offsets.MonthEnd(3)
        t = t[(t.index >= DATA_FIRST) & (quarter_end <= DATA_LAST_END)]
    return trim_edges(t)


def overall(main_df, handoff_df):
    total = main_df[SUM_COLS].sum().to_frame().T
    r = add_ratios(total).iloc[0]
    out = {k: r[k] for k in ["orders", "late_pct", "super_late_pct", "avg_review", "avg_promised", "avg_actual", "cushion"]}
    h_orders = handoff_df["orders"].sum()
    out["seller_missed_pct"] = handoff_df["missed_orders"].sum() / h_orders * 100 if h_orders > 0 else np.nan
    return out


def make_delta(series, kind):
    """Change between the latest period and the one before it."""
    if series is None:
        return None
    s = series.dropna()
    if len(s) < 2:
        return None
    last, prev = s.iloc[-1], s.iloc[-2]
    diff = last - prev
    if kind != "count":
        diff = float(np.round(diff, 2 if kind == "score" else 1)) + 0.0     # avoids showing "-0.0"
    if kind == "count":
        return f"{diff:+,.0f} ({diff / prev * 100:+.1f}%)" if prev else None
    if kind == "pp":
        return f"{diff:+.1f} pp"
    if kind == "score":
        return f"{diff:+.2f}"
    return f"{diff:+.1f} d"


def get_status(days_difference):
    if days_difference >= 0:
        return "On Time"
    if days_difference >= -5:
        return "Late"
    return "Super Late"


# ----------------------------------------------------------------------------
# KPI cards (same layout as the reference: bordered card = metric + small trend chart)
# ----------------------------------------------------------------------------
CARD_SPECS = {
    "orders": dict(label="Delivered Orders", fmt="{:,.0f}", kind="count", chart="bar", zero=True, dcolor="normal",
                   help="Orders with a delivery date. Canceled / unavailable / in-transit orders are excluded."),
    "late_pct": dict(label="Late Delivery Rate", fmt="{:.1f}%", kind="pp", chart="area", zero=True, dcolor="inverse",
                     help="% of delivered orders that arrived after the estimated date (Late + Super Late)."),
    "super_late_pct": dict(label="Super Late (> 5 days)", fmt="{:.1f}%", kind="pp", chart="area", zero=True, dcolor="inverse",
                           help="% of delivered orders that arrived more than 5 days after the estimated date."),
    "avg_review": dict(label="Avg Review Score", fmt="{:.2f} ★", kind="score", chart="area", zero=False, dcolor="normal",
                       help="Average review score (1-5) of orders that have a review."),
    "avg_promised": dict(label="Avg Promised Delivery Time", fmt="{:.1f} days", kind="days", chart="area", zero=False,
                         dcolor="off", help="Estimated delivery date minus purchase date."),
    "avg_actual": dict(label="Avg Actual Delivery Time", fmt="{:.1f} days", kind="days", chart="area", zero=False,
                       dcolor="inverse", help="Actual delivery date minus purchase date."),
    "cushion": dict(label="Estimate Cushion (promised − actual)", fmt="{:+.1f} days", kind="days", chart="area",
                    zero=False, dcolor="normal",
                    help="Average promised days minus average actual days. Positive = estimates are padded on average; "
                         "a small cushion leaves little room for delays."),
    "seller_missed_pct": dict(label="Seller Missed Hand-off Deadline", fmt="{:.1f}%", kind="pp", chart="area", zero=True,
                              dcolor="inverse",
                              help="% of orders where the seller handed the parcel to the carrier after the shipping "
                                   "deadline (orders with valid hand-off dates only)."),
}


def spark(s, color, chart, zero_based):
    fig = go.Figure()
    if chart == "bar":
        fig.add_trace(go.Bar(x=s.index, y=s.values, marker_color=color))
    else:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=color, width=1.5),
                                 fill="tozeroy", fillcolor=color))
        lo, hi = float(s.min()), float(s.max())
        if zero_based:
            fig.update_yaxes(range=[0, hi * 1.15 if hi > 0 else 1])
        else:
            pad = (hi - lo) or abs(hi) or 1.0
            fig.update_yaxes(range=[lo - pad * 0.6, hi + pad * 0.15])
    fig.update_traces(hovertemplate="%{x|%b %Y}<br>%{y:,.1f}<extra></extra>")
    fig.update_layout(height=150, margin=dict(l=0, r=0, t=4, b=0), showlegend=False, bargap=0.15,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False, nticks=4, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=False, nticks=4, tickfont=dict(size=10))
    return fig


def kpi_row(main_df, handoff_df, table, keys):
    values = overall(main_df, handoff_df)
    for i, (col, key) in enumerate(zip(st.columns(len(keys)), keys)):
        spec = CARD_SPECS[key]
        series = table[key] if (not table.empty and key in table) else None
        value = values[key]
        text = "–" if pd.isna(value) else spec["fmt"].format(value)
        with col:
            with st.container(border=True):
                st.metric(spec["label"], text, delta=make_delta(series, spec["kind"]),
                          delta_color=spec["dcolor"], help=spec["help"])
                if series is not None and series.dropna().shape[0] >= 2:
                    show(spark(series.dropna(), CARD_COLORS[i % 4], spec["chart"], spec["zero"]),
                         config={"displayModeBar": False})
                else:
                    st.caption("Not enough periods to draw a trend.")


# ----------------------------------------------------------------------------
# Headline numbers (always all data, all regions)
# ----------------------------------------------------------------------------
by_month_all = summarize(main_all, ["month"])
worst_months = sorted(by_month_all[by_month_all["orders"] >= 300].nlargest(3, "late_pct")["month"].tolist())
period_flag = main_all["month"].isin(worst_months).map({True: "Worst 3 months", False: "Other months"})
by_period = summarize(main_all.assign(period=period_flag), ["period"]).set_index("period")

ne = main_all[main_all["region"] == "Northeast"][SUM_COLS].sum()
rest = main_all[main_all["region"] != "Northeast"][SUM_COLS].sum()
ne_late = ne["late_orders"] / ne["orders"] * 100
rest_late = rest["late_orders"] / rest["orders"] * 100

early_share = days_all[days_all["days_late"] < 0]["orders"].sum() / days_all["orders"].sum() * 100
status_all = summarize(main_all, ["delivery_status"]).set_index("delivery_status")
state_all = summarize(main_all, ["customer_state"])
best_state = state_all.loc[state_all["late_pct"].idxmin()]
worst_state = state_all.loc[state_all["late_pct"].idxmax()]

miss_all = handoff_all.groupby("seller_missed")[["orders", "late_orders"]].sum()
late_if_missed = miss_all.loc[True, "late_orders"] / miss_all.loc[True, "orders"] * 100
late_if_met = miss_all.loc[False, "late_orders"] / miss_all.loc[False, "orders"] * 100
seller_share_all = handoff_all["late_missed_orders"].sum() / handoff_all["late_orders"].sum() * 100

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
t_all = period_table(base_main, base_h, freq)
t_sel = period_table(sel_main, sel_h, freq)

st.title("Last Mile Logistics Auditor")
st.caption(f"Olist marketplace · purchases from {min_d:%b %Y} to {max_d:%b %Y} · "
           f"{int(main_all['orders'].sum()):,} delivered orders · time frame: {freq.lower()}")
st.info(
    f"**Answer for the CEO: both.** The Northeast is late **{ne_late:.1f}%** of the time vs **{rest_late:.1f}%** in the "
    f"rest of the country (a *regional* problem), and in three months ({', '.join(worst_months)}) the late rate was "
    f"**{by_period.loc['Worst 3 months', 'late_pct']:.1f}%** vs **{by_period.loc['Other months', 'late_pct']:.1f}%** in all "
    f"other months, in every region (a *time-related* problem). On average {early_share:.0f}% of orders arrive before the "
    f"estimate, but when we miss it the average review falls from **{status_all.loc['On Time', 'avg_review']:.2f}** to "
    f"**{status_all.loc['Late', 'avg_review']:.2f}** (late) and **{status_all.loc['Super Late', 'avg_review']:.2f}** (super late)."
)
with st.expander("Full summary (all data, whatever the filters say)"):
    st.markdown(f"""
1. **A regional problem in the Northeast.** Late **{ne_late:.1f}%** vs **{rest_late:.1f}%** elsewhere.
   State late rates go from {best_state['late_pct']:.1f}% ({best_state['customer_state']}) to {worst_state['late_pct']:.1f}% ({worst_state['customer_state']}).
2. **A problem in a few months, in every region.** The 3 worst months were chosen *because* they were the worst, so their share
   of late orders is partly built in; the gap in the rate is the better evidence. The data does not say what caused it.
3. **The estimates are only half the story.** {early_share:.0f}% of orders arrive *before* the estimated date, so on average the
   estimates are padded, not too optimistic. But the padding differs by state, so some states run out of cushion (Delay tab).
4. **Late deliveries hurt reviews a lot:** {status_all.loc['On Time', 'avg_review']:.2f} on time,
   {status_all.loc['Late', 'avg_review']:.2f} late, {status_all.loc['Super Late', 'avg_review']:.2f} super late.
5. **Sellers are a smaller but real part.** Orders are late {late_if_missed:.1f}% of the time when the seller missed the hand-off
   deadline vs {late_if_met:.1f}% when it did not, but only {seller_share_all:.0f}% of late orders involve a late seller.
   The dataset has no carrier field, so it cannot say which carrier is slow.
""")

# ----------------------------------------------------------------------------
# KPI rows
# ----------------------------------------------------------------------------
st.subheader("All-Time Statistics")
st.caption(f"All months (respecting the region filter). Arrows show the change in the latest "
           f"{'month' if freq == 'Monthly' else 'quarter'} vs the one before.")
kpi_row(base_main, base_h, t_all, ["orders", "late_pct", "super_late_pct", "avg_review"])

st.subheader("Selected Duration")
st.caption(f"{start_date:%b %Y} → {end_date:%b %Y}" + ("  ·  incomplete quarters are hidden from the trend charts" if freq == "Quarterly" else ""))
kpi_row(sel_main, sel_h, t_sel, ["orders", "late_pct", "super_late_pct", "avg_review"])

st.subheader("Promise vs Reality")
st.caption("Are the estimated delivery dates realistic? Selected duration.")
kpi_row(sel_main, sel_h, t_sel, ["avg_promised", "avg_actual", "cushion", "seller_missed_pct"])

# ----------------------------------------------------------------------------
# Trends: nationwide problem or regional problem?
# ----------------------------------------------------------------------------
st.subheader("Delivery performance over time")
st.caption("Nationwide problem or regional problem? If every line jumps in the same periods, it is nationwide.")
left, right = st.columns(2)

with left:
    st.markdown("**Late and Super Late rate**")
    if len(t_sel) >= 2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=t_sel.index, y=t_sel["orders"], name="Orders (left axis)",
                             marker_color="rgba(150,150,150,0.30)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=t_sel.index, y=t_sel["late_pct"], name="Late % (all late orders)", mode="lines+markers",
                                 line=dict(color=STATUS_COLORS["Late"], width=2.5)), secondary_y=True)
        fig.add_trace(go.Scatter(x=t_sel.index, y=t_sel["super_late_pct"], name="Super Late %", mode="lines+markers",
                                 line=dict(color=STATUS_COLORS["Super Late"], width=2.5)), secondary_y=True)
        fig.update_yaxes(title_text="orders", secondary_y=False, showgrid=False)
        fig.update_yaxes(title_text="% of orders", secondary_y=True, rangemode="tozero")
        show(style(fig, 380))
    else:
        st.info("Select a longer date range (or the monthly time frame) to see the trend.")

with right:
    st.markdown("**Late rate by region**")
    reg = add_period(sel_main, freq).groupby(["period", "region"])[["orders", "late_orders"]].sum().reset_index()
    reg["late_pct"] = reg["late_orders"] / reg["orders"] * 100
    reg = reg[reg["orders"] >= 15]                      # tiny samples are too noisy to draw
    if len(t_sel) >= 2:
        reg = reg[reg["period"].isin(t_sel.index)]
    if reg["period"].nunique() >= 2:
        fig = go.Figure()
        for region_name in REGION_ORDER:
            sub = reg[reg["region"] == region_name]
            if not sub.empty:
                fig.add_trace(go.Scatter(x=sub["period"], y=sub["late_pct"], name=region_name, mode="lines+markers",
                                         line=dict(color=REGION_COLORS[region_name], width=2)))
        if len(t_sel) >= 2:
            fig.add_trace(go.Scatter(x=t_sel.index, y=t_sel["late_pct"], name="All selected", mode="lines",
                                     line=dict(color="rgba(200,200,200,0.9)", width=2, dash="dash")))
        fig.update_yaxes(title_text="% of orders late", rangemode="tozero")
        show(style(fig, 380))
    else:
        st.info("Not enough data points per region for this selection.")

# ----------------------------------------------------------------------------
# Deep-dive tabs
# ----------------------------------------------------------------------------
tab_delay, tab_geo, tab_rev, tab_cat, tab_seller, tab_about = st.tabs(
    ["⏱️ Delay", "🗺️ Geography", "⭐ Reviews", "📦 Categories", "🏭 Seller vs carrier", "ℹ️ About"])

# ---------------------------------------------------------------- Delay (Story 2)
with tab_delay:
    st.markdown("### How late are orders compared to the promised date?")
    st.caption("User Story 2 · `Days_Difference = estimated delivery date − actual delivery date` (dates only). "
               "Positive = early, negative = late. On Time: 0 or more · Late: 1-5 days late · Super Late: more than 5 days late.")

    c_left, c_right = st.columns([2, 1])
    with c_left:
        st.markdown("**How early or late orders arrive**")
        hist = days.groupby("days_late")["orders"].sum().reset_index()
        hist["days_difference"] = -hist["days_late"]
        hist["status"] = hist["days_difference"].apply(get_status)
        fig = go.Figure()
        for status_name in STATUS_ORDER:
            sub = hist[hist["status"] == status_name]
            fig.add_trace(go.Bar(x=sub["days_difference"], y=sub["orders"], name=status_name,
                                 marker_color=STATUS_COLORS[status_name]))
        fig.update_layout(barmode="relative")
        fig.update_xaxes(title_text="Days_Difference (negative = late)")
        fig.update_yaxes(title_text="orders")
        show(style(fig, 380))
        st.caption("The first and last bars include everything further out. This chart only uses the region filter.")
    with c_right:
        st.markdown("**Status of delivered orders**")
        by_status = summarize(sel_main, ["delivery_status"]).set_index("delivery_status").reindex(STATUS_ORDER).reset_index()
        by_status["share_%"] = by_status["orders"] / by_status["orders"].sum() * 100
        show_table(by_status[["delivery_status", "orders", "share_%"]].rename(
            columns={"delivery_status": "Status", "orders": "Orders", "share_%": "Share %"}).round(1))
        st.caption("Orders that were canceled, unavailable or not yet delivered have no delivery date and are not included.")

    st.markdown("### Do we pad the estimate the same way everywhere?")
    st.caption("Average promised days vs average actual days (purchase to delivery), by state. A small gap means a small cushion.")
    state_days = summarize(sel_main, ["customer_state", "region"]).sort_values("avg_promised")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=state_days["customer_state"], y=state_days["avg_promised"], name="Promised", marker_color="#1d6fa5"))
    fig.add_trace(go.Bar(x=state_days["customer_state"], y=state_days["avg_actual"], name="Actual", marker_color="#d62828"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="average days")
    show(style(fig, 380))
    with st.expander("State table (sorted by cushion)"):
        tbl = state_days.sort_values("cushion")[["customer_state", "region", "orders", "avg_promised", "avg_actual", "cushion", "late_pct"]]
        show_table(tbl.rename(columns={"customer_state": "State", "region": "Region", "orders": "Orders",
                                       "avg_promised": "Avg promised (d)", "avg_actual": "Avg actual (d)",
                                       "cushion": "Cushion (d)", "late_pct": "Late %"}).round(1))

# ---------------------------------------------------------------- Geography (Story 3)
with tab_geo:
    st.markdown("### Which states have the highest share of late deliveries?")
    st.caption("User Story 3 · selected duration. Distances are measured from Sao Paulo, where most sellers are.")

    state_view = summarize(sel_main, ["customer_state", "region"]).merge(
        states[["customer_state", "lat", "lng", "distance_km"]], on="customer_state", how="left")
    national_late = overall(sel_main, sel_h)["late_pct"]

    # lighter version of YlOrRd (the darkest red is cut off so the black state codes stay readable)
    MAP_SCALE = [[i / 4, sample_colorscale("YlOrRd", [0.8 * i / 4])[0]] for i in range(5)]

    def make_map(view):
        """Each state is a filled shape drawn from the local geojson, so the map needs no internet and no map tiles."""
        values = view.set_index("customer_state")
        low, high = view["late_pct"].min(), view["late_pct"].max()
        fig = go.Figure()
        for feature in geojson["features"]:
            code = feature["properties"]["sigla"]
            geometry = feature["geometry"]
            polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
            xs, ys = [], []
            for polygon in polygons:
                ring = polygon[0]                      # outer border of the shape
                xs += [point[0] for point in ring] + [None]
                ys += [point[1] for point in ring] + [None]
            if code in values.index:
                row = values.loc[code]
                share = 0.5 if high == low else (row["late_pct"] - low) / (high - low)
                color = sample_colorscale(MAP_SCALE, [share])[0]
                text = f"<b>{code}</b><br>late: {row['late_pct']:.1f}%<br>orders: {int(row['orders']):,}"
            else:
                color = "#3a3f4b"
                text = f"{code}<br>not in the selection"
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=color,
                                     line=dict(color="#1A1D24", width=1), text=text, hoverinfo="text",
                                     hoveron="fills", showlegend=False))
        fig.add_trace(go.Scatter(x=view["lng"], y=view["lat"], mode="text", text=view["customer_state"],
                                 textfont=dict(size=11, color="black"), hoverinfo="skip", showlegend=False))
        # invisible point, only there to show the colour bar
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(colorscale=MAP_SCALE, cmin=low, cmax=high, color=[low], showscale=True,
                                             colorbar=dict(title="% late", thickness=15, len=0.7))))
        fig.update_xaxes(visible=False, range=[-74, -33])
        fig.update_yaxes(visible=False, range=[-34.5, 5.5], scaleanchor="x", scaleratio=1)
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        return fig

    map_col, bar_col = st.columns([3, 2])
    with map_col:
        st.markdown("**Late delivery rate by state** (hover for details)")
        show(make_map(state_view))

    with bar_col:
        st.markdown("**Ranking (dashed line = average of the selection)**")
        ranked = state_view.sort_values("late_pct")
        fig = go.Figure()
        for region_name in REGION_ORDER:
            sub = ranked[ranked["region"] == region_name]
            if not sub.empty:
                fig.add_trace(go.Bar(
                    x=sub["late_pct"], y=sub["customer_state"], orientation="h", name=region_name,
                    marker_color=REGION_COLORS[region_name],
                    text=[f"{p:.1f}%  (n={n:,})" for p, n in zip(sub["late_pct"], sub["orders"])],
                    textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}: %{x:.1f}% late<extra></extra>"))
        fig.add_vline(x=national_late, line_dash="dash", line_color="rgba(200,200,200,0.9)")
        fig.update_layout(barmode="overlay")
        fig.update_yaxes(categoryorder="array", categoryarray=ranked["customer_state"].tolist())
        fig.update_xaxes(title_text="% late (Late + Super Late)", range=[0, max(ranked["late_pct"].max() * 1.45, 1)])
        show(style(fig, 600))
    st.caption("Small states have few orders (see n), so their rates can swing a lot.")

    st.markdown("### Are remote states hit harder?")
    st.caption(f"Remote = more than {REMOTE_KM:,} km from Sao Paulo.")
    dist = state_view.dropna(subset=["distance_km"])
    remote = dist.groupby(dist["distance_km"] > REMOTE_KM)[["orders", "late_orders"]].sum()
    remote["pct"] = remote["late_orders"] / remote["orders"] * 100
    worst_sel = state_view[state_view["orders"] >= 30].sort_values("late_pct", ascending=False).head(1)
    corr = dist["distance_km"].rank().corr(dist["late_pct"].rank()) if len(dist) >= 5 else np.nan   # Spearman, no scipy needed

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Worst state (≥ 30 orders)", "–" if worst_sel.empty else f"{worst_sel['customer_state'].iloc[0]} · {worst_sel['late_pct'].iloc[0]:.1f}%")
    m2.metric("Late rate · remote states", pct(remote.loc[True, "pct"]) if True in remote.index else "–")
    m3.metric("Late rate · other states", pct(remote.loc[False, "pct"]) if False in remote.index else "–")
    m4.metric("Distance vs late % (rank corr.)", "–" if pd.isna(corr) else f"{corr:+.2f}",
              help="Spearman rank correlation across states. Values near 0 mean distance alone does not explain lateness.")

    sc_col, tbl_col = st.columns([3, 2])
    with sc_col:
        fig = go.Figure()
        for region_name in REGION_ORDER:
            sub = dist[dist["region"] == region_name]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["distance_km"], y=sub["late_pct"], mode="markers+text", name=region_name,
                    text=sub["customer_state"], textposition="top center",
                    marker=dict(size=np.clip(np.sqrt(sub["orders"]) * 0.7, 8, 46), color=REGION_COLORS[region_name], opacity=0.8),
                    customdata=sub["orders"].to_numpy(),
                    hovertemplate="%{text}: %{y:.1f}% late<br>%{x:,.0f} km · n=%{customdata:,}<extra></extra>"))
        fig.add_hline(y=national_late, line_dash="dot", line_color="rgba(200,200,200,0.7)")
        fig.update_xaxes(title_text="distance to Sao Paulo (km)")
        fig.update_yaxes(title_text="% late", rangemode="tozero")
        show(style(fig, 420))
    with tbl_col:
        st.markdown("**Regions**")
        region_tbl = summarize(sel_main, ["region"]).sort_values("late_pct", ascending=False)
        show_table(region_tbl[["region", "orders", "late_pct", "super_late_pct", "avg_review"]].rename(
            columns={"region": "Region", "orders": "Orders", "late_pct": "Late %", "super_late_pct": "Super late %",
                     "avg_review": "Avg review"}),
            column_config={"Orders": st.column_config.NumberColumn(format="%d"),
                           "Late %": st.column_config.NumberColumn(format="%.1f"),
                           "Super late %": st.column_config.NumberColumn(format="%.1f"),
                           "Avg review": st.column_config.NumberColumn(format="%.2f")})
    st.info("**Audit note (all data):** distance alone does not explain it. AM, AP, AC and RO are further from Sao Paulo "
            "than the Northeast states, yet have the lowest late rates because their promised dates carry a cushion of "
            "about 20 days. The Northeast is the real hot spot, and Rio de Janeiro, close to the hub, has more late "
            "orders than the whole Northeast, mostly in the three spike months.")

    with st.expander("State scorecard (table)"):
        card = state_view.sort_values("late_pct", ascending=False)[
            ["customer_state", "region", "orders", "late_orders", "late_pct", "super_late_pct", "avg_review",
             "avg_promised", "avg_actual", "cushion", "distance_km"]]
        show_table(card, column_config={
            "late_pct": st.column_config.NumberColumn("late %", format="%.1f"),
            "super_late_pct": st.column_config.NumberColumn("super late %", format="%.1f"),
            "avg_review": st.column_config.NumberColumn("avg review", format="%.2f"),
            "avg_promised": st.column_config.NumberColumn("avg promised (d)", format="%.1f"),
            "avg_actual": st.column_config.NumberColumn("avg actual (d)", format="%.1f"),
            "cushion": st.column_config.NumberColumn("cushion (d)", format="%.1f"),
            "distance_km": st.column_config.NumberColumn("km to Sao Paulo", format="%.0f")})
        st.download_button("Download state scorecard (CSV)", data=card.to_csv(index=False).encode("utf-8"),
                           file_name="state_scorecard.csv", mime="text/csv")

# ---------------------------------------------------------------- Reviews (Story 4)
with tab_rev:
    st.markdown("### Do late deliveries cause bad reviews?")
    st.caption("User Story 4 · selected duration (the day-by-day chart only uses the region filter).")

    rev = summarize(sel_main, ["delivery_status"]).set_index("delivery_status").reindex(STATUS_ORDER)
    on_rows = sel_main[sel_main["delivery_status"] == "On Time"]
    late_rows = sel_main[sel_main["delivery_status"] != "On Time"]
    on_avg = on_rows["review_sum"].sum() / on_rows["review_count"].sum() if on_rows["review_count"].sum() else np.nan
    late_avg = late_rows["review_sum"].sum() / late_rows["review_count"].sum() if late_rows["review_count"].sum() else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg score · On Time", "–" if pd.isna(on_avg) else f"{on_avg:.2f} ★")
    c2.metric("Avg score · Late (Late + Super Late)", "–" if pd.isna(late_avg) else f"{late_avg:.2f} ★",
              delta=None if pd.isna(late_avg) or pd.isna(on_avg) else f"{late_avg - on_avg:+.2f} vs on time")
    c3.metric("1-star reviews · On Time", pct(rev.loc["On Time", "one_star_pct"]))
    c4.metric("1-star reviews · Super Late", pct(rev.loc["Super Late", "one_star_pct"]))

    l_col, r_col = st.columns([2, 3])
    with l_col:
        st.markdown("**Average review score by delivery status**")
        fig = go.Figure(go.Bar(
            x=STATUS_ORDER, y=rev["avg_review"], marker_color=[STATUS_COLORS[s] for s in STATUS_ORDER],
            text=[num(v) for v in rev["avg_review"]], textposition="outside",
            customdata=rev["review_count"].fillna(0).to_numpy(),
            hovertemplate="%{x}: %{y:.2f} ★ (n=%{customdata:,.0f})<extra></extra>"))
        fig.update_yaxes(range=[0, 5.5], title_text="average review score")
        show(style(fig, 400))
    with r_col:
        st.markdown("**Review score for each day of delay**")
        day_rev = days.groupby("days_late")[["review_sum", "review_count", "orders"]].sum().reset_index()
        day_rev = day_rev[day_rev["days_late"].between(-20, 25) & (day_rev["review_count"] >= 30)]
        day_rev["avg_review"] = day_rev["review_sum"] / day_rev["review_count"]
        if len(day_rev) >= 3:
            dot_colors = [STATUS_COLORS["On Time"] if v <= 0 else STATUS_COLORS["Late"] if v <= 5 else STATUS_COLORS["Super Late"]
                          for v in day_rev["days_late"]]
            fig = go.Figure(go.Scatter(
                x=day_rev["days_late"], y=day_rev["avg_review"], mode="lines+markers",
                line=dict(color="rgba(180,180,180,0.8)", width=2), marker=dict(size=8, color=dot_colors),
                customdata=day_rev["review_count"].to_numpy(),
                hovertemplate="%{x} days late: %{y:.2f} ★ (n=%{customdata:,})<extra></extra>"))
            fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(200,200,200,0.8)",
                          annotation_text="estimated date", annotation_position="top left")
            fig.add_vline(x=5.5, line_dash="dash", line_color=STATUS_COLORS["Super Late"],
                          annotation_text="Super Late (> 5 days)", annotation_position="top right")
            fig.update_xaxes(title_text="days late (negative = delivered early)")
            fig.update_yaxes(title_text="average review score", range=[0.8, 5])
            show(style(fig, 400))
            st.caption("Days with fewer than 30 reviews are hidden.")
        else:
            st.info("Not enough reviews per day of delay in this selection.")

    # does the review gap show up inside each state?
    sr = sel_main.assign(is_late=sel_main["delivery_status"] != "On Time")
    rs = sr.pivot_table(index="customer_state", columns="is_late", values="review_sum", aggfunc="sum")
    rc = sr.pivot_table(index="customer_state", columns="is_late", values="review_count", aggfunc="sum")
    consistency = ""
    if True in rc.columns and False in rc.columns:
        enough = (rc[True] >= 30) & (rc[False] >= 30)
        if enough.any():
            gap = (rs[False] / rc[False] - rs[True] / rc[True])[enough]
            consistency = (f" Inside each state the same pattern holds: in **{int((gap > 0).sum())} of {len(gap)}** states "
                           "with 30+ late and 30+ on-time reviews, late orders score lower.")
    st.info("**Audit note:** the score stays about the same however early the parcel arrives, then drops right after the "
            "promised date, so being on time matters more than being fast." + consistency +
            " This is observational data, so it shows a strong link rather than a proven cause.")

# ---------------------------------------------------------------- Categories (Bonus)
with tab_cat:
    st.markdown("### Which product categories are hardest to deliver on time?")
    st.caption("Bonus story · category names translated to English (two missing ones translated by hand). "
               "Selected duration and region filter. An order with products from different categories counts once in each.")

    s_col1, s_col2 = st.columns(2)
    min_orders = s_col1.slider("Minimum orders per category", 50, 2000, 300, step=50, key="cat_min",
                               help="Hides small categories whose rates would be noisy.")
    top_n = s_col2.slider("Categories to show on each side", 5, 15, 10, key="cat_top")

    cat_table = cats.groupby("category")[["orders", "late_orders", "review_sum", "review_count"]].sum().reset_index()
    cat_table["late_pct"] = cat_table["late_orders"] / cat_table["orders"] * 100
    cat_table["avg_review"] = cat_table["review_sum"] / cat_table["review_count"]
    cat_table = cat_table[cat_table["orders"] >= min_orders].sort_values("late_pct")

    if len(cat_table) < 2:
        st.warning("Not enough categories with that many orders. Lower the minimum.")
    else:
        shown = pd.concat([cat_table.head(top_n), cat_table.tail(top_n)]) if len(cat_table) > 2 * top_n else cat_table
        n_easy = min(top_n, len(shown))
        colors = ["#2a9d8f"] * n_easy + ["#d62828"] * (len(shown) - n_easy)
        sel_late_pct = overall(sel_main, sel_h)["late_pct"]
        l_col, r_col = st.columns([3, 2])
        with l_col:
            st.markdown("**Easiest and hardest categories by late rate**")
            fig = go.Figure(go.Bar(
                x=shown["late_pct"], y=[c.replace("_", " ") for c in shown["category"]], orientation="h",
                marker_color=colors, text=[f"{p:.1f}%" for p in shown["late_pct"]], textposition="outside",
                cliponaxis=False, customdata=shown[["orders", "avg_review"]].to_numpy(),
                hovertemplate="%{y}: %{x:.1f}% late<br>%{customdata[0]:,} orders · %{customdata[1]:.2f} ★<extra></extra>"))
            fig.add_vline(x=sel_late_pct, line_dash="dash", line_color="rgba(200,200,200,0.9)")
            fig.update_xaxes(title_text="% late", range=[0, shown["late_pct"].max() * 1.2])
            show(style(fig, max(420, 26 * len(shown))))
        with r_col:
            st.markdown("**Furniture vs electronics**")
            grp = groups.groupby("group")[["orders", "late_orders", "review_sum", "review_count"]].sum()
            for name in ["Furniture", "Electronics"]:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    if name in grp.index and grp.loc[name, "orders"] > 0:
                        row = grp.loc[name]
                        a, b, c = st.columns(3)
                        a.metric("Orders", f"{int(row['orders']):,}")
                        b.metric("Late %", pct(row["late_orders"] / row["orders"] * 100))
                        c.metric("Avg review", num(row["review_sum"] / row["review_count"]))
                    else:
                        st.caption("No orders in this selection.")
            st.caption("Furniture = every category with “furniture” in the name. Electronics = electronics, computers, "
                       "computers_accessories, telephony, fixed_telephony and audio together. An order counts once per group.")
            st.info("**Audit note (all data):** furniture is not harder to deliver than electronics (about 7.2% vs 6.9% late). "
                    "Category differences are small next to the differences between states and between months.")
        with st.expander("Table with all categories"):
            show_table(cat_table.sort_values("late_pct", ascending=False)[["category", "orders", "late_pct", "avg_review"]]
                       .rename(columns={"category": "Category", "orders": "Orders", "late_pct": "Late %", "avg_review": "Avg review"})
                       .round(2))

# ---------------------------------------------------------------- Candidate's choice: seller vs carrier
with tab_seller:
    st.markdown("### Candidate's choice: is a late order the seller's fault, or does the delay happen after hand-off?")
    st.markdown(
        "**Why it matters:** the fix is different. If sellers hand parcels over late, Veridi can enforce deadlines and "
        "penalties. If the delay happens *after* hand-off, the problem sits in shipping or in a promise that is too short, "
        "and no seller scorecard will fix it. Each order has a seller deadline (`shipping_limit_date`); handing over "
        "after it counts as a seller miss. **The dataset has no carrier field, so individual carriers cannot be compared.**"
    )
    h = sel_h
    if h["orders"].sum() == 0:
        st.info("No orders with valid hand-off dates in the current selection.")
    else:
        by_missed = h.groupby("seller_missed")[["orders", "late_orders"]].sum().reindex([False, True])
        by_missed["late_pct"] = by_missed["late_orders"] / by_missed["orders"] * 100
        total_late = h["late_orders"].sum()
        share_missed_late = h["late_missed_orders"].sum() / total_late * 100 if total_late > 0 else np.nan

        st.caption(f"Based on {int(h['orders'].sum()):,} of {int(sel_main['orders'].sum()):,} delivered orders in the selection "
                   "(orders without a carrier hand-off date, or with dates in the wrong order, are not included).")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Seller missed the deadline", pct(h["missed_orders"].sum() / h["orders"].sum() * 100),
                  help="% of orders handed over after the seller's deadline.")
        c2.metric("Late % · seller on time", pct(by_missed.loc[False, "late_pct"]))
        c3.metric("Late % · seller missed deadline", pct(by_missed.loc[True, "late_pct"]))
        c4.metric("Late orders despite an on-time seller", "–" if pd.isna(share_missed_late) else f"{100 - share_missed_late:.0f}%",
                  help="Share of late orders where the seller handed the parcel over on time, so the delay happened after hand-off.")

        l_col, m_col, r_col = st.columns(3)
        with l_col:
            st.markdown("**Where the time goes (average days)**")
            times = h.groupby("delivery_status")[["orders", "seller_days_sum", "after_handoff_days_sum"]].sum().reindex(STATUS_ORDER)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=STATUS_ORDER, y=times["seller_days_sum"] / times["orders"],
                                 name="Before hand-off (seller)", marker_color="#8d99ae"))
            fig.add_trace(go.Bar(x=STATUS_ORDER, y=times["after_handoff_days_sum"] / times["orders"],
                                 name="After hand-off", marker_color="#1d6fa5"))
            fig.update_layout(barmode="stack")
            fig.update_yaxes(title_text="days")
            show(style(fig, 400))
        with m_col:
            st.markdown("**Late rate vs seller deadline**")
            fig = go.Figure(go.Bar(
                x=["Seller on time", "Seller missed deadline"], y=by_missed["late_pct"].to_numpy(),
                marker_color=["#2a9d8f", "#d62828"], text=[pct(v) for v in by_missed["late_pct"]], textposition="outside"))
            top = by_missed["late_pct"].max()
            fig.update_yaxes(title_text="% of orders late", range=[0, max(top * 1.25, 1) if pd.notna(top) else 1])
            show(style(fig, 400))
        with r_col:
            st.markdown("**Late orders where the seller was also late**")
            late_h = h[h["delivery_status"] != "On Time"]
            by_reg = late_h.groupby("region")[["orders", "late_missed_orders", "after_handoff_days_sum"]].sum().reset_index()
            by_reg["seller_missed_share"] = by_reg["late_missed_orders"] / by_reg["orders"] * 100
            by_reg["after_handoff"] = by_reg["after_handoff_days_sum"] / by_reg["orders"]
            by_reg = by_reg.sort_values("seller_missed_share")
            fig = go.Figure(go.Bar(
                x=by_reg["seller_missed_share"], y=by_reg["region"], orientation="h",
                marker_color=[REGION_COLORS[r] for r in by_reg["region"]],
                text=[f"{v:.0f}%" for v in by_reg["seller_missed_share"]], textposition="outside", cliponaxis=False,
                customdata=by_reg[["orders", "after_handoff"]].to_numpy(),
                hovertemplate="%{y}: %{x:.0f}% of %{customdata[0]:,} late orders had a late seller<br>"
                              "avg %{customdata[1]:.1f} days after hand-off<extra></extra>"))
            fig.update_xaxes(title_text="% of late orders", range=[0, max(by_reg["seller_missed_share"].max() * 1.3, 1)])
            show(style(fig, 400))

        p_miss, p_met = by_missed.loc[True, "late_pct"], by_missed.loc[False, "late_pct"]
        miss_text = (f"when the seller misses the deadline an order is about {p_miss / p_met:.0f}× more likely to be late "
                     f"({p_miss:.1f}% vs {p_met:.1f}%)" if pd.notna(p_miss) and pd.notna(p_met) and p_met > 0
                     else "the late rate is higher when the seller misses the deadline")
        st.info(f"**Audit note:** {miss_text}, but most late orders follow an on-time hand-off. "
                "In the notebook's what-if, if every seller met its deadline about 12% of late orders (760) "
                "would disappear. In the worst Northeast states only 14-22% of late orders involve a late seller, so the delay "
                "mostly happens after hand-off or in a promise that is too short.")
        st.markdown(
            """
| Problem found | What Veridi can do |
|---|---|
| Delay after hand-off in the Northeast (no carrier field in the data) | Get carrier and tracking data for these states, then decide if carriers or hubs need to change |
| Estimated date not adjusted per state | Set the promised date from how long deliveries really take in that state, not a flat cushion |
| Three bad months (Nov 2017, Feb 2018, Mar 2018) | Investigate what happened; the cause is unknown from the data |
| Sellers missing deadlines | Seller scorecards and penalties (smaller but real effect) |
"""
        )

# ---------------------------------------------------------------- About
with tab_about:
    st.markdown("### About this dashboard")
    st.markdown("""
**Data:** Olist Brazilian E-Commerce dataset (Kaggle), about 99,000 orders from 2016 to 2018.
The dashboard reads small summary tables in `dashboard_data/`, created by the notebook `last_mile_logistics_audit.ipynb`.
Each table keeps sums and counts, so percentages are recalculated correctly after any filter.

**How the numbers are built**
- One review is kept per order (the latest), so joining reviews does not duplicate orders.
- Only orders with status `delivered` and a delivery date are used. Canceled, unavailable and in-progress orders are left out.
- `Days_Difference` compares dates only (the estimate has no time of day).
- Late = Late + Super Late. Super Late = more than 5 days after the estimated date.
- Product categories are translated with the translation file, plus two categories it misses (translated by hand).
- Trend charts group orders by *purchase* month (or quarter). Sparse first/last periods are trimmed, and quarters the data does not fully cover are hidden.

**What the filters do**
- *Start / End date* and *Region* change every number and chart, except the delay-distribution and day-by-day review charts (region only, their tables have no month) and the “Full summary” (always all data).
- The category and furniture/electronics tables have no state column, so they respond to dates and region only.

**Limits**
- Observational data, so the review results show a strong link, not a proven cause.
- There is no carrier field, so nothing here says which carrier is slow.
- The 3 worst months are chosen because they were the worst, and the cause is unknown.
- Small states and categories have few orders, so their percentages are noisy.
""")
