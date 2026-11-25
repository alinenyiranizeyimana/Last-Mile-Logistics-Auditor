"""
Last Mile Logistics Auditor - Veridi Logistics
Chart-first Streamlit dashboard on the Olist Brazilian E-Commerce dataset.

Reads the summary tables in ./dashboard_data/ (created by section 8 of the notebook):
dash_main, dash_states, dash_days, dash_categories, dash_groups, dash_handoff (.csv) and brazil_states.geojson.
Every table keeps sums and counts, so percentages are recalculated correctly after any filter.

Run:  streamlit run app.py
"""
import inspect
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

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
BLUE, RED, GREEN, ORANGE, GREY = "#1d6fa5", "#d62828", "#2a9d8f", "#f4a261", "#8d99ae"
REMOTE_KM = 1500
MIN_MONTH_ORDERS = 300
GRID = "rgba(128,128,128,0.18)"
HEAT = "YlOrRd"

SUM_COLS = ["orders", "late_orders", "super_late_orders", "review_sum", "review_count",
            "one_star", "promised_sum", "actual_sum"]

_chart_ids = itertools.count()

# ----------------------------------------------------------------------------
# Look & feel
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
    @media (prefers-color-scheme: dark) {section[data-testid="stSidebar"] {background-color: #05070a;}}
    div[data-testid="stMetricValue"] {font-size: 1.9rem; font-weight: 700;}
    div[data-testid="stMetricLabel"] p {font-size: 0.8rem; opacity: 0.8; text-transform: uppercase; letter-spacing: .03em;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
    h3 {padding-top: 0.4rem; padding-bottom: 0.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _stretch_kwargs(func):
    """Full-width argument for the installed Streamlit version."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}
    if "width" in params and isinstance(params["width"].default, str):
        return {} if params["width"].default == "stretch" else {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


PLOTLY_KW = _stretch_kwargs(st.plotly_chart)
TABLE_KW = _stretch_kwargs(st.dataframe)
NO_BAR = {"displayModeBar": False}


def show(fig, **kwargs):
    st.plotly_chart(fig, key=f"plot_{next(_chart_ids)}", config=NO_BAR, **PLOTLY_KW, **kwargs)


def show_table(df, **kwargs):
    st.dataframe(df, hide_index=True, **TABLE_KW, **kwargs)


def style(fig, height=340, **layout):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=24, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      **layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def pct(x):
    return "–" if pd.isna(x) else f"{x:.1f}%"


def num(x, digits=2):
    return "–" if pd.isna(x) else f"{x:.{digits}f}"


def mini_fig(fig, height=70):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def spark(x, y, color):
    """Tiny filled line for a KPI card."""
    fig = go.Figure(go.Scatter(x=list(x), y=list(y), mode="lines", line=dict(color=color, width=2),
                               fill="tozeroy", fillcolor=rgba(color, 0.16),
                               hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.update_yaxes(rangemode="tozero")
    return mini_fig(fig)


def mini_bars(x, y, colors):
    """Tiny bars for a KPI card."""
    fig = go.Figure(go.Bar(x=list(x), y=list(y), marker_color=colors,
                           hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    return mini_fig(fig)


def card(label, value, fig=None, delta=None, delta_color="normal", help=None):
    """KPI card: number on top, a small chart under it."""
    with st.container(border=True):
        st.metric(label, value, delta=delta, delta_color=delta_color, help=help)
        if fig is not None:
            show(fig)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
REQUIRED_FILES = ["dash_main.csv", "dash_states.csv", "dash_days.csv", "dash_categories.csv",
                  "dash_groups.csv", "dash_handoff.csv", "brazil_states.geojson"]
missing_files = [f for f in REQUIRED_FILES if not (DATA / f).exists()]
if missing_files:
    st.error("Dashboard data not found in `dashboard_data/`: " + ", ".join(missing_files) + ". "
             "Keep that folder next to `app.py`.")
    st.stop()


@st.cache_data(show_spinner="Loading data...")
def load_data():
    main = pd.read_csv(DATA / "dash_main.csv")
    states = pd.read_csv(DATA / "dash_states.csv")
    days = pd.read_csv(DATA / "dash_days.csv")
    cats = pd.read_csv(DATA / "dash_categories.csv")
    groups = pd.read_csv(DATA / "dash_groups.csv")
    handoff = pd.read_csv(DATA / "dash_handoff.csv")

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
# Sidebar
# ----------------------------------------------------------------------------
min_d = DATA_FIRST.date()
max_d = DATA_LAST_END.date()

with st.sidebar:
    st.markdown("## 🚚 Last Mile Auditor")
    st.caption("Veridi Logistics")
    st.markdown("### ⚙️ Filters")
    start_date = st.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d)
    end_date = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d)
    regions = st.multiselect("Region", REGION_ORDER, default=REGION_ORDER)
    st.caption("Whole purchase months. Delay histogram and day-by-day reviews use the region filter only.")
    st.markdown("---")
    st.caption("Olist e-commerce (Kaggle) · delivered orders only.")

if start_date is None or end_date is None or start_date > end_date:
    st.warning("Start date must be on or before the end date.")
    st.stop()
if not regions:
    st.warning("Select at least one region.")
    st.stop()

start_month = start_date.strftime("%Y-%m")
end_month = end_date.strftime("%Y-%m")
REG_SHOWN = [r for r in REGION_ORDER if r in regions]


def by_region(df):
    return df[df["region"].isin(regions)]


def in_window(df):
    return df[(df["month"] >= start_month) & (df["month"] <= end_month)]


base_main = by_region(main_all)
sel_main = in_window(base_main)
base_h = by_region(handoff_all)
sel_h = in_window(base_h)
cats = in_window(by_region(cats_all))
groups = in_window(by_region(groups_all))
days = by_region(days_all)

if sel_main["orders"].sum() == 0:
    st.warning("No delivered orders match the filters. Widen the dates or regions.")
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
    return add_ratios(df.groupby(by)[SUM_COLS].sum().reset_index())


def overall(main_df, handoff_df):
    total = main_df[SUM_COLS].sum().to_frame().T
    r = add_ratios(total).iloc[0]
    out = {k: r[k] for k in ["orders", "late_pct", "super_late_pct", "avg_review", "avg_promised", "avg_actual", "cushion"]}
    h_orders = handoff_df["orders"].sum()
    out["seller_missed_pct"] = handoff_df["missed_orders"].sum() / h_orders * 100 if h_orders > 0 else np.nan
    return out


def get_status(days_difference):
    if days_difference >= 0:
        return "On Time"
    if days_difference >= -5:
        return "Late"
    return "Super Late"


def wmedian(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return values[order][np.searchsorted(cum, cum[-1] / 2)]


def heatmap(z, x, y, title="% late", fmt=".0f", scale=HEAT, zmin=None, zmax=None):
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=scale, zmin=zmin, zmax=zmax,
                               texttemplate="%{z:" + fmt + "}", textfont=dict(size=10),
                               colorbar=dict(title=title, thickness=12),
                               hovertemplate="%{y} · %{x}: %{z:" + fmt + "}<extra></extra>"))
    fig.update_xaxes(type="category", tickangle=-90)
    return fig


# --- headline numbers (all data, all regions)
by_month_all = summarize(main_all, ["month"])
valid_months = by_month_all[by_month_all["orders"] >= MIN_MONTH_ORDERS]["month"]
worst_months = sorted(by_month_all[by_month_all["orders"] >= 300].nlargest(3, "late_pct")["month"].tolist())
period_flag = main_all["month"].isin(worst_months).map({True: "Worst 3 months", False: "Other months"})
by_period = summarize(main_all.assign(period=period_flag), ["period"]).set_index("period")

ne = main_all[main_all["region"] == "Northeast"][SUM_COLS].sum()
rest = main_all[main_all["region"] != "Northeast"][SUM_COLS].sum()
ne_late = ne["late_orders"] / ne["orders"] * 100
rest_late = rest["late_orders"] / rest["orders"] * 100

status_all = summarize(main_all, ["delivery_status"]).set_index("delivery_status").reindex(STATUS_ORDER)
region_all = summarize(main_all, ["region"]).set_index("region").reindex(REGION_ORDER)

# --- selection
monthly_sel = summarize(sel_main, ["month"])
monthly_sel = monthly_sel[monthly_sel["month"].isin(valid_months)].sort_values("month")
state_sel = summarize(sel_main, ["customer_state", "region"])
region_sel = summarize(sel_main, ["region"]).set_index("region").reindex(REG_SHOWN)
status_sel = summarize(sel_main, ["delivery_status"]).set_index("delivery_status").reindex(STATUS_ORDER)
vals = overall(sel_main, sel_h)

h_month = sel_h.groupby("month")[["orders", "missed_orders"]].sum().reset_index()
h_month = h_month[h_month["month"].isin(valid_months)].sort_values("month")
h_month["missed_pct"] = h_month["missed_orders"] / h_month["orders"] * 100

# ----------------------------------------------------------------------------
# Header + answer cards (all data)
# ----------------------------------------------------------------------------
st.title("🚚 Last Mile Logistics Auditor")
st.caption(f"Olist · {min_d:%b %Y} – {max_d:%b %Y} · {int(main_all['orders'].sum()):,} delivered orders")

st.markdown("#### CEO question: regional or nationwide?  **Both.**")
a1, a2, a3 = st.columns(3)
with a1:
    reg_late = region_all["late_pct"]
    card("Regional · Northeast late rate", f"{ne_late:.1f}%",
         mini_bars(REGION_ORDER, reg_late, [REGION_COLORS[r] for r in REGION_ORDER]),
         delta=f"{ne_late - rest_late:+.1f} pts vs rest ({rest_late:.1f}%)", delta_color="inverse")
with a2:
    m_all = by_month_all[by_month_all["orders"] >= MIN_MONTH_ORDERS].sort_values("month")
    card("Time · 3 worst months late rate", f"{by_period.loc['Worst 3 months', 'late_pct']:.1f}%",
         spark(m_all["month"], m_all["late_pct"], RED),
         delta=f"{by_period.loc['Worst 3 months', 'late_pct'] - by_period.loc['Other months', 'late_pct']:+.1f} pts vs other months",
         delta_color="inverse")
with a3:
    card("Impact · review when late", f"{status_all.loc['Late', 'avg_review']:.2f} ★",
         mini_bars(STATUS_ORDER, status_all["avg_review"], [STATUS_COLORS[s] for s in STATUS_ORDER]),
         delta=f"{status_all.loc['Late', 'avg_review'] - status_all.loc['On Time', 'avg_review']:+.2f} vs on time "
               f"({status_all.loc['On Time', 'avg_review']:.2f})")

# ----------------------------------------------------------------------------
# KPI cards (selection) with sparklines
# ----------------------------------------------------------------------------
st.markdown(f"#### Key numbers · {start_date:%b %Y} → {end_date:%b %Y}")
mx, ms = monthly_sel["month"], monthly_sel
has_trend = len(monthly_sel) >= 2


def trend(col, color):
    return spark(mx, ms[col], color) if has_trend else None


k = st.columns(4)
with k[0]:
    card("Delivered orders", f"{vals['orders']:,.0f}", trend("orders", BLUE))
with k[1]:
    card("Late rate", pct(vals["late_pct"]), trend("late_pct", ORANGE), help="Late + Super Late")
with k[2]:
    card("Super Late (> 5 days)", pct(vals["super_late_pct"]), trend("super_late_pct", RED))
with k[3]:
    card("Avg review", "–" if pd.isna(vals["avg_review"]) else f"{vals['avg_review']:.2f} ★", trend("avg_review", GREEN))
k = st.columns(4)
with k[0]:
    card("Avg promised time", f"{vals['avg_promised']:.1f} d", trend("avg_promised", BLUE))
with k[1]:
    card("Avg actual time", f"{vals['avg_actual']:.1f} d", trend("avg_actual", RED))
with k[2]:
    card("Estimate cushion", f"{vals['cushion']:+.1f} d", trend("cushion", GREEN), help="Promised − actual days")
with k[3]:
    card("Seller missed deadline", pct(vals["seller_missed_pct"]),
         spark(h_month["month"], h_month["missed_pct"], GREY) if len(h_month) >= 2 else None)

# ----------------------------------------------------------------------------
# Overview charts
# ----------------------------------------------------------------------------
o1, o2 = st.columns([1, 2])
with o1:
    with st.container(border=True):
        st.markdown("**Delivery status**")
        fig = go.Figure(go.Pie(labels=STATUS_ORDER, values=status_sel["orders"], hole=0.62, sort=False,
                               marker=dict(colors=[STATUS_COLORS[s] for s in STATUS_ORDER]),
                               textinfo="percent", hovertemplate="%{label}: %{value:,} orders<extra></extra>"))
        fig.add_annotation(text=f"<b>{vals['late_pct']:.1f}%</b><br>late", showarrow=False, font=dict(size=18))
        show(style(fig, 300))
with o2:
    with st.container(border=True):
        st.markdown("**Orders and late rate by month**")
        if has_trend:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=monthly_sel["month"], y=monthly_sel["orders"], name="Orders", marker_color=rgba(BLUE, 0.45)))
            fig.add_trace(go.Scatter(x=monthly_sel["month"], y=monthly_sel["late_pct"], name="Late %", yaxis="y2",
                                     mode="lines+markers", line=dict(color=RED, width=2.5)))
            style(fig, 300)
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title=dict(text="% late"), rangemode="tozero"))
            fig.update_xaxes(type="category", tickangle=-90)
            show(fig)
        else:
            st.info("Pick 2+ months.")

o3, o4, o5 = st.columns(3)
with o3:
    with st.container(border=True):
        st.markdown("**Late rate by region**")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=REG_SHOWN, y=region_sel["late_pct"] - region_sel["super_late_pct"], name="Late",
                             marker_color=STATUS_COLORS["Late"]))
        fig.add_trace(go.Bar(x=REG_SHOWN, y=region_sel["super_late_pct"], name="Super Late",
                             marker_color=STATUS_COLORS["Super Late"]))
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title_text="% of orders")
        show(style(fig, 300))
with o4:
    with st.container(border=True):
        st.markdown("**8 worst states**")
        worst8 = state_sel[state_sel["orders"] >= 30].nlargest(8, "late_pct").sort_values("late_pct")
        fig = go.Figure(go.Bar(x=worst8["late_pct"], y=worst8["customer_state"], orientation="h",
                               marker_color=[REGION_COLORS[r] for r in worst8["region"]],
                               text=[f"{v:.1f}%" for v in worst8["late_pct"]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{x:.1f}% late<extra></extra>"))
        fig.update_xaxes(range=[0, max(worst8["late_pct"].max() * 1.25, 1)] if not worst8.empty else None)
        show(style(fig, 300))
with o5:
    with st.container(border=True):
        st.markdown("**Review by status**")
        fig = go.Figure(go.Bar(x=STATUS_ORDER, y=status_sel["avg_review"], marker_color=[STATUS_COLORS[s] for s in STATUS_ORDER],
                               text=[num(v) for v in status_sel["avg_review"]], textposition="outside"))
        fig.update_yaxes(range=[0, 5.5])
        show(style(fig, 300))

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_delay, tab_geo, tab_time, tab_rev, tab_cat, tab_seller, tab_about = st.tabs(
    ["⏱️ Delay", "🗺️ Geography", "📅 Over time", "⭐ Reviews", "📦 Categories", "🏭 Seller vs carrier", "ℹ️ About"])

# ---------------------------------------------------------------- Delay
with tab_delay:
    hist = days.groupby("days_late")["orders"].sum().reset_index().sort_values("days_late")
    hist["days_difference"] = -hist["days_late"]
    hist["status"] = hist["days_difference"].apply(get_status)
    tot = hist["orders"].sum()
    early_pct = hist.loc[hist["days_difference"] > 0, "orders"].sum() / tot * 100
    on_day_pct = hist.loc[hist["days_difference"] == 0, "orders"].sum() / tot * 100
    mean_diff = (hist["days_difference"] * hist["orders"]).sum() / tot
    med_diff = wmedian(hist["days_difference"], hist["orders"])

    d = st.columns(4)
    d[0].container(border=True).metric("Arrive early", pct(early_pct))
    d[1].container(border=True).metric("Arrive on the day", pct(on_day_pct))
    d[2].container(border=True).metric("Avg days early (−late)", f"{mean_diff:+.1f}")
    d[3].container(border=True).metric("Median days early", f"{med_diff:+.0f}")

    c_left, c_right = st.columns([3, 2])
    with c_left:
        with st.container(border=True):
            st.markdown("**Early / late vs promised date** (negative = late)")
            fig = go.Figure()
            for s in STATUS_ORDER:
                sub = hist[hist["status"] == s]
                fig.add_trace(go.Bar(x=sub["days_difference"], y=sub["orders"], name=s, marker_color=STATUS_COLORS[s]))
            fig.update_layout(barmode="relative")
            fig.update_xaxes(title_text="days early (+) / late (−)")
            show(style(fig, 340))
    with c_right:
        with st.container(border=True):
            st.markdown("**Cumulative share of orders**")
            hs = hist.sort_values("days_difference")
            fig = go.Figure(go.Scatter(x=hs["days_difference"], y=hs["orders"].cumsum() / tot * 100, mode="lines",
                                       line=dict(color=BLUE, width=3), fill="tozeroy", fillcolor=rgba(BLUE, 0.15),
                                       hovertemplate="%{x} d: %{y:.1f}% of orders arrive by then<extra></extra>"))
            fig.add_vline(x=-0.5, line_dash="dash", line_color="rgba(200,200,200,0.9)")
            fig.update_xaxes(title_text="days early (+) / late (−)")
            fig.update_yaxes(title_text="% of orders", range=[0, 100])
            show(style(fig, 340))

    state_days = state_sel.sort_values("avg_promised")
    with st.container(border=True):
        st.markdown("**Promised vs actual days, by state**")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=state_days["customer_state"], y=state_days["avg_promised"], name="Promised", marker_color=BLUE))
        fig.add_trace(go.Bar(x=state_days["customer_state"], y=state_days["avg_actual"], name="Actual", marker_color=RED))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text="avg days")
        show(style(fig, 340))

    e1, e2 = st.columns(2)
    with e1:
        with st.container(border=True):
            st.markdown("**Cushion by state** (promised − actual, days)")
            cs = state_sel.sort_values("cushion")
            fig = go.Figure(go.Bar(x=cs["customer_state"], y=cs["cushion"],
                                   marker_color=[RED if v < 8 else ORANGE if v < 12 else GREEN for v in cs["cushion"]],
                                   hovertemplate="%{x}: %{y:.1f} days<extra></extra>"))
            fig.add_hline(y=vals["cushion"], line_dash="dash", line_color="rgba(200,200,200,0.9)")
            fig.update_yaxes(title_text="days")
            show(style(fig, 340))
    with e2:
        with st.container(border=True):
            st.markdown("**Cushion vs late rate** (bubble = orders)")
            fig = go.Figure()
            for r in REG_SHOWN:
                sub = state_sel[state_sel["region"] == r]
                fig.add_trace(go.Scatter(x=sub["cushion"], y=sub["late_pct"], mode="markers+text", name=r,
                                         text=sub["customer_state"], textposition="top center",
                                         marker=dict(size=np.clip(np.sqrt(sub["orders"]) * 0.7, 8, 44),
                                                     color=REGION_COLORS[r], opacity=0.8),
                                         hovertemplate="%{text}: cushion %{x:.1f} d · %{y:.1f}% late<extra></extra>"))
            fig.update_xaxes(title_text="cushion (days)")
            fig.update_yaxes(title_text="% late", rangemode="tozero")
            show(style(fig, 340))

    with st.expander("State table"):
        tbl = state_days.sort_values("cushion")[["customer_state", "region", "orders", "avg_promised", "avg_actual", "cushion", "late_pct"]]
        show_table(tbl.rename(columns={"customer_state": "State", "region": "Region", "orders": "Orders",
                                       "avg_promised": "Promised (d)", "avg_actual": "Actual (d)",
                                       "cushion": "Cushion (d)", "late_pct": "Late %"}).round(1))

# ---------------------------------------------------------------- Geography
with tab_geo:
    state_view = state_sel.merge(states[["customer_state", "lat", "lng", "distance_km"]], on="customer_state", how="left")
    national_late = vals["late_pct"]
    MAP_SCALE = [[i / 4, sample_colorscale("YlOrRd", [0.8 * i / 4])[0]] for i in range(5)]

    dist = state_view.dropna(subset=["distance_km"])
    remote = dist.groupby(dist["distance_km"] > REMOTE_KM)[["orders", "late_orders"]].sum()
    remote["pct"] = remote["late_orders"] / remote["orders"] * 100
    worst_sel = state_view[state_view["orders"] >= 30].sort_values("late_pct", ascending=False).head(1)
    corr = dist["distance_km"].rank().corr(dist["late_pct"].rank()) if len(dist) >= 5 else np.nan

    g = st.columns(4)
    g[0].container(border=True).metric("Worst state (≥30 orders)",
                                       "–" if worst_sel.empty else f"{worst_sel['customer_state'].iloc[0]} · {worst_sel['late_pct'].iloc[0]:.1f}%")
    g[1].container(border=True).metric(f"Late · remote (>{REMOTE_KM:,} km)", pct(remote.loc[True, "pct"]) if True in remote.index else "–")
    g[2].container(border=True).metric("Late · other states", pct(remote.loc[False, "pct"]) if False in remote.index else "–")
    g[3].container(border=True).metric("Distance vs late (rank corr.)", "–" if pd.isna(corr) else f"{corr:+.2f}",
                                       help="Near 0 = distance alone does not explain lateness.")

    def make_map(view):
        values = view.set_index("customer_state")
        low, high = view["late_pct"].min(), view["late_pct"].max()
        fig = go.Figure()
        for feature in geojson["features"]:
            code = feature["properties"]["sigla"]
            geometry = feature["geometry"]
            polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
            xs, ys = [], []
            for polygon in polygons:
                ring = polygon[0]
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
        with st.container(border=True):
            st.markdown("**Late rate by state**")
            show(make_map(state_view))
    with bar_col:
        with st.container(border=True):
            st.markdown("**Ranking** (dashed = average)")
            ranked = state_view.sort_values("late_pct")
            fig = go.Figure()
            for r in REGION_ORDER:
                sub = ranked[ranked["region"] == r]
                if not sub.empty:
                    fig.add_trace(go.Bar(
                        x=sub["late_pct"], y=sub["customer_state"], orientation="h", name=r, marker_color=REGION_COLORS[r],
                        text=[f"{p:.1f}%  (n={n:,})" for p, n in zip(sub["late_pct"], sub["orders"])],
                        textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x:.1f}% late<extra></extra>"))
            fig.add_vline(x=national_late, line_dash="dash", line_color="rgba(200,200,200,0.9)")
            fig.update_layout(barmode="overlay")
            fig.update_yaxes(categoryorder="array", categoryarray=ranked["customer_state"].tolist())
            fig.update_xaxes(title_text="% late", range=[0, max(ranked["late_pct"].max() * 1.5, 1)])
            show(style(fig, 560))

    s1, s2 = st.columns(2)
    with s1:
        with st.container(border=True):
            st.markdown("**Distance vs late rate** (bubble = orders)")
            fig = go.Figure()
            for r in REGION_ORDER:
                sub = dist[dist["region"] == r]
                if not sub.empty:
                    fig.add_trace(go.Scatter(
                        x=sub["distance_km"], y=sub["late_pct"], mode="markers+text", name=r,
                        text=sub["customer_state"], textposition="top center",
                        marker=dict(size=np.clip(np.sqrt(sub["orders"]) * 0.7, 8, 46), color=REGION_COLORS[r], opacity=0.8),
                        customdata=sub["orders"].to_numpy(),
                        hovertemplate="%{text}: %{y:.1f}% late<br>%{x:,.0f} km · n=%{customdata:,}<extra></extra>"))
            fig.add_hline(y=national_late, line_dash="dot", line_color="rgba(200,200,200,0.7)")
            fig.update_xaxes(title_text="km to Sao Paulo")
            fig.update_yaxes(title_text="% late", rangemode="tozero")
            show(style(fig, 400))
    with s2:
        with st.container(border=True):
            st.markdown("**Share of orders vs share of late orders, by region**")
            tot_o, tot_l = region_sel["orders"].sum(), region_sel["late_orders"].sum()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=REG_SHOWN, y=region_sel["orders"] / tot_o * 100, name="% of orders", marker_color=BLUE))
            fig.add_trace(go.Bar(x=REG_SHOWN, y=region_sel["late_orders"] / tot_l * 100 if tot_l else 0, name="% of late orders",
                                 marker_color=RED))
            fig.update_layout(barmode="group")
            fig.update_yaxes(title_text="%")
            show(style(fig, 400))

    t1, t2 = st.columns(2)
    with t1:
        with st.container(border=True):
            st.markdown("**Where late orders come from** (size = late orders, colour = late %)")
            tm = state_view[state_view["late_orders"] > 0]
            reg_tot = tm.groupby("region")[["orders", "late_orders"]].sum().reset_index()
            reg_tot["late_pct"] = reg_tot["late_orders"] / reg_tot["orders"] * 100
            fig = go.Figure(go.Treemap(
                labels=reg_tot["region"].tolist() + tm["customer_state"].tolist(),
                parents=[""] * len(reg_tot) + tm["region"].tolist(),
                values=reg_tot["late_orders"].tolist() + tm["late_orders"].tolist(),
                branchvalues="total",
                marker=dict(colors=reg_tot["late_pct"].tolist() + tm["late_pct"].tolist(), colorscale=HEAT,
                            colorbar=dict(title="% late", thickness=12)),
                textinfo="label+value", hovertemplate="%{label}: %{value:,} late orders<extra></extra>"))
            show(style(fig, 400))
    with t2:
        with st.container(border=True):
            st.markdown("**Late vs super late, by state**")
            ss = state_view.sort_values("late_pct", ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ss["customer_state"], y=ss["late_pct"] - ss["super_late_pct"], name="Late",
                                 marker_color=STATUS_COLORS["Late"]))
            fig.add_trace(go.Bar(x=ss["customer_state"], y=ss["super_late_pct"], name="Super Late",
                                 marker_color=STATUS_COLORS["Super Late"]))
            fig.update_layout(barmode="stack")
            fig.update_yaxes(title_text="% of orders")
            show(style(fig, 400))

    st.caption("AM, AP, AC, RO are far from Sao Paulo but rarely late (≈20-day cushion). Northeast is the hot spot. "
               "Small states = few orders.")

    with st.expander("State scorecard"):
        card_tbl = state_view.sort_values("late_pct", ascending=False)[
            ["customer_state", "region", "orders", "late_orders", "late_pct", "super_late_pct", "avg_review",
             "avg_promised", "avg_actual", "cushion", "distance_km"]]
        show_table(card_tbl, column_config={
            "late_pct": st.column_config.NumberColumn("late %", format="%.1f"),
            "super_late_pct": st.column_config.NumberColumn("super late %", format="%.1f"),
            "avg_review": st.column_config.NumberColumn("avg review", format="%.2f"),
            "avg_promised": st.column_config.NumberColumn("promised (d)", format="%.1f"),
            "avg_actual": st.column_config.NumberColumn("actual (d)", format="%.1f"),
            "cushion": st.column_config.NumberColumn("cushion (d)", format="%.1f"),
            "distance_km": st.column_config.NumberColumn("km to SP", format="%.0f")})
        st.download_button("Download CSV", data=card_tbl.to_csv(index=False).encode("utf-8"),
                           file_name="state_scorecard.csv", mime="text/csv")

# ---------------------------------------------------------------- Over time
with tab_time:
    spike_share_orders = base_main.loc[base_main["month"].isin(worst_months), "orders"].sum() / base_main["orders"].sum() * 100
    spike_share_late = base_main.loc[base_main["month"].isin(worst_months), "late_orders"].sum() / base_main["late_orders"].sum() * 100
    rj = main_all[main_all["customer_state"] == "RJ"]
    rj_late = rj["late_orders"].sum()
    rj_spike_share = rj.loc[rj["month"].isin(worst_months), "late_orders"].sum() / rj_late * 100 if rj_late else np.nan

    c = st.columns(4)
    c[0].container(border=True).metric("Late · normal months", pct(by_period.loc["Other months", "late_pct"]))
    c[1].container(border=True).metric("Late · 3 worst months", pct(by_period.loc["Worst 3 months", "late_pct"]),
                                       help=", ".join(worst_months))
    c[2].container(border=True).metric("Orders in worst 3", pct(spike_share_orders))
    c[3].container(border=True).metric("Late orders in worst 3", pct(spike_share_late))

    if has_trend:
        with st.container(border=True):
            st.markdown("**Late rate by purchase month**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly_sel["month"], y=monthly_sel["late_pct"], name="Late %", mode="lines+markers",
                                     line=dict(color=STATUS_COLORS["Late"], width=2.5)))
            fig.add_trace(go.Scatter(x=monthly_sel["month"], y=monthly_sel["super_late_pct"], name="Super Late %",
                                     mode="lines+markers", line=dict(color=STATUS_COLORS["Super Late"], width=2.5)))
            wv = monthly_sel[monthly_sel["month"].isin(worst_months)]
            if not wv.empty:
                fig.add_trace(go.Scatter(x=wv["month"], y=wv["late_pct"], name="3 worst", mode="markers",
                                         marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=RED, width=2)),
                                         hoverinfo="skip"))
            fig.update_xaxes(type="category", tickangle=-90)
            fig.update_yaxes(title_text="% of orders", rangemode="tozero")
            show(style(fig, 360))

        with st.container(border=True):
            st.markdown("**Late % by region and month** (spike hits every region)")
            hm = sel_main[sel_main["month"].isin(valid_months)].groupby(["region", "month"])[["orders", "late_orders"]].sum().reset_index()
            hm = hm[hm["orders"] >= 30]
            hm["late_pct"] = hm["late_orders"] / hm["orders"] * 100
            piv = hm.pivot(index="region", columns="month", values="late_pct").reindex(REG_SHOWN)
            show(style(heatmap(piv.values, piv.columns.tolist(), piv.index.tolist()), 300))

        t1, t2 = st.columns(2)
        with t1:
            with st.container(border=True):
                st.markdown("**Avg review by month**")
                fig = go.Figure(go.Scatter(x=monthly_sel["month"], y=monthly_sel["avg_review"], mode="lines+markers",
                                           line=dict(color=GREEN, width=2.5), fill="tozeroy", fillcolor=rgba(GREEN, 0.12)))
                fig.update_xaxes(type="category", tickangle=-90)
                fig.update_yaxes(range=[1, 5], title_text="★")
                show(style(fig, 320))
        with t2:
            with st.container(border=True):
                st.markdown("**Promised vs actual days by month**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=monthly_sel["month"], y=monthly_sel["avg_promised"], name="Promised",
                                         mode="lines+markers", line=dict(color=BLUE, width=2.5)))
                fig.add_trace(go.Scatter(x=monthly_sel["month"], y=monthly_sel["avg_actual"], name="Actual",
                                         mode="lines+markers", line=dict(color=RED, width=2.5)))
                fig.update_xaxes(type="category", tickangle=-90)
                fig.update_yaxes(title_text="days")
                show(style(fig, 320))
    else:
        st.info("Pick 2+ months for the trend charts.")

    w1, w2 = st.columns(2)
    flag = base_main["month"].isin(worst_months).map({True: "3 worst months", False: "Normal months"})
    reg_spike = base_main.assign(period=flag).groupby(["region", "period"])[["orders", "late_orders"]].sum().reset_index()
    reg_spike["late_pct"] = reg_spike["late_orders"] / reg_spike["orders"] * 100
    with w1:
        with st.container(border=True):
            st.markdown("**Normal vs worst months, by region** (all dates)")
            fig = go.Figure()
            for period_name, color in [("Normal months", GREEN), ("3 worst months", RED)]:
                sub = reg_spike[reg_spike["period"] == period_name].set_index("region").reindex(REG_SHOWN)
                fig.add_trace(go.Bar(x=sub.index, y=sub["late_pct"], name=period_name, marker_color=color,
                                     text=[pct(v) for v in sub["late_pct"]], textposition="outside"))
            fig.update_layout(barmode="group")
            fig.update_yaxes(title_text="% late", rangemode="tozero")
            show(style(fig, 340))
    with w2:
        with st.container(border=True):
            st.markdown("**Worst 3 months: share of orders vs late orders**")
            fig = go.Figure(go.Bar(x=["% of orders", "% of late orders"], y=[spike_share_orders, spike_share_late],
                                   marker_color=[BLUE, RED], text=[pct(spike_share_orders), pct(spike_share_late)],
                                   textposition="outside"))
            fig.update_yaxes(range=[0, max(spike_share_late, spike_share_orders) * 1.25], title_text="%")
            show(style(fig, 340))
    st.caption(f"Rio de Janeiro: {int(rj_late):,} late orders vs {int(ne['late_orders']):,} in the whole Northeast; "
               f"{rj_spike_share:.0f}% of Rio's late orders fall in the 3 worst months. Cause unknown from the data.")

# ---------------------------------------------------------------- Reviews
with tab_rev:
    on_rows = sel_main[sel_main["delivery_status"] == "On Time"]
    late_rows = sel_main[sel_main["delivery_status"] != "On Time"]
    on_avg = on_rows["review_sum"].sum() / on_rows["review_count"].sum() if on_rows["review_count"].sum() else np.nan
    late_avg = late_rows["review_sum"].sum() / late_rows["review_count"].sum() if late_rows["review_count"].sum() else np.nan

    c = st.columns(4)
    c[0].container(border=True).metric("Score · on time", "–" if pd.isna(on_avg) else f"{on_avg:.2f} ★")
    c[1].container(border=True).metric("Score · late", "–" if pd.isna(late_avg) else f"{late_avg:.2f} ★",
                                       delta=None if pd.isna(late_avg) or pd.isna(on_avg) else f"{late_avg - on_avg:+.2f}")
    c[2].container(border=True).metric("1-star · on time", pct(status_sel.loc["On Time", "one_star_pct"]))
    c[3].container(border=True).metric("1-star · super late", pct(status_sel.loc["Super Late", "one_star_pct"]))

    l_col, r_col = st.columns([2, 3])
    with l_col:
        with st.container(border=True):
            st.markdown("**Avg review by delivery status**")
            fig = go.Figure(go.Bar(
                x=STATUS_ORDER, y=status_sel["avg_review"], marker_color=[STATUS_COLORS[s] for s in STATUS_ORDER],
                text=[num(v) for v in status_sel["avg_review"]], textposition="outside",
                customdata=status_sel["review_count"].fillna(0).to_numpy(),
                hovertemplate="%{x}: %{y:.2f} ★ (n=%{customdata:,.0f})<extra></extra>"))
            fig.update_yaxes(range=[0, 5.5], title_text="avg review")
            show(style(fig, 360))
    with r_col:
        with st.container(border=True):
            st.markdown("**Review score by days late**")
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
                    hovertemplate="%{x} d late: %{y:.2f} ★ (n=%{customdata:,})<extra></extra>"))
                fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(200,200,200,0.8)")
                fig.add_vline(x=5.5, line_dash="dash", line_color=RED)
                fig.update_xaxes(title_text="days late (negative = early)")
                fig.update_yaxes(title_text="avg review", range=[0.8, 5])
                show(style(fig, 360))
            else:
                st.info("Not enough reviews.")

    r1, r2, r3 = st.columns(3)
    with r1:
        with st.container(border=True):
            st.markdown("**1-star share by status**")
            fig = go.Figure(go.Bar(x=STATUS_ORDER, y=status_sel["one_star_pct"], marker_color=[STATUS_COLORS[s] for s in STATUS_ORDER],
                                   text=[pct(v) for v in status_sel["one_star_pct"]], textposition="outside"))
            fig.update_yaxes(title_text="% of reviews", range=[0, max(status_sel["one_star_pct"].max() * 1.2, 1)])
            show(style(fig, 330))
    with r2:
        with st.container(border=True):
            st.markdown("**Review by region and status**")
            rs_tbl = summarize(sel_main, ["region", "delivery_status"])
            fig = go.Figure()
            for s in STATUS_ORDER:
                sub = rs_tbl[rs_tbl["delivery_status"] == s].set_index("region").reindex(REG_SHOWN)
                fig.add_trace(go.Bar(x=REG_SHOWN, y=sub["avg_review"], name=s, marker_color=STATUS_COLORS[s]))
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 5], title_text="avg review")
            show(style(fig, 330))
    with r3:
        with st.container(border=True):
            st.markdown("**State: late rate vs review** (bubble = orders)")
            sv = state_sel.dropna(subset=["avg_review"])
            fig = go.Figure()
            for r in REG_SHOWN:
                sub = sv[sv["region"] == r]
                if not sub.empty:
                    fig.add_trace(go.Scatter(x=sub["late_pct"], y=sub["avg_review"], mode="markers+text", name=r,
                                             text=sub["customer_state"], textposition="top center",
                                             marker=dict(size=np.clip(np.sqrt(sub["orders"]) * 0.6, 7, 40),
                                                         color=REGION_COLORS[r], opacity=0.8),
                                             hovertemplate="%{text}: %{x:.1f}% late · %{y:.2f} ★<extra></extra>"))
            if len(sv) >= 3:
                slope, intercept = np.polyfit(sv["late_pct"], sv["avg_review"], 1)
                xs = np.array([sv["late_pct"].min(), sv["late_pct"].max()])
                fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="trend", showlegend=False,
                                         line=dict(color="rgba(200,200,200,0.7)", dash="dot"), hoverinfo="skip"))
            fig.update_xaxes(title_text="% late", rangemode="tozero")
            fig.update_yaxes(title_text="avg review")
            show(style(fig, 330))

    sr = sel_main.assign(is_late=sel_main["delivery_status"] != "On Time")
    rs = sr.pivot_table(index="customer_state", columns="is_late", values="review_sum", aggfunc="sum")
    rc = sr.pivot_table(index="customer_state", columns="is_late", values="review_count", aggfunc="sum")
    if True in rc.columns and False in rc.columns:
        enough = (rc[True] >= 30) & (rc[False] >= 30)
        if enough.any():
            gap = (rs[False] / rc[False] - rs[True] / rc[True])[enough].sort_values()
            with st.container(border=True):
                st.markdown(f"**Review drop when late, by state** · late scores lower in {int((gap > 0).sum())} of {len(gap)} states")
                fig = go.Figure(go.Bar(x=gap.index, y=gap.values, marker_color=[GREEN if v > 0 else RED for v in gap.values],
                                       hovertemplate="%{x}: %{y:.2f} ★ lower when late<extra></extra>"))
                fig.update_yaxes(title_text="★ lost when late")
                show(style(fig, 300))
    st.caption("Score barely moves however early the parcel arrives, then drops right after the promised date. "
               "Observational data: strong link, not proven cause.")

# ---------------------------------------------------------------- Categories
with tab_cat:
    s_col1, s_col2 = st.columns(2)
    min_orders = s_col1.slider("Min orders per category", 50, 2000, 300, step=50, key="cat_min")
    top_n = s_col2.slider("Categories per side", 5, 15, 10, key="cat_top")

    cat_table = cats.groupby("category")[["orders", "late_orders", "review_sum", "review_count"]].sum().reset_index()
    cat_table["late_pct"] = cat_table["late_orders"] / cat_table["orders"] * 100
    cat_table["avg_review"] = cat_table["review_sum"] / cat_table["review_count"]
    cat_table = cat_table[cat_table["orders"] >= min_orders].sort_values("late_pct")

    if len(cat_table) < 2:
        st.warning("Not enough categories. Lower the minimum.")
    else:
        shown = pd.concat([cat_table.head(top_n), cat_table.tail(top_n)]) if len(cat_table) > 2 * top_n else cat_table
        n_easy = min(top_n, len(shown))
        colors = [GREEN] * n_easy + [RED] * (len(shown) - n_easy)

        grp = groups.groupby("group")[["orders", "late_orders", "review_sum", "review_count"]].sum()
        fe = [n for n in ["Furniture", "Electronics"] if n in grp.index and grp.loc[n, "orders"] > 0]

        c = st.columns(4)
        c[0].container(border=True).metric("Categories shown", f"{len(cat_table)}")
        c[1].container(border=True).metric("Easiest", cat_table.iloc[0]["category"].replace("_", " "),
                                           delta=f"{cat_table.iloc[0]['late_pct']:.1f}% late", delta_color="off")
        c[2].container(border=True).metric("Hardest", cat_table.iloc[-1]["category"].replace("_", " "),
                                           delta=f"{cat_table.iloc[-1]['late_pct']:.1f}% late", delta_color="off")
        if len(fe) == 2:
            fl = grp.loc["Furniture", "late_orders"] / grp.loc["Furniture", "orders"] * 100
            el = grp.loc["Electronics", "late_orders"] / grp.loc["Electronics", "orders"] * 100
            c[3].container(border=True).metric("Furniture vs electronics", f"{fl:.1f}% vs {el:.1f}%", delta=f"{fl - el:+.1f} pts", delta_color="off")
        else:
            c[3].container(border=True).metric("Furniture vs electronics", "–")

        l_col, r_col = st.columns([3, 2])
        with l_col:
            with st.container(border=True):
                st.markdown("**Easiest and hardest categories**")
                fig = go.Figure(go.Bar(
                    x=shown["late_pct"], y=[x.replace("_", " ") for x in shown["category"]], orientation="h",
                    marker_color=colors, text=[f"{p:.1f}%" for p in shown["late_pct"]], textposition="outside",
                    cliponaxis=False, customdata=shown[["orders", "avg_review"]].to_numpy(),
                    hovertemplate="%{y}: %{x:.1f}% late<br>%{customdata[0]:,} orders · %{customdata[1]:.2f} ★<extra></extra>"))
                fig.add_vline(x=vals["late_pct"], line_dash="dash", line_color="rgba(200,200,200,0.9)")
                fig.update_xaxes(title_text="% late", range=[0, shown["late_pct"].max() * 1.2])
                show(style(fig, max(420, 24 * len(shown))))
        with r_col:
            with st.container(border=True):
                st.markdown("**Furniture vs electronics**")
                if fe:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=fe, y=[grp.loc[n, "late_orders"] / grp.loc[n, "orders"] * 100 for n in fe],
                                         marker_color=[ORANGE, BLUE][:len(fe)],
                                         text=[pct(grp.loc[n, "late_orders"] / grp.loc[n, "orders"] * 100) for n in fe],
                                         textposition="outside"))
                    fig.update_yaxes(title_text="% late", rangemode="tozero")
                    show(style(fig, 200))
                    fig = go.Figure(go.Bar(x=fe, y=[grp.loc[n, "review_sum"] / grp.loc[n, "review_count"] for n in fe],
                                           marker_color=[ORANGE, BLUE][:len(fe)],
                                           text=[num(grp.loc[n, "review_sum"] / grp.loc[n, "review_count"]) for n in fe],
                                           textposition="outside"))
                    fig.update_yaxes(title_text="avg review", range=[0, 5.5])
                    show(style(fig, 200))
                else:
                    st.caption("No orders in this selection.")

        x1, x2 = st.columns(2)
        with x1:
            with st.container(border=True):
                st.markdown("**Late rate vs review** (bubble = orders)")
                ct = cat_table.dropna(subset=["avg_review"])
                fig = go.Figure(go.Scatter(
                    x=ct["late_pct"], y=ct["avg_review"], mode="markers",
                    marker=dict(size=np.clip(np.sqrt(ct["orders"]) * 0.45, 7, 36), color=ct["late_pct"], colorscale=HEAT, opacity=0.8),
                    text=[x.replace("_", " ") for x in ct["category"]],
                    hovertemplate="%{text}: %{x:.1f}% late · %{y:.2f} ★<extra></extra>"))
                fig.update_xaxes(title_text="% late", rangemode="tozero")
                fig.update_yaxes(title_text="avg review")
                show(style(fig, 360))
        with x2:
            with st.container(border=True):
                st.markdown("**Biggest categories by orders**")
                big = cat_table.nlargest(10, "orders").sort_values("orders")
                fig = go.Figure(go.Bar(x=big["orders"], y=[x.replace("_", " ") for x in big["category"]], orientation="h",
                                       marker=dict(color=big["late_pct"], colorscale=HEAT),
                                       text=[f"{p:.1f}% late" for p in big["late_pct"]], textposition="outside", cliponaxis=False,
                                       hovertemplate="%{y}: %{x:,} orders<extra></extra>"))
                fig.update_xaxes(range=[0, big["orders"].max() * 1.3])
                show(style(fig, 360))
        st.caption("Category gaps are small next to state and month gaps. Orders with mixed categories count once in each.")
        with st.expander("All categories"):
            show_table(cat_table.sort_values("late_pct", ascending=False)[["category", "orders", "late_pct", "avg_review"]]
                       .rename(columns={"category": "Category", "orders": "Orders", "late_pct": "Late %", "avg_review": "Avg review"})
                       .round(2))

# ---------------------------------------------------------------- Seller vs carrier
with tab_seller:
    h = sel_h
    if h["orders"].sum() == 0:
        st.info("No orders with valid hand-off dates in this selection.")
    else:
        by_missed = h.groupby("seller_missed")[["orders", "late_orders"]].sum().reindex([False, True])
        by_missed["late_pct"] = by_missed["late_orders"] / by_missed["orders"] * 100
        total_late = h["late_orders"].sum()
        share_missed_late = h["late_missed_orders"].sum() / total_late * 100 if total_late > 0 else np.nan
        p_miss, p_met = by_missed.loc[True, "late_pct"], by_missed.loc[False, "late_pct"]
        avoidable = (h["late_missed_orders"].sum() - h["missed_orders"].sum() * p_met / 100) if pd.notna(p_met) else np.nan

        c = st.columns(5)
        c[0].container(border=True).metric("Seller missed deadline", pct(h["missed_orders"].sum() / h["orders"].sum() * 100))
        c[1].container(border=True).metric("Late · seller on time", pct(p_met))
        c[2].container(border=True).metric("Late · seller missed", pct(p_miss),
                                           delta=f"{p_miss / p_met:.1f}× higher" if pd.notna(p_miss) and pd.notna(p_met) and p_met > 0 else None,
                                           delta_color="inverse")
        c[3].container(border=True).metric("Late orders, seller on time",
                                           "–" if pd.isna(share_missed_late) else f"{100 - share_missed_late:.0f}%")
        c[4].container(border=True).metric("Avoidable late orders", "–" if pd.isna(avoidable) else f"{avoidable:,.0f}",
                                           help="Late orders that would disappear if sellers who missed the deadline had the on-time late rate.")

        l_col, m_col, r_col = st.columns(3)
        with l_col:
            with st.container(border=True):
                st.markdown("**Where the time goes (avg days)**")
                times = h.groupby("delivery_status")[["orders", "seller_days_sum", "after_handoff_days_sum"]].sum().reindex(STATUS_ORDER)
                fig = go.Figure()
                fig.add_trace(go.Bar(x=STATUS_ORDER, y=times["seller_days_sum"] / times["orders"], name="Seller (before hand-off)", marker_color=GREY))
                fig.add_trace(go.Bar(x=STATUS_ORDER, y=times["after_handoff_days_sum"] / times["orders"], name="After hand-off", marker_color=BLUE))
                fig.update_layout(barmode="stack")
                fig.update_yaxes(title_text="days")
                show(style(fig, 360))
        with m_col:
            with st.container(border=True):
                st.markdown("**Late rate vs seller deadline**")
                fig = go.Figure(go.Bar(
                    x=["Seller on time", "Seller missed"], y=by_missed["late_pct"].to_numpy(),
                    marker_color=[GREEN, RED], text=[pct(v) for v in by_missed["late_pct"]], textposition="outside"))
                top = by_missed["late_pct"].max()
                fig.update_yaxes(title_text="% late", range=[0, max(top * 1.25, 1) if pd.notna(top) else 1])
                show(style(fig, 360))
        with r_col:
            with st.container(border=True):
                st.markdown("**Late orders: who was late first?**")
                lm = h["late_missed_orders"].sum()
                fig = go.Figure(go.Pie(labels=["Seller on time (delay after hand-off)", "Seller missed deadline"],
                                       values=[total_late - lm, lm], hole=0.6, sort=False,
                                       marker=dict(colors=[BLUE, RED]), textinfo="percent"))
                show(style(fig, 360))

        late_h = h[h["delivery_status"] != "On Time"]
        by_reg = late_h.groupby("region")[["orders", "late_missed_orders", "after_handoff_days_sum"]].sum().reset_index()
        by_reg["seller_missed_share"] = by_reg["late_missed_orders"] / by_reg["orders"] * 100
        by_reg["after_handoff"] = by_reg["after_handoff_days_sum"] / by_reg["orders"]
        by_reg = by_reg.sort_values("seller_missed_share")
        reg_miss = h.groupby(["region", "seller_missed"])[["orders", "late_orders"]].sum().reset_index()
        reg_miss["late_pct"] = reg_miss["late_orders"] / reg_miss["orders"] * 100

        q1, q2, q3 = st.columns(3)
        with q1:
            with st.container(border=True):
                st.markdown("**Late orders with a late seller, by region**")
                fig = go.Figure(go.Bar(
                    x=by_reg["seller_missed_share"], y=by_reg["region"], orientation="h",
                    marker_color=[REGION_COLORS[r] for r in by_reg["region"]],
                    text=[f"{v:.0f}%" for v in by_reg["seller_missed_share"]], textposition="outside", cliponaxis=False,
                    customdata=by_reg[["orders", "after_handoff"]].to_numpy(),
                    hovertemplate="%{y}: %{x:.0f}% of %{customdata[0]:,} late orders<br>"
                                  "avg %{customdata[1]:.1f} d after hand-off<extra></extra>"))
                fig.update_xaxes(title_text="% of late orders", range=[0, max(by_reg["seller_missed_share"].max() * 1.3, 1)])
                show(style(fig, 340))
        with q2:
            with st.container(border=True):
                st.markdown("**Late rate by region: seller on time vs missed**")
                fig = go.Figure()
                for flag_val, name, color in [(False, "Seller on time", GREEN), (True, "Seller missed", RED)]:
                    sub = reg_miss[reg_miss["seller_missed"] == flag_val].set_index("region").reindex(REG_SHOWN)
                    fig.add_trace(go.Bar(x=REG_SHOWN, y=sub["late_pct"], name=name, marker_color=color))
                fig.update_layout(barmode="group")
                fig.update_yaxes(title_text="% late")
                show(style(fig, 340))
        with q3:
            with st.container(border=True):
                st.markdown("**Seller deadline misses by month**")
                if len(h_month) >= 2:
                    fig = go.Figure(go.Scatter(x=h_month["month"], y=h_month["missed_pct"], mode="lines+markers",
                                               line=dict(color=GREY, width=2.5), fill="tozeroy", fillcolor=rgba(GREY, 0.18)))
                    fig.update_xaxes(type="category", tickangle=-90)
                    fig.update_yaxes(title_text="% of orders", rangemode="tozero")
                    show(style(fig, 340))
                else:
                    st.info("Pick 2+ months.")

        st.caption(f"Based on {int(h['orders'].sum()):,} of {int(sel_main['orders'].sum()):,} delivered orders (valid hand-off dates only). "
                   "No carrier field in the data, so carriers cannot be compared.")
        st.markdown(
            """
| Problem | Action |
|---|---|
| Delay after hand-off in the Northeast | Get carrier + tracking data, then fix carriers or hubs |
| One flat estimate cushion | Set promised date per state from real delivery times |
| Bad months (Nov 2017, Feb 2018, Mar 2018) | Investigate; cause unknown from the data |
| Sellers missing deadlines | Scorecards and penalties (smaller effect) |
"""
        )

# ---------------------------------------------------------------- About
with tab_about:
    st.markdown("""
**Data:** Olist Brazilian E-Commerce (Kaggle), ~99,000 orders, 2016-2018. Tables built in notebook section 8 (`dashboard_data/`); they keep sums and counts so every % is recalculated after filtering.

**Rules**
- Only `delivered` orders with a delivery date. One review per order (latest).
- `Days_Difference` = estimated − actual delivery date (dates only).
- Late = Late + Super Late. Super Late = more than 5 days late.
- Monthly charts group by purchase month and hide months with < 300 orders.

**Filters:** dates + region change everything except the delay histogram and day-by-day reviews (region only), normal-vs-worst-months charts (region only) and the top answer cards (all data).

**Limits:** observational data (link, not cause) · no carrier field · the 3 worst months were picked because they were worst, cause unknown · small states and categories are noisy.
""")
