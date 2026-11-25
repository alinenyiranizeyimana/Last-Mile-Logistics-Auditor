import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st

st.set_page_config(page_title='Last Mile Logistics Audit', page_icon=':truck:', layout='wide')

DATA = Path(__file__).parent / 'dashboard_data'

STATUS_ORDER = ['On Time', 'Late', 'Super Late']
STATUS_COLORS = {'On Time': '#2a9d8f', 'Late': '#f4a261', 'Super Late': '#d62828'}
REGION_ORDER = ['North', 'Northeast', 'Central-West', 'Southeast', 'South']
REGION_COLORS = {'North': '#6a4c93', 'Northeast': '#d62828', 'Central-West': '#f4a261',
                 'Southeast': '#2a9d8f', 'South': '#1d6fa5'}

# columns that can be added up (the tables hold sums and counts, so I can recalculate % after filtering)
SUM_COLS = ['orders', 'late_orders', 'super_late_orders', 'review_sum', 'review_count',
            'one_star', 'promised_sum', 'actual_sum']


# ---------- loading data ----------

@st.cache_data
def load_data():
    main = pd.read_csv(DATA / 'dash_main.csv')
    states = pd.read_csv(DATA / 'dash_states.csv')
    days = pd.read_csv(DATA / 'dash_days.csv')
    cats = pd.read_csv(DATA / 'dash_categories.csv')
    groups = pd.read_csv(DATA / 'dash_groups.csv')
    handoff = pd.read_csv(DATA / 'dash_handoff.csv')
    with open(DATA / 'brazil_states.geojson') as f:
        geojson = json.load(f)

    # Late = Late + Super Late
    main['late_orders'] = main['orders'].where(main['delivery_status'] != 'On Time', 0)
    main['super_late_orders'] = main['orders'].where(main['delivery_status'] == 'Super Late', 0)
    return main, states, days, cats, groups, handoff, geojson


# ---------- small helper functions ----------

def show(fig):
    # newer streamlit versions use width='stretch', older ones use use_container_width
    fig.update_layout(template='plotly_white', margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, theme=None, width='stretch')
    except TypeError:
        st.plotly_chart(fig, theme=None, use_container_width=True)


def summarize(df, by):
    # add up the sums and counts for each group, then calculate the percentages
    out = df.groupby(by)[SUM_COLS].sum().reset_index()
    out['late_pct'] = out['late_orders'] / out['orders'] * 100
    out['super_late_pct'] = out['super_late_orders'] / out['orders'] * 100
    out['avg_review'] = out['review_sum'] / out['review_count']
    out['one_star_pct'] = out['one_star'] / out['review_count'] * 100
    out['avg_promised'] = out['promised_sum'] / out['orders']
    out['avg_actual'] = out['actual_sum'] / out['orders']
    return out


def pct(x):
    return 'n/a' if pd.isna(x) else f'{x:.1f}%'


def num(x, digits=2):
    return 'n/a' if pd.isna(x) else f'{x:.{digits}f}'


# lighter version of YlOrRd (the darkest red is cut off so the black state codes can still be read)
MAP_SCALE = [[i / 4, sample_colorscale('YlOrRd', [0.8 * i / 4])[0]] for i in range(5)]


def make_map(state_view, geojson):
    # I draw each state as a filled shape. This way the map needs no internet and no map tiles.
    values = state_view.set_index('customer_state')
    low, high = state_view['late_pct'].min(), state_view['late_pct'].max()

    fig = go.Figure()
    for feature in geojson['features']:
        code = feature['properties']['sigla']
        geometry = feature['geometry']
        if geometry['type'] == 'MultiPolygon':
            polygons = geometry['coordinates']
        else:
            polygons = [geometry['coordinates']]

        xs, ys = [], []
        for polygon in polygons:
            ring = polygon[0]                      # outer border of the shape
            xs += [point[0] for point in ring] + [None]
            ys += [point[1] for point in ring] + [None]

        if code in values.index:
            row = values.loc[code]
            share = 0.5 if high == low else (row['late_pct'] - low) / (high - low)
            color = sample_colorscale(MAP_SCALE, [share])[0]
            text = f"<b>{code}</b><br>late: {row['late_pct']:.1f}%<br>orders: {int(row['orders']):,}"
        else:
            color = '#e5e5e5'
            text = f'{code}<br>not in the selection'
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', fill='toself', fillcolor=color, line=dict(color='white', width=1),
                                 text=text, hoverinfo='text', hoveron='fills', showlegend=False))

    # state codes on top of the shapes
    fig.add_trace(go.Scatter(x=state_view['lng'], y=state_view['lat'], mode='text', text=state_view['customer_state'],
                             textfont=dict(size=11, color='black'), hoverinfo='skip', showlegend=False))

    # invisible points, only there to show the colour bar
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', hoverinfo='skip', showlegend=False,
                             marker=dict(colorscale=MAP_SCALE, cmin=low, cmax=high, color=[low], showscale=True,
                                         colorbar=dict(title='% late', thickness=15, len=0.7))))

    fig.update_xaxes(visible=False, range=[-74, -33])
    fig.update_yaxes(visible=False, range=[-34.5, 5.5], scaleanchor='x', scaleratio=1)
    fig.update_layout(height=650, title='Late delivery rate by state (hover for details)')
    return fig


def get_status(days_difference):
    if days_difference >= 0:
        return 'On Time'
    elif days_difference >= -5:
        return 'Late'
    else:
        return 'Super Late'


main_all, states, days_all, cats_all, groups_all, handoff_all, geojson = load_data()

# ---------- sidebar filters ----------

st.sidebar.title('Filters')
regions = st.sidebar.multiselect('Region', REGION_ORDER, default=REGION_ORDER)
months = sorted(main_all['month'].unique())
start_month, end_month = st.sidebar.select_slider('Purchase month', options=months, value=(months[0], months[-1]))
st.sidebar.caption('The filters change every tab except Summary (which always uses all the data). '
                   'The review-by-delay chart only uses the region filter.')

if len(regions) == 0:
    st.warning('Pick at least one region in the sidebar.')
    st.stop()

in_filter = main_all['region'].isin(regions) & (main_all['month'] >= start_month) & (main_all['month'] <= end_month)
main = main_all[in_filter]
days = days_all[days_all['region'].isin(regions)]
cats = cats_all[cats_all['region'].isin(regions) & (cats_all['month'] >= start_month) & (cats_all['month'] <= end_month)]
groups = groups_all[groups_all['region'].isin(regions) & (groups_all['month'] >= start_month) & (groups_all['month'] <= end_month)]
handoff = handoff_all[handoff_all['region'].isin(regions) & (handoff_all['month'] >= start_month) & (handoff_all['month'] <= end_month)]

if main['orders'].sum() == 0:
    st.warning('No orders for this selection.')
    st.stop()

# ---------- title and KPIs ----------

st.title('Last Mile Logistics Audit')
st.caption('Veridi Logistics | Olist Brazilian e-commerce data | Data Engineering Project Challenge - Aline Nyiranizeyimana')

total = main[SUM_COLS].sum()
total_all = main_all[SUM_COLS].sum()
late_pct = total['late_orders'] / total['orders'] * 100
national_late = total_all['late_orders'] / total_all['orders'] * 100
super_late_pct = total['super_late_orders'] / total['orders'] * 100

on_time_rows = main[main['delivery_status'] == 'On Time']
late_rows = main[main['delivery_status'] != 'On Time']
review_on_time = on_time_rows['review_sum'].sum() / on_time_rows['review_count'].sum()
review_late = late_rows['review_sum'].sum() / late_rows['review_count'].sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Delivered orders', f"{int(total['orders']):,}")
if abs(late_pct - national_late) < 0.05:
    c2.metric('Late (Late + Super Late)', pct(late_pct))      # all data selected, nothing to compare with
else:
    c2.metric('Late (Late + Super Late)', pct(late_pct), delta=f'{late_pct - national_late:+.1f} pts vs all data', delta_color='inverse')
c3.metric('Super Late (> 5 days)', pct(super_late_pct))
c4.metric('Avg days: promised / actual', f"{total['promised_sum'] / total['orders']:.1f} / {total['actual_sum'] / total['orders']:.1f}")
c5.metric('Avg review: on time / late', f'{num(review_on_time)} / {num(review_late)}')

tab_summary, tab_delay, tab_where, tab_reviews, tab_cats, tab_seller, tab_about = st.tabs(
    ['Summary', 'Delay (Story 2)', 'Where (Story 3)', 'Reviews (Story 4)', 'Categories (Bonus)',
     'Seller vs after hand-off', 'About'])


# ---------- Summary (always all data) ----------

with tab_summary:
    st.header('Is Veridi failing specific regions, or is it nationwide?')
    st.caption('This tab always uses all the data, whatever the filters say.')

    by_region = summarize(main_all, ['region'])
    by_state = summarize(main_all, ['customer_state'])
    by_month = summarize(main_all, ['month'])
    by_status = summarize(main_all, ['delivery_status']).set_index('delivery_status')

    # the 3 months with the highest late % (only months with 300+ orders)
    worst_months = sorted(by_month[by_month['orders'] >= 300].nlargest(3, 'late_pct')['month'].tolist())
    main_all['period'] = main_all['month'].isin(worst_months).map({True: 'Worst 3 months', False: 'Other months'})
    by_period = summarize(main_all, ['period']).set_index('period')

    ne = main_all[main_all['region'] == 'Northeast'][SUM_COLS].sum()
    rest = main_all[main_all['region'] != 'Northeast'][SUM_COLS].sum()
    ne_late = ne['late_orders'] / ne['orders'] * 100
    rest_late = rest['late_orders'] / rest['orders'] * 100

    days_total = days_all['orders'].sum()
    early_share = days_all[days_all['days_late'] < 0]['orders'].sum() / days_total * 100

    h_missed = handoff_all.assign(late=handoff_all['delivery_status'] != 'On Time')
    h_missed['late_orders'] = h_missed['orders'].where(h_missed['late'], 0)
    miss = h_missed.groupby('seller_missed')[['orders', 'late_orders']].sum()
    late_if_missed = miss.loc[True, 'late_orders'] / miss.loc[True, 'orders'] * 100
    late_if_met = miss.loc[False, 'late_orders'] / miss.loc[False, 'orders'] * 100
    seller_share = miss.loc[True, 'late_orders'] / miss['late_orders'].sum() * 100

    best_state = by_state.loc[by_state['late_pct'].idxmin()]
    worst_state = by_state.loc[by_state['late_pct'].idxmax()]
    worst_months_text = ', '.join(worst_months)

    st.markdown(f"""
**Short answer: both, and they are two different problems.**

1. **A regional problem in the Northeast.** The Northeast is late **{ne_late:.1f}%** of the time, against **{rest_late:.1f}%** in the rest of the country.
   Late rates go from {best_state['late_pct']:.1f}% ({best_state['customer_state']}) to {worst_state['late_pct']:.1f}% ({worst_state['customer_state']}).
2. **A problem in a few months, in every region.** In the 3 worst months ({worst_months_text}) the late rate was **{by_period.loc['Worst 3 months', 'late_pct']:.1f}%**,
   against **{by_period.loc['Other months', 'late_pct']:.1f}%** in all the other months. I picked these months because they were the worst, so the share of late orders
   in them is partly built in. The gap in the rate is the better evidence. The data does not say what caused it.
3. **The CEO's idea about the estimates is only half right.** {early_share:.0f}% of orders arrive *before* the estimated date, so on average the estimates are padded, not too optimistic.
   But the padding is not the same in every state (see the Delay tab), so some states run out of cushion.
4. **Late deliveries hurt reviews a lot.** Average review score: **{by_status.loc['On Time', 'avg_review']:.2f}** on time,
   **{by_status.loc['Late', 'avg_review']:.2f}** late, **{by_status.loc['Super Late', 'avg_review']:.2f}** super late.
5. **Sellers are a smaller but real part.** Orders are late {late_if_missed:.1f}% of the time when the seller missed the hand-off deadline, against {late_if_met:.1f}% when the seller was on time.
   But only {seller_share:.0f}% of late orders involve a late seller. The dataset has no carrier field, so it cannot say which carrier is slow.
""")

    st.subheader('Late % in the worst 3 months vs the other months, by region')
    period_region = summarize(main_all, ['region', 'period'])
    fig = px.bar(period_region, x='region', y='late_pct', color='period', barmode='group',
                 category_orders={'region': REGION_ORDER, 'period': ['Other months', 'Worst 3 months']},
                 color_discrete_map={'Other months': '#2a9d8f', 'Worst 3 months': '#d62828'},
                 text=period_region['late_pct'].round(1), labels={'late_pct': '% of orders late', 'region': '', 'period': ''})
    fig.update_traces(textposition='outside')
    show(fig)


# ---------- Delay (Story 2) ----------

with tab_delay:
    st.header('How late are orders compared to the promised date?')
    st.markdown('`Days_Difference = estimated delivery date - actual delivery date` (dates only). '
                'Positive = early, negative = late. **On Time**: 0 or more. **Late**: 1 to 5 days late. **Super Late**: more than 5 days late.')

    left, right = st.columns([2, 1])

    hist = days.groupby('days_late')['orders'].sum().reset_index()
    hist['days_difference'] = -hist['days_late']
    hist['status'] = hist['days_difference'].apply(get_status)
    fig = px.bar(hist, x='days_difference', y='orders', color='status', category_orders={'status': STATUS_ORDER},
                 color_discrete_map=STATUS_COLORS, labels={'days_difference': 'Days_Difference (negative = late)', 'orders': 'orders'},
                 title='How early or late orders arrive')
    with left:
        show(fig)
        st.caption('The first and last bars include everything further out. This chart only uses the region filter.')

    with right:
        by_status = summarize(main, ['delivery_status']).set_index('delivery_status').reindex(STATUS_ORDER).reset_index()
        by_status['share_%'] = by_status['orders'] / by_status['orders'].sum() * 100
        st.subheader('Status of delivered orders')
        st.dataframe(by_status[['delivery_status', 'orders', 'share_%']].round(1), hide_index=True)
        st.caption('Orders that were canceled, unavailable or not yet delivered have no delivery date and are not included.')

    st.subheader('Do we pad the estimate the same way everywhere?')
    st.caption('Average promised days vs average actual days (purchase to delivery), by state. A small gap means a small cushion.')
    state_days = summarize(main, ['customer_state', 'region'])
    state_days['cushion_days'] = state_days['avg_promised'] - state_days['avg_actual']
    state_days = state_days.sort_values('avg_promised')
    long = state_days.melt(id_vars=['customer_state'], value_vars=['avg_promised', 'avg_actual'],
                           var_name='what', value_name='days')
    long['what'] = long['what'].map({'avg_promised': 'Promised', 'avg_actual': 'Actual'})
    fig = px.bar(long, x='customer_state', y='days', color='what', barmode='group',
                 color_discrete_map={'Promised': '#1d6fa5', 'Actual': '#d62828'}, labels={'customer_state': '', 'days': 'average days', 'what': ''})
    show(fig)
    st.dataframe(state_days[['customer_state', 'region', 'orders', 'avg_promised', 'avg_actual', 'cushion_days', 'late_pct']]
                 .sort_values('cushion_days').round(1), hide_index=True)


# ---------- Where (Story 3) ----------

with tab_where:
    st.header('Which states have the most late deliveries?')

    state_view = summarize(main, ['customer_state', 'region']).merge(
        states[['customer_state', 'lat', 'lng', 'distance_km']], on='customer_state', how='left')

    fig = make_map(state_view, geojson)
    show(fig)

    st.subheader('Late % by state')
    bar_data = state_view.sort_values('late_pct')
    fig = px.bar(bar_data, x='late_pct', y='customer_state', orientation='h', color='region',
                 color_discrete_map=REGION_COLORS, hover_data={'orders': True},
                 category_orders={'customer_state': bar_data['customer_state'].tolist()[::-1], 'region': REGION_ORDER},
                 labels={'late_pct': '% late (Late + Super Late)', 'customer_state': ''})
    fig.add_vline(x=late_pct, line_dash='dash', line_color='black', annotation_text='average of selection',
                  annotation_position='bottom right')
    fig.update_layout(height=750)
    show(fig)
    st.caption('Small states have few orders (hover to see the number), so their percentages move around a lot.')

    st.subheader('Are remote states hit harder?')
    st.caption('Distance from the state to Sao Paulo (where most sellers are) against late %.')
    fig = px.scatter(state_view, x='distance_km', y='late_pct', size='orders', size_max=45, color='region', text='customer_state',
                     color_discrete_map=REGION_COLORS, labels={'distance_km': 'distance to Sao Paulo (km)', 'late_pct': '% late'})
    fig.update_traces(textposition='top center')
    show(fig)
    if len(state_view) >= 5:
        # spearman correlation = normal correlation of the ranks (this way scipy is not needed)
        corr = state_view['distance_km'].rank().corr(state_view['late_pct'].rank())
        st.write(f'Spearman correlation between distance and late % for the states in view: **{corr:.2f}**. '
                 'A value near 0 means distance on its own does not explain it.')

    st.subheader('By region')
    region_table = summarize(main, ['region']).set_index('region').reindex([r for r in REGION_ORDER if r in regions]).reset_index()
    st.dataframe(region_table[['region', 'orders', 'late_pct', 'super_late_pct', 'avg_review']].round(2), hide_index=True)

    st.subheader('Is it the same every month?')
    monthly = summarize(main, ['month'])
    monthly = monthly[monthly['orders'] >= 100]
    monthly = monthly.rename(columns={'late_pct': 'Late %', 'super_late_pct': 'Super Late %'})
    fig = px.line(monthly, x='month', y=['Late %', 'Super Late %'], markers=True,
                  color_discrete_sequence=['#f4a261', '#d62828'], labels={'value': '% of orders', 'month': '', 'variable': ''})
    show(fig)
    st.caption('Months with fewer than 100 orders in the selection are hidden.')


# ---------- Reviews (Story 4) ----------

with tab_reviews:
    st.header('Do late deliveries cause bad reviews?')

    rev = summarize(main, ['delivery_status']).set_index('delivery_status').reindex(STATUS_ORDER).reset_index()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(rev, x='delivery_status', y='avg_review', color='delivery_status', color_discrete_map=STATUS_COLORS,
                     text=rev['avg_review'].round(2), labels={'avg_review': 'average review score', 'delivery_status': ''},
                     title='Average review score')
        fig.update_yaxes(range=[0, 5])
        fig.update_layout(showlegend=False)
        show(fig)
    with col2:
        fig = px.bar(rev, x='delivery_status', y='one_star_pct', color='delivery_status', color_discrete_map=STATUS_COLORS,
                     text=rev['one_star_pct'].round(1), labels={'one_star_pct': '% of reviews with 1 star', 'delivery_status': ''},
                     title='Share of 1-star reviews')
        fig.update_layout(showlegend=False)
        show(fig)

    st.subheader('Review score for each day of delay')
    day_rev = days.groupby('days_late')[['review_sum', 'review_count', 'orders']].sum().reset_index()
    day_rev = day_rev[day_rev['days_late'].between(-20, 25) & (day_rev['review_count'] >= 30)]
    day_rev['avg_review'] = day_rev['review_sum'] / day_rev['review_count']
    fig = px.line(day_rev, x='days_late', y='avg_review', markers=True, hover_data={'review_count': True},
                  labels={'days_late': 'days late (negative = delivered early)', 'avg_review': 'average review score'})
    fig.add_vline(x=0.5, line_dash='dash', line_color='black')
    fig.add_vline(x=5.5, line_dash='dash', line_color='#d62828')
    fig.update_yaxes(range=[1, 5])
    show(fig)
    st.caption('The score stays about the same however early the parcel is, then drops quickly after the promised date. '
               'Days with fewer than 30 reviews are hidden. This chart only uses the region filter.')
    st.info('This is observational data, so it cannot prove cause for sure. But the drop is very large and it shows up in every state with enough reviews (checked in the notebook).')


# ---------- Categories (Bonus) ----------

with tab_cats:
    st.header('Product categories in English')

    col1, col2 = st.columns(2)
    min_orders = col1.slider('Minimum orders per category', 50, 2000, 300, step=50)
    top_n = col2.slider('Categories to show on each side', 5, 15, 10)

    cat_table = cats.groupby('category')[['orders', 'late_orders', 'review_sum', 'review_count']].sum().reset_index()
    cat_table['late_pct'] = cat_table['late_orders'] / cat_table['orders'] * 100
    cat_table['avg_review'] = cat_table['review_sum'] / cat_table['review_count']
    cat_table = cat_table[cat_table['orders'] >= min_orders].sort_values('late_pct')

    if len(cat_table) < 2:
        st.warning('Not enough categories with that many orders. Lower the minimum.')
    else:
        if len(cat_table) > 2 * top_n:
            shown = pd.concat([cat_table.head(top_n), cat_table.tail(top_n)])
        else:
            shown = cat_table
        shown = shown.assign(side=['Easiest'] * min(top_n, len(shown)) + ['Hardest'] * (len(shown) - min(top_n, len(shown))))
        fig = px.bar(shown, x='late_pct', y='category', orientation='h', color='side',
                     color_discrete_map={'Easiest': '#2a9d8f', 'Hardest': '#d62828'}, hover_data={'orders': True},
                     category_orders={'category': shown['category'].tolist()[::-1]},
                     labels={'late_pct': '% late', 'category': '', 'side': ''})
        fig.add_vline(x=late_pct, line_dash='dash', line_color='black', annotation_text='average of selection',
                      annotation_position='bottom right')
        fig.update_layout(height=max(400, 28 * len(shown)))
        show(fig)
        st.caption('An order with products from different categories is counted once in each of them.')

        with st.expander('Table with all categories'):
            st.dataframe(cat_table.sort_values('late_pct', ascending=False)[['category', 'orders', 'late_pct', 'avg_review']].round(2),
                         hide_index=True)

    st.subheader('Furniture vs electronics')
    grp = groups.groupby('group')[['orders', 'late_orders', 'review_sum', 'review_count']].sum()
    g1, g2 = st.columns(2)
    for column, name in [(g1, 'Furniture'), (g2, 'Electronics')]:
        if name in grp.index and grp.loc[name, 'orders'] > 0:
            row = grp.loc[name]
            column.metric(f'{name} - late %', pct(row['late_orders'] / row['orders'] * 100))
            column.caption(f"{int(row['orders']):,} orders | average review {num(row['review_sum'] / row['review_count'])}")
    st.caption('Furniture = all categories with "furniture" in the name. Electronics = electronics, computers, computers_accessories, '
               'telephony, fixed_telephony and audio together. An order is counted once per group.')


# ---------- Seller vs after hand-off ----------

with tab_seller:
    st.header('Is a late order the seller handing over late, or a delay after hand-off?')
    st.markdown('Each order has a purchase date, a date it was handed to the logistics partner and a delivery date. '
                'The seller has a deadline to hand the parcel over (`shipping_limit_date`). '
                '**The dataset has no carrier field, so this cannot say anything about individual carriers.**')

    h = handoff.assign(late=handoff['delivery_status'] != 'On Time')
    h['late_orders'] = h['orders'].where(h['late'], 0)
    h['late_and_missed'] = h['late_orders'].where(h['seller_missed'], 0)

    times = h.groupby('delivery_status')[['orders', 'seller_days_sum', 'after_handoff_days_sum']].sum().reindex(STATUS_ORDER)
    times['Before hand-off (seller)'] = times['seller_days_sum'] / times['orders']
    times['After hand-off'] = times['after_handoff_days_sum'] / times['orders']

    col1, col2 = st.columns(2)
    with col1:
        long = times.reset_index().melt(id_vars='delivery_status', value_vars=['Before hand-off (seller)', 'After hand-off'],
                                        var_name='stage', value_name='days')
        fig = px.bar(long, x='delivery_status', y='days', color='stage', barmode='stack',
                     color_discrete_map={'Before hand-off (seller)': '#8d99ae', 'After hand-off': '#1d6fa5'},
                     labels={'days': 'average days', 'delivery_status': '', 'stage': ''}, title='Where the time goes')
        show(fig)
    with col2:
        m = h.groupby('seller_missed')[['orders', 'late_orders']].sum()
        m['late_pct'] = m['late_orders'] / m['orders'] * 100
        m = m.rename(index={False: 'Seller met deadline', True: 'Seller missed deadline'}).reset_index()
        fig = px.bar(m, x='seller_missed', y='late_pct', color='seller_missed',
                     color_discrete_map={'Seller met deadline': '#2a9d8f', 'Seller missed deadline': '#d62828'},
                     text=m['late_pct'].round(1), labels={'late_pct': '% of orders late', 'seller_missed': ''},
                     title='Late % by seller punctuality')
        fig.update_layout(showlegend=False)
        show(fig)

    total_late = h['late_orders'].sum()
    if total_late > 0:
        share = h['late_and_missed'].sum() / total_late * 100
        st.metric('Late orders where the seller missed the hand-off deadline', pct(share))
        st.write(f'So about **{100 - share:.0f}%** of late orders happen even though the seller handed the parcel over on time. '
                 'Late orders also spend longer after hand-off, but that is partly by definition (they took longer overall).')

    st.subheader('By region')
    reg = h.groupby('region')[['orders', 'late_orders', 'late_and_missed']].sum().reset_index()
    reg['late_pct'] = reg['late_orders'] / reg['orders'] * 100
    reg['late_orders_with_late_seller_%'] = reg['late_and_missed'] / reg['late_orders'] * 100
    st.dataframe(reg[['region', 'orders', 'late_orders', 'late_pct', 'late_orders_with_late_seller_%']].round(1), hide_index=True)
    st.caption('Orders without a carrier hand-off date (or with dates in the wrong order) are not in this tab, so totals are a little lower than in the other tabs.')


# ---------- About ----------

with tab_about:
    st.header('About this dashboard')
    st.markdown("""
**Data:** Olist Brazilian E-Commerce dataset (Kaggle), about 99,000 orders from 2016 to 2018.
The dashboard reads small summary tables in `dashboard_data/`, created by the notebook `last_mile_logistics_audit.ipynb`.

**How the numbers are built**
- One review is kept per order (the latest), so joining reviews does not duplicate orders.
- Only orders with status `delivered` and a delivery date are used. Canceled, unavailable and in-progress orders are left out.
- `Days_Difference` compares dates only (the estimate has no time of day).
- Late = Late + Super Late. Super Late = more than 5 days after the estimated date.
- Product categories are translated with the translation file, plus two categories it misses (translated by hand).

**Limits**
- Observational data, so the review results show a strong link, not a proven cause.
- There is no carrier field, so nothing here says which carrier is slow.
- The 3 worst months are chosen because they were the worst, and the cause is unknown.
- Small states and categories have few orders, so their percentages are noisy.
""")
