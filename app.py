
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

TEAM = ["Soham Admane", "Vaishnav Balpande", "Lokesh Parate"]

st.set_page_config(
    page_title="E-Commerce Sales & Customer Behavior Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

NAVY = "#222B52"
BLUE = "#3B82F6"
INDIGO = "#4F46E5"
GREEN = "#10B981"
AMBER = "#F59E0B"
ROSE = "#EF4444"
VIOLET = "#8B5CF6"
SLATE = "#64748B"
PALETTE = [BLUE, GREEN, AMBER, VIOLET, "#EC4899", "#06B6D4", SLATE, "#F97316"]
STATUS_COLORS = {"On-Time": GREEN, "Slight Delay": AMBER, "Major Delay": ROSE, "Cancelled": SLATE}

pio.templates["dav"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#1E293B"),
        title=dict(font=dict(size=16, color=NAVY), x=0, xanchor="left"),
        colorway=PALETTE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=30),
        hoverlabel=dict(font=dict(family="Inter, Segoe UI, sans-serif")),
        xaxis=dict(gridcolor="rgba(148,163,184,0.18)", zeroline=False),
        yaxis=dict(gridcolor="rgba(148,163,184,0.18)", zeroline=False),
    )
)
pio.templates.default = "plotly_white+dav"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def show(fig):
    """Render a Plotly figure full-width with the toolbar hidden."""
    try:
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def show_table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def add_play_controls(fig, frame_ms=350, transition_ms=250, redraw=False):
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="left", showactive=False,
        x=1.0, y=1.02, xanchor="right", yanchor="bottom", pad=dict(r=0, t=0),
        bgcolor="#EEF2FF", bordercolor="#C7D2FE", font=dict(color=NAVY, size=12),
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, {"frame": {"duration": frame_ms, "redraw": redraw},
                              "transition": {"duration": transition_ms, "easing": "cubic-in-out"},
                              "fromcurrent": False, "mode": "immediate"}]),
            dict(label="❚❚ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0}, "mode": "immediate"}]),
        ],
    )])
    return fig


def start_at_end(fig):
    """Show the final animation frame on load, so the chart is never empty before Play."""
    if fig.frames:
        for trace, final in zip(fig.data, fig.frames[-1].data):
            trace.update(final)
        if fig.layout.sliders:
            fig.layout.sliders[0].active = len(fig.frames) - 1
    return fig


def reveal_line(x, y, title, x_title, y_title, color, frame_ms=140):
    """Line/area chart that draws itself point by point when Play is pressed."""
    x, y = list(x), list(y)
    n = len(x)
    fig = go.Figure(
        data=[go.Scatter(x=x, y=y, mode="lines+markers", fill="tozeroy",
                         line=dict(color=color, width=3, shape="spline"),
                         marker=dict(size=7, color=ROSE), fillcolor="rgba(59,130,246,0.12)",
                         name=y_title)],
        frames=[go.Frame(data=[go.Scatter(x=x[:k], y=y[:k])], name=str(k)) for k in range(1, n + 1)],
    )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, showlegend=False,
                      xaxis=dict(type="category", categoryorder="array", categoryarray=x,
                                 range=[-0.5, n - 0.5], tickangle=-45),
                      yaxis=dict(range=[0, max(y) * 1.12 if n else 1]))
    return add_play_controls(fig, frame_ms=frame_ms, transition_ms=0, redraw=True)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, .stTabs, button, input { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Hide Streamlit developer chrome for a clean presentation */
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    header[data-testid="stHeader"] { background: transparent; height: 0; }
    .main .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 1.4rem; padding-bottom: 2.5rem; padding-left: 2.4rem; padding-right: 2.4rem; max-width: 100%;
    }

    @keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes shimmer { 0% { background-position: 0% 50%; } 100% { background-position: 100% 50%; } }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #222B52 0%, #33407A 55%, #4F46E5 100%);
        background-size: 200% 200%;
        animation: fadeUp .6s ease both, shimmer 12s ease-in-out infinite alternate;
        border-radius: 18px; padding: 26px 32px; color: #F8F7F0; margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(34, 43, 82, 0.25);
    }
    .hero .eyebrow { font-size: .78rem; letter-spacing: 1.6px; text-transform: uppercase; opacity: .8; font-weight: 600; }
    .hero h1 { font-size: 2.15rem; font-weight: 800; margin: 6px 0 6px 0; color: #FFFFFF; letter-spacing: -0.6px; line-height: 1.15; }
    .hero p { font-size: .98rem; opacity: .88; margin: 0 0 16px 0; }
    .team { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    .team-label { font-size: .8rem; opacity: .8; margin-right: 4px; font-weight: 600; }
    .chip {
        display: inline-flex; align-items: center; gap: 8px; background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22); padding: 5px 12px 5px 5px; border-radius: 999px;
        font-size: .9rem; font-weight: 600; color: #FFFFFF;
    }
    .avatar {
        width: 26px; height: 26px; border-radius: 50%; background: #F8F7F0; color: #222B52;
        display: inline-flex; align-items: center; justify-content: center; font-size: .72rem; font-weight: 800;
    }

    /* Section + card styling */
    .section-title { font-size: 1.05rem; font-weight: 700; color: #222B52; margin: 4px 0 2px 0; }
    .muted { color: #64748B; font-size: .85rem; }
    .insight {
        background: rgba(79, 70, 229, 0.06); border: 1px solid rgba(79, 70, 229, 0.18); border-radius: 12px;
        padding: 11px 16px; font-size: .93rem; color: #1E293B; margin: 4px 0 14px 0; animation: fadeUp .5s ease both;
    }

    /* KPI cards */
    .kpi-container {
        background: #FFFFFF; border: 1px solid rgba(148, 163, 184, 0.28); border-radius: 14px;
        padding: 14px 16px; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05); height: 100%;
        animation: fadeUp .55s ease both; transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
    }
    .kpi-container:hover { transform: translateY(-3px); border-color: rgba(79,70,229,.45); box-shadow: 0 8px 22px rgba(79,70,229,.12); }
    .kpi-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; gap: 6px; }
    .kpi-label { font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .6px; color: #64748B; }
    .pill { font-size: .68rem; font-weight: 700; padding: 2px 8px; border-radius: 12px; white-space: nowrap; }
    .pill-green { background: rgba(16,185,129,.14); color: #059669; }
    .pill-blue { background: rgba(59,130,246,.14); color: #2563EB; }
    .pill-amber { background: rgba(245,158,11,.16); color: #B45309; }
    .pill-rose { background: rgba(239,68,68,.14); color: #DC2626; }
    .kpi-value { font-size: 1.6rem; font-weight: 800; color: #0F172A; line-height: 1.2; margin-bottom: 2px; }
    .kpi-footer { font-size: .75rem; color: #64748B; }

    /* Product spotlight */
    .product-showcase {
        background: #FFFFFF; border: 1px solid rgba(148,163,184,.28); border-radius: 14px; padding: 16px 20px;
        box-shadow: 0 2px 10px rgba(15,23,42,.05); animation: fadeUp .6s ease both;
    }
    .product-badge { background: rgba(79,70,229,.12); color: #4F46E5; font-size: .72rem; font-weight: 700;
        padding: 3px 9px; border-radius: 6px; display: inline-block; margin-bottom: 8px; }
    .stat-box { background: #F1F5F9; padding: 8px 12px; border-radius: 8px; }
    .stat-label { font-size: .7rem; color: #64748B; font-weight: 600; }

    /* Charts fade in */
    [data-testid="stPlotlyChart"] { animation: fadeUp .7s ease both; }

    /* Tabs */
    hr { margin-top: 1rem !important; margin-bottom: .4rem !important; opacity: .25 !important; }
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important; width: 100% !important; gap: 4px !important;
        border-bottom: 1px solid rgba(148,163,184,.3) !important; margin-bottom: 1.1rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1 1 0px !important; justify-content: center !important; padding: 12px 6px !important;
        font-size: .9rem !important; font-weight: 600 !important; white-space: nowrap !important; border-radius: 8px 8px 0 0 !important;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: rgba(79,70,229,.06) !important; }
    .stTabs [aria-selected="true"] { color: #4F46E5 !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #4F46E5 !important; height: 3px !important; }

    /* Footer */
    .footer { text-align: center; color: #64748B; font-size: .85rem; padding: 10px 0 4px 0; }
    .footer b { color: #222B52; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading dataset...")
def load_dataset(mode="fast"):
    """
    Load dataset from CSV converted from online_retail_II.xlsx
    Supports fast sampling mode (100k) or full dataset (1M+) with memory caching.
    """
    if mode == "fast":
        candidates = ["ecommerce_sample.csv"]
    else:
        candidates = ["ecommerce_data_cleaned.csv", "ecommerce_data_cleaned.csv.gz"]
    file_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    df = pd.read_csv(file_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df

chips = "".join(
    f'<span class="chip"><span class="avatar">{"".join(w[0] for w in name.split())}</span>{name}</span>'
    for name in TEAM
)
st.markdown(f"""
<div class="hero">
    <div class="eyebrow">DAV Lab Mini-Project &nbsp;·&nbsp; Group Project</div>
    <h1>E-Commerce Sales &amp; Customer Behavior Analytics</h1>
    <p>1M+ online retail transactions (Dec 2009 – Dec 2011) &nbsp;·&nbsp; Revenue · Categories · Locations · Delivery · Reviews · Trends · Forecasting</p>
    <div class="team"><span class="team-label">Group Members:</span>{chips}</div>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    head_l, head_r = st.columns([3, 2])
    with head_l:
        st.markdown('<div class="section-title">Project Controls & Dataset Filters</div>'
                    '<div class="muted">Leave a filter empty to include everything. Charts update instantly.</div>',
                    unsafe_allow_html=True)
    with head_r:
        mode_label = st.radio("Dataset mode", ["Fast (100k sample)", "Full (1M+ rows)"], horizontal=True,
                              help="Fast mode uses a representative 100k-row sample for instant interaction.")
    data_mode = "fast" if mode_label.startswith("Fast") else "full"
    df_raw = load_dataset(mode=data_mode)

    years_available = sorted(df_raw["Year"].unique().tolist())
    min_date = df_raw["Date"].min().date()
    max_date = df_raw["Date"].max().date()
    top_countries = df_raw["Country"].value_counts().head(15).index.tolist()
    categories = sorted(df_raw["Product_Category"].unique().tolist())
    segments = sorted(df_raw["Customer_Segment"].dropna().unique().tolist())

    r1_col1, r1_col2, r1_col3 = st.columns([1.0, 1.2, 1.4])
    with r1_col1:
        selected_years = st.multiselect("Year(s)", years_available, default=[], placeholder="All years")
    with r1_col2:
        date_range = st.date_input("Invoice date range", value=(), min_value=min_date, max_value=max_date)
    with r1_col3:
        selected_countries = st.multiselect("Customer country", top_countries, default=[], placeholder="All countries")

    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        selected_categories = st.multiselect("Product category", categories, default=[], placeholder="All categories")
    with r2_col2:
        selected_segments = st.multiselect("Customer RFM segment", segments, default=[], placeholder="All segments")

df_filtered = df_raw

if selected_years:
    df_filtered = df_filtered[df_filtered["Year"].isin(selected_years)]

has_date_filter = bool(date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2)
if has_date_filter:
    start_d, end_d = date_range
    df_filtered = df_filtered[(df_filtered["Date"].dt.date >= start_d) & (df_filtered["Date"].dt.date <= end_d)]

if selected_categories:
    df_filtered = df_filtered[df_filtered["Product_Category"].isin(selected_categories)]

if selected_countries:
    df_filtered = df_filtered[df_filtered["Country"].isin(selected_countries)]

if selected_segments:
    df_filtered = df_filtered[df_filtered["Customer_Segment"].isin(selected_segments)]

if df_filtered.empty:
    st.warning("No transactions match the selected filters. Please widen your selection.")
    st.stop()

df_valid = df_filtered[~df_filtered["Is_Cancelled"]]

total_rev = df_valid["Total_Amount"].sum()
total_txns = len(df_filtered)
total_orders = df_valid["Invoice"].nunique()
unique_customers = df_valid["Customer ID"].nunique()
n_countries = df_valid["Country"].nunique()
avg_order_value = total_rev / total_orders if total_orders > 0 else 0
ontime_orders = (df_filtered["Delivery_Status"] == "On-Time").sum()
ontime_rate = (ontime_orders / total_txns * 100) if total_txns > 0 else 0
cancel_rate = df_filtered["Is_Cancelled"].mean() * 100
avg_review = df_filtered["Review_Rating"].mean()

rev_by_year = df_valid.groupby("Year")["Total_Amount"].sum()
if 2010 in rev_by_year.index and 2011 in rev_by_year.index and rev_by_year[2010] > 0:
    yoy = (rev_by_year[2011] - rev_by_year[2010]) / rev_by_year[2010] * 100
    rev_pill = (f"{yoy:+.1f}% vs 2010", "pill-green" if yoy >= 0 else "pill-rose")
else:
    rev_pill = ("Selected period", "pill-blue")

kpis = [
    ("Total Revenue", f"£{total_rev:,.0f}", rev_pill, "Completed sales; 2011 data ends 9 Dec"),
    ("Total Orders", f"{total_orders:,}", (f"{cancel_rate:.1f}% cancelled", "pill-amber"), "Unique invoices"),
    ("Customers", f"{unique_customers:,}", (f"{n_countries} countries", "pill-blue"), "Distinct registered buyers"),
    ("Avg Order Value", f"£{avg_order_value:,.2f}", ("per invoice", "pill-blue"), "Revenue ÷ orders"),
    ("On-Time Rate", f"{ontime_rate:.1f}%", ("On target" if ontime_rate >= 70 else "Below target",
                                             "pill-green" if ontime_rate >= 70 else "pill-rose"), "Delivered within SLA"),
    ("Avg Review", f"{avg_review:.2f} ★", ("out of 5", "pill-amber"), "Customer satisfaction"),
]
for i, (col, (label, value, (pill_text, pill_cls), foot)) in enumerate(zip(st.columns(6), kpis)):
    with col:
        st.markdown(f"""
        <div class="kpi-container" style="animation-delay: {i * 0.08:.2f}s;">
            <div class="kpi-header">
                <span class="kpi-label">{label}</span>
                <span class="pill {pill_cls}">{pill_text}</span>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-footer">{foot}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f'<div class="muted" style="margin-top:10px;">Showing <b>{total_txns:,}</b> transaction lines '
            f'({"100k sample" if data_mode == "fast" else "full dataset"}). Press <b>▶ Play</b> on animated charts to watch the story unfold.</div>',
            unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Revenue",
    "🛍️ Categories",
    "🌍 Locations",
    "🚚 Delivery",
    "⭐ Reviews",
    "🕒 Trends",
    "🤖 Forecasting & ML"
])

with tab1:
    monthly_rev = df_valid.groupby("Year_Month")["Total_Amount"].sum().reset_index()
    if not monthly_rev.empty:
        peak = monthly_rev.loc[monthly_rev["Total_Amount"].idxmax()]
        insight(f"Revenue peaks in <b>{peak['Year_Month']}</b> at <b>£{peak['Total_Amount']:,.0f}</b>. "
                f"November is the strongest month each year, so Q4 stock planning matters most.")

    col_rev1, col_rev2 = st.columns([6.8, 3.2])

    with col_rev1:
        fig_monthly = reveal_line(monthly_rev["Year_Month"], monthly_rev["Total_Amount"],
                                  "Monthly Sales Trend (animated build-up)", "Month", "Revenue (£)", BLUE)
        fig_monthly.update_layout(height=390)
        show(fig_monthly)

    with col_rev2:
        top_single_item = df_valid.groupby(["Description", "StockCode", "Product_Category"]).agg(
            Revenue=("Total_Amount", "sum"),
            Units=("Quantity", "sum"),
            Avg_Price=("Price", "mean")
        ).reset_index().sort_values(by="Revenue", ascending=False).iloc[0]
        share = top_single_item["Revenue"] / total_rev * 100 if total_rev else 0

        st.markdown(f"""
        <div class="product-showcase">
            <span class="product-badge">Top Best Seller</span>
            <h4 style="margin: 0 0 6px 0; color: {NAVY}; font-size: 1.05rem;">{top_single_item['Description']}</h4>
            <div class="muted" style="margin-bottom: 12px;">
                Category: <b>{top_single_item['Product_Category']}</b> &nbsp;|&nbsp; SKU: <b>{top_single_item['StockCode']}</b>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px;">
                <div class="stat-box"><div class="stat-label">GROSS SALES</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {GREEN};">£{top_single_item['Revenue']:,.0f}</div></div>
                <div class="stat-box"><div class="stat-label">UNITS SOLD</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {BLUE};">{top_single_item['Units']:,}</div></div>
            </div>
            <div class="muted">
                Avg unit price: <b>£{top_single_item['Avg_Price']:.2f}</b><br>
                Share of total revenue: <b>{share:.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col_rev3, col_rev4 = st.columns(2)

    with col_rev3:
        fig_hist = px.histogram(
            df_valid[df_valid["Total_Amount"].between(1, 150)],
            x="Total_Amount",
            nbins=30,
            title="Distribution of Transaction Values (£1 to £150)",
            labels={"Total_Amount": "Order Amount (£)"},
            color_discrete_sequence=[VIOLET]
        )
        fig_hist.update_layout(height=350, yaxis_title="Order Frequency", bargap=0.05)
        show(fig_hist)

    with col_rev4:
        top_products = df_valid.groupby("Description")["Total_Amount"].sum().nlargest(10).reset_index()
        fig_top_prod = px.bar(
            top_products,
            x="Total_Amount",
            y="Description",
            orientation="h",
            title="Top 10 Best-Selling Items by Revenue",
            labels={"Total_Amount": "Total Sales (£)", "Description": ""},
            color="Total_Amount",
            color_continuous_scale="Blues"
        )
        fig_top_prod.update_layout(yaxis=dict(autorange="reversed"), height=350, coloraxis_showscale=False)
        show(fig_top_prod)

with tab2:
    cat_summary = df_valid.groupby("Product_Category").agg(
        Revenue=("Total_Amount", "sum"),
        Quantity=("Quantity", "sum"),
        Avg_Price=("Price", "mean"),
        Orders=("Invoice", "nunique")
    ).reset_index().sort_values(by="Revenue", ascending=False).reset_index(drop=True)

    total_cat_rev = cat_summary["Revenue"].sum()
    cat_summary["Pct"] = (cat_summary["Revenue"] / total_cat_rev * 100).round(1)
    cat_colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cat_summary["Product_Category"])}

    named = cat_summary[cat_summary["Product_Category"] != "Other General Retail"]
    if not named.empty:
        insight(f"<b>{named.iloc[0]['Product_Category']}</b> is the largest named category with "
                f"<b>{named.iloc[0]['Pct']}%</b> of revenue. Press ▶ Play on the race chart to watch categories grow month by month.")

    segments_html = '<div style="display: flex; height: 16px; border-radius: 8px; overflow: hidden; margin-bottom: 12px;">'
    pills_html = '<div style="display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 14px;">'
    for _, row in cat_summary.iterrows():
        color = cat_colors[row["Product_Category"]]
        segments_html += f'<div style="width: {row["Pct"]}%; background-color: {color};" title="{row["Product_Category"]}: {row["Pct"]}%"></div>'
        pills_html += (f'<div style="display: flex; align-items: center; gap: 6px; font-size: .82rem;">'
                       f'<span style="width: 10px; height: 10px; border-radius: 50%; background-color: {color}; display:inline-block;"></span>'
                       f'<b>{row["Pct"]}%</b> <span style="color: #64748B;">{row["Product_Category"]}</span></div>')
    st.markdown(segments_html + '</div>' + pills_html + '</div>', unsafe_allow_html=True)

    col_cat1, col_cat2 = st.columns(2)

    with col_cat1:
        fig_donut = px.pie(
            cat_summary,
            names="Product_Category",
            values="Revenue",
            hole=0.55,
            title="Revenue Share by Product Category",
            color="Product_Category",
            color_discrete_map=cat_colors
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent", sort=False)
        fig_donut.update_layout(height=400)
        show(fig_donut)

    with col_cat2:
        race = (df_valid.groupby(["Year_Month", "Product_Category"])["Total_Amount"].sum()
                .unstack(fill_value=0).sort_index().cumsum()
                .stack().rename("Cumulative_Revenue").reset_index())
        cat_order = cat_summary["Product_Category"].tolist()
        fig_race = px.bar(
            race,
            x="Cumulative_Revenue",
            y="Product_Category",
            orientation="h",
            animation_frame="Year_Month",
            color="Product_Category",
            color_discrete_map=cat_colors,
            category_orders={"Product_Category": cat_order},
            range_x=[0, race["Cumulative_Revenue"].max() * 1.1],
            title="Category Revenue Race (cumulative, animated)",
            labels={"Cumulative_Revenue": "Cumulative Revenue (£)", "Product_Category": "", "Year_Month": "Month"}
        )
        fig_race.update_layout(height=400, showlegend=False)
        add_play_controls(fig_race, frame_ms=380, transition_ms=300)
        show(start_at_end(fig_race))

    st.markdown('<div class="section-title">Category Metrics Breakdown</div>', unsafe_allow_html=True)
    cat_display = cat_summary[["Product_Category", "Revenue", "Pct", "Quantity", "Avg_Price", "Orders"]].copy()
    cat_display.columns = ["Category", "Revenue", "Share %", "Units", "Avg Price", "Orders"]
    cat_display["Revenue"] = cat_display["Revenue"].map("£{:,.2f}".format)
    cat_display["Units"] = cat_display["Units"].map("{:,}".format)
    cat_display["Avg Price"] = cat_display["Avg Price"].map("£{:.2f}".format)
    cat_display["Orders"] = cat_display["Orders"].map("{:,}".format)
    show_table(cat_display)

with tab3:
    geo_df = df_valid.groupby("Country").agg(
        Revenue=("Total_Amount", "sum"),
        Orders=("Invoice", "nunique"),
        Customers=("Customer ID", "nunique")
    ).reset_index()
    geo_df["Avg_Order_Value"] = geo_df["Revenue"] / geo_df["Orders"].clip(lower=1)

    uk_rev = geo_df.loc[geo_df["Country"] == "United Kingdom", "Revenue"].sum()
    intl_df = geo_df[geo_df["Country"] != "United Kingdom"].sort_values(by="Revenue", ascending=False)
    if total_rev and not intl_df.empty:
        insight(f"The UK brings <b>{uk_rev / total_rev * 100:.1f}%</b> of revenue. "
                f"<b>{intl_df.iloc[0]['Country']}</b> is the biggest export market. Press ▶ Play to watch the map spread over time.")

    geo_time = (df_valid.groupby(["Year_Month", "Country"])["Total_Amount"].sum()
                .unstack(fill_value=0).sort_index().cumsum()
                .stack().rename("Revenue").reset_index())
    geo_time = geo_time[geo_time["Revenue"] > 0]
    geo_time["Log_Revenue"] = np.log10(geo_time["Revenue"])
    tick_vals = [v for v in range(1, 9) if geo_time["Log_Revenue"].min() - 1 <= v <= geo_time["Log_Revenue"].max() + 1]
    fig_map = px.choropleth(
        geo_time,
        locations="Country",
        locationmode="country names",
        color="Log_Revenue",
        animation_frame="Year_Month",
        hover_name="Country",
        hover_data={"Revenue": ":,.0f", "Log_Revenue": False, "Year_Month": False},
        range_color=[geo_time["Log_Revenue"].min(), geo_time["Log_Revenue"].max()],
        color_continuous_scale="Plasma",
        title="Global Revenue Footprint Over Time (cumulative, animated)",
        labels={"Year_Month": "Month"}
    )
    fig_map.update_layout(height=470, margin=dict(r=0, t=60, l=0, b=0),
                          geo=dict(showframe=False, projection_type="natural earth", bgcolor="rgba(0,0,0,0)"),
                          coloraxis_colorbar=dict(title="Revenue", tickvals=tick_vals,
                                                  ticktext=[f"£{10 ** v:,.0f}" for v in tick_vals]))
    add_play_controls(fig_map, frame_ms=450, transition_ms=0, redraw=True)
    show(start_at_end(fig_map))

    col_geo1, col_geo2 = st.columns(2)

    with col_geo1:
        fig_intl = px.bar(
            intl_df.head(12),
            x="Revenue",
            y="Country",
            orientation="h",
            title="Top 12 International Markets (excl. UK)",
            labels={"Revenue": "Revenue (£)", "Country": ""},
            color="Revenue",
            color_continuous_scale="Teal"
        )
        fig_intl.update_layout(yaxis=dict(autorange="reversed"), height=360, coloraxis_showscale=False)
        show(fig_intl)

    with col_geo2:
        dom_df = pd.DataFrame({
            "Market": ["United Kingdom (Domestic)", "International (Export)"],
            "Revenue": [uk_rev, intl_df["Revenue"].sum()]
        })
        fig_dom = px.pie(
            dom_df,
            names="Market",
            values="Revenue",
            title="Domestic vs International Revenue",
            color_discrete_sequence=[BLUE, AMBER],
            hole=0.55
        )
        fig_dom.update_layout(height=360)
        show(fig_dom)

with tab4:
    shipped = df_filtered[~df_filtered["Is_Cancelled"]]
    insight(f"<b>{ontime_rate:.1f}%</b> of orders arrive on time, with an average transit of "
            f"<b>{shipped['Delivery_Days'].mean():.1f} days</b>. International orders take the longest. "
            f"<span class='muted'>(Delivery times are simulated in the data pipeline.)</span>")

    col_del1, col_del2 = st.columns(2)

    with col_del1:
        status_df = df_filtered["Delivery_Status"].value_counts().reset_index()
        status_df.columns = ["Status", "Count"]
        fig_del_donut = px.pie(
            status_df,
            names="Status",
            values="Count",
            hole=0.62,
            title="Delivery Status Breakdown",
            color="Status",
            color_discrete_map=STATUS_COLORS
        )
        fig_del_donut.update_traces(textposition="inside", textinfo="percent")
        fig_del_donut.update_layout(
            annotations=[dict(text=f"<b>{ontime_rate:.0f}%</b><br>on time", x=0.5, y=0.5, font_size=18, showarrow=False)],
            height=370
        )
        show(fig_del_donut)

    with col_del2:
        fig_del_hist = px.histogram(
            shipped,
            x="Delivery_Days",
            color="Delivery_Status",
            nbins=15,
            title="Distribution of Transit Duration (Days)",
            labels={"Delivery_Days": "Transit Days", "Delivery_Status": "Status"},
            color_discrete_map=STATUS_COLORS
        )
        fig_del_hist.update_layout(barmode="stack", height=370, yaxis_title="Shipments", bargap=0.05)
        show(fig_del_hist)

    top_ship_countries = df_filtered["Country"].value_counts().head(6).index.tolist()
    box_del_df = shipped[shipped["Country"].isin(top_ship_countries)]
    fig_del_box = px.box(
        box_del_df,
        x="Country",
        y="Delivery_Days",
        color="Country",
        title="Delivery Duration Spread Across Major Destination Countries",
        labels={"Delivery_Days": "Delivery Duration (Days)", "Country": ""}
    )
    fig_del_box.update_layout(height=350, showlegend=False)
    show(fig_del_box)

with tab5:
    rating_by_status = df_filtered.groupby("Delivery_Status")["Review_Rating"].mean()
    if "On-Time" in rating_by_status.index:
        delayed = rating_by_status.drop(labels=["On-Time", "Cancelled"], errors="ignore")
        msg = f"On-time orders average <b>{rating_by_status['On-Time']:.2f} ★</b>"
        if not delayed.empty:
            msg += f" while delayed orders average <b>{df_filtered[df_filtered['Delivery_Status'].isin(delayed.index)]['Review_Rating'].mean():.2f} ★</b>"
        insight(msg + ". Faster delivery means happier customers. <span class='muted'>(Ratings are simulated in the data pipeline.)</span>")

    col_rev_a, col_rev_b = st.columns(2)

    with col_rev_a:
        rating_dist = df_filtered["Review_Rating"].value_counts().sort_index(ascending=False).reset_index()
        rating_dist.columns = ["Rating", "Count"]
        rating_dist["Rating_Label"] = rating_dist["Rating"].apply(lambda x: f"{x} ★")
        rating_dist["Percentage"] = (rating_dist["Count"] / rating_dist["Count"].sum() * 100).round(1)
        fig_rating_bar = px.bar(
            rating_dist,
            x="Percentage",
            y="Rating_Label",
            orientation="h",
            text="Percentage",
            title="Review Rating Breakdown (% of orders)",
            labels={"Percentage": "Percentage (%)", "Rating_Label": ""},
            color="Rating",
            color_continuous_scale="Emrld"
        )
        fig_rating_bar.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
        fig_rating_bar.update_layout(yaxis=dict(autorange="reversed"), height=360, coloraxis_showscale=False)
        show(fig_rating_bar)

    with col_rev_b:
        rev_cat_df = df_filtered["Review_Category"].value_counts().reset_index()
        rev_cat_df.columns = ["Category", "Count"]
        fig_rev_cat = px.pie(
            rev_cat_df,
            names="Category",
            values="Count",
            hole=0.55,
            title="Review Classification Share (np.select, DAV Practical 8)",
            color="Category",
            color_discrete_map={"Excellent": GREEN, "Good": BLUE, "Average": AMBER, "Poor / Needs Improvement": ROSE}
        )
        fig_rev_cat.update_layout(height=360)
        show(fig_rev_cat)

    st.markdown('<div class="section-title">Punctuality vs Satisfaction</div>', unsafe_allow_html=True)
    review_pool = df_filtered[~df_filtered["Is_Cancelled"]]
    sample_reviews = review_pool.sample(n=min(1500, len(review_pool)), random_state=42)
    rng = np.random.default_rng(42)
    sample_reviews = sample_reviews.assign(
        Days_Jitter=sample_reviews["Delivery_Days"] + rng.uniform(-0.25, 0.25, len(sample_reviews)),
        Rating_Jitter=sample_reviews["Review_Rating"] + rng.uniform(-0.18, 0.18, len(sample_reviews)),
    )
    fig_scatter_rev = px.scatter(
        sample_reviews,
        x="Days_Jitter",
        y="Rating_Jitter",
        color="Delivery_Status",
        title="Delivery Duration vs Customer Rating",
        labels={"Days_Jitter": "Delivery Time (Days)", "Rating_Jitter": "Customer Rating (1-5)", "Delivery_Status": "Status"},
        color_discrete_map=STATUS_COLORS,
        opacity=0.55,
        hover_data={"Delivery_Days": True, "Review_Rating": True, "Days_Jitter": False, "Rating_Jitter": False}
    )
    fig_scatter_rev.update_layout(height=370)
    show(fig_scatter_rev)

with tab6:
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hourly_df = df_valid.groupby("Hour")["Invoice"].nunique().reset_index()
    dow_df = df_valid.groupby("DayOfWeek")["Total_Amount"].sum().reindex(days_order).dropna().reset_index()
    if not hourly_df.empty and not dow_df.empty:
        insight(f"Orders peak at <b>{int(hourly_df.loc[hourly_df['Invoice'].idxmax(), 'Hour'])}:00</b> and "
                f"<b>{dow_df.loc[dow_df['Total_Amount'].idxmax(), 'DayOfWeek']}</b> is the best sales day. "
                f"Press ▶ Play to step through each weekday's hourly pattern.")

    col_trend1, col_trend2 = st.columns(2)

    with col_trend1:
        hour_dow = df_valid.groupby(["DayOfWeek", "Hour"])["Invoice"].nunique().reset_index(name="Orders")
        hour_dow = (hour_dow.set_index(["DayOfWeek", "Hour"])
                    .reindex(pd.MultiIndex.from_product([[d for d in days_order if d in hour_dow["DayOfWeek"].unique()],
                                                         sorted(df_valid["Hour"].unique())], names=["DayOfWeek", "Hour"]),
                             fill_value=0).reset_index())
        fig_hour = px.bar(
            hour_dow,
            x="Hour",
            y="Orders",
            animation_frame="DayOfWeek",
            category_orders={"DayOfWeek": days_order},
            color="Orders",
            color_continuous_scale="Purples",
            range_y=[0, hour_dow["Orders"].max() * 1.15],
            range_color=[0, hour_dow["Orders"].max()],
            title="Hourly Order Velocity by Weekday (animated)",
            labels={"Hour": "Hour of Day (24h)", "Orders": "Distinct Orders", "DayOfWeek": "Day"}
        )
        fig_hour.update_layout(height=380, coloraxis_showscale=False, bargap=0.15)
        add_play_controls(fig_hour, frame_ms=700, transition_ms=450)
        show(fig_hour)

    with col_trend2:
        fig_dow = px.bar(
            dow_df,
            x="DayOfWeek",
            y="Total_Amount",
            title="Total Sales Revenue by Day of the Week",
            labels={"DayOfWeek": "", "Total_Amount": "Revenue (£)"},
            color="Total_Amount",
            color_continuous_scale="Purples"
        )
        fig_dow.update_layout(height=380, coloraxis_showscale=False)
        show(fig_dow)

    st.markdown('<div class="section-title">Customer RFM Behavioural Segmentation (Recency, Frequency, Monetary)</div>', unsafe_allow_html=True)
    rfm_counts = df_valid.groupby("Customer_Segment")["Customer ID"].nunique().reset_index()
    rfm_counts.columns = ["Segment", "Customer_Count"]
    fig_rfm = px.bar(
        rfm_counts.sort_values(by="Customer_Count", ascending=False),
        x="Customer_Count",
        y="Segment",
        orientation="h",
        text="Customer_Count",
        title="Customers per Behavioural Segment",
        labels={"Customer_Count": "Number of Buyers", "Segment": ""},
        color="Customer_Count",
        color_continuous_scale="Viridis"
    )
    fig_rfm.update_traces(textposition="outside", cliponaxis=False)
    fig_rfm.update_layout(yaxis=dict(autorange="reversed"), height=330, coloraxis_showscale=False)
    show(fig_rfm)

with tab7:
    st.caption("Implemented via **DAV Practical 6 (ARIMA & smoothing)** and **DAV Practical 3 (Linear Regression)**.")

    daily_ts = df_valid.groupby("Date")["Total_Amount"].sum()
    daily_ts.index = pd.to_datetime(daily_ts.index)
    daily_ts = daily_ts.asfreq("D").fillna(0)

    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        sma = daily_ts.rolling(window=7).mean()
        ema = daily_ts.ewm(span=7, adjust=False).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=daily_ts.index, y=daily_ts.values, mode="lines", name="Actual Daily", line=dict(color="#94A3B8", width=1), opacity=0.45))
        fig_ma.add_trace(go.Scatter(x=sma.index, y=sma.values, mode="lines", name="7-Day SMA", line=dict(color=BLUE, width=2.5)))
        fig_ma.add_trace(go.Scatter(x=ema.index, y=ema.values, mode="lines", name="7-Day EMA", line=dict(color=ROSE, width=2.2, dash="dash")))
        fig_ma.update_layout(title="Revenue Smoothing: 7-Day SMA & EMA", yaxis_title="Revenue (£)", height=400,
                             legend=dict(orientation="h", y=-0.2))
        show(fig_ma)

    with col_ml2:
        forecast_days = st.slider("ARIMA forecast horizon (days)", 7, 30, 14)
        if len(daily_ts) < 30:
            st.info("Select a wider date range (at least 30 days) to fit the ARIMA model.")
        else:
            try:
                res = ARIMA(daily_ts, order=(1, 1, 1)).fit()
                forecast = res.get_forecast(steps=forecast_days)
                forecast_mean = forecast.predicted_mean
                conf_int = forecast.conf_int()
                recent_actual = daily_ts.tail(45)

                fx = [recent_actual.index[-1]] + list(forecast_mean.index)
                fy = [recent_actual.values[-1]] + list(forecast_mean.values)
                lo = [recent_actual.values[-1]] + list(conf_int.iloc[:, 0].values)
                hi = [recent_actual.values[-1]] + list(conf_int.iloc[:, 1].values)

                fig_arima = go.Figure(
                    data=[
                        go.Scatter(x=recent_actual.index, y=recent_actual.values, mode="lines+markers", name="Recent Actual",
                                   line=dict(color=BLUE, width=2), marker=dict(size=5)),
                        go.Scatter(x=fx, y=lo, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
                        go.Scatter(x=fx, y=hi, mode="lines", fill="tonexty", fillcolor="rgba(239,68,68,0.18)",
                                   line=dict(width=0), name="95% Confidence"),
                        go.Scatter(x=fx, y=fy, mode="lines+markers", name=f"{forecast_days}-Day Forecast",
                                   line=dict(color=ROSE, width=2.5, dash="dash"), marker=dict(size=6)),
                    ],
                    frames=[go.Frame(data=[go.Scatter(x=fx[:k], y=lo[:k]), go.Scatter(x=fx[:k], y=hi[:k]),
                                           go.Scatter(x=fx[:k], y=fy[:k])], traces=[1, 2, 3], name=str(k))
                            for k in range(1, len(fx) + 1)],
                )
                y_top = max(recent_actual.max(), max(hi)) * 1.1
                fig_arima.update_layout(title=f"ARIMA(1,1,1) {forecast_days}-Day Revenue Forecast (animated)",
                                        yaxis_title="Revenue (£)", height=400,
                                        xaxis=dict(range=[recent_actual.index[0], fx[-1]]),
                                        yaxis=dict(range=[min(0, min(lo)), y_top]),
                                        legend=dict(orientation="h", y=-0.2))
                add_play_controls(fig_arima, frame_ms=160, transition_ms=0, redraw=True)
                show(fig_arima)

                recent_avg = daily_ts.tail(forecast_days).mean()
                insight(f"Forecast average: <b>£{forecast_mean.mean():,.0f}/day</b> for the next {forecast_days} days, "
                        f"vs <b>£{recent_avg:,.0f}/day</b> over the last {forecast_days} days.")
            except Exception as e:
                st.warning(f"ARIMA forecasting status: {e}")

    col_ml3, col_ml4 = st.columns(2)

    with col_ml3:
        num_cols = ["Quantity", "Price", "Total_Amount", "Delivery_Days", "Review_Rating"]
        corr_matrix = df_filtered[num_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation Matrix Heatmap (DAV Practicals 3 & 7)"
        )
        fig_corr.update_layout(height=400)
        show(fig_corr)

    with col_ml4:
        st.markdown('<div class="section-title">Linear Regression Performance (DAV Practical 3)</div>', unsafe_allow_html=True)
        reg_df = df_valid[(df_valid["Quantity"] <= df_valid["Quantity"].quantile(0.99))
                          & (df_valid["Price"] <= df_valid["Price"].quantile(0.99))]
        if len(reg_df) < 20:
            st.info("Not enough rows in the current filter to train the regression model.")
        else:
            X = reg_df[["Quantity", "Price", "Delivery_Days"]].fillna(0)
            y = reg_df["Total_Amount"].fillna(0)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            lr = LinearRegression()
            lr.fit(X_tr, y_tr)
            y_preds = lr.predict(X_te)
            r2 = r2_score(y_te, y_preds)
            mse_val = mean_squared_error(y_te, y_preds)

            st.markdown("**Target (y):** `Total_Amount` (£) &nbsp;|&nbsp; **Features (X):** `Quantity`, `Price`, `Delivery_Days`")
            st.caption(f"Trained on {len(X_tr):,} completed sales (80/20 split); cancelled lines and top-1% quantity/price outliers removed.")
            m1, m2 = st.columns(2)
            m1.metric("Model R² Score", f"{r2:.4f}")
            m2.metric("Mean Squared Error", f"{mse_val:,.2f}")

            pred_sample = pd.DataFrame({"Actual (£)": y_te.values[:6], "Predicted (£)": y_preds[:6]}).round(2)
            st.caption("Sample actual vs predicted values:")
            show_table(pred_sample)

st.divider()
st.markdown(f"""
<div class="footer">
    DAV Lab Mini-Project &nbsp;·&nbsp; Group Members: <b>{TEAM[0]}</b> · <b>{TEAM[1]}</b> · <b>{TEAM[2]}</b><br>
    Built with Python, Pandas, Plotly &amp; Streamlit
</div>
""", unsafe_allow_html=True)
