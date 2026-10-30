
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
np.random.seed(42)

print("=" * 80)
print("E-COMMERCE SALES AND CUSTOMER BEHAVIOR ANALYTICS")
print("DAV Lab Mini Project - Soham Admane, Vaishnav Balpande, Lokesh Parate")
print("=" * 80)

print("\n--- SECTION 1: DATA INGESTION & OVERVIEW ---")
df = pd.read_csv("ecommerce_sample.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 Records (df.head()):")
print(df.head())

print("\nLast 5 Records (df.tail()):")
print(df.tail())

print("\nData Columns:")
print(df.columns.tolist())

print("\nData Types (df.dtypes):")
print(df.dtypes)

print("\nMissing Values Count (df.isnull().sum()):")
print(df.isnull().sum())

print("\nStatistical Summary (df.describe()):")
print(df.describe())

print("\n--- SECTION 2: REVENUE ANALYTICS ---")
valid_sales = df[~df["Is_Cancelled"]]

total_revenue = valid_sales["Total_Amount"].sum()
total_orders = valid_sales["Invoice"].nunique()
avg_order_val = total_revenue / total_orders
print(f"Total Gross Revenue: £{total_revenue:,.2f}")
print(f"Total Completed Orders: {total_orders:,}")
print(f"Average Order Value (AOV): £{avg_order_val:.2f}")

monthly_sales = valid_sales.groupby("Year_Month")["Total_Amount"].sum().reset_index()
print("\nMonthly Sales Trend Data:")
print(monthly_sales.head(10))

max_rev = monthly_sales["Total_Amount"].max()
max_month = monthly_sales.loc[monthly_sales["Total_Amount"].idxmax(), "Year_Month"]

plt.figure(figsize=(14, 5))
plt.plot(monthly_sales["Year_Month"], monthly_sales["Total_Amount"], marker="o", color="blue", linewidth=2)
plt.annotate(
    f"Peak: £{max_rev:,.0f}",
    xy=(max_month, max_rev),
    xytext=(-60, -35),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="red", lw=1.5)
)
plt.title("Monthly Revenue Trend with Peak Highlighted (DAV Practical 7 Style)", fontsize=14)
plt.xlabel("Month (Year-Month)", fontsize=12)
plt.ylabel("Revenue (£)", fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("fig1_monthly_revenue_trend.png")
plt.close()
print("Saved fig1_monthly_revenue_trend.png")

plt.figure(figsize=(10, 5))
sns.histplot(valid_sales[valid_sales["Total_Amount"] < 100]["Total_Amount"], bins=30, kde=True, color="purple")
plt.title("Histogram of Transaction Revenue (< £100) with KDE", fontsize=14)
plt.xlabel("Transaction Revenue (£)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.tight_layout()
plt.savefig("fig2_revenue_histogram_kde.png")
plt.close()
print("Saved fig2_revenue_histogram_kde.png")

print("\n--- SECTION 3: PRODUCT CATEGORY ANALYTICS ---")
cat_revenue = valid_sales.groupby("Product_Category")["Total_Amount"].sum().sort_values(ascending=False)
cat_orders = valid_sales.groupby("Product_Category")["Quantity"].sum().sort_values(ascending=False)

print("\nRevenue by Product Category:")
print(cat_revenue)

plt.figure(figsize=(10, 5))
cat_revenue.plot(kind="bar", color="steelblue")
plt.title("Vertical Bar Chart of Revenue by Product Category", fontsize=14)
plt.xlabel("Category", fontsize=12)
plt.ylabel("Revenue (£)", fontsize=12)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("fig3_category_revenue_bar.png")
plt.close()
print("Saved fig3_category_revenue_bar.png")

plt.figure(figsize=(10, 5))
cat_orders.sort_values(ascending=True).plot(kind="barh", color="teal")
plt.title("Horizontal Bar Chart of Quantity Sold by Category", fontsize=14)
plt.xlabel("Quantity Sold", fontsize=12)
plt.ylabel("Category", fontsize=12)
plt.tight_layout()
plt.savefig("fig4_category_quantity_barh.png")
plt.close()
print("Saved fig4_category_quantity_barh.png")

plt.figure(figsize=(8, 8))
cat_revenue.plot(kind="pie", autopct="%1.1f%%", startangle=90, wedgeprops=dict(width=0.4))
plt.title("Donut Chart of Revenue Share by Category (DAV Practical 8 Style)", fontsize=14)
plt.ylabel("")
plt.tight_layout()
plt.savefig("fig5_category_donut_chart.png")
plt.close()
print("Saved fig5_category_donut_chart.png")

print("\n--- SECTION 4: CUSTOMER LOCATIONS ANALYTICS ---")
country_summary = valid_sales.groupby("Country").agg({
    "Total_Amount": "sum",
    "Invoice": "nunique",
    "Customer ID": "nunique"
}).rename(columns={"Total_Amount": "Revenue", "Invoice": "Orders", "Customer ID": "Customers"})
country_summary = country_summary.sort_values(by="Revenue", ascending=False)

print("\nTop 10 Countries by Revenue:")
print(country_summary.head(10))

fig_geo = px.choropleth(
    country_summary.reset_index(),
    locations="Country",
    locationmode="country names",
    color="Revenue",
    hover_name="Country",
    hover_data={"Revenue": ":,.2f", "Orders": ":,", "Customers": ":,"},
    color_continuous_scale="Viridis",
    title="Global Revenue Distribution by Customer Location (Plotly Express)"
)
fig_geo.write_html("fig_geo_choropleth.html")
print("Saved fig_geo_choropleth.html (Interactive Plotly Map)")

print("\n--- SECTION 5: DELIVERY TIMES ANALYTICS ---")
print("Delivery Status Breakdown:")
deliv_counts = df["Delivery_Status"].value_counts()
print(deliv_counts)

top_5_countries = df["Country"].value_counts().head(5).index
filtered_deliv = df[df["Country"].isin(top_5_countries) & (~df["Is_Cancelled"])]

plt.figure(figsize=(10, 5))
sns.boxplot(data=filtered_deliv, x="Country", y="Delivery_Days", palette="Set2")
plt.title("Box-and-Whisker Plot of Delivery Days for Top 5 Countries", fontsize=14)
plt.xlabel("Country", fontsize=12)
plt.ylabel("Delivery Time (Days)", fontsize=12)
plt.tight_layout()
plt.savefig("fig6_delivery_days_boxplot.png")
plt.close()
print("Saved fig6_delivery_days_boxplot.png")

plt.figure(figsize=(7, 7))
deliv_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90, colors=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
plt.title("Pie Chart of Fulfillment Delivery Status", fontsize=14)
plt.ylabel("")
plt.tight_layout()
plt.savefig("fig7_delivery_status_pie.png")
plt.close()
print("Saved fig7_delivery_status_pie.png")

print("\n--- SECTION 6: REVIEWS & CUSTOMER SATISFACTION ---")
rating_counts = df["Review_Rating"].value_counts().sort_index(ascending=False)
print("Review Rating Distribution (1 to 5 Stars):")
print(rating_counts)

plt.figure(figsize=(10, 5))
sample_for_scatter = df[~df["Is_Cancelled"]].sample(n=1000, random_state=42)
sns.scatterplot(
    data=sample_for_scatter,
    x="Delivery_Days",
    y="Review_Rating",
    hue="Delivery_Status",
    palette="viridis",
    alpha=0.6,
    s=70
)
plt.title("Relationship between Delivery Days and Customer Review Rating", fontsize=14)
plt.xlabel("Delivery Time (Days)", fontsize=12)
plt.ylabel("Review Rating (1-5 Stars)", fontsize=12)
plt.tight_layout()
plt.savefig("fig8_delivery_vs_rating_scatter.png")
plt.close()
print("Saved fig8_delivery_vs_rating_scatter.png")

cat_ratings = df.groupby("Product_Category")["Review_Rating"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
cat_ratings.plot(kind="bar", color="coral", ylim=(1, 5))
plt.title("Average Customer Review Rating across Product Categories", fontsize=14)
plt.xlabel("Product Category", fontsize=12)
plt.ylabel("Average Rating (Stars)", fontsize=12)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("fig9_category_ratings_bar.png")
plt.close()
print("Saved fig9_category_ratings_bar.png")

print("\n--- SECTION 7: PURCHASING TRENDS ---")
hourly_orders = valid_sales.groupby("Hour")["Invoice"].nunique()
plt.figure(figsize=(10, 4))
plt.plot(hourly_orders.index, hourly_orders.values, marker="o", color="darkblue", linestyle="-", linewidth=2)
plt.title("Hourly Purchasing Trend (Peak Order Hours)", fontsize=14)
plt.xlabel("Hour of Day (24-Hour Format)", fontsize=12)
plt.ylabel("Number of Unique Invoices", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("fig10_hourly_orders_trend.png")
plt.close()
print("Saved fig10_hourly_orders_trend.png")

days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily_orders = valid_sales.groupby("DayOfWeek")["Total_Amount"].sum().reindex(days_order).dropna()

plt.figure(figsize=(9, 4))
daily_orders.plot(kind="bar", color="mediumseagreen")
plt.title("Total Revenue by Day of Week", fontsize=14)
plt.xlabel("Day of Week", fontsize=12)
plt.ylabel("Revenue (£)", fontsize=12)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("fig11_dayofweek_revenue_bar.png")
plt.close()
print("Saved fig11_dayofweek_revenue_bar.png")

print("\n--- SECTION 8: TIME SERIES DECOMPOSITION & ARIMA FORECASTING ---")
daily_sales = valid_sales.groupby("Date")["Total_Amount"].sum()
daily_sales.index = pd.to_datetime(daily_sales.index)
daily_sales = daily_sales.asfreq("D").fillna(0)

sma_7 = daily_sales.rolling(window=7).mean()
ema_7 = daily_sales.ewm(span=7, adjust=False).mean()

plt.figure(figsize=(14, 5))
plt.plot(daily_sales.index, daily_sales.values, label="Daily Revenue", alpha=0.4, color="gray")
plt.plot(sma_7.index, sma_7.values, label="7-Day SMA", color="blue", linewidth=2)
plt.plot(ema_7.index, ema_7.values, label="7-Day EMA", color="red", linestyle="--", linewidth=2)
plt.title("Daily Revenue Trend with 7-Day SMA and EMA (DAV Practical 6 Style)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Revenue (£)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig12_timeseries_sma_ema.png")
plt.close()
print("Saved fig12_timeseries_sma_ema.png")

try:
    decomp = seasonal_decompose(daily_sales, model="additive", period=30)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    decomp.observed.plot(ax=axes[0], color="black", title="Observed Daily Revenue")
    decomp.trend.plot(ax=axes[1], color="blue", title="Trend Component")
    decomp.seasonal.plot(ax=axes[2], color="green", title="Seasonal Component (30-day Cycle)")
    decomp.resid.plot(ax=axes[3], color="red", title="Residual Irregular Component")
    plt.tight_layout()
    plt.savefig("fig13_seasonal_decompose.png")
    plt.close()
    print("Saved fig13_seasonal_decompose.png")
except Exception as e:
    print(f"Seasonal decomposition note: {e}")

try:
    arima_model = ARIMA(daily_sales, order=(1, 1, 1))
    arima_res = arima_model.fit()
    forecast = arima_res.get_forecast(steps=14)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    plt.figure(figsize=(12, 5))
    plt.plot(daily_sales.tail(60).index, daily_sales.tail(60).values, marker="o", label="Recent Actual Sales", color="blue")
    plt.plot(forecast_mean.index, forecast_mean.values, marker="x", color="red", linestyle="--", label="14-Day ARIMA Forecast")
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="pink", alpha=0.3, label="95% Confidence Interval")
    plt.title("ARIMA(1,1,1) 14-Day Revenue Forecast (DAV Practical 6 Style)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Revenue (£)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig14_arima_forecast.png")
    plt.close()
    print("Saved fig14_arima_forecast.png")
except Exception as e:
    print(f"ARIMA forecast note: {e}")

print("\n--- SECTION 9: CORRELATION MATRIX & LINEAR REGRESSION ---")
numeric_df = df[["Quantity", "Price", "Total_Amount", "Delivery_Days", "Review_Rating"]].copy()
corr = numeric_df.corr()
print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap (DAV Practical 3 & 7 Style)", fontsize=14)
plt.tight_layout()
plt.savefig("fig15_correlation_matrix.png")
plt.close()
print("Saved fig15_correlation_matrix.png")

X = numeric_df[["Quantity", "Price", "Delivery_Days"]].fillna(0)
y = numeric_df["Total_Amount"].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nLinear Regression Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared Score (R2): {r2:.4f}")

pred_comparison = pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10]})
print("\nSample Actual vs Predicted Values:")
print(pred_comparison)

print("\n--- SECTION 10: INTERACTIVE PLOTLY VISUALIZATION ---")
cat_grouped = df[~df["Is_Cancelled"]].groupby("Product_Category").agg({
    "Quantity": "mean",
    "Price": "mean",
    "Review_Rating": "mean"
}).reset_index()

fig_grouped = px.bar(
    cat_grouped,
    x="Product_Category",
    y=["Quantity", "Price", "Review_Rating"],
    barmode="group",
    title="Interactive Grouped Bar Chart of Category Metrics (DAV Practical 8 Style)",
    labels={"value": "Average Metric Value", "variable": "Feature"}
)
fig_grouped.write_html("fig_grouped_interactive.html")
print("Saved fig_grouped_interactive.html (Interactive Plotly Chart)")

print("\n" + "=" * 80)
print("ALL 10 SECTIONS COMPLETED SUCCESSFULLY!")
print("Figures and HTML charts generated matching DAV Practical 1 through 8.")
print("=" * 80)