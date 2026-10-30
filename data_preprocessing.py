import pandas as pd
import numpy as np
import time

print("Starting E-Commerce Data Cleaning and Enrichment Pipeline...")
start_time = time.time()

np.random.seed(42)

print("Reading online_retail_II.csv...")
df = pd.read_csv("online_retail_II.csv")
print(f"Loaded raw dataset with shape: {df.shape}")

print("Cleaning data...")
df["Description"] = df["Description"].astype(str).str.strip().str.upper()
df["StockCode"] = df["StockCode"].astype(str).str.strip()
df["Country"] = df["Country"].astype(str).str.strip()

invalid_desc = ["NAN", "UNSPECIFIED", "?", "DAMAGED", "LOST", "CHECK", "COUNT"]
df = df[~df["Description"].isin(invalid_desc)]

df["Is_Cancelled"] = df["Invoice"].astype(str).str.startswith("C")

df = df[df["Price"] > 0]

df["Total_Amount"] = df["Quantity"] * df["Price"]

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Year"] = df["InvoiceDate"].dt.year
df["Month"] = df["InvoiceDate"].dt.month
df["Month_Name"] = df["InvoiceDate"].dt.strftime("%B")
df["Year_Month"] = df["InvoiceDate"].dt.strftime("%Y-%m")
df["DayOfWeek"] = df["InvoiceDate"].dt.strftime("%A")
df["Hour"] = df["InvoiceDate"].dt.hour
df["Date"] = df["InvoiceDate"].dt.date

print("Assigning product categories using np.select...")
desc = df["Description"]

cond_home = desc.str.contains(r"HEART|T-LIGHT|CANDLE|HOLDER|DOORMAT|MAT|CLOCK|FRAME|CUSHION|MIRROR|LIGHT|LANTERN|VASE|WALL", regex=True, na=False)
cond_kitchen = desc.str.contains(r"MUG|CUP|BOWL|PLATE|CUTTERS|CAKE|BOTTLE|TEAPOT|SPOON|FORK|GLASS|DISH|JAR|TRAY|COASTER|LUNCH|KITCHEN", regex=True, na=False)
cond_stationery = desc.str.contains(r"NOTEBOOK|PEN|PENCIL|PAPER|CARD|CARDS|STICKERS|STAMP|CALENDAR|ENVELOPE|TAPE|ERASER", regex=True, na=False)
cond_bags = desc.str.contains(r"BAG|TOTE|PURSE|WALLET|POUCH|SCARF|JEWELLERY|NECKLACE|BRACELET|CHARM", regex=True, na=False)
cond_gifts = desc.str.contains(r"CHRISTMAS|XMAS|EASTER|BIRTHDAY|PARTY|GIFT|WRAP|RIBBON|FESTIVE|HALLOWEEN|ADVENT", regex=True, na=False)
cond_toys = desc.str.contains(r"TOY|GAME|DOLL|PUZZLE|PLUSH|TEDDY|BALLOONS|PLAYING CARDS|YO-YO|CRAFT|CHILDRENS", regex=True, na=False)
cond_beauty = desc.str.contains(r"SOAP|BATH|LIP GLOSS|COMPACT|SLIPPERS|SOCKS|LAVENDER|SCENT", regex=True, na=False)

category_conditions = [
    cond_home,
    cond_kitchen,
    cond_stationery,
    cond_bags,
    cond_gifts,
    cond_toys,
    cond_beauty
]

category_choices = [
    "Home & Living",
    "Kitchen & Dining",
    "Stationery & Office",
    "Bags & Accessories",
    "Gifts & Seasonal",
    "Toys & Games",
    "Health & Beauty"
]

df["Product_Category"] = np.select(category_conditions, category_choices, default="Other General Retail")

print("Modeling realistic delivery fulfillment times...")
is_uk = df["Country"] == "United Kingdom"
is_europe = df["Country"].isin(["Germany", "France", "EIRE", "Spain", "Netherlands", "Belgium", "Switzerland", "Portugal", "Italy", "Sweden"])

n = len(df)
noise = np.random.normal(0, 0.8, n)

uk_days = np.clip(np.round(3.0 + noise), 1, 7)
europe_days = np.clip(np.round(6.0 + noise * 1.2), 3, 12)
intl_days = np.clip(np.round(11.0 + noise * 1.5), 6, 20)

conditions_delivery = [
    df["Is_Cancelled"],
    is_uk,
    is_europe
]
choices_delivery = [
    0, # Cancelled orders have 0 delivery days
    uk_days,
    europe_days
]
df["Delivery_Days"] = np.select(conditions_delivery, choices_delivery, default=intl_days).astype(int)

cond_status_cancelled = df["Is_Cancelled"]
cond_status_ontime = (is_uk & (df["Delivery_Days"] <= 3)) | (is_europe & (df["Delivery_Days"] <= 6)) | (~is_uk & ~is_europe & (df["Delivery_Days"] <= 11))
cond_status_slight = (is_uk & (df["Delivery_Days"] <= 5)) | (is_europe & (df["Delivery_Days"] <= 8)) | (~is_uk & ~is_europe & (df["Delivery_Days"] <= 14))

delivery_status_conditions = [
    cond_status_cancelled,
    cond_status_ontime,
    cond_status_slight
]
delivery_status_choices = [
    "Cancelled",
    "On-Time",
    "Slight Delay"
]
df["Delivery_Status"] = np.select(delivery_status_conditions, delivery_status_choices, default="Major Delay")

print("Modeling customer reviews and satisfaction ratings...")
rand_review = np.random.rand(n)

cond_rev_cancelled = df["Is_Cancelled"]
cond_rev_ontime = df["Delivery_Status"] == "On-Time"
cond_rev_slight = df["Delivery_Status"] == "Slight Delay"

rating_ontime = np.where(rand_review < 0.65, 5, np.where(rand_review < 0.90, 4, 3))
rating_slight = np.where(rand_review < 0.20, 5, np.where(rand_review < 0.60, 4, np.where(rand_review < 0.90, 3, 2)))
rating_delay = np.where(rand_review < 0.05, 5, np.where(rand_review < 0.20, 4, np.where(rand_review < 0.50, 3, np.where(rand_review < 0.85, 2, 1))))
rating_cancelled = np.where(rand_review < 0.70, 1, np.where(rand_review < 0.90, 2, 3))

conditions_rating = [
    cond_rev_cancelled,
    cond_rev_ontime,
    cond_rev_slight
]
choices_rating = [
    rating_cancelled,
    rating_ontime,
    rating_slight
]
df["Review_Rating"] = np.select(conditions_rating, choices_rating, default=rating_delay).astype(int)

cond_exc = df["Review_Rating"] == 5
cond_good = df["Review_Rating"] == 4
cond_avg = df["Review_Rating"] == 3

rating_cat_conditions = [cond_exc, cond_good, cond_avg]
rating_cat_choices = ["Excellent", "Good", "Average"]
df["Review_Category"] = np.select(rating_cat_conditions, rating_cat_choices, default="Poor / Needs Improvement")

print("Computing Customer RFM Segments...")
valid_cust = df[df["Customer ID"].notnull() & (~df["Is_Cancelled"])]
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = valid_cust.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "Invoice": "nunique",
    "Total_Amount": "sum"
}).rename(columns={"InvoiceDate": "Recency", "Invoice": "Frequency", "Total_Amount": "Monetary"})

rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["RFM_Score"] = rfm["R_Score"] * 100 + rfm["F_Score"] * 10 + rfm["M_Score"]

cond_champions = (rfm["R_Score"] >= 4) & (rfm["F_Score"] >= 3)
cond_loyal = (rfm["R_Score"] >= 3) & (rfm["F_Score"] >= 3)
cond_potential = (rfm["R_Score"] >= 3) & (rfm["F_Score"] <= 2)
cond_at_risk = (rfm["R_Score"] <= 2) & (rfm["F_Score"] >= 3)
cond_hibernating = (rfm["R_Score"] <= 2) & (rfm["F_Score"] <= 2)

rfm_conds = [cond_champions, cond_loyal, cond_potential, cond_at_risk]
rfm_choices = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk Customers"]
rfm["Customer_Segment"] = np.select(rfm_conds, rfm_choices, default="Hibernating / Inactive")

df = df.merge(rfm[["Customer_Segment"]], on="Customer ID", how="left")
df["Customer_Segment"] = df["Customer_Segment"].fillna("Guest / Unregistered")

print(f"Final cleaned dataset shape: {df.shape}")
print("Saving cleaned dataset to ecommerce_data_cleaned.csv...")
df.to_csv("ecommerce_data_cleaned.csv", index=False)

print("Creating representative stratified sample of 100,000 rows for lightning-fast Streamlit dashboard exploration...")
sample_df = df.sample(n=100000, random_state=42).sort_values("InvoiceDate")
sample_df.to_csv("ecommerce_sample.csv", index=False)

elapsed = time.time() - start_time
print(f"Data pipeline complete in {elapsed:.1f}s!")
print(f"Cleaned CSV: ecommerce_data_cleaned.csv ({len(df)} rows)")
print(f"Sample CSV: ecommerce_sample.csv ({len(sample_df)} rows)")
