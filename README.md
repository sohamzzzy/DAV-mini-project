# E-Commerce Sales & Customer Behavior Analytics
**Data Analysis and Visualization (DAV) Lab Mini-Project**

---

## 👥 Project Information & Team Members

- **Course / Subject:** Data Analysis and Visualization (DAV) Lab
- **Section:** C2
- **Batch:** B1

| Sr. No. | Student Name | Roll Number | Role |
| :---: | :--- | :---: | :--- |
| 1 | **Vaishnav Balpande** | `C2-B1-01` | Data Pipeline & Analytics |
| 2 | **Soham Admane** | `C2-B1-08` | Dashboard & Modeling |
| 3 | **Lokesh Parate** | `C2-B1-15` | Visualizations & Documentation |

---

## 📌 Problem Statement

Online retailers record millions of transactional invoice lines every year, but raw transaction tables alone do not reveal which products, regions, time windows, or customer profiles actually drive growth. Without proactive visual analytics, fulfillment bottlenecks, negative reviews, seasonal inventory surges, and customer churn risk go unnoticed until revenue has already dropped.

### Objective
To build an end-to-end analytical pipeline and an interactive executive web dashboard that turns **1,067,371 raw transaction records** (UCI Online Retail II dataset, Dec 2009 – Dec 2011) into actionable business intelligence across seven dimensions:
1. **Revenue Analytics:** Monthly sales trends, revenue distribution, order values, and best-selling items.
2. **Product Categories:** Rule-based keyword categorization via `numpy.select` into 7 major retail categories.
3. **Customer Locations:** Domestic (UK) vs. international export markets with an interactive global choropleth map.
4. **Delivery & Logistics:** Fulfillment transit days modeling and on-time SLA compliance tracking.
5. **Customer Satisfaction:** Review ratings (1 to 5 stars) correlated against fulfillment speed.
6. **Purchasing Trends:** 24-hour order velocity, day-of-week buying behavior, and **RFM (Recency, Frequency, Monetary)** customer segmentation.
7. **Forecasting & ML:** 7-day SMA/EMA smoothing, **ARIMA(1,1,1)** revenue forecasting, correlation heatmap, and **Linear Regression** evaluated via $R^2$ and MSE.

---

## 🧪 Curriculum Mapping: DAV Practicals 1 to 8

| Practical | Syllabus Topic | Implementation in Project |
| :---: | :--- | :--- |
| **Lab 1** | Ingestion & Schema Inspection | Reading CSV/XLSX, validating shapes, schemas, and missing fields. |
| **Lab 2** | Data Cleaning & Preprocessing | Whitespace stripping, zero-price filtering, cancellation flagging (`'C'`). |
| **Lab 3** | Correlation & Linear Regression | Correlation matrix (`df.corr()`), train-test split, OLS regression ($R^2$, MSE). |
| **Lab 4** | Multidimensional Groupings | Multi-level aggregations by country, category, and time dimensions. |
| **Lab 5** | Data Binning & Discretization | Quartile binning using `pd.qcut` for RFM customer segmentation. |
| **Lab 6** | Time Series & Forecasting | 7-day SMA/EMA, additive seasonal decomposition, and ARIMA(1,1,1). |
| **Lab 7** | Static Visuals & Annotations | Seaborn theme styling, peak revenue annotations, and scatter plots. |
| **Lab 8** | Conditional Logic & Dashboards | `numpy.select` categorization, horizontal bars, donut charts, and Plotly widgets. |

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Interactive Web Dashboard:** Streamlit
- **Data Manipulation & Logic:** Pandas, NumPy
- **Interactive Visualization:** Plotly Express & Plotly Graph Objects
- **Statistical & Time Series Modeling:** Statsmodels, SciPy
- **Machine Learning & Evaluation:** Scikit-Learn
- **Static Publication Visuals:** Matplotlib, Seaborn
- **Excel Ingestion Engine:** OpenPyXL

---

## 📂 Repository Structure

```
├── app.py                                   # Streamlit Interactive Executive Web Dashboard (7 Tabs)
├── data_preprocessing.py                    # ETL pipeline, regex categorization & RFM segmentation
├── ecommerce_analytics_lab.py               # Standalone batch script generating 15 PNGs + 2 HTMLs
├── E_Commerce_Analytics_Project.ipynb       # 25-cell interactive Jupyter Notebook
├── PROJECT_DOCUMENTATION_REPORT.md          # Comprehensive technical report & limitations critique
├── E_Commerce_Analytics_Presentation.pptx   # Final 15-slide project presentation deck
├── ecommerce_sample.csv                     # Stratified 100,000-row sample for fast execution
├── requirements.txt                         # Full Python package dependencies list
└── .gitignore                               # Excludes files exceeding GitHub's 100MB limit
```

---

## 🚀 Quickstart & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/sohamzzzy/DAV-mini-project.git
cd DAV-mini-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
python -m streamlit run app.py
```
> The dashboard will open in your browser at `http://localhost:8501`.

### 4. Run the Standalone Lab Analytics Script
```bash
python ecommerce_analytics_lab.py
```

### 5. Launch the Jupyter Notebook
```bash
python -m notebook E_Commerce_Analytics_Project.ipynb
```
