```python
# ============================================================
# APPAREL & TEXTILES E-COMMERCE - EXPLORATORY DATA ANALYSIS
# Week 2 Internship Project
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_PATH = "data/e-shop clothing 2008.csv"
OUTPUT_DIR = "visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("APPAREL & TEXTILES E-COMMERCE EDA")
print("=" * 60)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ------------------------------------------------------------
# 3. BASIC DATA INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()


# ------------------------------------------------------------
# 4. MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("\nNo missing values found.")
else:
    print("\nMissing values detected.")


# ------------------------------------------------------------
# 5. DUPLICATE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)


# ------------------------------------------------------------
# 6. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe(include="all"))


# ------------------------------------------------------------
# 7. DATA CLEANING
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Remove exact duplicate rows
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# Convert selected columns to categorical
categorical_columns = [
    "month",
    "day",
    "country",
    "page 1 (main category)",
    "page 2 (clothing model)",
    "colour",
    "location",
    "model photography",
    "price 2",
    "page"
]

for column in categorical_columns:
    if column in df.columns:
        df[column] = df[column].astype("category")


# ------------------------------------------------------------
# 8. CREATE DATE COLUMN
# ------------------------------------------------------------

if all(column in df.columns for column in ["year", "month", "day"]):

    df["date"] = pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + df["month"].astype(str)
        + "-"
        + df["day"].astype(str),
        errors="coerce"
    )

    print("\nDate column created successfully.")


# ------------------------------------------------------------
# 9. CATEGORY MAPPING
# ------------------------------------------------------------

category_mapping = {
    1: "Trousers",
    2: "Skirts",
    3: "Blouses",
    4: "Sale"
}

if "page 1 (main category)" in df.columns:

    df["category_name"] = (
        df["page 1 (main category)"]
        .astype(str)
        .map(category_mapping)
    )

    print("\nCategory distribution:")
    print(df["category_name"].value_counts())


# ------------------------------------------------------------
# 10. MONTHLY ACTIVITY ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MONTHLY ACTIVITY")
print("=" * 60)

monthly_activity = df.groupby(
    "month",
    observed=True
).size()

print(monthly_activity)


# ------------------------------------------------------------
# 11. VISUALIZATION 1
# MONTHLY ACTIVITY LINE CHART
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    monthly_activity.index.astype(str),
    monthly_activity.values,
    marker="o",
    linewidth=2
)

plt.title("Monthly Online Shopping Activity")
plt.xlabel("Month")
plt.ylabel("Number of Records")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "monthly_activity.png"),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 12. CATEGORY ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CATEGORY ANALYSIS")
print("=" * 60)

category_counts = df["category_name"].value_counts()

print(category_counts)

category_percentage = (
    category_counts / category_counts.sum()
) * 100

print("\nCategory Percentage:")
print(category_percentage.round(2))


# ------------------------------------------------------------
# 13. VISUALIZATION 2
# CATEGORY BAR CHART
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

category_counts.plot(kind="bar")

plt.title("Apparel Category Distribution")
plt.xlabel("Category")
plt.ylabel("Number of Records")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "category_distribution.png"),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 14. PRICE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PRICE ANALYSIS")
print("=" * 60)

print("\nPrice Statistics:")

print(df["price"].describe())

print("\nMedian Price:")
print(df["price"].median())

print("\nMean Price:")
print(df["price"].mean())


# ------------------------------------------------------------
# 15. VISUALIZATION 3
# PRICE DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    df["price"].dropna(),
    bins=30
)

plt.axvline(
    df["price"].median(),
    linestyle="--",
    linewidth=2,
    label="Median Price"
)

plt.title("Distribution of Apparel Prices")
plt.xlabel("Price (USD)")
plt.ylabel("Frequency")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "price_distribution.png"),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 16. PRICE BY CATEGORY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PRICE BY CATEGORY")
print("=" * 60)

category_price = df.groupby(
    "category_name",
    observed=True
)["price"].agg(
    ["mean", "median", "min", "max"]
)

print(category_price)


# ------------------------------------------------------------
# 17. VISUALIZATION 4
# PRICE BY CATEGORY BOXPLOT
# ------------------------------------------------------------

categories = df["category_name"].dropna().unique()

price_data = [
    df.loc[
        df["category_name"] == category,
        "price"
    ].dropna()
    for category in categories
]

plt.figure(figsize=(9, 5))

plt.boxplot(
    price_data,
    labels=categories
)

plt.title("Price Distribution by Apparel Category")
plt.xlabel("Category")
plt.ylabel("Price (USD)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "price_by_category.png"),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 18. PRODUCT MODEL ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TOP CLOTHING MODELS")
print("=" * 60)

top_models = (
    df["page 2 (clothing model)"]
    .value_counts()
    .head(10)
)

print(top_models)


# ------------------------------------------------------------
# 19. TOP PRODUCT VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

top_models.sort_values().plot(kind="barh")

plt.title("Top 10 Most Viewed Clothing Models")
plt.xlabel("Number of Records")
plt.ylabel("Clothing Model")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top_clothing_models.png"),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 20. COUNTRY ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TOP COUNTRIES")
print("=" * 60)

top_countries = df["country"].value_counts().head(10)

print(top_countries)


# ------------------------------------------------------------
# 21. SESSION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SESSION ANALYSIS")
print("=" * 60)

session_summary = df.groupby("session ID").agg(
    total_clicks=("order", "count"),
    unique_products=("page 2 (clothing model)", "nunique"),
    average_price=("price", "mean")
)

print("\nSession Summary:")
print(session_summary.head())

print("\nSession Statistics:")
print(session_summary.describe())


# ------------------------------------------------------------
# 22. OUTLIER ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PRICE OUTLIER ANALYSIS")
print("=" * 60)

Q1 = df["price"].quantile(0.25)
Q3 = df["price"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

price_outliers = df[
    (df["price"] < lower_limit)
    | (df["price"] > upper_limit)
]

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

print(
    "Number of price outliers:",
    len(price_outliers)
)


# ------------------------------------------------------------
# 23. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

numeric_columns = df.select_dtypes(
    include=np.number
).columns

correlation_matrix = df[
    numeric_columns
].corr()

print(correlation_matrix)


# ------------------------------------------------------------
# 24. CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.imshow(
    correlation_matrix,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix of Numerical Variables")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_heatmap.png"),
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 25. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nNumber of Products:")
print(df["page 2 (clothing model)"].nunique())

print("\nNumber of Countries:")
print(df["country"].nunique())

print("\nAverage Price:")
print(round(df["price"].mean(), 2))

print("\nMedian Price:")
print(round(df["price"].median(), 2))

print("\nMost Popular Category:")
print(category_counts.idxmax())

print("\nMost Viewed Clothing Model:")
print(top_models.index[0])

print("\nEDA completed successfully!")

print("\nVisualizations saved in:")
print(OUTPUT_DIR)
```

