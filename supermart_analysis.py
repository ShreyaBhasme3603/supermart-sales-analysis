"""
Supermart Sales Analysis
Data Analytics + AI/ML Forecasting Project
Run: python supermart_analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("supermart_sales_data.csv", parse_dates=["Order Date","Ship Date"])

print("\n--- BASIC OVERVIEW ---")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nMissing values:\n", df.isnull().sum())
print("\nTotal Sales:", round(df["Sales"].sum(),2))
print("Total Profit:", round(df["Profit"].sum(),2))
print("Total Orders:", df["Order ID"].nunique())
print("Average Order Value:", round(df["Sales"].mean(),2))

# Monthly analysis
df["Order Month"] = df["Order Date"].dt.to_period("M").astype(str)
monthly = df.groupby("Order Month").agg(
    Sales=("Sales","sum"),
    Profit=("Profit","sum"),
    Orders=("Order ID","nunique")
).reset_index()

# Category analysis
category = df.groupby("Category").agg(
    Sales=("Sales","sum"),
    Profit=("Profit","sum"),
    Quantity=("Quantity","sum")
).sort_values("Sales", ascending=False)

print("\n--- CATEGORY ANALYSIS ---")
print(category)

print("\n--- REGION ANALYSIS ---")
print(df.groupby("Region").agg(
    Sales=("Sales","sum"),
    Profit=("Profit","sum")
).sort_values("Sales", ascending=False))

# Visualizations
plt.figure(figsize=(9,5))
plt.plot(monthly["Order Month"], monthly["Sales"], marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
category["Sales"].plot(kind="bar")
plt.title("Sales by Category")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

# AI/ML component: simple sales forecasting
monthly["Index"] = range(len(monthly))
X = monthly[["Index"]]
y = monthly["Sales"]

model = LinearRegression()
model.fit(X, y)

future = pd.DataFrame({"Index": range(len(monthly), len(monthly)+3)})
future["Forecast Sales"] = model.predict(future[["Index"]])

print("\n--- AI/ML SALES FORECAST ---")
print(future)
print("Model R2 on historical monthly data:", round(model.score(X,y),4))

print("\nProject completed successfully.")
