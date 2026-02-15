# =========================================================
# APPLIED DATA SCIENCE
# Assignment 1 – Python for Data Handling
# Dataset: Vehicle Car Data (Kaggle)
# =========================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("car data.csv")
print("Dataset Loaded Successfully")
print("="*100)

# ---------------------------------------------------------
# 3. Clean Column Names
# ---------------------------------------------------------
df.columns = df.columns.str.strip().str.lower()
print("Columns after cleaning:")
print(df.columns.tolist())
print("="*100)

# ---------------------------------------------------------
# 4. Dataset Exploration
# ---------------------------------------------------------
print("First 5 Records:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()
print("="*100)

# ---------------------------------------------------------
# 5. Check Missing Values and Zero Values
# ---------------------------------------------------------
print("Missing Values:")
print(df.isnull().sum())

numerical_cols = df.select_dtypes(include=np.number).columns
print("\nZero Values in Numerical Columns:")
print((df[numerical_cols] == 0).sum())
print("="*100)

# ---------------------------------------------------------
# 6. Remove Duplicate Records
# ---------------------------------------------------------
print("Duplicate Records:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates Removed")
print("="*100)

# ---------------------------------------------------------
# 7. Handle Missing Values
# ---------------------------------------------------------
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing Values Handled")
print("="*100)

# ---------------------------------------------------------
# 8. Detect Price Columns Automatically
# ---------------------------------------------------------
price_cols = [col for col in df.columns if "price" in col]
print("Detected Price Columns:", price_cols)

df[price_cols] = df[price_cols].apply(pd.to_numeric, errors='coerce')
df[price_cols] = df[price_cols].fillna(df[price_cols].mean())
print("="*100)

# ---------------------------------------------------------
# 9. Feature Engineering – Car Age
# ---------------------------------------------------------
df['car_age'] = 2024 - df['year']
print("Car Age Column Created")
print("="*100)

# ---------------------------------------------------------
# 10. Statistical Measures
# ---------------------------------------------------------
print("Selling Price Statistics")
print("Mean:", df['selling_price'].mean())
print("Median:", df['selling_price'].median())
print("Mode:", df['selling_price'].mode()[0])
print("Skewness:", df['selling_price'].skew())
print("="*100)

# ---------------------------------------------------------
# 11. Basic Visualization (For Output Screenshots)
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df['selling_price'], kde=True)
plt.title("Distribution of Selling Price")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='fuel_type', data=df)
plt.title("Fuel Type Distribution")
plt.show()

print("Preprocessing Completed Successfully")
