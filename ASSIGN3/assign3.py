# =========================================================
# APPLIED DATA SCIENCE
# Assignment – Data Visualization using Pandas
# Dataset: Global YouTube Statistics (Kaggle)
# =========================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("Global YouTube Statistics.csv", encoding='latin1')
print("Dataset Loaded Successfully")
print("="*100)

# ---------------------------------------------------------
# 3. Clean Column Names
# ---------------------------------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Columns after cleaning:")
print(df.columns.tolist())
print("="*100)

# ---------------------------------------------------------
# 4. Dataset Exploration
# ---------------------------------------------------------
print("First 5 Records:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("="*100)

# ---------------------------------------------------------
# 5. Handle Missing Values
# ---------------------------------------------------------
numerical_cols = df.select_dtypes(include=np.number).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing Values Handled")
print("="*100)

# ---------------------------------------------------------
# 6. HISTOGRAM
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
plt.hist(df['subscribers'], bins=30)
plt.title("Distribution of Subscribers")
plt.xlabel("Subscribers")
plt.ylabel("Frequency")
plt.show()

# ---------------------------------------------------------
# 7. BOXPLOT
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x=df['highest_yearly_earnings'])
plt.title("Boxplot of Highest Yearly Earnings")
plt.show()

# ---------------------------------------------------------
# 8. SCATTERPLOT
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
plt.scatter(df['subscribers'], df['video_views'])
plt.title("Subscribers vs Video Views")
plt.xlabel("Subscribers")
plt.ylabel("Video Views")
plt.show()

# ---------------------------------------------------------
# 9. BAR CHART
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
df['channel_type'].value_counts().plot(kind='bar')
plt.title("Channel Type Distribution")
plt.xlabel("Channel Type")
plt.ylabel("Count")
plt.show()

print("Data Visualization Completed Successfully")
