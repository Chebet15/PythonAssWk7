import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -------------------------
# Task 1: Load and Explore the Dataset
# -------------------------
try:
    # Load iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Clean dataset (no missing values in Iris, but if there were:)
    df = df.dropna()  # or df.fillna(df.mean())

except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# -------------------------
# Task 2: Basic Data Analysis
# -------------------------
# Compute basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Grouping by species and calculating mean
species_mean = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(species_mean)

# Interesting findings example:
# We can observe that petal length and width are key features distinguishing species.

# -------------------------
# Task 3: Data Visualization
# -------------------------
plt.style.use('seaborn-v0_8')

# 1. Line Chart - trend of sepal length across samples
plt.figure(figsize=(8,5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title('Line Chart: Sepal Length Across Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# 2. Bar Chart - average petal length per species
plt.figure(figsize=(8,5))
sns.barplot(x='species', y='petal length (cm)', data=df, estimator=np.mean)
plt.title('Bar Chart: Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram - distribution of sepal width
plt.figure(figsize=(8,5))
plt.hist(df['sepal width (cm)'], bins=15, edgecolor='black')
plt.title('Histogram: Sepal Width Distribution')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot - sepal length vs petal length
plt.figure(figsize=(8,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()
