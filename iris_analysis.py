import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset from sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    df = None

if df is not None:
    # Display first few rows
    print("First 5 rows:")
    print(df.head())
    
    # Explore structure
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Clean dataset (fill or drop missing values)
    if df.isnull().values.any():
        df = df.dropna()
        print("Missing values dropped.")
    else:
        print("No missing values found.")
    
    # Task 2: Basic Data Analysis
    print("\nBasic statistics:")
    print(df.describe())
    
    # Group by species and compute mean of numerical columns
    print("\nMean values by species:")
    print(df.groupby('target').mean())
    
    # Task 3: Data Visualization
    sns.set(style="whitegrid")
    
    # Line chart: sepal length over samples
    plt.figure(figsize=(8,4))
    plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
    plt.title('Sepal Length Over Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Sepal Length (cm)')
    plt.legend()
    plt.show()
    
    # Bar chart: average petal length per species
    plt.figure(figsize=(8,4))
    species_means = df.groupby('target')['petal length (cm)'].mean()
    species_names = [iris.target_names[i] for i in species_means.index]
    plt.bar(species_names, species_means)
    plt.title('Average Petal Length per Species')
    plt.xlabel('Species')
    plt.ylabel('Average Petal Length (cm)')
    plt.show()
    
    # Histogram: sepal width distribution
    plt.figure(figsize=(8,4))
    plt.hist(df['sepal width (cm)'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Scatter plot: sepal length vs petal length
    plt.figure(figsize=(8,4))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df, palette='Set2', legend='full')
    plt.title('Sepal Length vs Petal Length by Species')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species', labels=iris.target_names)
    plt.show()
else:
    print("Dataset not available for analysis.")
