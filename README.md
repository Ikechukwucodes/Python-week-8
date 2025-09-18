~# Iris Dataset Analysis

This project analyzes the classic Iris dataset using Python, pandas, matplotlib, and seaborn. The script `iris_analysis.py` performs data loading, exploration, basic analysis, and visualization.

## Features

### 1. Load and Explore the Dataset
- Loads the Iris dataset using `sklearn.datasets.load_iris()`.
- Converts the dataset to a pandas DataFrame.
- Displays the first few rows with `.head()`.
- Shows data types and checks for missing values.
- Cleans the dataset by dropping any missing values (if present).
- Handles errors during loading with try/except.

### 2. Basic Data Analysis
- Computes basic statistics (mean, median, standard deviation) using `.describe()`.
- Groups data by species and calculates the mean of numerical columns for each group.
- Prints findings and summary statistics.

### 3. Data Visualization
Creates four types of plots:
- **Line Chart:** Shows sepal length trends over sample index.
- **Bar Chart:** Compares average petal length across species.
- **Histogram:** Displays the distribution of sepal width.
- **Scatter Plot:** Visualizes the relationship between sepal length and petal length, colored by species.

All plots are customized with titles, axis labels, and legends for clarity.

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies with:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage
Run the script in your terminal:
```
python iris_analysis.py
```

## Error Handling
- If required packages are missing, the script will raise an ImportError.
- If the dataset fails to load, an error message will be printed.
- Missing data is handled by dropping rows with missing values.

## Customization
You can modify the script to use a different CSV dataset by replacing the data loading section. Update the analysis and visualization sections as needed for your dataset.

## References
- [Iris Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- [scikit-learn documentation](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)
