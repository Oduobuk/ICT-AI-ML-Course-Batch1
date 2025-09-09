# Month 1, Week 4: Deep Dive into Data Cleaning and Visualization

### **Objective:**
To develop practical skills in identifying and handling common data quality issues. To create foundational data visualizations for the purpose of Exploratory Data Analysis (EDA).

---

## Part 1: Data Cleaning

Data cleaning is the process of fixing or removing incorrect, corrupted, incorrectly formatted, duplicate, or incomplete data within a dataset. It is one of the most important and time-consuming tasks in data science.

### **1.1 Handling Missing Values**

Pandas represents missing values with `np.nan`. Identifying them is the first step.

*   **Identifying Missing Values:**
    *   `df.isnull()`: Returns a boolean DataFrame of the same size, indicating `True` for `NaN`.
    *   `df.isnull().sum()`: A common and useful chain to count `NaN`s in each column.

*   **Strategies for Handling `NaN`:**
    1.  **Dropping:** The simplest strategy.
        *   `df.dropna()`: Drops any row containing at least one `NaN`.
        *   `df.dropna(axis='columns')`: Drops any column containing `NaN`.
        *   **Caution:** Be careful not to drop too much data.

    2.  **Imputation (Filling):** The preferred strategy when data is valuable.
        *   `df.fillna(0)`: Fills all `NaN`s with a specific value (e.g., 0).
        *   **Dynamic Imputation:** It is often better to fill with a calculated statistic.
            ```python
            # Fill missing values in a column with the mean of that column
            mean_value = df['my_column'].mean()
            df['my_column'].fillna(mean_value, inplace=True)
            ```

### **1.2 Handling Duplicates**

*   `df.duplicated()`: Returns a boolean Series indicating duplicate rows.
*   `df.drop_duplicates()`: Returns a DataFrame with duplicate rows removed.

### **1.3 Handling Outliers**

Outliers are data points that differ significantly from other observations. They can be legitimate data or errors.

*   **Identification:** A common method is using the Interquartile Range (IQR).
    *   An outlier is often defined as a value less than `Q1 - 1.5 * IQR` or greater than `Q3 + 1.5 * IQR`.
*   **Strategy:** The strategy depends on the domain. Sometimes they are removed, other times they are capped or transformed (e.g., with a log function).

---

## Part 2: Introduction to Data Visualization

Visualization is crucial for exploring data, identifying patterns, and communicating results.

### **2.1 Matplotlib: The Foundation**

*   Matplotlib is the fundamental plotting library in Python. It is highly customizable but can be complex.
*   **Core Objects:** The `Figure` (the overall window) and the `Axes` (the individual plot).

    ```python
    import matplotlib.pyplot as plt

    # Create a figure and an axes
    fig, ax = plt.subplots()

    # Plot data
    ax.plot(x_data, y_data)

    # Customize
    ax.set_title("My Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    plt.show() # Display the plot
    ```

### **2.2 Seaborn: High-Level Statistical Graphics**

*   Seaborn is built on top of Matplotlib and provides a more high-level, aesthetically pleasing interface for common statistical plots.

*   **Common Plots for EDA:**
    *   **`sns.histplot(data=df, x='column')`:** Histogram to see the distribution of a single numerical variable.
    *   **`sns.boxplot(data=df, x='category', y='value')`:** Box plot to see distributions across categories.
    *   **`sns.scatterplot(data=df, x='col1', y='col2')`:** Scatter plot to see the relationship between two numerical variables.
    *   **`sns.heatmap(df.corr(), annot=True)`:** Heatmap, often used to visualize the correlation matrix of a DataFrame.

### **Summary of Deep Dive:**
Effective data cleaning is a prerequisite for any meaningful analysis. Visualization is not just for final reports; it is a primary tool for exploration and understanding your data (EDA). Seaborn provides powerful, one-line commands for most common EDA plots.
