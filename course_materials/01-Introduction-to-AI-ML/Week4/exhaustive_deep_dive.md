# Month 1, Week 4: Exhaustive Deep Dive

### **Objective:**

To gain mastery over data cleaning by handling complex data types and to
develop a nuanced understanding of visualization for effective data
storytelling and analysis.

## Part 1: Advanced Data Cleaning Techniques

### **1.1 String Manipulation for Cleaning Text Data**

Real-world data is messy. Text columns often require significant
cleaning using Pandas' built-in string methods, which can be accessed
via the `.str` accessor on a Series.

-   **Common** `.str` **Methods:**
    -   `.lower()` / `.upper()`: Convert text to a consistent case.
    -   `.strip()`: Remove leading/trailing whitespace.
    -   `.replace(old, new)`: Replace a substring.
    -   `.contains(pattern)`: Check for the existence of a substring,
        returns a boolean Series (useful for filtering).
    -   `.extract(regex_pattern)`: Extract capture groups from a regular
        expression, returning a new DataFrame.

```{=html}
<!-- -->
```
-   import pandas as pd
        df = pd.DataFrame({'product_code': [' ABC-123 ', 'def-456', 'GHI-789-US']})

        # Chain multiple operations
        df['product_code_clean'] = df['product_code'].str.strip().str.upper().str.replace('-', '')

### **1.2 Custom Transformations with** `.apply()`

When built-in functions are not enough, `.apply()` lets you use your own
function on a Series (column) or DataFrame.

-   **Use Case:** Applying a complex data cleaning rule or creating a
    new feature from multiple columns.

```{=html}
<!-- -->
```
-   def get_price_category(price):
            if price > 1000:
                return 'High'
            elif price > 200:
                return 'Medium'
            else:
                return 'Low'

        # df['price_category'] = df['price'].apply(get_price_category)

    **Performance Warning:** `.apply()` is essentially a loop and can be
    slow on large datasets. Use vectorized operations (NumPy/Pandas
    built-in functions) whenever possible.

## Part 2: Advanced Data Visualization

### **2.1 Customizing Matplotlib Plots**

Full control over your plots is essential for professional reports and
presentations.

-   **Anatomy of a Plot:** Figure, Axes, Title, Labels, Ticks, Legend,
    Spines.

-   **Working with Subplots:** Creating a grid of plots to compare
    different views of the data.

```{=html}
<!-- -->
```
-   import matplotlib.pyplot as plt
        # Create a 2x2 grid of plots
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

        # Access each subplot by its index
        axes[0, 0].hist(data_for_plot1)
        axes[0, 0].set_title('Plot 1')

        axes[1, 1].scatter(x_data, y_data)
        axes[1, 1].set_title('Plot 4')

        plt.tight_layout() # Adjusts subplot params for a tight layout
        plt.show()

```{=html}
<!-- -->
```
-   **Adding Annotations:** Using `ax.text()` or `ax.annotate()` to add
    text or arrows to highlight specific data points.

### **2.2 Advanced Seaborn Plots for Deeper Insights**

-   `sns.pairplot(df)`**:** Creates a grid of scatterplots for every
    pair of numerical columns in a DataFrame, with histograms on the
    diagonal. It is the single most effective command for getting a
    quick overview of your entire dataset.

-   `sns.jointplot(data=df, x='col1', y='col2')`**:** A combination of a
    scatter plot and histograms for two variables. It allows you to see
    both the relationship and the individual distributions.

-   `sns.catplot(data=df, x='category', y='value', kind='bar')`**:** A
    "figure-level" interface for categorical plots. The `kind` parameter
    can be changed to `box`, `violin`, `strip`, etc., making it
    incredibly versatile for comparing distributions across categories.

-   `sns.lmplot(data=df, x='col1', y='col2')`**:** A scatter plot with a
    linear regression line automatically fitted and displayed, including
    the confidence interval. Excellent for visualizing linear
    relationships.

### **Summary of Exhaustive Deep Dive:**

Cleaning real-world data often requires custom logic and string
manipulation. For visualization, moving beyond single plots to grids of
subplots (`subplots`, `pairplot`) allows for richer comparisons.
Understanding the Matplotlib object hierarchy (Figure, Axes) is the key
to unlocking full customization for creating publication-quality
graphics.
