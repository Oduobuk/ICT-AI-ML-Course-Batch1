# Data Visualization with Python

## Table of Contents
1. [Introduction to Data Visualization](#introduction-to-data-visualization)
2. [Matplotlib Fundamentals](#matplotlib-fundamentals)
3. [Seaborn for Statistical Visualization](#seaborn-for-statistical-visualization)
4. [Plotly for Interactive Visualizations](#plotly-for-interactive-visualizations)
5. [Best Practices and Design Principles](#best-practices-and-design-principles)

## Introduction to Data Visualization

### Why Visualize Data?
- Identify patterns and trends
- Detect outliers and anomalies
- Communicate insights effectively
- Support data-driven decisions

### Types of Visualizations
- **Univariate Analysis**: Histograms, Box Plots, Violin Plots
- **Bivariate Analysis**: Scatter Plots, Line Charts, Bar Charts
- **Multivariate Analysis**: Heatmaps, Pair Plots, Parallel Coordinates
- **Geospatial**: Maps, Choropleth Maps
- **Time Series**: Line Charts, Area Charts, Candlestick Charts

## Matplotlib Fundamentals

### Basic Plotting
```python
import matplotlib.pyplot as plt
import numpy as np

# Create a simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linestyle='--', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
```

### Multiple Plots
```python
# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# First subplot
ax1.plot(x, np.sin(x), 'r-')
ax1.set_title('Sine Wave')

# Second subplot
ax2.plot(x, np.cos(x), 'b--')
ax2.set_title('Cosine Wave')

plt.tight_layout()
plt.show()
```

## Seaborn for Statistical Visualization

### Distribution Plots
```python
import seaborn as sns
import pandas as pd

# Load sample dataset
tips = sns.load_dataset('tips')

# Distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(data=tips, x='total_bill', kde=True, bins=30)
plt.title('Distribution of Total Bill Amounts')
plt.show()
```

### Categorical Plots
```python
# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='total_bill', data=tips, hue='sex')
plt.title('Total Bill by Day and Gender')
plt.show()

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True)
plt.title('Distribution of Bills by Day and Gender')
plt.show()
```

### Heatmaps
```python
# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = tips.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

## Plotly for Interactive Visualizations

### Basic Interactive Plot
```python
import plotly.express as px

# Interactive scatter plot
fig = px.scatter(
    tips,
    x='total_bill',
    y='tip',
    color='sex',
    size='size',
    hover_data=['day', 'time'],
    title='Tips by Total Bill and Gender'
)
fig.show()
```

### Interactive 3D Plot
```python
# 3D scatter plot
fig = px.scatter_3d(
    tips,
    x='total_bill',
    y='tip',
    z='size',
    color='sex',
    title='3D View of Tips Data'
)
fig.show()
```

## Best Practices and Design Principles

### Design Principles
1. **Clarity**: Make sure your visualization is easy to understand
2. **Accuracy**: Represent data truthfully
3. **Efficiency**: Convey maximum information with minimum ink
4. **Aesthetics**: Use colors and styles effectively

### Common Pitfalls to Avoid
- Overcrowding the visualization
- Using misleading scales
- Poor color choices
- Missing labels or titles
- 3D when 2D would suffice

### Choosing the Right Chart
- **Comparison**: Bar charts, Column charts
- **Distribution**: Histograms, Box plots, Violin plots
- **Composition**: Pie charts, Stacked bars, Treemaps
- **Relationship**: Scatter plots, Bubble charts
- **Trends**: Line charts, Area charts

## Exercises
1. Create a dashboard with multiple visualizations
2. Build an interactive visualization with Plotly
3. Design an effective visualization for a given dataset
4. Critique and improve existing visualizations

## Resources
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Plotly Documentation](https://plotly.com/python/)
- [The Visual Display of Quantitative Information by Edward Tufte](https://www.edwardtufte.com/tufte/books_vdqi)
