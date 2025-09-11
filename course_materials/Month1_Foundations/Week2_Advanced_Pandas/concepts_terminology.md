# Week 2: Advanced Pandas and Data Visualization - Concepts and Terminology

## Core Concepts

### Day 1: Advanced Pandas Techniques

#### Key Concepts
1. **MultiIndex**
   - Hierarchical indexing for higher-dimensional data
   - Enables complex data aggregation and reshaping

2. **GroupBy Operations**
   - Split-apply-combine pattern
   - Aggregation, transformation, and filtering

3. **Time Series Handling**
   - DateTime indexing and operations
   - Resampling and frequency conversion
   - Time zone handling

4. **Performance Optimization**
   - Vectorized operations
   - Memory usage optimization
   - Efficient data types

### Day 2: Data Visualization

#### Key Concepts
1. **Visual Encoding**
   - Position, length, angle, area, color
   - Choosing the right chart type

2. **Statistical Visualization**
   - Distribution plots
   - Relationship plots
   - Categorical plots

3. **Interactive Visualization**
   - Event handling
   - Linked interactions
   - Tooltips and annotations

### Day 3: Practical Applications

#### Key Concepts
1. **Data Analysis Pipeline**
   - Data loading and inspection
   - Cleaning and preprocessing
   - Feature engineering

2. **Exploratory Data Analysis (EDA)**
   - Univariate analysis
   - Bivariate analysis
   - Multivariate analysis

3. **Dashboard Creation**
   - Layout design
   - Interactive components
   - Deployment considerations

## Technical Terminology

### Pandas-Specific Terms
- **DataFrame**: 2D labeled data structure with columns of potentially different types
- **Series**: 1D labeled array capable of holding any data type
- **Index**: Immutable sequence used for indexing and alignment
- **GroupBy**: Mechanism for grouping data by some criteria
- **Pivot Table**: Data summarization tool

### Visualization Terms
- **Aesthetic**: Visual property of an object in a plot (color, size, shape)
- **Facet**: Subset of data displayed in a separate panel
- **Geom**: Geometric object that represents data (point, line, bar)
- **Theme**: Non-data components of a plot (background, grid lines)

## Technology Stack

### Core Libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical visualization
- **Plotly**: Interactive visualization
- **Dash**: Web-based dashboards

### Recommended Tools
- Jupyter Notebook/Lab for interactive development
- VS Code with Python extension
- Git for version control
- Conda for environment management

## Learning Resources

### Documentation
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Plotly Python Documentation](https://plotly.com/python/)

### Recommended Reading
- "Python for Data Analysis" by Wes McKinney
- "Data Visualization with Python and JavaScript" by Kyran Dale
- "Storytelling with Data" by Cole Nussbaumer Knaflic

## Common Pitfalls and Solutions

### Data Cleaning
- **Issue**: Inconsistent date formats  
  **Solution**: Use `pd.to_datetime()` with format specification

- **Issue**: Missing values  
  **Solution**: Understand the nature of missingness before imputation

### Visualization
- **Issue**: Overplotting  
  **Solution**: Use transparency, jitter, or 2D histograms

- **Issue**: Misleading axis scaling  
  **Solution**: Always start numerical axes at zero for bar charts

## Assessment Criteria

### Coding Exercises
- Correctness of implementation
- Code efficiency
- Documentation and comments

### Visualizations
- Appropriate chart type selection
- Clear labeling and titles
- Effective use of color and annotations

### Analysis
- Logical flow of analysis
- Appropriate statistical methods
- Clear interpretation of results
