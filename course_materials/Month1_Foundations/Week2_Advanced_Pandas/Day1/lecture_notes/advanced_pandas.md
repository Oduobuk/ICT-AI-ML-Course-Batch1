# Advanced Pandas Techniques

## Table of Contents
1. [Advanced Data Selection](#advanced-data-selection)
2. [MultiIndex and Advanced Indexing](#multiindex-and-advanced-indexing)
3. [GroupBy Operations](#groupby-operations)
4. [Time Series Handling](#time-series-handling)
5. [Performance Optimization](#performance-optimization)

## Advanced Data Selection

### Boolean Indexing
```python
# Select rows where Age > 30
df[df['Age'] > 30]

# Multiple conditions
mask = (df['Age'] > 30) & (df['Pclass'] == 1)
df[mask]
```

### Query Method
```python
# Equivalent to boolean indexing but more readable
df.query('Age > 30 and Pclass == 1')
```

### Where and Mask
```python
# Replace values where condition is False
df['Age'].where(df['Age'] > 30, 30)  # Replace ages <= 30 with 30

# Opposite of where
df['Age'].mask(df['Age'] > 30, 30)  # Replace ages > 30 with 30
```

## MultiIndex and Advanced Indexing

### Creating MultiIndex
```python
# From columns
df.set_index(['Pclass', 'Sex'])

# From tuples
index = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2)])
```

### Slicing with MultiIndex
```python
# Select all rows where first level is 'A'
df.xs('A', level=0)

# Cross-section with multiple levels
df.xs(('A', 1), level=[0, 1])
```

## GroupBy Operations

### Basic Grouping
```python
# Group by single column
grouped = df.groupby('Pclass')

# Multiple aggregations
grouped.agg({
    'Age': ['mean', 'min', 'max'],
    'Fare': ['sum', 'mean']
})
```

### Custom Aggregation
```python
def range_func(x):
    return x.max() - x.min()

df.groupby('Pclass')['Age'].agg(['mean', 'std', range_func])
```

### Transform and Filter
```python
# Create new column with group means
df['Age_Mean_By_Class'] = df.groupby('Pclass')['Age'].transform('mean')

# Filter groups
df.groupby('Pclass').filter(lambda x: len(x) > 100)  # Only keep classes with >100 passengers
```

## Time Series Handling

### Date Ranges
```python
# Create date range
pd.date_range('2023-01-01', periods=5, freq='D')
```

### Resampling
```python
# Convert daily data to monthly
monthly = df.resample('M').mean()

# Custom resampling
df.resample('W').agg({'A': 'sum', 'B': 'mean'})
```

## Performance Optimization

### Vectorized Operations
```python
# Slow: Using apply with Python function
df['new_col'] = df['col'].apply(lambda x: x * 2)

# Fast: Vectorized operation
df['new_col'] = df['col'] * 2
```

### Categorical Data
```python
# Convert to category type for memory efficiency
df['category'] = df['category'].astype('category')
```

### Chunk Processing
```python
# Process large files in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process(chunk)
```

## Best Practices
1. Use vectorized operations instead of loops
2. Minimize copies of DataFrames
3. Use appropriate data types
4. Leverage built-in methods
5. Use `inplace=True` for memory efficiency

## Exercises
1. Load a dataset and perform advanced filtering
2. Create and manipulate a MultiIndex DataFrame
3. Perform complex groupby operations
4. Optimize a slow pandas operation

## Resources
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Python Data Science Handbook - Chapter 3](https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
