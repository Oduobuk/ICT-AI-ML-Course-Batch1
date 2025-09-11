# Teaching Guide: Advanced Pandas - Day 1

## 🎯 Learning Objectives
By the end of this session, students should be able to:
1. Perform advanced data selection and filtering in Pandas
2. Work with MultiIndex DataFrames
3. Use GroupBy operations for data aggregation
4. Handle time series data effectively
5. Optimize Pandas code for better performance

## ⏰ Time Allocation (3-hour session)

### 1. Introduction (15 min)
- **Objective**: Set the stage for the day's topics
- **Key Points**:
  - Why advanced Pandas skills are essential for data analysis
  - Real-world applications of today's topics
  - Brief overview of what we'll cover

### 2. Advanced Data Selection (30 min)

#### Concepts to Cover:
- Boolean indexing
- The `query()` method
- `where()` vs `mask()`
- `isin()`, `between()`, and other filtering methods

#### Teaching Approach:
1. **Demonstration** (10 min):
   ```python
   # Boolean indexing
   df[df['Age'] > 30]
   
   # Using query()
   df.query('Age > 30 and Pclass == 1')
   
   # where() vs mask()
   df['Age'].where(df['Age'] > 30, 30)  # Replace ages <= 30 with 30
   df['Age'].mask(df['Age'] > 30, 30)    # Replace ages > 30 with 30
   ```

2. **Common Pitfalls**:
   - Chained indexing issues
   - Using `&` vs `and` in conditions
   - Performance implications of different selection methods

### 3. MultiIndex and Advanced Indexing (45 min)

#### Concepts to Cover:
- Creating MultiIndex DataFrames
- Indexing and slicing with `.loc[]`
- Stacking and unstacking
- Cross-sections with `.xs()`

#### Teaching Approach:
1. **Interactive Exercise** (15 min):
   ```python
   # Create a MultiIndex DataFrame
   index = pd.MultiIndex.from_tuples([
       ('Group A', 'Week 1'), 
       ('Group A', 'Week 2'),
       ('Group B', 'Week 1'),
       ('Group B', 'Week 2')
   ])
   
   df = pd.DataFrame({
       'Value': [10, 15, 12, 18],
       'Count': [100, 150, 120, 180]
   }, index=index)
   
   # Slicing examples
   df.loc['Group A']
   df.loc[('Group A', 'Week 1'):('Group B', 'Week 1')]
   ```

2. **Real-world Analogy**:
   - Compare MultiIndex to Excel pivot tables
   - Think of it as a tree structure where each level adds more specificity

### 4. GroupBy Operations (45 min)

#### Concepts to Cover:
- Split-apply-combine pattern
- Aggregation methods (mean, sum, count, etc.)
- Transformation vs. filtration
- The `agg()` method for multiple operations

#### Teaching Approach:
1. **Live Coding** (20 min):
   ```python
   # Basic groupby
   df.groupby('Pclass')['Fare'].mean()
   
   # Multiple aggregations
   df.groupby('Pclass').agg({
       'Age': ['mean', 'min', 'max'],
       'Fare': ['sum', 'count']
   })
   
   # Custom aggregation
   def range_func(x):
       return x.max() - x.min()
   
   df.groupby('Pclass')['Age'].agg(range_func)
   ```

2. **Common Mistakes**:
   - Forgetting to reset_index()
   - Misusing transform vs apply
   - Performance considerations with large datasets

### 5. Time Series Handling (30 min)

#### Concepts to Cover:
- DateTimeIndex
- Resampling
- Shifting and lagging
- Rolling windows

#### Teaching Approach:
```python
# Create time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'value': np.random.randn(100).cumsum()
}, index=dates)

# Resampling
df.resample('W').mean()

# Rolling windows
df.rolling(window=7).mean()
```

### 6. Performance Optimization (15 min)

#### Key Points:
- Vectorized operations vs loops
- Using appropriate dtypes
- Memory usage optimization
- The `eval()` and `query()` methods for large datasets

### 7. Wrap-up and Q&A (15 min)

#### Discussion Questions:
1. When would you choose to use MultiIndex instead of multiple columns?
2. How would you optimize a slow GroupBy operation?
3. What are some real-world scenarios where time series resampling is useful?

## 🧑‍🏫 Teaching Tips

### Engagement Strategies:
- Start with a real-world dataset that demonstrates why these techniques matter
- Use the "I do, we do, you do" approach for coding examples
- Include quick comprehension checks throughout the session

### Common Student Questions:
1. **Q**: When should I use `query()` vs boolean indexing?
   **A**: Use `query()` for better readability with complex conditions, especially when working with column names that have spaces.

2. **Q**: How do I know if I should use MultiIndex?
   **A**: When you find yourself creating complex column names like 'sales_2022_Q1', it's probably time for a MultiIndex.

3. **Q**: What's the most efficient way to aggregate data?
   **A**: Built-in aggregation methods (mean, sum, etc.) are fastest. For custom operations, test performance with `%timeit`.

## 📝 Assessment Ideas

### In-class Activity:
- Have students work in pairs to analyze a dataset using the techniques covered
- Each pair presents one insight they discovered using these methods

### Homework:
- Provide a dataset with missing values and ask students to clean and analyze it
- Include tasks that require combining multiple techniques (e.g., groupby with time series resampling)

## 📚 Additional Resources
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Python Data Science Handbook - Chapter 3](https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html)
- [Modern Pandas](https://tomaugspurger.github.io/modern-1-intro.html)
