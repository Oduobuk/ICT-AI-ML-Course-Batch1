# Week 2 Quiz: Advanced Pandas and Data Visualization

## Instructions
- **Time Limit:** 45 minutes
- **Total Points:** 100
- **Attempts Allowed:** 2
- **Due:** End of Week 2

## Multiple Choice Questions (5 points each)

1. **Which of the following methods is most efficient for applying a function to each element in a pandas Series?**
   A) Using `apply()` with a lambda function  
   B) Using `applymap()`  
   C) Using vectorized operations  
   D) Using `iterrows()`
   
   **Answer:** C

2. **What is the purpose of the `groupby()` function in pandas?**
   A) To sort the DataFrame by a specific column  
   B) To split the data into groups based on some criteria  
   C) To remove duplicate rows from the DataFrame  
   D) To merge two DataFrames together
   
   **Answer:** B

3. **Which of the following is NOT a valid way to handle missing data in pandas?**
   A) `df.dropna()`  
   B) `df.fillna(value)`  
   C) `df.remove_na()`  
   D) `df.interpolate()`
   
   **Answer:** C

4. **When creating a line plot with Matplotlib, which parameter would you use to change the line style to dashed?**
   A) `line_style='--'`  
   B) `style='dashed'`  
   C) `linestyle='--'`  
   D) `line_type='dashed'`
   
   **Answer:** C

5. **What is the main advantage of using Seaborn over Matplotlib?**
   A) It's faster for large datasets  
   B) It provides a higher-level interface for statistical graphics  
   C) It supports 3D visualizations  
   D) It can create interactive plots
   
   **Answer:** B

## Short Answer Questions (15 points each)

6. **Explain the difference between `loc` and `iloc` in pandas. Provide an example of when you would use each.**

   **Answer:**  
   `loc` is label-based indexing, while `iloc` is integer-location based indexing. 
   - Use `loc` when you want to select data by label: `df.loc[df['column'] > 5]`
   - Use `iloc` when you want to select data by position: `df.iloc[0:5, 2:4]`

7. **Describe how you would create a pivot table in pandas and what information it provides.**

   **Answer:**  
   A pivot table is created using `pd.pivot_table()` and provides a way to summarize and analyze data in a DataFrame. It allows you to group data by one or more columns and apply aggregations. For example:
   ```python
   pd.pivot_table(df, values='sales', index='region', 
                 columns='month', aggfunc='sum')
   ```
   This would show total sales by region for each month.

## Coding Exercise (30 points)

8. **Complete the following function that takes a DataFrame and returns a new DataFrame with the following modifications:**
   - Remove any rows with missing values in the 'age' column
   - Create a new column 'age_group' that categorizes ages into 'Child' (<18), 'Adult' (18-65), and 'Senior' (65+)
   - Calculate the average 'income' for each age group
   - Return a DataFrame with two columns: 'age_group' and 'avg_income'

```python
def analyze_income_by_age(df):
    """
    Analyze income by age groups in the given DataFrame.
    
    Args:
        df: DataFrame with 'age' and 'income' columns
        
    Returns:
        DataFrame with average income by age group
    """
    # Remove rows with missing age values
    df_clean = df.dropna(subset=['age']).copy()
    
    # Create age groups
    bins = [0, 18, 65, 120]
    labels = ['Child', 'Adult', 'Senior']
    df_clean['age_group'] = pd.cut(df_clean['age'], bins=bins, labels=labels)
    
    # Calculate average income by age group
    result = df_clean.groupby('age_group')['income'].mean().reset_index()
    result.columns = ['age_group', 'avg_income']
    
    return result
```

## Extra Credit (10 points)

9. **Explain one way to optimize the performance of a pandas operation on a large dataset.**

   **Answer:**  
   One way to optimize performance is to use vectorized operations instead of loops. For example, instead of using `apply()` with a custom function on each row, use built-in pandas methods that operate on entire columns at once. Other optimizations include using appropriate data types (e.g., `category` for low-cardinality strings), using `read_csv()` with `dtype` parameter to specify column types, and using `chunksize` for very large files that don't fit in memory.

## Submission Instructions
1. Save this file as `week2_quiz_<your_name>.md`
2. Fill in your answers directly in this document
3. Submit via the course learning management system by the due date
