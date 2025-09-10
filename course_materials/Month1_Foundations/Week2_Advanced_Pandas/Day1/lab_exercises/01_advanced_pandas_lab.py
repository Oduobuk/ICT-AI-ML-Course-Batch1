"""
Advanced Pandas Lab Exercise
"""
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create sample dataset
def create_sample_data():
    """Create a sample sales dataset."""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    products = ['A', 'B', 'C', 'D', 'E']
    categories = ['Electronics', 'Clothing', 'Home', 'Electronics', 'Clothing']
    
    data = {
        'date': np.random.choice(dates, 10000),
        'product': np.random.choice(products, 10000, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'category': np.random.choice(categories, 10000, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
        'quantity': np.random.randint(1, 10, 10000),
        'price': np.random.uniform(10, 1000, 10000).round(2)
    }
    
    df = pd.DataFrame(data)
    df['revenue'] = df['quantity'] * df['price']
    return df

# Exercise 1: Data Exploration
def explore_data(df):
    """
    Perform basic data exploration.
    
    Args:
        df: Input DataFrame
        
    Returns:
        dict: Dictionary containing exploration results
    """
    # 1. Display basic information about the dataset
    print("1. Dataset Info:")
    print("-" * 50)
    print(df.info())
    
    # 2. Display summary statistics
    print("\n2. Summary Statistics:")
    print("-" * 50)
    print(df.describe())
    
    # 3. Check for missing values
    print("\n3. Missing Values:")
    print("-" * 50)
    print(df.isnull().sum())
    
    # 4. Unique values in categorical columns
    print("\n4. Unique Values:")
    print("-" * 50)
    for col in ['product', 'category']:
        print(f"{col}: {df[col].nunique()} unique values")
    
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict()
    }

# Exercise 2: Advanced Filtering
def filter_data(df):
    """
    Filter data based on conditions.
    
    Args:
        df: Input DataFrame
        
    Returns:
        tuple: Filtered DataFrames
    """
    # 1. Filter for high-revenue transactions (> $500)
    high_revenue = df[df['revenue'] > 500]
    
    # 2. Filter for specific products using isin
    selected_products = df[df['product'].isin(['A', 'C', 'E'])]
    
    # 3. Filter using query method
    query_result = df.query('quantity > 5 and price < 100')
    
    # 4. Filter using between
    mid_range = df[df['price'].between(100, 500)]
    
    return high_revenue, selected_products, query_result, mid_range

# Exercise 3: GroupBy Operations
def groupby_analysis(df):
    """
    Perform groupby operations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame: Aggregated results
    """
    # 1. Group by category and calculate metrics
    category_stats = df.groupby('category').agg({
        'revenue': ['sum', 'mean', 'count'],
        'quantity': 'sum'
    })
    
    # 2. Group by multiple columns
    product_category_stats = df.groupby(['category', 'product']).agg({
        'revenue': ['sum', 'mean'],
        'quantity': ['sum', 'mean']
    })
    
    # 3. Pivot table
    pivot = pd.pivot_table(
        df,
        values='revenue',
        index='category',
        columns='product',
        aggfunc=['sum', 'mean'],
        fill_value=0
    )
    
    return category_stats, product_category_stats, pivot

# Exercise 4: Time Series Analysis
def time_series_analysis(df):
    """
    Perform time series analysis.
    
    Args:
        df: Input DataFrame with date column
        
    Returns:
        tuple: Time series DataFrames
    """
    # Convert date to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Set date as index
    ts_df = df.set_index('date')
    
    # Daily revenue
    daily_revenue = ts_df['revenue'].resample('D').sum()
    
    # Weekly summary
    weekly_stats = ts_df['revenue'].resample('W').agg(['sum', 'mean', 'count'])
    
    # Monthly revenue by category
    monthly_category = ts_df.groupby('category')['revenue'].resample('M').sum().unstack(level=0)
    
    return daily_revenue, weekly_stats, monthly_category

def main():
    # Create sample data
    print("Creating sample data...")
    df = create_sample_data()
    
    # Exercise 1: Data Exploration
    print("\n" + "="*50)
    print("EXERCISE 1: DATA EXPLORATION")
    print("="*50)
    explore_data(df)
    
    # Exercise 2: Advanced Filtering
    print("\n" + "="*50)
    print("EXERCISE 2: ADVANCED FILTERING")
    print("="*50)
    high_rev, sel_prod, query_res, mid_range = filter_data(df)
    print(f"High revenue transactions: {len(high_rev)}")
    print(f"Selected products transactions: {len(sel_prod)}")
    print(f"Query results: {len(query_res)}")
    print(f"Mid-range price items: {len(mid_range)}")
    
    # Exercise 3: GroupBy Operations
    print("\n" + "="*50)
    print("EXERCISE 3: GROUPBY OPERATIONS")
    print("="*50)
    cat_stats, prod_cat_stats, pivot_table = groupby_analysis(df)
    print("\nCategory Statistics:")
    print(cat_stats)
    
    # Exercise 4: Time Series Analysis
    print("\n" + "="*50)
    print("EXERCISE 4: TIME SERIES ANALYSIS")
    print("="*50)
    daily, weekly, monthly = time_series_analysis(df)
    print("\nWeekly Stats:")
    print(weekly.head())

if __name__ == "__main__":
    main()
