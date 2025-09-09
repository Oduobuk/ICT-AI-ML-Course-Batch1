# Month 1, Week 3: Exhaustive Deep Dive

### **Objective:**
To achieve a professional-level understanding of advanced data manipulation techniques, focusing on performance, complex indexing, and multi-level data organization, as detailed in *Python for Data Analysis, 3rd Edition*.

---

## Part 1: Advanced NumPy

### **1.1 Array Broadcasting**

Broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. It is a powerful mechanism that allows for efficient vectorized operations without making explicit copies of data.

**The Rules of Broadcasting:**
1.  If the arrays do not have the same rank, prepend the shape of the lower-rank array with 1s until both shapes have the same length.
2.  The two arrays are said to be *compatible* in a dimension if they have the same size in that dimension, or if one of the arrays has size 1 in that dimension.
3.  If compatible, the result of the operation is an array with a shape equal to the element-wise maximum of the two input array shapes.
4.  If not compatible, a `ValueError` is raised.

    ```python
    import numpy as np

    # Example: Add a 1D array to a 2D array
    arr = np.arange(12).reshape((4, 3)) # Shape (4, 3)
    row_mean = arr.mean(axis=1) # Shape (4,)

    # To subtract the mean from each row, reshape the mean array
    # arr's shape is (4, 3). row_mean's shape is (4,). 
    # To make them compatible, we reshape row_mean to (4, 1).
    # The (4, 1) array is then "broadcast" across the (4, 3) array.
    demeaned_arr = arr - row_mean.reshape((4, 1))
    ```

### **1.2 Advanced Indexing: `np.ix_`**

For constructing an indexer that allows you to select elements from a multi-dimensional array based on the cross-product of indices from different dimensions.

    ```python
    arr = np.arange(9).reshape((3, 3))
    # Select rows 0 and 2, and columns 1 and 2
    # This gives you elements (0,1), (0,2), (2,1), (2,2)
    subset = arr[np.ix_([0, 2], [1, 2])]
    ```

---

## Part 2: Advanced Pandas

### **2.1 Multi-level Indexing (Hierarchical Indexing)**

A `MultiIndex` allows you to have multiple index levels on an axis. It's a powerful way to work with higher-dimensional data in a lower-dimensional form (like a Series or DataFrame).

    ```python
    import pandas as pd
    data = pd.Series(np.random.randn(9),
                     index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'c'],
                            [1, 2, 3, 1, 3, 1, 2, 2, 3]])
    # The index is now a MultiIndex
    # data['b'] -> Selects all data with outer index 'b'
    # data[:, 2] -> Selects all data with inner index 2
    ```

### **2.2 Advanced GroupBy Operations: `transform` and `apply`**

While `agg` returns an aggregated result, `transform` and `apply` allow for more complex operations.

*   **`transform`:** Returns an object that is indexed the same as the one being grouped. It is used for broadcasting results of an aggregation back to the original data shape.

    ```python
    # Example: Subtract the group-wise mean from each value in the group
    # df.groupby('key')['data'].transform(lambda x: x - x.mean())
    ```

*   **`apply`:** The most general GroupBy method. It splits the object into pieces, invokes the passed function on each piece, and then attempts to stitch the results together. It offers maximum flexibility.

    ```python
    # Example: Select the top 2 rows from each group sorted by a column
    # def top(df, n=2, column='value'):
    #     return df.sort_values(by=column, ascending=False).head(n)
    # df.groupby('group_key').apply(top)
    ```

### **2.3 Combining Datasets: `merge`, `join`, and `concat`**

*   **`pd.concat`:** Stacks objects along an axis. Best for combining DataFrames with the same columns.
*   **`pd.merge`:** Connects rows in DataFrames based on one or more keys (like a database JOIN).
    *   **Types of joins:** `inner`, `outer`, `left`, `right`.
    *   Crucial for combining data from different sources.
*   **`df.join`:** A convenient method for combining data from two DataFrames using their indexes.

### **2.4 Performance with Categorical Data**

If a DataFrame column contains a small number of distinct values (e.g., status, group, gender), converting it to the `category` dtype can save memory and significantly speed up operations like `groupby`.

    ```python
    df['status'] = df['status'].astype('category')
    ```