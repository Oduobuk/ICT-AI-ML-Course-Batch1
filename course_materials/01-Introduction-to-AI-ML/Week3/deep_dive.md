# Month 1, Week 3: Deep Dive into NumPy and Pandas

### **Objective:**

To move beyond introductory concepts and gain a practical, in-depth
understanding of the core mechanics of NumPy and Pandas. This guide will
focus on the "how" and "why" of common operations.

### **Reference Text:**

-   *Python for Data Analysis, 3rd Edition* by Wes McKinney. Chapters 4
    (NumPy) and 5 (Pandas) are essential reading.

## Part 1: NumPy Deep Dive

### **1.1 The NumPy** `ndarray`**: A Multidimensional Array Object**

-   **Homogeneous Data:** Unlike Python lists, all elements of a NumPy
    array must be of the same data type (e.g., all `int32` or all
    `float64`). This is the key to its performance.

-   **Attributes:** Every array has a `shape` (a tuple indicating the
    size of each dimension) and a `dtype` (an object describing the data
    type).

```{=html}
<!-- -->
```
-   import numpy as np
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        print(arr.shape)  # Output: (2, 3)
        print(arr.dtype)  # Output: int64

### **1.2 Universal Functions (ufuncs): Fast Element-Wise Operations**

A `ufunc` performs element-wise operations on data in `ndarrays`. This
is the core of "vectorization," which avoids slow Python `for` loops.

-   **Unary ufuncs:** Operate on a single array (e.g., `np.sqrt`,
    `np.exp`).

-   **Binary ufuncs:** Operate on two arrays (e.g., `np.add`,
    `np.maximum`).

```{=html}
<!-- -->
```
-   arr = np.arange(10)

        # Instead of this:
        # result = []
        # for x in arr:
        #     result.append(x * 2)

        # Do this (Vectorized):
        result = arr * 2

### **1.3 Array Indexing and Slicing**

This is more powerful than list slicing.

-   **Slicing:** Slices on NumPy arrays are *views* on the original
    array, not copies. Modifying a slice will modify the original array.

```{=html}
<!-- -->
```
-   arr = np.arange(10)
        arr_slice = arr[5:8]
        arr_slice[1] = 12345
        print(arr) # Output: [    0     1     2     3     4 12345     6     7     8     9]
        # To create a copy, you must be explicit: arr[5:8].copy()

```{=html}
<!-- -->
```
-   **Boolean Indexing:** Select data based on a condition. This is
    extremely powerful.

```{=html}
<!-- -->
```
-   data = np.random.randn(5, 4)
        # Select all rows where the value in the first column is negative
        negative_rows = data[data[:, 0] < 0]

## Part 2: Pandas Deep Dive

### **2.1 The DataFrame: The Ultimate Data Structure**

A DataFrame is a collection of Series, where each Series is a column.
All columns in a DataFrame share the same index.

### **2.2 Essential Functionality: Indexing, Selection, and Filtering**

This is the most critical area to master.

-   `loc` **(Label-based indexing):** Selects data based on index
    *labels*.

    -   `df.loc['row_label']`
    -   `df.loc[:, ['col1', 'col2']`

-   `iloc` **(Integer-position based indexing):** Selects data based on
    integer *position*.

    -   `df.iloc[0]` \# First row
    -   `df.iloc[:, 0:2]` \# First two columns

-   **Conditional Selection (Boolean Indexing):** Similar to NumPy, but
    used far more often.

```{=html}
<!-- -->
```
-   import pandas as pd
        df = pd.DataFrame(np.random.randn(5, 3), columns=['a', 'b', 'c'])

        # Select all rows where column 'a' is greater than 0
        positive_a = df[df['a'] > 0]

        # Select rows where column 'b' is > 0 AND column 'c' is < 0
        complex_filter = df[(df['b'] > 0) & (df['c'] < 0)]

### **2.3 Handling Missing Data (**`NaN`**)**

-   `isnull()`: Returns a boolean DataFrame indicating `True` for `NaN`
    values.
-   `dropna()`: Drops rows (or columns) with missing values.
-   `fillna(value)`: Fills `NaN` values with a specified value (e.g., 0,
    or the mean of the column `df['col'].mean()`).

### **2.4 GroupBy: Split-Apply-Combine**

This is the cornerstone of data aggregation.

1.  **Split:** The data is split into groups based on some criteria
    (e.g., values in a column).

2.  **Apply:** A function is applied to each group independently (e.g.,
    `sum()`, `mean()`, `count()`).

3.  **Combine:** The results of the function applications are combined
    into a result DataFrame.

-   # Example: Calculate the average value of column 'c' for each unique category in column 'a'
        # df.groupby('a')['c'].mean()

### **Summary of Deep Dive:**

Mastering NumPy's vectorized operations and slicing (especially boolean
indexing) is key to writing efficient code. For Pandas, proficiency in
using `loc`, `iloc`, and boolean filtering to select subsets of data is
non-negotiable. The `groupby` mechanic is the most powerful tool for
data analysis and summarization.
