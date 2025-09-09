# Month 1, Week 3: Numerical and Data-Driven Programming

## Lesson 1: Introduction to NumPy and Pandas

### **Objective:**
To introduce the two most critical Python libraries for data science: NumPy for numerical operations and Pandas for data manipulation. Students will understand why these libraries are used and learn to perform basic operations.

### **Key Concepts:**

1.  **Why Not Just Use Python Lists and Dictionaries?**
    *   **Performance:** Standard Python is slow for numerical computations on large datasets.
    *   **Functionality:** Python's built-in types lack the advanced mathematical and data manipulation functions needed for data analysis.

2.  **Introduction to NumPy (Numerical Python)**
    *   **The NumPy Array:** The core of NumPy is the `ndarray` (n-dimensional array) object.
        *   It is a fast, flexible container for large datasets in Python.
        *   Arrays enable you to perform mathematical operations on whole blocks of data at once (vectorization).
    *   **Creating Arrays:**
        *   `np.array()`: Convert a Python list into a NumPy array.
        *   `np.arange()`: Like the Python `range` function but returns an array.
        *   `np.zeros()`, `np.ones()`: Create arrays of zeros or ones.
    *   **Key Advantage:** Vectorized operations are significantly faster than iterating with a `for` loop.

3.  **Introduction to Pandas**
    *   **Purpose:** Built on top of NumPy, Pandas provides high-level data structures and functions designed to make working with structured or tabular data intuitive and flexible.
    *   **Core Data Structures:**
        *   **a) Series:** A one-dimensional labeled array, like a single column in a spreadsheet. Each element has an index.
        *   **b) DataFrame:** A two-dimensional labeled data structure with columns of potentially different types, like a full spreadsheet or a SQL table.

4.  **Creating Pandas Objects**
    *   **Series:** `pd.Series(data, index=index)`
    *   **DataFrame:** `pd.DataFrame(data)` where `data` is typically a dictionary of lists or a NumPy array.

5.  **First Steps with Pandas**
    *   **Loading Data:** The most common way to create a DataFrame is by loading a file: `pd.read_csv('file.csv')`.
    *   **Inspecting Data:**
        *   `.head()`: View the first few rows.
        *   `.info()`: Get a concise summary of the DataFrame (column types, non-null values).
        *   `.describe()`: Generate descriptive statistics.

### **Summary:**
NumPy provides the low-level, high-performance array objects that are the foundation of data science in Python. Pandas builds on NumPy to provide the powerful DataFrame object, which is the primary tool for data cleaning, transformation, and analysis. This week, we will master the fundamentals of both.
