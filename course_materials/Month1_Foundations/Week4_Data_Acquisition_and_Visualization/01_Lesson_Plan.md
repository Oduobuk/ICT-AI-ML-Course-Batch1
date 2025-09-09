# Month 1, Week 4: Data Acquisition and Visualization

## Lesson 1: Data Acquisition and Introduction to SQL

### **Objective:**
To understand the various ways data is acquired for analysis, with a primary focus on querying relational databases using SQL and loading data from standard file formats like CSV.

### **Key Concepts:**

1.  **Where Does Data Come From?**
    *   **Files:** CSV, Excel, JSON, XML, Parquet.
    *   **Databases:** Relational (SQL) and NoSQL databases.
    *   **APIs:** Web services that provide data on request.
    *   **Web Scraping:** Extracting data from websites (a more advanced topic).

2.  **Introduction to SQL (Structured Query Language)**
    *   **What is it?** SQL is the standard language for interacting with relational databases (e.g., PostgreSQL, MySQL, SQL Server).
    *   **Why is it essential?** A vast amount of the world's corporate data resides in SQL databases. Being able to retrieve your own data is a fundamental skill.
    *   **Core SQL Commands:**
        *   `SELECT`: Used to query the database and retrieve data.
        *   `FROM`: Specifies the table to retrieve the data from.
        *   `WHERE`: Filters records based on a condition.

    *   **Simple Query Structure:**
        ```sql
        SELECT column1, column2
        FROM table_name
        WHERE condition;
        ```

    *   **Example:** Get the names and salaries of all employees in the 'engineering' department.
        ```sql
        SELECT name, salary
        FROM employees
        WHERE department = 'engineering';
        ```

3.  **Joining Tables**
    *   Data is often stored across multiple tables.
    *   `JOIN`: Combines rows from two or more tables based on a related column between them.
    *   **Example:** Get employee names and their department names.
        ```sql
        SELECT employees.name, departments.department_name
        FROM employees
        JOIN departments ON employees.department_id = departments.id;
        ```

4.  **Loading Data with Pandas**
    *   Pandas provides easy-to-use functions to read data from various sources into a DataFrame.
    *   **From CSV:** `pd.read_csv('path/to/your/file.csv')`
    *   **From Excel:** `pd.read_excel('path/to/your/file.xlsx', sheet_name='Sheet1')`
    *   **From SQL:** Pandas can directly query a database and load the results into a DataFrame.
        ```python
        import pandas as pd
        from sqlalchemy import create_engine

        # Connection string to your database
        engine = create_engine('postgresql://user:password@host:port/database')
        query = "SELECT * FROM employees;"

        df = pd.read_sql(query, engine)
        ```

### **Summary:**
Data acquisition is the first step in any data science project. SQL is the most important language for retrieving data from databases. Pandas provides a powerful and convenient toolkit for loading data from both files and databases, creating the DataFrames we will use for the rest of our analysis.
