# Week 2 Teaching Notes: Advanced Pandas and Data Visualization

## 📅 Weekly Overview

### Day 1: Advanced Pandas Techniques
- **Key Concepts**: MultiIndex, GroupBy operations, time series handling
- **Common Challenges**:
  - Understanding hierarchical indexing
  - Memory optimization with large datasets
  - Efficient data aggregation

### Day 2: Data Visualization
- **Key Concepts**:
  - Matplotlib customization
  - Statistical visualization with Seaborn
  - Interactive visualizations with Plotly
- **Common Challenges**:
  - Choosing the right plot type
  - Handling large datasets in visualizations
  - Customizing plot aesthetics

### Day 3: Practical Applications
- **Key Concepts**:
  - End-to-end data analysis pipeline
  - Feature engineering
  - Model building and evaluation
- **Common Challenges**:
  - Data cleaning decisions
  - Feature selection
  - Interpreting model results

## 🎯 Learning Objectives

By the end of this week, students should be able to:
1. Perform advanced data manipulation with Pandas
2. Create effective visualizations for different data types
3. Build a complete data analysis pipeline
4. Optimize code for better performance
5. Create interactive dashboards

## 🛠️ Technical Setup

### Required Packages
- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- plotly >= 5.0.0
- scikit-learn >= 0.24.0

### Dataset
- Titanic dataset (included in seaborn)
- Netflix dataset (for assignment)
- Sample sales data (for exercises)

## 📝 Teaching Strategies

### Active Learning
- Live coding demonstrations
- Pair programming exercises
- Code review sessions
- Debugging challenges

### Assessment Methods
- In-class exercises
- Weekly quiz
- Practical assignment
- Code review

## 🧩 Common Pitfalls and Solutions

### 1. Memory Usage with Large Datasets
- **Issue**: Students may struggle with memory errors
- **Solution**:
  - Use `dtype` parameter when reading data
  - Process data in chunks
  - Use category dtype for low-cardinality strings

### 2. Plot Customization
- **Issue**: Students often create hard-to-read visualizations
- **Solution**:
  - Emphasize the importance of labels and titles
  - Teach color theory basics
  - Show examples of good vs. bad visualizations

### 3. Performance Optimization
- **Issue**: Inefficient code that takes too long to run
- **Solution**:
  - Teach vectorized operations
  - Show timing comparisons
  - Introduce profiling tools

## 🎓 Teaching Tips

### For Day 1 (Advanced Pandas)
- Start with simple examples before moving to complex ones
- Use real-world analogies for MultiIndex
- Show the equivalence between different methods (e.g., groupby vs. pivot_table)

### For Day 2 (Data Visualization)
- Begin with the "why" before the "how"
- Show multiple ways to create the same plot
- Emphasize the importance of storytelling with data

### For Day 3 (Practical Applications)
- Break down the analysis into clear steps
- Demonstrate the iterative nature of data analysis
- Show how to handle unexpected data issues

## 🔍 Debugging Common Errors

### 1. SettingWithCopyWarning
- **Cause**: Chained indexing
- **Fix**: Use `.loc[]` for assignment

### 2. MemoryError
- **Cause**: Large dataset in memory
- **Fix**: Use chunking or reduce memory usage

### 3. Plot Not Displaying
- **Cause**: Missing `plt.show()` or incorrect backend
- **Fix**: Ensure proper import and display commands

## 📚 Additional Resources

### Recommended Reading
- [Python for Data Analysis](https://wesmckinney.com/book/) by Wes McKinney
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) by Jake VanderPlas
- [Storytelling with Data](https://www.storytellingwithdata.com/) by Cole Nussbaumer Knaflic

### Online Courses
- [Data Visualization with Python](https://www.coursera.org/learn/python-for-data-visualization)
- [Advanced Pandas](https://www.datacamp.com/courses/advanced-pandas)

### Cheat Sheets
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Matplotlib Cheat Sheet](https://matplotlib.org/cheatsheets/)
- [Seaborn Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Seaborn_Cheat_Sheet.pdf)

## 🏆 Assessment Guidelines

### Grading Rubric for Week 2 Assignment

| Criteria | Excellent (9-10) | Good (7-8) | Needs Improvement (0-6) |
|----------|------------------|------------|-------------------------|
| **Code Quality** | Clean, efficient, well-documented code | Mostly clean with minor issues | Hard to follow, lacks documentation |
| **Data Cleaning** | Thorough and well-justified | Adequate but could be improved | Incomplete or inappropriate methods |
| **Analysis** | Insightful, thorough analysis | Basic analysis with some insights | Superficial or incorrect analysis |
| **Visualizations** | Clear, informative, well-formatted | Mostly clear with minor issues | Unclear or inappropriate visualizations |
| **Documentation** | Excellent explanations and comments | Adequate documentation | Lacking documentation |

### Common Feedback Points
1. **Code Organization**
   - Use functions for reusable code
   - Include docstrings and comments
   - Follow PEP 8 style guide

2. **Data Analysis**
   - Justify data cleaning decisions
   - Consider edge cases
   - Validate assumptions

3. **Visualizations**
   - Label axes and include units
   - Use appropriate chart types
   - Consider color blindness

## 🚀 Extension Activities

### For Advanced Students
1. Create an interactive dashboard using Dash or Streamlit
2. Implement a machine learning model on the dataset
3. Optimize the code for better performance

### For Students Needing Extra Help
1. Provide additional practice exercises
2. Offer one-on-one code reviews
3. Suggest relevant tutorials or documentation
