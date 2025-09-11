# Teaching Guide: Data Visualization - Day 2

## 🎯 Learning Objectives
By the end of this session, students should be able to:
1. Create effective visualizations using Matplotlib and Seaborn
2. Choose appropriate chart types for different data types
3. Customize plots for better readability and impact
4. Create interactive visualizations with Plotly
5. Apply visualization best practices

## ⏰ Time Allocation (3-hour session)

### 1. Introduction (15 min)
- **Objective**: Understand the importance of data visualization
- **Key Points**:
  - Why visualization matters in data analysis
  - The visualization pipeline
  - Overview of Python's visualization ecosystem

### 2. Matplotlib Fundamentals (30 min)

#### Concepts to Cover:
- Figure and axes objects
- Basic plot types (line, bar, scatter, histogram)
- Customizing plots (labels, titles, legends)
- Subplots and layouts

#### Teaching Approach:
1. **Live Demonstration** (15 min):
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   
   # Basic line plot
   x = np.linspace(0, 10, 100)
   y = np.sin(x)
   
   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(x, y, label='sin(x)', color='blue', linestyle='--')
   ax.set_title('Sine Wave')
   ax.set_xlabel('X-axis')
   ax.set_ylabel('Y-axis')
   ax.legend()
   plt.show()
   ```

2. **Common Pitfalls**:
   - Overcrowded plots
   - Poor color choices
   - Missing labels or context

### 3. Statistical Visualization with Seaborn (45 min)

#### Concepts to Cover:
- Relationship plots (scatter, line, relplot)
- Distribution plots (histogram, KDE, distplot)
- Categorical plots (box, violin, barplot)
- Matrix plots (heatmap, pairplot)

#### Teaching Approach:
1. **Interactive Exercise** (20 min):
   ```python
   import seaborn as sns
   
   # Load example dataset
   tips = sns.load_dataset('tips')
   
   # Create a figure with multiple subplots
   fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   
   # Scatter plot
   sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time', ax=axes[0,0])
   
   # Box plot
   sns.boxplot(data=tips, x='day', y='total_bill', hue='sex', ax=axes[0,1])
   
   # KDE plot
   sns.kdeplot(data=tips, x='total_bill', hue='time', ax=axes[1,0])
   
   # Heatmap
   correlation = tips.corr()
   sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=axes[1,1])
   
   plt.tight_layout()
   plt.show()
   ```

2. **Best Practices**:
   - Choosing the right plot for your data
   - Effective use of color and style
   - Labeling and annotations

### 4. Interactive Visualizations with Plotly (45 min)

#### Concepts to Cover:
- Introduction to Plotly Express
- Creating interactive plots
- Customizing interactivity
- Combining multiple plots in Dash

#### Teaching Approach:
1. **Live Coding** (20 min):
   ```python
   import plotly.express as px
   
   # Create an interactive scatter plot
   fig = px.scatter(tips, x='total_bill', y='tip', color='time',
                   title='Total Bill vs Tip by Time',
                   labels={'total_bill': 'Total Bill ($)', 'tip': 'Tip ($)'},
                   hover_data=['day', 'size'])
   
   # Add trendline
   fig.update_traces(
       marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
       selector=dict(mode='markers')
   )
   
   # Customize layout
   fig.update_layout(
       xaxis_title='Total Bill ($)',
       yaxis_title='Tip ($)',
       legend_title='Time of Day',
       font=dict(family='Arial', size=12, color='black')
   )
   
   fig.show()
   ```

2. **Hands-on Exercise**:
   - Have students create an interactive visualization with tooltips
   - Add dropdown menus or sliders for filtering

### 5. Visualization Best Practices (30 min)

#### Key Points:
- The data-ink ratio
- Color theory for data visualization
- Accessibility considerations
- Common visualization pitfalls to avoid

#### Teaching Approach:
- Show examples of good vs. bad visualizations
- Discuss the importance of context and storytelling
- Demonstrate how to make visualizations accessible

### 6. Case Study: Telling a Story with Data (30 min)

#### Activity:
- Present a dataset and guide students through creating a visualization that tells a story
- Focus on:
  - Identifying the key message
  - Choosing appropriate visual encodings
  - Adding context and annotations
  - Ensuring clarity and impact

### 7. Wrap-up and Q&A (15 min)

#### Discussion Questions:
1. How does the choice of visualization affect the interpretation of data?
2. What are some strategies for visualizing high-dimensional data?
3. How can we make visualizations more accessible to people with color vision deficiencies?

## 🧑‍🏫 Teaching Tips

### Engagement Strategies:
- Start with a "before and after" visualization example
- Use real-world datasets that students can relate to
- Encourage students to critique visualizations

### Common Student Questions:
1. **Q**: How do I choose between a bar chart and a line chart?
   **A**: Use bar charts for categorical data and line charts for continuous data over time.

2. **Q**: What's the best way to handle too many categories in a pie chart?
   **A**: Consider using a bar chart instead, or group smaller categories into 'Other'.

3. **Q**: How can I make my visualizations more professional-looking?
   **A**: Focus on consistency, proper labeling, and removing unnecessary elements.

## 📝 Assessment Ideas

### In-class Activity:
- Give students a messy visualization and have them improve it
- Have students present their visualizations and explain their design choices

### Homework:
- Create a dashboard with multiple linked visualizations
- Write a short report explaining the insights gained from the visualizations

## 📚 Additional Resources
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Plotly Documentation](https://plotly.com/python/)
- [From Data to Viz](https://www.data-to-viz.com/)
- [The Visual Display of Quantitative Information](https://www.edwardtufte.com/tufte/books_vdqi) by Edward Tufte
