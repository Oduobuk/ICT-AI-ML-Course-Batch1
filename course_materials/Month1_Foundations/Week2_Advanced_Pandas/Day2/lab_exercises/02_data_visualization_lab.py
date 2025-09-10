"""
Data Visualization Lab Exercise
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set style for better-looking plots
plt.style.use('seaborn')
sns.set_palette('husl')

# Exercise 1: Load and Explore Data
def load_explore_data():
    """
    Load and explore the sample dataset.
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    # Load sample dataset
    df = sns.load_dataset('titanic')
    
    # Display basic information
    print("Dataset Info:")
    print("-" * 50)
    print(df.info())
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print(df.describe(include='all'))
    
    return df

# Exercise 2: Basic Plots with Matplotlib
def basic_plots(df):
    """
    Create basic plots using Matplotlib.
    
    Args:
        df: Input DataFrame
    """
    # 1. Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Age distribution
    axes[0, 0].hist(df['age'].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Count')
    
    # Plot 2: Fare distribution by class
    for cls in sorted(df['class'].unique()):
        subset = df[df['class'] == cls]['fare']
        axes[0, 1].hist(subset, alpha=0.5, label=cls, bins=20)
    axes[0, 1].set_title('Fare Distribution by Class')
    axes[0, 1].set_xlabel('Fare')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    
    # Plot 3: Survival rate by class
    survival_rate = df.groupby('class')['survived'].mean()
    axes[1, 0].bar(survival_rate.index, survival_rate.values, color='lightgreen')
    axes[1, 0].set_title('Survival Rate by Class')
    axes[1, 0].set_ylabel('Survival Rate')
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 4: Scatter plot of age vs. fare
    axes[1, 1].scatter(df['age'], df['fare'], alpha=0.5, c=df['survived'], cmap='viridis')
    axes[1, 1].set_title('Age vs. Fare (Color by Survival)')
    axes[1, 1].set_xlabel('Age')
    axes[1, 1].set_ylabel('Fare')
    
    plt.tight_layout()
    plt.show()

# Exercise 3: Advanced Plots with Seaborn
def advanced_plots(df):
    """
    Create advanced statistical plots using Seaborn.
    
    Args:
        df: Input DataFrame
    """
    # Set up the figure with a grid of plots
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Box plot of age by class and survival
    plt.subplot(2, 2, 1)
    sns.boxplot(x='class', y='age', hue='survived', data=df)
    plt.title('Age Distribution by Class and Survival')
    
    # Plot 2: Violin plot of fare by class and survival
    plt.subplot(2, 2, 2)
    sns.violinplot(x='class', y='fare', hue='survived', split=True, data=df)
    plt.ylim(0, 200)  # Limit y-axis for better visualization
    plt.title('Fare Distribution by Class and Survival')
    
    # Plot 3: Heatmap of correlation matrix
    plt.subplot(2, 2, 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    
    # Plot 4: Pair plot of numerical variables
    plt.subplot(2, 2, 4)
    # For demonstration, we'll use a subset of the data for better performance
    sns.pairplot(df[['age', 'fare', 'pclass', 'survived']].dropna(), 
                 hue='survived', 
                 plot_kws={'alpha': 0.6})
    plt.suptitle('Pair Plot of Numerical Variables')
    
    plt.tight_layout()
    plt.show()

# Exercise 4: Interactive Plots with Plotly
def interactive_plots(df):
    """
    Create interactive plots using Plotly.
    
    Args:
        df: Input DataFrame
    """
    # Interactive scatter plot
    fig1 = px.scatter(
        df,
        x='age',
        y='fare',
        color='survived',
        size='pclass',
        hover_data=['class', 'sex', 'embark_town'],
        title='Interactive Scatter Plot: Age vs. Fare',
        labels={'age': 'Age', 'fare': 'Fare ($)', 'survived': 'Survived'}
    )
    
    # Interactive box plot
    fig2 = px.box(
        df,
        x='class',
        y='age',
        color='survived',
        title='Interactive Box Plot: Age by Class and Survival',
        labels={'class': 'Class', 'age': 'Age', 'survived': 'Survived'}
    )
    
    # Interactive parallel coordinates plot
    fig3 = px.parallel_categories(
        df,
        dimensions=['class', 'sex', 'embark_town', 'survived'],
        color='age',
        color_continuous_scale=px.colors.sequential.Plasma,
        title='Interactive Parallel Categories Plot',
        labels={'class': 'Class', 'sex': 'Gender', 'embark_town': 'Embarkation', 'survived': 'Survived'}
    )
    
    # Save the figures to HTML files
    fig1.write_html('scatter_plot.html')
    fig2.write_html('box_plot.html')
    fig3.write_html('parallel_categories.html')
    
    print("Interactive plots have been saved as HTML files.")
    return fig1, fig2, fig3

def main():
    # Exercise 1: Load and explore data
    print("Loading and exploring data...")
    df = load_explore_data()
    
    # Exercise 2: Basic plots with Matplotlib
    print("\nCreating basic plots with Matplotlib...")
    basic_plots(df)
    
    # Exercise 3: Advanced plots with Seaborn
    print("\nCreating advanced plots with Seaborn...")
    advanced_plots(df)
    
    # Exercise 4: Interactive plots with Plotly
    print("\nCreating interactive plots with Plotly...")
    interactive_plots(df)
    
    print("\nLab exercise completed!")

if __name__ == "__main__":
    main()
