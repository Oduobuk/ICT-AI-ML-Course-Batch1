"""
Descriptive Statistics Lab - Week 3, Day 1

This lab will help you practice calculating and interpreting descriptive statistics using Python.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Exercise 1: Basic Descriptive Statistics
# =============================================================================
print("\n=== Exercise 1: Basic Descriptive Statistics ===\n")

# Sample dataset: Daily website visitors for a month (in thousands)
visitors = np.array([12.5, 15.2, 13.8, 14.9, 16.1, 15.5, 14.2, 
                    13.7, 16.8, 17.2, 15.9, 16.5, 18.2, 17.8,
                    16.9, 15.4, 14.8, 16.2, 17.5, 18.1, 17.9,
                    16.8, 15.2, 14.9, 16.5, 17.8, 18.5, 19.2, 18.9, 20.1])

# 1.1 Calculate basic statistics
mean_visitors = np.mean(visitors)
median_visitors = np.median(visitors)
mode_visitors = stats.mode(visitors, keepdims=True)
std_visitors = np.std(visitors, ddof=1)  # ddof=1 for sample standard deviation
var_visitors = np.var(visitors, ddof=1)
q1 = np.percentile(visitors, 25)
q3 = np.percentile(visitors, 75)
iqr = q3 - q1

print(f"Mean visitors: {mean_visitors:.2f}K")
print(f"Median visitors: {median_visitors:.2f}K")
print(f"Mode visitors: {mode_visitors.mode[0]:.2f}K (appears {mode_visitors.count[0]} times)")
print(f"Standard deviation: {std_visitors:.2f}K")
print(f"Variance: {var_visitors:.2f}K²")
print(f"IQR: {iqr:.2f}K (Q1: {q1:.2f}K, Q3: {q3:.2f}K)")

# 1.2 Visualize the distribution
plt.figure(figsize=(12, 5))

# Histogram with KDE
plt.subplot(1, 2, 1)
sns.histplot(visitors, kde=True, bins=8)
plt.axvline(mean_visitors, color='r', linestyle='--', label=f'Mean: {mean_visitors:.2f}K')
plt.axvline(median_visitors, color='g', linestyle='-', label=f'Median: {median_visitors:.2f}K')
plt.title('Distribution of Daily Visitors')
plt.xlabel('Number of Visitors (thousands)')
plt.legend()

# Box plot
plt.subplot(1, 2, 2)
sns.boxplot(x=visitors)
plt.title('Box Plot of Daily Visitors')
plt.xlabel('Number of Visitors (thousands)')

plt.tight_layout()
plt.show()

# =============================================================================
# Exercise 2: Probability Distributions
# =============================================================================
print("\n=== Exercise 2: Probability Distributions ===\n")

def plot_distribution(samples, title, xlabel='Value', ylabel='Density'):
    """Helper function to plot distributions"""
    plt.figure(figsize=(10, 5))
    sns.histplot(samples, kde=True, stat='density')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# 2.1 Normal Distribution
mu, sigma = 100, 15  # mean and standard deviation
normal_samples = np.random.normal(mu, sigma, 1000)
plot_distribution(normal_samples, 'Normal Distribution (μ=100, σ=15)')

# 2.2 Binomial Distribution
n, p = 10, 0.5  # number of trials, probability of success
binomial_samples = np.random.binomial(n, p, 1000)
plot_distribution(binomial_samples, 'Binomial Distribution (n=10, p=0.5)')

# 2.3 Poisson Distribution
lam = 5  # rate parameter
poisson_samples = np.random.poisson(lam, 1000)
plot_distribution(poisson_samples, 'Poisson Distribution (λ=5)')

# =============================================================================
# Exercise 3: Central Limit Theorem Demonstration
# =============================================================================
print("\n=== Exercise 3: Central Limit Theorem ===\n")

def clt_demonstration(population, sample_size=30, n_samples=1000, title=''):
    """Demonstrate CLT by sampling from a non-normal population"""
    sample_means = [np.mean(np.random.choice(population, sample_size)) 
                   for _ in range(n_samples)]
    
    plt.figure(figsize=(12, 5))
    
    # Population distribution
    plt.subplot(1, 2, 1)
    sns.histplot(population, kde=True)
    plt.title(f'Population Distribution\nSkew: {stats.skew(population):.2f}, ' +
              f'Kurtosis: {stats.kurtosis(population):.2f}')
    
    # Sampling distribution of the mean
    plt.subplot(1, 2, 2)
    sns.histplot(sample_means, kde=True)
    plt.title(f'Sampling Distribution of Mean (n={sample_size})\n' +
             f'Mean: {np.mean(sample_means):.2f}, ' +
             f'Std: {np.std(sample_means, ddof=1):.2f}')
    
    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.show()

# 3.1 CLT with Exponential distribution
exponential_pop = np.random.exponential(scale=1.0, size=10000)
clt_demonstration(exponential_pop, title='CLT with Exponential Distribution')

# 3.2 CLT with Uniform distribution
uniform_pop = np.random.uniform(0, 10, 10000)
clt_demonstration(uniform_pop, title='CLT with Uniform Distribution')

# =============================================================================
# Exercise 4: Real-world Data Analysis
# =============================================================================
print("\n=== Exercise 4: Real-world Data Analysis ===\n")

# Load the tips dataset from seaborn
tips = sns.load_dataset('tips')
print("\nFirst few rows of the tips dataset:")
print(tips.head())

# 4.1 Calculate descriptive statistics by day
print("\nDescriptive statistics by day:")
print(tips.groupby('day')['total_bill'].describe())

# 4.2 Visualize the distributions
plt.figure(figsize=(12, 10))

# Box plot of total bill by day
plt.subplot(2, 2, 1)
sns.boxplot(x='day', y='total_bill', data=tips)
plt.title('Total Bill by Day')

# Violin plot of tips by time and day
plt.subplot(2, 2, 2)
sns.violinplot(x='day', y='tip', hue='time', data=tips, split=True)
plt.title('Tip Distribution by Day and Time')

# Scatter plot of total bill vs tip with regression line
plt.subplot(2, 2, 3)
sns.regplot(x='total_bill', y='tip', data=tips, scatter_kws={'alpha':0.5})
plt.title('Total Bill vs Tip with Regression Line')

# Heatmap of correlation matrix
plt.subplot(2, 2, 4)
numeric_cols = tips.select_dtypes(include=[np.number]).corr()
sns.heatmap(numeric_cols, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()

# 4.3 Calculate skewness and kurtosis
print("\nSkewness and Kurtosis:")
for col in ['total_bill', 'tip']:
    print(f"\n{col.capitalize()}:")
    print(f"  Skewness: {tips[col].skew():.4f}")
    print(f"  Kurtosis: {tips[col].kurtosis():.4f}")

print("\nLab exercises completed!")
