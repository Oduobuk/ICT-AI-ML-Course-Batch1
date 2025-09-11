# Day 1: Descriptive Statistics & Probability

## 📚 Learning Objectives
By the end of this session, you will be able to:
- Calculate and interpret measures of central tendency and dispersion
- Understand and work with common probability distributions
- Apply the Central Limit Theorem in data analysis
- Calculate and interpret statistical moments
- Use Python to perform statistical analysis

## 📊 1. Measures of Central Tendency

### Mean (Arithmetic Average)
- **Definition**: The sum of all values divided by the number of values
- **Formula**: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$
- **Python Implementation**:
  ```python
  import numpy as np
  data = [10, 20, 30, 40, 50]
  mean = np.mean(data)
  print(f"Mean: {mean}")  # Output: 30.0
  ```
- **When to use**: For normally distributed data without outliers

### Median
- **Definition**: The middle value when data is ordered
- **Python Implementation**:
  ```python
  median = np.median(data)
  print(f"Median: {median}")  # Output: 30.0
  ```
- **When to use**: For skewed distributions or data with outliers

### Mode
- **Definition**: The most frequent value in the dataset
- **Python Implementation**:
  ```python
  from scipy import stats
  mode = stats.mode([1, 2, 2, 3, 4])
  print(f"Mode: {mode.mode[0]}, Count: {mode.count[0]}")  # Output: 2, 2
  ```
- **When to use**: For categorical data or identifying most common values

## 📈 2. Measures of Dispersion

### Range
- **Definition**: Difference between maximum and minimum values
- **Formula**: $Range = max(x) - min(x)$

### Variance
- **Definition**: Average of squared differences from the mean
- **Formula**: $s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$
- **Python Implementation**:
  ```python
  variance = np.var(data, ddof=1)  # ddof=1 for sample variance
  print(f"Variance: {variance:.2f}")
  ```

### Standard Deviation
- **Definition**: Square root of variance
- **Formula**: $s = \sqrt{s^2}$
- **Python Implementation**:
  ```python
  std_dev = np.std(data, ddof=1)
  print(f"Standard Deviation: {std_dev:.2f}")
  ```

### Interquartile Range (IQR)
- **Definition**: Range between the first (25th) and third (75th) quartiles
- **Python Implementation**:
  ```python
  Q1 = np.percentile(data, 25)
  Q3 = np.percentile(data, 75)
  IQR = Q3 - Q1
  print(f"IQR: {IQR}")
  ```

## 🎲 3. Probability Distributions

### Normal Distribution (Gaussian)
- **Properties**:
  - Symmetric, bell-shaped curve
  - Defined by mean (μ) and standard deviation (σ)
  - 68-95-99.7 rule
- **Python Visualization**:
  ```python
  import matplotlib.pyplot as plt
  import numpy as np
  from scipy.stats import norm
  
  x = np.linspace(-4, 4, 1000)
  plt.plot(x, norm.pdf(x, 0, 1), 'r-', lw=2, label='μ=0, σ=1')
  plt.title('Standard Normal Distribution')
  plt.legend()
  plt.show()
  ```

### Binomial Distribution
- **When to use**: For binary outcomes with fixed trials
- **Parameters**: n (trials), p (probability of success)
- **Example**:
  ```python
  from scipy.stats import binom
  
  n, p = 10, 0.5
  x = np.arange(0, n+1)
  plt.bar(x, binom.pmf(x, n, p))
  plt.title('Binomial Distribution (n=10, p=0.5)')
  plt.show()
  ```

### Poisson Distribution
- **When to use**: For counting events in fixed intervals
- **Parameter**: λ (average rate)
- **Example**:
  ```python
  from scipy.stats import poisson
  
  mu = 3
  x = np.arange(0, 15)
  plt.bar(x, poisson.pmf(x, mu))
  plt.title('Poisson Distribution (λ=3)')
  plt.show()
  ```

## 🎯 4. Central Limit Theorem (CLT)

### Key Points:
1. The sampling distribution of the mean approaches a normal distribution as sample size increases
2. True regardless of the population distribution shape
3. Foundation for many statistical tests

### Demonstration:
```python
import numpy as np
import matplotlib.pyplot as plt

# Non-normal population
data = np.random.exponential(scale=1.0, size=10000)

# Plot population distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(data, bins=30, density=True)
plt.title('Population Distribution (Exponential)')

# Sampling distribution of the mean
sample_means = [np.mean(np.random.choice(data, 30)) for _ in range(1000)]

plt.subplot(1, 2, 2)
plt.hist(sample_means, bins=30, density=True)
plt.title('Sampling Distribution of Mean (n=30)')
plt.tight_layout()
plt.show()
```

## 📏 5. Statistical Moments

### 1. Mean (First Moment)
- Measures central tendency

### 2. Variance (Second Central Moment)
- Measures dispersion

### 3. Skewness (Third Standardized Moment)
- Measures asymmetry
- **Interpretation**:
  - Positive: Right-skewed (tail on right)
  - Negative: Left-skewed (tail on left)
  - Zero: Symmetrical
- **Python Implementation**:
  ```python
  from scipy.stats import skew
  skewness = skew(data)
  print(f"Skewness: {skewness:.2f}")
  ```

### 4. Kurtosis (Fourth Standardized Moment)
- Measures tail heaviness
- **Interpretation**:
  - Positive: Heavy tails (leptokurtic)
  - Negative: Light tails (platykurtic)
  - Zero: Normal distribution
- **Python Implementation**:
  ```python
  from scipy.stats import kurtosis
  kurt = kurtosis(data)
  print(f"Kurtosis: {kurt:.2f}")
  ```

## 💻 Practical Exercise

### Dataset: Student Exam Scores
```python
import pandas as pd
import seaborn as sns

# Sample data
data = {
    'student_id': range(1, 101),
    'exam_score': np.random.normal(70, 15, 100).clip(0, 100)  # Normal dist, mean=70, std=15
}
df = pd.DataFrame(data)

# Calculate statistics
print("Descriptive Statistics:")
print(f"Mean: {df['exam_score'].mean():.2f}")
print(f"Median: {df['exam_score'].median():.2f}")
print(f"Standard Deviation: {df['exam_score'].std():.2f}")
print(f"Skewness: {df['exam_score'].skew():.2f}")
print(f"Kurtosis: {df['exam_score'].kurtosis():.2f}")

# Visualize
grid = sns.displot(df['exam_score'], kde=True)
grid.fig.suptitle('Distribution of Exam Scores')
plt.show()
```

## 📝 Summary
- **Measures of Central Tendency**: Mean, median, mode
- **Measures of Dispersion**: Range, variance, standard deviation, IQR
- **Probability Distributions**: Normal, Binomial, Poisson
- **Central Limit Theorem**: Foundation for inferential statistics
- **Statistical Moments**: Mean, variance, skewness, kurtosis

## 📚 Additional Resources
1. [Think Stats](https://greenteapress.com/thinkstats2/)
2. [Python Data Science Handbook - Statistics](https://jakevdp.github.io/PythonDataScienceHandbook/05.14-image-features.html)
3. [Scipy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
