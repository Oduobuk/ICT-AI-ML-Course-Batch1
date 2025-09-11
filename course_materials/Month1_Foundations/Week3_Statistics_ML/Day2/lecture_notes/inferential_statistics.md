# Day 2: Inferential Statistics

## 📚 Learning Objectives
By the end of this session, you will be able to:
- Understand the fundamentals of statistical inference
- Perform and interpret hypothesis tests
- Calculate and interpret confidence intervals
- Understand Type I and Type II errors
- Apply statistical tests using Python

## 🔍 1. Introduction to Statistical Inference

### Population vs Sample
- **Population**: The entire group we want to study
- **Sample**: A subset of the population used for analysis
- **Parameter vs Statistic**:
  - Parameter: A numerical characteristic of a population (e.g., population mean μ)
  - Statistic: A numerical characteristic of a sample (e.g., sample mean x̄)

### Sampling Methods
1. **Simple Random Sampling**: Every member has an equal chance of being selected
2. **Stratified Sampling**: Population divided into subgroups (strata), then random samples from each
3. **Cluster Sampling**: Population divided into clusters, random clusters selected, all members in chosen clusters are sampled
4. **Systematic Sampling**: Selecting every k-th element from a list

### Sampling Distribution
- The distribution of a statistic (e.g., mean) over many samples
- The Central Limit Theorem states that the sampling distribution of the mean will be approximately normal for large enough sample sizes

## 📊 2. Confidence Intervals

### Concept
A range of values that is likely to contain a population parameter with a certain level of confidence.

### Formula for Mean (σ known)
$$\bar{x} \pm z_{\alpha/2} \left(\frac{\sigma}{\sqrt{n}}\right)$$

### Formula for Mean (σ unknown)
$$\bar{x} \pm t_{\alpha/2, n-1} \left(\frac{s}{\sqrt{n}}\right)$$

### Python Implementation
```python
import numpy as np
from scipy import stats

# Sample data
data = [72, 75, 71, 76, 74, 72, 74, 75, 73, 74]
confidence = 0.95

# Calculate confidence interval
mean = np.mean(data)
sem = stats.sem(data)  # Standard error of the mean
ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)

print(f"{confidence*100}% Confidence Interval: {ci}")
```

## 🎯 3. Hypothesis Testing

### Null and Alternative Hypotheses
- **Null Hypothesis (H₀)**: A statement of no effect or no difference
- **Alternative Hypothesis (H₁)**: What we want to prove

### Types of Tests
1. **One-sample t-test**: Compare sample mean to a known value
2. **Two-sample t-test**: Compare means of two independent samples
3. **Paired t-test**: Compare means from the same group at different times
4. **ANOVA**: Compare means of more than two groups
5. **Chi-square test**: Test relationships between categorical variables

### p-values and Significance
- **p-value**: Probability of observing the data if the null hypothesis is true
- **Significance level (α)**: Threshold for rejecting the null (commonly 0.05)
- **Decision Rule**: If p-value < α, reject H₀

## 📉 4. Common Statistical Tests

### One-sample t-test
```python
from scipy import stats

# Sample data and hypothesized mean
sample = [72, 75, 71, 76, 74, 72, 74, 75, 73, 74]
pop_mean = 70  # Hypothesized population mean

# Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(sample, pop_mean)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")
```

### Two-sample t-test
```python
# Two independent samples
group1 = [72, 75, 71, 76, 74, 72, 74, 75, 73, 74]
group2 = [68, 72, 69, 70, 71, 70, 69, 72, 71, 70]

# Perform two-sample t-test
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

### Paired t-test
```python
# Measurements before and after treatment
before = [72, 75, 71, 76, 74, 72, 74, 75, 73, 74]
after = [68, 72, 70, 73, 72, 70, 71, 74, 70, 72]

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(before, after)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
```

### One-way ANOVA
```python
# Three or more groups
group1 = [72, 75, 71, 76, 74]
group2 = [68, 72, 70, 71, 70]
group3 = [65, 68, 67, 69, 70]

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)
print(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
```

### Chi-square Test
```python
from scipy.stats import chi2_contingency

# Contingency table
#         | Pass | Fail |
# Group A |  30  |  10  |
# Group B |  20  |  20  |
observed = [[30, 10],
            [20, 20]]

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(observed)
print(f"Chi2: {chi2:.4f}, p-value: {p:.4f}")
```

## ⚠️ 5. Type I and Type II Errors

### Definitions
- **Type I Error (False Positive)**: Rejecting a true null hypothesis
  - Probability = α (significance level)
- **Type II Error (False Negative)**: Failing to reject a false null hypothesis
  - Probability = β

### Power of a Test
- Power = 1 - β (probability of correctly rejecting a false null)
- Factors affecting power:
  1. Sample size
  2. Effect size
  3. Significance level (α)
  4. Variability in the data

### Calculating Power
```python
from statsmodels.stats.power import TTestPower

# Initialize power analysis
power_analysis = TTestPower()

# Calculate power
power = power_analysis.solve_power(
    effect_size=0.5,  # Medium effect size
    nobs=50,          # Sample size
    alpha=0.05,       # Significance level
    power=None        # What we're solving for
)
print(f"Power: {power:.4f}")

# Calculate required sample size
sample_size = power_analysis.solve_power(
    effect_size=0.5,
    alpha=0.05,
    power=0.8,  # Desired power
    nobs=None   # What we're solving for
)
print(f"Required sample size: {sample_size:.0f}")
```

## 📝 6. Practical Example: A/B Testing

### Scenario
An e-commerce website wants to test if a new webpage design (B) leads to higher conversion rates than the current design (A).

### Data
- **Group A (Control)**: 1000 visitors, 50 conversions (5%)
- **Group B (Variant)**: 1000 visitors, 65 conversions (6.5%)

### Hypothesis Test
```python
from statsmodels.stats.proportion import proportions_ztest

# Number of successes and sample sizes
count = [50, 65]     # Conversions
nobs = [1000, 1000]  # Total visitors

# Perform two-proportion z-test
z_stat, p_value = proportions_ztest(count, nobs)
print(f"z-statistic: {z_stat:.4f}, p-value: {p_value:.4f}")

# Calculate confidence interval for the difference in proportions
from statsmodels.stats.proportion import confint_proportions_2indep

ci_low, ci_upp = confint_proportions_2indep(
    count1=65, nobs1=1000,  # Group B
    count2=50, nobs2=1000,  # Group A
    compare='diff',
    alpha=0.05
)
print(f"95% CI for difference in proportions: [{ci_low:.4f}, {ci_upp:.4f}]")
```

### Interpretation
- If p-value < 0.05, we can conclude that the new design (B) leads to a statistically significant increase in conversion rates.
- The confidence interval tells us the range of plausible values for the true difference in conversion rates.

## 📚 7. Key Takeaways

1. **Statistical Inference** allows us to make conclusions about a population based on sample data.
2. **Confidence Intervals** provide a range of plausible values for a population parameter.
3. **Hypothesis Testing** helps us make decisions about population parameters based on sample statistics.
4. **Common Tests** include t-tests, ANOVA, and chi-square tests, each appropriate for different types of data and research questions.
5. **Type I and Type II Errors** are inherent risks in hypothesis testing that we must consider.
6. **Power Analysis** helps determine the sample size needed to detect an effect of a certain size.

## 🔍 8. Further Reading

1. [Scipy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
2. [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
3. [Interactive Statistical Analysis with Python](https://github.com/rougier/statistical-analysis-python-tutorial)
4. [Think Stats: Exploratory Data Analysis in Python](https://greenteapress.com/thinkstats2/)

## 💡 9. Practice Questions

1. **Multiple Choice**:
   What does a p-value of 0.03 mean?
   a) There's a 3% chance the null hypothesis is true
   b) There's a 3% chance of observing the data if the null is true
   c) There's a 97% chance the alternative hypothesis is true
   d) The effect size is 0.03

   **Answer**: b) There's a 3% chance of observing the data if the null is true

2. **True/False**:
   A 95% confidence interval means there's a 95% probability that the true population parameter lies within the interval.
   
   **Answer**: False. The confidence level refers to the long-run proportion of confidence intervals that contain the true parameter, not the probability for any specific interval.

3. **Short Answer**:
   When would you use a paired t-test instead of a two-sample t-test?
   
   **Answer**: A paired t-test is used when the samples are related or matched in some way, such as measurements taken from the same subjects before and after a treatment. A two-sample t-test is used for independent samples.
