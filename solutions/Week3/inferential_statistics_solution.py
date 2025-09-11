"""
Inferential Statistics Lab - Solutions

This file contains the solutions to the Inferential Statistics lab exercises.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import power
from statsmodels.stats.proportion import proportions_ztest

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# Exercise 1: Confidence Intervals - SOLUTION
# =============================================================================
print("\n=== Exercise 1: Confidence Intervals - SOLUTION ===\n")

# Sample data: Daily time spent on a website (in minutes)
time_spent = np.random.normal(loc=25, scale=5, size=100)  # 100 users

# Calculate 95% confidence interval for the mean
def calculate_ci(data, confidence=0.95):
    """Calculate confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    
    # Calculate the t-critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    # Calculate margin of error
    margin = t_crit * std_err
    
    return mean, mean - margin, mean + margin

# Calculate and print CI
mean, ci_low, ci_high = calculate_ci(time_spent)
print("=== Expected Output ===")
print(f"Sample mean: {mean:.2f} minutes")
print(f"95% Confidence Interval: [{ci_low:.2f}, {ci_high:.2f}] minutes")

# =============================================================================
# Exercise 2: One-Sample t-Test - SOLUTION
# =============================================================================
print("\n=== Exercise 2: One-Sample t-Test - SOLUTION ===\n")

# Sample data
session_durations = np.random.normal(loc=28, scale=7, size=50)
pop_mean = 30  # Claimed population mean

# Perform one-sample t-test
t_stat, p_value = stats.ttest_1samp(session_durations, pop_mean)

alpha = 0.05
print("=== Expected Output ===")
print(f"Sample mean: {np.mean(session_durations):.2f} minutes")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Make a decision
if p_value < alpha:
    print(f"\nConclusion: Reject the null hypothesis (p < {alpha}).")
    print("There is significant evidence that the average session duration is not 30 minutes.")
else:
    print(f"\nConclusion: Fail to reject the null hypothesis (p >= {alpha}).")
    print("There is not enough evidence to conclude that the average session duration differs from 30 minutes.")

# =============================================================================
# Exercise 3: Two-Sample t-Test - SOLUTION
# =============================================================================
print("\n=== Exercise 3: Two-Sample t-Test - SOLUTION ===\n")

# Generate sample data
np.random.seed(42)
scores_method_a = np.random.normal(loc=75, scale=10, size=30)
scores_method_b = np.random.normal(loc=82, scale=8, size=30)

# Check for equal variances
levene_stat, levene_p = stats.levene(scores_method_a, scores_method_b)
equal_var = levene_p > 0.05

# Perform two-sample t-test
t_stat, p_value = stats.ttest_ind(scores_method_a, scores_method_b, 
                                 equal_var=equal_var)

print("=== Expected Output ===")
print(f"Method A mean: {np.mean(scores_method_a):.2f}")
print(f"Method B mean: {np.mean(scores_method_b):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Make a decision
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: Reject the null hypothesis.")
    print("There is a statistically significant difference in test scores between the two teaching methods.")
    
    # Check which method has higher scores
    if np.mean(scores_method_a) > np.mean(scores_method_b):
        print("Method A (Traditional) has significantly higher scores.")
    else:
        print("Method B (Interactive) has significantly higher scores.")
else:
    print("\nConclusion: Fail to reject the null hypothesis.")
    print("There is no statistically significant difference in test scores between the two teaching methods.")

# =============================================================================
# Exercise 4: Chi-Square Test for Independence - SOLUTION
# =============================================================================
print("\n=== Exercise 4: Chi-Square Test for Independence - SOLUTION ===\n")

# Contingency table
contingency_table = pd.DataFrame({
    'Product A': [45, 55],
    'Product B': [30, 20],
    'Product C': [25, 25]
}, index=['Male', 'Female'])

print("Contingency Table:")
print(contingency_table)
print()

# Perform chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print("=== Expected Output ===")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

# Make a decision
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: Reject the null hypothesis.")
    print("There is a statistically significant association between gender and product preference.")
else:
    print("\nConclusion: Fail to reject the null hypothesis.")
    print("There is no statistically significant association between gender and product preference.")

# =============================================================================
# Exercise 5: Power Analysis - SOLUTION
# =============================================================================
print("\n=== Exercise 5: Power Analysis - SOLUTION ===\n")

# Parameters
effect_size = 0.5  # Medium effect size
alpha = 0.05
power_level = 0.80

# Calculate required sample size
analysis = sm.stats.power.TTestIndPower()
required_n = analysis.solve_power(
    effect_size=effect_size,
    power=power_level,
    alpha=alpha,
    ratio=1.0  # Equal sample sizes
)

print("=== Expected Output ===")
print(f"Required sample size per group: {np.ceil(required_n):.0f}")
print(f"Total required sample size: {2 * np.ceil(required_n):.0f}")

print("\nAll solutions have been executed. Compare your results with the expected outputs above.")
