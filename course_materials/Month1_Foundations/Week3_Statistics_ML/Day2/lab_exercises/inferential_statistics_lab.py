"""
Inferential Statistics Lab - Week 3, Day 2

This lab will help you practice hypothesis testing, confidence intervals,
and other inferential statistics concepts using Python.
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
# Exercise 1: Confidence Intervals
# =============================================================================
print("\n=== Exercise 1: Confidence Intervals ===\n")

# Sample data: Daily time spent on a website (in minutes)
time_spent = np.random.normal(loc=25, scale=5, size=100)  # 100 users

# 1.1 Calculate the 95% confidence interval for the mean
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
print(f"Sample mean: {mean:.2f} minutes")
print(f"95% Confidence Interval: [{ci_low:.2f}, {ci_high:.2f}] minutes")

# 1.2 Visualize the confidence interval
plt.figure(figsize=(10, 4))
plt.errorbar(x=0, y=mean, yerr=[[mean - ci_low], [ci_high - mean]], 
             fmt='o', capsize=5, markersize=8)
plt.axhline(y=25, color='r', linestyle='--', label='Population Mean')
plt.xlim(-0.5, 0.5)
plt.xticks([])
plt.ylabel('Time Spent (minutes)')
plt.title('95% Confidence Interval for Mean Time Spent')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('confidence_interval.png')
print("\nSaved visualization: confidence_interval.png")

# =============================================================================
# Exercise 2: One-Sample t-Test
# =============================================================================
print("\n=== Exercise 2: One-Sample t-Test ===\n")

# Scenario: A company claims their app's average session duration is 30 minutes.
# We have a sample of 50 sessions with the following durations (in minutes):
session_durations = np.random.normal(loc=28, scale=7, size=50)

# 2.1 Perform one-sample t-test
pop_mean = 30  # Claimed population mean
t_stat, p_value = stats.ttest_1samp(session_durations, pop_mean)

print(f"Sample mean: {np.mean(session_durations):.2f} minutes")
alpha = 0.05
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# 2.2 Make a decision
if p_value < alpha:
    print(f"Conclusion: Reject the null hypothesis (p < {alpha}).")
    print("There is significant evidence that the average session duration is not 30 minutes.")
else:
    print(f"Conclusion: Fail to reject the null hypothesis (p >= {alpha}).")
    print("There is not enough evidence to conclude that the average session duration differs from 30 minutes.")

# =============================================================================
# Exercise 3: Two-Sample t-Test
# =============================================================================
print("\n=== Exercise 3: Two-Sample t-Test ===\n")

# Scenario: Compare test scores between two teaching methods
# Method A: Traditional lecture
# Method B: Interactive learning

# Generate sample data
np.random.seed(42)
scores_method_a = np.random.normal(loc=75, scale=10, size=30)
scores_method_b = np.random.normal(loc=82, scale=8, size=30)

# 3.1 Visualize the distributions
plt.figure(figsize=(10, 5))
sns.kdeplot(scores_method_a, label='Method A (Traditional)', fill=True)
sns.kdeplot(scores_method_b, label='Method B (Interactive)', fill=True)
plt.xlabel('Test Scores')
plt.title('Distribution of Test Scores by Teaching Method')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('teaching_methods_distribution.png')
print("\nSaved visualization: teaching_methods_distribution.png")

# 3.2 Check for equal variances
levene_stat, levene_p = stats.levene(scores_method_a, scores_method_b)
print(f"\nLevene's test for equal variances: p-value = {levene_p:.4f}")
if levene_p > 0.05:
    print("Assuming equal variances (p > 0.05)")
    equal_var = True
else:
    print("Not assuming equal variances (p <= 0.05)")
    equal_var = False

# 3.3 Perform two-sample t-test
t_stat, p_value = stats.ttest_ind(scores_method_a, scores_method_b, 
                                 equal_var=equal_var)

print(f"\nMethod A mean: {np.mean(scores_method_a):.2f}")
print(f"Method B mean: {np.mean(scores_method_b):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# 3.4 Make a decision
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
# Exercise 4: Chi-Square Test for Independence
# =============================================================================
print("\n=== Exercise 4: Chi-Square Test for Independence ===\n")

# Scenario: Test if there's an association between gender and product preference
# Create a contingency table
# Rows: Gender (Male, Female)
# Columns: Product Preference (A, B, C)
contingency_table = pd.DataFrame({
    'Product A': [45, 55],
    'Product B': [30, 20],
    'Product C': [25, 25]
}, index=['Male', 'Female'])

print("Contingency Table:")
print(contingency_table)
print()

# 4.1 Perform chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

# 4.2 Make a decision
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: Reject the null hypothesis.")
    print("There is a statistically significant association between gender and product preference.")
else:
    print("\nConclusion: Fail to reject the null hypothesis.")
    print("There is no statistically significant association between gender and product preference.")

# 4.3 Visualize the data
plt.figure(figsize=(10, 5))
contingency_table.T.plot(kind='bar', stacked=True)
plt.title('Product Preference by Gender')
plt.xlabel('Product')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Gender')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('product_preference.png')
print("\nSaved visualization: product_preference.png")

# =============================================================================
# Exercise 5: Power Analysis
# =============================================================================
print("\n=== Exercise 5: Power Analysis ===\n")

# Scenario: Determine the required sample size for an A/B test
# We want to detect an effect size of 0.5 with 80% power at alpha = 0.05

effect_size = 0.5  # Medium effect size
alpha = 0.05
power_level = 0.80

# 5.1 Calculate required sample size
analysis = sm.stats.power.TTestIndPower()
required_n = analysis.solve_power(
    effect_size=effect_size,
    power=power_level,
    alpha=alpha,
    ratio=1.0  # Equal sample sizes
)

print(f"Required sample size per group: {np.ceil(required_n):.0f}")
print(f"Total required sample size: {2 * np.ceil(required_n):.0f}")

# 5.2 Create power curve
effect_sizes = np.linspace(0.1, 1.0, 10)
sample_sizes = np.arange(10, 201, 10)

# Create a grid of effect sizes and sample sizes
X, Y = np.meshgrid(effect_sizes, sample_sizes)
Z = np.zeros_like(X, dtype=float)

for i in range(len(sample_sizes)):
    for j in range(len(effect_sizes)):
        Z[i, j] = analysis.solve_power(
            effect_size=effect_sizes[j],
            nobs1=sample_sizes[i],
            alpha=alpha,
            power=None
        )

# Plot power curve
plt.figure(figsize=(10, 6))
contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour, label='Statistical Power')
plt.axhline(y=required_n, color='r', linestyle='--', 
            label=f'Required N ({np.ceil(required_n):.0f} per group)')
plt.axvline(x=effect_size, color='r', linestyle='--', 
            label=f'Effect Size ({effect_size})')
plt.xlabel('Effect Size (Cohen\'s d)')
plt.ylabel('Sample Size per Group')
plt.title('Power Analysis: Effect Size vs. Sample Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('power_analysis.png')
print("\nSaved visualization: power_analysis.png")

print("\nLab exercises completed!")
