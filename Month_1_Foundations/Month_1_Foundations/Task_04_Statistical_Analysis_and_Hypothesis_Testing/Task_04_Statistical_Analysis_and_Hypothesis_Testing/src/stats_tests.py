"""Statistical test selection and hypothesis-testing utilities.

The central idea is a *test selection helper*: rather than picking a test
by habit, we first check the assumptions (normality via Shapiro-Wilk,
equal variance via Levene's test) and let that decide between the
parametric test (t-test / ANOVA) and its non-parametric counterpart
(Mann-Whitney U / Kruskal-Wallis).
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


@dataclass
class TestResult:
    hypothesis: str
    test_used: str
    statistic: float
    p_value: float
    significant: bool
    effect_size_name: str
    effect_size_value: float
    notes: str = ""


# --------------------------------------------------------------------------
# Assumption checks (drive the test-selection helper)
# --------------------------------------------------------------------------
def check_normality(sample: pd.Series, alpha: float = ALPHA) -> bool:
    """Shapiro-Wilk normality test. Returns True if the sample looks
    normally distributed (p >= alpha). Subsamples to 5000 points if larger,
    since Shapiro-Wilk is not defined for very large n.
    """
    sample = sample.dropna()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    if len(sample) < 3:
        return False
    _, p = stats.shapiro(sample)
    return p >= alpha


def check_equal_variance(*samples, alpha: float = ALPHA) -> bool:
    """Levene's test for equal variances across 2+ groups."""
    _, p = stats.levene(*samples)
    return p >= alpha


# --------------------------------------------------------------------------
# Effect size calculators
# --------------------------------------------------------------------------
def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d for two independent samples (pooled standard deviation)."""
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_std


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation effect size, derived from the Mann-Whitney U statistic."""
    return 1 - (2 * u_stat) / (n1 * n2)


def eta_squared_from_anova(groups: List[pd.Series]) -> float:
    """Eta-squared effect size for a one-way ANOVA (proportion of variance
    explained by group membership).
    """
    all_values = pd.concat(groups)
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((all_values - grand_mean) ** 2).sum()
    return ss_between / ss_total


def epsilon_squared_from_kruskal(h_stat: float, n: int, k: int) -> float:
    """Epsilon-squared effect size for Kruskal-Wallis (n=total obs, k=n groups)."""
    return (h_stat - k + 1) / (n - k)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V effect size for two categorical variables."""
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    r, k = ct.shape
    return float(np.sqrt((chi2 / n) / (min(r - 1, k - 1))))


# --------------------------------------------------------------------------
# Confidence interval functions
# --------------------------------------------------------------------------
def ci_mean_difference(a: pd.Series, b: pd.Series, confidence: float = 0.95):
    """CI for the difference in means of two independent samples (Welch)."""
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dof = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1)
    )
    t_crit = stats.t.ppf((1 + confidence) / 2, dof)
    margin = t_crit * se
    return diff - margin, diff + margin


def ci_mean(sample: pd.Series, confidence: float = 0.95):
    """CI for a single sample mean."""
    n = len(sample)
    mean = sample.mean()
    se = sample.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_crit * se
    return mean - margin, mean + margin


def ci_correlation(r: float, n: int, confidence: float = 0.95):
    """CI for a Pearson correlation coefficient via the Fisher z-transform."""
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + confidence) / 2)
    lo, hi = z - z_crit * se, z + z_crit * se
    return np.tanh(lo), np.tanh(hi)


# --------------------------------------------------------------------------
# Test selection helper - the core "pick the right test" function
# --------------------------------------------------------------------------
def select_and_run_two_group_test(group_a: pd.Series, group_b: pd.Series, hypothesis: str) -> TestResult:
    """Compare two independent numeric groups: checks normality on each
    group, then picks Welch's t-test (normal, allowing unequal variance)
    or Mann-Whitney U (non-normal) accordingly.
    """
    normal_a = check_normality(group_a)
    normal_b = check_normality(group_b)

    if normal_a and normal_b:
        stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)
        effect_name, effect_val = "Cohen's d", cohens_d(group_a, group_b)
        test_used = "Welch's t-test"
        notes = "Both groups passed Shapiro-Wilk normality; Welch's t-test used (does not assume equal variance)."
    else:
        stat, p = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
        effect_name = "Rank-biserial correlation"
        effect_val = rank_biserial_from_u(stat, len(group_a), len(group_b))
        test_used = "Mann-Whitney U"
        notes = "At least one group failed Shapiro-Wilk normality; Mann-Whitney U (non-parametric) used instead."

    return TestResult(
        hypothesis=hypothesis, test_used=test_used, statistic=float(stat), p_value=float(p),
        significant=p < ALPHA, effect_size_name=effect_name, effect_size_value=float(effect_val), notes=notes,
    )


def select_and_run_multi_group_test(groups: List[pd.Series], hypothesis: str) -> TestResult:
    """Compare 3+ independent numeric groups: checks normality (all groups)
    and equal variance, then picks one-way ANOVA or Kruskal-Wallis.
    """
    all_normal = all(check_normality(g) for g in groups)
    equal_var = check_equal_variance(*groups)

    if all_normal and equal_var:
        stat, p = stats.f_oneway(*groups)
        effect_name, effect_val = "Eta-squared", eta_squared_from_anova(groups)
        test_used = "One-way ANOVA"
        notes = "All groups passed normality and Levene's equal-variance test; one-way ANOVA used."
    else:
        stat, p = stats.kruskal(*groups)
        n_total = sum(len(g) for g in groups)
        effect_name = "Epsilon-squared"
        effect_val = epsilon_squared_from_kruskal(stat, n_total, len(groups))
        test_used = "Kruskal-Wallis H"
        notes = "Normality and/or equal-variance assumption failed; Kruskal-Wallis (non-parametric) used instead."

    return TestResult(
        hypothesis=hypothesis, test_used=test_used, statistic=float(stat), p_value=float(p),
        significant=p < ALPHA, effect_size_name=effect_name, effect_size_value=float(effect_val), notes=notes,
    )


def run_chi_square_test(x: pd.Series, y: pd.Series, hypothesis: str) -> TestResult:
    """Chi-square test of independence between two categorical variables,
    with Cramer's V as the effect size.
    """
    ct = pd.crosstab(x, y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return TestResult(
        hypothesis=hypothesis, test_used="Chi-square test of independence", statistic=float(chi2), p_value=float(p),
        significant=p < ALPHA, effect_size_name="Cramer's V", effect_size_value=cramers_v(x, y),
        notes=f"Contingency table shape {ct.shape}, dof={dof}.",
    )


def run_correlation_test(x: pd.Series, y: pd.Series, hypothesis: str, method: str = "pearson") -> TestResult:
    """Pearson or Spearman correlation test between two continuous variables."""
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
        test_used = "Pearson correlation"
    else:
        r, p = stats.spearmanr(x, y)
        test_used = "Spearman correlation"
    return TestResult(
        hypothesis=hypothesis, test_used=test_used, statistic=float(r), p_value=float(p),
        significant=p < ALPHA, effect_size_name="r (correlation coefficient)", effect_size_value=float(r),
        notes="Effect size interpretation: |r|<0.1 negligible, 0.1-0.3 small, 0.3-0.5 moderate, >0.5 large.",
    )


# --------------------------------------------------------------------------
# Multiple comparison correction
# --------------------------------------------------------------------------
def bonferroni_correction(p_values: List[float]) -> List[float]:
    """Bonferroni-corrected p-values (multiply by number of tests, cap at 1.0)."""
    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def holm_correction(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction (less conservative than plain Bonferroni)."""
    m = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * p_values[idx]
        running_max = max(running_max, adjusted)
        corrected[idx] = min(running_max, 1.0)
    return corrected.tolist()


def results_to_dataframe(results: List[TestResult], corrected_p: List[float] = None) -> pd.DataFrame:
    """Convert a list of TestResult objects into a tidy summary DataFrame,
    optionally attaching multiple-comparison-corrected p-values.
    """
    rows = []
    for i, r in enumerate(results):
        row = {
            "Hypothesis": r.hypothesis,
            "Test used": r.test_used,
            "Statistic": round(r.statistic, 4),
            "p-value": round(r.p_value, 6),
            "Significant (alpha=0.05)": r.significant,
            "Effect size": f"{r.effect_size_name} = {r.effect_size_value:.4f}",
        }
        if corrected_p is not None:
            row["Holm-corrected p-value"] = round(corrected_p[i], 6)
            row["Significant after correction"] = corrected_p[i] < ALPHA
        rows.append(row)
    return pd.DataFrame(rows)
