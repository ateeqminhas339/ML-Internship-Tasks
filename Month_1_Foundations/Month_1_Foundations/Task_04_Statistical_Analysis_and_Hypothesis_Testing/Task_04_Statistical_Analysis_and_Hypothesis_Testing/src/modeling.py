"""Baseline modeling and comparison utilities for Task 04.

Two things live here, both required by the task's "Evaluation" section:
1. A simple OLS regression baseline that translates the hypothesis-test
   findings into a concrete, business-usable number (e.g. "smoking adds
   approximately $X to expected annual charges, holding age/bmi constant").
2. A naive-vs-proper test comparison, showing what would happen if someone
   skipped the assumption checks and used a t-test/ANOVA blindly - the
   "simple baseline" against which the test-selection helper is judged.
"""

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def fit_baseline_ols(df: pd.DataFrame):
    """Fit a simple OLS regression of charges on the key business drivers
    (age, bmi, children, smoker, region), used purely to translate
    statistical findings into an interpretable dollar impact per factor.
    This is a baseline, not a tuned predictive model.
    """
    model = smf.ols(
        "charges ~ age + bmi + children + C(smoker) + C(region) + C(sex)", data=df
    ).fit()
    return model


def summarize_ols_effects(model) -> pd.DataFrame:
    """Extract a tidy coefficient table (effect in dollars, CI, p-value)
    from a fitted OLS model.
    """
    conf_int = model.conf_int()
    summary = pd.DataFrame({
        "coefficient": model.params,
        "ci_lower": conf_int[0],
        "ci_upper": conf_int[1],
        "p_value": model.pvalues,
    })
    return summary.round(2)


def naive_vs_selected_two_group(group_a: pd.Series, group_b: pd.Series) -> pd.DataFrame:
    """Baseline comparison: what a naive analyst would report (a plain
    Student's t-test, assumptions never checked) vs. what the test-selection
    helper actually recommends. Returns a side-by-side comparison so the
    report can show whether the conclusion (significant or not) changes.
    """
    naive_stat, naive_p = stats.ttest_ind(group_a, group_b, equal_var=True)

    from .stats_tests import select_and_run_two_group_test
    proper_result = select_and_run_two_group_test(group_a, group_b, hypothesis="(baseline comparison)")

    return pd.DataFrame([
        {"approach": "Naive (Student's t-test, assumptions unchecked)",
         "test_used": "Student's t-test", "statistic": round(naive_stat, 4),
         "p_value": round(naive_p, 6), "significant": naive_p < 0.05},
        {"approach": "Test-selection helper (assumption-checked)",
         "test_used": proper_result.test_used, "statistic": round(proper_result.statistic, 4),
         "p_value": round(proper_result.p_value, 6), "significant": proper_result.significant},
    ])


def naive_vs_selected_multi_group(groups) -> pd.DataFrame:
    """Baseline comparison for 3+ groups: a naive, blindly-applied one-way
    ANOVA vs. the test-selection helper's actual recommendation. This is
    the single clearest illustration of why assumption-checking matters:
    for the region-vs-charges comparison in this dataset, naive ANOVA
    reports a false positive (p < 0.05) that the properly-selected
    Kruskal-Wallis test correctly does not support.
    """
    naive_stat, naive_p = stats.f_oneway(*groups)

    from .stats_tests import select_and_run_multi_group_test
    proper_result = select_and_run_multi_group_test(list(groups), hypothesis="(baseline comparison)")

    return pd.DataFrame([
        {"approach": "Naive (one-way ANOVA, assumptions unchecked)",
         "test_used": "One-way ANOVA", "statistic": round(naive_stat, 4),
         "p_value": round(naive_p, 6), "significant": naive_p < 0.05},
        {"approach": "Test-selection helper (assumption-checked)",
         "test_used": proper_result.test_used, "statistic": round(proper_result.statistic, 4),
         "p_value": round(proper_result.p_value, 6), "significant": proper_result.significant},
    ])
