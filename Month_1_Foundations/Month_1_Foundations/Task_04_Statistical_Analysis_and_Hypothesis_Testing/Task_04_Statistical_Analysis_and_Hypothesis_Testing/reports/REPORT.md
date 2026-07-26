# Task 04 — Statistical Analysis and Hypothesis Testing — Report

Author: Ateeq (@ateeqminhas339)
Date: 2026-07-26

## 1. Executive Summary (≤5 bullets)

- **Smoking is the dominant, statistically robust cost driver**: smokers pay ~$23,800 more
  per year on average (OLS, holding age/BMI/children/region/sex constant), confirmed by
  Mann-Whitney U (p < 0.001, large effect size).
- **Region does NOT significantly affect charges** once tested properly (Kruskal-Wallis,
  p = 0.202) — despite a naive, assumption-unchecked ANOVA falsely suggesting it does
  (p = 0.033). This is the report's central methodological finding.
- **BMI has a small but statistically significant positive relationship with charges**
  (Pearson r ≈ 0.198, 95% CI excludes 0; OLS effect ≈ $339 per BMI unit).
- **Smoking status is weakly associated with sex** (Chi-square p = 0.006, Cramer's V ≈ 0.075
  — a small effect, detectable only due to the large sample size).
- After Holm-Bonferroni correction across all 4 hypotheses, three (H1, H3, H4) remain
  significant; H2 (region) remains non-significant — reinforcing that region should not be
  used as a pricing factor on this evidence.

## 2. Business Problem & Framing

A health insurance company must decide which customer attributes can legitimately justify
different premiums. Getting this wrong in either direction is costly: underpricing
high-risk customers (e.g. smokers) erodes margins, while using a factor that does **not**
actually predict cost (e.g. region, if unsupported) risks both financial mispricing and a
discrimination claim under insurance regulation. This task's goal is to answer three
concrete questions — does smoking, BMI, or region affect charges — with statistically
defensible evidence, not intuition, and to make the methodology itself auditable (which
test was used, why, and what would have happened with a less careful approach).

## 3. Data Overview

| Property | Value |
|---|---|
| Source | Kaggle — Medical Cost Personal Datasets |
| Raw shape | 1,338 rows x 7 columns |
| Cleaned shape | 1,337 rows x 9 columns (1 exact duplicate row dropped) |
| Missing values | 0 (confirmed across all columns) |
| Numeric variables | `age`, `bmi`, `children`, `charges` |
| Categorical variables | `sex`, `smoker`, `region` (+ derived `bmi_category`, `age_group`) |
| Target of interest | `charges` (individual medical costs billed) |

## 4. Methodology

All statistical logic lives in reusable, documented functions under `src/`, orchestrated
from `notebook.ipynb`:

1. **Cleaning** (`data_loader.clean_data`) — dropped the single exact duplicate row
   (a data-entry artifact, not a genuine repeat observation), standardized categorical
   text casing, and added `bmi_category` (WHO cutoffs) and `age_group` for interpretability.
2. **Assumption checks** (`stats_tests.check_normality`, `check_equal_variance`) —
   Shapiro-Wilk for normality per group, Levene's test for equal variance across groups.
3. **Test selection helper** (`select_and_run_two_group_test`,
   `select_and_run_multi_group_test`) — routes automatically: Welch's t-test if both
   groups are normal (Mann-Whitney U otherwise); one-way ANOVA if all groups are normal
   *and* variances are equal (Kruskal-Wallis otherwise).
4. **Categorical association** (`run_chi_square_test`) — chi-square test of independence
   with Cramer's V as effect size.
5. **Correlation** (`run_correlation_test`) — Pearson correlation with a Fisher
   z-transform confidence interval.
6. **Effect size calculator** — Cohen's d, rank-biserial correlation (derived from the
   Mann-Whitney U statistic), eta-squared (ANOVA), epsilon-squared (Kruskal-Wallis), and
   Cramer's V, all implemented explicitly rather than relying on a single library default.
7. **Confidence interval functions** — for a mean difference (Welch-style, unequal
   variance), a single mean, and a correlation coefficient.
8. **Multiple comparison correction** (`bonferroni_correction`, `holm_correction`) —
   applied across all 4 hypotheses tested on the same dataset.
9. **Baseline comparison** (`modeling.naive_vs_selected_two_group`,
   `naive_vs_selected_multi_group`) — the required "simple baseline": what a naive
   analyst who skips assumption-checking (plain Student's t-test / one-way ANOVA) would
   have concluded, compared directly against the test-selection helper's output.
10. **OLS regression baseline** (`modeling.fit_baseline_ols`) — translates the hypothesis
    findings into an interpretable dollar effect per factor, holding others constant.

## 5. Results & Key Visualizations

| Hypothesis | Test used | Statistic | p-value | Effect size | Holm-corrected p | Significant? |
|---|---|---|---|---|---|---|
| H1: Smokers vs. non-smokers charges differ | Mann-Whitney U | 283,859 | < 0.001 | Rank-biserial = -0.949 | < 0.001 | Yes |
| H2: Charges differ by region | Kruskal-Wallis H | 4.62 | 0.202 | Epsilon-sq = 0.001 | 0.202 | No |
| H3: Smoking status associated with sex | Chi-square | 7.47 | 0.006 | Cramer's V = 0.075 | 0.019 | Yes |
| H4: BMI correlates with charges | Pearson | r = 0.198 | < 0.001 | r = 0.198 | < 0.001 | Yes |

**Naive baseline comparison (the key methodological result):**

| Approach | Test | p-value | Significant? |
|---|---|---|---|
| Naive one-way ANOVA (region, assumptions unchecked) | ANOVA | 0.033 | **Yes (false positive)** |
| Test-selection helper (region, assumption-checked) | Kruskal-Wallis | 0.202 | No |

**OLS baseline coefficients** (dollar effect on charges, holding other factors constant):

| Factor | Effect ($) | 95% CI | p-value |
|---|---|---|---|
| Smoker (yes vs. no) | +23,847 | (23,036, 24,658) | < 0.001 |
| Age (per year) | +257 | (233, 280) | < 0.001 |
| BMI (per unit) | +339 | (283, 395) | < 0.001 |
| Children (per child) | +475 | (204, 745) | < 0.001 |
| Region: southeast (vs. northeast) | -1,035 | (-1,975, -96) | 0.031 |
| Region: southwest (vs. northeast) | -960 | (-1,898, -22) | 0.045 |
| Region: northwest (vs. northeast) | -349 | (-1,285, 586) | 0.464 |
| Sex: male (vs. female) | -129 | (-783, 524) | 0.697 |
| Model R-squared | 0.751 | — | — |

Figures generated in `figures/` (5 total, meeting the minimum requirement):

1. `01_exploratory_overview.png` — charges distribution, charges by smoker, BMI vs. charges
2. `02_multiple_comparison_correction.png` — raw vs. Holm/Bonferroni-corrected p-values
3. `03_effect_sizes_summary.png` — effect size magnitude across all 4 hypotheses
4. `04_naive_vs_selected_region.png` — naive ANOVA vs. properly-selected Kruskal-Wallis p-values
5. `05_ols_baseline_effects.png` — OLS coefficients with significance highlighted

## 6. Limitations & Risks

- The OLS regression coefficients for region are individually significant for southeast
  and southwest (vs. the northeast baseline category) even though the omnibus
  Kruskal-Wallis test found no overall region effect; this is a known tension between
  omnibus non-parametric tests and per-category linear regression coefficients, and
  should be flagged rather than silently resolved — the omnibus test result is treated
  as authoritative here since it doesn't assume normality/linearity.
- Cramer's V for the smoker-sex association (0.075) is statistically significant only
  because of the large sample size (n=1,337); practically, this is a negligible-to-small
  effect and should not be treated as a meaningful business finding on its own.
- The OLS model is a baseline for interpretability, not a tuned predictive model — no
  interaction terms (e.g. smoker x BMI, a well-documented interaction in this dataset)
  were included, so the individual coefficients may understate the true combined effect
  of smoking and obesity together.
- Multiple comparison correction was applied across 4 pre-specified hypotheses; if
  additional exploratory comparisons are run later, they should be included in the
  correction to avoid inflating the false-positive rate further.

## 7. Recommendation / Next Steps

- **Premiums can be justified by smoking status, age, BMI, and number of children** — all
  show robust, statistically significant, and practically meaningful effects.
- **Region should NOT be used as a pricing factor** based on this evidence — the omnibus
  test does not support a genuine regional difference, and using it risks an unjustified
  discrimination exposure.
- Investigate a smoker x BMI interaction term in a follow-up regression, since this
  dataset is well known to show a compounding effect between the two.
- Re-run the naive-vs-selected baseline comparison on any future dataset before trusting
  ANOVA/t-test results by default — this task demonstrated a real, consequential case
  where skipping the assumption check changes the business conclusion.

## 8. References

- Dataset: [Medical Cost Personal Datasets — Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Task 01 (EDA methodology precedent): `Week_1_Task/` in this repository
- SciPy documentation: [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
  (`shapiro`, `levene`, `ttest_ind`, `mannwhitneyu`, `f_oneway`, `kruskal`,
  `chi2_contingency`, `pearsonr`)
- statsmodels documentation: [OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)
