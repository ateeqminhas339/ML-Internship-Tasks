# Task 04 — Statistical Analysis and Hypothesis Testing

Hypothesis-driven statistical analysis of the [Medical Cost Personal](https://www.kaggle.com/datasets/mirichoi0218/insurance)
(insurance) dataset, addressing a health insurer's premium-pricing question: do smokers,
BMI, and region genuinely affect medical charges?

## What this task does

- Implements a **statistical test selection helper** that checks normality (Shapiro-Wilk)
  and equal variance (Levene's test) first, then automatically picks the correct test:
  Welch's t-test vs. Mann-Whitney U (2 groups), one-way ANOVA vs. Kruskal-Wallis (3+ groups)
- Implements an **effect size calculator**: Cohen's d, rank-biserial correlation,
  eta-squared, epsilon-squared, Cramer's V, and Pearson/Spearman r
- Implements **confidence interval functions**: for a mean difference, a single mean,
  and a correlation coefficient (Fisher z-transform)
- Applies **Holm-Bonferroni multiple comparison correction** across 4 simultaneous hypotheses
- **Compares against a naive baseline** (blindly applying t-test/ANOVA without checking
  assumptions) — and finds a real case where the naive approach reaches the **wrong**
  conclusion (region falsely appears significant)
- Fits a simple OLS regression baseline to translate findings into dollar impact per factor

## Folder structure

```
Task_04_Statistical_Analysis_and_Hypothesis_Testing/
├── README.md                  # this file
├── notebook.ipynb             # main solution notebook (run this)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # loading + documented cleaning
│   ├── stats_tests.py         # test selection helper, effect sizes, CIs, multi-comparison correction
│   └── modeling.py            # OLS baseline + naive-vs-selected test comparison
├── figures/                   # 5 generated charts (.png)
├── reports/
│   └── REPORT.md              # written report
├── data/
│   └── insurance.csv
├── requirements.txt
└── .gitignore
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```
Run all cells top-to-bottom (Kernel → Restart & Run All). Figures are saved automatically
to `figures/`.

## Key results

| Hypothesis | Test used | p-value | Effect size | Significant? |
|---|---|---|---|---|
| H1: Smokers vs. non-smokers charges differ | Mann-Whitney U | < 0.001 | Rank-biserial ≈ 0.95 | Yes |
| H2: Charges differ by region | Kruskal-Wallis | 0.202 | Epsilon-sq ≈ 0.001 | **No** |
| H3: Smoking associated with sex | Chi-square | 0.006 | Cramer's V ≈ 0.075 | Yes (small effect) |
| H4: BMI correlates with charges | Pearson | < 0.001 | r ≈ 0.198 | Yes (small-moderate) |

**Critical finding:** a naive, assumption-unchecked one-way ANOVA reports region as
significant (p = 0.033) — the properly-selected Kruskal-Wallis test correctly shows it is
**not** (p = 0.202). Pricing premiums by region on this data would be statistically
unjustified. See `reports/REPORT.md` for the full write-up.
