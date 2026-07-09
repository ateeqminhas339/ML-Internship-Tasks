"""
Telco Customer Churn - EDA Dashboard
Streamlit app covering univariate, bivariate, multivariate, distribution,
correlation, and hypothesis-driven analysis on the Telco Customer Churn dataset.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Telco Customer Churn - EDA Dashboard", layout="wide")


# ----------------------------------------------------------------------
# Data loading & cleaning (cached so it only runs once)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")

    # TotalCharges has blank strings for tenure == 0 customers -> coerce & fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)
    return df


df_raw = load_data()

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [
    c for c in df_raw.columns
    if c not in numeric_cols + ["Churn", "Churn_binary", "customerID"]
]

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------
st.sidebar.header("Filters")

contract_options = df_raw["Contract"].unique().tolist()
selected_contracts = st.sidebar.multiselect(
    "Select Contract Type", contract_options, default=contract_options
)

internet_options = df_raw["InternetService"].unique().tolist()
selected_internet = st.sidebar.multiselect(
    "Select Internet Service", internet_options, default=internet_options
)

df = df_raw[
    df_raw["Contract"].isin(selected_contracts)
    & df_raw["InternetService"].isin(selected_internet)
].copy()

if df.empty:
    st.warning("No rows match the selected filters. Adjust the filters in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------
# Header + top-level metrics
# ----------------------------------------------------------------------
st.title("📊 Telco Customer Churn - EDA Dashboard")
st.caption(
    "Interactive dashboard exploring the Telco Customer Churn dataset "
    "([Kaggle source](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))."
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Churned Customers", f"{(df['Churn'] == 'Yes').sum():,}")
col3.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

tabs = st.tabs([
    "🧩 Missing Data",
    "📈 Univariate",
    "🔀 Bivariate",
    "🧮 Multivariate",
    "🔗 Correlation",
    "🧪 Hypothesis Testing",
    "📝 Insights",
])

# ----------------------------------------------------------------------
# Tab 1 - Missing data patterns
# ----------------------------------------------------------------------
with tabs[0]:
    st.subheader("Missing Data Patterns")
    missing_summary = (
        df_raw.isna().sum().to_frame("missing_count")
        .assign(missing_pct=lambda x: (x["missing_count"] / len(df_raw) * 100).round(2))
        .sort_values("missing_count", ascending=False)
    )
    st.dataframe(missing_summary, use_container_width=True)
    st.markdown(
        "**Finding:** the raw `TotalCharges` column contains 11 blank values. "
        "Every one of them belongs to a customer with `tenure == 0` "
        "(a brand-new customer with no billing history yet) — this is a "
        "structurally explainable pattern rather than a random data-entry "
        "error, so the missing values were imputed with `0` instead of the "
        "column mean."
    )

# ----------------------------------------------------------------------
# Tab 2 - Univariate analysis
# ----------------------------------------------------------------------
with tabs[1]:
    st.subheader("Churn Distribution")
    churn_counts = df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "count"]
    fig = px.pie(churn_counts, names="Churn", values="count", hole=0.4,
                 color="Churn", color_discrete_map={"Yes": "#c44e52", "No": "#4c72b0"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numeric Feature Distribution")
    num_col = st.selectbox("Choose a numeric column", numeric_cols, key="uni_numeric")
    fig = px.histogram(df, x=num_col, nbins=30, marginal="box",
                        title=f"Distribution of {num_col}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Categorical Feature Counts")
    cat_col = st.selectbox("Choose a categorical column", categorical_cols, key="uni_cat")
    counts = df[cat_col].value_counts().reset_index()
    counts.columns = [cat_col, "count"]
    fig = px.bar(counts, x=cat_col, y="count", title=f"Counts of {cat_col}")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Tab 3 - Bivariate analysis
# ----------------------------------------------------------------------
with tabs[2]:
    st.subheader("Churn Rate by Category")
    cat_col_b = st.selectbox("Choose a categorical column", categorical_cols, key="bi_cat")
    ct = pd.crosstab(df[cat_col_b], df["Churn"], normalize="index") * 100
    ct = ct.reset_index().melt(id_vars=cat_col_b, var_name="Churn", value_name="pct")
    fig = px.bar(ct, x=cat_col_b, y="pct", color="Churn", barmode="stack",
                 color_discrete_map={"Yes": "#c44e52", "No": "#4c72b0"},
                 title=f"Churn rate (%) by {cat_col_b}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numeric Feature vs. Churn")
    num_col_b = st.selectbox("Choose a numeric column", numeric_cols, key="bi_num")
    fig = px.box(df, x="Churn", y=num_col_b, color="Churn",
                 color_discrete_map={"Yes": "#c44e52", "No": "#4c72b0"},
                 title=f"{num_col_b} by Churn")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Tab 4 - Multivariate analysis
# ----------------------------------------------------------------------
with tabs[3]:
    st.subheader("Contract Type x Internet Service -> Churn Rate")
    pivot = pd.pivot_table(
        df, index="Contract", columns="InternetService",
        values="Churn_binary", aggfunc="mean",
    ) * 100
    fig = px.imshow(pivot, text_auto=".1f", color_continuous_scale="Reds",
                     labels=dict(color="Churn rate (%)"),
                     title="Churn Rate (%) by Contract x Internet Service")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tenure Group x Contract -> Churn Rate")
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    )
    grp = (
        df.groupby(["tenure_group", "Contract"])["Churn_binary"]
        .mean().mul(100).reset_index(name="churn_rate")
    )
    fig = px.bar(grp, x="tenure_group", y="churn_rate", color="Contract",
                 barmode="group", title="Churn Rate (%) by Tenure Group and Contract")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pairwise Numeric Relationships")
    fig = px.scatter_matrix(df, dimensions=numeric_cols, color="Churn",
                             color_discrete_map={"Yes": "#c44e52", "No": "#4c72b0"})
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Tab 5 - Correlation analysis
# ----------------------------------------------------------------------
with tabs[4]:
    st.subheader("Correlation Heatmap (Numeric Features)")
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Point-biserial Correlation with Churn")
    pb_results = []
    for col in numeric_cols:
        r, p = stats.pointbiserialr(df["Churn_binary"], df[col])
        pb_results.append({"feature": col, "correlation": r, "p_value": p})
    pb_df = pd.DataFrame(pb_results).sort_values("correlation")
    fig = px.bar(pb_df, x="correlation", y="feature", orientation="h",
                 title="Point-biserial Correlation with Churn")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cramer's V (Categorical Association with Churn)")

    def cramers_v(x, y):
        ct = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(ct)[0]
        n = ct.sum().sum()
        r, k = ct.shape
        return np.sqrt((chi2 / n) / (min(r - 1, k - 1)))

    cramers_results = [
        {"feature": c, "cramers_v": cramers_v(df[c], df["Churn"])}
        for c in categorical_cols
    ]
    cramers_df = pd.DataFrame(cramers_results).sort_values("cramers_v", ascending=False)
    fig = px.bar(cramers_df, x="cramers_v", y="feature", orientation="h",
                 title="Cramer's V by Feature")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Tab 6 - Hypothesis testing
# ----------------------------------------------------------------------
with tabs[5]:
    st.subheader("Hypothesis-Driven Testing (alpha = 0.05)")

    def chi2_test(col):
        ct = pd.crosstab(df[col], df["Churn"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        return chi2, p

    def mannwhitney_test(col):
        a = df.loc[df["Churn"] == "Yes", col]
        b = df.loc[df["Churn"] == "No", col]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return u, p

    hypotheses = [
        ("H1: Contract type is associated with churn", "chi2", "Contract"),
        ("H2: Tenure differs between churned and retained customers", "mwu", "tenure"),
        ("H3: Internet service type is associated with churn", "chi2", "InternetService"),
        ("H4: MonthlyCharges differs between churned and retained customers", "mwu", "MonthlyCharges"),
        ("H5: TechSupport subscription is associated with churn", "chi2", "TechSupport"),
    ]

    rows = []
    for label, kind, col in hypotheses:
        if kind == "chi2":
            stat, p = chi2_test(col)
            test_name = "Chi-square"
        else:
            stat, p = mannwhitney_test(col)
            test_name = "Mann-Whitney U"
        rows.append({
            "Hypothesis": label,
            "Test": test_name,
            "Statistic": round(stat, 2),
            "p-value": f"{p:.4g}",
            "Result (alpha=0.05)": "Reject H0 (significant)" if p < 0.05 else "Fail to reject H0",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown(
        "Each row states a null hypothesis (no association / no difference) "
        "and reports whether the data provides enough evidence to reject it "
        "at the 5% significance level."
    )

# ----------------------------------------------------------------------
# Tab 7 - Insights
# ----------------------------------------------------------------------
with tabs[6]:
    st.subheader("Key Insights")
    st.markdown(
        """
- **Missing data** in `TotalCharges` is structural (tenure = 0 customers), not random.
- **Class imbalance:** churn is a minority class (~27% overall) — matters for future modelling.
- **Contract type** is the strongest churn driver: month-to-month customers churn far more
  than one/two-year contracts.
- **Tenure** and **MonthlyCharges** both differ significantly by churn status — churners tend
  to be newer customers paying more per month.
- **Fiber-optic internet** customers churn more than DSL/no-internet customers, and this
  compounds with contract type (highest churn = month-to-month + fiber optic).
- **Lack of add-on services** (Online Security, Tech Support) is strongly associated with
  higher churn — potential retention levers.
- `tenure` and `TotalCharges` are highly collinear — worth addressing before modelling.
        """
    )
    st.caption("Use the sidebar filters to explore how these patterns change across contract and internet-service segments.")
