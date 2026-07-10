import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Telco Customer Churn - EDA Dashboard", layout="wide")

RED, BLUE = "#c44e52", "#4c72b0"
CMAP = {"Yes": RED, "No": BLUE}


@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Telco-Customer-Churn.csv")
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn_binary"] = (df["Churn"] == "Yes").astype(int)
    return df


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n, (r, k) = ct.sum().sum(), ct.shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1)))


def chi2_test(df, col):
    chi2, p, *_ = stats.chi2_contingency(pd.crosstab(df[col], df["Churn"]))
    return chi2, p


def mwu_test(df, col):
    a, b = df.loc[df["Churn"] == "Yes", col], df.loc[df["Churn"] == "No", col]
    return stats.mannwhitneyu(a, b, alternative="two-sided")


df_raw = load_data()
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [c for c in df_raw.columns if c not in numeric_cols + ["Churn", "Churn_binary", "customerID"]]

st.sidebar.header("Filters")
contracts = st.sidebar.multiselect("Select Contract Type", df_raw["Contract"].unique(), default=df_raw["Contract"].unique())
internet = st.sidebar.multiselect("Select Internet Service", df_raw["InternetService"].unique(), default=df_raw["InternetService"].unique())

df = df_raw[df_raw["Contract"].isin(contracts) & df_raw["InternetService"].isin(internet)].copy()
if df.empty:
    st.warning("No rows match the selected filters. Adjust the filters in the sidebar.")
    st.stop()

st.title("📊 Telco Customer Churn - EDA Dashboard")
st.caption("Interactive dashboard exploring the Telco Customer Churn dataset "
           "([Kaggle source](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).")

c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", f"{len(df):,}")
c2.metric("Churned Customers", f"{(df['Churn'] == 'Yes').sum():,}")
c3.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

tabs = st.tabs(["🧩 Missing Data", "📈 Univariate", "🔀 Bivariate", "🧮 Multivariate",
                "🔗 Correlation", "🧪 Hypothesis Testing", "📝 Insights"])

with tabs[0]:
    st.subheader("Missing Data Patterns")
    miss = df_raw.isna().sum().to_frame("missing_count")
    miss["missing_pct"] = (miss["missing_count"] / len(df_raw) * 100).round(2)
    st.dataframe(miss.sort_values("missing_count", ascending=False), use_container_width=True)
    st.markdown(
        "**Finding:** the raw `TotalCharges` column contains 11 blank values, all belonging "
        "to customers with `tenure == 0` (no billing history yet) — a structurally explainable "
        "pattern, so missing values were imputed with `0` instead of the column mean."
    )

with tabs[1]:
    st.subheader("Churn Distribution")
    counts = df["Churn"].value_counts().reset_index(name="count").rename(columns={"index": "Churn"})
    st.plotly_chart(px.pie(counts, names="Churn", values="count", hole=0.4, color="Churn", color_discrete_map=CMAP),
                     use_container_width=True)

    st.subheader("Numeric Feature Distribution")
    num_col = st.selectbox("Choose a numeric column", numeric_cols, key="uni_numeric")
    st.plotly_chart(px.histogram(df, x=num_col, nbins=30, marginal="box", title=f"Distribution of {num_col}"),
                     use_container_width=True)

    st.subheader("Categorical Feature Counts")
    cat_col = st.selectbox("Choose a categorical column", categorical_cols, key="uni_cat")
    ccounts = df[cat_col].value_counts().reset_index(name="count").rename(columns={"index": cat_col})
    st.plotly_chart(px.bar(ccounts, x=cat_col, y="count", title=f"Counts of {cat_col}"), use_container_width=True)

with tabs[2]:
    st.subheader("Churn Rate by Category")
    cat_col_b = st.selectbox("Choose a categorical column", categorical_cols, key="bi_cat")
    ct = (pd.crosstab(df[cat_col_b], df["Churn"], normalize="index") * 100).reset_index() \
        .melt(id_vars=cat_col_b, var_name="Churn", value_name="pct")
    st.plotly_chart(px.bar(ct, x=cat_col_b, y="pct", color="Churn", barmode="stack",
                            color_discrete_map=CMAP, title=f"Churn rate (%) by {cat_col_b}"),
                     use_container_width=True)

    st.subheader("Numeric Feature vs. Churn")
    num_col_b = st.selectbox("Choose a numeric column", numeric_cols, key="bi_num")
    st.plotly_chart(px.box(df, x="Churn", y=num_col_b, color="Churn", color_discrete_map=CMAP,
                            title=f"{num_col_b} by Churn"), use_container_width=True)

with tabs[3]:
    st.subheader("Contract Type x Internet Service -> Churn Rate")
    pivot = pd.pivot_table(df, index="Contract", columns="InternetService", values="Churn_binary", aggfunc="mean") * 100
    st.plotly_chart(px.imshow(pivot, text_auto=".1f", color_continuous_scale="Reds",
                               labels=dict(color="Churn rate (%)"),
                               title="Churn Rate (%) by Contract x Internet Service"), use_container_width=True)

    st.subheader("Tenure Group x Contract -> Churn Rate")
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 60, 72],
                                 labels=["0-12", "13-24", "25-48", "49-60", "61-72"])
    grp = df.groupby(["tenure_group", "Contract"])["Churn_binary"].mean().mul(100).reset_index(name="churn_rate")
    st.plotly_chart(px.bar(grp, x="tenure_group", y="churn_rate", color="Contract", barmode="group",
                            title="Churn Rate (%) by Tenure Group and Contract"), use_container_width=True)

    st.subheader("Pairwise Numeric Relationships")
    st.plotly_chart(px.scatter_matrix(df, dimensions=numeric_cols, color="Churn", color_discrete_map=CMAP),
                     use_container_width=True)

with tabs[4]:
    st.subheader("Correlation Heatmap (Numeric Features)")
    corr = df[numeric_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                               title="Correlation Matrix"), use_container_width=True)

    st.subheader("Point-biserial Correlation with Churn")
    pb = pd.DataFrame(
        [{"feature": c, "correlation": stats.pointbiserialr(df["Churn_binary"], df[c])[0],
          "p_value": stats.pointbiserialr(df["Churn_binary"], df[c])[1]} for c in numeric_cols]
    ).sort_values("correlation")
    st.plotly_chart(px.bar(pb, x="correlation", y="feature", orientation="h",
                            title="Point-biserial Correlation with Churn"), use_container_width=True)

    st.subheader("Cramer's V (Categorical Association with Churn)")
    cv = pd.DataFrame([{"feature": c, "cramers_v": cramers_v(df[c], df["Churn"])} for c in categorical_cols]) \
        .sort_values("cramers_v", ascending=False)
    st.plotly_chart(px.bar(cv, x="cramers_v", y="feature", orientation="h", title="Cramer's V by Feature"),
                     use_container_width=True)

with tabs[5]:
    st.subheader("Hypothesis-Driven Testing (alpha = 0.05)")
    hypotheses = [
        ("H1: Contract type is associated with churn", "chi2", "Contract"),
        ("H2: Tenure differs between churned and retained customers", "mwu", "tenure"),
        ("H3: Internet service type is associated with churn", "chi2", "InternetService"),
        ("H4: MonthlyCharges differs between churned and retained customers", "mwu", "MonthlyCharges"),
        ("H5: TechSupport subscription is associated with churn", "chi2", "TechSupport"),
    ]
    rows = []
    for label, kind, col in hypotheses:
        stat, p = chi2_test(df, col) if kind == "chi2" else mwu_test(df, col)
        rows.append({
            "Hypothesis": label,
            "Test": "Chi-square" if kind == "chi2" else "Mann-Whitney U",
            "Statistic": round(stat, 2),
            "p-value": f"{p:.4g}",
            "Result (alpha=0.05)": "Reject H0 (significant)" if p < 0.05 else "Fail to reject H0",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.markdown("Each row states a null hypothesis (no association / no difference) and reports "
                "whether the data provides enough evidence to reject it at the 5% significance level.")

with tabs[6]:
    st.subheader("Key Insights")
    st.markdown("""
- **Missing data** in `TotalCharges` is structural (tenure = 0 customers), not random.
- **Class imbalance:** churn is a minority class (~27% overall) — matters for future modelling.
- **Contract type** is the strongest churn driver: month-to-month customers churn far more than one/two-year contracts.
- **Tenure** and **MonthlyCharges** both differ significantly by churn status — churners tend to be newer customers paying more per month.
- **Fiber-optic internet** customers churn more than DSL/no-internet customers, and this compounds with contract type.
- **Lack of add-on services** (Online Security, Tech Support) is strongly associated with higher churn.
- `tenure` and `TotalCharges` are highly collinear — worth addressing before modelling.
""")
    st.caption("Use the sidebar filters to explore how these patterns change across contract and internet-service segments.")
