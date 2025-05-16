import pandas as pd
import numpy as np
from scipy.stats import skew
from dotenv import load_dotenv
import os

# Optional: LLM fallback
try:
    from backend.llm_generator import deepseek_fallback
except ImportError:
    deepseek_fallback = None

load_dotenv()


def explain_numeric_summary(df: pd.DataFrame) -> str:
    report = []
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        return "No numeric features found in the dataset."

    for col in numeric_df.columns:
        stats = numeric_df[col].describe()
        skewness = skew(numeric_df[col].dropna())
        line = (
            f"📊 **{col}**: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"min={stats['min']:.2f}, max={stats['max']:.2f}, skewness={skewness:.2f}"
        )
        if abs(skewness) > 1:
            line += " → ⚠️ Highly skewed"
        elif abs(skewness) > 0.5:
            line += " → ⚠️ Moderately skewed"
        report.append(line)

    return "\n".join(report)


def explain_missing_data(df: pd.DataFrame) -> str:
    missing = df.isnull().sum()
    total = df.shape[0]
    report = []
    for col, count in missing.items():
        if count > 0:
            percent = (count / total) * 100
            report.append(f"🕳️ Column **{col}** has {count} missing values ({percent:.1f}%)")
    return "\n".join(report) or "✅ No missing values found."


def explain_correlations(df: pd.DataFrame) -> str:
    corr = df.select_dtypes(include=np.number).corr()
    report = []
    threshold = 0.75
    for col1 in corr.columns:
        for col2 in corr.columns:
            if col1 != col2 and abs(corr.loc[col1, col2]) >= threshold:
                report.append(f"🔗 High correlation between **{col1}** and **{col2}**: {corr.loc[col1, col2]:.2f}")
    return "\n".join(report) or "✅ No highly correlated feature pairs."


def generate_explanations(df: pd.DataFrame) -> str:
    explanations = [
        "### 🧠 Data Summary Explanation",
        explain_numeric_summary(df),
        "\n### ❓ Missing Values Analysis",
        explain_missing_data(df),
        "\n### 🔍 Correlation Insights",
        explain_correlations(df),
    ]

    return "\n\n".join(explanations)


def generate_llm_explanation(df: pd.DataFrame) -> str:
    """Optional LLM explanation fallback if enabled."""
    if deepseek_fallback is None:
        return "⚠️ DeepSeek is not available. Cannot generate LLM explanation."

    try:
        sample = df.head(5).to_markdown()
        prompt = f"Analyze this dataset sample and explain any patterns or issues:\n\n{sample}"
        return deepseek_fallback(prompt)
    except Exception as e:
        return f"❌ LLM fallback failed: {e}"
