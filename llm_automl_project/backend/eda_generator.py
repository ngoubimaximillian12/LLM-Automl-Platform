import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF

def generate_eda_report(df: pd.DataFrame, output_dir: str = "data/eda_report"):
    os.makedirs(output_dir, exist_ok=True)
    for col in df.columns:
        try:
            plt.figure()
            df[col].dropna().hist(bins=20)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.savefig(f"{output_dir}/{col}_hist.png")
            plt.close()
        except Exception as e:
            print(f"Could not plot histogram for {col}: {e}")
    try:
        plt.figure(figsize=(10, 8))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()
    except Exception as e:
        print(f"Could not generate heatmap: {e}")
    return f"EDA report images saved to: {output_dir}"

def export_eda_to_pdf(output_dir: str = "data/eda_report", output_pdf: str = "data/eda_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for img in sorted(os.listdir(output_dir)):
        if img.endswith(".png"):
            pdf.add_page()
            pdf.image(f"{output_dir}/{img}", x=10, y=10, w=180)
    pdf.output(output_pdf)
    return output_pdf
