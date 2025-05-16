import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF

def generate_eda_report(df: pd.DataFrame, output_dir: str = "data/eda_report"):
    os.makedirs(output_dir, exist_ok=True)

    for col in df.columns:
        try:
            plt.figure(figsize=(6, 4))
            df[col].dropna().hist(bins=20, edgecolor='black')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            filepath = os.path.join(output_dir, f"{col}_hist.png")
            plt.savefig(filepath)
            plt.close()
        except Exception as e:
            print(f"⚠️ Could not plot histogram for {col}: {e}")

    # Correlation heatmap
    try:
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] >= 2:
            plt.figure(figsize=(10, 8))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
            plt.close()
    except Exception as e:
        print(f"⚠️ Could not generate heatmap: {e}")

    return f"EDA report images saved to: {output_dir}"


def export_eda_to_pdf(output_dir: str = "data/eda_report", output_pdf: str = "data/eda_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"{output_dir} does not exist")

    image_files = [f for f in sorted(os.listdir(output_dir)) if f.endswith(".png")]
    if not image_files:
        raise ValueError("No image files found in EDA report directory.")

    for img_file in image_files:
        full_path = os.path.join(output_dir, img_file)
        pdf.add_page()
        pdf.image(full_path, x=10, y=10, w=180)

    pdf.output(output_pdf)
    return output_pdf
