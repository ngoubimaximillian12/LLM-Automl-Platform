import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

DATA_DIR = "data"

def show_data_preview():
    st.subheader("🔍 Data Preview + Insights")

    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".csv", ".xlsx", ".json", ".parquet"))]
    if not files:
        st.info("📁 No uploaded datasets found.")
        return

    selected_file = st.selectbox("📂 Choose file to explore", files)

    try:
        if selected_file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, selected_file))
        elif selected_file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(DATA_DIR, selected_file))
        elif selected_file.endswith(".json"):
            df = pd.read_json(os.path.join(DATA_DIR, selected_file))
        elif selected_file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(DATA_DIR, selected_file))
        else:
            st.error("Unsupported file format.")
            return

        st.write(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(100))

        st.markdown("### 🧾 Dataset Summary Stats")
        st.json({
            "Missing values": int(df.isnull().sum().sum()),
            "Duplicate rows": int(df.duplicated().sum()),
            "Numeric columns": len(df.select_dtypes(include='number').columns),
            "Categorical columns": len(df.select_dtypes(include='object').columns),
            "Memory Usage (MB)": round(df.memory_usage().sum() / 1e6, 2)
        })

        target_col = st.selectbox("🎯 Choose target column (optional)", df.columns)
        if target_col:
            st.markdown("### 📊 Target Distribution")
            st.bar_chart(df[target_col].value_counts())

        st.markdown("### 📈 Auto Visualizations")
        numeric_df = df.select_dtypes(include='number').dropna()
        if numeric_df.empty:
            st.warning("⚠️ No numeric columns found for plotting.")
            return

        # 1. Histogram
        st.markdown("**1️⃣ Histogram (First numeric column)**")
        fig, ax = plt.subplots()
        ax.hist(numeric_df.iloc[:, 0], bins=20, color="skyblue")
        ax.set_xlabel(numeric_df.columns[0])
        st.pyplot(fig)

        # 2. Correlation heatmap
        st.markdown("**2️⃣ Correlation Heatmap**")
        fig, ax = plt.subplots()
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # 3. PCA
        if numeric_df.shape[1] >= 2:
            st.markdown("**3️⃣ PCA (2D Projection)**")
            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric_df)
            pca = PCA(n_components=2)
            comp = pca.fit_transform(scaled)
            pca_df = pd.DataFrame(comp, columns=["PC1", "PC2"])
            fig, ax = plt.subplots()
            sns.scatterplot(data=pca_df, x="PC1", y="PC2", ax=ax)
            st.pyplot(fig)

        # 4. KMeans Clustering
        st.markdown("**4️⃣ KMeans Clustering (k=3)**")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(numeric_df)
        fig, ax = plt.subplots()
        sns.scatterplot(x=comp[:, 0], y=comp[:, 1], hue=labels, ax=ax)
        st.pyplot(fig)

        # 5. Boxplot
        st.markdown("**5️⃣ Boxplot**")
        fig, ax = plt.subplots()
        sns.boxplot(data=numeric_df, ax=ax)
        st.pyplot(fig)

        # 6. Outlier Detection
        st.markdown("### 🚨 Outlier Flagging (Z-Score)")
        z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std()
        outliers = (abs(z_scores) > 3).sum()
        st.write("Outliers per column:")
        st.json(outliers.to_dict())

        # 7. Categorical Summary
        categoricals = df.select_dtypes(include='object').columns.tolist()
        if categoricals:
            st.markdown("### 🔤 Categorical Feature Summary")
            for col in categoricals:
                st.markdown(f"**{col}**: {df[col].nunique()} unique values")
                st.bar_chart(df[col].value_counts().head(10))

        # 8. Column Data Types
        st.markdown("### 🧬 Column Types")
        dtype_df = pd.DataFrame(df.dtypes, columns=["Type"]).reset_index().rename(columns={"index": "Column"})
        dtype_df["Type"] = dtype_df["Type"].astype(str)
        st.dataframe(dtype_df)

        # 9. Download button
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=df.to_csv(index=False),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        # 10. LLM-based fallback insight
        try:
            from backend.llm_generator import deepseek_fallback
            preview_text = df.head(3).to_markdown()
            llm_response = deepseek_fallback(f"Based on this dataset, give EDA insights:\n\n{preview_text}")
            st.markdown("### 🤖 LLM Insight")
            st.markdown(llm_response)
        except:
            pass

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
