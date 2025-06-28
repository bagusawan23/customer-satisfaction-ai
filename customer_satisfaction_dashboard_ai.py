import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor

# Fungsi melatih model langsung
@st.cache_resource
def train_model(X, y):
    numeric_features = ["sla_fulfillment", "num_complaints_per_100_orders", "response_time_hours", "delivery_time_diff_days"]
    categorical_features = ["region", "product_type"]
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor())
    ])
    
    pipeline.fit(X, y)
    return pipeline

# Upload data
st.title("📦 Prediksi & Rekomendasi Kepuasan Pelanggan")
file = st.file_uploader("Upload file Excel", type="xlsx")

if file:
    df = pd.read_excel(file)
    if "customer_satisfaction" in df.columns:
        X = df.drop(columns=["customer_satisfaction"])
        y = df["customer_satisfaction"]
        model = train_model(X, y)
        
        st.success("Model berhasil dilatih dari data Excel yang diupload.")
        
        preds = model.predict(X)
        df["predicted_satisfaction"] = preds
        st.dataframe(df.head())

        # Visualisasi
        st.subheader("Distribusi Prediksi Kepuasan")
        fig, ax = plt.subplots()
        sns.histplot(preds, bins=20, kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Kolom 'customer_satisfaction' tidak ditemukan. Pastikan file Excel memuatnya.")
else:
    st.info("Silakan upload file Excel.")
