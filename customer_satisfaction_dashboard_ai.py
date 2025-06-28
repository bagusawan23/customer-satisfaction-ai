import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

# Load model
@st.cache_resource
def load_model():
    return joblib.load("customer_satisfaction_model_full.pkl")

model = load_model()

# OpenRouter API
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "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"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# UI
st.title("📦 Prediksi & Rekomendasi AI untuk Kepuasan Pelanggan")
uploaded_file = st.file_uploader("📁 Upload file Excel", type=["xlsx"])

if uploaded_file is not None:
    df_input = pd.read_excel(uploaded_file)
    try:
        predictions = model.predict(df_input)
        df_input["predicted_customer_satisfaction"] = predictions
        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df_input)

        # Visualisasi distribusi
        st.subheader("🔍 Distribusi Prediksi Kepuasan")
        fig, ax = plt.subplots()
        sns.histplot(predictions, bins=20, kde=True, ax=ax)
        ax.set_xlabel("Predicted Customer Satisfaction")
        st.pyplot(fig)

        # Analisis AI berdasarkan agregat data
        avg_sla = df_input["sla_fulfillment"].mean()
        avg_complaint = df_input["num_complaints_per_100_orders"].mean()
        avg_response = df_input["response_time_hours"].mean()
        avg_delay = df_input["delivery_time_diff_days"].mean()

        prompt = f"""
Kepuasan pelanggan 2021 turun dari 98.2% ke 85%. Target 2022 adalah 95%.

Fitur yang tersedia:
- SLA pemenuhan order: {avg_sla:.1f}%
- Jumlah komplain per 100 order: {avg_complaint:.1f}
- Waktu respons layanan pelanggan: rata-rata {avg_response:.1f} jam
- Keterlambatan pengiriman: rata-rata {avg_delay:.1f} hari

Tugas kamu:
1. Apakah target 95% bisa dicapai?
2. Rekomendasikan minimal 3 strategi berbasis data untuk mencapainya.
Jawab dalam Bahasa Indonesia.
"""

        if st.button("🤖 Minta Rekomendasi AI"):
            payload = {
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [
                    {"role": "system", "content": "Kamu adalah analis AI ahli logistik dan pelayanan pelanggan."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                st.subheader("🧠 Rekomendasi AI:")
                st.markdown(content)
            else:
                st.error(f"Gagal memanggil API: {response.status_code}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("Silakan upload file Excel terlebih dahulu.")
