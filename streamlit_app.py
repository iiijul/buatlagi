pip install -r requirements.txt
streamlit run streamlit_app.pycd /workspaces/buatlagi
python3 -m streamlit run streamlit_app.pyimport streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import pickle
import io
from pathlib import Path

st.set_page_config(page_title="BuatLagi — Dashboard Anggaran", layout="wide", initial_sidebar_state="expanded")

# --- Helpers
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_model(path: str):
    model_path = Path(path)
    if not model_path.exists():
        return None
    try:
        # Try joblib first
        model = joblib.load(path)
        return model
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception:
        return None

# --- Paths
DATA_PATH = "data/02_realisasi_anggaran_klasifikasi.csv"
MODEL_PATH = "model/knn.pkcls"

# --- Load
df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# --- Sidebar controls
with st.sidebar:
    st.title("Kontrol")
    st.markdown("Filter data untuk eksplorasi dan prediksi")
    kementerian = st.multiselect("Kementerian", options=sorted(df['nama_kementerian'].unique()), default=None)
    provinsi = st.multiselect("Provinsi", options=sorted(df['provinsi'].unique()), default=None)
    jenis = st.multiselect("Jenis Belanja", options=sorted(df['jenis_belanja_utama'].unique()), default=None)
    min_pagu = st.slider("Pagu (miliar) minimal", float(df['pagu_miliar'].min()), float(df['pagu_miliar'].max()), float(df['pagu_miliar'].min()))
    sample_size = st.slider("Jumlah baris contoh (untuk preview)", 5, 200, 25)
    st.divider()
    st.markdown("## Model & Prediksi")
    st.write("Model ditemukan:" , "✅" if model is not None else "❌")
    st.caption("Jika model tidak cocok, gunakan tombol 'Prediksi dari baris' untuk memakai fitur numerik dari dataset.")
    st.button("Segarkan data/model")

# --- Filter data
filt = df.copy()
if kementerian:
    filt = filt[filt['nama_kementerian'].isin(kementerian)]
if provinsi:
    filt = filt[filt['provinsi'].isin(provinsi)]
if jenis:
    filt = filt[filt['jenis_belanja_utama'].isin(jenis)]
filt = filt[filt['pagu_miliar'] >= min_pagu]

# --- Layout: KPIs and plots
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Jumlah Satker", len(filt))
with kpi2:
    st.metric("Rata-rata Pagu (miliar)", f"{filt['pagu_miliar'].mean():,.2f}")
with kpi3:
    st.metric("Rata-rata Skor IKPA", f"{filt['skor_ikpa'].mean():,.2f}")
with kpi4:
    achieved = (filt['realisasi_tercapai_95persen'] == 'Ya').mean() * 100
    st.metric("Realisasi >=95% (%)", f"{achieved:.1f}%")

st.markdown("---")

# Main columns
left, right = st.columns((2,1))

with left:
    st.subheader("Visualisasi Interaktif")
    # Scatter: pagu vs realisasi_tw3_persen
    fig = px.scatter(filt, x='pagu_miliar', y='realisasi_tw3_persen', color='nama_kementerian', hover_data=['kode_satker','provinsi','skor_ikpa'], size='jumlah_spm', opacity=0.8,
                     title='Pagu vs Realisasi TW3 (%)')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.histogram(filt, x='pagu_miliar', nbins=40, color='jenis_belanja_utama', title='Distribusi Pagu (miliar)')
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.box(filt, x='tipe_satker', y='skor_ikpa', color='tipe_satker', title='Skor IKPA per Tipe Satker')
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Komposisi per Kementerian / Provinsi")
    sun = px.sunburst(filt, path=['nama_kementerian','provinsi','tipe_satker'], values='pagu_miliar', color='skor_ikpa', title='Komposisi Pagu: Kementerian → Provinsi → Tipe')
    st.plotly_chart(sun, use_container_width=True)

    st.subheader("Tabel Data (preview)")
    st.dataframe(filt.head(sample_size))
    csv = filt.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV (filtered)", data=csv, file_name='filtered_buatlagi.csv', mime='text/csv')

with right:
    st.subheader("Panel Prediksi & Analisis")
    if model is None:
        st.warning("Model tidak ditemukan atau tidak dapat dimuat. Prediksi menggunakan model lokal tidak tersedia.")
    else:
        st.success("Model berhasil dimuat. Coba prediksi interaktif di bawah.")

    st.markdown("**Pilih Satker untuk Prediksi (pakai fitur numerik yang ada di dataset)**")
    kode = st.selectbox("Pilih kode_satker", options=filt['kode_satker'].tolist())
    row = filt[filt['kode_satker'] == kode].head(1)
    st.write(row.T)

    numeric_cols = list(df.select_dtypes(include=['number']).columns)
    chosen_features = st.multiselect("Fitur numerik untuk prediksi (urut penting)", options=numeric_cols, default=numeric_cols)

    if st.button("Prediksi dari baris ini"):
        X = row[chosen_features].fillna(0)
        # Ensure shape
        try:
            pred = model.predict(X)
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
            st.markdown("### Hasil Prediksi")
            st.write(pd.DataFrame({'prediksi':pred}))
            if proba is not None:
                st.write(pd.DataFrame(proba, columns=[f'P_{c}' for c in model.classes_]))
                # Show pie chart for probabilities
                probs = proba[0]
                labels = [str(c) for c in model.classes_]
                figp = px.pie(values=probs, names=labels, title='Probabilitas Prediksi')
                st.plotly_chart(figp, use_container_width=True)
        except Exception as e:
            st.error(f"Prediksi gagal: {e}")

    st.markdown("---")
    st.markdown("**Batch predict untuk filtered dataset**")
    if model is not None and st.button("Jalankan Prediksi Batch" ):
        try:
            Xf = filt[chosen_features].fillna(0)
            preds = model.predict(Xf)
            filt_result = filt.copy()
            filt_result['prediksi_model'] = preds
            st.dataframe(filt_result[['kode_satker','nama_kementerian','provinsi','pagu_miliar','prediksi_model']].head(100))
            st.success("Prediksi batch selesai — tampilkan di tabel.")
        except Exception as e:
            st.error(f"Prediksi batch gagal: {e}")

# --- Footer
st.markdown("---")
st.caption("Dashboard dibuat otomatis oleh asisten. Ingin fitur tambahan (SHAP, clustering, atau peta)? Beri tahu saya.")
