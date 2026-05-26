import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import pickle
from pathlib import Path
import traceback

try:
    import shap
except Exception:
    shap = None


def get_model_labels(model, proba=None):
    """Return a list of class labels for a model or infer from proba length."""
    if model is None:
        return []
    # common sklearn attribute
    if hasattr(model, 'classes_'):
        return [str(c) for c in list(model.classes_)]
    # some wrappers may use other names
    if hasattr(model, 'classes'):
        try:
            return [str(c) for c in list(model.classes)]
        except Exception:
            pass
    if hasattr(model, 'class_names'):
        try:
            return [str(c) for c in list(model.class_names)]
        except Exception:
            pass
    # Orange models may expose domain info
    try:
        domain = getattr(model, 'domain', None)
        if domain is not None:
            class_var = getattr(domain, 'class_var', None)
            if class_var is not None and hasattr(class_var, 'values'):
                return [str(v) for v in list(class_var.values)]
    except Exception:
        pass
    # fallback: infer from proba length
    if proba is not None:
        try:
            return [f'Class_{i}' for i in range(len(proba))]
        except Exception:
            pass
    return []

st.set_page_config(
    page_title="BuatLagi — Dashboard Anggaran",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROVINCE_COORDINATES = {
    'Aceh': (4.6951, 96.7494),
    'Bali': (-8.3405, 115.0920),
    'Bangka Belitung': (-2.7416, 106.4410),
    'Banten': (-6.1204, 106.1500),
    'Bengkulu': (-3.8000, 102.2656),
    'DKI Jakarta': (-6.2088, 106.8456),
    'DI Yogyakarta': (-7.7956, 110.3695),
    'Jawa Barat': (-6.9147, 107.6098),
    'Jawa Tengah': (-7.1500, 110.1403),
    'Jawa Timur': (-7.2504, 112.7688),
    'Kalimantan Timur': (0.5533, 117.1573),
    'Lampung': (-5.4264, 105.2613),
    'Nusa Tenggara Barat': (-8.5831, 116.2370),
    'Nusa Tenggara Timur': (-9.6949, 120.2273),
    'Papua': (-4.2693, 137.0800),
    'Riau': (0.5070, 101.4478),
    'Sulawesi Selatan': (-5.1477, 119.4327),
    'Sulawesi Tengah': (-0.4658, 121.1873),
    'Sulawesi Utara': (1.4783, 124.8390),
    'Sumatera Barat': (-0.9471, 100.4172),
    'Sumatera Utara': (2.9948, 99.0134),
}

DATA_PATH = "data/02_realisasi_anggaran_klasifikasi.csv"
MODEL_PATH = "model/knn.pkcls"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_model(path: str):
    model_path = Path(path)
    if not model_path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def try_load_model(path: str):
    """Attempt to load model and return (model, error_message).
    Keeps original load_model for caching but exposes error details for UI.
    """
    model_path = Path(path)
    if not model_path.exists():
        return None, f"Model file not found: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e:
        tb = traceback.format_exc()
        # try pickle as fallback
        try:
            with open(path, 'rb') as f:
                m = pickle.load(f)
                return m, None
        except Exception:
            return None, tb

@st.cache_data
def build_shap_explainer(_model, background: pd.DataFrame):
    if shap is None or _model is None:
        return None
    try:
        return shap.Explainer(_model.predict, background)
    except Exception:
        try:
            return shap.KernelExplainer(_model.predict, background)
        except Exception:
            return None

@st.cache_data
def get_province_centroids():
    return PROVINCE_COORDINATES

@st.cache_data
def prepare_map_data(df: pd.DataFrame):
    province_df = (
        df.groupby('provinsi', as_index=False)
        .agg(
            pagu_miliar=('pagu_miliar', 'sum'),
            skor_ikpa=('skor_ikpa', 'mean'),
            jumlah_satker=('kode_satker', 'count'),
        )
    )
    coords = get_province_centroids()
    province_df['lat'] = province_df['provinsi'].map(lambda key: coords.get(key, (None, None))[0])
    province_df['lon'] = province_df['provinsi'].map(lambda key: coords.get(key, (None, None))[1])
    return province_df.dropna(subset=['lat', 'lon'])

@st.cache_data
def get_numeric_columns(df: pd.DataFrame):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['realisasi_tercapai_95persen']]


def prepare_model_input(model, df_input: pd.DataFrame):
    """Prepare numpy array matching model.domain.attributes ordering when available.
    Accepts a DataFrame (one or more rows) and returns a 2D list/array.
    """
    # If Orange model with domain, use domain.attributes ordering
    if model is not None and hasattr(model, 'domain'):
        attrs = [a.name for a in model.domain.attributes]
        rows = []
        for _, r in df_input.iterrows():
            vals = []
            for a in attrs:
                if '=' in a:
                    # one-hot encoded category like 'tipe_satker=Kantor Pusat'
                    col, val = a.split('=', 1)
                    v = r.get(col, None)
                    try:
                        vals.append(1.0 if str(v) == val else 0.0)
                    except Exception:
                        vals.append(0.0)
                else:
                    # numeric attribute
                    v = r.get(a, 0)
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(0.0)
            rows.append(vals)
        return rows
    # Fallback: use numeric columns from dataframe order
    # Ensure df_input is DataFrame
    if isinstance(df_input, pd.Series):
        df_input = df_input.to_frame().T
    numeric = [c for c in df_input.select_dtypes(include=[np.number]).columns]
    return df_input[numeric].fillna(0).values.tolist()

# --- Load data and model
with st.spinner("Memuat data dan model..."):
    df = load_data(DATA_PATH)
    model, model_error = try_load_model(MODEL_PATH)
    numeric_cols = get_numeric_columns(df)
    shap_explainer = None
    if model is not None and shap is not None:
        background = df[numeric_cols].dropna().sample(n=min(50, len(df)), random_state=42)
        shap_explainer = build_shap_explainer(model, background)

st.title("📊 BuatLagi Dashboard Anggaran")
st.markdown(
    "Dashboard interaktif untuk eksplorasi realisasi anggaran, prediksi model, dan interpretabilitas SHAP."
)

with st.sidebar:
    st.header("Filter eksplorasi")
    kementerian = st.multiselect("Kementerian", sorted(df['nama_kementerian'].unique()), default=None)
    provinsi = st.multiselect("Provinsi", sorted(df['provinsi'].unique()), default=None)
    jenis = st.multiselect("Jenis Belanja", sorted(df['jenis_belanja_utama'].unique()), default=None)
    min_pagu = st.slider(
        "Pagu (miliar) minimal",
        float(df['pagu_miliar'].min()),
        float(df['pagu_miliar'].max()),
        float(df['pagu_miliar'].min()),
    )
    sample_size = st.slider("Jumlah baris contoh", 5, 200, 25)
    st.divider()
    st.header("Model dan prediksi")
    st.write("Model dimuat:", "✅" if model is not None else "❌")
    if model is None and 'model_error' in locals() and model_error is not None:
        with st.expander("Detail error pemuatan model"):
            st.code(model_error)
    st.write("SHAP dimuat:", "✅" if shap_explainer is not None else "⚠️")
    st.caption(
        "Gunakan tab Prediksi untuk melakukan inferensi baris, batch, atau input manual."
    )

filt = df.copy()
if kementerian:
    filt = filt[filt['nama_kementerian'].isin(kementerian)]
if provinsi:
    filt = filt[filt['provinsi'].isin(provinsi)]
if jenis:
    filt = filt[filt['jenis_belanja_utama'].isin(jenis)]
filt = filt[filt['pagu_miliar'] >= min_pagu]

if filt.empty:
    st.warning("Tidak ada data yang cocok dengan filter. Coba atur ulang filter di sidebar.")
    st.stop()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Satker", len(filt))
with kpi2:
    st.metric("Rata-rata Pagu", f"{filt['pagu_miliar'].mean():,.2f} miliar")
with kpi3:
    st.metric("Rata-rata Skor IKPA", f"{filt['skor_ikpa'].mean():.2f}")
with kpi4:
    achieved = (filt['realisasi_tercapai_95persen'] == 'Ya').mean() * 100
    st.metric("Realisasi >=95%", f"{achieved:.1f}%")

tabs = st.tabs(["Ringkasan", "Peta Provinsi", "Prediksi"])

with tabs[0]:
    st.subheader("Insight Utama")
    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = px.scatter(
            filt,
            x='pagu_miliar',
            y='realisasi_tw3_persen',
            color='nama_kementerian',
            size='jumlah_spm',
            hover_data=['kode_satker', 'provinsi', 'skor_ikpa'],
            title='Pagu vs Realisasi TW3 (%)',
        )
        fig_scatter.update_layout(height=520)
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        fig_hist = px.histogram(
            filt,
            x='pagu_miliar',
            color='jenis_belanja_utama',
            nbins=35,
            title='Distribusi Pagu per Jenis Belanja',
        )
        fig_hist.update_layout(height=250)
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_box = px.box(
            filt,
            x='tipe_satker',
            y='skor_ikpa',
            color='tipe_satker',
            title='Skor IKPA menurut Tipe Satker',
        )
        fig_box.update_layout(height=250)
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Komposisi Pagu dan Realisasi")
    fig_sunburst = px.sunburst(
        filt,
        path=['nama_kementerian', 'provinsi', 'tipe_satker'],
        values='pagu_miliar',
        color='skor_ikpa',
        color_continuous_scale='Blues',
        title='Komposisi Pagu: Kementerian → Provinsi → Tipe Satker',
    )
    st.plotly_chart(fig_sunburst, use_container_width=True)

    st.subheader("Data Raw (preview)")
    st.dataframe(filt.head(sample_size))
    csv = filt.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV filtered",
        data=csv,
        file_name='buatlagi_filtered.csv',
        mime='text/csv',
    )

with tabs[1]:
    st.subheader("Peta Interaktif per Provinsi")
    map_data = prepare_map_data(filt)
    if map_data.empty:
        st.warning("Tidak cukup data peta provinsi untuk ditampilkan.")
    else:
        fig_map = px.scatter_mapbox(
            map_data,
            lat='lat',
            lon='lon',
            size='pagu_miliar',
            color='skor_ikpa',
            hover_name='provinsi',
            hover_data={
                'pagu_miliar': ':.2f',
                'skor_ikpa': ':.2f',
                'jumlah_satker': True,
                'lat': False,
                'lon': False,
            },
            color_continuous_scale='Viridis',
            size_max=35,
            zoom=4,
            title='Peta Provinsi: Total Pagu dan Skor IKPA',
            mapbox_style='open-street-map',
        )
        fig_map.update_layout(height=650, margin={'l':0, 'r':0, 't':40, 'b':0})
        st.plotly_chart(fig_map, use_container_width=True)

with tabs[2]:
    st.subheader("Prediksi Real-Time dan Interpretabilitas")
    row_container, manual_container = st.columns(2)

    with row_container:
        st.markdown("### Prediksi berdasarkan baris data")
        selected_code = st.selectbox("Pilih kode_satker", filt['kode_satker'].tolist())
        selected_row = filt[filt['kode_satker'] == selected_code].head(1)
        st.dataframe(selected_row.T)
        X_row_prepared = None  # Initialize outside button scope
        if model is not None:
            if st.button("Prediksi baris terpilih"):
                # Prepare input according to model domain (handles Orange one-hot features)
                X_row_prepared = prepare_model_input(model, selected_row)
                try:
                    prediction = model.predict(X_row_prepared)
                    # Orange wrapper may return (pred_array, proba_array) tuple
                    pred_vals = prediction[0] if isinstance(prediction, tuple) else prediction
                    try:
                        raw = pred_vals[0]
                    except Exception:
                        raw = pred_vals
                    labels_map = get_model_labels(model)
                    if labels_map:
                        try:
                            idx = int(raw)
                            display_label = labels_map[idx] if 0 <= idx < len(labels_map) else str(raw)
                        except Exception:
                            display_label = str(raw)
                    else:
                        display_label = str(raw)
                    st.success(f"Prediksi: {display_label}")
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_row_prepared)[0]
                        labels = get_model_labels(model, proba)
                        if not labels:
                            labels = [f'Class_{i}' for i in range(len(proba))]
                        proba_df = pd.DataFrame([proba], columns=[f'P_{c}' for c in labels])
                        st.dataframe(proba_df)
                        fig_p = px.pie(
                            values=proba,
                            names=labels,
                            title='Probabilitas Prediksi',
                        )
                        st.plotly_chart(fig_p, use_container_width=True)
                except Exception as exc:
                    st.error(f"Prediksi baris gagal: {exc}")
        else:
            st.warning("Model tidak tersedia untuk prediksi baris.")

        if shap_explainer is not None and X_row_prepared is not None:
            st.markdown("### SHAP Interpretasi untuk Baris Terpilih")
            try:
                shap_values = shap_explainer(X_row_prepared)
                contribution = pd.DataFrame({
                    'feature': numeric_cols,
                    'shap_value': shap_values.values[0],
                })
                contribution = contribution.assign(weight=contribution['shap_value'].abs()).sort_values('weight', ascending=False).head(10)
                fig_shap = px.bar(
                    contribution,
                    x='shap_value',
                    y='feature',
                    orientation='h',
                    color='shap_value',
                    color_continuous_scale='RdBu',
                    title='Top 10 Kontribusi Fitur SHAP',
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as exc:
                st.info(f"SHAP tidak dapat dijalankan untuk model ini: {exc}")

    with manual_container:
        st.markdown("### Prediksi menggunakan input manual")
        with st.form(key='manual_input_form'):
            manual_values = {}
            tipe_options = sorted(df['tipe_satker'].unique())
            tipe_choice = st.selectbox('Tipe Satker', tipe_options)
            for col in numeric_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                step = max((max_val - min_val) / 100.0, 0.01)
                manual_values[col] = st.number_input(
                    label=col,
                    value=float(df[col].median()),
                    min_value=min_val,
                    max_value=max_val,
                    step=step,
                )
            # include categorical selection for tipe_satker so we can build one-hot features
            manual_values['tipe_satker'] = tipe_choice
            submit = st.form_submit_button("Prediksi Manual")
            if submit:
                if model is None:
                    st.warning("Model tidak tersedia untuk prediksi manual.")
                else:
                    X_manual_df = pd.DataFrame([manual_values])
                    X_manual_prepared = prepare_model_input(model, X_manual_df)
                    try:
                        prediction = model.predict(X_manual_prepared)
                        pred_vals = prediction[0] if isinstance(prediction, tuple) else prediction
                        try:
                            raw = pred_vals[0]
                        except Exception:
                            raw = pred_vals
                        labels_map = get_model_labels(model)
                        if labels_map:
                            try:
                                idx = int(raw)
                                display_label = labels_map[idx] if 0 <= idx < len(labels_map) else str(raw)
                            except Exception:
                                display_label = str(raw)
                        else:
                            display_label = str(raw)
                        st.success(f"Prediksi: {display_label}")
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_manual_prepared)[0]
                            labels = get_model_labels(model, proba)
                            if not labels:
                                labels = [f'Class_{i}' for i in range(len(proba))]
                            proba_df = pd.DataFrame([proba], columns=[f'P_{c}' for c in labels])
                            st.dataframe(proba_df)
                            fig_p = px.pie(
                                values=proba,
                                names=labels,
                                title='Probabilitas Prediksi Manual',
                            )
                            st.plotly_chart(fig_p, use_container_width=True)
                        if shap_explainer is not None:
                            shap_values = shap_explainer(X_manual)
                            contribution = pd.DataFrame({
                                'feature': numeric_cols,
                                'shap_value': shap_values.values[0],
                            })
                            contribution = contribution.assign(weight=contribution['shap_value'].abs()).sort_values('weight', ascending=False).head(10)
                            fig_shap = px.bar(
                                contribution,
                                x='shap_value',
                                y='feature',
                                orientation='h',
                                color='shap_value',
                                color_continuous_scale='RdBu',
                                title='SHAP Kontribusi untuk Input Manual',
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)
                    except Exception as exc:
                        st.error(f"Prediksi manual gagal: {exc}")

    st.markdown("---")
    st.markdown("### Prediksi Batch Filtered")
    if model is not None and st.button("Jalankan Prediksi Batch pada Filter"):
        try:
            X_batch_prepared = prepare_model_input(model, filt)
            preds = model.predict(X_batch_prepared)
            # normalize Orange wrapper output
            if isinstance(preds, tuple):
                preds = preds[0]
            labels_map = get_model_labels(model)
            try:
                import numpy as _np
                is_array = isinstance(preds, (_np.ndarray, list))
            except Exception:
                is_array = isinstance(preds, list)
            if labels_map and is_array:
                try:
                    preds_mapped = [labels_map[int(p)] for p in preds]
                except Exception:
                    preds_mapped = preds
            else:
                preds_mapped = preds
            batch_result = filt.copy()
            batch_result['prediksi_model'] = preds_mapped
            st.dataframe(batch_result[['kode_satker', 'nama_kementerian', 'provinsi', 'pagu_miliar', 'prediksi_model']].head(100))
            st.success("Batch prediksi selesai.")
        except Exception as exc:
            st.error(f"Prediksi batch gagal: {exc}")

st.markdown("---")
st.caption("Dashboard dibuat oleh asisten; jalankan dengan `python3 -m streamlit run streamlit_app.py` untuk memastikan lingkungan yang sama dengan pip.")
