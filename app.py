import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import time

# =========================================================================
# 1. KONFIGURASI HALAMAN & CSS CUSTOM (SIDEBAR PREMIUM & METRIC)
# =========================================================================
st.set_page_config(
    page_title="Sistem Monitoring & EWS 2023", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Styling Dasar Layout Utama */
    .main { background-color: #f8f9fa; }
    
    /* Styling Metrik Box */
    .metric-box { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #64748b; margin-bottom: 15px; }
    .metric-box-ontime { background-color: #f0fdf4; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #16a34a; margin-bottom: 15px; }
    .metric-box-beresiko { background-color: #fef2f2; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #dc2626; margin-bottom: 15px; }
    .metric-title { font-size: 14px; font-weight: bold; color: #475569; margin-bottom: 5px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #1e293b; }
    .metric-delta { font-size: 14px; font-weight: 500; color: #16a34a; }
    .metric-delta-red { font-size: 14px; font-weight: 500; color: #dc2626; }
    
  
    /* Menyembunyikan lingkaran/bulatan radio default Streamlit */
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    
    /* Menghilangkan label teks default bawaan st.radio agar bersih */
    div[data-testid="stRadio"] > label {
        display: none !important;
    }
    
    /* Mengatur jarak antar item menu navigasi */
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 10px !important;
    }
    
    /* Membuat komponen kotak dasar item menu */
    div[data-testid="stRadio"] label:has(input[type="radio"]) {
        background-color: #ffffff;
        color: #1e293b !important;
        border: 1px solid #e2e8f0;
        padding: 14px 18px !important;
        border-radius: 10px !important;
        cursor: pointer;
        transition: all 0.25s ease-in-out !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        margin-bottom: 2px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Efek Interaktif Hover (Saat kursor mouse melintas di atas menu) */
    div[data-testid="stRadio"] label:has(input[type="radio"]):hover {
        background-color: #f1f5f9;
        border-color: #cbd5e1;
        transform: translateX(6px); /* Efek bergeser smooth ke kanan */
    }
    
    /* Efek Saat Menu Aktif / Terpilih (Selected State) */
    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) {
        background-color: #0f172a !important; /* Warna Dark Slate Premium */
        border-color: #0f172a !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.18) !important;
    }
    
    /* Memaksa warna teks di dalam menu yang aktif menjadi putih tebal */
    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)


# =========================================================================
# 2. AUTO-PIPELINE BACKEND (MACHINE LEARNING)
# =========================================================================
@st.cache_resource 
def run_automatic_ml_pipeline():
    try:
        df = pd.read_csv('dataset_akt23.csv', on_bad_lines='skip') 
    except FileNotFoundError:
        st.error("Error: File 'dataset_akt23.csv' tidak ditemukan.")
        st.stop()

    df = df.drop_duplicates()
    df['IPK'] = df['IPK'].fillna(df['IPK'].median())
    df['IPS_Sem6'] = df['IPS_Sem6'].fillna(df['IPS_Sem6'].median())
    df['Jml_MK_Gagal'] = df['Jml_MK_Gagal'].fillna(0)
    df['SKS_Lulus'] = df['SKS_Lulus'].fillna(df['SKS_Lulus'].median())
    df['Jumlah_Bimbingan'] = df['Jumlah_Bimbingan'].fillna(0)
    df['Status_Skripsi'] = df['Status_Skripsi'].astype(str).str.replace('\n', '', regex=True).str.strip()
    
    df['Rata_IP'] = (df['IPK'] + df['IPS_Sem6']) / 2
    df['Progress_SKS'] = df['SKS_Lulus'] / 144
    df['Tren_IP'] = df['IPS_Sem6'] - df['IPK']
    
    tahapan_aktif = ['Judul ACC', 'Seminar Proposal', 'Seminar Hasil', 'Sidang Komprehensif']
    
    def batasan_aturan(row):
        if (row['IPK'] >= 3.00) and (row['Jml_MK_Gagal'] <= 2) and (row['Jumlah_Bimbingan'] > 5) and (row['Status_Skripsi'] in tahapan_aktif):
            return 'Ontime'
        else:
            return 'Beresiko'
            
    df['Target'] = df.apply(batasan_aturan, axis=1)
    
    le_status = LabelEncoder()
    df['Status_Skripsi_Encoded'] = le_status.fit_transform(df['Status_Skripsi'])
    
    le_target = LabelEncoder()
    df['Target_Encoded'] = le_target.fit_transform(df['Target'])
    
    fitur = ['IPK', 'IPS_Sem6', 'Jml_MK_Gagal', 'SKS_Lulus', 'Jumlah_Bimbingan', 
             'Status_Skripsi_Encoded', 'Rata_IP', 'Progress_SKS', 'Tren_IP']
    
    X = df[fitur]
    y = df['Target_Encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, le_status, le_target, df, fitur

model, le_status, le_target, df_clean, fitur_model = run_automatic_ml_pipeline()

# Fungsi Aturan Rekomendasi
def dapatkan_rekomendasi_tindakan(row):
    rekomendasi = []
    if row['Jml_MK_Gagal'] > 2:
        rekomendasi.append("Her-registrasi kelas perbaikan untuk MK Gagal")
    if row['Jumlah_Bimbingan'] <= 5:
        rekomendasi.append("Pemanggilan oleh Kaprodi/Dosen Pembimbing untuk bimbingan intensif")
    if row['Status_Skripsi'] == 'Belum Ambil':
        rekomendasi.append("Peringatan batas akhir pendaftaran Judul Tugas Akhir")
    if row['IPK'] < 3.00:
        rekomendasi.append("Monitoring nilai semester akhir (Target minimal IPK aman)")
    if len(rekomendasi) == 0:
        rekomendasi.append("Monitoring progress bab skripsi dan jadwalkan seminar hasil")
    return " • ".join(rekomendasi)

# Fungsi Aturan Level Risiko (DISINKRONKAN AGAR COCOK DENGAN STRUKTUR URUTAN)
def tentukan_tingkat_risiko(row):
    if row['Jml_MK_Gagal'] >= 3 or row['IPK'] < 2.5 or row['Status_Skripsi'] == 'Belum Ambil':
        return 'Kritis'
    elif row['Status_Kelulusan'] == 'Ontime':
        return 'Aman'
    else:
        return 'Waspada'


# =========================================================================
# 3. NAVIGASI MENU UTAMA (SIDEBAR STYLING)
# =========================================================================
with st.sidebar:
    # Header khusus untuk mempercantik Sidebar bagian atas
    st.markdown("<h3 style='text-align: center; color: #1e293b; margin-bottom: 20px;'> NAVIGASI SISTEM</h3>", unsafe_allow_html=True)
    
    # Penggunaan st.sidebar.radio yang sudah dimodifikasi otomatis oleh CSS di atas
    pilihan_menu = st.sidebar.radio(
        "Navigasi Utama",
        ["Dashboard Monitoring", "Early Warning System", "Fitur Prediksi"]
    )
    st.markdown("---")
    st.caption("Model AI: Random Forest Classifier\n\n Data Angkatan 2023 Active")

# Memproses Prediksi Massal AI untuk sinkronisasi seluruh menu
df_proses = df_clean.copy()
df_proses['Prediksi_ID'] = model.predict(df_proses[fitur_model])
df_proses['Status_Kelulusan'] = le_target.inverse_transform(df_proses['Prediksi_ID'])
df_proses['Tingkat Risiko'] = df_proses.apply(tentukan_tingkat_risiko, axis=1)


# =========================================================================
# MENU 1: DASHBOARD MONITORING
# =========================================================================
if pilihan_menu == "Dashboard Monitoring":
    st.title("Dashboard Monitoring Mahasiswa")
    st.markdown("Ringkasan statistik data real-time mahasiswa angkatan 2023.")
    st.markdown("---")
    
    m_total = len(df_proses)
    m_beresiko = len(df_proses[df_proses['Status_Kelulusan'] == 'Beresiko'])
    m_ontime = m_total - m_beresiko
    persen_ontime = (m_ontime / m_total) * 100
    persen_beresiko = (m_beresiko / m_total) * 100
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-title">TOTAL MAHASISWA AKTIF</div><div class="metric-value">{m_total} Orang</div><div class="metric-delta" style="color:#64748b;"> Mahasiswa Angkatan 2023</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box-ontime"><div class="metric-title">PREDIKSI TEPAT WAKTU (ONTIME)</div><div class="metric-value">{m_ontime} Orang</div><div class="metric-delta">▲ {persen_ontime:.1f}% Dari Total</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box-beresiko"><div class="metric-title">PREDIKSI BERESIKO (TERHAMBAT)</div><div class="metric-value">{m_beresiko} Orang</div><div class="metric-delta-red">▼ {persen_beresiko:.1f}% Perlu Perhatian</div></div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("Analisis Distribusi Status Kelulusan")
    
    col_chart1, col_chart2 = st.columns([1, 1])
    with col_chart1:
        df_pie = pd.DataFrame({'Status': ['Tepat Waktu (Ontime)', 'Beresiko (Terhambat)'], 'Jumlah': [m_ontime, m_beresiko]})
        fig_pie = px.pie(df_pie, values='Jumlah', names='Status', title='Proporsi Status Kelulusan', color='Status', color_discrete_map={'Tepat Waktu (Ontime)': '#16a34a', 'Beresiko (Terhambat)': '#dc2626'}, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_chart2:
        df_bar = df_proses.groupby(['Status_Skripsi', 'Status_Kelulusan']).size().reset_index(name='Jumlah')
        fig_bar = px.bar(df_bar, x='Status_Skripsi', y='Jumlah', color='Status_Kelulusan', title='Distribusi Berdasarkan Tahap Skripsi', barmode='group', color_discrete_map={'Ontime': '#16a34a', 'Beresiko': '#dc2626'})
        st.plotly_chart(fig_bar, use_container_width=True)


# =========================================================================
# MENU 2: EARLY WARNING SYSTEM (EWS)
# =========================================================================
elif pilihan_menu == "Early Warning System":
    st.title("Early Warning System (EWS)")
    st.markdown("Monitor tingkat risiko mahasiswa dan kirim **Notifikasi Peringatan Otomatis** ke kontak mahasiswa.")
    st.markdown("---")
    
    df_ews = df_proses.copy()
    df_ews['Rekomendasi Tindakan'] = df_ews.apply(dapatkan_rekomendasi_tindakan, axis=1)
    df_ews['Pilih'] = False 
    
    urutan = {'Kritis': 1, 'Waspada': 2, 'Aman': 3}
    df_ews['Sort_Idx'] = df_ews['Tingkat Risiko'].map(urutan)
    df_ews = df_ews.sort_values('Sort_Idx').drop('Sort_Idx', axis=1)

    kolom_tampil = ['Pilih', 'NIM', 'Nama', 'Tingkat Risiko', 'IPK', 'Jml_MK_Gagal', 'Jumlah_Bimbingan', 'Status_Skripsi', 'Rekomendasi Tindakan']
    df_ews = df_ews[kolom_tampil]

    def beri_warna_risiko(val):
        if 'Kritis' in str(val): return 'background-color: #fca5a5; color: #7f1d1d; font-weight: bold;'
        elif 'Waspada' in str(val): return 'background-color: #fef08a; color: #713f12; font-weight: bold;'
        elif 'Aman' in str(val): return 'background-color: #bbf7d0; color: #14532d; font-weight: bold;'
        return ''

    try:
        df_styled = df_ews.style.map(beri_warna_risiko, subset=['Tingkat Risiko'])
    except AttributeError:
        df_styled = df_ews.style.applymap(beri_warna_risiko, subset=['Tingkat Risiko'])


    edited_df = st.data_editor(
        df_styled,
        hide_index=True,
        column_config={"Pilih": st.column_config.CheckboxColumn("Pilih Notif", help="Centang untuk kirim notifikasi", default=False)},
        disabled=["NIM", "Nama", "Tingkat Risiko", "IPK", "Jml_MK_Gagal", "Jumlah_Bimbingan", "Status_Skripsi", "Rekomendasi Tindakan"],
        use_container_width=True,
        height=400
    )
    
    if st.button("📤 Kirim Notifikasi Terpilih", type="primary"):
        mhs_terpilih = edited_df[edited_df['Pilih'] == True]
        if len(mhs_terpilih) > 0:
            with st.spinner("Menghubungkan ke server notifikasi falkultas..."):
                time.sleep(1.2)
            st.success(f"**BERHASIL!** Notifikasi dikirim ke **{len(mhs_terpilih)}** mahasiswa: {', '.join(mhs_terpilih['Nama'].tolist())}")
            st.balloons()
        else:
            st.error("Silakan centang kotak di kolom 'Pilih Notif' terlebih dahulu.")


# =========================================================================
# MENU 3: FITUR PREDIKSI
# =========================================================================
elif pilihan_menu == "Fitur Prediksi":
    st.title("Fitur Analisis & Prediksi Status Kelulusan")
    st.markdown("Gunakan salah satu tab di bawah ini untuk melihat hasil prediksi kecerdasan buatan.")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Prediksi Berdasarkan Dataset", "Prediksi Mandiri (Input Manual)"])
    
    with tab1:
        st.subheader("Cari & Analisis Mahasiswa Langsung dari Database")
        st.markdown("Pilih mahasiswa dari daftar di bawah ini untuk membaca data aslinya dan memprediksi status kelulusannya secara otomatis.")
        
        df_proses['Pencarian_Mhs'] = df_proses['NIM'].astype(str) + " - " + df_proses['Nama']
        
        pilihan_mhs = st.selectbox(
            "Silakan Ketik NIM atau Nama Mahasiswa:",
            options=df_proses['Pencarian_Mhs'].tolist(),
            index=0
        )
        
        data_mhs = df_proses[df_proses['Pencarian_Mhs'] == pilihan_mhs].iloc[0]
        
        st.markdown("Hasil Analisis AI")
        if data_mhs['Status_Kelulusan'] == "Ontime":
            st.success(f"Mahasiswa **{data_mhs['Nama']} ({data_mhs['NIM']})** saat ini diprediksi berada di jalur **TEPAT WAKTU (Ontime)**.")
        else:
            st.error(f"Mahasiswa **{data_mhs['Nama']} ({data_mhs['NIM']})** saat ini diprediksi **BERESIKO (Terhambat Lulus)**.")
            
        st.markdown("Detail Rekomendasi:")
        rekomendasi_mhs = dapatkan_rekomendasi_tindakan(data_mhs)
        for poin in rekomendasi_mhs.split(" • "):
            st.write(f"- {poin}")
            
        st.markdown("---")
        st.markdown("Ringkasan Data Akademik Tercatat:")
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Indeks Prestasi Kumulatif (IPK)", f"{data_mhs['IPK']:.2f}")
        c_m2.metric("Jumlah MK Gagal", f"{int(data_mhs['Jml_MK_Gagal'])} Mata Kuliah")
        c_m3.metric("Bimbingan Skripsi", f"{int(data_mhs['Jumlah_Bimbingan'])} Kali")
        c_m4.metric("Tahapan Skripsi", str(data_mhs['Status_Skripsi']))
        
    with tab2:
        st.subheader("Simulasi Prediksi Mandiri (Input Nilai Manual)")
        st.markdown("Masukkan data akademik secara bebas di bawah ini untuk melihat simulasi prediksi model.")
        
        col_l, col_r = st.columns(2)
        with col_l:
            m_nama = st.text_input("Nama Mahasiswa Simulan:", value="Aditya Lestari")
            m_ipk = st.number_input("IPK Terakhir:", min_value=0.0, max_value=4.0, value=3.37, step=0.01, key="manual_ipk")
            m_ips = st.number_input("IPS Semester Terakhir:", min_value=0.0, max_value=4.0, value=3.30, step=0.01, key="manual_ips")
            m_gagal = st.number_input("Jumlah MK Gagal:", min_value=0, max_value=50, value=2, step=1, key="manual_gagal")
            
        with col_r:
            m_sks = st.number_input("SKS Telah Lulus:", min_value=0, max_value=160, value=112, step=1, key="manual_sks")
            opsi_status = list(le_status.classes_) 
            m_status = st.selectbox("Tahap Skripsi Saat Ini:", opsi_status, key="manual_status")
            m_bimbingan = st.number_input("Jumlah Konsultasi/Bimbingan:", min_value=0, max_value=100, value=4, step=1, key="manual_bimbingan")
            
        if st.button("Hitung Prediksi Mandiri", type="primary"):
            calc_rata_ip = (m_ipk + m_ips) / 2
            calc_progress_sks = m_sks / 144
            calc_tren_ip = m_ips - m_ipk
            calc_status_encoded = le_status.transform([m_status])[0]
            
            df_input_baru = pd.DataFrame([[
                m_ipk, m_ips, m_gagal, m_sks, m_bimbingan, 
                calc_status_encoded, calc_rata_ip, calc_progress_sks, calc_tren_ip
            ]], columns=fitur_model)
            
            id_hasil = model.predict(df_input_baru)[0]
            teks_hasil = le_target.inverse_transform([id_hasil])[0]
            
            st.markdown("Hasil Analisis")
            if teks_hasil == "Ontime":
                st.success(f"Status **{m_nama}** saat ini diprediksi **TEPAT WAKTU (Ontime)**.")
            else:
                st.error(f"Status **{m_nama}** saat ini diprediksi **BERESIKO (Terlambat)**.")
                
                st.markdown("Saran & Rekomendasi Tindakan:")
                baris_analisis = {'IPK': m_ipk, 'Jml_MK_Gagal': m_gagal, 'Jumlah_Bimbingan': m_bimbingan, 'Status_Skripsi': m_status}
                rekomendasi_teks = dapatkan_rekomendasi_tindakan(baris_analisis)
                
                for poin in rekomendasi_teks.split(" • "):
                    st.write(f"- {poin}")