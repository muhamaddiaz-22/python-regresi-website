# 1. IMPORT LIBRARY YANG DIBUTUHKAN
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io

# 2. KONFIGURASI TAMPILAN WEBSITE
st.set_page_config(page_title="Kalkulator Regresi Linear", layout="wide")
st.title("📊 Kalkulator Regresi Linear Sederhana")
# Membuat layout dengan dua kolom
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Masukkan Data Anda")
    # 3. MEMBUAT FORM INPUT DATA
    with st.form(key='data_form'):
        # Input untuk data X
        x_input = st.text_area(
            "Masukkan data X (pisahkan dengan koma)",
            "0, 0.091, 0.041, 0.191, 0.241, 0.291"
        )
        # Input untuk data Y
        y_input = st.text_area(
            "Masukkan data Y (pisahkan dengan koma)",
            "0.0005, 0.0005, 0.0015, 0.003, 0.0045, 0.0055"
        )
        
        # Tombol untuk memproses data
        submit_button = st.form_submit_button(label='🚀 Proses & Buat Grafik')

# 4. PROSES DAN TAMPILKAN HASIL JIKA TOMBOL DITEKAN
if submit_button:
    try:
        # Mengubah string input menjadi list angka (float)
        data_x_str = x_input.split(',')
        data_y_str = y_input.split(',')
        
        data_x = np.array([float(x.strip()) for x in data_x_str])
        data_y = np.array([float(y.strip()) for y in data_y_str])

        # Validasi: jumlah data X dan Y harus sama
        if len(data_x) != len(data_y):
            st.error("Jumlah data X dan Y harus sama!")
        # Validasi: butuh minimal 2 data untuk regresi
        elif len(data_x) < 2:
            st.error("Dibutuhkan minimal 2 pasang data untuk melakukan regresi.")
        else:
            # --- MULAI PERHITUNGAN REGRESI (SAMA SEPERTI KODE ASLI) ---
            X = data_x.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, data_y)

            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(X, data_y)
            # --- SELESAI PERHITUNGAN REGRESI ---

            with col2:
                st.header("Hasil Analisis")
                
                # 5. MEMBUAT PLOT (SEPERTI KODE ASLI TAPI UNTUK STREAMLIT)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X, data_y, color='blue', label='Data Praktikum Anda')
                y_prediksi = model.predict(X)
                ax.plot(X, y_prediksi, color='red', linewidth=2, label='Garis Regresi')

                ax.set_title('Grafik Hasil Regresi Linear', fontsize=16)
                ax.set_xlabel('Variabel Independen (X)', fontsize=12)
                ax.set_ylabel('Variabel Dependen (Y)', fontsize=12)
                ax.legend(loc='upper left')
                ax.grid(True)
                
                # Menampilkan plot di website
                st.pyplot(fig)
                
                # 6. MENAMPILKAN HASIL PERHITUNGAN
                st.subheader("Hasil Perhitungan")
                st.markdown(f"**Persamaan Garis:** `$Y = {slope:.4f}X + {intercept:.4f}$`")
                st.markdown(f"**Slope (Gradien), m:** `{slope:.4f}`")
                st.markdown(f"**Intercept (Titik Potong Y), c:** `{intercept:.4f}`")
                st.markdown(f"**Koefisien Determinasi ($R^2$):** `{r_squared:.4f}`")

                # Opsi untuk download gambar
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300)
                st.download_button(
                    label="📥 Unduh Grafik (.png)",
                    data=buf.getvalue(),
                    file_name="grafik_regresi.png",
                    mime="image/png"
                )

    except ValueError:
        st.error("Format data salah. Pastikan semua data adalah angka dan dipisahkan dengan koma.")
    except Exception as e:
        st.error(f"Terjadi error: {e}")