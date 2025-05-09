import streamlit as st
import joblib
import gzip
import numpy as np

# Load model dan scaler
@st.cache_resource
def load_resources():
    with gzip.open("model.sav.gz", "rb") as f:
        model = joblib.load(f)
    scaler = joblib.load("scaler.sav")  # scaler tetap tidak dikompresi
    return scaler, model

scaler, model = load_resources()

class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

def map_to_major(career):
    career = career.lower()
    if "engineer" in career or "scientist" in career or "developer" in career:
        return "Teknik"
    elif "doctor" in career or "nurse" in career:
        return "Kesehatan"
    elif "lawyer" in career or "government" in career or "politician" in career:
        return "Sosial"
    elif "artist" in career or "writer" in career or "designer" in career:
        return "Humaniora"
    elif "teacher" in career or "professor" in career:
        return "Pendidikan"
    elif "business" in career or "entrepreneur" in career or "accountant" in career or "banker" in career or "investor" in career:
        return "Ekonomi"
    else:
        return "Lainnya"

def recommend(gender, part_time_job, absence_days, extracurricular_activities,
              weekly_self_study_hours, math_score, history_score, physics_score,
              chemistry_score, biology_score, english_score, geography_score,
              total_score, average_score):

    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    features = np.array([[gender_encoded, part_time_job_encoded, absence_days,
                          extracurricular_activities_encoded, weekly_self_study_hours,
                          math_score, history_score, physics_score, chemistry_score,
                          biology_score, english_score, geography_score,
                          total_score, average_score]])

    scaled = scaler.transform(features)
    probabilities = model.predict_proba(scaled)
    top_indices = np.argsort(-probabilities[0])[:5]
    top_results = [(class_names[i], probabilities[0][i]) for i in top_indices]
    recommended_major = map_to_major(top_results[0][0])
    return top_results, recommended_major

# ======================== Streamlit UI ========================

st.title("🎓 Sistem Rekomendasi Studi dan Karier")

st.header("📋 Masukkan Data Akademik Anda")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    part_time_job = st.checkbox("Punya Pekerjaan Paruh Waktu")
    extracurricular_activities = st.checkbox("Aktif Ekstrakurikuler")
    absence_days = st.number_input("Jumlah Hari Absen", min_value=0, max_value=365, value=5)
    weekly_self_study_hours = st.number_input("Jam Belajar Mandiri per Minggu", min_value=0, value=5)

with col2:
    math_score = st.slider("Nilai Matematika", 0, 100, 80)
    history_score = st.slider("Nilai Sejarah", 0, 100, 75)
    physics_score = st.slider("Nilai Fisika", 0, 100, 70)
    chemistry_score = st.slider("Nilai Kimia", 0, 100, 72)
    biology_score = st.slider("Nilai Biologi", 0, 100, 74)
    english_score = st.slider("Nilai Bahasa Inggris", 0, 100, 78)
    geography_score = st.slider("Nilai Geografi", 0, 100, 73)

total_score = math_score + history_score + physics_score + chemistry_score + biology_score + english_score + geography_score
average_score = total_score / 7

if st.button("🔍 Lihat Rekomendasi"):
    top_careers, major = recommend(gender, part_time_job, absence_days,
                                   extracurricular_activities, weekly_self_study_hours,
                                   math_score, history_score, physics_score,
                                   chemistry_score, biology_score, english_score,
                                   geography_score, total_score, average_score)

    st.subheader("📌 Top 5 Prediksi Karier:")
    for career, prob in top_careers:
        st.write(f"- **{career}**: {prob*100:.2f}%")

    st.markdown(f"### 🎓 Rekomendasi Jurusan: **{major}**")
