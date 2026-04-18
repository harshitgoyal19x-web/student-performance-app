import streamlit as st
import numpy as np
from xgboost import XGBRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Student Predictor", page_icon="🎓")

# ---------------- SESSION STATE INIT ----------------
if "study_hours" not in st.session_state:
    st.session_state.study_hours = 0
    st.session_state.attendance = 0
    st.session_state.sleep_hours = 0
    st.session_state.prev_marks = 0
    st.session_state.consistency = 0
    st.session_state.stress = 0

# ---------------- RESET BUTTON ----------------
if st.button("🔄 Reset App"):
    st.session_state.study_hours = 0
    st.session_state.attendance = 0
    st.session_state.sleep_hours = 0
    st.session_state.prev_marks = 0
    st.session_state.consistency = 0
    st.session_state.stress = 0
    st.rerun()

# ---------------- UI ----------------
st.title("🎓 Student Performance Predictor")
st.caption("💡 Smooth + Fast + Reset Working")

# ---------------- SLIDERS ----------------
study_hours = st.slider("📚 Study Hours", 0, 16, key="study_hours")
attendance = st.slider("🏫 Attendance (%)", 0, 100, key="attendance")
sleep_hours = st.slider("😴 Sleep Hours", 0, 12, key="sleep_hours")
prev_marks = st.slider("📊 Previous Marks", 0, 100, key="prev_marks")
consistency = st.slider("📅 Study Consistency (Days/week)", 0, 7, key="consistency")
stress = st.slider("😵 Stress Level (0-10)", 0, 10, key="stress")

# ---------------- MODEL DATA ----------------
X = np.array([
    [2,60,6,50,3,6],
    [5,80,7,65,5,4],
    [8,90,8,85,6,3],
    [6,75,7,70,4,5],
    [9,95,8,90,6,2],
    [3,65,6,55,3,6],
    [7,85,7,75,5,4],
    [10,90,8,88,6,3],
    [4,70,6,60,3,5]
])

y = np.array([40,70,85,75,90,55,78,88,60])

# ---------------- FAST MODEL (CACHE) ----------------
@st.cache_resource
def load_model():
    model = XGBRegressor(n_estimators=50)  # reduced for speed
    model.fit(X, y)
    return model

model = load_model()

# ---------------- PREDICT ----------------
if st.button("🎯 Predict Marks"):
    prediction = model.predict([[
        study_hours,
        attendance,
        sleep_hours,
        prev_marks,
        consistency,
        stress
    ]])

    result = prediction[0]

    st.success(f"📊 Predicted Marks: {result:.2f}")
    st.progress(int(result))

    if result < 50:
        st.error("⚠️ Poor Performance")
    elif result < 75:
        st.warning("🙂 Average Performance")
    else:
        st.success("🔥 Excellent Performance")
