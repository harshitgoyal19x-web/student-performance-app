import streamlit as st
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="Student Predictor", page_icon="🎓")

# 🔄 RESET BUTTON (TOP)
if st.button("🔄 Reset App"):
    st.session_state.clear()
    st.rerun()

st.title("🎓 Student Performance Predictor")

st.caption("💡 Tip: Refresh page (Ctrl + R) for full reset")

# INPUTS
study_hours = st.slider("📚 Study Hours", 0, 16, 5)
attendance = st.slider("🏫 Attendance (%)", 0, 100, 70)
sleep_hours = st.slider("😴 Sleep Hours", 0, 12, 7)
prev_marks = st.slider("📊 Previous Marks", 0, 100, 60)
consistency = st.slider("📅 Study Consistency (Days/week)", 0, 7, 4)
stress = st.slider("😵 Stress Level (0-10)", 0, 10, 5)

# MODEL DATA
X = np.array([
    [2,60,6,50,3,6],
    [5,80,7,65,5,4],
    [8,90,8,85,6,3],
    [6,75,7,70,4,5],
    [9,95,8,90,6,2],
    [3,65,6,55,3,6],
    [7,85,7,75,5,4],
    [10,90,7,88,6,3],
    [12,95,8,92,7,2]
])

y = np.array([40,70,90,65,95,50,80,88,96])

model = XGBRegressor(n_estimators=120)
model.fit(X, y)

# PREDICT
if st.button("Predict Marks"):
    prediction = model.predict([[
        study_hours,
        attendance,
        sleep_hours,
        prev_marks,
        consistency,
        stress
    ]])[0]

    st.subheader("📊 Result Analysis")
    st.success(f"📊 Predicted Marks: {prediction:.2f}")

    st.progress(int(prediction))

    if prediction < 50:
        st.error("🔴 Poor Performance")
    elif prediction < 75:
        st.warning("🟡 Average Performance")
    else:
        st.success("🟢 Excellent Performance 🚀")

    # Suggestions
    if study_hours < 4:
        st.info("📌 Increase study hours")
    if sleep_hours < 5:
        st.info("📌 Improve sleep")
    if stress > 7:
        st.info("📌 Reduce stress")
