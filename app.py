import streamlit as st
import numpy as np
from xgboost import XGBRegressor

# Page config
st.set_page_config(page_title="Student Predictor", page_icon="🎓")

# RESET BUTTON
if st.button("🔄 Reset App"):
    st.session_state.clear()
    st.rerun()

# Title
st.title("🎓 Student Performance Predictor")
st.caption("💡 Tip: Click reset button for full reset")

# SLIDERS (DEFAULT = 0 for proper reset)
study_hours = st.slider("📚 Study Hours", 0, 16, 0)
attendance = st.slider("🏫 Attendance (%)", 0, 100, 0)
sleep_hours = st.slider("😴 Sleep Hours", 0, 12, 0)
prev_marks = st.slider("📊 Previous Marks", 0, 100, 0)
consistency = st.slider("📅 Study Consistency (Days/week)", 0, 7, 0)
stress = st.slider("😵 Stress Level (0-10)", 0, 10, 0)

# MODEL DATA
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

# TRAIN MODEL
model = XGBRegressor(n_estimators=120)
model.fit(X, y)

# PREDICT BUTTON
if st.button("🎯 Predict Marks"):
    prediction = model.predict([[
        study_hours,
        attendance,
        sleep_hours,
        prev_marks,
        consistency,
        stress
    ]])

    st.success(f"📊 Predicted Marks: {prediction[0]:.2f}")

    # Progress bar
    st.progress(int(prediction[0]))

    # Performance message
    if prediction[0] < 50:
        st.error("⚠️ Poor Performance")
    elif prediction[0] < 75:
        st.warning("🙂 Average Performance")
    else:
        st.success("🔥 Excellent Performance")
