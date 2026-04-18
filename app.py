import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="Student Predictor", page_icon="🎓")

# RESET BUTTON (WORKING)
if st.button("🔄 Reset App"):
    st.session_state.clear()
    st.rerun()

# Title
st.title("🎓 Student Performance Predictor")
st.caption("💡 Tip: Click Reset App for full reset")

# INPUTS (with unique keys for reset)
study_hours = st.slider("📊 Study Hours", 0, 16, 5, key="study")
attendance = st.slider("🏫 Attendance (%)", 0, 100, 70, key="att")
sleep_hours = st.slider("😴 Sleep Hours", 0, 12, 7, key="sleep")
prev_marks = st.slider("📈 Previous Marks", 0, 100, 60, key="marks")
consistency = st.slider("📅 Study Consistency (Days/week)", 0, 7, 4, key="cons")
stress = st.slider("😵 Stress Level (0-10)", 0, 10, 5, key="stress")

# DATA (simple training data)
X = np.array([
    [2, 60, 6, 50, 3, 6],
    [5, 75, 7, 65, 4, 5],
    [8, 85, 6, 75, 5, 4],
    [6, 70, 7, 60, 4, 5],
    [9, 90, 6, 85, 6, 3],
    [4, 65, 8, 55, 3, 6],
    [7, 80, 7, 70, 5, 4],
    [10, 95, 6, 90, 6, 2]
])

y = np.array([50, 65, 80, 70, 90, 55, 75, 95])

# MODEL (stable)
model = LinearRegression()
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
    ]])[0]

    st.success(f"📊 Predicted Marks: {prediction:.2f}")
    st.progress(int(min(max(prediction, 0), 100)))

    # Performance label
    if prediction < 50:
        st.error("🔴 Poor Performance")
    elif prediction < 75:
        st.warning("🟡 Average Performance")
    else:
        st.success("🟢 Excellent Performance")
