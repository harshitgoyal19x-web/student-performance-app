import streamlit as st
import numpy as np
from xgboost import XGBRegressor

st.set_page_config(page_title="Student Predictor", page_icon="🎓")

# DEFAULT VALUES
defaults = {
    "study": 5,
    "att": 70,
    "sleep": 7,
    "marks": 60,
    "cons": 4,
    "stress": 5
}

# RESET BUTTON
if st.button("🔄 Reset App"):
    for key in defaults:
        st.session_state[key] = defaults[key]
    st.session_state["result"] = None
    st.rerun()

# TITLE
st.title("🎓 Student Performance Predictor")
st.caption("Tip: Use Reset button")

# SLIDERS (WITH KEYS)
study = st.slider("Study Hours", 0, 16, defaults["study"], key="study")
att = st.slider("Attendance (%)", 0, 100, defaults["att"], key="att")
sleep = st.slider("Sleep Hours", 0, 12, defaults["sleep"], key="sleep")
marks = st.slider("Previous Marks", 0, 100, defaults["marks"], key="marks")
cons = st.slider("Consistency", 0, 7, defaults["cons"], key="cons")
stress = st.slider("Stress Level", 0, 10, defaults["stress"], key="stress")

# MODEL DATA
X = np.array([
    [2,60,6,50,3,6],
    [5,80,7,65,5,4],
    [8,90,8,85,6,3],
    [6,75,7,70,4,5],
    [9,95,8,90,6,2],
    [3,65,5,55,3,7],
    [7,85,7,75,5,4],
    [10,98,8,95,6,1],
    [4,70,6,60,4,6]
])

y = np.array([40,70,85,65,95,50,80,98,60])

model = XGBRegressor(n_estimators=120)
model.fit(X, y)

# PREDICT
if st.button("Predict Marks"):
    pred = model.predict([[study, att, sleep, marks, cons, stress]])[0]
    st.session_state["result"] = pred

# SHOW RESULT
if "result" in st.session_state and st.session_state["result"] is not None:
    p = st.session_state["result"]

    st.success(f"Predicted Marks: {p:.2f}")
    st.progress(min(int(p), 100))

    if p < 50:
        st.error("Poor Performance")
    elif p < 75:
        st.warning("Average Performance")
    else:
        st.success("Excellent 🚀")
