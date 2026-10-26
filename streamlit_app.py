import streamlit as st
import requests

# 1. Page setup
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
st.title("Diabetes Risk Predictor")

# Define the API URL once:
# Local chalane ke liye:
API_URL = "http://127.0.0.1:8000"
# Live Streamlit Cloud ke liye (jab ngrok chalayein toh neeche wali line uncomment kar ke apna link dalein):
# API_URL = "https://your-ngrok-url.ngrok-free.app"

# 2. Eight input widgets laid out in two columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", min_value=0, max_value=17, value=2, step=1)
    glucose = st.slider("Glucose", min_value=0, max_value=200, value=120, step=1)
    blood_pressure = st.slider("BloodPressure", min_value=0, max_value=130, value=70, step=1)
    skin_thickness = st.slider("SkinThickness", min_value=0, max_value=100, value=20, step=1)

with col2:
    insulin = st.slider("Insulin", min_value=0, max_value=850, value=79, step=1)
    bmi = st.slider("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    age = st.slider("Age", min_value=18, max_value=90, value=30, step=1)

# 3. Predict button
if st.button("Predict", type="primary", use_container_width=True):
    payload = {
        "Pregnancies": int(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(dpf),
        "Age": int(age)
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            label = result["result_label"]
            conf = result["confidence"]

            # 4. Clear result display
            if label == "Diabetic":
                st.error(f"Result: {label}")
            else:
                st.success(f"Result: {label}")

            st.metric(label="Confidence", value=f"{conf * 100:.1f}%")
        else:
            st.warning("Could not get a valid response from the server. Please make sure FastAPI is running first.")

    # 5. Graceful failure
    except requests.exceptions.RequestException:
        st.warning("Could not connect to the API. Please start the FastAPI server first (`uvicorn main:app --reload`).")
        