import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ── 1. Page Setup & Custom Enterprise Styling ────────────────────────────────
st.set_page_config(
    page_title="GlucoGuard AI | Clinical Risk Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the API URL once (as required)
API_URL = "https://drizzly-pager-syndrome.ngrok-free.dev"

# Custom CSS for a sleek, modern Healthcare SaaS look
st.markdown("""
<style>
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        margin: 0.4rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.98rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
        margin-bottom: 0.6rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(248, 250, 252, 0.05);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 14px 18px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State for Patient Presets & Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

if "preset" not in st.session_state:
    st.session_state.preset = {
        "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70,
        "SkinThickness": 20, "Insulin": 79, "BMI": 25.0,
        "DiabetesPedigreeFunction": 0.50, "Age": 30
    }


def load_preset(preg, gluc, bp, skin, ins, bmi_val, dpf_val, age_val):
    st.session_state.preset = {
        "Pregnancies": preg, "Glucose": gluc, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": ins, "BMI": bmi_val,
        "DiabetesPedigreeFunction": dpf_val, "Age": age_val
    }


# ── Sidebar: System Telemetry & 1-Click Patient Presets ──────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Backend Telemetry")
    st.caption(f"**Endpoint:** `{API_URL}`")

    if st.button("🔄 Check FastAPI Health", use_container_width=True):
        try:
            ping = requests.get(f"{API_URL}/", timeout=5)
            if ping.status_code == 200:
                info = ping.json()
                st.success(f"Online | Model Loaded: {info.get('model_loaded', True)}")
            else:
                st.warning(f"Server responded with HTTP {ping.status_code}")
        except requests.exceptions.RequestException:
            st.error("Offline — Start FastAPI (`uvicorn main:app --reload`)")

    st.divider()
    st.markdown("### 🧪 Quick Patient Presets")
    st.caption("Load realistic clinical profiles for instant testing:")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🟢 Healthy", use_container_width=True):
            load_preset(1, 95, 68, 22, 80, 22.4, 0.25, 26)
            st.rerun()
    with c2:
        if st.button("🔴 High Risk", use_container_width=True):
            load_preset(6, 168, 88, 35, 240, 36.5, 1.15, 52)
            st.rerun()

    if st.button("🟡 Borderline / Pre-Diabetic", use_container_width=True):
        load_preset(3, 135, 78, 28, 145, 29.2, 0.65, 41)
        st.rerun()

    st.divider()
    st.markdown("### 🧬 Model Architecture")
    st.info(
        "**Algorithm:** K-Nearest Neighbours (K=5)\n\n"
        "**Preprocessing:** StandardScaler Pipeline\n\n"
        "**Dataset:** Pima Indians Diabetes (768 cohorts)\n\n"
        "**Validation Accuracy:** 70.13%"
    )

# ── Hero Title Banner ────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown("""
<div class="hero-banner">
    <span class="badge">CLINICAL DECISION SUPPORT SYSTEM • v2.4</span>
    <h1>GlucoGuard™ Diagnostic Assessment Suite</h1>
    <p>Real-time KNN inference via FastAPI microservice with automated biomarker risk profiling.</p>
</div>
""", unsafe_allow_html=True)

# ── 2. Eight Input Widgets in 3 Categorized Clinical Columns ─────────────────
st.subheader("📋 Patient Biomarkers & Vitals Input")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("##### 🩸 Metabolic Panel")
    glucose = st.slider(
        "Glucose (mg/dL)", min_value=0, max_value=200,
        value=st.session_state.preset["Glucose"], step=1,
        help="Plasma glucose concentration (2-hour oral glucose tolerance test). Normal fasting: <100 mg/dL."
    )
    insulin = st.slider(
        "Insulin (mu U/ml)", min_value=0, max_value=850,
        value=st.session_state.preset["Insulin"], step=1,
        help="2-Hour serum insulin level. Normal range: 16–166 mu U/ml."
    )
    dpf = st.number_input(
        "DiabetesPedigreeFunction", min_value=0.0, max_value=2.5,
        value=float(st.session_state.preset["DiabetesPedigreeFunction"]), step=0.01,
        help="Genetic diabetes likelihood score based on family history."
    )

with col2:
    st.markdown("##### ❤️ Physical & Cardiovascular")
    bmi = st.slider(
        "BMI (kg/m²)", min_value=0.0, max_value=70.0,
        value=float(st.session_state.preset["BMI"]), step=0.1,
        help="Body Mass Index. Normal: 18.5–24.9 | Overweight: 25–29.9 | Obese: ≥30."
    )
    blood_pressure = st.slider(
        "BloodPressure (mm Hg)", min_value=0, max_value=130,
        value=st.session_state.preset["BloodPressure"], step=1,
        help="Diastolic blood pressure. Normal diastolic: <80 mm Hg."
    )
    skin_thickness = st.slider(
        "SkinThickness (mm)", min_value=0, max_value=100,
        value=st.session_state.preset["SkinThickness"], step=1,
        help="Triceps skin fold thickness (mm), used to estimate body fat."
    )

with col3:
    st.markdown("##### 👤 Patient Demographics")
    age = st.slider(
        "Age (years)", min_value=18, max_value=90,
        value=st.session_state.preset["Age"], step=1,
        help="Patient age in years."
    )
    pregnancies = st.slider(
        "Pregnancies", min_value=0, max_value=17,
        value=st.session_state.preset["Pregnancies"], step=1,
        help="Number of times pregnant."
    )

    # Live mini-summary inside column 3
    bmi_category = (
        "Underweight" if bmi < 18.5 else
        "Normal Weight" if bmi < 25 else
        "Overweight" if bmi < 30 else "Obese Class"
    )
    st.caption(f"📌 **Live Vitals Tag:** `{bmi_category}` | `{age} yrs`")

st.write("")

# ── 3. Predict Button & API Call ─────────────────────────────────────────────
predict_clicked = st.button("⚡ Run Diagnostic Prediction", type="primary", use_container_width=True)

if predict_clicked:
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

    with st.spinner("Querying FastAPI Inference Engine..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                result_label = data["result_label"]
                confidence = data["confidence"]
                pred_int = data["prediction"]

                st.divider()
                st.subheader("📊 Diagnostic Evaluation Report")

                res_col1, res_col2 = st.columns([1.4, 1], gap="large")

                with res_col1:
                    # 4. Clear Result Display (Required st.error / st.success)
                    if result_label == "Diabetic":
                        st.error(f"🚨 **Clinical Classification: {result_label}** (High Risk Detected)")
                        st.markdown(
                            "The patient's biomarker profile closely matches diabetic cohorts in the "
                            "feature space. Confirmatory **HbA1c** and fasting plasma glucose testing is advised."
                        )
                    else:
                        st.success(f"✅ **Clinical Classification: {result_label}** (Low Risk Profile)")
                        st.markdown(
                            "The patient's biomarker profile aligns with non-diabetic cohorts. "
                            "Continue routine annual wellness screening and healthy lifestyle maintenance."
                        )

                    # Visual Confidence Bar
                    st.write("")
                    st.caption(" KNN Nearest-Neighbour Consensus Strength")
                    st.progress(min(max(float(confidence), 0.0), 1.0))

                with res_col2:
                    # Required st.metric confidence display as percentage
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric(
                            label="Model Confidence",
                            value=f"{confidence * 100:.1f}%",
                            delta="High Certainty" if confidence >= 0.75 else "Moderate"
                        )
                    with m2:
                        st.metric(
                            label="Predicted Class ID",
                            value=f"Class {pred_int}",
                            delta=result_label
                        )

                # ── Clinical Biomarker Flag Breakdown ────────────────────────
                st.write("")
                st.markdown("#### 🔬 Automated Biomarker Threshold Analysis")
                f1, f2, f3, f4 = st.columns(4)

                with f1:
                    g_flag = "🔴 Elevated" if glucose >= 140 else ("🟡 Borderline" if glucose >= 100 else "🟢 Optimal")
                    st.metric("Glucose Status", f"{glucose} mg/dL", g_flag)
                with f2:
                    b_flag = "🔴 Obese" if bmi >= 30 else ("🟡 Overweight" if bmi >= 25 else "🟢 Normal")
                    st.metric("BMI Index", f"{bmi:.1f} kg/m²", b_flag)
                with f3:
                    bp_flag = "🔴 High Diastolic" if blood_pressure >= 80 else "🟢 Normal"
                    st.metric("Diastolic BP", f"{blood_pressure} mmHg", bp_flag)
                with f4:
                    dpf_flag = "🔴 Strong Genetic" if dpf >= 0.8 else "🟢 Low/Avg"
                    st.metric("Pedigree Score", f"{dpf:.2f}", dpf_flag)

                # Save to Session History
                st.session_state.history.insert(0, {
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Result": result_label,
                    "Confidence": f"{confidence * 100:.1f}%",
                    "Glucose": glucose,
                    "BMI": bmi,
                    "BP": blood_pressure,
                    "Age": age
                })

                # Expandable Raw JSON Inspector for Technical Defense / Demo
                with st.expander("🛠️ View Raw FastAPI Request & Response Payload (JSON)"):
                    j1, j2 = st.columns(2)
                    with j1:
                        st.markdown("**POST Request (`/predict`)**")
                        st.json(payload)
                    with j2:
                        st.markdown("**HTTP 200 Response**")
                        st.json(data)

            else:
                st.warning(
                    f"⚠️ FastAPI returned HTTP {response.status_code}. "
                    "Please make sure the FastAPI server is running properly and `model.pkl` is loaded."
                )

        # 5. Graceful Failure (Friendly message instead of raw traceback)
        except requests.exceptions.RequestException:
            st.warning(
                "🔌 **Could not connect to the FastAPI backend.** "
                "Please start your FastAPI server first by running `uvicorn main:app --reload` "
                "and verify that your ngrok tunnel is active."
            )

# ── Session Audit Log & CSV Export ───────────────────────────────────────────
if st.session_state.history:
    st.divider()
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.subheader("🗂️ Session Diagnostic Log")
    with h_col2:
        df_hist = pd.DataFrame(st.session_state.history)
        st.download_button(
            "📥 Export Log (CSV)",
            data=df_hist.to_csv(index=False).encode("utf-8"),
            file_name="diabetes_predictions_log.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)