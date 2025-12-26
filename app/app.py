import streamlit as st
import pandas as pd
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# ===============================
# PATH SETUP
# ===============================
project_root = Path(__file__).parent.parent
model_path = project_root / "src" / "output" / "spark_rf_model"

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="IT Salary Prediction", layout="centered")
st.title("IT SALARY PREDICTION ANALYSIS")
st.markdown("*Using Linear Regression Model (Spark ML)*")

# ===============================
# LOAD SPARK + MODEL
# ===============================
@st.cache_resource
def load_spark_assets():
    spark = SparkSession.builder.appName("Salary_App").getOrCreate()
    model = PipelineModel.load(str(model_path.absolute()))
    return spark, model

spark, model = load_spark_assets()

# ===============================
# SESSION STATE
# ===============================
if 'form_key' not in st.session_state:
    st.session_state.form_key = 0


# ===============================
# FORM UI (GIỮ NGUYÊN)
# ===============================
def create_form():
    with st.form(key=f"salary_form_{st.session_state.form_key}"):

        st.subheader("Enter Prediction Information")
        col1, col2 = st.columns(2)

        with col1:
            work_year_label = st.selectbox(
                "Working Year*",
                ["---", "2020", "2021", "2022", "2023", "2024", "2025"],
                key=f"work_year_{st.session_state.form_key}"
            )
            work_year = int(work_year_label) if work_year_label != "---" else None

            remote_ratio_label = st.selectbox(
                "Remote Work Ratio*",
                ["---", "0", "50", "100"],
                key=f"remote_ratio_{st.session_state.form_key}"
            )
            remote_ratio = int(remote_ratio_label) if remote_ratio_label != "---" else None

        with col2:
            experience_label = st.selectbox(
                "Experience Level*",
                ["---", "Fresher (EN)", "Junior (MI)", "Senior (SE)", "Executive (EX)"],
                key=f"experience_{st.session_state.form_key}"
            )
            experience_level = experience_label.split(" ")[-1][1:-1] if experience_label != "---" else None

            employment_label = st.selectbox(
                "Employment Type*",
                ["---", "Full-time (FT)", "Part-time (PT)", "Contract (CT)", "Freelance (FL)"],
                key=f"employment_{st.session_state.form_key}"
            )
            employment_type = employment_label.split(" ")[-1][1:-1] if employment_label != "---" else None

        company_size_label = st.selectbox(
            "Company Size*",
            ["---", "Small (S)", "Medium (M)", "Large (L)"],
            key=f"company_size_{st.session_state.form_key}"
        )
        company_size = company_size_label.split(" ")[-1][1:-1] if company_size_label != "---" else None

        country_options = [
            "---", "United States (US)", "United Kingdom (GB)", "India (IN)",
            "Canada (CA)", "Germany (DE)", "France (FR)", "Vietnam (VN)",
            "Japan (JP)", "Australia (AU)"
        ]

        company_location_label = st.selectbox(
            "Company Location*",
            country_options,
            key=f"company_loc_{st.session_state.form_key}"
        )
        company_location = company_location_label.split(" ")[-1][1:-1] if company_location_label != "---" else None

        employee_residence_label = st.selectbox(
            "Employee Residence Country*",
            country_options,
            key=f"employee_res_{st.session_state.form_key}"
        )
        employee_residence = employee_residence_label.split(" ")[-1][1:-1] if employee_residence_label != "---" else None

        job_title = st.selectbox(
            "Job Title*",
            ["---", "Data Engineer", "Data Scientist", "Machine Learning Engineer",
             "Data Analyst", "Data Architect"],
            key=f"job_title_{st.session_state.form_key}"
        )
        job_title = job_title if job_title != "---" else None

        submitted = st.form_submit_button("PREDICT SALARY", type="primary")
        reset = st.form_submit_button("RESET")

        return submitted, reset, {
            'work_year': work_year,
            'remote_ratio': remote_ratio,
            'experience_level': experience_level,
            'employment_type': employment_type,
            'company_size': company_size,
            'company_location': company_location,
            'employee_residence': employee_residence,
            'job_title': job_title
        }


submitted, reset, form_data = create_form()

# ===============================
# RESET
# ===============================
if reset:
    st.session_state.form_key += 1
    st.rerun()

# ===============================
# PREDICT (SPARK)
# ===============================
if submitted:
    missing_fields = [field for field, value in form_data.items() if value is None]

    if missing_fields:
        st.error("❌ Please fill in all fields")
    else:
        # 1. Tạo Pandas DF từ input
        input_data = pd.DataFrame([form_data])

        try:
            # 2. Chuyển sang Spark DF
            spark_input = spark.createDataFrame(input_data)

            # 3. Predict (Pipeline tự chạy Indexer/Scaler/RF)
            result = model.transform(spark_input)

            prediction = result.select("prediction").collect()[0][0]

            st.success("✅ Prediction Successful!")
            st.metric("Predicted Salary (USD/year)", f"${round(prediction):,}")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")