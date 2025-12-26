import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# Thêm thư mục gốc vào PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.preprocess import encode_features

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="IT Salary Prediction", layout="centered")
st.title("IT SALARY PREDICTION ANALYSIS")
st.markdown("*Using Linear Regression Model*")


# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('notebook/model.pkl')
        encoder = joblib.load('notebook/encoder.pkl')
        scaler = joblib.load('notebook/scaler.pkl')
        return model, encoder, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


model, encoder, scaler = load_model()

# --- KHỞI TẠO SESSION STATE ---
if 'form_key' not in st.session_state:
    st.session_state.form_key = 0


# --- TẠO FORM ĐỘNG ---
def create_form():
    with st.form(key=f"salary_form_{st.session_state.form_key}"):
        st.subheader("Enter Prediction Information")

        col1, col2 = st.columns(2)

        with col1:
            # Work Year selectbox
            work_year_options = ["---", "2020", "2021", "2022", "2023", "2024", "2025"]
            work_year_label = st.selectbox(
                "Working Year*",
                options=work_year_options,
                index=0,
                key=f"work_year_{st.session_state.form_key}"
            )
            work_year = int(work_year_label) if work_year_label != "---" else None

            # Remote Ratio selectbox
            remote_ratio_options = ["---", "0", "50", "100"]
            remote_ratio_label = st.selectbox(
                "Remote Work Ratio*",
                options=remote_ratio_options,
                index=0,
                key=f"remote_ratio_{st.session_state.form_key}"
            )
            remote_ratio = int(remote_ratio_label) if remote_ratio_label != "---" else None

        with col2:
            # Experience Level selectbox
            experience_options = ["---", "Fresher (EN)", "Junior (MI)", "Senior (SE)",
                                  "Executive (EX)"]
            experience_label = st.selectbox(
                "Experience Level*",
                options=experience_options,
                index=0,
                key=f"experience_{st.session_state.form_key}"
            )
            experience_level = experience_label.split(" ")[-1][
                               1:-1] if experience_label != "---" else None

            # Employment Type selectbox
            employment_options = ["---", "Full-time (FT)", "Part-time (PT)", "Contract (CT)",
                                  "Freelance (FL)"]
            employment_label = st.selectbox(
                "Employment Type*",
                options=employment_options,
                index=0,
                key=f"employment_{st.session_state.form_key}"
            )
            employment_type = employment_label.split(" ")[-1][1:-1] if employment_label != "---" else None

        # Company Size selectbox
        company_size_options = ["---", "Small (S)", "Medium (M)", "Large (L)"]
        company_size_label = st.selectbox(
            "Company Size*",
            options=company_size_options,
            index=0,
            key=f"company_size_{st.session_state.form_key}"
        )
        company_size = company_size_label.split(" ")[-1][1:-1] if company_size_label != "---" else None

        # Country selectboxes
        country_options = ["---", "United States (US)", "United Kingdom (GB)", "India (IN)",
                           "Canada (CA)", "Germany (DE)", "France (FR)", "Vietnam (VN)",
                           "Japan (JP)", "Australia (AU)"]

        company_location_label = st.selectbox(
            "Company Location*",
            options=country_options,
            index=0,
            key=f"company_loc_{st.session_state.form_key}"
        )
        company_location = company_location_label.split(" ")[-1][
                           1:-1] if company_location_label != "---" else None

        employee_residence_label = st.selectbox(
            "Employee Residence Country*",
            options=country_options,
            index=0,
            key=f"employee_res_{st.session_state.form_key}"
        )
        employee_residence = employee_residence_label.split(" ")[-1][
                             1:-1] if employee_residence_label != "---" else None

        # Job Title selectbox
        job_title_options = ["---", "Data Engineer", "Data Scientist",
                             "Machine Learning Engineer", "Data Analyst", "Data Architect"]
        job_title = st.selectbox(
            "Job Title*",
            options=job_title_options,
            index=0,
            key=f"job_title_{st.session_state.form_key}"
        )
        job_title = job_title if job_title != "---" else None

        # --- NÚT PREDICT SALARY VÀ RESET ---
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


# Tạo form và nhận kết quả
submitted, reset, form_data = create_form()

# --- XỬ LÝ RESET ---
if reset:
    st.session_state.form_key += 1
    st.rerun()

# --- XỬ LÝ DỰ BÁO ---
if submitted and model is not None and encoder is not None and scaler is not None:
    # Ánh xạ key sang tên trường dễ hiểu
    FIELD_LABELS = {
        "work_year": "Working Year",
        "remote_ratio": "Remote Work Ratio",
        "experience_level": "Experience Level",
        "employment_type": "Employment Type",
        "company_size": "Company Size",
        "company_location": "Company Location",
        "employee_residence": "Employee Residence Country",
        "job_title": "Job Title"
    }

    # Kiểm tra các trường bị thiếu
    missing_fields = [field for field, value in form_data.items() if value is None]

    if missing_fields:
        missing_labels = [FIELD_LABELS[field] for field in missing_fields]
        st.error("❌ Please fill in all required fields:\n\n- " + "\n- ".join(missing_labels))
    else:
        input_data = pd.DataFrame({
            'work_year': [form_data['work_year']],
            'experience_level': [form_data['experience_level']],
            'employment_type': [form_data['employment_type']],
            'job_title': [form_data['job_title']],
            'employee_residence': [form_data['employee_residence']],
            'remote_ratio': [form_data['remote_ratio']],
            'company_location': [form_data['company_location']],
            'company_size': [form_data['company_size']]
        })

        try:
            X, _, _ = encode_features(
                input_data,
                encoder=encoder,
                scaler=scaler,
                fit_encoder=False,
                fit_scaler=False
            )

            prediction = model.predict(X)[0]
            predicted_salary = round(prediction)

            st.success("✅ Prediction Successful!")
            st.metric(
                label="**Predicted Salary (USD/year)**",
                value=f"${predicted_salary:,}",
                delta="Estimated result"
            )

            with st.expander("📊 View Input Details"):
                st.dataframe(input_data, hide_index=True)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
