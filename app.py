import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Dropout Risk Predictor", page_icon="🎓", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dropout_model.pkl")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


bundle = load_bundle()
pipe = bundle["pipeline"]
cat_options = bundle["categorical_options"]


def predict_risk(df: pd.DataFrame) -> float:
    """Return probability of dropout (0-1) for the first row of df."""
    if hasattr(pipe, "predict_proba"):
        return float(pipe.predict_proba(df)[0, 1])
    score = float(pipe.decision_function(df)[0])      # e.g. LinearSVC
    return 1.0 / (1.0 + np.exp(-score))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About the model")
    st.write(f"**Algorithm:** {bundle['model_name']}")
    m = bundle["cv_metrics"]
    st.write("**10-fold CV performance**")
    st.write(f"- Accuracy: {m['accuracy']:.3f}")
    st.write(f"- Precision: {m['precision']:.3f}")
    st.write(f"- Recall: {m['recall']:.3f}")
    st.write(f"- F1-score: {m['f1']:.3f}")
    st.write(f"- ROC-AUC: {m['roc_auc']:.3f}")
    st.caption("Trained on 100,000 student records. This tool supports early "
               "intervention and should not be the only basis for any decision about a student.")

st.title("🎓 Student Dropout Risk Predictor")
st.write("Enter a student's details and click **Predict** to estimate the probability of dropping out.")

YES_NO = {"No": 0, "Yes": 1}
vals = {}

# ---------------------------------------------------------------- inputs
st.subheader("1. Background")
c1, c2, c3 = st.columns(3)
vals["age"] = c1.number_input("Age", 15, 30, 21)
vals["gender"] = c2.selectbox("Gender", cat_options["gender"])
vals["nationality"] = c3.selectbox("Nationality", cat_options["nationality"])
vals["family_income_level"] = c1.selectbox("Family income level", ["Low", "Middle", "High"], index=1)
vals["first_generation"] = YES_NO[c2.selectbox("First-generation student", list(YES_NO))]
vals["disability_status"] = YES_NO[c3.selectbox("Disability", list(YES_NO))]
vals["international_student"] = YES_NO[c1.selectbox("International student", list(YES_NO))]

st.subheader("2. Enrolment & finances")
c1, c2, c3 = st.columns(3)
vals["course_type"] = c1.selectbox("Course type", cat_options["course_type"])
vals["study_mode"] = c2.selectbox("Study mode", cat_options["study_mode"])
vals["semester"] = c3.slider("Semester", 1, 8, 3)
vals["scholarship_holder"] = YES_NO[c1.selectbox("Scholarship holder", list(YES_NO), index=1)]
vals["tuition_up_to_date"] = YES_NO[c2.selectbox("Tuition up to date", list(YES_NO), index=1)]
vals["part_time_job"] = YES_NO[c3.selectbox("Part-time job", list(YES_NO))]
vals["weekly_work_hours"] = c1.slider("Weekly work hours", 0, 40, 8)

st.subheader("3. Academic performance")
c1, c2, c3 = st.columns(3)
vals["admission_grade"] = c1.number_input("Admission grade (0-200)", 0.0, 200.0, 133.0)
vals["previous_qualification_grade"] = c2.number_input("Previous qualification grade (0-200)", 0.0, 200.0, 139.0)
vals["gpa_semester_1"] = c3.number_input("GPA semester 1 (0-4)", 0.0, 4.0, 2.6, step=0.1)
vals["gpa_semester_2"] = c1.number_input("GPA semester 2 (0-4)", 0.0, 4.0, 2.0, step=0.1)
vals["units_enrolled"] = c2.number_input("Units enrolled", 1, 24, 13)
vals["units_passed"] = c3.number_input("Units passed", 0, 24, 10)
vals["units_failed"] = c1.number_input("Units failed", 0, 17, 3)

st.subheader("4. Engagement & wellbeing")
c1, c2, c3 = st.columns(3)
vals["attendance_rate"] = c1.slider("Attendance rate (%)", 0.0, 100.0, 75.0)
vals["assignments_submitted_rate"] = c2.slider("Assignments submitted (%)", 0.0, 100.0, 71.0)
vals["missed_deadlines"] = c3.number_input("Missed deadlines", 0, 20, 2)
vals["library_visits_per_month"] = c1.number_input("Library visits per month", 0, 30, 4)
vals["lms_logins_per_week"] = c2.number_input("LMS logins per week", 0.0, 30.0, 7.0, step=0.5)
vals["extracurricular_activities"] = YES_NO[c3.selectbox("Extracurricular activities", list(YES_NO), index=1)]
vals["stress_level"] = c1.slider("Stress level (1-10)", 1, 10, 6)
vals["satisfaction_score"] = c2.slider("Satisfaction score (1-10)", 1, 10, 6)
vals["mental_health_support"] = YES_NO[c3.selectbox("Receives mental-health support", list(YES_NO))]
vals["commute_time_mins"] = c1.number_input("Commute time (minutes)", 0, 120, 23)
vals["sleep_hours"] = c2.slider("Sleep hours per night", 4.0, 10.0, 6.5, step=0.1)

if vals["units_passed"] + vals["units_failed"] > vals["units_enrolled"]:
    st.warning("Units passed + units failed is greater than units enrolled - please double-check.")

# ---------------------------------------------------------------- prediction
if st.button("🔮 Predict dropout risk", type="primary"):
    row = pd.DataFrame([vals])[bundle["feature_columns"]]     # same column order as training
    prob = predict_risk(row)

    st.divider()
    st.subheader("Result")
    a, b = st.columns([1, 2])
    a.metric("Dropout probability", f"{prob * 100:.1f}%")
    with b:
        st.progress(min(max(prob, 0.0), 1.0))
        if prob < 0.30:
            st.success("🟢 Low risk")
        elif prob < 0.60:
            st.warning("🟡 Medium risk - consider monitoring this student.")
        else:
            st.error("🔴 High risk - early support or intervention is recommended.")
