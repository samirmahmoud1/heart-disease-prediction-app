import time
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Load CSS
# =========================
def load_css():
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =========================
# Load Artifacts
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("heart_disease_model.pkl")
    selected_features = joblib.load("selected_features.pkl")
    return model, selected_features

model, selected_features = load_model()

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("test.csv")
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Create binary target
    df["target"] = df["thallium"].apply(lambda x: 0 if x == 3 else 1)

    # Feature engineering
    df["heart_stress"] = df["st_depression"] * df["exercise_angina"]
    df["age_group"] = pd.cut(
        df["age"],
        bins=[20, 40, 55, 70, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return df

df = load_data()

# =========================
# Sidebar
# =========================
st.sidebar.markdown(
    """
    <div style='text-align:center; padding-top:10px;'>
        <h2 class='sidebar-title'>🫀 Heart Project</h2>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "EDA", "Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class='sidebar-box'>
        <p><b>Project Type:</b> Medical ML App</p>
        <p><b>Final Model:</b> Logistic Regression</p>
        <p><b>Use Case:</b> Early Heart Disease Risk Screening</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# HOME PAGE
# =========================
if page == "Home":
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-left">
                <div class="badge">AI-Powered Medical Decision Support</div>
                <h1>Heart Disease Prediction System</h1>
                <p>
                    A professional machine learning web application designed to support
                    early risk screening for heart disease using clinical indicators,
                    exploratory analytics, and real-time prediction.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{df.shape[0]//1000}K+</h3>
                <p>Patient Records</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h3>75.4%</h3>
                <p>Best Accuracy</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h3>3</h3>
                <p>Models Compared</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{len(selected_features)}</h3>
                <p>Selected Features</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## Project Overview")

    c1, c2 = st.columns([1.2, 1])

    with c1:
      with st.container():
        st.markdown(
            """
            <div class="glass-card">
            """,
            unsafe_allow_html=True
        )

        st.markdown("### About This Project")
        st.write(
            "This application leverages machine learning to predict the likelihood "
            "of heart disease based on clinical patient data. It aims to support "
            "early detection and risk assessment."
        )

        st.markdown("### Core Capabilities")
        st.write("✔ Analyze patient clinical indicators")
        st.write("✔ Identify important health risk patterns")
        st.write("✔ Compare multiple machine learning models")
        st.write("✔ Generate real-time heart disease risk predictions")

        st.markdown(
            """
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Workflow</h3>
                <div class="timeline-step">1. Data Cleaning</div>
                <div class="timeline-step">2. Exploratory Data Analysis</div>
                <div class="timeline-step">3. Feature Engineering</div>
                <div class="timeline-step">4. Model Training</div>
                <div class="timeline-step">5. Hyperparameter Tuning</div>
                <div class="timeline-step">6. Deployment with Streamlit</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## Key Medical Indicators")

    x1, x2, x3 = st.columns(3)

    cards = [
        (
            "Max Heart Rate",
            "One of the strongest predictive indicators identified by the model, showing an inverse relationship with disease risk."
        ),
        (
            "ST Depression",
            "Represents exercise-induced cardiac stress and contributes strongly to heart disease prediction."
        ),
        (
            "Exercise Angina",
            "A clinically meaningful symptom that helps distinguish high-risk patients from lower-risk cases."
        ),
    ]

    for col, (title, text) in zip([x1, x2, x3], cards):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <h4>{title}</h4>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
# =========================
# EDA PAGE
# =========================
elif page == "EDA":
    st.markdown(
        """
        <div class="glass-card">
            <h1 class='page-title'>Exploratory Data Analysis</h1>
            <p class='page-subtitle'>
                Interactive visual exploration of the clinical dataset to understand
                distributions, relationships, and patterns across the most relevant medical indicators.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{df.shape[0]:,}</h3>
                <p>Total Records</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{df.shape[1]}</h3>
                <p>Total Features</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k3:
        healthy_pct = df["target"].value_counts(normalize=True).get(0, 0) * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{healthy_pct:.1f}%</h3>
                <p>Healthy Cases</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with k4:
        disease_pct = df["target"].value_counts(normalize=True).get(1, 0) * 100
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{disease_pct:.1f}%</h3>
                <p>Disease Cases</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>📊 Target Distribution</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig_target = px.histogram(
            df,
            x="target",
            color="target",
            barmode="group",
            color_discrete_sequence=["#8ecae6", "#219ebc"]
        )
        fig_target.update_layout(
            template="simple_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            xaxis_title="Target",
            yaxis_title="Count",
            height=420,
            showlegend=True,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig_target.update_traces(marker_line_width=0)
        st.plotly_chart(fig_target, use_container_width=True)

    with row1_col2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>👤 Age Distribution</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig_age = px.histogram(
            df,
            x="age",
            nbins=30,
            color_discrete_sequence=["#2a9d8f"]
        )
        fig_age.update_layout(
            template="simple_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            xaxis_title="Age",
            yaxis_title="Count",
            height=420,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        fig_age.update_traces(marker_line_width=0)
        st.plotly_chart(fig_age, use_container_width=True)

    # Row 2
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🧪 Cholesterol vs Target</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig_chol = px.box(
            df,
            x="target",
            y="cholesterol",
            color="target",
            color_discrete_sequence=["#8ecae6", "#219ebc"]
        )
        fig_chol.update_layout(
            template="simple_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            xaxis_title="Target",
            yaxis_title="Cholesterol",
            height=420,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_chol, use_container_width=True)

    with row2_col2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>❤️ Max Heart Rate vs Target</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig_hr = px.box(
            df,
            x="target",
            y="max_hr",
            color="target",
            color_discrete_sequence=["#8ecae6", "#219ebc"]
        )
        fig_hr.update_layout(
            template="simple_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            xaxis_title="Target",
            yaxis_title="Max Heart Rate",
            height=420,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    # Row 3
    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>⚡ ST Depression vs Target</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig_st = px.box(
            df,
            x="target",
            y="st_depression",
            color="target",
            color_discrete_sequence=["#8ecae6", "#219ebc"]
        )
        fig_st.update_layout(
            template="simple_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            xaxis_title="Target",
            yaxis_title="ST Depression",
            height=420,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_st, use_container_width=True)

    with row3_col2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>🩺 Chest Pain Type by Target</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        chest_counts = (
            df.groupby(["chest_pain_type", "target"])
            .size()
            .reset_index(name="count")
        )

        fig_cp = px.bar(
            chest_counts,
            x="chest_pain_type",
            y="count",
            color="target",
            barmode="group",
            color_discrete_sequence=["#8ecae6", "#219ebc"]
        )
        fig_cp.update_layout(
            template="simple_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="white",
            xaxis_title="Chest Pain Type",
            yaxis_title="Count",
            height=420,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_cp, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <h3>🔗 Correlation Heatmap</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    corr = df[
        [
            "age",
            "sex",
            "chest_pain_type",
            "bp",
            "cholesterol",
            "fbs_over_120",
            "ekg_results",
            "max_hr",
            "exercise_angina",
            "st_depression",
            "slope_of_st",
            "number_of_vessels_fluro",
            "heart_stress",
            "target",
        ]
    ].corr()

    heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Blues",
            zmin=-1,
            zmax=1,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
        )
    )

    heatmap.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        height=720,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(heatmap, use_container_width=True)

    st.markdown("## EDA Insights")

    i1, i2 = st.columns(2)

    with i1:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Distribution Insights</h3>
                <ul>
                    <li><b>Target classes</b> are relatively balanced, supporting stable model training.</li>
                    <li><b>Age</b> is concentrated mostly among middle-aged and older adults.</li>
                    <li><b>Cholesterol</b> contributes to prediction but does not separate classes strongly on its own.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Relationship Insights</h3>
                <ul>
                    <li><b>Max heart rate</b> shows an inverse relationship with disease risk.</li>
                    <li><b>ST depression</b> and <b>exercise angina</b> appear to be strong predictive indicators.</li>
                    <li><b>Chest pain type</b> provides useful class separation in grouped analysis.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# =========================
# PREDICTION PAGE
# =========================
elif page == "Prediction":
    st.markdown("<h1 class='page-title'>Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='page-subtitle'>Enter patient clinical information to generate a prediction.</p>",
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Model Accuracy", "75.4%")

    with m2:
        st.metric("Model Type", "Logistic Regression")

    with m3:
        st.metric("Dataset Size", "270K+")

    st.markdown(
        """
        <div class="glass-card">
            <h3>Clinical Input Form</h3>
            <p>
                Fill in the patient’s core clinical indicators below.
                The model will estimate the likelihood of heart disease risk.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='form-wrapper'>", unsafe_allow_html=True)

    st.markdown("### Demographic Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 29, 77, 54)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    with col2:
        chest_pain_type = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
        bp = st.slider("Blood Pressure", 90, 200, 130)

    with col3:
        cholesterol = st.slider("Cholesterol", 120, 570, 245)
        max_hr = st.slider("Max Heart Rate", 70, 205, 150)

    st.markdown("### Stress & ECG Indicators")
    col4, col5, col6 = st.columns(3)

    with col4:
        exercise_angina = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col5:
        st_depression = st.slider("ST Depression", 0.0, 6.5, 1.0, 0.1)

    with col6:
        slope_of_st = st.selectbox("Slope of ST", [1, 2, 3])
        number_of_vessels_fluro = st.selectbox("Number of Vessels Fluro", [0, 1, 2, 3])

    heart_stress = st_depression * exercise_angina

    input_data = pd.DataFrame([{
        "max_hr": max_hr,
        "cholesterol": cholesterol,
        "bp": bp,
        "age": age,
        "st_depression": st_depression,
        "chest_pain_type": chest_pain_type,
        "number_of_vessels_fluro": number_of_vessels_fluro,
        "exercise_angina": exercise_angina,
        "heart_stress": heart_stress,
        "slope_of_st": slope_of_st
    }])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Risk")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        with st.spinner("Analyzing patient data..."):
            time.sleep(1.2)

            prediction = model.predict(input_data)[0]

            try:
                probability = model.predict_proba(input_data)[0][1]
            except Exception:
                probability = None

        if prediction == 1:
            st.markdown(
                """
                <div class="result-card danger">
                    <h2>Higher Risk Detected</h2>
                    <p>
                        The model predicts an elevated likelihood of heart disease.
                        This result should be interpreted as a screening support output,
                        not a final medical diagnosis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="result-card safe">
                    <h2>Lower Risk Detected</h2>
                    <p>
                        The model predicts a lower likelihood of heart disease
                        based on the provided clinical values.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        if probability is not None:
            st.markdown(
                f"""
                <div class="probability-box">
                    <h3>Predicted Risk Probability</h3>
                    <p>{probability:.2%}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(int(probability * 100))

        st.markdown(
            """
            <div class="glass-card">
                <h4>Clinical Note</h4>
                <p>
                    This prediction is based on statistical patterns learned from historical data.
                    It should be used as a supportive screening tool and not a substitute for professional medical diagnosis.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )