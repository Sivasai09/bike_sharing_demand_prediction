import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# BikeWise - Bike Rental Demand Prediction
# Design 1: Modern Dashboard (Blue + Cyan)
# Existing model, input fields, feature encoding and prediction
# logic are retained.
# ------------------------------------------------------------

st.set_page_config(
    page_title="BikeWise | Bike Rental Demand Prediction",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    return joblib.load("bike_sharing_model.pkl")

model = load_model()

# Exact 20-feature order used by the saved model
features = [
    "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "temp", "atemp", "hum", "windspeed", "day", "is_weekend",
    "season_fall", "season_spring", "season_summer", "season_winter",
    "weathersit_Clear", "weathersit_Heavy Rain",
    "weathersit_Light Snow", "weathersit_Mist"
]

# ------------------------------ DESIGN ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #06172b;
    --navy2: #08233d;
    --panel: #092b43;
    --field: #102f49;
    --cyan: #20e5e1;
    --cyan2: #00bce8;
    --white: #f4fbff;
    --muted: #b8d0df;
}
.stApp {
    background:
      radial-gradient(ellipse at 72% 8%, #07558b 0%, transparent 33%),
      radial-gradient(ellipse at 48% 62%, #092c4c 0%, transparent 48%),
      linear-gradient(135deg, #041326 0%, #061d36 55%, #031226 100%);
    color: var(--white);
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
.block-container {
    max-width: 1600px;
    padding-top: 1rem;
    padding-bottom: 1.5rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #04152b 0%, #06294b 65%, #07345a 100%);
    border-right: 1px solid #0878a8;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

.brand {
    text-align: center;
    padding: 9px 4px 20px;
    margin-bottom: 16px;
    border-bottom: 1px solid #12618b;
}
.brand-icon {
    color: #20e5e1;
    font-size: 43px;
    line-height: 1.15;
    filter: drop-shadow(0 0 11px #00d9ef);
}
.brand-name {
    color: #f5fbff;
    font-size: 20px;
    font-weight: 800;
}
.brand-name span { color: #20e5e1; }
.nav-item {
    padding: 10px 12px;
    margin: 7px 0;
    border-radius: 9px;
    color: #e2f1fa;
    font-size: 13px;
    font-weight: 600;
}
.nav-active {
    color: #fff;
    background: linear-gradient(90deg, #087be9, #064dbb);
    box-shadow: 0 5px 18px #006bd455;
}
.sidebar-tagline {
    margin-top: 40px;
    color: #20e5e1;
    font-size: 13px;
    line-height: 1.5;
    font-style: italic;
    font-weight: 700;
    text-align: center;
    transform: rotate(-5deg);
}
.sidebar-city {
    margin-top: 42px;
    text-align: center;
    font-size: 42px;
    opacity: .8;
    filter: drop-shadow(0 0 10px #008bd8);
}

/* Design 1 hero panel */
.hero-card {
    min-height: 340px;
    overflow: hidden;
    border: 1px solid #087fb2;
    border-radius: 18px;
    background: linear-gradient(145deg, #06233f 0%, #07518a 55%, #03182e 100%);
    box-shadow: 0 12px 34px #0008, inset 0 0 24px #087cb533;
    margin-bottom: 16px;
}
.hero-copy { padding: 25px 24px 12px; }
.hero-kicker {
    color: #24e6df;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 8px;
}
.hero-title {
    color: #f7fcff;
    font-size: clamp(25px, 2.7vw, 39px);
    font-weight: 800;
    line-height: 1.08;
    margin: 0 0 10px;
}
.hero-title span { color: #20e5e1; }
.hero-sub {
    color: #d5eaf6;
    font-size: 15px;
    margin: 0;
}
.hero-photo {
    height: 240px;
    background-image:
      linear-gradient(180deg, #06233f00 60%, #061b30aa 100%),
      url('https://images.unsplash.com/photo-1485965120184-e220f721d03e?auto=format&fit=crop&w=1200&q=88');
    background-size: cover;
    background-position: center 54%;
    border-top: 1px solid #1688b7;
}

/* Prediction form card */
.form-card {
    border: 1px solid #087fb2;
    border-radius: 18px;
    padding: 17px 17px 15px;
    background: linear-gradient(160deg, #082c49 0%, #061b32 100%);
    box-shadow: 0 12px 34px #0008, inset 0 0 20px #0077aa22;
}
.form-title {
    color: #f5fcff;
    font-size: 18px;
    font-weight: 800;
    padding: 0 0 12px;
    margin-bottom: 10px;
    border-bottom: 1px solid #12618b;
}
.section-heading {
    color: #f5fbff;
    font-size: 21px;
    font-weight: 800;
    margin: 5px 0 13px;
}
.group-label {
    color: #24e6df;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.7px;
    text-transform: uppercase;
    margin: 9px 0 5px;
}

/* Inputs */
label, .stMarkdown, p { color: #eaf5fb; }
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #102f49 !important;
    border: 1px solid #245b7b !important;
    border-radius: 9px !important;
    color: #f5fbff !important;
}
div[data-testid="stNumberInput"] button {
    background: #e5edf2 !important;
    color: #092033 !important;
}
div[data-testid="stSlider"] { padding-top: 0; }
div[data-testid="stSlider"] [role="slider"] {
    background-color: #20e5d1 !important;
    border: 2px solid #20e5d1 !important;
    box-shadow: 0 0 8px #20e5d177;
}
div[data-testid="stSlider"] [data-testid="stTickBar"] { background: #314a60; }
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #ff4c59 !important;
    font-weight: 800 !important;
}
div[data-testid="stSlider"] [data-testid="stSliderTrack"] {
    background: #33485d !important;
}
.stButton > button, div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 45px;
    color: #032033 !important;
    background: linear-gradient(100deg, #12d9d3, #00aee9) !important;
    border: 0 !important;
    border-radius: 11px !important;
    font-weight: 800 !important;
    box-shadow: 0 7px 20px #00bfe955;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    color: #032033 !important;
    background: linear-gradient(100deg, #53f2e4, #39c8ff) !important;
    border: 0 !important;
}

/* Prediction result */
.result-card {
    display: flex;
    align-items: center;
    gap: 18px;
    border: 1px solid #0788a9;
    border-radius: 16px;
    padding: 16px 19px;
    margin-top: 15px;
    background: linear-gradient(105deg, #063b50 0%, #06253e 100%);
    box-shadow: 0 9px 25px #0007, inset 0 0 20px #00d5d522;
}
.result-check {
    flex: 0 0 42px;
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #032a35;
    font-size: 25px;
    font-weight: 900;
    background: linear-gradient(135deg, #16f0c5, #00b9df);
    box-shadow: 0 0 18px #00e3d477;
}
.result-label {
    color: #20e5d1;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.7px;
    text-transform: uppercase;
}
.result-caption { color: #d5e8f2; font-size: 12px; margin-top: 5px; }
.result-number {
    color: #20e5e1;
    font-size: 37px;
    line-height: 1.1;
    font-weight: 800;
}
.result-description {
    color: #d5e8f2;
    font-size: 11px;
    line-height: 1.45;
    border-left: 1px solid #1689a4;
    padding-left: 17px;
}
.result-description b { color: #20e5e1; }

@media (max-width: 1050px) {
    .hero-photo { height: 190px; }
}
@media (max-width: 700px) {
    .hero-card { min-height: auto; }
    .hero-photo { height: 170px; }
    .result-card { flex-wrap: wrap; }
    .result-description { border-left: 0; padding-left: 0; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------ SIDEBAR ------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🚲</div>
        <div class="brand-name">Bike<span>Wise</span></div>
    </div>
    <div class="nav-item nav-active">⌂ &nbsp; Home</div>
    <div class="nav-item">❖ &nbsp; About Project</div>
    <div class="nav-item">▥ &nbsp; Insights</div>
    <div class="sidebar-tagline">Ride Smart...<br>Predict Better...</div>
    <div class="sidebar-city">▥ ▥ ▥</div>
    """, unsafe_allow_html=True)

# ------------------------------ MAIN LAYOUT ------------------------------
hero_col, form_col = st.columns([1.05, 1.0], gap="medium")

with hero_col:
    st.markdown("""
    <div class="hero-card">
      <div class="hero-copy">
        <div class="hero-kicker">BIKE RENTAL</div>
        <div class="hero-title">Bike Rental<br><span>Demand Prediction</span></div>
        <p class="hero-sub">Analyze. Predict. Plan.</p>
      </div>
      <div class="hero-photo"></div>
    </div>
    """, unsafe_allow_html=True)

with form_col:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<div class="form-title">▥ &nbsp; Prediction Form</div>', unsafe_allow_html=True)

    # All original input controls are kept in the prediction form.
    with st.form("bike_prediction_form"):
        col_a, col_b = st.columns(2, gap="small")

        with col_a:
            yr_label = st.selectbox("Year", ["2011", "2012"])
            mnth = st.selectbox("Month", list(range(1, 13)), index=5)
            day = st.number_input("Day of month", min_value=1, max_value=31, value=15, step=1)
            holiday_choice = st.selectbox("Public holiday?", ["No", "Yes"], index=0)
            workingday_choice = st.selectbox("Working day?", ["Working Day", "No work"], index=0)
            season = st.selectbox(
                "Season", ["fall", "spring", "summer", "winter"],
                index=1, format_func=lambda x: x.capitalize()
            )
            weathersit = st.selectbox(
                "Weather situation",
                ["Clear", "Mist", "Light Snow", "Heavy Rain"],
                index=0
            )

        with col_b:
            hr = st.slider("Hour of day", min_value=0, max_value=23, value=8)
            weekday = st.selectbox(
                "Day of week",
                options=list(range(7)),
                format_func=lambda x: [
                    "Sunday", "Monday", "Tuesday", "Wednesday",
                    "Thursday", "Friday", "Saturday"
                ][x],
                index=0
            )
            temp = st.slider(
                "Temperature (scaled)", min_value=0.0, max_value=1.0,
                value=0.50, step=0.01, format="%.2f"
            )
            atemp = st.slider(
                "Feels-like temperature (scaled)", min_value=0.0, max_value=1.0,
                value=0.50, step=0.01, format="%.2f"
            )
            hum = st.slider(
                "Humidity (scaled)", min_value=0.0, max_value=1.0,
                value=0.50, step=0.01, format="%.2f"
            )
            windspeed = st.slider(
                "Wind speed (scaled)", min_value=0.0, max_value=1.0,
                value=0.20, step=0.01, format="%.2f"
            )

        submitted = st.form_submit_button("➤  Predict Demand")

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------ PREDICTION ------------------------------
if submitted:
    # Preserve the original label encoding and feature construction.
    yr = 0 if yr_label == "2011" else 1
    holiday = 0 if holiday_choice == "No" else 1
    workingday = 1 if workingday_choice == "Working Day" else 0
    is_weekend = 1 if weekday in [0, 6] else 0

    input_data = {
        "yr": yr,
        "mnth": mnth,
        "hr": hr,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "day": day,
        "is_weekend": is_weekend,
        "season_fall": 1 if season == "fall" else 0,
        "season_spring": 1 if season == "spring" else 0,
        "season_summer": 1 if season == "summer" else 0,
        "season_winter": 1 if season == "winter" else 0,
        "weathersit_Clear": 1 if weathersit == "Clear" else 0,
        "weathersit_Heavy Rain": 1 if weathersit == "Heavy Rain" else 0,
        "weathersit_Light Snow": 1 if weathersit == "Light Snow" else 0,
        "weathersit_Mist": 1 if weathersit == "Mist" else 0
    }

    missing_features = [col for col in features if col not in input_data]
    if missing_features:
        st.error(f"Missing model features: {missing_features}")
    else:
        input_df = pd.DataFrame(
            [[input_data[col] for col in features]],
            columns=features
        )
        prediction = model.predict(input_df)
        predicted_count = max(0, round(float(prediction[0])))
        st.session_state["bike_prediction"] = predicted_count

# Show the prediction result after the user submits the form.
if "bike_prediction" in st.session_state:
    predicted_count = st.session_state["bike_prediction"]
    st.markdown(f"""
    <div class="result-card">
      <div class="result-check">✓</div>
      <div>
        <div class="result-label">YOUR PREDICTION</div>
        <div class="result-caption"><b>Estimated bike rentals</b></div>
        <div class="result-number">{predicted_count:,}</div>
      </div>
      <div class="result-description">
        The model estimates approximately<br>
        <b>{predicted_count:,}</b> bike rentals for the selected conditions.
        <br><i>This is a machine-learning estimate.</i>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="result-card">
      <div class="result-check">✓</div>
      <div>
        <div class="result-label">YOUR PREDICTION</div>
        <div class="result-caption"><b>Estimated bike rentals</b></div>
        <div class="result-number">—</div>
      </div>
      <div class="result-description">
        The model estimates bike rentals for the selected conditions.
        <br><i>This is a machine-learning estimate.</i>
      </div>
    </div>
    """, unsafe_allow_html=True)
