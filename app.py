import streamlit as st
import pandas as pd
import os
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="FoodSight",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================== THEME / CSS ====================
def load_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --green: #00d66b;
            --green-dark: #00b85c;
            --black: #080b09;
            --panel: #101411;
            --panel-soft: #151b17;
            --muted: #8c9890;
            --line: rgba(0, 214, 107, .22);
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--black) !important;
            color: #f4f7f5 !important;
            font-family: 'Inter', sans-serif !important;
        }
        [data-testid="stHeader"] { background: transparent !important; }
        [data-testid="stToolbar"] { right: 1rem; }
        #MainMenu, footer { visibility: hidden; }

        .block-container {
            max-width: 1180px !important;
            padding: 2.2rem 2rem 3rem !important;
        }

        [data-testid="stVerticalBlock"] { gap: 1rem; }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 2rem;
            font-size: 1.05rem;
            font-weight: 800;
            letter-spacing: -.4px;
        }
        .brand-mark {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--green);
            color: #061109;
            font-size: 19px;
        }
        .green { color: var(--green); }

        /* LEFT PANEL */
        .hero-panel {
            background: var(--green);
            border-radius: 24px;
            padding: 1.1rem;
            min-height: 470px;
            display: flex;
            align-items: stretch;
            box-shadow: 0 18px 55px rgba(0, 214, 107, .10);
        }
        .hero-inner {
            width: 100%;
            background: #090d0a;
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 18px;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            justify-content: center; /* Center content since footer is removed */
        }
        .hero-kicker {
            color: var(--green);
            font-size: .9rem;
            font-weight: 800;
        }
        .hero-title {
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1.08;
            font-weight: 800;
            letter-spacing: -1.8px;
            margin: 2.2rem 0 1rem;
            color: #f7faf8;
        }
        .hero-copy {
            color: #aab5ae;
            font-size: .98rem;
            line-height: 1.7;
            max-width: 440px;
        }

        /* RIGHT PANEL & MAIN BUTTONS (VISUAL FOCUS) */
        .st-key-home_predict button,
        .st-key-home_history button {
            min-height: 80px !important;
            height: 30% !important;
            flex: 1;
            
            /* Bright green gradient background */
            background: linear-gradient(135deg, #2dff8c 0%, #00d66b 100%) !important;
            
            /* Dark, bold text */
            color: #031a0b !important;
            font-size: 3rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.3px;
            
            /* Rounded corners and subtle green glow */
            border: none !important;
            border-radius: 30px !important;
            box-shadow: 0 8px 25px rgba(0, 214, 107, 0.3) !important;
            
            /* Layout inside button */
            text-align: left !important;
            justify-content: flex-start !important;
            padding: 2rem !important;
            
            /* Hover animation setup */
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        }
        
        /* Hover Animation */
        .st-key-home_predict button:hover,
        .st-key-home_history button:hover {
            transform: translateY(-6px) scale(1.02) !important;
            box-shadow: 0 15px 35px rgba(0, 214, 107, 0.5) !important;
            background: linear-gradient(135deg, #4dffa1 0%, #00e673 100%) !important;
            color: #000000 !important;
        }

        /* OTHER ELEMENTS */
        .eyebrow {
            color: var(--green);
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: 1.7px;
            font-weight: 800;
            margin-bottom: .8rem;
        }
        .section-heading {
            font-size: 1.5rem;
            color: #f5f8f6;
            font-weight: 800;
            letter-spacing: -.6px;
            margin: .5rem 0 .3rem;
        }
        .subheading {
            color: var(--muted);
            font-size: .9rem;
            margin-bottom: 1.2rem;
        }
        .small-label {
            color: var(--green);
            font-size: .75rem;
            text-transform: uppercase;
            letter-spacing: 1.4px;
            font-weight: 800;
        }
        .metric-card {
            background: var(--panel-soft);
            border: 1px solid var(--line);
            border-radius: 15px;
            padding: 1.1rem .8rem;
            text-align: center;
            min-height: 120px;
        }
        .metric-emoji { font-size: 1.5rem; }
        .metric-value {
            color: #f5f8f6;
            font-size: 1.15rem;
            font-weight: 800;
            margin: .35rem 0 .15rem;
            overflow-wrap: anywhere;
        }
        .metric-label { color: var(--muted); font-size: .76rem; }
        .feature-copy {
            color: var(--muted);
            font-size: .82rem;
            line-height: 1.55;
        }

        /* STANDARD BUTTONS */
        .stButton > button, .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] > button {
            background: var(--green) !important;
            color: #061109 !important;
            border: 0 !important;
            border-radius: 11px !important;
            min-height: 2.8rem;
            font-weight: 800 !important;
            transition: .2s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            background: var(--green-dark) !important;
            color: #061109 !important;
            border: 0 !important;
            transform: translateY(-1px);
        }
        .stButton > button[kind="secondary"] {
            background: transparent !important;
            color: #eaf1ec !important;
            border: 1px solid rgba(255,255,255,.25) !important;
        }

        [data-testid="stFileUploader"] section {
            background: #111713 !important;
            border: 1px dashed rgba(0,214,107,.45) !important;
            border-radius: 14px !important;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: #111713 !important;
            color: #f5f8f6 !important;
            border-color: rgba(255,255,255,.12) !important;
            border-radius: 10px !important;
        }
        [data-testid="stRadio"] label, [data-testid="stWidgetLabel"] {
            color: #dce5df !important;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }
        .footer-note {
            color: #66736b;
            text-align: center;
            font-size: .75rem;
            margin-top: 2.2rem;
        }

        @media (max-width: 760px) {
            .block-container { padding: 1.2rem 1rem 2rem !important; }
            .hero-panel { min-height: auto; }
            .hero-inner { padding: 1.5rem; min-height: 340px; }
            .st-key-home_predict button,
            .st-key-home_history button { min-height: 120px !important; font-size: 1.3rem !important; }
            .hero-title { margin-top: 2rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================== MODEL & DATA ====================
@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=101)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, 101))
    model.load_state_dict(torch.load("model/best_efficientnet_b3.pth", map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def load_classes():
    with open("classes.txt", "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]


@st.cache_data
def load_calories_csv():
    return pd.read_csv("food_calories_usda.csv")


transform = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ==================== HELPER FUNCTIONS ====================
def predict_food(image, model, class_names):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
    top_probs, top_indices = torch.topk(probs, 3)
    return (
        [class_names[i] for i in top_indices.numpy()],
        [float(p) for p in top_probs.numpy()],
    )


def get_calories(food_class, df_calories):
    row = df_calories[df_calories["food_class"] == food_class]
    if not row.empty:
        value = row.iloc[0]["calories_per_100g"]
        return float(value) if pd.notna(value) and value != "" else None
    return None


def get_portion_weight(portion_size, custom_grams=None):
    if portion_size == "Other (custom)" and custom_grams is not None:
        return int(custom_grams)
    return {
        "Small (150g)": 150,
        "Medium (250g)": 250,
        "Large (400g)": 400,
        "Extra Large (600g)": 600,
    }.get(portion_size, 250)


def save_feedback(image, predicted_food, actual_food):
    os.makedirs(f"retraining_data/{actual_food}", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"retraining_data/{actual_food}/{timestamp}.jpg"
    image.save(filename)
    return f"Saved correction: {predicted_food} -> {actual_food}"


def log_meal(food, calories, portion_grams):
    log_file = "daily_log.csv"
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "food": food,
        "portion_grams": portion_grams,
        "calories": round(calories),
    }
    df = pd.read_csv(log_file) if os.path.exists(log_file) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(log_file, index=False)


def render_metric_card(emoji, value, label):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-emoji">{emoji}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def go_to(page):
    st.session_state["page"] = page
    st.rerun()


def render_brand():
    st.markdown(
        """
        <div class="brand-row">
            <span class="brand-mark">🍲</span>
            <span>Food<span class="green">Sight</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================== PAGES ====================
def show_home():
    render_brand()
    
    st.markdown("""
    <style>
    [data-testid="stHorizontalBlock"] > div:nth-child(2) > div[data-testid="stVerticalBlock"] {
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center; /* Center vertically */
        gap: 1.5rem; /* Spacing between buttons */
    }
    </style>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="hero-panel">
                <div class="hero-inner">
                    <div class="hero-kicker">● &nbsp; YOUR SMART FOOD COMPANION</div>
                    <div>
                        <div class="hero-title">
                            Know what’s<br>on your <span class="green">plate.</span>
                        </div>
                        <div class="hero-copy">
                            Upload a food photo, identify your meal, and estimate
                            its calories in just a few steps.
                        </div>
                    </div>
                    <!-- Footer text removed as requested -->
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.container():
            if st.button(
                "📸  Calorie estimation  →",
                key="home_predict",
                use_container_width=True,
            ):
                go_to("predict")

            if st.button(
                "📊  Meal history  →",
                key="home_history",
                use_container_width=True,
            ):
                go_to("history")

    st.markdown(
        '<div class="footer-note">POWERED BY EFFICIENTNET-B3 & USDA FOOD DATA</div>',
        unsafe_allow_html=True,
    )


def show_predict():
    render_brand()
    top_left, top_right = st.columns([1, 5])
    with top_left:
        if st.button("← Home", use_container_width=True):
            go_to("home")

    st.markdown('<div class="eyebrow">FOOD ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Predict your meal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subheading">Upload a photo, check the prediction, and calculate an estimated calorie count.</div>',
        unsafe_allow_html=True,
    )

    upload_col, detail_col = st.columns([1, 1], gap="large")
    with upload_col:
        st.markdown('<div class="small-label">STEP 01</div>', unsafe_allow_html=True)
        st.markdown("### Upload a food photo")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="food_upload",
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your uploaded meal", use_container_width=True)
            if st.button("Identify food  →", type="primary", use_container_width=True):
                with st.spinner("Analyzing your meal..."):
                    model = load_model()
                    class_names = load_classes()
                    top_classes, top_confidences = predict_food(image, model, class_names)
                st.session_state["pred"] = top_classes[0]
                st.session_state["conf"] = top_confidences[0]
                st.session_state["top_3"] = top_classes
                st.session_state["predicted_image"] = image
                st.rerun()

    with detail_col:
        st.markdown('<div class="small-label">STEP 02</div>', unsafe_allow_html=True)
        st.markdown("### Prediction")
        if "pred" in st.session_state:
            predicted = st.session_state["pred"].replace("_", " ").title()
            st.markdown(f"## <span class='green'>{predicted}</span>", unsafe_allow_html=True)
            st.caption(f"Model confidence: {st.session_state['conf']:.1%}")
            with st.expander("See other possible predictions"):
                for food in st.session_state.get("top_3", [])[1:]:
                    st.write("• " + food.replace("_", " ").title())
        else:
            st.markdown(
                '<p class="feature-copy">Your food prediction will appear here after you upload an image and run the model.</p>',
                unsafe_allow_html=True,
            )

    if "pred" in st.session_state and uploaded_file:
        st.markdown('<div class="small-label">STEP 03</div>', unsafe_allow_html=True)
        st.markdown("### Confirm details and calculate calories")

        with st.form("calculate_form"):
            is_correct = st.radio(
                "Is the prediction correct?",
                ["Yes (Correct)", "No (I'll correct it)"],
                horizontal=True,
            )
            correction = st.text_input(
                "Correct food name (if wrong)",
                placeholder="e.g. pizza, steak, sushi",
            )
            portion = st.selectbox(
                "Portion size",
                [
                    "Small (150g)",
                    "Medium (250g)",
                    "Large (400g)",
                    "Extra Large (600g)",
                    "Other (custom)",
                ],
                index=1,
            )
            custom_grams = None
            if portion == "Other (custom)":
                custom_grams = st.number_input(
                    "Enter portion in grams",
                    min_value=1,
                    max_value=5000,
                    value=250,
                    step=10,
                )
            submitted = st.form_submit_button(
                "Calculate calories  →",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            class_names = load_classes()
            if is_correct == "No (I'll correct it)":
                if not correction.strip():
                    st.error("Please enter the correct food name.")
                    st.stop()
                actual_food = correction.strip().lower().replace(" ", "_")
                if actual_food not in class_names:
                    st.error(
                        f"'{correction}' is not one of the supported food classes. "
                        "The image will still be saved for future training."
                    )
                    save_feedback(
                        st.session_state["predicted_image"],
                        st.session_state["pred"],
                        actual_food,
                    )
                    st.stop()
                save_feedback(
                    st.session_state["predicted_image"],
                    st.session_state["pred"],
                    actual_food,
                )
                st.info("Your correction has been saved for retraining.")
            else:
                actual_food = st.session_state["pred"]

            df_calories = load_calories_csv()
            calories_per_100g = get_calories(actual_food, df_calories)
            if calories_per_100g is None:
                st.error(f"No calorie data found for '{actual_food}' in the USDA dataset.")
            else:
                grams = get_portion_weight(portion, custom_grams)
                total = (grams / 100) * calories_per_100g
                log_meal(actual_food, total, grams)

                st.markdown('<div class="small-label">YOUR RESULT</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-heading">Meal summary</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    render_metric_card("🍽️", actual_food.replace("_", " ").title(), "Food")
                with col2:
                    render_metric_card("⚖️", f"{grams} g", "Portion")
                with col3:
                    render_metric_card("🔥", f"{round(total)} kcal", "Estimated calories")


def show_history():
    render_brand()
    top_left, _ = st.columns([1, 5])
    with top_left:
        if st.button("← Home", use_container_width=True):
            go_to("home")

    st.markdown('<div class="eyebrow">YOUR TRACKER</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Meal history</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subheading">A record of the meals you have analyzed and logged.</div>',
        unsafe_allow_html=True,
    )

    if os.path.exists("daily_log.csv"):
        df = pd.read_csv("daily_log.csv")
        if not df.empty:
            total_calories = int(df["calories"].sum())
            total_meals = len(df)
            metric1, metric2 = st.columns(2)
            with metric1:
                render_metric_card("🔥", f"{total_calories:,} kcal", "Total logged calories")
            with metric2:
                render_metric_card("🍽️", total_meals, "Meals logged")

            st.markdown("### Your entries")
            st.dataframe(df.sort_values(["date", "time"], ascending=False), use_container_width=True)
            st.download_button(
                "Download history (CSV)  ↓",
                df.to_csv(index=False),
                "calorie_history.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.info("No meals logged yet.")
    else:
        st.info("No meals logged yet. Your analyzed meals will appear here.")


# ==================== APP ROUTER ====================
def main():
    load_css()
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    if st.session_state["page"] == "home":
        show_home()
    elif st.session_state["page"] == "predict":
        show_predict()
    elif st.session_state["page"] == "history":
        show_history()


if __name__ == "__main__":
    main()