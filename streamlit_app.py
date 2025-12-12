import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model and scaler
# -------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

best_model, scaler = load_model_and_scaler()

# -------------------------------
# Features (ordered)
# -------------------------------
features = [
    "voice_mail_messages",
    "customer_service_calls",
    "international_plan",
    "international_charge",
    "international_calls",
    "day_mins",
    "evening_mins",
    "night_mins",
    "total_charge"
]

# Display names without underscores
feature_labels = {
    "voice_mail_messages": "Voice Mail Messages",
    "customer_service_calls": "Customer Service Calls",
    "international_plan": "International Plan",
    "international_charge": "International Charge",
    "international_calls": "International Calls",
    "day_mins": "Day Minutes",
    "evening_mins": "Evening Minutes",
    "night_mins": "Night Minutes",
    "total_charge": "Total Charge"
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📞", layout="centered")

# -------------------------------
# Custom CSS with background image fix
# -------------------------------
st.markdown("""
    <style>
        /* Background Image */
        .stApp {
            background-image: url("https://logmanager.com/wp-content/uploads/2024/10/telco-img-768x439.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* Add translucent overlay to make text readable */
        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: transparent;
            z-index: 0;
        }

        /* Ensure content stays above overlay */
        .block-container {
            position: relative;
            z-index: 1;
        }

        /* Button styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.05);
        }

        /* Inputs */
        .stTextInput, .stNumberInput, .stSelectbox {
            background-color: #f9f9f9 !important;
            border-radius: 10px;
            padding: 8px;
        }

        /* Prediction box */
        .prediction-box {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
        .churn {
            background-color: #ffcccc;
            color: #b30000;
        }
        .not-churn {
            background-color: #ccffcc;
            color: #006600;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("📞 Telecom Customer Churn Prediction")
st.write("Enter customer details below:")

# -------------------------------
# Collect inputs (ordered & clean labels)
# -------------------------------
user_inputs = {}
for feature in features:
    label = feature_labels[feature]
    if feature == "international_plan":
        user_inputs[feature] = st.selectbox(label, ["No", "Yes"])
    else:
        user_inputs[feature] = st.number_input(label, value=0.0)

# Convert categorical to numeric
user_inputs["international_plan"] = 1 if user_inputs["international_plan"] == "Yes" else 0

# Create dataframe
user_df = pd.DataFrame([user_inputs])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Churn"):
    try:
        X_scaled = scaler.transform(user_df)
        probs = best_model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)

        if preds[0] == 1:
            st.markdown(
                f"<div class='prediction-box churn'>🚨 Prediction: Churn<br>Probability: {round(probs[0],2)}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='prediction-box not-churn'>✅ Prediction: Not Churn<br>Probability: {round(probs[0],2)}</div>",
                unsafe_allow_html=True,
            )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")