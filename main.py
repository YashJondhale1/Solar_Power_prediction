import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sidebar Navigation
st.sidebar.title("🔆 Solar Power Generation Prediction")
st.sidebar.markdown("A smart ML-based system for predicting solar power generation.")
app_mode = st.sidebar.selectbox("Navigate", ["Home", "Prediction", "About"])

# Load dataset
data = pd.read_csv("C:\\Users\\hp\\OneDrive\\Desktop\\jy\\Solar_power\\solarpowergeneration.csv")
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def predict_solar_power(input_values):
    input_df = pd.DataFrame([input_values])
    return model.predict(input_df)[0]

if app_mode == "Home":
    # import Image from pillow to open images
    from PIL import Image
    img = Image.open("img.png")

    # display image using streamlit
    # width is used to set the width of an image
    st.image(img)
    st.markdown(
        "<p style='text-align: center; font-size: 22px;'>"
        "🌞 Welcome to the Solar Power Generation Predictor! "
        "This system utilizes machine learning to estimate solar power generation based on various environmental factors.🌞"
        "</p>",
        unsafe_allow_html=True,
    )
    st.write("### Why Use This Predictor?")
    st.markdown("- Uses historical data for accurate forecasting.\n- Helps optimize solar energy usage.\n- User-friendly interface for easy predictions.")

elif app_mode == "Prediction":
    st.header("🔢 Predict Solar Power Generation")
    st.write("Enter the required values to get a solar power prediction.")
    
    input_values = {}
    for col in X.columns:
        input_values[col] = st.number_input(f"Enter value for {col}", value=0.0)
    
    if st.button("Predict"):
        st.success(f"Predicted Solar Power Generation: {predict_solar_power(input_values):.2f} kW")

    # Model Performance
    st.subheader("📊 Model Performance")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"Variance Score (R²): {r2_score(y_test, y_pred):.2f}")
    st.write(f"Model Accuracy: {r2_score(y_test, y_pred) * 100:.2f}%")

elif app_mode == "About":
    st.title("ℹ️ About")
    # import Image from pillow to open images
    from PIL import Image
    img = Image.open("img2.png")

    # display image using streamlit
    # width is used to set the width of an image
    st.image(img)
    st.markdown(
        "This project is an AI-powered tool for predicting solar power generation based on environmental parameters. "
        "It utilizes a machine learning model trained on historical solar data to provide real-time predictions."
    )
    st.write("### Features:")
    st.markdown("- Accurate solar power prediction.\n- Supports multiple environmental factors.\n- Helps in energy optimization and planning.")

# Sidebar Feedback Section
st.sidebar.markdown("---")
feedback = st.sidebar.text_area("💬 Feedback")
if st.sidebar.button("Submit Feedback"):
    if feedback:
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.warning("Please provide feedback before submitting.")
