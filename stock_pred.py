import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the scaler and model
scaler = joblib.load('mms.h5')
model = load_model('stock_model.h5')

# Page configuration
st.set_page_config(page_title="📊 Stock Price Predictor", page_icon="💹", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        /* Background color for the whole app */
        body {
            background-color: #f0f2f6;
        }

        /* Critical style for the main header */
        .critical-header {
            color: #e63946;
            text-align: center;
            font-size: 3em;
            font-family: 'Arial Black', sans-serif;
        }

        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #264653;
            color: white;
            font-size: 1.1em;
        }

        /* Slider styling */
        .stSlider > div {
            color: #1d3557;
        }

        /* Styling for success messages */
        .stAlert {
            background-color: #388E3C;
            border: 2px solid #388E3C;
            border-radius: 10px;
        }

        /* Button customization */
        div.stButton > button {
            color: white;
            background-color: #388E3C;
            border: none;
            font-size: 1.2em;
            padding: 10px 20px;
            border-radius: 5px;
        }

        /* Footer style */
        .footer {
            text-align: center;
            font-size: 1.2em;
            padding: 10px;
            background-color: #3F51B5;
            color#FFFFFF;
            border-radius: 5px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# # Sidebar configuration
# # Sidebar Header and Feedback Section
# st.sidebar.markdown("""
# <div class='sidebar-content' style='padding: 15px; background-color: #f0f4f8; border-radius: 10px;'>
#     <h2 style='color: #2a9d8f;'>📊 Forecast Google Stock Price Trends</h2>
#     <hr style='border: 1px solid #264653;'>

#     <h3 style='color: #264653;'>💬 Feedback</h3>
#     <p style='font-size: 1.1em; color: #333;'>We value your opinion! Rate our app below:</p>
# </div>
# """, unsafe_allow_html=True)

# # Custom Star Rating System
# # Sidebar Header and Feedback Section
# st.sidebar.markdown("""
# <div class='sidebar-content' style='padding: 15px; background-color: #f0f4f8; border-radius: 10px;'>
#     <h2 style='color: #2a9d8f;'>📊 Forecast Google Stock Price Trends</h2>
#     <hr style='border: 1px solid #264653;'>

#     <h3 style='color: #264653;'>💬 Feedback</h3>
#     <p style='font-size: 1.1em; color: #333;'>We value your opinion! Please leave your feedback below:</p>
# </div>
# """, unsafe_allow_html=True)

# Add Text Input for Feedback
user_feedback = st.sidebar.text_area(
    "Write your feedback here:", 
    placeholder="Type your thoughts about this app...",
    max_chars=300
)

# Feedback Submission Button
if st.sidebar.button("Submit Feedback"):
    st.sidebar.markdown(f"""
    <div style='margin-top: 20px; background-color: #e9ecef; padding: 10px; border-radius: 5px; text-align: center;'>
        <h4 style='color: #2a9d8f;'>Thank You! 🙏</h4>
        <p style='font-size: 1.1em; color: #333;'>You shared:</p>
        <p style='font-size: 1em; color: #264653;'>{user_feedback}</p>
        <p style='font-size: 0.9em; color: #333;'>Your feedback helps us improve!</p>
    </div>
    """, unsafe_allow_html=True)


rating = st.sidebar.radio(
    "",
    [
        "⭐ 1 - Poor",
        "⭐⭐ 2 - Fair",
        "⭐⭐⭐ 3 - Good",
        "⭐⭐⭐⭐ 4 - Very Good",
        "⭐⭐⭐⭐⭐ 5 - Excellent"
    ],
    index=2
)

# Feedback Confirmation
st.sidebar.markdown(f"""
<div style='margin-top: 20px; background-color: #e9ecef; padding: 10px; border-radius: 5px; text-align: center;'>
    <h4 style='color: #2a9d8f;'>Thank You! 🙏</h4>
    <p style='font-size: 1.1em; color: #333;'>You rated us:</p>
    <p style='font-size: 1.5em; color: #ffba08;'>{rating.split(" - ")[0]}</p>
    <p style='font-size: 0.9em; color: #333;'>Your feedback helps us improve!</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<p>Welcome to the Stock Price Predictor. Adjust the sliders to input recent stock data and see the prediction for future prices!</p></div>", unsafe_allow_html=True)

# Main header with critical style
st.markdown("<div class='critical-header'>🔮Google Stock Price Forecasting Tool</div>", unsafe_allow_html=True)

import streamlit as st

# Sidebar Introduction section
st.sidebar.markdown("""
    <div style='background-color: #8A2BE2; color: white; padding: 15px; border-radius: 10px; margin-top: 10px;'>
        <h2 style='text-align: center;'>🚀 How This App Works:</h2>
        <ul style='font-size: 1.1em;'>
            <li>📈 Use the sliders to input recent stock prices for the past 60 days.</li>
            <li>🔍 Click <b>Predict</b> to calculate the stock price for Day 61.</li>
            <li>📊 View the prediction below!</li>
        </ul>
    </div>
""", unsafe_allow_html=True)


# User input for stock prices
st.header("📉 Set Recent Price Values")

# Use sliders to gather data from users
columns = st.columns(10)  # Use 10 columns for a compact design
prices = []

# Distribute sliders in columns
for i in range(60):
    with columns[i % 10]:  # Use modulo to cycle through columns
        price = st.slider(f"Day {i+1}", min_value=500.0, max_value=3000.0, value=1500.0, step=0.1, key=f"slider_{i+1}")
        prices.append(price)

# Prediction section
if st.button("🔮 Predict Stock Price"):
    # Convert input prices to a numpy array
    inputs = np.array(prices).reshape(-1, 1)
    inputs_scaled = scaler.transform(inputs)

    # Prepare the test data
    x_test = []
    x_test.append(inputs_scaled[:, 0])  # Use the user-provided 60 data points
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions_scaled = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions_scaled)

    # Display the prediction result
    st.markdown(f"""
        <div style='background-color: #3F51B5; color: white; padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <h3 style='text-align: center;'>📊 Predicted Stock Price</h3>
            <p style='text-align: center; font-size: 1.5em;'>📈 Predicted Price for Day 61: <b>${predictions.flatten()[0]:.2f}</b></p>
        </div>
    """, unsafe_allow_html=True)
    st.balloons()

    # Note with explanation
    st.markdown("""
        <div style='background-color: #3F51B5; color: white; padding: 15px; border-radius: 10px; margin-top: 20px;'>
            <h4>⚠️ Note:</h4>
            <p>
                This model is not trained for precise predictions. Due to computational limitations, training was limited.
                Treat the results as approximations. For accurate forecasts, use more advanced models with larger datasets.
                Thank you for understanding!
            </p>
        </div>
    """, unsafe_allow_html=True)

# Footer section
st.markdown("""
    <div class='footer'>
        © 2024 - Built with passion by Mohammed Naser Uddin.  
    </div>
""", unsafe_allow_html=True)
