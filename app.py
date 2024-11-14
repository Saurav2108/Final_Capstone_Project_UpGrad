import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# Load models and scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('best_fraud_Prediction_model.pkl')

# Inject custom CSS to style the sidebar and the main content
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fbc2eb, #a6c0fe, #c2e9fb, #e0c3fc); /* Gradient from Pink to Violet */
        color: #333;
    }
    .stApp {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fbc2eb, #a6c0fe, #c2e9fb, #e0c3fc); /* Gradient for the app */
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #00ff7f, #00d2ff, #7f00ff, #ff007f); /* Gradient for sidebar block: Green, Blue, Violet, Pink */
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .info-block {
        font-size: 18px;
        font-weight: bold;
        color: #fff;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        background: linear-gradient(135deg, #00ff7f, #00d2ff, #7f00ff, #ff007f); /* Gradient from Green, Blue, Violet, Pink */
    }
    .result-text {
        font-size: 24px; /* Larger font size for results */
        font-weight: bold;
        color: #333;
        text-align: center;
        margin-top: 20px;
    }
    .result-probability {
        font-size: 24px; /* Larger font size for probability */
        font-weight: bold;
        color: #007bff; /* Blue color for probability */
        text-align: center;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with custom block
with st.sidebar:
    st.markdown(
        """
        <div class="info-block">
            Pranav Lokhande<br>
            Batch Data Science Boot Camp
        </div>
        """,
        unsafe_allow_html=True
    )

# Main content
st.title('Credit Card Fraud Prediction')

# Create a text area for user to paste all feature values
st.header('Paste Transaction Details')
input_text = st.text_area('Enter feature values separated by commas:', '')

# Button to process the input
if st.button('Predict'):
    if input_text.strip():
        try:
            # Parse the input text
            input_values = list(map(float, input_text.split(',')))
            
            # Check if the input length matches the expected number of features
            expected_length = 30  # Number of features in your model
            if len(input_values) != expected_length:
                st.error(f"Expected {expected_length} values, but got {len(input_values)}. Please check the input.")
            else:
                # Create DataFrame from the input
                input_df = pd.DataFrame([input_values], columns=[
                    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                    'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                    'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Time'
                ])
                
                # Rearrange columns to match the model's expected order: Time first, Amount last
                input_df = input_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 
                                     'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 
                                     'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
                
                # Apply scaling to 'Amount' and 'Time' features
                if isinstance(scaler, RobustScaler):
                    # Transform Amount and Time separately
                    input_df[['Amount', 'Time']] = scaler.transform(input_df[['Amount', 'Time']])
                else:
                    st.error("Unexpected scaler type. Please check your scaler.")
                
                # Make prediction
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)[:, 1]
                
                # Display results with larger font sizes
                st.markdown(f'<p class="result-text">Prediction: {"Fraud" if prediction[0] == 1 else "Not Fraud"}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="result-probability">Probability of Fraud: {prediction_proba[0]:.4f}</p>', unsafe_allow_html=True)
        except ValueError as ve:
            st.error(f"Value error: {ve}")
        except Exception as e:
            st.error(f"Error processing input: {e}")
    else:
        st.error("Please paste the feature values.")
