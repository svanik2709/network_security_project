import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained models
autoencoder = load_model('autoencoder_model.keras')
privacy_model = load_model('privacy_risk_model.keras')

# Streamlit App
st.title("Network Security Project Demo 🔒")
st.write("Autoencoder Reconstruction & Privacy Risk Prediction")

# Input fields
st.write("### Generate Random Test Data:")
if st.button("Generate and Predict"):
    # Generate random data
    test_network_data = np.random.rand(5, 20)
    test_user_data = np.random.rand(5, 10, 5)

    st.write("##### Test Network Data:")
    st.dataframe(test_network_data)

    st.write("##### Test User Behavior Data (reshaped for viewing):")
    st.dataframe(test_user_data.reshape(5, -1))  # <-- IMPORTANT: Reshape 3D to 2D

    # Make predictions
    reconstructed = autoencoder.predict(test_network_data)
    privacy_predictions = privacy_model.predict(test_user_data)

    # Display results
    st.write("##### Autoencoder Reconstruction:")
    st.dataframe(reconstructed)

    st.write("##### Privacy Risk Predictions:")
    st.dataframe(privacy_predictions)
