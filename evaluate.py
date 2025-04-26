from tensorflow.keras.models import load_model
import numpy as np

autoencoder = load_model('autoencoder_model.keras')
privacy_model = load_model('privacy_risk_model.keras')


# Generate some dummy test data
test_network_data = np.random.rand(5, 20)
test_user_data = np.random.rand(5, 10, 5)

# Predict using Autoencoder
reconstructed = autoencoder.predict(test_network_data)
print("\nAutoencoder Reconstruction:")
print(reconstructed)

# Predict using Privacy Risk BiLSTM
privacy_predictions = privacy_model.predict(test_user_data)
print("\nPrivacy Risk Predictions:")
print(privacy_predictions)
