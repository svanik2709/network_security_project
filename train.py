import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.autoencoder import build_autoencoder
from models.privacy_risk_bilstm import build_privacy_risk_bilstm

# Example Data
network_data = np.random.rand(1000, 20)
user_behavior_data = np.random.rand(1000, 10, 5)
privacy_labels = np.random.randint(0, 3, 1000)

# Split data
net_train, net_test = train_test_split(network_data, test_size=0.2, random_state=42)
user_train, user_test, label_train, label_test = train_test_split(user_behavior_data, privacy_labels, test_size=0.2, random_state=42)

# Build Models
autoencoder = build_autoencoder(input_dim=20)
privacy_model = build_privacy_risk_bilstm(input_shape=(10, 5), num_classes=3)

# Train Autoencoder
print("Training Autoencoder...")
autoencoder.fit(net_train, net_train, epochs=20, batch_size=32, validation_split=0.2)

# Train Privacy Risk BiLSTM
print("Training Privacy Risk BiLSTM...")
privacy_model.fit(user_train, label_train, epochs=20, batch_size=32, validation_split=0.2)


# Save models
autoencoder.save('autoencoder_model.keras')
privacy_model.save('privacy_risk_model.keras')
