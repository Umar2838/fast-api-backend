import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("Model.keras")
# Load the scaler
with open("scalar.pkl", "rb") as f:
    scaler = pickle.load(f)


# Ye User upload karega
df = pd.read_csv("Test_data.csv") 

# Separate time and features
time = df.iloc[:, 0]  # First column as time
features = df.iloc[:, 1:]  # Remaining columns as features

# Scale the features
scaled_features = scaler.transform(features)

# Make predictions
predictions = model.predict(scaled_features)

import matplotlib.pyplot as plt
if(True): #User tick kare agar to true
    actual_values = pd.read_csv("Test_Results.csv")
    actual_values = actual_values.iloc[:,:]
    pred_values = predictions.flatten()
    actual_values = actual_values.to_numpy().flatten()
    
    # Plot both actual and predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(time, actual_values, label='Actual Values', color='red', linewidth=2)
    plt.plot(time, pred_values, label='Predicted Values', color='blue', linewidth=2)
    
    # Graph details
    plt.title("Comparison of Actual vs Predicted Loss Over Time")
    plt.xlabel("Time (hours)")
    plt.ylabel("Loss (MW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    
    
    # Convert predictions to 1D array if it's not already
    pred_values = predictions.flatten()
    
    # Plot predictions vs time
    plt.figure(figsize=(10, 5))
    plt.plot(time, pred_values, linestyle='-', marker='', color='blue', label='ANN Predictions')
    
    plt.xlabel("Time (hours)")
    plt.ylabel("Predicted Loss (MW)")
    plt.title("ANN Loss Predictions over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

import pandas as pd

# Step 1: Compute row-wise sum across all 11 columns
total_loss = features.sum(axis=1)

# Step 2: Flatten predictions to 1D array if not already
pred_loss = predictions.flatten()

# Step 3: Create new DataFrame
loss_df = pd.DataFrame({
    'Time (hours)' : time,
    'Total Load': total_loss,
    'Predicted Loss': pred_loss
})

# Optional: Display first few rows
loss_df.to_csv("Results.csv") #GIve user option to download this