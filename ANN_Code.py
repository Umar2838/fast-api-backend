import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU
from sklearn.model_selection import train_test_split

df = pd.read_csv('Dataset.csv')
X = df.drop('System Loss', axis = 1)
y = df['System Loss']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 23)

from sklearn.preprocessing import StandardScaler
import pickle

scaler=StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
with open("scalar.pkl", "wb") as f:
    pickle.dump(scaler, f)
    
from keras.callbacks import ModelCheckpoint

# Define model
model = Sequential()
model.add(Dense(64, input_dim=12))
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(64))
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(32))
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.summary()

# Save the best model based on validation loss
checkpoint = ModelCheckpoint(
    filepath='Model.keras',   # ✅ updated extension
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.3,
    epochs=150,
    callbacks=[checkpoint]
)


 
# import matplotlib.pyplot as plt

# # Plot Training and Validation Loss
# # plt.figure(figsize=(12, 5))

# # Loss Plot
# plt.plot(history.history['loss'], label='Training Loss', color='blue')
# plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
# plt.title('Loss over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Squared Error (Loss)')
# plt.legend()
# plt.grid(True)

# # # Mean Absolute Error Plot
# # plt.subplot(1, 2, 2)
# # plt.plot(history.history['mae'], label='Training MAE', color='blue')
# # plt.plot(history.history['val_mae'], label='Validation MAE', color='orange')
# # plt.title('MAE over Epochs')
# # plt.xlabel('Epochs')
# # plt.ylabel('Mean Absolute Error')
# # plt.legend()
# # plt.grid(True)

# plt.tight_layout()
# plt.show()
   
# from matplotlib import pyplot as plt
# # plot the training and validation accuracy and loss at each epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'y', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
    
    
# acc = history.history['mae']
# val_acc = history.history['val_mae']
# plt.plot(epochs, acc, 'y', label='Training MAE')
# plt.plot(epochs, val_acc, 'r', label='Validation MAE')
# plt.title('Training and validation MAE')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
    
#Predict on test data
# predictions = model.predict(X_test_scaled[76:81])
# print("Predicted values are: ", predictions)
# print("Real values are: ", y_test[76:81])
    
# # #Comparison with other models..
# # #Neural network - from the current code
# mse_neural, mae_neural = model.evaluate(X_test_scaled, y_test)
# print('Mean squared error from neural net: ', mse_neural)
# print('Mean absolute error from neural net: ', mae_neural)

# from sklearn.metrics import r2_score

# # Actual and predicted values
# y_pred = model.predict(X_test_scaled)

# # R² score
# r2 = r2_score(y_test, y_pred)
# print("R² Score:", r2)

# Save the entire model to an HDF5 file
# model.save('regression_model.h5')
# print("Model saved as 'regression_model.h5'")
