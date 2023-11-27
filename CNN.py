import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.metrics import mean_squared_error

# Assuming X_train, X_test, y_train, y_test are obtained from previous steps

# Reshaping the data for CNN (Adding a channel dimension)
X_train_cnn = np.expand_dims(X_train.values, axis=-1)
X_test_cnn = np.expand_dims(X_test.values, axis=-1)

# CNN Model Architecture
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)  # Output layer (for regression)
])

# Compile the CNN model
cnn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the CNN model
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the CNN model on the test dataset
cnn_predictions = cnn_model.predict(X_test_cnn)
cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_predictions))
print(f"CNN Model RMSE on Test Set: {cnn_rmse}")
