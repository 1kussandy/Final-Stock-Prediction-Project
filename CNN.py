import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def cnn_model(X_train, X_test, y_train, y_test):
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    cnn_model.add(MaxPooling1D(pool_size=1))  # Updated pool_size to 1

    cnn_model.add(Flatten())
    cnn_model.add(Dense(50, activation='relu'))
    cnn_model.add(Dense(1))

    cnn_model.compile(optimizer='adam', loss='mean_squared_error')
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    cnn_predictions = cnn_model.predict(X_test)
    
    cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_predictions))

    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.square(cnn_predictions - y_test)))

    # Calculate the range of the target variable (replace min and max with your actual values)
    min_target_value = np.min(y_test)
    max_target_value = np.max(y_test)
    target_range = max_target_value - min_target_value

    # Calculate accuracy in percentage
    accuracy_percentage = ((1 - (rmse / target_range)) * 100).round(2)

    print(f"CNN Model RMSE on Test Set: {cnn_rmse}")
    print(f"Accuracy Percentage: {accuracy_percentage}%")

    return cnn_predictions

# Test the CNN model with dummy data
if __name__ == "__main__":
    # Dummy data for testing purposes
    np.random.seed(42)
    X_train_dummy = np.random.rand(100, 3, 2)  # Example shape for X_train
    y_train_dummy = np.random.rand(100,)  # Example shape for y_train
    X_test_dummy = np.random.rand(20, 3, 2)  # Example shape for X_test
    y_test_dummy = np.random.rand(20,)  # Example shape for y_test

    # Call CNN model function with dummy data
    cnn_predictions_dummy = cnn_model(X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy)
