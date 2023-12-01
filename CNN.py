from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten

def cnn_model(x_train, x_test, y_train, y_test):
    # Ensure the input data is 3D
    # THIS WILL PRINT 1 
    # DATA IS ALREADY IN 1D
    print(len(x_train.shape))

    # Create a simple CNN model
    model = Sequential()
    # model.add(Conv2D)
    # input shape needs to be 3D
    #takes sequence length and
    # model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(x_train.shape[1], 1)))
    # model.add(Flatten())

    ## Error at this line
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    return model







# import numpy as np
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
# from tensorflow.keras.layers import Dropout


# def cnn_model(X_train, X_test, y_train, y_test):
#     cnn_model = Sequential()
#     cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
#     cnn_model.add(MaxPooling1D(pool_size=1))

#     cnn_model.add(Flatten())
#     cnn_model.add(Dense(50, activation='relu'))
#     cnn_model.add(Dense(1))

#     cnn_model.compile(optimizer='adam', loss='mean_squared_error')
#     cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

#     cnn_predictions = cnn_model.predict(X_test)

#     cnn_rmse = np.sqrt(mean_squared_error(y_test, cnn_predictions))
#     print(f"CNN Model RMSE on Test Set: {cnn_rmse}")

#     # Calculate accuracy in percentage
#     rmse = np.sqrt(np.mean(np.square(cnn_predictions - y_test)))
#     min_target_value = np.min(y_test)
#     max_target_value = np.max(y_test)
#     target_range = max_target_value - min_target_value
#     accuracy_percentage = ((1 - (rmse / target_range)) * 100).round(2)
#     print(f"Accuracy Percentage: {accuracy_percentage}%")

#     return cnn_predictions

# if __name__ == "__main__":
#     # Dummy data for testing purposes
#     np.random.seed(42)
#     X_train_dummy = np.random.rand(100, 3, 1)  # Example shape for X_train
#     y_train_dummy = np.random.rand(100,)  # Example shape for y_train
#     X_test_dummy = np.random.rand(20, 3, 1)  # Example shape for X_test
#     y_test_dummy = np.random.rand(20,)  # Example shape for y_test

#     # Call CNN model function with dummy data
#     cnn_predictions_dummy = cnn_model(X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy)