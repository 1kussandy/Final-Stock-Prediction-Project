import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data_path):
    # Load the stocks dataset
    stocks_data = pd.read_csv(data_path)  # Replace 'stocks.csv' with your file path

    # Perform preprocessing on the stocks dataset
    # Assuming 'symbol', 'date', 'open', 'high', 'low', 'close', and 'volume' columns are present
    # Extracting 'symbol' and 'date' as non-numeric features, and other columns as numeric features
    X = stocks_data[['open', 'high', 'low', 'volume']]  # Features (numeric columns)
    y = stocks_data['close']  # Target variable

    # Normalize/Scale numerical features for model training
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_path = 'stocks.csv'  # Replace with your dataset path
    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    print("Preprocessed Data Shapes:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Save preprocessed data if needed
    # ...

