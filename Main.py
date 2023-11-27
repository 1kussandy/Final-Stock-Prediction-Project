import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Task & Data Selection - Obtaining historical stock price data using yfinance
ticker_symbol = 'AAPL'  # Apple Inc. ticker symbol
start_date = '2010-01-01'  # Start date for historical data retrieval
end_date = '2022-01-01'  # End date for historical data retrieval

# Fetch historical stock price data using yfinance
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Step 2: Preprocessing
# Data Cleaning: Handling missing values
stock_data.dropna(inplace=True)

# Feature Engineering: Adding a 50-day Moving Average as a new feature
stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()

# Normalization/Scaling: Scale numerical features (Close and 50_MA) for model training
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data[['Close', '50_MA']])
scaled_df = pd.DataFrame(scaled_data, columns=['Close_Scaled', '50_MA_Scaled'])

# Concatenating scaled features with the original dataset
stock_data = pd.concat([stock_data, scaled_df], axis=1)

# Train-Test Split: Divide data into training and testing sets
# Considering 'Close' as the target variable for prediction
X = stock_data[['Close_Scaled', '50_MA_Scaled']]  # Features
y = stock_data['Close']  # Target variable

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Displaying the preprocessed data and the split sets
print("Preprocessed Stock Data:")
print(stock_data.head())

print("\nTraining and Testing Set Sizes:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
