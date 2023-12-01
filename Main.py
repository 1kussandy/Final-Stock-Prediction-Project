import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Features import extract_features
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from CNN import cnn_model
from sklearn.metrics import mean_absolute_error

def importData():
    training_file_path = './stock.csv'
    training_data = pd.read_csv(training_file_path, header=None, delimiter=',', names=["symbol","date","open","high","low","close","volume"])
    return training_data

if __name__ == "__main__":

    # 100 % correct
    training_data = importData()
    training_features = extract_features(training_data)

    print('Extracted Features.')

    [x_train, x_test, y_train, y_test] = train_test_split(training_features, training_data['close'], test_size=0.2, random_state=42)

    print(x_train)
    x_train = np.array(x_train)
    x_test = np.array(x_test)    
    print(x_train)
    print('Finished Splitting and Scaling Data.')

    ####--- CREATE MODLES HERE ---####

    # errors in this function
    cnn = cnn_model(x_train, x_test, y_train, y_test)
    y_pred = cnn.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error (MAE): {mae}')





