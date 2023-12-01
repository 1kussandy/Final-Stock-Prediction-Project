import numpy as np

def extract_features(data):

    features = []
    
    for index, row in data.iterrows():
        open_value = row['open']
        high_value = row['high']
        low_value = row['low']
        volume_value = row['volume']

        # Convert empty strings to NaN and then fill NaN with 0
        open_value = float(open_value) if open_value != '' else 0
        high_value = float(high_value) if high_value != '' else 0
        low_value = float(low_value) if low_value != '' else 0
        volume_value = float(volume_value) if volume_value != '' else 0

        feature = [
            open_value,
            data.loc[int(index) - 1, 'open'] if index != 0 else 0,
            data.loc[int(index) - 2, 'open'] if index != 1 and index != 0 else 0,
            data.loc[int(index)+1,'open'] if index!= data.shape[0] - 1 else 0,
            data.loc[int(index) + 2, 'open'] if index != data.shape[0] - 2 and index != data.shape[0] - 1 else 0,

            high_value,
            data.loc[int(index) - 1, 'high'] if index != 0 else 0,
            data.loc[int(index) - 2, 'high'] if index != 1 and index != 0 else 0,
            data.loc[int(index)+1,'high'] if index!=data.shape[0] - 1 else 0,
            data.loc[int(index) + 2, 'high'] if index != data.shape[0] - 2 and index != data.shape[0] - 1 else 0,

            low_value,
            data.loc[int(index) - 1, 'low'] if index != 0 else 0,
            data.loc[int(index) - 2, 'low'] if index != 1 and index != 0 else 0,
            data.loc[int(index)+1,'low'] if index!=data.shape[0] - 1 else 0,
            data.loc[int(index) + 2, 'low'] if index != data.shape[0] - 2 and index != data.shape[0] - 1 else 0,

            volume_value,
            data.loc[int(index) - 1, 'volume'] if index != 0 else 0,
            data.loc[int(index) - 2, 'volume'] if index != 1 and index != 0 else 0,
            data.loc[int(index)+1,'volume'] if index!=data.shape[0] - 1 else 0,
            data.loc[int(index) + 2, 'volume'] if index != data.shape[0] - 2 and index != data.shape[0] - 1 else 0,
        ]
        #Add Feature
        features.append(feature)
    print(f'features {np.array(features)}')
    return np.array(features)



