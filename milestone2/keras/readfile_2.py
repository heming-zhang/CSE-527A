import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def read_train_file(train_path):
    train_df = pd.read_csv(train_path)
    rows, columns = train_df.shape
    # MAP WILDERNESS TO CLASSIFICATION TYPE
    wilderness_area_list = sorted(list(set(train_df['Wilderness_Area'])))
    wilderness_area_dict = {k: v for v, k in enumerate(wilderness_area_list)}
    train_df['Wilderness_Area'] = train_df['Wilderness_Area'].map(wilderness_area_dict)
    wilderness_list = train_df['Wilderness_Area']
    # MAP COVER_TYPE TO CLASSIFICATION TYPE
    cover_type_list = sorted(list(set(train_df['Cover_Type'])))
    cover_type_dict = {k: v for v, k in enumerate(cover_type_list)}
    train_df['Cover_Type'] = train_df['Cover_Type'].map(cover_type_dict)
    cover_list = train_df['Cover_Type']
    # CLASSIFY SOIL_TYPE
    soil_type_list = list(train_df['Soil_Type'])
    climate_type_list = []
    geo_type_list = []
    for soil_type in soil_type_list:
        climate_type = int(soil_type / 1000)
        climate_type_list.append(climate_type)
        geo_type = int((soil_type % 1000) / 100)
        geo_type_list.append(geo_type)
    yTr = np.array(train_df['Cover_Type']).reshape(rows, 1)
    # NORMALIZE TRAINNING DATA SET
    train_df = train_df.iloc[:, 1 : -3]
    train_df_mean = train_df.mean()
    train_df_std = train_df.std()
    normalized_df = normalize(train_df, train_df_mean, train_df_std)
    # COMBINE TRAINING DATA SET INTO NUMPY ARRAY
    class_df = pd.DataFrame({
        'Wilderness_Area': wilderness_list,
        'Climater_Type': climate_type_list,
        'Geo_Type': geo_type_list
    })
    train_df = pd.concat([normalized_df, class_df], axis = 1)
    xTr = np.array(train_df)
    np.save('./xTr.npy', xTr)
    np.save('./yTr.npy', yTr)
    print(xTr.shape)
    print(yTr.shape)
    return train_df_mean, train_df_std


def read_test_file(test_path, train_df_mean, train_df_std):
    test_df = pd.read_csv(test_path)
    rows, columns = test_df.shape
    # MAP WILDERNESS TO CLASSIFICATION TYPE
    wilderness_area_list = sorted(list(set(test_df['Wilderness_Area'])))
    wilderness_area_dict = {k: v for v, k in enumerate(wilderness_area_list)}
    test_df['Wilderness_Area'] = test_df['Wilderness_Area'].map(wilderness_area_dict)
    wilderness_list = test_df['Wilderness_Area']
    # CLASSIFY SOIL_TYPE
    soil_type_list = list(test_df['Soil_Type'])
    climate_type_list = []
    geo_type_list = []
    for soil_type in soil_type_list:
        climate_type = int(soil_type / 1000)
        climate_type_list.append(climate_type)
        geo_type = int((soil_type % 1000) / 100)
        geo_type_list.append(geo_type)
    # NORMALIZE TEST DATA SET
    test_df = test_df.iloc[:, 1 : -2]
    normalized_df = normalize(test_df, train_df_mean, train_df_std)
    # COMBINE TRAINING DATA SET INTO NUMPY ARRAY
    class_df = pd.DataFrame({
        'Wilderness_Area': wilderness_list,
        'Climater_Type': climate_type_list,
        'Geo_Type': geo_type_list
    })
    test_df = pd.concat([normalized_df, class_df], axis = 1)
    print(test_df)
    xTe = np.array(test_df)
    np.save('./xTe.npy', xTe)
    print(xTe.shape)


def num_encode(column):
    # binary encode
    # enc = OneHotEncoder(sparse=False)
    enc = OneHotEncoder()
    column = column.reshape(-1, 1)
    enc.fit(column)
    encode_col = enc.transform(column).toarray()
    return encode_col


def normalize(df, df_mean, df_std):
    normalized_df = (df - df_mean) / df_std
    return normalized_df


if __name__ == "__main__":
    train_path = './train.csv'
    train_df_mean, train_df_std = read_train_file(train_path)
    test_path = './test.csv'
    read_test_file(test_path, train_df_mean, train_df_std)