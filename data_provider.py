import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_data(path):
    def is_hit(value):
        if value > 77:
            return 1
        return 0

    data = pd.read_csv(path)
    garbage_columns = [
        'playlist_id',
        'track_album_id',
        'track_id',
        'track_popularity',
        'track_name',
        'track_album_name',
        'track_album_release_date',
    ]
    popularity_percent = data['track_popularity']
    data = data.drop(garbage_columns, axis=1)
    data['is_hit'] = popularity_percent.map(is_hit)

    return data


def split_data(data: pd.DataFrame):
    row_count = len(data)
    train_count = int((row_count / 3) * 2)
    test_count = int(row_count - train_count)

    train = data.head(train_count)
    test = data.tail(test_count)
    return train, test


def fix_nan_data(data: pd.DataFrame):
    data.fillna(data.median(axis=0), inplace=True)


def encode_label(column_name: str, train: pd.DataFrame, test: pd.DataFrame):
    label_encoder = LabelEncoder()
    label_encoder.fit(train[column_name].to_list() + test[column_name].to_list())
    train[column_name] = label_encoder.transform(train[column_name])
    test[column_name] = label_encoder.transform(test[column_name])


def encode_labels(target_columns: list[str], train: pd.DataFrame, test: pd.DataFrame):
    for column in target_columns:
        encode_label(column, train, test)


def get_columns_with_string_value_type(data: pd.DataFrame):
    return [c for c in data.columns if data[c].dtype == 'O']


def scale_data(target_columns, data: pd.DataFrame):
    for column in target_columns:
        data[column] = StandardScaler().fit_transform(data[column].values.reshape(-1, 1))


def prepare_data(target_column_name, train: pd.DataFrame, test: pd.DataFrame):
    columns_to_encode_repeats = get_columns_with_string_value_type(train) + get_columns_with_string_value_type(test)
    columns_to_encode = list(set(columns_to_encode_repeats))

    fix_nan_data(train)
    fix_nan_data(test)

    encode_labels(columns_to_encode, train, test)

    scale_data([c for c in train.columns if c != target_column_name], train)
    scale_data(test.columns, test)


def split_data_to_target_and_other(target_name, data: pd.DataFrame):
    y = data[target_name]
    data.drop(target_name, axis=1, inplace=True)
    return y, data
