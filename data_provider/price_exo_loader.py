import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


DEFAULT_KNOWN_EXO_FEATURES = [
    'forecast_光伏',
    'forecast_水电',
    'forecast_火电',
    'forecast_风电',
    'forecast_新能源',
    'forecast_总和',
    'forecast_负荷',
    'forecast_非市场机组',
    'temperature',
    'wind_speed_ten',
    'wind_speed_fifty',
    'wind_speed_eighty',
    'wind_speed_hundred',
    'hour_precipitation',
    'cloud_cover',
    'solar_radiation',
]

DERIVED_FEATURES = {
    'forecast_火电': [
        'forecast_总和',
        'forecast_新能源',
        'forecast_水电',
        'forecast_非市场机组',
    ],
}

DEFAULT_UNKNOWN_EXO_FEATURES = [
    'quantity_储能',
    'quantity_风电',
    'quantity_光伏',
    'quantity_火电',
    'quantity_水电',
    'quantity_总出清电量',
    'load',
]


def parse_feature_list(value, default_features):
    if value is None:
        return list(default_features)
    value = value.strip()
    if value == '':
        return list(default_features)
    if value.lower() in {'none', 'null', 'empty'}:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def time_features_from_datetime(datetimes):
    hour = datetimes.dt.hour.to_numpy(dtype=np.float32)
    minute = datetimes.dt.minute.to_numpy(dtype=np.float32)
    dayofweek = datetimes.dt.dayofweek.to_numpy(dtype=np.float32)

    hour_angle = 2 * np.pi * hour / 24.0
    minute_angle = 2 * np.pi * minute / 60.0
    dow_angle = 2 * np.pi * dayofweek / 7.0

    return np.stack([
        np.sin(hour_angle),
        np.cos(hour_angle),
        np.sin(minute_angle),
        np.cos(minute_angle),
        np.sin(dow_angle),
        np.cos(dow_angle),
    ], axis=1).astype(np.float32)


def add_derived_features(df):
    if all(col in df.columns for col in DERIVED_FEATURES['forecast_火电']):
        df['forecast_火电'] = (
            pd.to_numeric(df['forecast_总和'], errors='coerce')
            - pd.to_numeric(df['forecast_新能源'], errors='coerce')
            - pd.to_numeric(df['forecast_水电'], errors='coerce')
            - pd.to_numeric(df['forecast_非市场机组'], errors='coerce')
        )
    return df


class Dataset_PriceExo(Dataset):
    def __init__(self, args, flag='train'):
        assert flag in ['train', 'test']
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.target = getattr(args, 'target', 'price')
        self.expected_minutes = getattr(args, 'price_interval_minutes', 15)
        self.test_start_hour = getattr(args, 'test_start_hour', 0)
        self.test_start_minute = getattr(args, 'test_start_minute', 15)
        self.known_features = parse_feature_list(
            getattr(args, 'known_exo_features', None),
            DEFAULT_KNOWN_EXO_FEATURES,
        )
        self.unknown_features = parse_feature_list(
            getattr(args, 'unknown_exo_features', None),
            DEFAULT_UNKNOWN_EXO_FEATURES,
        )
        self.root_path = args.root_path
        self.data_path = args.data_path

        self.__read_data__()

    def __read_data__(self):
        csv_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(csv_path, encoding='utf-8-sig')
        df_raw = add_derived_features(df_raw)
        if 'datetime' not in df_raw.columns:
            raise ValueError('Dataset_PriceExo requires a datetime column')
        if self.target not in df_raw.columns:
            raise ValueError('target column {} not found'.format(self.target))

        required = [self.target] + self.known_features + self.unknown_features
        missing = [col for col in required if col not in df_raw.columns]
        if missing:
            raise ValueError('columns not found in {}: {}'.format(csv_path, ', '.join(missing)))

        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_raw = df_raw.sort_values('datetime').reset_index(drop=True)
        df_raw[required] = df_raw[required].apply(pd.to_numeric, errors='coerce')
        df_raw[required] = df_raw[required].ffill().bfill().fillna(0.0)

        n = len(df_raw)
        num_test = int(n * 0.2)
        num_train = n - num_test

        border1s = [0, n - num_test - self.seq_len]
        border2s = [num_train, n]
        type_map = {'train': 0, 'test': 1}
        set_type = type_map[self.flag]
        border1 = max(0, border1s[set_type])
        border2 = border2s[set_type]

        train_slice = slice(border1s[0], border2s[0])
        self.price_scaler = StandardScaler()
        self.price_scaler.fit(df_raw[[self.target]].iloc[train_slice].values)

        self.known_scaler = None
        if self.known_features:
            self.known_scaler = StandardScaler()
            self.known_scaler.fit(df_raw[self.known_features].iloc[train_slice].values)

        self.unknown_scaler = None
        if self.unknown_features:
            self.unknown_scaler = StandardScaler()
            self.unknown_scaler.fit(df_raw[self.unknown_features].iloc[train_slice].values)

        df = df_raw.iloc[border1:border2].reset_index(drop=True)
        self.datetime = df['datetime']
        self.price = self.price_scaler.transform(df[[self.target]].values).astype(np.float32)
        self.known_exo = self._transform_optional(df, self.known_features, self.known_scaler)
        self.unknown_exo = self._transform_optional(df, self.unknown_features, self.unknown_scaler)
        self.time_features = time_features_from_datetime(self.datetime)
        self.valid_starts = self._build_valid_starts(self.datetime)

    @staticmethod
    def _transform_optional(df, features, scaler):
        if not features:
            return np.zeros((len(df), 0), dtype=np.float32)
        return scaler.transform(df[features].values).astype(np.float32)

    def _build_valid_starts(self, datetimes):
        total_len = self.seq_len + self.pred_len
        if len(datetimes) < total_len:
            return []

        deltas = datetimes.diff().dt.total_seconds().div(60).to_numpy()
        continuous_edge = np.ones(len(datetimes), dtype=bool)
        continuous_edge[1:] = deltas[1:] == self.expected_minutes

        valid_starts = []
        for start in range(0, len(datetimes) - total_len + 1):
            end = start + total_len
            if not continuous_edge[start + 1:end].all():
                continue
            if self.flag == 'test':
                forecast_time = datetimes.iloc[start + self.seq_len]
                if forecast_time.hour != self.test_start_hour or forecast_time.minute != self.test_start_minute:
                    continue
            valid_starts.append(start)
        return valid_starts

    def __getitem__(self, index):
        start = self.valid_starts[index]
        hist_end = start + self.seq_len
        future_end = hist_end + self.pred_len

        unknown_dim = self.unknown_exo.shape[-1]
        unknown_future_mask = np.ones((self.pred_len, unknown_dim), dtype=np.float32)

        return {
            'price_hist': torch.from_numpy(self.price[start:hist_end]),
            'price_future': torch.from_numpy(self.price[hist_end:future_end]),
            'known_hist_exo': torch.from_numpy(self.known_exo[start:hist_end]),
            'known_future_exo': torch.from_numpy(self.known_exo[hist_end:future_end]),
            'unknown_hist_exo': torch.from_numpy(self.unknown_exo[start:hist_end]),
            'unknown_future_mask': torch.from_numpy(unknown_future_mask),
            'time_hist': torch.from_numpy(self.time_features[start:hist_end]),
            'time_future': torch.from_numpy(self.time_features[hist_end:future_end]),
        }

    def __len__(self):
        return len(self.valid_starts)

    def inverse_price(self, data):
        shape = data.shape
        return self.price_scaler.inverse_transform(data.reshape(-1, 1)).reshape(shape)
