# -*- coding: utf-8 -*-
"""Data loader and base learner for Evoluationary Regressor Chains Testing.

Data loaders, all datasets are indexed in size of (t, m, d),
where t is the timestamp of data,
m is the number of data streams,
and d is the dimension of the feature
"""
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrainStreams(Dataset):
    """Train dataset."""

    def __init__(self, device=device):
        """Initialize a TrainStreams object."""
        super(TrainStreams, self).__init__()
        df = pd.read_csv('data/multistream.central.csv')
        self.x = torch.zeros((373327, 8, 11), device=device)
        self.y = torch.zeros((373327, 8), device=device)
        x_period = df['PERIODS'].to_numpy()
        x_stations = df[['TRIP_ORGIN_STATION',
                         'TRIP_DESTINATION_STATION']].to_numpy()
        x_platforms = df['PLANNED_ARRIVAL_PLATFORM'].to_numpy()
        for i in range(8):
            self.x[:, i, 0] = torch.from_numpy(x_period).to(device)
            self.x[:, i, 1:3] = torch.from_numpy(x_stations).to(device)
            self.x[:, i, 3] = torch.from_numpy(x_platforms).to(device)
            x_others = df[['SEGMENT_DIRECTION_NAME', 'TRIP_DURATION_SEC',
                           'DWELL_TIME', 'DAY_OF_YEAR', 'DAY_OF_WEEK', 'TIME',
                           'CAR{}_PSNGLD_ARRIVE'.format(i + 1)]].to_numpy()
            y = df['CAR{}_PSNGLD_DEPART'.format(i + 1)].to_numpy()
            self.x[:, i, 4:] = torch.from_numpy(x_others).to(device)
            self.y[:, i] = torch.from_numpy(y).to(device)
        del df

    def __len__(self):
        """Return the length."""
        return 373327

    def __getitem__(self, key):
        """Return data with given index."""
        return self.x[key, :, :], self.y[key, :]


class BaseLearnerTrain(nn.Module):
    """A simple linear model with embedding layers."""

    def __init__(self, device=device):
        """Initialize a new base learner."""
        super(BaseLearnerTrain, self).__init__()
        self.embedding_period = nn.Embedding(4, 2)
        self.embedding_stations = nn.Embedding(162, 3)
        self.embedding_platforms = nn.Embedding(460, 3)
        self.fc_period = nn.Linear(2, 1, bias=False)
        self.fc_stations = nn.Linear(6, 1, bias=False)
        self.fc_platforms = nn.Linear(3, 1, bias=False)
        self.fcn = nn.Linear(7, 1, bias=False)
        self.fcy = nn.Linear(1, 1)
        nn.init.kaiming_uniform_(self.fc_period.weight)
        nn.init.kaiming_uniform_(self.fc_stations.weight)
        nn.init.kaiming_uniform_(self.fc_platforms.weight)
        nn.init.kaiming_uniform_(self.fcn.weight)
        nn.init.kaiming_uniform_(self.fcy.weight)
        nn.init.zeros_(self.fcy.bias)
        self.to(device)

    def forward(self, x, y):
        """Calculate forward."""
        if len(x.shape) == 1:
            x = x.view(1, -1)
        n, d = x.shape
        x1 = self.embedding_period(x[:, 0].to(torch.int32))
        x1 = self.fc_period(x1.view(n, -1))
        x2 = self.embedding_stations(x[:, 1:3].to(torch.int32))
        x2 = self.fc_stations(x2.view(n, -1))
        x3 = self.embedding_platforms(x[:, 3].to(torch.int32))
        x3 = self.fc_platforms(x3.view(n, -1))
        x4 = self.fcn(x[:, 4:])
        x5 = self.fcy(y)
        xx = x1 + x2 + x3 + x4 + x5
        return xx


if __name__ == "__main__":
    print('Test SydneyTrain dataset...')
    train = TrainStreams()
    for i, data in enumerate(train[:10]):
        x, y = data
        print(x.shape, y.shape)
    print('Test complete!')
    print()
    x, y = train[:10]
    print("Test BaseLearnerTrain forward method...")
    learner = BaseLearnerTrain()
    y_predict = learner(x[:, 0, :], y[:, 0:1])
    print("y_predict = ", y_predict)
    print("Test complete.")
