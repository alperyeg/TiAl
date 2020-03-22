import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def load_data(data_loader='', filename='TNM.wfl', names=('A', 'B', 'C', 'D'),
              standardize=False, test_size=0.2, batch_size=32, save=False):
    """
    Loads from `filename` values represented sa columns and stores them as a
    table in `data`.

    :param data_loader, string, if the path of the file is specified then a
           `Pytorch dataloader` is loaded and returned, otherwise it will be
           created using `filename`
    :param filename: string, path to file
    :param names: string, defines name of columns for the data object
           number should be equal to column number in the file
    :param standardize: bool, if `True` then standardize features by removing 
           the mean and scaling to unit variance
    :param test_size: float, Percentage to split the dataset to train and test,
           should be in [0,1)
    :param batch_size: int, size of the batch
    :param save: string, if `True` then a pickable dataloader as `dataloader\.pt`
           will be saved, only in combination when creating a dataloader
    """
    try:
        dataloader = torch.load('dataloader.pt')
        return dataloader['train_loader'], dataloader['test_loader']
    except FileNotFoundError:
        data = pd.read_table('TNM.wfl', delim_whitespace=True,
                             names=names)
        train_data, test_data = _create_dataloader(data)
        if standardize:
            scaler = Normalizer()
            # Take all columns exluding the first one
            fitted = scaler.fit_transform(data.T)
            # print("in fitted: ", scaler.scale_)
            # returns a numpy array which is reshaped into a DataFrame
            fitted = pd.DataFrame(fitted.T, columns=names)
            # Concatenation with the first column
            train_data, test_data = _create_dataloader(fitted)
            if save:
                torch.save('dataloader.pt', {
                           'train_loader': train_data,
                           'test_loader': test_data})
        return train_data, test_data


def _create_dataloader(data, batch_size=32, standardize=True, test_size=0.2):
    """
    Creates train and test `Pytorch` dataloaders

    :return: Pytorch train and test dataloader
    """
    # TODO
    x = data.iloc[:, 0:3]
    y = data.iloc[:, 3:]
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        train_size=1-test_size,
                                                        test_size=test_size)
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    # TODO adapt to correct shapes with .view()
    x_train = torch.from_numpy(x_train.values)
    x_test = torch.from_numpy(x_test.values)
    train_set = TensorDataset(x_train, torch.from_numpy(y_train.values))
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    test_set = TensorDataset(x_test, torch.from_numpy(y_test.values))
    test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


# Testing
if __name__ == "__main__":
    train, test = load_data()
    print(train, test)
