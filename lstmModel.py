import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential


def get_train_test_data(df):
    """
    Split the data into training and testing sets

    Parameters
    ----------
    df : pandas.DataFrame
        The input data

    Returns
    -------
    train_data : pandas.DataFrame
        The training data
    test_data : pandas.DataFrame
        The testing data
    """
    train_data = df[
        (df["date_saisie"] >= "2006-01-01") & (df["date_saisie"] <= "2018-12-31")
    ]
    test_data = df[
        (df["date_saisie"] >= "2019-01-01") & (df["date_saisie"] <= "2019-12-31")
    ]
    return train_data, test_data


class lstm_model(BaseEstimator, RegressorMixin):
    def __init__(self, seq_length=10):
        """
        Initialize the LSTM model

        Parameters
        ----------
        seq_length : int
            The length of the input sequences

        Returns
        -------
        None
        """
        self.seq_length = seq_length  # The length of the input sequences
        self.X_train = None  # The training input data
        self.y_train = None  # The training output data
        self.X_test = None  # The testing input data
        self.y_test = None  # The testing output data
        self.y_pred = None  # The predicted output data
        self.mse = None  # The mean squared error
        self.mape = None  # The mean absolute percentage error
        self.model = Sequential()  # The LSTM model
        self.model.add(LSTM(64, input_shape=(seq_length, 1)))  # Add an LSTM layer
        self.model.add(Dense(1))  # Add a dense layer
        self.model.compile(
            loss="mean_squared_error", optimizer="adam"
        )  # Compile the model

    def create_sequences(self, data, seq_length):
        """
        Create input and output sequences for LSTM

        Parameters
        ----------
        data : numpy.ndarray
            The input data
        seq_length : int
            The length of the input sequences

        Returns
        -------
        X : numpy.ndarray
            The input sequences
        y : numpy.ndarray
            The output sequences
        """
        X = []
        y = []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def fit(self, train_data, epochs=10, batch_size=32):
        """
        Train the LSTM model

        Parameters
        ----------
        train_data : pandas.DataFrame
            The training data
        epochs : int
            The number of epochs
        batch_size : int
            The batch size

        Returns
        -------
        None
        """
        X_train, y_train = self.create_sequences(
            train_data["montant_HT"].values, self.seq_length
        )
        self.X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        self.y_train = y_train
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, test_data):
        """
        Predict the output data

        Parameters
        ----------
        test_data : pandas.DataFrame
            The testing data

        Returns
        -------
        y_pred : numpy.ndarray
            The predicted output data
        """
        X_test, y_test = self.create_sequences(
            test_data["montant_HT"].values, self.seq_length
        )
        self.X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        self.y_test = y_test
        self.y_pred = self.model.predict(X_test)
        return self.y_pred

    def evaluate(self):
        """
        Evaluate the LSTM model : compute the mean squared error and the mean absolute percentage error

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.mse = self.model.evaluate(self.X_test, self.y_test)
        self.mape = np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100
        print("Mean Squared Error:", self.mse)
        print("Mean Absolute Percentage Error:", self.mape, "%")
