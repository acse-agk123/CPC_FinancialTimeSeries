# Name: Antony Krymski
# Username: agk-123

import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils.utilities import sharpe_ratio, scale, split, window, load_data, evals_initialise, eval

def test_sharpe_ratio():
    returns = tf.constant([0.01, 0.02, 0.03, 0.04], dtype=tf.float32)
    result = sharpe_ratio(returns).numpy()
    expected = (tf.reduce_mean(returns) / tf.math.reduce_std(returns)) * np.sqrt(252.0)
    assert np.isclose(result, expected.numpy(), atol=1e-5)

def test_scale():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    result = scale(a)
    expected = (a - np.min(a, axis=0)) / (np.max(a, axis=0) - np.min(a, axis=0)) * 2 - 1
    np.testing.assert_array_almost_equal(result, expected, decimal=5)

def test_split():
    X = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([7, 8, 9, 10, 11, 12])
    X_train, X_test, y_train, y_test = split(X, y, test_size=0.5)
    assert len(X_train) == 3
    assert len(X_test) == 3
    np.testing.assert_array_equal(X_train, np.array([1, 2, 3]))
    np.testing.assert_array_equal(X_test, np.array([4, 5, 6]))

def test_window():
    data = np.array([1, 2, 3, 4, 5, 6])
    timesteps = 3
    result = window(data, timesteps)
    expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    np.testing.assert_array_equal(result, expected)

def test_load_data():
    timeseries = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    w = 3
    X_train, X_test, y_train, y_test = load_data(timeseries, w=w, test_ratio=0.2)
    assert X_train.shape[0] == 5
    assert X_test.shape[0] == 2

def test_evals_initialise():
    df = evals_initialise()
    assert df.shape == (2, 0)

def test_eval():
    y_test = np.array([1, 2, 3, 4])
    y_pred = np.array([1.1, 2.1, 2.9, 4.0])
    result = eval('TestModel', y_pred, y_test)
    expected_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    expected_mae = mean_absolute_error(y_test, y_pred)
    assert np.isclose(result[0], expected_rmse, atol=1e-5)
    assert np.isclose(result[1], expected_mae, atol=1e-5)