# Name: Antony Krymski
# Username: agk-123

from .imports import *

plt.style.use("bmh")
plt.rcParams.update({'font.size': 10})

def sharpe_ratio(returns):
    """
    Calculate the Sharpe ratio for a given set of returns.

    The Sharpe ratio measures the risk-adjusted return of an investment.
    It is calculated as the mean return divided by the standard deviation 
    of the return, scaled by the square root of the number of trading days 
    in a year (252 days).

    Args:
        returns (np.array): An array of return values.

    Returns:
        float: The Sharpe ratio of the returns.
    """
    mean_return = tf.reduce_mean(returns)
    std_return = tf.math.reduce_std(returns)
    sharpe = mean_return / std_return
    return sharpe * np.sqrt(252.0)

def scale(a):
    """
    Perform min-max scaling on the input array.

    The function scales the input array to a range of [-1, 1].

    Args:
        a (np.array): The input array to be scaled.

    Returns:
        np.array: The scaled array.
    """
    maxv = np.max(a, axis=0)
    minv = np.min(a, axis=0)
    return (a - minv) / (maxv - minv) * 2. - 1.

def split(X, y, test_size=0.33):
    """
    Split the data into training and testing sets.

    This function performs a time-series split based on the provided test size ratio.

    Args:
        X (np.array): Input features.
        y (np.array): Target values.
        test_size (float): Proportion of the dataset to include in the test split (default is 0.33).

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    i = int(np.floor(len(X) * (1 - test_size)))
    X_train, y_train = X[:i], y[:i]
    X_test, y_test = X[i:], y[i:]
    return X_train, X_test, y_train, y_test

def window(data, timesteps, strides=1):
    """
    Create a windowed version of the input data.

    This function generates overlapping windows of a specified length 
    (timesteps) from the input data, with a specified stride.

    Args:
        data (np.array): The input data array.
        timesteps (int): The length of each window.
        strides (int): The step size between windows (default is 1).

    Returns:
        np.array: An array of windowed data.
    """
    x = []
    for i in range(0, len(data) - timesteps, strides):
        x.append(data[i:i+timesteps])
    return np.array(x)

def download(ticker='^GSPC'):
    """
    Download historical price data for a given ticker from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (default is '^GSPC' for S&P 500).

    Returns:
        pd.DataFrame: A DataFrame containing the historical price data.
    """
    url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=0&period2=9999999999&interval=1d&events=history&includeAdjustedClose=true'
    return pd.read_csv(url, index_col='Date', parse_dates=True)

def plot(x, big=False, allocations=False):
    """
    Plot a time series.

    Args:
        x (np.array or pd.Series): The data to plot.
        big (bool): If True, plot with a larger figure size (default is False).
        allocations (bool): If True, add a title 'Allocations' to the plot (default is False).
    """
    if big:
        plt.figure(figsize=(16, 8))
    else:
        plt.figure(figsize=(10, 4))
    plt.margins(x=0, y=0)
    if allocations:
        plt.title('Allocations')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(x)
    plt.show()

def plot_forecast(x, y, y_pred):
    """
    Plot actual vs. forecasted time series data.

    Args:
        x (np.array or pd.Series): Historical data.
        y (np.array or pd.Series): Actual future data.
        y_pred (np.array or pd.Series): Forecasted future data.
    """
    t1 = np.arange(0, len(x), 1)
    t2 = np.arange(len(x), len(x) + len(y), 1)
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.margins(x=0, y=0)
    ax.plot(t1, x, color='blue')
    ax.plot(t2, y, color='blue', label='Actual')
    ax.plot(t2, y_pred, color='red', label='Forecast')
    ax.axvline(x=len(x), color='gray', linestyle='--')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

def plot_timeseries(**kwargs):
    """
    Plot multiple time series on the same figure.

    Accepts keyword arguments to specify the series to plot, 
    their labels, and optional title and index.

    Args:
        **kwargs: Keyword arguments for series to plot, title, and index.
    """
    fig, (ax) = plt.subplots(1, 1, sharex=True, figsize=(16, 8))
    ax.margins(x=0, y=0)
    index = kwargs.get('index', None)
    for key, value in kwargs.items():
        if key == 'title':
            ax.set_title(value)
        elif key == 'index':
            pass
        else:
            if index is None:
                ax.plot(value, label=key)
            else:
                ax.plot(index, value, label=key)
                
    # Make the axis tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=25)
    
    plt.legend(loc='upper left', fontsize=25)
    plt.tight_layout()
    plt.show()

def plot_training(history, metric='loss'):
    """
    Plot training and validation metrics over epochs.

    Args:
        history (tf.keras.callbacks.History): The history object from a Keras model fit.
        metric (str): The metric to plot (default is 'loss').
    """
    plot_timeseries(index=range(history.params['epochs']), title=metric, train=history.history[metric], test=history.history['val_' + metric])
    
def load_data(timeseries, w=None, test_ratio=0.2):
    """
    Load and preprocess time series data.

    This function removes outliers, scales the data, generates lags,
    and splits the data into training and testing sets.

    Args:
        timeseries (pd.Series): The input time series data.
        w (int): The window length for generating lags.
        test_ratio (float): The proportion of the dataset to include in the test split (default is 0.2).

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = timeseries.values

    # Remove outliers
    outliers = np.quantile(np.abs(X), 0.99)
    X = np.clip(X, -outliers, outliers)
    print('Outliers', outliers)

    # Scale the data
    X = X / outliers

    # Generate lags
    X = window(X, w).astype(np.float32)
    y = timeseries[w:]
    return split(X, y, test_ratio)

def evals_initialise():
    """
    Initialize a DataFrame for evaluation metrics.

    Returns:
        pd.DataFrame: An empty DataFrame with index set to evaluation metrics.
    """
    return pd.DataFrame(index=['RMSE', 'MAE'])

def eval(name, y_pred, y_test):
    """
    Evaluate model performance using RMSE and MAE.

    Args:
        name (str): The name of the model or experiment.
        y_pred (np.array): The predicted values.
        y_test (np.array): The true values.

    Returns:
        list: A list containing RMSE and MAE values.
    """
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return [rmse, mae]
