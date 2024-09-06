# Name: Antony Krymski
# Username: agk-123

# Import TensorFlow and Keras components
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Conv1D, Conv2D, Flatten, Dense, LSTM, Reshape, MaxPooling2D, LayerNormalization, TimeDistributed, GRU, Lambda, BatchNormalization, LeakyReLU, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint
except ImportError as e:
    raise ImportError(f"TensorFlow or its Keras components could not be imported. Please ensure that TensorFlow is installed properly. Original error: {e}")

# Import Matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(f"Matplotlib could not be imported. Please ensure that Matplotlib is installed. Original error: {e}")

# Import NumPy
try:
    import numpy as np
except ImportError as e:
    raise ImportError(f"NumPy could not be imported. Please ensure that NumPy is installed. Original error: {e}")

# Import Pandas
try:
    import pandas as pd
except ImportError as e:
    raise ImportError(f"Pandas could not be imported. Please ensure that Pandas is installed. Original error: {e}")

# Import Scikit-Learn components
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
except ImportError as e:
    raise ImportError(f"Scikit-Learn or its components could not be imported. Please ensure that Scikit-Learn is installed. Original error: {e}")
