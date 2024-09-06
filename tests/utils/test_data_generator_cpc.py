# Name: Antony Krymski
# Username: agk-123

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from utils.data_generator_cpc import DataGenerator

def test_data_generator_initialization():
    data = np.random.randn(100, 1)
    batch_size = 32
    timesteps = 10
    n_windows = 3
    
    generator = DataGenerator(data, batch_size=batch_size, timesteps=timesteps, n_windows=n_windows)
    
    assert generator.batch_size == batch_size
    assert generator.timesteps == timesteps
    assert generator.n_windows == n_windows
    assert generator.shuffle is True
    assert generator.data.shape == data.shape
    assert len(generator.indexes) == len(data) - (timesteps * n_windows) - timesteps

def test_data_generator_len():
    data = np.random.randn(100, 1)
    batch_size = 32
    timesteps = 10
    n_windows = 3
    
    generator = DataGenerator(data, batch_size=batch_size, timesteps=timesteps, n_windows=n_windows)
    
    expected_len = len(generator.indexes) // batch_size
    assert len(generator) == expected_len

def test_data_generator_getitem():
    data = np.random.randn(100, 1)
    batch_size = 32
    timesteps = 10
    n_windows = 3
    
    generator = DataGenerator(data, batch_size=batch_size, timesteps=timesteps, n_windows=n_windows)
    
    (x, y), labels = generator.__getitem__(0)
    
    # Check the shapes of the output
    assert x.shape == (batch_size, n_windows, timesteps, 1)
    assert y.shape == (batch_size, timesteps, 1)
    assert labels.shape == (batch_size,)
    
    # Check the contents are numeric and within the expected range
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(labels))

def test_data_generator_on_epoch_end():
    data = np.random.randn(100, 1)
    batch_size = 32
    timesteps = 10
    n_windows = 3

    # Initialize with shuffle=True
    generator = DataGenerator(data, batch_size=batch_size, timesteps=timesteps, n_windows=n_windows)
    
    # Copy the initial state of indexes
    initial_indexes = generator.indexes.copy()
    
    # Shuffle should be True initially, so indexes should change
    generator.on_epoch_end()
    assert not np.array_equal(initial_indexes, generator.indexes), "Indexes did not change after shuffle."

    # Reset indexes and re-initialize with shuffle=False
    generator.indexes = initial_indexes.copy()
    generator.shuffle = False
    generator.on_epoch_end()
    
    # With shuffle=False, indexes should remain the same
    assert np.array_equal(initial_indexes, generator.indexes), "Indexes changed despite shuffle being False."


def test_data_generator_batch_content():
    data = np.random.randn(100, 1)
    batch_size = 32
    timesteps = 10
    n_windows = 3
    
    generator = DataGenerator(data, batch_size=batch_size, timesteps=timesteps, n_windows=n_windows)
    (x, y), labels = generator.__getitem__(0)

    # Check that for each element in the batch, half are positive samples and half are negative samples
    assert np.sum(labels == 1) == batch_size // 2
    assert np.sum(labels == 0) == batch_size // 2
