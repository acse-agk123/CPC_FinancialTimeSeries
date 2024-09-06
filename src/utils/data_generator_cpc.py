# Name: Antony Krymski
# Username: agk-123

from .imports import *

class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator class for Keras models using the Sequence API.

    This generator yields batches of data for training or evaluation in a
    way that is memory efficient and supports shuffling and multiple time 
    window generation. The generator creates both positive and negative samples.

    Attributes:
        data (np.array): The input data array.
        batch_size (int): The size of each batch.
        shuffle (bool): Whether to shuffle the data after each epoch.
        timesteps (int): Number of timesteps for each window.
        n_windows (int): Number of windows to generate for each sample.
        indexes (np.array): Indexes of the data, used for shuffling.
    """
    
    def __init__(self, data, batch_size=32, shuffle=True, timesteps=None, n_windows=None):
        """
        Initializes the data generator with the provided parameters.

        Args:
            data (np.array): The input data array.
            batch_size (int): Size of each batch (default is 32).
            shuffle (bool): Whether to shuffle the data after each epoch (default is True).
            timesteps (int): Number of timesteps for each window.
            n_windows (int): Number of windows to generate for each sample.
        """
        self.timesteps = timesteps
        self.n_windows = n_windows
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data
        
        # Calculate the indices for data excluding incomplete windows
        self.indexes = np.arange(len(self.data) - (timesteps * n_windows) - timesteps)
        self.shuffle = shuffle
        self.on_epoch_end()  # Initialize data shuffling

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        Returns:
            int: The number of batches per epoch.
        """
        return len(self.indexes) // self.batch_size

    def get_window(self, idx):
        """
        Retrieves a single window of data starting at the specified index.

        Args:
            idx (int): Starting index for the window.

        Returns:
            np.array: A slice of the data corresponding to the window.
        """
        return self.data[idx:idx+self.timesteps, 0]

    def __getitem__(self, batch):
        """
        Generates one batch of data.

        Args:
            batch (int): The index of the batch.

        Returns:
            tuple: A tuple containing the batch of inputs (x, y) and the corresponding labels.
        """
        x = []  # List to store input windows
        y = []  # List to store target outputs
        labels = []  # List to store labels (1 for positive, 0 for negative)

        # Select indexes for the current batch
        indexes = self.indexes[batch*self.batch_size//2:(batch+1)*self.batch_size//2]

        for idx in indexes:
            windows = []

            # Generate input windows
            for w in range(self.n_windows):
                window = self.get_window(idx + (w * self.timesteps))
                windows.append(window)

            # Generate positive output window
            y_positive = self.get_window(idx + (self.n_windows * self.timesteps))

            # Generate negative output window by adding noise
            y_negative = self.get_window(idx + ((self.n_windows-1) * self.timesteps))
            y_negative = np.random.normal(np.mean(y_negative), np.std(y_negative), len(y_negative))

            # Append positive sample
            x.append(windows)
            y.append(y_positive)
            labels.append(1)

            # Append negative sample
            x.append(windows)
            y.append(y_negative)
            labels.append(0)

        x = np.array(x).astype(np.float32)
        y = np.array(y).astype(np.float32)
        labels = np.array(labels).astype(np.float32)

        # Expand dimensions to add a channel dimension
        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)

        return (x, y), labels

    def on_epoch_end(self):
        """
        Updates indexes after each epoch. Shuffles the data if shuffle is set to True.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)
