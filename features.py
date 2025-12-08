# ============================================================================
# FEATURE EXTRACTION TRANSFORMERS
# ============================================================================
# Custom sklearn-compatible transformers for feature extraction

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from numpy.polynomial import Polynomial


class Mean(TransformerMixin, BaseEstimator):
    """
    Custom transformer that computes mean across a specified axis.
    Used as a feature extraction step: reduces time dimension to a single value per channel.
    This extracts the average signal amplitude as a feature, which captures the overall
    magnitude of the hemodynamic response.
    
    Example: If input is (n_trials, n_channels, n_times), and axis=-1,
    output is (n_trials, n_channels) - one mean value per channel per trial.
    """

    def __init__(self, axis=0):
        """
        Args:
            axis: Axis along which to compute mean (typically -1 for time dimension)
        """
        self.axis = axis

    def fit(self, X, y=None, **fit_params):
        """
        Fit method (no-op for this transformer, but required by sklearn interface).
        This transformer doesn't need to learn anything from the data.
        """
        return self

    def transform(self, X):
        """
        Compute mean across specified axis.
        
        Args:
            X: Input data (typically shape: n_samples, n_channels, n_times)
            
        Returns:
            Mean values (reduces dimension by 1, e.g., n_samples, n_channels)
        """
        return np.mean(X, axis=self.axis)


class Slope(TransformerMixin, BaseEstimator):
    """
    Custom transformer that computes slope (rate of change) of the signal.
    Used as a feature extraction step: captures the trend/direction of hemodynamic response.
    The slope indicates whether HbO/HbR is increasing or decreasing during the task,
    which is important because motor imagery typically causes an increase in HbO and
    decrease in HbR in the motor cortex.
    
    Example: If input is (n_trials, n_channels, n_times), output is (n_trials, n_channels)
    - one slope value per channel per trial.
    """

    def __init__(self, n_len, method="interval"):
        """
        Args:
            n_len: Length of time series (number of time points) - used for polynomial fitting
            method: "interval" for simple slope calculation, or "polynomial" for polynomial fit
        """
        self.n_len = n_len
        self.method = method

    def fit(self, X, y=None, **fit_params):
        """
        Fit method (no-op, but required by sklearn interface).
        This transformer doesn't need to learn anything from the data.
        """
        return self

    def transform(self, X):
        """
        Compute slope for each channel in each trial.
        Slope measures the rate of change: positive = increasing, negative = decreasing.
        
        Args:
            X: Input data of shape (n_trials, n_channels, n_times)
            
        Returns:
            Slopes of shape (n_trials, n_channels) - one slope value per channel per trial
        """
        n_trials, n_channels, n_times = X.shape
        slopes_of_trials = []

        if self.method == "interval":
            # Simple slope: (last_value - first_value) / n_times
            # Fast approximation of the overall trend
            # This gives the average rate of change over the entire epoch
            slopes_of_trials = (X[:, :, n_times - 1] - X[:, :, 0]) / n_times
        else:
            # Polynomial fit: fit a line to the time series and extract slope
            # More accurate but slower (accounts for all points, not just first and last)
            for trial_idx in range(n_trials):
                slopes = []
                for channel_idx in range(n_channels):
                    times = np.arange(self.n_len)  # Time points: [0, 1, 2, ..., n_len-1]
                    # Fit polynomial of degree 1 (linear) to get slope
                    # This finds the best-fit line through all time points
                    slope, intercept = Polynomial.fit(times, X[trial_idx, channel_idx], 1)
                    slopes.append(slope)
                slopes_of_trials.append(slopes)
        return np.array(slopes_of_trials)

