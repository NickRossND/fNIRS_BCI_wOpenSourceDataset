# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================
# Functions for preprocessing fNIRS data: standardization and main preprocessing pipeline

import numpy as np
import pandas as pd
import mne
from mne import Epochs, events_from_annotations
from config import n_jobs


def exponential_moving_standardize(data, factor_new=0.001, eps=1e-4):
    """
    Apply exponential moving standardization to remove slow drifts in fNIRS data.
    This is an adaptive normalization method that accounts for non-stationarity.
    
    For each time point, it calculates a running mean and standard deviation using
    exponential moving average, then standardizes the data. This removes slow drifts
    while preserving the hemodynamic response signal.
    
    Args:
        data: numpy array of shape (n_channels, n_times) - one channel's time series
        factor_new: Smoothing factor for exponential moving average (smaller = more smoothing, slower adaptation)
        eps: Small value to prevent division by zero in standardization
        
    Returns:
        standardized: Standardized data with same shape as input (n_channels, n_times)
    """
    # Transpose to (n_times, n_channels) for easier processing with pandas
    data = data.T
    df = pd.DataFrame(data)
    
    # Calculate exponential moving average (EMA) of the mean
    meaned = df.ewm(alpha=factor_new).mean()
    
    # Remove the moving mean (demean the data)
    demeaned = df - meaned
    
    # Calculate squared deviations from the moving mean
    squared = demeaned * demeaned
    
    # Calculate EMA of the variance (squared deviations)
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    
    # Standardize: divide by moving standard deviation
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    
    # Transpose back to (n_channels, n_times) to match original format
    return standardized.T


def preprocessing_fnirs_func(raw_data, dataset, preprocessing_params, subject_idx=0):
    """
    Complete preprocessing pipeline for fNIRS data.
    Converts raw light intensity → optical density → hemoglobin concentration → filtered → epochs.
    
    This function transforms raw fNIRS measurements into machine-learning-ready data.
    
    Args:
        raw_data: Raw fNIRS data (light intensity measurements from the fNIRS system)
        dataset: FineMI dataset object with configuration
        preprocessing_params: Dictionary with preprocessing settings (filter parameters, etc.)
        subject_idx: Subject index (for potential subject-specific processing)
        
    Returns:
        X_fnirs: Preprocessed epoch data (n_epochs, n_channels, n_timepoints)
        Y_fnirs: Class labels (0-indexed, e.g., [0, 0, 1, 1, ...])
        info_fnirs: Metadata about the data (channel names, sampling rate, etc.)
        freq_bounds: List of frequency bands used (for compatibility with other code)
    """
    original_annotations = raw_data.annotations.copy()

    # Step 1: Convert raw light intensity to optical density (OD)
    # OD = -log(I/I0) where I is measured intensity, I0 is reference intensity
    optical_density = mne.preprocessing.nirs.optical_density(raw_data)

    # Step 2: Convert optical density to hemoglobin concentration using Beer-Lambert law
    # This gives us HbO (oxyhemoglobin) and HbR (deoxyhemoglobin) concentrations in µM
    hemo_data = mne.preprocessing.nirs.beer_lambert_law(optical_density)

    # Step 3: Apply bandpass filter to remove noise
    if preprocessing_params["use_band_pass"]:
        iir_params = {
            "output": "sos",
            "order": preprocessing_params["filter_order"],
            "ftype": preprocessing_params["filter_type"]
        }
        hemo_data = hemo_data.filter(preprocessing_params["lower_bound"],
                                                           preprocessing_params["upper_bound"],
                                                           method='iir',
                                                           iir_params=iir_params,
                                                           n_jobs=n_jobs)

    # Step 4: Apply exponential moving standardization (optional)
    if preprocessing_params["moving_average_std"]:
        hemo_data = hemo_data.apply_function(exponential_moving_standardize,
                                                                   n_jobs=n_jobs,
                                                                   channel_wise=False)

    # Step 5: Extract epochs and convert to numpy arrays
    # Extract event markers from annotations
    hemo_data.set_annotations(original_annotations)
    events, event_dict = events_from_annotations(hemo_data, event_id=dataset.event_id_fnirs)
    # DEBUG: Check if events were found
    # print(f"DEBUG: Found {len(events)} events in hemo_data")
    # print(f"DEBUG: hemo_data.annotations: {hemo_data.annotations}")
    # print(f"DEBUG: event_dict: {event_dict}")
    # if len(events) == 0:
    #     print("ERROR: No events found! Cannot create epochs.")
    #     raise ValueError("No events found in preprocessed data. Check annotations and event_id_fnirs.")
    
    # Extract epochs: segments data into time windows around each event
    epochs = Epochs(hemo_data, events, event_id=dataset.event_id_fnirs, 
                    tmin=dataset.tmin, tmax=dataset.tmax, proj=True,
                    baseline=(dataset.baseline_tmin, dataset.baseline_tmax), 
                    preload=True, verbose=False)
    
    # Convert epochs to numpy array: shape (n_epochs, n_channels, n_timepoints)
    X_fnirs = epochs.get_data()
    
    # extract labels and convert to 0-indexed
    # epochs.events[:, -1] gets the event ID (last column) for each event
    label = epochs.events[:, -1]
    label_names = np.unique(label)
    for label_idx, label_name in enumerate(label_names):
        label[label == label_name] = label_idx
    Y_fnirs = label

    info_fnirs = epochs.info

    # freq band info 
    freq_bounds = [
        {
            "fmin": preprocessing_params["lower_bound"],
            "fmax": preprocessing_params["upper_bound"]
        }
    ]
    return X_fnirs, Y_fnirs, info_fnirs
