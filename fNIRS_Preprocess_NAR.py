# Import MNE-Python library for neuroimaging data processing
import mne
from matplotlib.ticker import FormatStrFormatter
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_cnt, read_raw_nirx
from config import n_jobs

import numpy as np
import matplotlib.pyplot as plt
import pickle
import gc  
import os.path

# Time window for epoch extraction: epochs will be extracted from -4 seconds before to 14 seconds after each event
# This captures the baseline period (-4 to 0s) and the motor imagery task period (0 to 14s)
tmin = -4.  # Start time relative to event (negative = before event, captures baseline)
tmax = 14.  # End time relative to event (positive = after event, captures full task period)

# Number of subjects in the FineMI dataset
num_subjects = 18

# Sampling rate of the dataset in Hz (1000 Hz = 1000 data points per second)
sample_rate = 7.8125

# Preprocessing Settings
# Filter order: higher order = sharper frequency cutoff, but more computational cost
# Order 6 is a good balance for fNIRS data
filtOrder = 6
# Filter type: Butterworth filter provides flat frequency response in passband (no ripples)
filtType = "butter"

# Package Global Settings
seed = 1  # Random seed for reproducibility (ensures same results on repeated runs)
# Set MNE logging level to WARNING to suppress INFO and DEBUG messages (reduces console output)
mne.set_log_level(verbose="WARNING")


def load_rawData(subject_name):
    n_blocks = 8  # Each subject has 8 experimental blocks (sessions)
    file_train_list = []  # List to store individual block data before concatenation
    if subject_name == '1':
        file = read_raw_nirx("./Data/FineMI/subject" + subject_name + "/fNIRS/block1-4", preload=True)
        # Remove last 40 annotations (likely artifacts or unwanted events at end of recording)
        idx_to_remove = np.arange(-40, 0)
        # Delete annotations at specified indices
        file.annotations.delete(idx_to_remove)
        # Crop the raw data to only include periods with annotations (removes leading/trailing data without events)
        file.crop_by_annotations()
        file_train_list.append(file)
        # Block5-8: Load individual files for blocks 5 through 8
        for block_idx in range(4, n_blocks):  # range(4, 8) = [4, 5, 6, 7]
            file_name = "./Data/FineMI/subject" + subject_name + "/fNIRS/block" + str(block_idx + 1)
            file = read_raw_nirx(file_name, preload=True)
            file_train_list.append(file)
    elif subject_name == '9':
        # Subject 9's fNIRS data is missing from the dataset
        print('subject 9 fNIRS data is not available')
        return None, None
    else:
        # Standard case: all other subjects have individual block files (blocks 1-8)
        for block_idx in range(n_blocks):  # range(8) = [0, 1, 2, 3, 4, 5, 6, 7]
            file_name = "./Data/FineMI/subject" + subject_name + "/fNIRS/block" + str(block_idx + 1)
            file = read_raw_nirx(file_name, preload=True)
            # Special fix for subject 5, block 6: remove first annotation (likely artifact or bad marker)
            if subject_name == "5" and block_idx == 5:
                file.annotations.delete(0)  # Delete annotation at index 0
                file.crop_by_annotations()
            file_train_list.append(file)
    file_train = concatenate_raws(file_train_list)
    file_test = None
    print("Subject " + subject_name + " file loaded successfully")
    return file_train, file_test

    
def preprocessData(file_train, subject_name, tmin, tmax):
    """
    Preprocesses raw fNIRS data through the standard pipeline:
    1. Convert to optical density (OD) - removes variability in source intensity
    2. Convert to hemoglobin concentration (HbO/HbR) using Beer-Lambert law
    3. Apply bandpass filter - removes noise outside hemodynamic response range
    4. Extract epochs - segment data into time windows around events
    
    Args:
        file_train: Raw fNIRS data (light intensity measurements)
        file_test: Test data (unused)
        subject_name: Subject identifier
        tmin: Epoch start time
        tmax: Epoch end time
        
    Returns:
        X_train: Preprocessed epoch data (numpy array)
        Y_train: Labels (task IDs)
        X_test: Test data (None)
        Y_test: Test labels (None)
        frequency_list: Frequency information (empty for fNIRS, used for EEG)
        epoch_info: Metadata about epochs
    """
        # Save epochs to disk for later use 
    if not os.path.exists("img/fnirs/subject" + subject_name):
        os.makedirs("img/fnirs/subject" + subject_name)
    # Step 1: Convert raw light intensity to optical density (OD)
    # OD = -log(I/I0) where I is measured intensity, I0 is reference intensity
    # This removes the effect of source intensity variations and makes data more comparable
    file_train_od = mne.preprocessing.nirs.optical_density(file_train)

    # Step 2: Calculate the scalp coupling index (SCI) using the optical density data
    # The SCI is a measure of the coupling between the source and detector channels
    # It is calculated as the ratio of the variance of the source channels to the variance of the detector channels
    # A higher SCI indicates a stronger coupling between the source and detector channels (ie good channels)
    # A lower SCI indicates a weaker coupling between the source and detector channels (ie bad channels)
    file_train_sci = mne.preprocessing.nirs.scalp_coupling_index(file_train_od, l_freq = 0.7, h_freq = 1.5, verbose = False)
    idx = np.where(file_train_sci < 0.5)[0]
    bad_channels = [file_train.ch_names[i] for i in idx]
    good_channels = [file_train.ch_names[i] for i in range(len(file_train_sci)) if i not in idx]
    print("Bad channels: ", bad_channels)
    print("Good channels: ", good_channels)
    file_train.info['bads'] = bad_channels
    # Step 3: Convert optical density to hemoglobin concentration using Beer-Lambert law
    # This gives us HbO (oxyhemoglobin) and HbR (deoxyhemoglobin) concentrations in µM (micromolar)
    # The Beer-Lambert law relates light absorption to chromophore concentration
    # HbO and HbR have different absorption spectra, allowing separation using multiple wavelengths
    file_train_hemo = mne.preprocessing.nirs.beer_lambert_law(file_train_od)

    # plot what the power spectal density looks like prior to band pass filtering 
    fig = file_train_hemo.compute_psd().plot(average=True, amplitude=False)
    fig.suptitle("Before filtering", weight="bold", size="x-large")
    fig.savefig("img/fnirs/subject" + subject_name + "/beforeFilt.png", dpi=300)
    plt.close(fig)

    # Configure bandpass filter parameters 
    iir_params = {
        "output": "sos",
        "order": filtOrder,
        "ftype": filtType
    }
    # filter the data using bandpass and plot PSD after the filtering
    file_train_hemoData = file_train_hemo.filter(0.01, 0.3, method='iir', iir_params=iir_params, n_jobs=n_jobs)
    fig = file_train_hemoData.compute_psd().plot(average=True, amplitude=False)
    fig.suptitle("After filtering", weight="bold", size="x-large")
    fig.savefig("img/fnirs/subject" + subject_name + "/afterFilt.png", dpi=300)
    plt.close(fig)
    print('done filtering')

    # set up function for extracting epochs related to each experimental condition
    # plot the events, their timings, and occurences 
    events, event_dict = mne.events_from_annotations(file_train_hemoData)
    fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=file_train_hemo.info["sfreq"])
    fig.savefig("img/fnirs/subject" + subject_name + "/events.png", dpi=300)
    plt.close(fig)
    epochs_data = Epochs(file_train_hemoData, events, event_id= event_dict, tmin =tmin, tmax = tmax, proj=True,
                        baseline=(-3.9, -2.), preload=True, verbose=False)

    training_data = epochs_data.get_data()
    training_labels = epochs_data.events[:, -1] - 1

    del file_train_hemoData # free up some memory

    # Save epochs to disk for later use 
    if not os.path.exists("tmp"):
        os.makedirs('tmp')
    # Save in MNE's FIF format (Fast Imaging Format) for efficient storage
    epochs_data.save("tmp/subject" + str(subject_name) + "_fNIRS-epo.fif", overwrite=True)

    return training_data, training_labels, epochs_data


if __name__ == "__main__":
    for subject in range(1, num_subjects + 1):
        subject_name = str(subject)
        file_train, file_test = load_rawData(subject_name)
        X_train, Y_train, epoch_info = preprocessData(file_train, subject_name, tmin, tmax)
        print("Subject " + subject_name + " data preprocessed successfully")
