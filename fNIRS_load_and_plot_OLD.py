# Before running this script, please ensure that the dataset was downloaded and saved in the "./Data/FineMI" directory

# Import MNE-Python library for neuroimaging data processing (EEG, fNIRS, etc.)
import mne
# Import formatter for customizing number display in plots
from matplotlib.ticker import FormatStrFormatter
# Import MNE classes: Epochs (time-locked data segments), pick_types (channel selection), events_from_annotations (event markers)
from mne import Epochs, pick_types, events_from_annotations
# Import functions to concatenate multiple recordings and read different file formats
from mne.io import concatenate_raws, read_raw_cnt, read_raw_nirx

import numpy as np
import matplotlib.pyplot as plt

import pickle
import gc  # Garbage collector for memory management
import os.path

# Dataset configuration: specifies which dataset to process
dataset_name = "FineMI"

# Time window for epoch extraction: epochs will be extracted from -4 seconds before to 14 seconds after each event
# This captures the baseline period (-4 to 0s) and the motor imagery task period (0 to 14s)
if dataset_name == "FineMI":
    tmin = -4.  # Start time relative to event (negative = before event, captures baseline)
    tmax = 14.  # End time relative to event (positive = after event, captures full task period)


# Number of subjects in the FineMI dataset
if dataset_name == "FineMI":
    n_subjects = 18

# Starting subject number for processing loop (1 = start from subject 1)
start_subject = 1

# Sampling rate of the dataset in Hz (1000 Hz = 1000 data points per second)
if dataset_name == "FineMI":
    sample_rate_dataset = 1000

# Preprocessing Settings
# Filter order: higher order = sharper frequency cutoff, but more computational cost
# Order 6 is a good balance for fNIRS data
filter_order_fnirs = 6
# Filter type: Butterworth filter provides flat frequency response in passband (no ripples)
filter_type_fnirs = "butter"

# Package Global Settings
seed = 1  # Random seed for reproducibility (ensures same results on repeated runs)
# Set MNE logging level to WARNING to suppress INFO and DEBUG messages (reduces console output)
mne.set_log_level(verbose="WARNING")
# Number of parallel jobs for processing (2 = use 2 CPU cores simultaneously for faster processing)
n_jobs = 2


# Matplotlib configuration for Chinese font support (Microsoft YaHei)
plt.rcParams['font.family'] = ['Microsoft YaHei']
# Prevent matplotlib from using minus sign that may not render properly with Chinese fonts
plt.rcParams['axes.unicode_minus'] = False

# Plotting configuration flags
plot_all_subjects = True
# load_mean_epochs = True
# If True, load pre-computed mean epochs from pickle files; if False, compute them from scratch
load_mean_epochs = False
plot_mean_of_selected_channels = True
plot_by_class = True  # Plot data separated by motor imagery class/task
plot_trends = True  # Generate time-series trend plots showing HbO/HbR changes over time
plot_topo = True  # Generate topographic maps showing spatial distribution of activation
# fNIRS channel configuration: defines source-detector pairs that form measurement channels
# Each dictionary specifies a source (S) and detector (D) pair
# In fNIRS, light is emitted from sources and detected at detectors; the path between them measures brain activity
# These represent the optode (optical sensor) positions on the scalp
channel_names = [{"S": 1, "D": 1},
                 {"S": 1, "D": 2},
                 {"S": 1, "D": 5},
                 {"S": 2, "D": 1},
                 {"S": 2, "D": 2},
                 {"S": 2, "D": 3},
                 {"S": 3, "D": 2},
                 {"S": 3, "D": 3},
                 {"S": 3, "D": 4},
                 {"S": 3, "D": 7},
                 {"S": 4, "D": 3},
                 {"S": 4, "D": 4},
                 {"S": 5, "D": 5},
                 {"S": 5, "D": 6},
                 {"S": 6, "D": 2},
                 {"S": 6, "D": 5},
                 {"S": 6, "D": 6},
                 {"S": 6, "D": 7},
                 {"S": 7, "D": 6},
                 {"S": 7, "D": 7},
                 {"S": 7, "D": 8},
                 {"S": 8, "D": 4},
                 {"S": 8, "D": 7},
                 {"S": 8, "D": 8}, ]
# Source optode names: locations where light emitters are placed (following 10-10 EEG system naming convention)
# These names indicate positions on the scalp (e.g., FCC1h = fronto-central-central, position 1, halfway)
source_names = ['FCC1h', 'FFC3h', 'FCC5h', 'FFT7h', 'CPP1h', 'CCP3h', 'CPP5h', 'TTP7h']
detector_names = ['FFC1h', 'FCC3h', 'FFC5h', 'FTT7h', 'CCP1h', 'CPP3h', 'CCP5h', 'TPP7h']


def load_raw_fnirs_data(subject_name, dataset_name):
    """
    Loads raw fNIRS data files for a given subject from disk.
    Handles special cases for subjects 1, 5, and 9 which have different file structures or missing data.
    
    Args:
        subject_name: String identifier for the subject (e.g., '1', '2')
        dataset_name: Name of the dataset being processed
        
    Returns:
        raw_file_train: MNE Raw object containing concatenated training data from all blocks
        raw_file_test: MNE Raw object for test data (None for this dataset)
    """

    if dataset_name == "FineMI":
        n_blocks = 8  # Each subject has 8 experimental blocks (sessions)
        raw_file_train_list = []  # List to store individual block data before concatenation

        print("Subject: ", subject_name)
        # Special handling for subject 1: blocks 1-4 are stored in a combined file
        if subject_name == '1':
            # subject 1 session 1
            # Block1-4: Load combined file for first 4 blocks
            file_name = "./Data/FineMI/subject" + subject_name + "/fNIRS/block1-4"
            # Read NIRx format fNIRS data and preload into memory (preload=True loads all data immediately)
            raw_file = read_raw_nirx(file_name, preload=True)
            # Remove last 40 annotations (likely artifacts or unwanted events at end of recording)
            # np.arange(-40, 0) creates array [-40, -39, ..., -1] representing last 40 indices
            idx_to_remove = np.arange(-40, 0)
            raw_file.annotations.delete(idx_to_remove)
            # Crop the raw data to only include periods with annotations (removes leading/trailing data without events)
            raw_file.crop_by_annotations()
            raw_file_train_list.append(raw_file)

            # Block5-8: Load individual files for blocks 5 through 8
            for block_idx in range(4, n_blocks):  # range(4, 8) = [4, 5, 6, 7]
                file_name = "./Data/FineMI/subject" + subject_name + "/fNIRS/block" + str(block_idx + 1)
                raw_file = read_raw_nirx(file_name, preload=True)
                raw_file_train_list.append(raw_file)
        elif subject_name == '9':
            # Subject 9's fNIRS data is missing from the dataset
            print('subject 9 fNIRS data is not available')
            return None, None
        else:
            # Standard case: all other subjects have individual block files (blocks 1-8)
            for block_idx in range(n_blocks):  # range(8) = [0, 1, 2, 3, 4, 5, 6, 7]
                file_name = "./Data/FineMI/subject" + subject_name + "/fNIRS/block" + str(block_idx + 1)
                raw_file = read_raw_nirx(file_name, preload=True)
                # Special fix for subject 5, block 6: remove first annotation (likely artifact or bad marker)
                if subject_name == "5" and block_idx == 5:
                    raw_file.annotations.delete(0)  # Delete annotation at index 0
                    raw_file.crop_by_annotations()
                raw_file_train_list.append(raw_file)

        # Concatenate all blocks into a single continuous Raw object (combines all 8 blocks end-to-end)
        raw_file_train = concatenate_raws(raw_file_train_list)
        raw_file_test = None  # This dataset doesn't have separate test data
        return raw_file_train, raw_file_test
    else:
        print("Unknown Dataset!")
        return None, None


def extract_epoch(raw_file, event_id, picks, tmin, tmax):
    """
    Extracts time-locked epochs from raw fNIRS data.
    An epoch is a time window around each event (e.g., motor imagery cue onset).
    This function segments the continuous data into these windows for analysis.
    
    Args:
        raw_file: MNE Raw object containing continuous fNIRS data
        event_id: Dictionary mapping annotation labels to event IDs (e.g., {"1.0": 1})
        picks: Which channels to include (None = all channels)
        tmin: Start time of epoch relative to event (seconds, negative = before event)
        tmax: End time of epoch relative to event (seconds, positive = after event)
        
    Returns:
        epochs: MNE Epochs object containing all extracted time windows
    """
    # Extract event markers from annotations in the raw file
    # events is a numpy array with shape (n_events, 3): [sample_number, 0, event_id]
    # The underscore _ discards the event_id mapping dictionary (we already have it)
    events, _ = events_from_annotations(raw_file, event_id=event_id)
    # Create Epochs object: segments data into time windows around each event
    # proj=True: apply projection vectors (for artifact removal, e.g., removing bad channels)
    # baseline=(-3.9, -2.): use time window from -3.9 to -2 seconds as baseline for correction
    #   This baseline period is subtracted from the signal to remove drift and normalize
    # preload=True: load all epoch data into memory for faster access (uses more RAM but faster)
    # verbose=False: suppress progress messages
    epochs = Epochs(raw_file, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=(-3.9, -2.), preload=True, verbose=False)
    return epochs


def get_data_from_raw_file(raw_file_train, raw_file_test, subject, dataset_name, picks_train, picks_test, tmin, tmax,
                           data_type="fNIRS"):
    """
    Extracts and processes epoch data from raw fNIRS files.
    This function:
    1. Extracts epochs (time windows) around events
    2. Converts epochs to numpy arrays for analysis
    3. Extracts labels (which motor imagery task was performed)
    4. Saves epochs to disk for later use
    
    Args:
        raw_file_train: MNE Raw object with training data
        raw_file_test: MNE Raw object with test data (unused here)
        subject: Subject identifier
        dataset_name: Name of dataset
        picks_train: Which channels to select (None = all)
        picks_test: Which channels to select for test (unused)
        tmin: Start time for epochs
        tmax: End time for epochs
        data_type: Type of data being processed ("fNIRS")
        
    Returns:
        train_data: numpy array of shape (n_epochs, n_channels, n_timepoints)
        train_labels: numpy array of class labels (0-7 for 8 motor imagery tasks)
        X_test: Test data (None here)
        Y_test: Test labels (None here)
        epoch_info: MNE Info object with channel and sampling information
    """

    if dataset_name == "FineMI":
        # Map annotation labels to event IDs
        # Each number corresponds to a different motor imagery task (1-8)
        # The ".0" suffix is how annotations are stored in the raw files
        event_id_train = {
            "1.0": 1,  # Task 1: Hand flexion/extension
            "2.0": 2,  # Task 2: Wrist flexion/extension
            "3.0": 3,  # Task 3: Wrist adduction/abduction
            "4.0": 4,  # Task 4: Elbow pronation/supination
            "5.0": 5,  # Task 5: Elbow flexion/extension
            "6.0": 6,  # Task 6: Shoulder pronation/supination
            "7.0": 7,  # Task 7: Shoulder adduction/abduction
            "8.0": 8,  # Task 8: Shoulder flexion/extension
        }

    # Extract epochs from raw data
    # Handle both single Raw object and list of Raw objects (for flexibility)
    if not type(raw_file_train) == list:
        # Single file: extract epochs directly
        epochs_train = extract_epoch(raw_file_train, event_id_train, picks_train, tmin, tmax)
        epoch_info = epochs_train.info  # Store metadata (channel names, sampling rate, etc.)
    else:
        # Multiple files: extract epochs from each, then combine
        epochs_train = [extract_epoch(raw_file_train_elem, event_id_train, picks_train, tmin, tmax) for
                        raw_file_train_elem in raw_file_train]
        epoch_info = epochs_train[0].info  # Use info from first epoch (should be same for all)

    # Convert epochs to numpy arrays for analysis
    # Shape: (n_epochs, n_channels, n_timepoints)
    # n_epochs = number of trials, n_channels = number of fNIRS channels, n_timepoints = samples in time window
    if not type(epochs_train) == list:
        # Single Epochs object: get data directly
        train_data = epochs_train.get_data()
    else:
        # Multiple Epochs objects: concatenate along epoch dimension (axis 0)
        train_data = np.concatenate(
            [epochs_train_elem.get_data() for epochs_train_elem in epochs_train])

    # Free memory: delete raw file data since we've extracted what we need
    del raw_file_train
    gc.collect()  # Force garbage collection to free memory immediately

    # Save epochs to disk for later use (avoids recomputing during plotting)
    if not os.path.exists("tmp"):
        os.makedirs('tmp')
    # Save in MNE's FIF format (Fast Imaging Format) for efficient storage
    epochs_train.save("tmp/subject" + str(subject) + "_" + data_type + "-epo.fif", overwrite=True)

    # Extract labels: which motor imagery task was performed in each epoch
    # epochs_train.events is array with shape (n_events, 3): [sample, 0, event_id]
    # events[:, -1] gets the event ID (last column) for each event
    # Subtract 1 to convert from 1-8 to 0-7 (Python uses 0-indexed arrays)
    if not type(epochs_train) == list:
        train_labels = epochs_train.events[:, -1] - 1
    else:
        # Multiple epochs: concatenate labels from all epoch objects
        train_labels = np.concatenate([epochs_train_elem.events[:, -1] - 1 for epochs_train_elem in epochs_train])

    # Free memory: delete epochs since we have the data arrays
    del epochs_train
    gc.collect()
    return train_data, train_labels, None, None, epoch_info


##############################################################################

def preprocessing_fnirs(raw_file_train, raw_file_test, subject_session_name, dataset_name, tmin, tmax):
    """
    Preprocesses raw fNIRS data through the standard pipeline:
    1. Convert to optical density (OD) - removes source intensity variations
    2. Convert to hemoglobin concentration (HbO/HbR) using Beer-Lambert law
    3. Apply bandpass filter - removes noise outside hemodynamic response range
    4. Extract epochs - segment data into time windows around events
    
    Args:
        raw_file_train: Raw fNIRS data (light intensity measurements)
        raw_file_test: Test data (unused)
        subject_session_name: Subject identifier
        dataset_name: Dataset name
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
    # Step 1: Convert raw light intensity to optical density (OD)
    # OD = -log(I/I0) where I is measured intensity, I0 is reference intensity
    # This removes the effect of source intensity variations and makes data more comparable
    raw_file_train_od = mne.preprocessing.nirs.optical_density(raw_file_train)

    # Step 2: Convert optical density to hemoglobin concentration using Beer-Lambert law
    # This gives us HbO (oxyhemoglobin) and HbR (deoxyhemoglobin) concentrations in µM (micromolar)
    # The Beer-Lambert law relates light absorption to chromophore concentration
    # HbO and HbR have different absorption spectra, allowing separation using multiple wavelengths
    raw_file_train_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_file_train_od)

    # Step 3: Configure bandpass filter parameters
    # IIR (Infinite Impulse Response) filter for frequency filtering
    iir_params = {
        "output": "sos",  # Second-order sections format (numerically stable, prevents filter instability)
        "order": filter_order_fnirs,  # Filter order (6 = 6th order Butterworth filter)
        "ftype": filter_type_fnirs  # Filter type (Butterworth = smooth frequency response)
    }

    # Apply bandpass filter: 0.01-0.3 Hz
    # This removes:
    # - Low frequencies (<0.01 Hz): slow drifts, physiological noise, movement artifacts
    # - High frequencies (>0.3 Hz): cardiac pulsation (~1 Hz), respiration (~0.2-0.3 Hz), other artifacts
    # The passband (0.01-0.3 Hz) contains the hemodynamic response signal (typically 0.01-0.1 Hz)
    raw_file_train_haemo = raw_file_train_haemo.filter(0.01, 0.3, method='iir', iir_params=iir_params,
                                                       n_jobs=n_jobs)

    # Step 4: Extract epochs and convert to numpy arrays
    # Pass None for picks_train/picks_test to use all channels
    X_train, Y_train, X_test, Y_test, epoch_info = get_data_from_raw_file(raw_file_train_haemo, None,
                                                                          subject_session_name,
                                                                          dataset_name,
                                                                          None, None,  # picks_train, picks_test (None = all channels)
                                                                          tmin,
                                                                          tmax, data_type="fNIRS")
    frequency_list = []  # fNIRS doesn't use frequency bands like EEG (EEG analyzes different frequency bands)
    return X_train, Y_train, X_test, Y_test, frequency_list, epoch_info


def plot_fNIRS(epochs, subject, plot_by_class=False, data_type="fNIRS"):
    """
    Generates visualization plots for fNIRS data.
    Creates two types of plots:
    1. Trend plots: Time-series of HbO/HbR changes for selected channels
    2. Topographic maps: Spatial distribution of HbO at specific time points
    
    Args:
        epochs: MNE Epochs object (unused parameter, data loaded from disk instead)
        subject: Subject identifier string
        plot_by_class: Whether to plot by class (unused, always plots by class)
        data_type: Type of data ("fNIRS")
    """
    # Define the 8 motor imagery tasks with descriptive names for plot labels
    tasks = [
        {
            "joint": "Hand",
            "move": "flexion/extension"
        },
        {
            "joint": "Wrist",
            "move": "flexion/extension"
        },
        {
            "joint": "Wrist",
            "move": "adduction/abduction"
        },
        {
            "joint": "Elbow",
            "move": "pronation/supination"
        },
        {
            "joint": "Elbow",
            "move": "flexion/extension"
        },
        {
            "joint": "Shoulder",
            "move": "pronation/supination"
        },
        {
            "joint": "Shoulder",
            "move": "adduction/abduction"
        },
        {
            "joint": "Shoulder",
            "move": "flexion/extension"
        }
    ]
    n_classes = len(tasks)  # 8 motor imagery tasks
    # Create output directory for this subject's plots
    dir_path = "img/fnirs/subject" + subject
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f'Created directory: {dir_path}')

    # Compute or load mean epochs for each task
    # Mean epochs = average across all subjects and all trials of the same task (grand average)
    mean_epoch_dict_list = []
    for c in range(n_classes):  # Loop through each of the 8 tasks
        filename = dir_path + "/" + str(c) + ".pkl"
        if load_mean_epochs:
            # Load pre-computed mean epochs from pickle file (faster if already computed)
            f_read = open(filename, 'rb')
            mean_epoch_dict = pickle.load(f_read)
            f_read.close()
        else:
            # Compute mean epochs by averaging across all subjects
            epochs_list = []
            # Load epochs from all 18 subjects (excluding subject 9 which has no data)
            for subject_name in range(1, 19):
                # Read saved epochs from disk (saved during preprocessing)
                epochs_elem = mne.read_epochs("tmp/subject" + str(subject_name) + "_" + data_type + "-epo.fif",
                                              preload=True)

                # Select only epochs for the current task (class c+1, since classes are 1-8 but c is 0-7)
                # epochs_elem[str(c + 1) + ".0"] selects epochs with event ID matching task number
                # Copy to avoid modifying original data
                epochs_list.append(epochs_elem[str(c + 1) + ".0"].copy())
                del epochs_elem
                gc.collect()
            # Concatenate epochs from all subjects into one Epochs object
            epochs_all = mne.concatenate_epochs(epochs_list)
            # Compute average across all epochs, separately for HbO and HbR
            # This gives the grand average evoked response for this task across all subjects
            # picks='hbo' selects only oxyhemoglobin channels, picks='hbr' selects deoxyhemoglobin channels
            mean_epoch_dict = {'HbO': epochs_all.average(picks='hbo'), 'HbR': epochs_all.average(picks='hbr')}
            # Rename channels: remove the '_hbo' or '_hbr' suffix for cleaner plotting
            # x[:-4] removes last 4 characters (e.g., "S1_D1_hbo" becomes "S1_D1")
            for condition in mean_epoch_dict:
                mean_epoch_dict[condition].rename_channels(lambda x: x[:-4])
            # Save computed mean epochs for future use (avoids recomputing)
            f_save = open(filename, 'wb')
            pickle.dump(mean_epoch_dict, f_save)
            f_save.close()
            del epochs_list
            del epochs_all
            gc.collect()
        mean_epoch_dict_list.append(mean_epoch_dict)

    # Generate trend plots: time-series of HbO/HbR changes over time
    if plot_trends:
        for c in range(n_classes):  # For each of the 8 tasks

            # Color scheme: HbO in magenta (#AA3377), HbR in blue ('b')
            color_dict = dict(HbO='#AA3377', HbR='b')

            # Select 4 channels around C3 (motor cortex area, primary motor area for hand movement)
            # These channels are in the region expected to show motor imagery activation
            # C3 is a standard EEG location over the left motor cortex
            selected_fNIRS_channels = ["CCP3h-FCC3h", "CCP3h-CCP5h", "FCC5h-CCP5h", "FCC5h-FCC3h"]
            picks_fNIRS = []
            # Convert channel names to MNE format (S1_D1, S2_D3, etc.)
            for selected_fNIRS_channel in selected_fNIRS_channels:
                S = selected_fNIRS_channel.split("-")[0]  # Source name (e.g., "CCP3h")
                D = selected_fNIRS_channel.split("-")[1]  # Detector name (e.g., "FCC3h")
                # Find index of source and detector in their respective lists
                S_idx = source_names.index(S)
                D_idx = detector_names.index(D)
                # Create MNE channel name format: "S1_D1" means source 1, detector 1
                # Add 1 because indices are 0-based but MNE uses 1-based numbering
                picks_fNIRS.append("S" + str(S_idx + 1) + "_" + "D" + str(D_idx + 1))

            # Create figure for trend plot (1 row, 1 column, 10x7 inches)
            fig_trend, axes_trend = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

            # Plot evoked responses: shows average HbO and HbR over time
            # combine="mean": average across the selected channels (shows overall response in motor cortex region)
            # This shows the overall hemodynamic response in the motor cortex region
            mne.viz.plot_compare_evokeds(mean_epoch_dict_list[c], combine="mean", picks=picks_fNIRS,
                                         axes=axes_trend, show=False,  # show=False prevents automatic display
                                         truncate_xaxis=False, truncate_yaxis=False, legend="lower right",
                                         ylim=dict(hbo=[-0.05, 0.15], hbr=[-0.05, 0.15]),  # Y-axis limits in µM (micromolar)
                                         colors=color_dict,
                                         title=tasks[c]["joint"] + " " + tasks[c]["move"])

            # Customize plot appearance for publication quality
            axes_trend.set_title(tasks[c]["joint"] + " " + tasks[c]["move"],
                                 fontdict={'fontsize': 20, 'fontweight': 'bold'})
            axes_trend.legend(["HbO", "HbR"], fontsize=24, loc="upper right")
            legend = axes_trend.get_legend()
            # Make legend text bold
            for text in legend.get_texts():
                text.set_fontweight('extra bold')
            # Make legend lines thicker (4.0 points)
            for line in legend.get_lines():
                line.set_linewidth(4.0)

            # Format axes: larger tick labels for readability
            axes_trend.tick_params(labelsize=24)
            axes_trend.set_xlabel("Times (s)", fontsize=24, fontweight='bold')
            axes_trend.set_ylabel("µM", fontsize=24, fontweight='bold')  # Micromolar units (concentration)
            # Make plot lines thicker for visibility
            for line in axes_trend.get_lines():
                line.set_linewidth(4.0)

            # Make tick labels bold
            for tick in axes_trend.get_xticklabels():
                tick.set_fontweight('bold')

            for tick in axes_trend.get_yticklabels():
                tick.set_fontweight('bold')

            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            fig_trend.show()
            # Save high-resolution figure (300 DPI for publication quality)
            fig_trend.savefig("img/fnirs/subject" + subject + "/trend_4_channel_around_C3_" + str(c) + ".png",
                              dpi=300)

    # Generate topographic maps: spatial distribution of activation across the scalp
    if plot_topo:
        for c in range(n_classes):  # For each of the 8 tasks
            times = 9.  # Time point to visualize (9 seconds after cue, in the middle of motor imagery period, 8-10s window)
            half_time = 1.  # Average over ±1 second around the time point (8-10s window) to reduce noise
            # Topomap settings: extrapolate to areas without sensors, use red-blue colormap (RdBu_r = reversed)
            # Red = positive (activation/increase), Blue = negative (deactivation/decrease)
            topomap_args = dict(extrapolate='local', cmap="RdBu_r")
            mean_epoch = mean_epoch_dict_list[c]
            # Value limits for colorbar: -0.2 to 0.2 µM (micromolar)
            vlim_hbo = (-0.2, 0.2)

            # Create figure with 2 subplots (for potential left/right hemisphere or different views)
            fig_topo_hbo, axes_topo_hbo = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

            # Plot topographic map: shows spatial distribution of HbO at the specified time
            # This visualizes which brain regions are most activated during the task
            # ch_type='hbo' specifies we're plotting oxyhemoglobin
            mean_epoch["HbO"].plot_topomap(ch_type='hbo', times=times, colorbar=True,
                                           axes=axes_topo_hbo,
                                           average=half_time,  # Average over time window to reduce noise
                                           vlim=vlim_hbo, **topomap_args)

            # Customize colorbar appearance
            cbar1 = axes_topo_hbo[0].images[0].colorbar

            # Format colorbar numbers to 2 decimal places
            formatter = FormatStrFormatter('%.2f')
            cbar1.ax.yaxis.set_major_formatter(formatter)
            cbar1.ax.title.set_size(20)
            cbar1.ax.tick_params(labelsize=20)
            axes_topo_hbo[0].set_title(tasks[c]["joint"] + " " + tasks[c]["move"] + "(HbO)", fontdict={'fontsize': 20})
            fig_topo_hbo.tight_layout()

            # Adjust position of second subplot (likely to make room for colorbar or adjust layout)
            # Get position bounds: left, bottom, width, height
            ll, bb, ww, hh = axes_topo_hbo[-1].get_position().bounds
            # Resize second subplot to 5% of original width (makes it very narrow, possibly for colorbar)
            axes_topo_hbo[-1].set_position([ll, bb, ww * 0.05, hh])

            plt.tight_layout()
            # Save high-resolution topographic map
            fig_topo_hbo.savefig("img/fnirs/subject" + subject + "/topo_hbo_" + str(c) + ".png", dpi=300)

    return None


if __name__ == "__main__":
    # Main execution: process all subjects and generate plots
    data_type = "fNIRS"
    # Loop through all subjects (1-18, skipping subject 9 which has no data)
    for subject in range(start_subject, 19):
        if subject == 9:
            continue  # Skip subject 9 (data not available)
        
        subject_session_name = str(subject)

        # Step 1: Load raw fNIRS data files from disk
        raw_file_train, raw_file_test = load_raw_fnirs_data(subject_session_name,
                                                            dataset_name)

        # Skip processing if data loading failed (e.g., subject 9)
        if raw_file_train is None:
            continue

        # Step 2: Preprocess data (OD conversion, Beer-Lambert, filtering, epoch extraction)
        # This converts raw light intensity → optical density → hemoglobin concentration → filtered → epochs
        X_train, Y_train, X_test, Y_test, frequency_list, epoch_info = preprocessing_fnirs(raw_file_train,
                                                                                           raw_file_test,
                                                                                           subject_session_name,
                                                                                           dataset_name, tmin,
                                                                                           tmax)

        print("\nLoading data of subject %d...  done.\n" % subject)
    
    # Step 3: Generate plots for all subjects
    # Create list of all subject names (1-18, excluding 9)
    subjectNames = [str(n) for n in range(1, 19) if n != 9]
    # Generate trend plots and topographic maps for each subject
    # Note: This uses the saved epoch files from the preprocessing step above
    for name in subjectNames:
        plot_fNIRS(None, name, plot_by_class, data_type)
