# ============================================================================
# DATASET CLASS
# ============================================================================
# FineMI dataset class for handling data loading and configuration

import numpy as np
from mne.io import concatenate_raws, read_raw_nirx


class FineMI():
    """
    Dataset class for FineMI motor imagery dataset.
    This class encapsulates all configuration, metadata, and data loading functionality for the FineMI dataset.
    It handles fNIRS data, manages subject-specific configurations, and stores preprocessing parameters.
    """
    def __init__(self, tmin=0, tmax=4, baseline_tmin=-2, baseline_tmax=0, class_of_interest=None, down_sample=False,
                 down_sample_rate=250, resample_fnirs=False, resample_rate_fnirs=250):
        """
        Initialize FineMI dataset configuration.
        
        Args:
            tmin: Start time of epoch relative to event (seconds) - when to start extracting data after motor imagery cue
            tmax: End time of epoch relative to event (seconds) - when to stop extracting data
            baseline_tmin: Start of baseline period for correction (negative = before event, e.g., -4s = 4s before cue)
            baseline_tmax: End of baseline period for correction (e.g., -2s = 2s before cue)
            class_of_interest: List of task IDs to classify (e.g., ["1", "7"] for tasks 1 and 7)
            down_sample: Whether to downsample EEG data (reduces sampling rate, saves memory)
            down_sample_rate: Target sampling rate for downsampling (Hz)
            resample_fnirs: Whether to resample fNIRS data
            resample_rate_fnirs: Target sampling rate for fNIRS resampling (Hz)
        """

        self.name = "FineMI"

        # Default to tasks 1 and 5 if no specific tasks specified
        if class_of_interest is None:
            class_of_interest = ["1", "5"]

        # Initialize data storage lists (will be populated when load() is called)
        self.subject_list = []  # List of subject IDs being processed
        self.use_all_subject_sessions = True  # Flag for using all sessions (vs. specific ones)
        self.subject_session_names_included = []  # List of subject-session identifiers

        self.exclude_trials = []  # List of trials to exclude (e.g., artifacts, bad trials)

        # Lists to store raw data files (one per subject)
        self.raw_file_fnirs_list = []  # For fNIRS data (MNE Raw objects)

        # Processed data storage (not used in this script, but available for other purposes)
        self.data_train = []
        self.label_train = []

        # Time window parameters for epoch extraction
        self.tmin = tmin  # Epoch start time (e.g., 3s after cue = start extracting at 3 seconds)
        self.tmax = tmax  # Epoch end time (e.g., 7s after cue = stop extracting at 7 seconds)
        # So with tmin=3, tmax=7, we extract a 4-second window from 3s to 7s after the motor imagery cue

        # Baseline correction parameters
        self.baseline_tmin = baseline_tmin  # Baseline start (e.g., -4s = 4s before cue)
        self.baseline_tmax = baseline_tmax  # Baseline end (e.g., -2s = 2s before cue)
        # Baseline correction subtracts the average signal during baseline period from the task period
        # This removes slow drifts and normalizes the signal

        # Resampling parameters
        self.resample_fnirs = resample_fnirs  # Whether to resample fNIRS
        self.resample_rate_fnirs = resample_rate_fnirs  # Target sampling rate for fNIRS resampling

        # Dataset metadata - fixed properties of the FineMI dataset
        self.sample_rate_fnirs = 7.8125  # fNIRS sampling rate (Hz) - much lower than EEG (typical for fNIRS)
        self.n_subjects = 18  # Total number of subjects in dataset
        self.n_sessions_per_subject = 1  # Number of recording sessions per subject (1 = single session per subject)
        self.n_classes = 8  # Number of motor imagery tasks in the dataset
        self.n_blocks = 8  # Number of experimental blocks per subject (each block contains multiple trials)
        
        # Define the 8 motor imagery tasks with descriptive names
        # Each task specifies which joint and what type of movement to imagine
        self.tasks = [
            {
                "joint": "Hand",
                "move": "flexion_extension"  # Task 1: Imagine opening/closing hand
            },
            {
                "joint": "Wrist",
                "move": "flexion_extension"  # Task 2: Imagine flexing/extending wrist
            },
            {
                "joint": "Wrist",
                "move": "adduction_abduction"  # Task 3: Imagine moving wrist side-to-side
            },
            {
                "joint": "Elbow",
                "move": "pronation_supination"  # Task 4: Imagine rotating forearm (palm up/down)
            },
            {
                "joint": "Elbow",
                "move": "flexion_extension"  # Task 5: Imagine bending/straightening elbow
            },
            {
                "joint": "Shoulder",
                "move": "pronation_supination"  # Task 6: Imagine rotating arm at shoulder
            },
            {
                "joint": "Shoulder",
                "move": "adduction_abduction"  # Task 7: Imagine moving arm away from/toward body
            },
            {
                "joint": "Shoulder",
                "move": "flexion_extension"  # Task 8: Imagine raising/lowering arm
            }
        ]

        # Create class names by combining joint and movement (e.g., "Hand_flexion_extension")
        class_names = [task["joint"] + "_" + task["move"] for task in self.tasks]
        
        # Filter to only include classes of interest if specified
        # This allows focusing on specific tasks rather than all 8 classes
        if class_of_interest is not None:
            class_names_of_interest = []
            for class_idx, class_name in enumerate(class_names):
                # class_idx is 0-7 (Python 0-indexed), but class_of_interest uses 1-8, so convert to string and add 1
                if str(class_idx + 1) in class_of_interest:
                    class_names_of_interest.append(class_name)
            self.class_names = class_names_of_interest
        else:
            # Use all 8 classes if no specific selection
            self.class_names = class_names

        self.path = ""  # Path to dataset directory (set in load() method)

        # Event ID mappings: maps annotation labels (strings) to numeric event IDs
        self.event_id_fnirs = {}  # For fNIRS data
        
        # Complete mapping of all 8 tasks for fNIRS format (note the ".0" suffix)
        event_id_fnirs = {
            "1.0": 1,  # hand open/close
            "2.0": 2,  # wrist flexion/extension
            "3.0": 3,  # wrist abduction/adduction
            "4.0": 4,  # elbow pronation/supination
            "5.0": 5,  # elbow flexion/extension
            "6.0": 6,  # shoulder pronation/supination
            "7.0": 7,  # shoulder abduction/adduction
            "8.0": 8,  # shoulder flexion/extension
        }
        
        # Filter event IDs to only include classes of interest
        # This ensures we only extract epochs for the tasks we want to classify
        if class_of_interest is not None:
            for c in class_of_interest:
                c = c + ".0"
                # Add to fNIRS event ID mapping (e.g., "1.0" -> 1)
                self.event_id_fnirs[c] = event_id_fnirs[c]
        else:
            # Use all event IDs if no specific selection
            self.event_id_fnirs = event_id_fnirs

    def load(self, subject_list=[1], session_list=[1], path="./Data/FineMI/", data_type=None):
        """
        Load raw data files for specified subjects from disk.
        Handles special cases for subjects 1 and 5 which have different file structures.
        This method reads the raw data files, handles subject-specific quirks, and concatenates blocks.
        
        Args:
            subject_list: List of subject IDs to load (e.g., [1, 2, 3])
            session_list: List of session IDs (usually [1] for this dataset - single session per subject)
            path: Path to dataset directory
            data_type: List of data types to load (e.g., ["fNIRS"] or ["EEG", "fNIRS"])
        """

        assert len(subject_list) > 0, "Use at least one subject!"

        # Reset data storage lists (clear any previous data)
        self.raw_file_train_list = []
        self.raw_file_fnirs_list = []

        # Store configuration
        self.subject_list = subject_list
        self.path = path

        # Process each subject in the list
        for subject in subject_list:

            # Load fNIRS data if requested
            if "fNIRS" in data_type:
                raw_file_list = []  # List to store individual block files before concatenation

                subject_session_name = str(subject)
                print("Subject session: subject", subject_session_name)
                
                # Special handling for subject 1: blocks 1-4 are stored in a combined file
                if subject_session_name == "1":
                    # Subject 1: Block1-4 are in a single combined file
                    file_name = "./Data/FineMI/subject" + subject_session_name + "/fNIRS/block1-4"
                    # Read NIRx format fNIRS data (NIRx is a common fNIRS system manufacturer)
                    raw_file = read_raw_nirx(file_name, preload=True)  # preload=True loads all data into memory

                    # Remove last 40 annotations (artifacts or unwanted events at end of recording)
                    idx_to_remove = np.arange(-40, 0)  # Creates array [-40, -39, ..., -1] representing last 40 indices
                    raw_file.annotations.delete(idx_to_remove)
                    # Crop the raw data to only include periods with annotations
                    raw_file.crop_by_annotations()
                    raw_file_list.append(raw_file)

                    # Block5-8: Load individual files for blocks 5 through 8
                    for block_idx in range(4, self.n_blocks):  # range(4, 8) = [4, 5, 6, 7]
                        file_name = "./Data/FineMI/subject" + subject_session_name + "/fNIRS/block" + str(
                            block_idx + 1)
                        raw_file = read_raw_nirx(file_name, preload=True)
                        raw_file_list.append(raw_file)
                else:
                    # Standard case: all other subjects have individual block files (blocks 1-8)
                    for block_idx in range(self.n_blocks):  # range(8) = [0, 1, 2, 3, 4, 5, 6, 7]
                        file_name = "./Data/FineMI/subject" + subject_session_name + "/fNIRS/block" + str(
                            block_idx + 1)
                        raw_file = read_raw_nirx(file_name, preload=True)
                        # Special fix for subject 5, block 6: remove first annotation (likely artifact or bad marker)
                        if subject_session_name == "5" and block_idx == 5:
                            raw_file.annotations.delete(0)  # Delete annotation at index 0
                            raw_file.crop_by_annotations()
                        raw_file_list.append(raw_file)

                # Concatenate all blocks into one continuous recording
                # This combines all 8 blocks end-to-end into a single Raw object
                raw_file_fnirs = concatenate_raws(raw_file_list)
                self.raw_file_fnirs_list.append(raw_file_fnirs)

