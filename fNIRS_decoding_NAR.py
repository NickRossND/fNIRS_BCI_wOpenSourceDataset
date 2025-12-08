# Before running this script, please ensure that the dataset was downloaded and saved in the "./Data/FineMI" directory

import time
import mne
import numpy as np
import random

# Import configuration
from config import (
    dataset_name, class_of_interest, data_type, preprocessing_params,
    use_models, params, scenario_name, K, valid_size, seed, n_jobs,
    start_subject, n_subjects, end_subject, target_subject
)

# Import modules
from dataset import FineMI
from ml_pipeline import test_on_each_subject

# Set MNE logging level to WARNING to suppress INFO and DEBUG messages
mne.set_log_level(verbose="WARNING")

# Set random seeds for reproducibility
np.random.seed(seed)
random.seed(seed)

# Initialize dataset object with specific time windows for epoch extraction
# tmin=3, tmax=7 means extract epochs from 3s to 7s after the motor imagery cue
# This captures the peak of the hemodynamic response (which occurs 4-6s after neural activation)
# baseline_tmin=-4, baseline_tmax=-2 means use -4s to -2s as baseline period (before the cue)
dataset = FineMI(tmin=-4, tmax=14, baseline_tmin=-4, baseline_tmax=-2, class_of_interest=class_of_interest)


if __name__ == "__main__":
    """
    Main execution block: runs the complete pipeline when script is executed directly.
    
    This script performs:
    1. Data loading for all subjects
    2. fNIRS preprocessing (OD → Beer-Lambert → filter → epochs)
    3. Feature extraction (Mean, Slope)
    4. Machine learning (SVM classification)
    5. Cross-validation evaluation
    6. Results saving to CSV files
    
    The script processes each subject independently, training and testing a model
    for each subject using only that subject's data (intra-subject analysis).
    """
    start_time = time.time()  # Record start time for performance measurement
    print("\n###############################################################################\n")
    print("Run one time")
    
    # Execute main processing function
    # This processes all subjects, trains models, evaluates performance, and saves results
    test_on_each_subject(dataset)
    
    end_time = time.time()  # Record end time
    time_cost = end_time - start_time  # Calculate total execution time
    print("\nRun time: %f (s)." % time_cost)  # Print execution time in seconds
