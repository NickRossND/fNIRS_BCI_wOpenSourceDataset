# Before running this script, please ensure that the dataset was downloaded and saved in the "./Data/FineMI" directory

import time
import mne
import numpy as np
import random

# Import configuration
from config import class_of_interest,use_models,seed

# Import modules
from dataset import FineMI
from ml_pipeline import intrasubject_tests
from prediction import predict_session_realtime
from bci_real_time_predictions import predict_session_realtime_with_timeline, binarize_timeline

# Set MNE logging level to WARNING to suppress INFO and DEBUG messages
mne.set_log_level(verbose="WARNING")

# Set random seeds for reproducibility
np.random.seed(seed)
random.seed(seed)

# Initialize dataset object with specific time windows for epoch extraction
# tmin=3, tmax=7 means extract epochs from 3s to 7s after the motor imagery cue
# This captures the peak of the hemodynamic response (which occurs 4-6s after neural activation)
# baseline_tmin=-4, baseline_tmax=-2 means use -4s to -2s as baseline period (before the cue)
dataset = FineMI(tmin=-4, tmax=10, baseline_tmin=-4, baseline_tmax=-2, class_of_interest=class_of_interest)


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
    dir_mark, trained_models = intrasubject_tests(dataset)

    #pick a subject/session + model to test "online"
    subj_id = 5     # 1-based
    sess_id = 8     # 1-based
    model_name = list(use_models.keys())[0]  # e.g., "ML:fNIRS_Union:Mean;Slope+Zscore+SVM"
    
    model = trained_models[(subj_id, sess_id, model_name)]
    y_pred, y_true, info = predict_session_realtime(
        model=model,
        dataset=dataset,
        subject_idx=subj_id - 1,  # convert to 0-based index
        session_idx=sess_id - 1    # convert to 0-based index
    )
    print("First 10 predictions:", y_pred[:10])
    print("First 10 true labels:", y_true[:10])

    # # If you want to mimic real-time, loop over epochs:
    # for i, epoch in enumerate(y_pred):
    #     print(f"Epoch {i}: predicted label = {epoch}")
    
    y_pred, y_true, timeline, time_axis = predict_session_realtime_with_timeline(model,dataset,subj_id-1,sess_id-1)
    binary_timeline = binarize_timeline(timeline, task_label=1)

    end_time = time.time()  # Record end time
    time_cost = end_time - start_time  # Calculate total execution time
    print("\nRun time: %f (s)." % time_cost)  # Print execution time in seconds
