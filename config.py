# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
# This file contains all configuration parameters for the fNIRS decoding pipeline

dataset_name = "FineMI"  # Name of the dataset being processed

# Which motor imagery tasks to classify (1-8, where each number represents a different task)
# ["1", "7"] means we're doing binary classification between task 1 (Hand flexion/extension) and task 7 (Shoulder adduction/abduction)
# This allows focusing on specific tasks rather than all 8 classes
class_of_interest = ["1","2"]

# Whether to use deoxyhemoglobin (HbR) in addition to oxyhemoglobin (HbO)
# True = use both HbO and HbR as features (doubles the number of channels)
# False = use only HbO (faster, but may lose information)
use_hbr = True

# Preprocessing parameters for fNIRS data
preprocessing_params = {
    "use_band_pass": True,  # Apply bandpass filter to remove noise outside hemodynamic response range
    "filter_order": 6,  # Filter order (6th order Butterworth filter - higher = sharper cutoff, but more computation)
    "filter_type": "butter",  # Butterworth filter type (provides flat frequency response in passband)
    "lower_bound": 0.01,  # Lower frequency cutoff (Hz) - removes slow drifts (<0.01 Hz)
    "upper_bound": 0.3,  # Upper frequency cutoff (Hz) - removes cardiac/respiratory artifacts (>0.3 Hz)
    # The passband (0.01-0.1 Hz) contains the hemodynamic response signal
    "moving_average_std": True,  # Apply exponential moving standardization, addtl low freq noise reduction
}

# Which machine learning models to use
# Format: "ML:DataType:FeatureExtraction+Preprocessing+Classifier"
# This example: Mean and Slope features extracted in parallel (Union), Z-score normalization, SVM classifier
use_models = {
    "ML:fNIRS_Union:Mean;Slope+Zscore+SVM": True,
}

# Hyperparameters for the machine learning models
params = {
    "SVM": {
        "kernel": "linear",  # Linear kernel for SVM (good for high-dimensional data, faster than RBF)
        "svm_C": 1  # Regularization parameter (1 = default, higher = less regularization, more complex decision boundary)
    }
}

one_time = True  # Flag for running once

# "intra_subject_cross_validation" means train and test on same subject
# evaluates how well the model generalizes within a single subject's data
scenario_name = "intra_subject_cross_validation"

K = 5  # Number of folds for K-fold cross-validation (5-fold CV: split data into 5 parts, use 4 for training, 1 for testing)
valid_size = 0.2  # Fraction of training data to use for validation (0.2 = 20% of training data held out for model selection)

plot_params = {}  # Parameters for plotting (empty = use defaults, not used in this script)

# Reproducibility settings: set random seeds so results are consistent across runs
seed = 0  # Random seed for reproducibility
n_jobs = 4  # Number of parallel jobs for sklearn operations (4 = use 4 CPU cores)

# Subject processing parameters
start_subject = 2  # First subject to process (1-indexed)
num_subjects = 17  # Total number of subjects in dataset
end_subject = num_subjects  # Last subject to process (18 = process all subjects)
target_subject = 3  # Specific subject to process if num_subjects == 1 (for single-subject analysis)

