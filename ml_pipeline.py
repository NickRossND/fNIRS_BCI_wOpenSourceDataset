# ============================================================================
# MACHINE LEARNING PIPELINE
# ============================================================================
# Main processing function that orchestrates the entire ML pipeline

import numpy as np
import pandas as pd
import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

from config import (
    dataset_name, class_of_interest, use_hbr, preprocessing_params,
    use_models, params, scenario_name, K, valid_size, seed, num_subjects,
    start_subject, end_subject, target_subject
)
from preprocessing import preprocessing_fnirs_func
from features import Mean, Slope
from cross_validation import cross_validation
from utils import mkdir
import os


def intrasubject_tests(dataset, dir_datetime_mark=None, datetime_mark=None):
    """
    Main function: process all subjects, train models, and evaluate performance.
    
    This is the orchestrator function that:
    1. Loads data for each subject
    2. Preprocesses fNIRS data (OD → Beer-Lambert → filter → epochs)
    3. Extracts features (Mean, Slope)
    4. Trains SVM classifier
    5. Evaluates using cross-validation
    6. Saves results to CSV files
    
    The function processes each subject independently (intra-subject analysis),
    meaning each subject's data is used to train and test a model for that subject only.
    
    Args:
        dataset: FineMI dataset object with configuration
        dir_datetime_mark: Optional directory name for results (if None, uses timestamp)
        datetime_mark: Optional timestamp for results (if None, generates new one)
        
    Returns:
        dir_datetime_mark: Directory name where results were saved
    """
    # Print configuration information for logging/reproducibility
    print("\n###############################################################################\n")
    datetime_mark = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(datetime_mark)
    print("Seed: ", seed)
    print("Dataset: ", dataset_name)
    print("Class of intereset: ", class_of_interest)
    print("Number of subjects: ", num_subjects)
    if num_subjects == 1:
        print("Target subject: ", target_subject)
    else: 
        print("Start subject: ", start_subject)
        print("End subject: ", end_subject)
    print("Tmin: ", str(dataset.tmin))
    print("Tmax: ", str(dataset.tmax))
    model_used = []
    for key in use_models:
        if use_models[key]:
            model_used.append(key)
    print("Model used:", str(model_used))
    print("Scenario: ", scenario_name)
    if "cross_validation" in scenario_name:
        print("K: ", K)

    print("------------------------------------------------------------------")
    print("Preprocessing settings: \n")
    print("Use band-pass filtering: ", preprocessing_params["use_band_pass"])

    # Build list of model names to use (extract from use_models dictionary)
    model_names = []
    model_index = dict()  # Maps model name to index in results array (for storing results)
    index = 0

    for model in use_models:
        if use_models[model]:
            model_names.append(model)
            model_index[model] = index
            index += 1

    print("------------------------------------------------------------------")
    print("valid size: ", valid_size)
    print("------------------------------------------------------------------")

    # Initialize arrays to store accuracies for all subjects and models
    # Shape: (num_subjects + 1, n_models) - extra row for average across subjects
    num_sessions_per_subject = dataset.num_sessions_per_subject
    train_acc_array = np.zeros((num_subjects * num_sessions_per_subject + 1, len(model_names)))  # Accuracy array of the training set. Dimension: num_subjects* n_sessions_in_1_subject + 1 (Average)
    valid_acc_array = np.zeros((num_subjects * num_sessions_per_subject + 1, len(model_names)))  # Accuracy array of the validation set. Dimension: num_subjects* n_sessions_in_1_subject + 1 (Average)
    test_acc_array = np.zeros((num_subjects * num_sessions_per_subject + 1, len(model_names)))  # Accuracy array of the test set. Dimension: num_subjects* n_sessions_in_1_subject + 1 (Average)

    subject_session_names = []  # List to store subject-session identifiers for results table
    trained_models = {} # store trained models for later use
    # Process each subject independently (intra-subject analysis)
    for subject_idx in range(start_subject - 1, end_subject):
        
        print("\n###############################################################################\n")
        
        # Handle case where only one subject is processed
        if num_subjects == 1:  # Run on 1 specified subject
            subject_idx = target_subject - 1
        subject_name = str(subject_idx + 1)
        # if subject_name == '9':
        #     continue
        # Process each session (usually just 1 session per subject for this dataset)
        n_sessions = dataset.num_sessions_per_subject if dataset.num_sessions_per_subject is not None else 1
        for session_idx in range(n_sessions):

            # Create subject-session identifier (e.g., "3s1" = subject 3, session 1)
            subject_session_name = subject_name + "s" + str(session_idx + 1)
            subject_session_names.append(subject_session_name)  # Record for results table

            print("Loading data of subject %s... " % subject_session_name)

            # Load raw data files for this subject from disk
            # This reads all 8 blocks and concatenates them into a single Raw object
            dataset.load([subject_idx + 1], session_list=[session_idx + 1])
            print("Loading data of subject %s... Done." % subject_session_name)

            Y_train_list = []  # Store labels for each model (usually same across models)
            
            # Process each model (usually just one model in this script)
            for model_name in model_names:
                X_train_list = []  # Store preprocessed data
                info_list = []  # Store metadata

                # Get fNIRS data for this subject (stored in dataset object after load())
                raw_data = dataset.raw_data_list[0]

                # Handle both single file and list of files (for flexibility)
                if type(raw_data) != list:
                    raw_data = [raw_data]
                
                X_fnirs_train = []  # List to store preprocessed data from each file
                Y_fnirs_train = []  # List to store labels from each file
                info_fnirs = []
                
                # Preprocess each raw file (usually just one concatenated file per subject)
                for raw_file_idx, raw_file in enumerate(raw_data):
                    # Apply preprocessing pipeline: OD → Beer-Lambert → filter → epochs
                    # This converts raw light intensity to machine-learning-ready data
                    X_fnirs_train_elem, Y_fnirs_train_elem, info_fnirs = preprocessing_fnirs_func(
                        raw_file,
                        dataset,
                        preprocessing_params,
                        subject_idx=subject_idx
                    )

                    # Separate HbO and HbR channels
                    # fNIRS data has interleaved channels: [HbO_ch1, HbR_ch1, HbO_ch2, HbR_ch2, ...]
                    # [:, ::2, :] selects every other channel starting from 0 (HbO channels: 0, 2, 4, ...)
                    # [:, 1::2, :] selects every other channel starting from 1 (HbR channels: 1, 3, 5, ...)
                    X_hbo_train_elem = X_fnirs_train_elem[:, ::2, :].copy()
                    X_hbr_train_elem = X_fnirs_train_elem[:, 1::2, :].copy()

                    # Combine HbO and HbR if use_hbr is True
                    if use_hbr:
                        # Concatenate along channel dimension: [all_HbO_channels, all_HbR_channels]
                        # This doubles the number of channels (e.g., 24 channels → 48 channels)
                        X_fnirs_train_elem = np.concatenate([X_hbo_train_elem, X_hbr_train_elem], axis=1)
                    else:
                        # Use only HbO channels (faster, but may lose information)
                        X_fnirs_train_elem = X_hbo_train_elem
                    
                    X_fnirs_train.append(X_fnirs_train_elem)
                    Y_fnirs_train.append(Y_fnirs_train_elem)
                
                # Concatenate data from all files (if multiple)
                X_fnirs_train = np.concatenate(X_fnirs_train)
                Y_fnirs_train = np.concatenate(Y_fnirs_train)

                X_train_list.append(X_fnirs_train)
                info_list.append(info_fnirs)
                Y_train = Y_fnirs_train.copy()

                assert (len(X_train_list) > 0)  # Ensure we have at least one data source

                # Get final training data (handle single vs multiple data types)
                X_train = X_train_list if len(X_train_list) > 1 else X_train_list[0]

                print("Preprocessing data of subject %s... Done." % subject_session_name)

                Y_train_list.append(Y_train)

                print("Building %s model of subject %s with scenario %s..." % (
                    model_name, subject_session_name, scenario_name))

                # Build machine learning pipeline
                # Pipeline steps:
                # 1. Feature extraction: Mean + Slope (in parallel via FeatureUnion)
                # 2. Standardization: Z-score normalization
                # 3. Classification: SVM
                
                n_times = X_train.shape[-1]  # Number of time points per epoch
                mean = Mean(axis=-1)  # Compute mean across time dimension (reduces n_times → 1)
                slope = Slope(n_times)  # Compute slope across time dimension (reduces n_times → 1)
                ss = StandardScaler()  # Z-score normalization (zero mean, unit variance)
                svm = SVC(kernel=params["SVM"]["kernel"], C=params["SVM"]["svm_C"])  # Linear SVM classifier
                
                # FeatureUnion combines Mean and Slope features in parallel
                # Output: concatenated features [mean_features, slope_features]
                # If we have 24 channels, this gives us 24 mean features + 24 slope features = 48 total features
                union = FeatureUnion([("mean", mean), ("slope", slope)])
                
                # Pipeline chains: feature extraction → normalization → classification
                # This ensures features are extracted, normalized, then classified in sequence
                model = Pipeline([("feature_union", union), ("standard_scale", ss), ('svm', svm)])
                
                # Perform cross-validation to evaluate model performance
                # This splits data into K folds, trains on K-1 folds, tests on 1 fold, repeats K times
                train_acc_elem, valid_acc_elem, test_acc_elem = cross_validation(model, X_train, Y_train, K=K,
                                                                                 subject_session_name=subject_session_name,
                                                                                 model_name=model_name,
                                                                                 valid_size=valid_size,
                                                                                 seed=seed)

                # Store results in arrays
                if num_subjects == 1:
                    subject_idx = 0  # Adjust index for single-subject case
                subject_session_idx = subject_idx * num_sessions_per_subject + session_idx
                train_acc_array[subject_session_idx][model_index[model_name]] = train_acc_elem
                valid_acc_array[subject_session_idx][model_index[model_name]] = valid_acc_elem
                test_acc_array[subject_session_idx][model_index[model_name]] = test_acc_elem
                print("Building %s model of subject %s with scenario %s. Done." % (
                    model_name, subject_session_name, scenario_name))

                # train final model on all data for this subject
                model.fit(X_train, Y_train)  
                subj_id  = subject_idx + 1
                sess_id = session_idx + 1
                trained_models[(subj_id, sess_id, model_name)] = model
    # Calculate average accuracies across all subjects (stored in last row of arrays)
    # This gives us the overall performance across all subjects
    train_acc_array[-1] = train_acc_array[:-1].mean(axis=0)
    if valid_size > 0:
        valid_acc_array[-1] = valid_acc_array[:-1].mean(axis=0)
    test_acc_array[-1] = test_acc_array[:-1].mean(axis=0)
    mean_test_acc = test_acc_array[-1]  # Overall mean test accuracy

    # Print results summary
    print("\ntrain_acc_array:", train_acc_array)
    print("\nvalid_acc_array:", valid_acc_array)
    print("\ntest_acc_array:", test_acc_array)
    print("\nMean test score: ", mean_test_acc)

    # Format results for CSV export
    # Convert numpy arrays to formatted strings (4 decimal places)
    result_train = []
    if valid_size > 0:
        result_valid = []
    result_test_acc = []

    subject_names = subject_session_names.copy()  # Copy to avoid modifying original list
    subject_names.append("Average")  # Add row label for average across subjects

    # Convert each accuracy value to formatted string
    for i in range(len(test_acc_array)):
        result_train.append([])
        if valid_size > 0:
            result_valid.append([])
        result_test_acc.append([])

        for j in range(len(test_acc_array[i])):
            # Format to 4 decimal places (e.g., "0.8523")
            result_train[i].append("%.4f" % train_acc_array[i][j])
            if valid_size > 0:
                result_valid[i].append("%.4f" % valid_acc_array[i][j])
            result_test_acc[i].append("%.4f" % test_acc_array[i][j])

    # Note: Subject 9 is skipped during processing (see line with "if subject_name == '9': continue")
    # So it never gets added to subject_session_names, and no deletion is needed here
    
    # Create pandas DataFrames for easy CSV export
    # Rows = subjects (plus "Average"), Columns = model names
    pdf_result_train = pd.DataFrame(result_train, index=subject_names, columns=model_names)
    if valid_size > 0:
        pdf_result_valid = pd.DataFrame(result_valid, index=subject_names, columns=model_names)
    pdf_result_test_acc = pd.DataFrame(result_test_acc, index=subject_names, columns=model_names)

    # Save results to CSV files
    if dir_datetime_mark is None:
        dir_datetime_mark = datetime_mark
    dir_path = "results/" + dir_datetime_mark
    if not os.path.exists(dir_path):
        mkdir(dir_path)  # Create results directory if it doesn't exist
    
    # Save CSV files with timestamp and dataset name in filename
    pdf_result_train.to_csv(dir_path + "/" + datetime_mark + "_dataset=" + dataset_name + "__train.csv")
    if valid_size > 0:
        pdf_result_valid.to_csv(dir_path + "/" + datetime_mark + "_dataset=" + dataset_name + "__valid.csv")
    pdf_result_test_acc.to_csv(dir_path + "/" + datetime_mark + "_dataset=" + dataset_name + "__test_acc.csv")

    return dir_datetime_mark, trained_models

