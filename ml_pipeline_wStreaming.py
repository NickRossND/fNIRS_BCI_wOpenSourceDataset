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
    start_subject, end_subject, target_subject, n_jobs
)
from preprocessing import preprocessing_fnirs_func
from features import Mean, Slope
from cross_validation import cross_validation
from utils import mkdir
import os
import time as _time


def stream_predict_continuous(model, raw, dataset, window_samples, threshold=0.5,
                              positive_label=None, realtime=True, print_interval_s=1.0,
                              stride=1):
    """
    Stream predictions from an MNE Raw object and print one binary prediction every print_interval_s seconds.

    Args:
        model: trained sklearn pipeline that accepts input shape (n_epochs, n_channels, n_times)
        raw: mne.io.Raw object containing continuous fNIRS data
        dataset: dataset object (used to obtain sample rate if raw.info missing)
        window_samples: number of time samples in each input window (must match training n_times)
        threshold: probability threshold for positive prediction (if predict_proba available)
        positive_label: label corresponding to an activation/task (used when only predict() is available)
        realtime: if True, sleep between predictions to simulate live acquisition
        print_interval_s: seconds between printed outputs (1.0 for once per second)
        stride: samples to advance per loop (1 = every sample considered)
    """
    # Get continuous data and time data
    data = raw.get_data()  # shape: (n_channels, n_total_samples)
    times = raw.times      # shape: (n_total_samples,)

    # samp frequency
    sfreq = None
    try:
        sfreq = raw.info.get('sfreq', None)
    except Exception:
        sfreq = None
    if sfreq is None:
        sfreq = getattr(dataset, 'sampling_rate', 7.8125)

    n_channels, n_total = data.shape

    # print once per print_interval_s secs 
    last_print_time = times[0] - print_interval_s
    pred_list = []
    for idx in range(0, n_total, stride):
        # window length 
        start = idx - window_samples + 1
        if start < 0:
            pad_len = -start
            pad = np.repeat(data[:, :1], pad_len, axis=1)
            window = np.concatenate([pad, data[:, :idx + 1]], axis=1)
        else:
            window = data[:, start: idx + 1]

        # make sure exact window len
        if window.shape[1] != window_samples:
            continue

        X_win = window[np.newaxis, ...]  # shape: (1, n_channels, window_samples)

        # binary prediction
        pred_bin = binary_predictions(model, X_win, threshold=threshold, positive_label=positive_label)[0]
        pred_list.append(pred_bin)
        # # Print only once per print_interval_s seconds
        # if (times[idx] - last_print_time) >= print_interval_s:
        #     print("Time %.3f s -> %d" % (times[idx], int(pred_bin)))
        #     last_print_time = times[idx]

        # if realtime:
        #     # Sleep to simulate real-time streaming
        #     _time.sleep(1.0 / float(sfreq))


def intrasubject_tests(dataset, dir_datetime_mark=None, datetime_mark=None):
    """
    main function: process subjects, train models, and eval performance.
    
    1. Loads data for each subject
    2. Preprocesses fNIRS data (OD, Beer-Lambert, filter, epochs)
    3. Extracts features (mean, slope)
    4. Trains SVM classifier
    5. Eval using cross-validation
    6. Saves results to CSV
    
    The function processes each subject independently (intra-subject analysis),
    
    Args:
        dataset: FineMI dataset object with configuration
        dir_datetime_mark: Optional directory name for results (if None, uses timestamp)
        datetime_mark: Optional timestamp for results (if None, generates new one)
        
    Returns:
        dir_datetime_mark: Directory name where results were saved
    """
    # Print config info for logging/reproducibility
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

    # list of model names used
    model_names = []
    model_index = dict()  # maps model name to index in results arr
    index = 0

    for model in use_models:
        if use_models[model]:
            model_names.append(model)
            model_index[model] = index
            index += 1

    print("------------------------------------------------------------------")
    print("valid size: ", valid_size)
    print("------------------------------------------------------------------")

    # initialize arrs to store accuracy data
    num_sessions_per_subject = dataset.num_sessions_per_subject
    train_acc_array = np.zeros((num_subjects * num_sessions_per_subject + 1, len(model_names)))  # Accuracy array of the training set. Dimension: num_subjects* num_sessions_in_1_subject + 1 (Average)
    valid_acc_array = np.zeros((num_subjects * num_sessions_per_subject + 1, len(model_names)))  # Accuracy array of the validation set. Dimension: num_subjects* num_sessions_in_1_subject + 1 (Average)
    test_acc_array = np.zeros((num_subjects * num_sessions_per_subject + 1, len(model_names)))  # Accuracy array of the test set. Dimension: num_subjects* num_sessions_in_1_subject + 1 (Average)

    subject_names = []  # store subject-session identifiers

    # process each subject (intra-subject)
    for subject_idx in range(start_subject - 1, end_subject):
        
        print("\n###############################################################################\n")
        
        if num_subjects == 1:  
            subject_idx = target_subject - 1
        subject_name = str(subject_idx + 1)
        # if subject_name == '9':
        #     continue
        # process each session
        num_sessions = dataset.num_sessions_per_subject if dataset.num_sessions_per_subject is not None else 1
        for session_idx in range(num_sessions):

            # Create subject-session identifier (e.g., "3s1" = subject 3, session 1)
            # subject_session_name = subject_name
            subject_names.append(subject_name)  # Record for results table

            print("Loading data of subject %s... " % subject_name)

            # load raw data files for subject
            dataset.load([subject_idx + 1], session_list=[session_idx + 1])
            print("Loading data of subject %s... Done." % subject_name)

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
                    # X_fnirs_train_elem, Y_fnirs_train_elem, info_fnirs, freq_bounds = preprocessing_fnirs_func(
                    #     raw_file, dataset, preprocessing_params, subject_idx=subject_idx)
                    X_fnirs_train_elem, Y_fnirs_train_elem, info_fnirs, freq_bounds, label_map, event_dict = preprocessing_fnirs_func(
                        raw_file, dataset, preprocessing_params, subject_idx=subject_idx)
                    # separate HbO and HbR channels
                    X_hbo_train_elem = X_fnirs_train_elem[:, ::2, :].copy()
                    X_hbr_train_elem = X_fnirs_train_elem[:, 1::2, :].copy()

                    # Combine HbO and HbR if use_hbr is True
                    if use_hbr:
                        # Concatenate along channel dimension: [all_HbO_channels, all_HbR_channels]
                        # This doubles the number of channels (e.g., 24 channels → 48 channels)
                        X_fnirs_train_elem = np.concatenate([X_hbo_train_elem, X_hbr_train_elem], axis=1)
                    else:
                        # only HbO channels 
                        X_fnirs_train_elem = X_hbo_train_elem
                    
                    X_fnirs_train.append(X_fnirs_train_elem)
                    Y_fnirs_train.append(Y_fnirs_train_elem)
                
                # concatenate data
                X_fnirs_train = np.concatenate(X_fnirs_train)
                Y_fnirs_train = np.concatenate(Y_fnirs_train)

                X_train_list.append(X_fnirs_train)
                info_list.append(info_fnirs)
                Y_train = Y_fnirs_train.copy()

                assert (len(X_train_list) > 0)  # Ensure we have at least one data source

                # Get final training data (handle single vs multiple data types)
                X_train = X_train_list if len(X_train_list) > 1 else X_train_list[0]

                print("Preprocessing data of subject %s... Done." % subject_name)

                Y_train_list.append(Y_train)

                print("Building %s model of subject %s with scenario %s..." % (
                    model_name, subject_name, scenario_name))

                # machine learning pipeline
                # 1. feature extraction: Mean + Slope 
                # 2. Standardization: Z-score normalization
                # 3. Classification: SVM
                
                n_times = X_train.shape[-1]  # Number of time points per epoch
                mean = Mean(axis=-1)  # Compute mean across time dimension (reduces n_times → 1)
                slope = Slope(n_times)  # Compute slope across time dimension (reduces n_times → 1)
                ss = StandardScaler()  # Z-score normalization (zero mean, unit variance)
                svm = SVC(kernel=params["SVM"]["kernel"], C=params["SVM"]["svm_C"])  # Linear SVM classifier
                
                # combines Mean and Slope features in parallel
                union = FeatureUnion([("mean", mean), ("slope", slope)])
                
                # pipeline chains: feature extraction to normalization to classification
                model = Pipeline([("feature_union", union), ("standard_scale", ss), ('svm', svm)])
                
                # cross-validation to evaluate model performance
                train_acc_elem, valid_acc_elem, test_acc_elem = cross_validation(model, X_train, Y_train, K=K,
                                                                                 subject_session_name=subject_name,
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
                    model_name, subject_name, scenario_name))

                # --- Train final model on full subject data for continuous streaming ---
                try:
                    final_model = model.fit(X_train, Y_train)
                    print("Trained final model on full data for subject %s, model %s." % (subject_name, model_name))

                    # Get a continuous Raw object to stream from. Use the first raw file if available.
                    # raw_data may be a list (handled above), so pick the first Raw for streaming.
                    raw_cont = raw_data[0] if isinstance(raw_data, (list, tuple)) else raw_data

                    # window_samples must match the number of time points per epoch used in training
                    window_samples = X_train.shape[-1]

                    # map class_of_interest (strings from config) -> integer indices used in Y_train
                    positive_label_idxs = None
                    if class_of_interest is not None:
                        positive_label_idxs = []
                        for lab in class_of_interest:
                            # try lookup in event_dict (map name -> original event code)
                            if isinstance(event_dict, dict) and lab in event_dict:
                                orig_code = event_dict[lab]
                            else:
                                try:
                                    orig_code = int(lab)
                                except Exception:
                                    continue
                            if orig_code in label_map:
                                positive_label_idxs.append(label_map[orig_code])

                    # if no mapping found, fall back to passing the original config value (keeps previous behaviour)
                    positive_label_to_pass = positive_label_idxs if (positive_label_idxs is not None and len(positive_label_idxs) > 0) else class_of_interest

                    # Start streaming predictions and print one prediction per second
                    stream_predict_continuous(final_model, raw_cont, dataset,
                                              window_samples=window_samples,
                                              threshold=0.5,
                                              positive_label=positive_label_to_pass,
                                              realtime=True,
                                              print_interval_s=1.0)
                except Exception as e:
                    # Don't break the whole pipeline if streaming fails; continue to next model
                    print("Warning: streaming predictions failed for subject %s model %s: %s" % (
                        subject_name, model_name, str(e)
                    ))

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

    subject_names = subject_names.copy()  # Copy to avoid modifying original list
    subject_names.append("Average")  # Add row label for average across subjects

    # Convert each accuracy value to formatted string
    for i in range(len(test_acc_array)):
        result_train.append([])
        if valid_size > 0:
            result_valid.append([])
        result_test_acc.append([])

        for j in range(len(test_acc_array[i])):
            result_train[i].append("%.4f" % train_acc_array[i][j])
            if valid_size > 0:
                result_valid[i].append("%.4f" % valid_acc_array[i][j])
            result_test_acc[i].append("%.4f" % test_acc_array[i][j])

    # subject 9 skipped during processing bc no data available   
    pdf_result_train = pd.DataFrame(result_train, index=subject_names, columns=model_names)
    if valid_size > 0:
        pdf_result_valid = pd.DataFrame(result_valid, index=subject_names, columns=model_names)
    pdf_result_test_acc = pd.DataFrame(result_test_acc, index=subject_names, columns=model_names)

    # save results to CSV files
    if dir_datetime_mark is None:
        dir_datetime_mark = datetime_mark
    dir_path = "results/" + dir_datetime_mark
    if not os.path.exists(dir_path):
        mkdir(dir_path)  # create results dir if doesn't exist
    
    # save CSV files with timestamp and dataset name
    pdf_result_train.to_csv(dir_path + "/" + datetime_mark + "_dataset=" + dataset_name + "__train.csv")
    if valid_size > 0:
        pdf_result_valid.to_csv(dir_path + "/" + datetime_mark + "_dataset=" + dataset_name + "__valid.csv")
    pdf_result_test_acc.to_csv(dir_path + "/" + datetime_mark + "_dataset=" + dataset_name + "__test_acc.csv")

    return dir_datetime_mark

def binary_predictions(model, X_test, threshold=0.5, positive_label=None):
    """
    Return array of 0/1 for each sample:
    - If model has predict_proba, use prob of positive class and threshold.
    - Else if model has decision_function, map decision>0 -> 1 (or threshold).
    - Else use predict() and map positive_label -> 1, others -> 0.
    """
    # try predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        # If positive_label is a list, sum probabilities for those classes
        if positive_label is None:
            # assume positive is class with index 1 if binary
            pos_idx = 1 if probs.shape[1] > 1 else 0
            probs_pos = probs[:, pos_idx]
        else:
            # support positive_label being a single label or a list/tuple
            classes = list(model.classes_)
            if isinstance(positive_label, (list, tuple, np.ndarray)):
                idxs = [classes.index(p) for p in positive_label]
                probs_pos = probs[:, idxs].sum(axis=1)
            else:
                pos_idx = classes.index(positive_label)
                probs_pos = probs[:, pos_idx]
        return (probs_pos >= threshold).astype(int)

    # try decision_function
    if hasattr(model, "decision_function"):
        dec = model.decision_function(X_test)
        # If binary decision_function returns 1D array (n_samples,), use threshold 0
        if dec.ndim == 1:
            return (dec >= 0).astype(int)
        # For multi-class decision_function, fallback to predict-based mapping

    # fallback to predict and map labels
    preds = model.predict(X_test)
    if positive_label is None:
        raise ValueError("positive_label must be provided when model has no predict_proba/usable decision_function")
    # support list/tuple of positive labels
    if isinstance(positive_label, (list, tuple, np.ndarray)):
        return np.isin(preds, list(positive_label)).astype(int)
    return (np.array(preds) == positive_label).astype(int)
