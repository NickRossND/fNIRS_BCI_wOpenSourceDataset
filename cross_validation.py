# ============================================================================
# CROSS-VALIDATION FUNCTION
# ============================================================================

import numpy as np
import math
import datetime
import gc
from sklearn.model_selection import KFold, ShuffleSplit


def cross_validation(model, X_train, Y_train, K=5, subject_session_name="", model_name="", valid_size=0.2, seed=1):
    """
    Perform K-fold cross-validation with nested train/validation split.
    
    This function implements a nested cross-validation scheme:
    1. Outer loop: K-fold CV splits data into K folds
    2. For each fold: uses 1 fold as test, K-1 folds as training
    3. Inner loop: further splits training data into train/validation sets
    4. Trains model and evaluates on train, validation, and test sets
    
    This provides an unbiased estimate of model performance and prevents overfitting.
    
    Args:
        model: sklearn model (Pipeline) to train and evaluate
        X_train: Feature data (n_samples, n_features) or (n_samples, n_channels, n_times)
        Y_train: Labels (n_samples,) - class labels for each sample
        K: Number of folds for cross-validation (5 = 5-fold CV)
        subject_session_name: Identifier for logging (e.g., "3s1")
        model_name: Model name for logging
        valid_size: Fraction of training data to use for validation (0.2 = 20%)
        seed: Random seed for reproducibility
        
    Returns:
        train_accuracy: Mean accuracy on training sets across all folds (should be high)
        valid_accuracy: Mean accuracy on validation sets across all folds (for model selection)
        test_accuracy: Mean accuracy on test sets across all folds (final performance estimate)
    """
    # Extract subject name from subject_session_name (e.g., "3s1" -> "3")
    subject_name = subject_session_name.split("s")[0]
    
    # Lists to store accuracies for each fold (will have K values each)
    train_accuracy_list = []
    valid_accuracy_list = []
    test_accuracy_list = []
    
    # Number of unique classes (for potential stratified splitting)
    n_classes = len(np.unique(Y_train))

    # Create K-fold cross-validator
    # shuffle=True: randomly shuffle data before splitting (prevents order bias)
    # random_state=seed: ensures reproducibility (same split every time)
    cv_test = KFold(K, shuffle=True, random_state=seed)

    # Outer loop: iterate through each fold
    for test_split_index, (train_index, test_index) in enumerate(cv_test.split(X_train, Y_train)):
        # test_index: indices for test set (1/K of data, e.g., 20% if K=5)
        # train_index: indices for training set (K-1/K of data, e.g., 80% if K=5)
        
        # Extract test set: 1 fold held out for final evaluation
        # This is the "unseen" data that the model has never seen during training
        X_test = X_train[test_index]
        Y_test = Y_train[test_index]

        # Extract training set: remaining K-1 folds
        # This will be further split into actual training and validation sets
        X_train_set = X_train[train_index]
        Y_train_set = Y_train[train_index]

        print("test_split_index: ", test_split_index)
        datetime_mark = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(datetime_mark)

        # Calculate number of samples for validation set
        # valid_size=0.2 means 20% of training data goes to validation
        n_valid = math.ceil(valid_size * X_train_set.shape[0])
        
        # Create validator for splitting training data into train/validation
        # ShuffleSplit randomly selects validation samples (not sequential)
        cv_valid = ShuffleSplit(test_size=n_valid, random_state=seed)

        # Split training set into actual training and validation sets
        # This is the "inner" split: from the K-1 folds, further split into train/validation
        X_train_train_idx, X_train_valid_idx = next(cv_valid.split(X_train_set, Y_train_set))
        X_train_train_set = X_train_set[X_train_train_idx]  # Data for model training
        Y_train_train_set = Y_train_set[X_train_train_idx]
        X_train_valid_set = X_train_set[X_train_valid_idx]  # Data for hyperparameter tuning/early stopping
        Y_train_valid_set = Y_train_set[X_train_valid_idx]

        print("\nsubject: %s of model: %s" % (subject_session_name, model_name))

        print("Training set data shape: ", X_train_train_set.shape)
        
        # Train the model on training set
        # This fits the entire pipeline: feature extraction → normalization → SVM
        model.fit(X_train_train_set, Y_train_train_set)
        
        # Evaluate on training set (should be high, indicates model can learn the patterns)
        train_accuracy_list.append(model.score(X_train_train_set, Y_train_train_set))
        
        # Evaluate on validation set (used for model selection and hyperparameter tuning)
        if valid_size > 0:
            valid_accuracy_list.append(model.score(X_train_valid_set, Y_train_valid_set))
        else:
            valid_accuracy_list.append(0)

        # Evaluate on test set (final performance estimate - this is the most important metric)
        # This tells us how well the model generalizes to unseen data
        test_accuracy_elem = model.score(X_test, Y_test)
        test_accuracy_list.append(test_accuracy_elem)

    # Calculate mean accuracies across all K folds
    # This gives us a robust estimate of performance (averaged over K different train/test splits)
    train_accuracy = np.array(train_accuracy_list).mean()
    valid_accuracy = np.array(valid_accuracy_list).mean()
    test_accuracy = np.array(test_accuracy_list).mean()

    print("Mean train score of subject %s of model %s: %.4f" % (subject_session_name, model_name, train_accuracy))
    print("Mean test score of subject %s of model %s: %.4f" % (subject_session_name, model_name, test_accuracy))
    
    # Clean up memory
    del model
    gc.collect()  # Force garbage collection to release unused memory
    
    return train_accuracy, valid_accuracy, test_accuracy

