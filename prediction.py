# prediction.py

import numpy as np
from preprocessing import preprocessing_fnirs_func
from config import preprocessing_params, use_hbr  # and data_type if you need it

def preprocess_session_fnirs(dataset, subject_idx, session_idx):
    """
    Mirror the training preprocessing for a single subject/session.

    Returns:
        X_session: (n_epochs, n_channels_out, n_times)
        Y_session: (n_epochs,)
        info_fnirs: MNE info object
    """
    # load subject/session,  same as in training
    dataset.load(subject_list=[subject_idx + 1], session_list=[session_idx + 1])
    
    raw_data = dataset.raw_data_list[0]

    # run the same fNIRS preprocessing pipeline as training
    X_fnirs, Y_fnirs, info_fnirs = preprocessing_fnirs_func(
        raw_data,
        dataset,
        preprocessing_params,
        subject_idx=subject_idx
    )

    # split HbO / HbR 
    X_hbo = X_fnirs[:, ::2, :].copy()
    X_hbr = X_fnirs[:, 1::2, :].copy()

    if use_hbr:
        # Use both HbO and HbR (concatenate along channel axis)
        X_session = np.concatenate([X_hbo, X_hbr], axis=1)
    else:
        # Use only HbO
        X_session = X_hbo

    return X_session, Y_fnirs, info_fnirs


def predict_session_realtime(model,dataset,subject_idx,session_idx):
    # preprocess session first
    X_session, Y_session, info = preprocess_session_fnirs(dataset=dataset,subject_idx=subject_idx, session_idx=session_idx)

    # run predictions for all epochs
    y_pred = model.predict(X_session)

    return y_pred, Y_session, info
