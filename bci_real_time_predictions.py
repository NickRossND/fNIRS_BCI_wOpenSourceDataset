import numpy as np
from mne import events_from_annotations
from preprocessing import preprocessing_fnirs_func
from config import preprocessing_params, use_hbr  

def build_prediction_timeline(y_pred, raw_data, events, dataset, default_value=-1):
    """
    Turn per-epoch predictions into a sample-wise timeline.

    Args:
        y_pred: (n_epochs,) predicted labels for each epoch
        raw_data: MNE Raw object for the session
        events: events array (n_events, 3) from events_from_annotations
        dataset: your FineMI dataset object (provides tmin, tmax, sampling_rate)
        default_value: value used where no prediction is available yet (e.g., -1)

    Returns:
        timeline: (n_samples_total,) array of label per sample
        time_axis: (n_samples_total,) array of time in seconds
    """
    sfreq = raw_data.info['sfreq']
    n_samples_total = raw_data.n_times

    # Initialize timeline with "no prediction yet"
    timeline = np.full(n_samples_total, default_value, dtype=float)

    n_epochs = len(y_pred)
    n_events = len(events)
    n = min(n_epochs, n_events)

    for i in range(n):
        pred = y_pred[i]
        onset_sample = events[i, 0]  # sample index of event i

        # epoch window in samples relative to event
        start_sample = int(onset_sample + dataset.tmin * sfreq)
        end_sample   = int(onset_sample + dataset.tmax * sfreq)

        # clip to valid range
        start_sample = max(0, start_sample)
        end_sample   = min(n_samples_total, end_sample)

        # timeline segment with prediction
        timeline[start_sample:end_sample] = pred

    # time in sec
    time_axis = np.arange(n_samples_total) / sfreq

    return timeline, time_axis

def predict_session_realtime_with_timeline(model,dataset,subject_idx,session_idx):
    # 1. Load raw data
    dataset.load([subject_idx + 1], session_list=[session_idx + 1])
    raw_data = dataset.raw_data_list[0]

    # 2. Preprocess to epochs (same as training)
    X_fnirs, Y_fnirs, info = preprocessing_fnirs_func(
        raw_data, dataset, preprocessing_params, subject_idx=subject_idx
    )
    X_hbo = X_fnirs[:, ::2, :].copy()
    X_hbr = X_fnirs[:, 1::2, :].copy()

    if use_hbr:
        X_session = np.concatenate([X_hbo, X_hbr], axis=1)
    else:
        X_session = X_hbo

    # 3. Per-epoch predictions
    y_pred = model.predict(X_session)

    # 4. Get events from annotations to know when epochs are centered
    events, event_dict = events_from_annotations(raw_data, event_id=dataset.event_id_fnirs)

    # 5. Build sample-wise timeline
    timeline, time_axis = build_prediction_timeline(
        y_pred=y_pred,
        raw_data=raw_data,
        events=events,
        dataset=dataset,
        default_value=-1
    )

    return y_pred, Y_fnirs, timeline, time_axis

def binarize_timeline(timeline, task_label=1, default_value=-1):
    """
    Convert multi-class timeline to binary (task vs not).

    Args:
        timeline: sample-wise predicted labels (int or -1 where no prediction)
        task_label: which label index represents "task"
        default_value: value that indicates "no prediction yet" (e.g., -1)

    Returns:
        binary_timeline: same shape, with 1=task, 0=non-task, default_value unchanged.
    """
    binary = np.zeros_like(timeline)
    mask_task = (timeline == task_label)
    binary[mask_task] = 1

    # Preserve "no prediction yet" if you want
    binary[timeline == default_value] = default_value

    return binary