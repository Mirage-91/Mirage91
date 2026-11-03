import pyxdf
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mappings import TARGET_CHAR_ENCODING, TARGET_CHAR_DECODING, EVENT, is_target_event, is_nontarget_event
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#defining the bandpass settings
iir_filter = {
    'h_freq': 10,
    'l_freq': 0.1,
    'method': "iir",
    'iir_params': dict(order=4, ftype='butter')
}

#to scale to V (from µV)
scaling_dict = {"eeg": 1e-6} 

def main():
    """Main function to process EEG data, extract features, and evaluate classification accuracy."""
    fname= "exam_eeg.xdf"
    temp_data = pyxdf.load_xdf(fname)

    #printing the markers 
    stream = temp_data[0]
    stream_idx = next(i for i, s in enumerate(stream) if s['info']['type'][0].lower() == 'eeg') #eeg stream
    marker_idx = next(i for i, s in enumerate(stream) if s['info']['type'][0].lower() == 'markers') #marker stream

    verify_markers(stream[marker_idx], NumScan=16, expected_interval=0.166)
    
    #setting AUX and BIP to misc 
    channel_meta = temp_data[0][stream_idx]['info']["desc"][0]["channels"][0]["channel"]
    all_channel_names = []
    for elem in channel_meta: 
        all_channel_names.append(elem['label'][0])
    all_channel_types = []
    for elem in channel_meta:
        channel_type = elem.get('type', ['eeg'])[0].lower()
        channel_name = elem['label'][0].strip().lower()
        if channel_name == 'bip' or channel_type == 'aux':
            all_channel_types.append('misc')
        else:
            all_channel_types.append(channel_type)

    # Store EEG time series as pandas DataFrame
    eeg_stream = pd.DataFrame(temp_data[0][stream_idx]['time_series'], columns=all_channel_names)[all_channel_names]

    # Gather the EEG info and create an MNE RawArray object
    eeg_info = temp_data[0][stream_idx]['info']
    tmp_info = mne.create_info(ch_names=list(eeg_stream.columns), ch_types=all_channel_types, sfreq=eeg_info["nominal_srate"][0])
    tmp_raw = mne.io.RawArray(eeg_stream.T, tmp_info)

    # Scale the EEG data to volts as MNE expects that
    tmp_raw.apply_function(lambda data: data * scaling_dict['eeg'], picks='eeg')

    # Pick the EEG channels
    eeg_picks = mne.pick_types(tmp_raw.info, eeg=True)
    
    #bandpass 
    tmp_raw.filter(**iir_filter, picks=eeg_picks, phase='forward') #phase='forward' makes it causal 
    final_sig = tmp_raw.copy()
    final_sig.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='ignore')

    #plotting montage
    final_sig.plot_sensors(show_names=True, show=False)

    #plot time series
    final_sig.plot(picks='eeg', duration=10, n_channels=len(eeg_picks), scalings='auto', show=False)

    #set rejection criteria
    # consider signal as flat, when below certain threshold
    # reject, when jump greater than some value in single epoch
    flat_criteria = None
    reject_criteria = None      # as required, we only performed visual analysis and saw no need to remove anything

    #extract markers
    events_tmp = extract_markers(stream, stream_idx, marker_idx)

    #epoching 
    epoch_picks = mne.pick_types(final_sig.info, eeg=True) #excluding BIP cause its not an actual channel 
    tmin = -0.2
    tmax = 0.8
    epochs_tmp= mne.Epochs(final_sig.copy(), events_tmp, event_id=None, tmin=tmin, baseline=(-0.2,0),
                                tmax=tmax, reject=reject_criteria, flat=flat_criteria, picks=epoch_picks,
                                reject_by_annotation=False, preload=True)
 

    # averaging    
    target_keys = list(filter(is_target_event, epochs_tmp.event_id.keys()))
    target_epochs = epochs_tmp[target_keys]

    nontarget_keys = list(filter(is_nontarget_event, epochs_tmp.event_id.keys()))
    nontarget_epochs = epochs_tmp[nontarget_keys]

    n_epochs = min(len(target_epochs), len(nontarget_epochs), 640)
    #for if all epochs get dropped 
    if n_epochs == 0:
        raise RuntimeError(
            f"Cannot average both conditions: TARGET={len(target_epochs)}, "
            f"NONTARGET={len(nontarget_epochs)}."
        )

    ## we randomly select targets of each group; is needed cause we most probably won't have the same num 
    ## of target and nontarget (cause of dropped epochs) but need to average with same number so both groups should be same size 
    rng = np.random.default_rng(13)
    target_selection = rng.choice(len(target_epochs), size=n_epochs, replace=False)
    nontarget_selection = rng.choice(len(nontarget_epochs), size=n_epochs, replace=False)
    target_evoked = target_epochs[target_selection].average()
    nontarget_evoked = nontarget_epochs[nontarget_selection].average()
    print(f"Averaging {n_epochs} epochs per condition (after rejection).")

    target_evoked.plot(picks='eeg', spatial_colors=True, window_title='Target average', show=False)
    nontarget_evoked.plot(picks='eeg', spatial_colors=True, window_title='Non-target average', show=False)

    #showing the P300 in one
    colors = ["green", "red"]
    title = 'P300 for Speller'
    mne.viz.plot_compare_evokeds({"TARGET": target_evoked, "NONTARGET": nontarget_evoked}, combine = "mean", show=False)
    target_evoked.comment = "TARGET"
    nontarget_evoked.comment = "NONTARGET"
    evokeds = [target_evoked, nontarget_evoked]
    mne.viz.plot_evoked_topo(evokeds,title=title, color=colors, show=False)

    # downsampling (50Hz) + cropping
    epochs_400ms = downsampling_and_cropping(epochs_tmp, eeg_info)

    ## Group the epochs by trial (i.e. by individual letter)
    epochs_by_trial = {}
    # Identify all unique trial identifiers as (char_code, instance) tuples --> custom event code
    unique_trials = set()
    for key in epochs_400ms.event_id.keys():
        char_code = int(key[3:5])  # Extracts digits 4 and 5
        instance = int(key[5:])    # Extracts digit 6 to the end
        unique_trials.add((char_code, instance))

    for char_code, instance in unique_trials:
        # Find all original keys matching this exact character and instance
        matching_keys = [
            k for k in epochs_400ms.event_id.keys() 
            if int(k[3:5]) == char_code and int(k[5:]) == instance
        ]
        
        # Translate the integer code to the letter and append the instance number
        char_label = TARGET_CHAR_DECODING.get(char_code, str(char_code))
        trial_name = f"{char_label}_{instance}"  # e.g., "A_1" or "A_2"
        
        epochs_by_trial[trial_name] = epochs_400ms[matching_keys]

    print(f"Shape of epochs grouped by trial: { {k: v.get_data().shape for k, v in epochs_by_trial.items()} }")

    ## Sort the epochs according to their corresponding row or column.
    for trial_name, trial_epochs in epochs_by_trial.items():        
        # Extract number from position 2 and 3 (row/col indices) for all flashes and get the sort indices
        sorted_idx = np.argsort([str(code)[1:3] for code in trial_epochs.events[:, 2]], kind="stable")

        # Apply the sorting, basically along the first axis
        epochs_by_trial[trial_name] = trial_epochs[sorted_idx]

    print(f"Shape of epochs after sorting by row/column: { {k: v.get_data().shape for k, v in epochs_by_trial.items()} }")

    ## Reshape each epoch to a feature vector with 140 entries (20 samples over 7 channels)
    # collapse (n_epochs, n_channels, n_samples) to (n_epochs, n_channels * n_samples)
    features_by_trial = { 
        name: epochs.get_data().reshape(176, -1) for name, epochs in epochs_by_trial.items()
    }       
    print(f"Shape of feature vectors by trial: { {k: v.shape for k, v in features_by_trial.items()} }") 

    ## Construct the target vectors
    targets_by_trial = {
        trial_name: 
            np.array(
                [1 if is_target_event(code) else 0 for code in trial_epochs.events[:, 2]]
            ) for trial_name, trial_epochs in epochs_by_trial.items()
    }
    print(f"Shape of target vectors by trial: { {k: v.shape for k, v in targets_by_trial.items()} }")

    ## Verification
    first_key = list(features_by_trial.keys())[0]

    print(f"\n--- Data for Trial: {first_key} ---")

    features = features_by_trial[first_key]
    print(f"\nFeatures Shape: {features.shape}")
    print(f"Features Array (first 2 flashes, sorted, truncated):\n{features[:2, :]}")

    targets = targets_by_trial[first_key]
    print(f"\nTargets Shape: {targets.shape}")
    print(f"Targets Array (first 20 flashes, sorted):\n{targets[:20]}")


    ## sLDA classification
    trials = list(features_by_trial.keys())
    n_trials = len(trials)
    max_scans = 16
    n_rows = 5

    # for boolean classification results (20 trials x 16 scans)
    correct_predictions = np.zeros((n_trials, max_scans))

    for test_idx, test_key in enumerate(trials):
        # select one letter as a test set, use the rest for training
        train_keys = [k for k in trials if k != test_key]
        
        # create feature and target vectors
        X_train = np.vstack([features_by_trial[k] for k in train_keys])
        y_train = np.concatenate([targets_by_trial[k] for k in train_keys])
        
        X_test = features_by_trial[test_key]
        y_test = targets_by_trial[test_key]
        
        # find the true row and column from the sorted target vector
        y_test_grouped = y_test.reshape(11, max_scans)  # row-major; left-to-right, top-to-bottom
        true_row = np.argmax(np.mean(y_test_grouped[:n_rows], axis=1))   # why mean? can cover-up labelling glitch
        true_col = np.argmax(np.mean(y_test_grouped[n_rows:], axis=1))
        
        # fit an sLDA to the training data with equal priors
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5])
        lda.fit(X_train, y_train)
        
        # evaluate the trained model on the test set using the decision function
        # returns a signed discriminant score with 176 entries
        scores = lda.decision_function(X_test) 
        
        # reshape the 176 scores into 11 rows (representing the 5 grid rows + 6 grid cols)
        # and 16 columns (representing the scans)
        scores_grouped = scores.reshape(11, max_scans)
        
        # repeat the evaluation for NumScans = 1, 2, ..., 16
        for scan_idx in range(max_scans):
            num_scans = scan_idx + 1
            
            # calculating the mean over the current NumScan yields a vector with eleven entries
            mean_scores = np.mean(scores_grouped[:, :num_scans], axis=1)
            
            # identify the most probable target letter by selecting the highest-scoring row and column
            pred_row = np.argmax(mean_scores[:n_rows])
            pred_col = np.argmax(mean_scores[n_rows:])
            
            # store the classification result 
            if pred_row == true_row and pred_col == true_col:
                correct_predictions[test_idx, scan_idx] = 1

    # check how many letters were classified correctly and convert to accuracy percentages
    accuracies = np.mean(correct_predictions, axis=0) * 100

    # determine the best number of scans (first scan level to reach the maximum accuracy)
    max_accuracy = np.max(accuracies)
    best_scan = np.argmax(accuracies) + 1 

    print(f"Optimal performance achieved: {max_accuracy:.1f}% accuracy at NumScan = {best_scan}")

    # plot all accuracies with respect to the corresponding NumScan and highlight the best NumScan
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_scans + 1), accuracies, marker='o', linewidth=2, label="Accuracy")

    plt.axvline(x=best_scan, color='red', linestyle='--', label=f"Best NumScan ({best_scan})")
    plt.plot(best_scan, max_accuracy, 'ro', markersize=8)

    plt.xlabel("Number of Scans (NumScan)")
    plt.ylabel("Accuracy (%)")
    plt.title("Classification Accuracy vs. Number of Scans")
    plt.xticks(range(1, max_scans + 1))
    plt.legend()
    plt.grid(True)
    # best num_scan is marked as the number of scans with maximal accuracy, however one has to consider 
    # the tradeoff between accuracy and the paradigm runtime for selection of num_scan as well


def extract_markers(markers, idx_stream, idx_markers):
    """Extract markers from marker_stream and use custom encoding."""
    events = []
    current_target_char = None
    total_target_char = {}

    marker_stream = markers[idx_markers]
    eeg_stream = markers[idx_stream]

    for i, marker in enumerate(marker_stream['time_series']):
        marker_str = marker[0]
        # Find the matching timestamp in the EEG stream to get the sample index
        marker_time = marker_stream['time_stamps'][i]
        sample_index = np.argmin(np.abs(eeg_stream['time_stamps'] - marker_time))

        # retrieve which letter was the target
        if "START_LETTER_" in marker_str:
            current_target_char = marker_str.split("START_LETTER_")[1]
            total_target_char[current_target_char] = total_target_char.get(current_target_char, 0) + 1
            continue
        elif current_target_char is None:
            print(f"WARNING: Marker '{marker_str}' found without a preceding 'START_LETTER_' marker.")
            continue  # Skip this marker since we don't have a current target character

        row = 9     # means, not a row-flash, but a column-flash
        col = 9     # means, not a column-flash, but a row-flash

        # retrieve, which row/column flashed
        if "ROW_" in marker_str:
            row = int(marker_str.split("ROW_")[1])
        elif "COL_" in marker_str:
            col = int(marker_str.split("COL_")[1])

        # we used a custom encoding for the marker string
        if "NONTARGET" in marker_str:
            marker_encoded_value = int(
                f"{EVENT['NONTARGET']}{row}{col}{TARGET_CHAR_ENCODING[current_target_char]}{total_target_char[current_target_char]}"
            )
            events.append([sample_index, 0, marker_encoded_value])
        elif "TARGET" in marker_str:
            marker_encoded_value = int(
                f"{EVENT['TARGET']}{row}{col}{TARGET_CHAR_ENCODING[current_target_char]}{total_target_char[current_target_char]}"
            )
            events.append([sample_index, 0, marker_encoded_value])

    return np.array(events)


def verify_markers(marker_stream, NumScan=16, expected_interval=0.166, Rows=5, Collums=6):
    """
    Verify the integrity of our marker stream. The expected inter-marker-interval is 166 ms, not 133 ms,
    because according to the assignment instructions for the paradigm, at a typical frame rate of 60 Hz,
    the stimulus duration is 33 ms, and the inter-stimulus interval is 133 ms.
    """
    #we extract the labels and time_stamps 
    labels = []
    for sample in marker_stream["time_series"]:  #this holds infos like start end and the target/nontargets
        labels.append(sample[0])
    timestamps = marker_stream["time_stamps"] #this gives us information on timing 

    #getting the indices for the start markers
    starts = []
    for idx, label in enumerate(labels): 
        if label.startswith("START_LETTER_"): 
            starts.append(idx)
    # figuring out were the end is
    if "END" in labels: 
        end = labels.index("END")
    else: 
        end = len(labels)
    
    expected_flashes = NumScan * (Rows + Collums) #176 in our case
    interval_values = []

    complete = 0
    for trial, start in enumerate(starts):
        if trial + 1 < len(starts): 
            stop = starts[trial + 1]
        else: 
            stop = end
        
        indices = []
        for i in range(start + 1, stop): 
            if labels[i].startswith(("TARGET_", "NONTARGET_")): 
                indices.append(i)

        row_counts = [0] * Rows
        column_counts = [0] * Collums
        target_count = 0

        #we count how many rows, collums and targets there are 
        for i in indices:
            parts = labels[i].split("_")
            if parts[0] == "TARGET":
                target_count += 1
            if parts[1] == "ROW":
                row_counts[int(parts[2])] += 1
            else:
                column_counts[int(parts[2])] += 1

        #check the amount of rows and collums flashed 
        rows_and_columns_ok = True
        for count in row_counts + column_counts:
            if count != NumScan:
                rows_and_columns_ok = False
                print(f'Number of Rows and Collums wrong, check your input')

        targets_ok = target_count == NumScan * 2
        if (len(indices) == expected_flashes and rows_and_columns_ok and targets_ok):
            complete += 1

        #gets the time interval between two flashes
        for i in range(len(indices) - 1):
            interval = timestamps[indices[i + 1]] - timestamps[indices[i]]
            interval_values.append(interval)

    print(f"Complete letters: {complete}/{len(starts)} (with {expected_flashes} flashes expected each).")
    if interval_values:
        mean_interval = sum(interval_values) / len(interval_values) * 1000
        print(f"Mean flash interval: {mean_interval:.1f} ms (expected {expected_interval * 1000:.1f} ms).")


def downsampling_and_cropping(epochs, eeg_info, target_Hz=50, tmin=0.2, tmax=0.6):
    """Downsample the epochs to a target sampling frequency and crop them to a specific time window."""
    original_Hz = float(eeg_info["nominal_srate"][0])
    downsampling_factor = int(original_Hz // target_Hz)
    print(f"Downsampling factor: {downsampling_factor}")

    downsampled_epochs = epochs.copy()
    downsampled_epochs.decimate(decim=downsampling_factor)

    epochs_400ms = downsampled_epochs.crop(tmin=tmin,tmax=tmax,include_tmax=False)

    print("Raw event array codes:", epochs_400ms.events[:10, 2])
    print("Dictionary mapping:", list(epochs_400ms.event_id.items())[:5])

    return epochs_400ms


if __name__ == '__main__':
    main()
    plt.show()
