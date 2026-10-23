"""Filter EEG, apply common-average reference, and provide epoch rejection."""

#imports 
from __future__ import annotations

import mne
import numpy as np
import scipy.signal as sps


MAINS_FREQ_HZ = 50.0  # Hz Graz, Austria

def filter_raw(raw: mne.io.BaseRaw, notch_freq: float = MAINS_FREQ_HZ, highpass_freq: float = 0.1, lowpass_freq: float | None = 60.0, order: int = 4,) -> mne.io.BaseRaw:
    """Notch + band-pass filter, CAUSAL (via scipy lfilter, not filtfilt).

    Changed from zero-phase to causal on purpose: this makes the offline
    filter response identical to what we get online. Zero-
    phase filtering uses future samples via filtfilt — great for
    offline-only analysis, but it's impossible to replicate live,
    since a live sample stream has no "future" to filter with. This removes
    that offline/online mismatch entirely, at the cost of a small phase
    delay (which happens with dig causal filter).
    """

    fs = raw.info["sfreq"]
    data = raw.get_data()

    # band-pass (0.1-60Hz)
    nyquist = fs / 2
    low = highpass_freq / nyquist
    high = lowpass_freq / nyquist
    sos_bp = sps.butter(order, [low, high], btype="band", output="sos")
    data = sps.sosfilt(sos_bp, data, axis=-1)

    #notch (50 Hz)
    b_notch, a_notch = sps.iirnotch(w0=notch_freq, Q=30, fs=fs) #Q is quality factor, determines the width of the notch filter
    data = sps.lfilter(b_notch, a_notch, data, axis=-1)

    raw_filtered = mne.io.RawArray(data, raw.info.copy(), verbose=False)
    raw_filtered.set_annotations(raw.annotations)
    return raw_filtered

def preprocess_run(raw: mne.io.BaseRaw, notch_freq: float = MAINS_FREQ_HZ, highpass_freq: float = 0.5, lowpass_freq: float = 40.0) -> mne.io.BaseRaw:
    """Filter the recording and apply common-average reference."""
    raw = filter_raw(raw, notch_freq, highpass_freq, lowpass_freq)
    raw.set_eeg_reference("average", verbose=False)
    return raw

#move this to epochs?
#very conservative thresholds, we dont want to throw away too much data since we have very little data
def reject_artifact_epochs(X: "object", ptp_threshold: float = 400e-6, grad_threshold: float = 25e-6, max_bad_channel_frac: float = 0.5,) -> "object":
    """Flag epochs likely contaminated by blinks/muscle/movement artifacts,
    using only information inside that epoch's own window (so it behaves
    identically offline and online).

    this checks per-channel, then rejects the epoch only if more
    than max_bad_channel_frac of channels are bad. 

    - ptp_threshold: max allowed peak-to-peak amplitude per channel (volts).
    - grad_threshold: max allowed sample-to-sample jump per channel (volts).
    - max_bad_channel_frac: epoch is rejected only if more than this
      fraction of channels exceed either threshold.

    Returns a boolean mask, shape (n_epochs,), True = keep.
    """

    X = np.asarray(X)
    ptp = X.max(axis=-1) - X.min(axis=-1)            # (n_epochs, n_channels)
    grad = np.abs(np.diff(X, axis=-1)).max(axis=-1)  # (n_epochs, n_channels)

    channel_bad = (ptp > ptp_threshold) | (grad > grad_threshold)
    bad_frac = channel_bad.mean(axis=-1)
    return bad_frac <= max_bad_channel_frac