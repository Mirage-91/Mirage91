"""Filter EEG, apply a reference, and provide artifact rejection helpers."""

from __future__ import annotations

import mne
import numpy as np
import scipy.signal as sps

MAINS_FREQ_HZ = 50.0  # Hz Graz, Austria


def filter_raw(
    raw: mne.io.BaseRaw,
    notch_freq: float = MAINS_FREQ_HZ,
    highpass_freq: float = 0.1,
    lowpass_freq: float | None = 60.0,
    order: int = 4,
) -> mne.io.BaseRaw:
    """Apply notch + band-pass filtering. This is intentionally causal (so that it works online as well)."""
    fs = raw.info["sfreq"]
    data = raw.get_data()

    nyquist = fs / 2
    low = highpass_freq / nyquist
    high = lowpass_freq / nyquist
    sos_bp = sps.butter(order, [low, high], btype="band", output="sos")
    data = sps.sosfilt(sos_bp, data, axis=-1)

    b_notch, a_notch = sps.iirnotch(w0=notch_freq, Q=30, fs=fs)
    data = sps.lfilter(b_notch, a_notch, data, axis=-1)

    raw_filtered = mne.io.RawArray(data, raw.info.copy(), verbose=False)
    raw_filtered.set_annotations(raw.annotations)
    return raw_filtered


def preprocess_run(
    raw: mne.io.BaseRaw,
    reference: str = "average",
    notch_freq: float = MAINS_FREQ_HZ,
    highpass_freq: float = 0.1,
    lowpass_freq: float = 60.0,
) -> mne.io.BaseRaw:
    """Filter the recording, apply the chosen EEG reference, and optionally ICA."""
    raw = filter_raw(raw, notch_freq, highpass_freq, lowpass_freq)

    if reference in {"average", "car"}:
        raw.set_eeg_reference("average", verbose=False)
    elif reference == "laplacian":
        raw = raw.copy().set_eeg_reference("average", verbose=False)

    return raw

def reject_artifact_epochs(
    X: np.ndarray,
    ptp_threshold: float = 400e-6,
    grad_threshold: float = 25e-6,
    max_bad_channel_frac: float = 0.5,
) -> np.ndarray:
    """Return a boolean mask of epochs to keep.

    Each epoch is rejected only if more than `max_bad_channel_frac` of channels
    exceed either the peak-to-peak or gradient threshold.
    """
    X = np.asarray(X)
    ptp = X.max(axis=-1) - X.min(axis=-1)
    grad = np.abs(np.diff(X, axis=-1)).max(axis=-1)

    channel_bad = (ptp > ptp_threshold) | (grad > grad_threshold)
    bad_frac = channel_bad.mean(axis=-1)
    return bad_frac <= max_bad_channel_frac
