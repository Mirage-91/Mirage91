"""utilities: loading .xdf recordings into MNE Raw objects, and working out the 
event/marker info.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pyxdf

# 4-class motor imagery task. Matches feature_extraction.py's event_mapping
# and the discrete commands the competition's UDP game interface expects
# (rest -> no command, the other three -> INPUT_A/B/C/D depending on game).
EVENT_ID = {"left_hand": 1, "right_hand": 2, "feet": 3, "rest": 4}


def load_xdf_to_raw(filepath: str | Path) -> mne.io.BaseRaw:
    """Load a single .xdf recording into an MNE Raw object with annotations
    attached from the 'paradigm' marker stream. (Same logic as main.py's
    load_xdf_to_raw — kept here so every script imports one copy.)
    """
    streams, header = pyxdf.load_xdf(str(filepath))

    eeg_stream = None
    marker_stream = None
    for s in streams:
        stype = s["info"]["type"][0]
        if stype == "EEG":
            eeg_stream = s
        elif stype == "Markers" and s["info"]["name"][0] == "paradigm":
            marker_stream = s

    if eeg_stream is None:
        raise ValueError(f"No EEG stream found in {filepath}")

    ch_names = [
        c["label"][0]
        for c in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    ]
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])

    ch_types = [
        "misc" if ch in ("x_dir", "y_dir", "z_dir") else "eeg" for ch in ch_names
    ]

    data = eeg_stream["time_series"].T
    data = data * 1e-6  # microvolts -> volts (MNE convention)

    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)

    if "FP2" in raw.ch_names:
        raw.rename_channels({"FP2": "Fp2"})

    montage = mne.channels.make_standard_montage("colin27_1020")
    raw.set_montage(montage, on_missing="warn", verbose=False)

    if marker_stream is not None:
        eeg_t0 = eeg_stream["time_stamps"][0]
        onsets = marker_stream["time_stamps"] - eeg_t0
        descriptions = [label[0].strip() for label in marker_stream["time_series"]]
        annotations = mne.Annotations(
            onset=onsets, duration=0.0, description=descriptions
        )
        raw.set_annotations(annotations)

    return raw


def get_events(raw: mne.io.BaseRaw, event_id: dict | None = None):
    """Extract events from annotations, keeping only the real task classes
    (drops e.g. 'pause'/'post_block' markers if present)."""
    event_id = event_id or EVENT_ID
    events, found_ids = mne.events_from_annotations(
        raw, event_id=event_id, verbose=False
    )
    return events, event_id


def find_xdf_runs(recordings_dir: str | Path) -> list[Path]:
    """Discover all .xdf files under a directory (non-recursive by default,
    falls back to recursive if nothing found at the top level)."""
    recordings_dir = Path(recordings_dir)
    runs = sorted(recordings_dir.glob("*.xdf"))
    if not runs:
        runs = sorted(recordings_dir.rglob("*.xdf"))
    return runs


def load_xdf_run(path: str | Path):
    """Load one run and return (raw, events) — the pair every downstream
    stage needs. This is the function train_classifier_rest_vs_movement.py
    expected from mirage_eeg.io but that was never uploaded."""
    raw = load_xdf_to_raw(path)
    events, _ = get_events(raw)
    return raw, events