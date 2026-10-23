"""Shared epoching. One design decision this file encodes: we epoch ONCE per
session with a WIDE window, and let each feature extractor crop out the
sub-window it actually needs (e.g. bandpower/CSP want 1-3s post-cue, ERDS
additionally wants a pre-cue baseline). That's what lets bandpower, CSP,
FBCSP, Riemannian and ERDS features all share one epoching pass instead of
each re-epoching the raw data with their own tmin/tmax,.

"""

from __future__ import annotations

import mne

#this defines the window so our epochs will go from -2.0 to 4.0
DEFAULT_TMIN = -2.0
DEFAULT_TMAX = 4.0
DEFAULT_BASELINE = None  # no MNE-level baseline correction; ERDSFeatures does
                          # its own baseline-relative normalization internally

def make_epochs(raw: mne.io.BaseRaw, events, event_id: dict, tmin: float = DEFAULT_TMIN, tmax: float = DEFAULT_TMAX, baseline=DEFAULT_BASELINE) -> mne.Epochs:
    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False,
    )