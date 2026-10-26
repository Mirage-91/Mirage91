from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import mne
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

mne.set_log_level("WARNING")

from classifiers import get_classifiers
from epochs import DEFAULT_BASELINE, DEFAULT_TMAX, DEFAULT_TMIN, make_epochs
from features import (
    HAS_PYRIEMANN,
    BandpowerFeatures,
    CSPFeatures,
    ERDSFeatures,
    FilterBankCSP,
    RiemannianFeatures,
)
from utils import EVENT_ID, find_xdf_runs, load_xdf_run
from preprocessing import preprocess_run, reject_artifact_epochs

#-----------------------------------------------------------------------
# default settings for the pipeline 

DEFAULT_BANDPOWER_CLASS_CONFIGS = {
    "left_hand": ("C3", 5, 10),
    "right_hand": ("C3", 5, 11),
    "feet": ("C3", 5, 13),
    "rest": ("C3", 5, 11),
}

#to change in both the parser and in the load_and_process_all 
#we are being very very conservative here but we dont have a lot of data so we want to make sure we dont throw away too much data
DEFAULT_PTP_THRESHOLD = 400e-6
DEFAULT_GRAD_THRESHOLD = 20e-6
DEFAULT_MAX_BAD_CHANNEL_FRAC = 0.3

#default for the ERDS features
DEFAULT_FREQUENCY_BANDS = [
    (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 28.0),
]

DEFAULT_ERDS_SPECS = [
    {"name": "C3_15_20Hz_ERS", "channel": "C3", "band": (15.0, 20.0), "window": (1.0, 3.5)},
    {"name": "C4_5_10Hz_ERS", "channel": "C4", "band": (5.0, 10.0), "window": (1.0, 3.5)},
]
#-----------------------------------------------------------------------------------------------------------------
#running the pipeline: 
#you can run the pipeline using a few different settings, run the following in the terminal: 

#python run_pipeline.py --recordings-dir recordings --cv group 
#python run_pipeline.py --recordings-dir <recordings dir> --cv group --downsample-freq 125

#this will run the pipeline on the recordings (you still need to add your specific recordings directory though!), using group cross-validation (i.e. each session is a separate fold).

#python run_pipeline.py --recordings-dir recordings --cv stratified 
#this will run the pipeline on the recordings in the recordings directory, using stratified cross-validation (i.e. each fold has a similar distribution of classes, but sessions can be split across folds).

def load_and_preprocess_all(
    recordings_dir: Path,
    reference: str,
    reject_artifacts: bool = True,
    ptp_threshold: float = DEFAULT_PTP_THRESHOLD,
    grad_threshold: float = DEFAULT_GRAD_THRESHOLD,
    max_bad_channel_frac: float = DEFAULT_MAX_BAD_CHANNEL_FRAC,
):
    """Load each .xdf run, preprocess it, and assemble the pooled dataset."""

    runs = find_xdf_runs(recordings_dir)
    if not runs:
        raise RuntimeError(f"No .xdf files found under {recordings_dir}")

    X_list, y_list, group_list = [], [], []
    sfreq, ch_names = None, None

    for run_idx, path in enumerate(runs):
        print(f"--- session {run_idx + 1}/{len(runs)}: {path.name}")
        raw, events = load_xdf_run(path)
        raw_clean = preprocess_run(raw, reference=reference)
        raw_clean.pick("eeg")

        epochs = make_epochs(raw_clean, events, event_id=EVENT_ID)
        if sfreq is None:
            sfreq = epochs.info["sfreq"]
            ch_names = list(epochs.info["ch_names"])

        X_session = epochs.get_data(copy=True)
        y_session = epochs.events[:, -1]
        n_before = len(X_session)

        if reject_artifacts:
            keep_mask = reject_artifact_epochs(
                X_session,
                ptp_threshold=ptp_threshold,
                grad_threshold=grad_threshold,
                max_bad_channel_frac=max_bad_channel_frac,
            )
            X_session = X_session[keep_mask]
            y_session = y_session[keep_mask]
            n_dropped = n_before - len(X_session)
            print(
                f"    {len(X_session)} epochs kept, {n_dropped} dropped as artifacts "
                f"({n_dropped / n_before:.0%})"
            )
        else:
            print(f"    {n_before} epochs")

        X_list.append(X_session)
        y_list.append(y_session)
        group_list.append(np.full(len(X_session), run_idx))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(group_list, axis=0)
    return X, y, groups, sfreq, ch_names


def build_feature_extractors(sfreq: float, ch_names: list[str], tmin: float) -> dict:
    n_samples = int(round((DEFAULT_TMAX - tmin) * sfreq)) + 1
    times = np.linspace(tmin, DEFAULT_TMAX, n_samples)
    extractors = {
        "Bandpower": BandpowerFeatures(DEFAULT_BANDPOWER_CLASS_CONFIGS, ch_names, sfreq, times),
        "CSP": CSPFeatures(n_components=6, sfreq=sfreq, times=times),
        "FBCSP": FilterBankCSP(bands=DEFAULT_FREQUENCY_BANDS, sfreq=sfreq, times=times),
        "Riemann": RiemannianFeatures(bands=DEFAULT_FREQUENCY_BANDS, sfreq=sfreq, times=times),
        "ERDS": ERDSFeatures(
            specs=DEFAULT_ERDS_SPECS,
            ch_names=ch_names,
            sfreq=sfreq,
            times=times,
            baseline_window=DEFAULT_BASELINE or (-2.0, 0.0),
        ),
    }

    return extractors


def run_benchmark(X, y, groups, sfreq, ch_names, tmin, n_splits=5, cv_strategy="group",
                   feature_names=None, classifier_names=None):
    """
    feature_names / classifier_names: optional lists to restrict the run to
    a subset, e.g. feature_names=["FBCSP", "Riemann"]. Unknown names raise
    immediately so a typo doesn't silently run everything (or nothing).
    """
    feature_extractors = build_feature_extractors(sfreq, ch_names, tmin)
    if feature_names:
        unknown = set(feature_names) - set(feature_extractors)
        if unknown:
            raise ValueError(f"unknown feature name(s) {sorted(unknown)}; "
                              f"available: {sorted(feature_extractors)}")
        feature_extractors = {k: v for k, v in feature_extractors.items() if k in feature_names}

    classifiers = get_classifiers()
    if classifier_names:
        unknown = set(classifier_names) - set(classifiers)
        if unknown:
            raise ValueError(f"unknown classifier name(s) {sorted(unknown)}; "
                              f"available: {sorted(classifiers)}")
        classifiers = {k: v for k, v in classifiers.items() if k in classifier_names}

    if cv_strategy == "group":
        n_splits = min(n_splits, len(np.unique(groups)))
        if n_splits < 2:
            raise RuntimeError(
                "Need at least 2 sessions for GroupKFold cross-validation; "
                f"found {len(np.unique(groups))}."
            )
        cv = GroupKFold(n_splits=n_splits)
        cv_kwargs = {"groups": groups}
    elif cv_strategy == "stratified":
        #NOTE: the GroupKFold is more realistic, but the StratifiedKFold will usually score higher because it can leak session info across folds. 
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_kwargs = {}
    else:
        raise ValueError(f"unknown cv_strategy: {cv_strategy!r} (expected 'group' or 'stratified')")

    results = []
    for feat_name, extractor in feature_extractors.items():
        for clf_name, clf in classifiers.items():
            pipeline = Pipeline([("features", extractor), ("classifier", clf)])
            try:
                scores = cross_val_score(
                    pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1, **cv_kwargs
                )
                mean_acc, std_acc = scores.mean(), scores.std()
                print(f"{feat_name:12s} + {clf_name:20s} | acc: {mean_acc:.3f} +/- {std_acc:.3f}")
                results.append((feat_name, clf_name, mean_acc, std_acc))
            except Exception as e:
                print(f"{feat_name:12s} + {clf_name:20s} | FAILED: {e}")
                traceback.print_exc(limit=1)

    results.sort(key=lambda r: r[2], reverse=True)
    print("\n" + "=" * 60)
    print("RANKED RESULTS (best first)")
    print("=" * 60)
    for feat_name, clf_name, mean_acc, std_acc in results:
        print(f"{feat_name:12s} + {clf_name:20s} | acc: {mean_acc:.3f} +/- {std_acc:.3f}")

    return results

#--------------------------------------------------------------------------------------------------------
def main():
    """Here all parser inputs are defined (for running the programm)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--reference", choices=["car", "laplacian"], default="car")
    parser.add_argument("--no-artifact-rejection", action="store_true",
                         help="disable peak-to-peak/gradient artifact rejection")
    parser.add_argument("--ptp-threshold", type=float, default=DEFAULT_PTP_THRESHOLD,
                         help="max peak-to-peak amplitude per channel per epoch, in volts")
    parser.add_argument("--grad-threshold", type=float, default=DEFAULT_GRAD_THRESHOLD,
                         help="max sample-to-sample jump per channel per epoch, in volts")
    parser.add_argument("--max-bad-channel-frac", type=float, default=DEFAULT_MAX_BAD_CHANNEL_FRAC,
                         help="reject epoch only if more than this fraction of channels are bad")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--features", type=str, nargs="+", default=None,
                         help="only run these feature extractors, e.g. "
                              "--features FBCSP Riemann. Default: run all.")
    parser.add_argument("--classifiers", type=str, nargs="+", default=None,
                         help="only run these classifiers, e.g. --classifiers "
                              "LDA \"sLDA (shrinkage)\". Default: run all.")
    parser.add_argument("--cv", choices=["group", "stratified"], default="group",
                         help="'group' = GroupKFold by session (realistic, default). "
                              "'stratified' = plain StratifiedKFold, ignores session "
                              "boundaries — usually scores higher but partly via "
                              "session leakage, see run_benchmark docstring/comment.")
    args = parser.parse_args()

    X, y, groups, sfreq, ch_names = load_and_preprocess_all(
        args.recordings_dir,
        reference=args.reference,
        reject_artifacts=not args.no_artifact_rejection,
        ptp_threshold=args.ptp_threshold,
        grad_threshold=args.grad_threshold,
        max_bad_channel_frac=args.max_bad_channel_frac,
    )
    print(f"\nPooled dataset: {X.shape[0]} trials, {X.shape[1]} channels, "
          f"sfreq={sfreq} Hz, {len(np.unique(groups))} sessions")

    run_benchmark(X, y, groups, sfreq, ch_names, tmin=DEFAULT_TMIN,
                  n_splits=args.n_splits, cv_strategy=args.cv,
                  feature_names=args.features, classifier_names=args.classifiers)


if __name__ == "__main__":
    main()