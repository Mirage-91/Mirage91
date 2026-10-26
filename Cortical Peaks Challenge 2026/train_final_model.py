"""Fit a chosen binary (or n-class) combo on ALL your data and save it,
ready for the online pipeline to load. This is the "final training" step —
run it once per game/command-set you're deploying, after you've picked a
winner using run_pipeline.py / test_binary_subtask.py's cross-validated
numbers. This script does NOT cross-validate — for deployment you want the
model trained on every trial you have, not held-out folds.
 
Usage:
    # dino_jump (right_hand vs rest) only
    python train_final_model.py --classes right_hand rest \
        --feature Riemann --classifier "sLDA (shrinkage)" \
        --model-name dino_jump
 
    # dino (jump and duck) as well as pong and pong ai
    python train_final_model.py --classes right_hand feet rest \
        --feature Riemann --classifier "SVM (linear)" \
        --model-name dino
 
    # ski and ski dyn— omit --classes to get all, sadly we cant go backwards yet (1 class missing)
    python train_final_model.py \
        --feature Riemann --classifier "sLDA (shrinkage)" \
        --model-name brainski
 

Saves to ./models/<model-name>.joblib — a dict containing the fitted
pipeline PLUS every setting needed to replicate preprocessing online
(reference, ch_names order, sfreq, epoch
tmin/tmax the model was trained on, and the class list in label order —
label i in the model's output corresponds to classes[i]).
"""
 
from __future__ import annotations
 
import argparse
import json
from pathlib import Path
 
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
 
from classifiers import get_classifiers
from epochs import DEFAULT_TMAX, DEFAULT_TMIN
from utils import EVENT_ID
from run_pipeline import (
    DEFAULT_GRAD_THRESHOLD,
    DEFAULT_MAX_BAD_CHANNEL_FRAC,
    DEFAULT_PTP_THRESHOLD,
    build_feature_extractors,
    load_and_preprocess_all,
)
 
# Keep generated models beside the project scripts, regardless of the
# directory the command is run from.
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--reference", choices=["car", "laplacian"], default="car")
    parser.add_argument("--no-artifact-rejection", action="store_true")
    parser.add_argument("--ptp-threshold", type=float, default=DEFAULT_PTP_THRESHOLD)
    parser.add_argument("--grad-threshold", type=float, default=DEFAULT_GRAD_THRESHOLD)
    parser.add_argument("--max-bad-channel-frac", type=float, default=DEFAULT_MAX_BAD_CHANNEL_FRAC)
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                         choices=list(EVENT_ID.keys()),
                         help="which classes to train on, e.g. --classes right_hand "
                              "rest, or --classes left_hand right_hand feet rest. "
                              "The order given is the label order the model outputs "
                              "(label 0 = first class listed, etc). Omit to use "
                              f"every class in EVENT_ID ({list(EVENT_ID.keys())}).")
    parser.add_argument("--feature", required=True,
                         choices=["Bandpower", "CSP", "FBCSP", "ERDS", "Riemann"])
    parser.add_argument("--classifier", required=True,
                         help='e.g. LDA, "sLDA (shrinkage)", "SVM (linear)", '
                              '"SVM (RBF)", "Random Forest", "Gradient Boosting"')
    parser.add_argument("--model-name", required=True,
                         help="output filename (without extension), e.g. 'jump_only'")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="where to save the model. Defaults to models/ beside this script.")
    args = parser.parse_args()
 
    classes = args.classes if args.classes is not None else list(EVENT_ID.keys())
    if len(classes) < 2:
        raise ValueError(f"need at least 2 classes to train a classifier, got {classes!r}")
    if len(set(classes)) != len(classes):
        raise ValueError(f"--classes has duplicate entries: {classes!r}")
 
    X, y, groups, sfreq, ch_names = load_and_preprocess_all(
        args.recordings_dir,
        reference=args.reference,
        reject_artifacts=not args.no_artifact_rejection,
        ptp_threshold=args.ptp_threshold,
        grad_threshold=args.grad_threshold,
        max_bad_channel_frac=args.max_bad_channel_frac,
    )
 
    # Map the chosen class names to their raw event codes, then to a
    # contiguous 0..n-1 label per the order `classes` was given in.
    class_ids = [EVENT_ID[c] for c in classes]
    id_to_label = {cid: label for label, cid in enumerate(class_ids)}
 
    mask = np.isin(y, class_ids)
    X_sub = X[mask]
    y_sub = np.array([id_to_label[v] for v in y[mask]])
 
    print(f"Training on ALL {len(y_sub)} trials across {len(classes)} classes:")
    for label, name in enumerate(classes):
        print(f"    label {label} = {name}: {(y_sub == label).sum()} trials")
 
    extractors = build_feature_extractors(sfreq, ch_names, DEFAULT_TMIN)
    classifiers = get_classifiers()
    if args.feature not in extractors:
        raise ValueError(f"unknown/unavailable feature extractor {args.feature!r}; "
                          f"available: {list(extractors)}")
    if args.classifier not in classifiers:
        raise ValueError(f"unknown classifier {args.classifier!r}; "
                          f"available: {list(classifiers)}")
 
    pipeline = Pipeline([
        ("features", extractors[args.feature]),
        ("classifier", classifiers[args.classifier]),
    ])
    pipeline.fit(X_sub, y_sub)
    train_acc = pipeline.score(X_sub, y_sub)
    print(f"Fit complete. Training-set accuracy (NOT a generalization estimate, "
          f"just a sanity check the fit worked): {train_acc:.3f}")
 
    args.models_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.models_dir / f"{args.model_name}.joblib"
 
    bundle = {
        "pipeline": pipeline,
        "feature_name": args.feature,
        "classifier_name": args.classifier,
        "classes": classes,   # classes[i] is the class name for predicted label i
        "sfreq": sfreq,
        "ch_names": ch_names,           # exact order the model expects channels in
        "reference": args.reference,
        "epoch_tmin": DEFAULT_TMIN,
        "epoch_tmax": DEFAULT_TMAX,
        "window_n_samples": int(round((DEFAULT_TMAX - DEFAULT_TMIN) * sfreq)) + 1,
    }
    joblib.dump(bundle, out_path)
    print(f"Saved to {out_path}")
 
    with open(args.models_dir / f"{args.model_name}.json", "w") as f:
        json.dump({k: v for k, v in bundle.items() if k != "pipeline"}, f, indent=2)
 
 
if __name__ == "__main__":
    main()
 