"""Quick test: how well can we tell a SUBSET of classes apart, instead of
all 4? Useful for any game that only needs some of your classes — e.g.
BrainSki 1 (Jump Only) needs just "feet vs rest" (2 classes); a jump+duck
or up/down style game needs "class_A vs class_B vs rest" (3 classes) so the
model can express "neither" as well as which of the two active commands.

Run AFTER run_pipeline.py has been run at least once with matching settings
(same --recordings-dir, --use-ica, --reference, rejection thresholds) so
this can reuse the cached preprocessed data instead of reloading from
scratch (~10min -> ~instant).

Run:
    # 2 classes (binary)
    python test_subtask.py --classes feet rest

    python test_subtask.py --classes right_hand rest \
        --recordings-dir data
      --features FBCSP Riemann \
      --classifiers LDA "sLDA (shrinkage)" "SVM (linear)" "SVM (RBF)"

    # 3 classes — flat joint classifier, e.g. for a jump+duck or up/down game
    python test_subtask.py --classes left_hand feet rest \
      --features FBCSP Riemann \
      --classifiers LDA "sLDA (shrinkage)" "SVM (linear)"

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from classifiers import get_classifiers
from utils import EVENT_ID
from run_pipeline import (
    DEFAULT_GRAD_THRESHOLD,
    DEFAULT_MAX_BAD_CHANNEL_FRAC,
    DEFAULT_PTP_THRESHOLD,
    DEFAULT_TMIN,
    build_feature_extractors,
    load_and_preprocess_all,
    run_benchmark,
)


def run_hierarchical_benchmark(X, y, groups, sfreq, ch_names, tmin,
                                class_a_id, class_b_id, rest_id,
                                n_splits, feature_names, classifier_names):
    """Cross-validated (GroupKFold) accuracy of the hierarchical decomposition:
    gate (active vs rest) + conditional (class_A vs class_B), fit fresh on
    each training fold, combined via chain-rule probabilities on each held-out
    fold. Same feature extractor + classifier is used for both stages per
    combo tested (keeps the grid a manageable size); mix-and-match stages
    manually via train_hierarchical_model.py later if one wins clearly.
    """
    extractors_all = build_feature_extractors(sfreq, ch_names, tmin)
    classifiers_all = get_classifiers()

    feat_names = feature_names or list(extractors_all)
    clf_names = classifier_names or list(classifiers_all)
    unknown_f = set(feat_names) - set(extractors_all)
    if unknown_f:
        raise ValueError(f"unknown feature name(s) {sorted(unknown_f)}; available: {sorted(extractors_all)}")
    unknown_c = set(clf_names) - set(classifiers_all)
    if unknown_c:
        raise ValueError(f"unknown classifier name(s) {sorted(unknown_c)}; available: {sorted(classifiers_all)}")

    mask_all = np.isin(y, [class_a_id, class_b_id, rest_id])
    X_all, y_all, groups_all = X[mask_all], y[mask_all], groups[mask_all]
    y_gate_all = np.where(np.isin(y_all, [class_a_id, class_b_id]), 1, 0)  # 1=active, 0=rest
    y_true_3class = np.where(y_all == class_a_id, 0, np.where(y_all == class_b_id, 1, 2))

    n_splits = min(n_splits, len(np.unique(groups_all)))
    if n_splits < 2:
        raise RuntimeError("Need at least 2 sessions represented across this class set.")
    cv = GroupKFold(n_splits=n_splits)

    results = []
    for feat_name in feat_names:
        for clf_name in clf_names:
            fold_accs = []
            for train_idx, test_idx in cv.split(X_all, y_true_3class, groups_all):
                X_train, X_test = X_all[train_idx], X_all[test_idx]
                y_gate_train = y_gate_all[train_idx]
                y_true_test = y_true_3class[test_idx]


                gate_pipe = Pipeline([
                    ("features", build_feature_extractors(sfreq, ch_names, tmin)[feat_name]),
                    ("classifier", get_classifiers()[clf_name]),
                ])
                gate_pipe.fit(X_train, y_gate_train)


                active_train_mask = y_gate_train == 1
                X_train_active = X_train[active_train_mask]
                y_train_active_raw = y_all[train_idx][active_train_mask]
                y_cond_train = np.where(y_train_active_raw == class_a_id, 0, 1)

                cond_pipe = Pipeline([
                    ("features", build_feature_extractors(sfreq, ch_names, tmin)[feat_name]),
                    ("classifier", get_classifiers()[clf_name]),
                ])
                cond_pipe.fit(X_train_active, y_cond_train)

                gate_probs = gate_pipe.predict_proba(X_test)
                p_active = gate_probs[:, list(gate_pipe.classes_).index(1)]
                p_rest = gate_probs[:, list(gate_pipe.classes_).index(0)]

                cond_probs = cond_pipe.predict_proba(X_test)
                p_a_given_active = cond_probs[:, list(cond_pipe.classes_).index(0)]
                p_b_given_active = cond_probs[:, list(cond_pipe.classes_).index(1)]

                combined = np.stack(
                    [p_active * p_a_given_active, p_active * p_b_given_active, p_rest], axis=1
                )
                pred = combined.argmax(axis=1)
                fold_accs.append((pred == y_true_test).mean())

            mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
            print(f"{feat_name:12s} + {clf_name:20s} | acc: {mean_acc:.3f} +/- {std_acc:.3f}")
            results.append((feat_name, clf_name, mean_acc, std_acc))

    results.sort(key=lambda r: r[2], reverse=True)
    print("\n" + "=" * 60)
    print("RANKED RESULTS (best first) — HIERARCHICAL")
    print("=" * 60)
    for feat_name, clf_name, mean_acc, std_acc in results:
        print(f"{feat_name:12s} + {clf_name:20s} | acc: {mean_acc:.3f} +/- {std_acc:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--reference", choices=["car", "laplacian"], default="car")
    parser.add_argument("--use-ica", action="store_true")
    parser.add_argument("--no-artifact-rejection", action="store_true")
    parser.add_argument("--ptp-threshold", type=float, default=DEFAULT_PTP_THRESHOLD)
    parser.add_argument("--grad-threshold", type=float, default=DEFAULT_GRAD_THRESHOLD)
    parser.add_argument("--max-bad-channel-frac", type=float, default=DEFAULT_MAX_BAD_CHANNEL_FRAC)
    parser.add_argument("--cache-dir", type=Path, default=Path(".pipeline_cache"))
    parser.add_argument("--classes", type=str, nargs="+", required=True,
                         choices=list(EVENT_ID.keys()),
                         help="2 or more classes to test, e.g. --classes feet rest, "
                              "or --classes left_hand feet rest. Order given is the "
                              "label order (label 0 = first class listed), matching "
                              "train_final_model.py's convention.")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--features", type=str, nargs="+", default=None,
                         help="only run these feature extractors, e.g. --features FBCSP Riemann")
    parser.add_argument("--classifiers", type=str, nargs="+", default=None,
                         help="only run these classifiers, e.g. --classifiers LDA \"sLDA (shrinkage)\"")
    args = parser.parse_args()

    if len(args.classes) < 2:
        raise ValueError(f"need at least 2 classes, got {args.classes!r}")
    if len(set(args.classes)) != len(args.classes):
        raise ValueError(f"--classes has duplicate entries: {args.classes!r}")

    X, y, groups, sfreq, ch_names = load_and_preprocess_all(
        args.recordings_dir,
        use_ica=args.use_ica,
        reference=args.reference,
        reject_artifacts=not args.no_artifact_rejection,
        ptp_threshold=args.ptp_threshold,
        grad_threshold=args.grad_threshold,
        max_bad_channel_frac=args.max_bad_channel_frac,
        cache_dir=args.cache_dir,
    )

    class_ids = [EVENT_ID[c] for c in args.classes]
    id_to_label = {cid: label for label, cid in enumerate(class_ids)}
    mask = np.isin(y, class_ids)
    X_sub, groups_sub = X[mask], groups[mask]
    y_sub = np.array([id_to_label[v] for v in y[mask]])

    print(f"\n{' vs '.join(args.classes)}: {len(y_sub)} trials")
    for label, name in enumerate(args.classes):
        print(f"    label {label} = {name}: {(y_sub == label).sum()} trials")
    print(f"    chance level: {1 / len(args.classes):.3f}")

    n_splits = min(args.n_splits, len(np.unique(groups_sub)))
    if n_splits < 2:
        raise RuntimeError("Need at least 2 sessions represented across this class set.")

    run_benchmark(X_sub, y_sub, groups_sub, sfreq, ch_names, tmin=DEFAULT_TMIN,
                  n_splits=n_splits, cv_strategy="group",
                  feature_names=args.features, classifier_names=args.classifiers)


if __name__ == "__main__":
    main()