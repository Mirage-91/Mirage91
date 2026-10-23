from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_classifiers() -> dict:
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "sLDA (shrinkage)": LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
        "SVM (linear)": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="linear"))]),
        "SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=1.0))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=67),
        "Gradient Boosting": GradientBoostingClassifier(random_state=67),
    }