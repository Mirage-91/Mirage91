"""Feature extractors, unified behind the sklearn Transformer interface
(fit(X, y), transform(X) -> feature matrix), where X is always an epoched
array of shape (n_trials, n_channels, n_times) as returned by
epochs.make_epochs(...).get_data().
 
Every extractor receives ``times`` (= ``epochs.times``), the clock of the
epochs, because get_data() returns a plain array without it. Time windows
given in seconds are turned into a boolean mask over that clock.
"""
 
from __future__ import annotations
 
import mne
import numpy as np
from scipy.linalg import eigh
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin
 
try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False
 
 
## Utility function only needed for the features
 
def _window_mask(times, window, n_times=None):
    """Boolean mask over the time axis selecting a window given in seconds.
 
    The epochs are longer than the part each feature needs (pre-cue baseline,
    cue reaction, end of trial), and different features use different windows.
    ``times`` is the clock of the epochs (``epochs.times``), so the window is
    compared directly in seconds and no index arithmetic is needed.
 
    The start is included and the end is excluded. A small tolerance avoids
    missing a sample because of float rounding (e.g. 0.5 vs 0.49999999).
    Use it as ``X[..., mask]``.
    """
    if times is None:
        raise ValueError("`times` is required: pass epochs.times to the feature extractor.")
    times = np.asarray(times)
    if n_times is not None and len(times) != n_times:
        raise ValueError(
            f"`times` has {len(times)} samples but X has {n_times}; "
            "they must come from the same epochs."
        )
    mask = (times >= window[0] - 1e-9) & (times < window[1] - 1e-9)
    if not mask.any():
        raise ValueError(
            f"Window {window} contains no samples (epochs span "
            f"{times[0]:.3f} to {times[-1]:.3f} s)."
        )
    return mask
 
 
# ============================================================
# 1. Bandpower features
# ============================================================
 
class BandpowerFeatures(BaseEstimator, TransformerMixin):
    # One log-bandpower feature per (channel, band) config
 
    def __init__(self, class_configs, ch_names, sfreq, times, window=(0.5, 3.5)):
        self.class_configs = class_configs
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.times = times
        self.window = window
 
    # fit() just precomputes the channel indices for each class config, so we don't have to look them up every time in transform()
    def fit(self, X, y=None):
        self.channel_indices_ = {
            name: self.ch_names.index(ch)
            for name, (ch, _, _) in self.class_configs.items()
        }
        return self
 
    # transform() computes the log-bandpower features for each trial, using the precomputed channel indices and the specified frequency bands and time window
    def transform(self, X):
        mask = _window_mask(self.times, self.window, X.shape[-1])
        feats = np.zeros((X.shape[0], len(self.class_configs)), dtype=float)
        for i, (name, (ch, l_freq, h_freq)) in enumerate(self.class_configs.items()):
            sig = X[:, self.channel_indices_[name], :]
            sig_filt = mne.filter.filter_data(
                sig, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq,
                method="iir", verbose=False
            )
            sig_win = sig_filt[:, mask]
            feats[:, i] = np.log(np.sum(sig_win ** 2, axis=-1) + 1e-12)
        return feats
 
 
# ============================================================
# 2. CSP features — one-vs-rest, per class
# ============================================================
 
# utility for csp
def _compute_cov(trial: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    cov = np.cov(trial)
    cov /= np.trace(cov)
    cov += reg * np.eye(cov.shape[0])  # is added so that the covariance matrix is positive definite and invertible
    return cov
 
 
def _csp_filters_binary(cov_a: np.ndarray, cov_b: np.ndarray, n_components: int) -> np.ndarray:
    eigvals, eigvecs = eigh(cov_a, cov_a + cov_b)
    idx = np.argsort(eigvals)
    sorted_vecs = eigvecs[:, idx]
    n = n_components // 2
    filters = np.concatenate([sorted_vecs[:, :n], sorted_vecs[:, -n:]], axis=1)
    return filters.T
 
 
class CSPFeatures(BaseEstimator, TransformerMixin):
 
    def __init__(self, n_components=6, l_freq=5.0, h_freq=25.0, sfreq=None, times=None, window=(1.0, 3.0)):
        self.n_components = n_components
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.sfreq = sfreq
        self.times = times
        self.window = window
 
    # fit() computes the class-wise covariance matrices and derives the CSP filters for each class, storing them for later use in transform().
    def fit(self, X, y):
        X_filt = mne.filter.filter_data(
            X, sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq,
            method="iir", verbose=False
        )
        mask = _window_mask(self.times, self.window, X.shape[-1])
        X_win = X_filt[..., mask]
 
        classes = np.unique(y)
        self.classes_ = classes
        class_covs = {}
        for c in classes:
            trials = X_win[y == c]
            covs = [_compute_cov(trial) for trial in trials]
            class_covs[c] = np.mean(covs, axis=0)
 
        self.filters_ = {}
        for c in classes:
            cov_c = class_covs[c]
            cov_rest = np.mean([class_covs[other] for other in classes if other != c], axis=0)
            self.filters_[c] = _csp_filters_binary(cov_c, cov_rest, self.n_components)
        return self
 
    # transform() applies the precomputed CSP filters to each trial, computing the log-variance of the filtered signals as features.
    def transform(self, X):
        X_filt = mne.filter.filter_data(
            X, sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq,
            method="iir", verbose=False
        )
        mask = _window_mask(self.times, self.window, X.shape[-1])
        X_win = X_filt[..., mask]
 
        feature_list = []
        for trial in X_win:
            trial_feats = []
            for c in self.classes_:
                W = self.filters_[c]
                filtered = W @ trial
                log_var = np.log(np.var(filtered, axis=1) + 1e-12)  # with the var we square and center
                trial_feats.append(log_var)
            feature_list.append(np.concatenate(trial_feats))
        return np.array(feature_list)
 
 
# ============================================================
# 3. Filter-Bank CSP (from train_classifier_rest_vs_movement.py)
# ============================================================
 
class FilterBankCSP(BaseEstimator, TransformerMixin):
    def __init__(self, bands, sfreq, times, n_components=4, window=(1.0, 3.0)):
        self.bands = bands
        self.sfreq = sfreq
        self.times = times
        self.n_components = n_components
        self.window = window
 
    def fit(self, X, y):
        from mne.decoding import CSP
 
        mask = _window_mask(self.times, self.window, X.shape[-1])
        self.csp_by_band_ = []
        for l_freq, h_freq in self.bands:
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False
            )
            X_win = X_band[..., mask]
            csp = CSP(n_components=self.n_components, reg="ledoit_wolf", log=True, norm_trace=False)
            csp.fit(X_win, y)
            self.csp_by_band_.append(csp)
        return self
 
    def transform(self, X):
        mask = _window_mask(self.times, self.window, X.shape[-1])
        band_features = []
        for (l_freq, h_freq), csp in zip(self.bands, self.csp_by_band_):
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False
            )
            X_win = X_band[..., mask]
            band_features.append(csp.transform(X_win))
        return np.concatenate(band_features, axis=1)
 
 
# ============================================================
# 4. Riemannian tangent-space features
# ============================================================
 
class RiemannianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, bands, sfreq, times, window=(1.0, 3.0)):
        self.bands = bands
        self.sfreq = sfreq
        self.times = times
        self.window = window
 
    def fit(self, X, y=None):
        mask = _window_mask(self.times, self.window, X.shape[-1])
        self.tangent_spaces_ = []
        for l_freq, h_freq in self.bands:
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False
            )
            X_win = X_band[..., mask]
            covs = Covariances(estimator="lwf").fit_transform(X_win)
            ts = TangentSpace(metric="riemann").fit(covs)
            self.tangent_spaces_.append(ts)
        return self
 
    def transform(self, X):
        mask = _window_mask(self.times, self.window, X.shape[-1])
        ts_features = []
        for (l_freq, h_freq), ts in zip(self.bands, self.tangent_spaces_):
            X_band = mne.filter.filter_data(
                X, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False
            )
            X_win = X_band[..., mask]
            covs = Covariances(estimator="lwf").transform(X_win)
            ts_features.append(ts.transform(covs))
        return np.concatenate(ts_features, axis=1)
 
 
# ============================================================
# 5. ERDS (baseline-relative power) features
# ============================================================
 
class ERDSFeatures(BaseEstimator, TransformerMixin):
    """Baseline-relative power change per (channel, band, window) spec. Needs
    the epoch to include a pre-cue baseline window."""
 
    def __init__(self, specs, ch_names, sfreq, times, baseline_window):
        self.specs = specs
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.times = times
        self.baseline_window = baseline_window
 
    def fit(self, X, y=None):
        self.channel_indices_ = {s["name"]: self.ch_names.index(s["channel"]) for s in self.specs}
        return self
 
    def transform(self, X):
        n_times = X.shape[-1]
        feats = np.zeros((X.shape[0], len(self.specs)), dtype=float)
        base_mask = _window_mask(self.times, self.baseline_window, n_times)
 
        for i, spec in enumerate(self.specs):
            sig = X[:, self.channel_indices_[spec["name"]], :]
            l_freq, h_freq = spec["band"]
            sig_filt = mne.filter.filter_data(
                sig, sfreq=self.sfreq, l_freq=l_freq, h_freq=h_freq, method="iir", verbose=False
            )
            power = np.abs(hilbert(sig_filt, axis=-1)) ** 2
 
            win_mask = _window_mask(self.times, spec["window"], n_times)
            baseline_power = power[:, base_mask].mean(axis=1)
            window_power = power[:, win_mask].mean(axis=1)
            feats[:, i] = (window_power - baseline_power) / (baseline_power + 1e-8)
 
        return feats
 