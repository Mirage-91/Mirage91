"""Online (streaming) classification pipeline.

Two data sources, same downstream code either way:
- LSLSource: connects to a real live LSL EEG stream (competition day).
- ReplaySource: replays one of your recorded .xdf files as if it were live,
  chunk by chunk. Nice for try out purposes without needing the hardware.

Usage:
    # Test against a recorded file (fast, no hardware needed):
    python online_pipeline.py --model models/jump_only.joblib \
        --replay-file /path/to/session1_run1.xdf --speed 20

    # Real competition run:
    python online_pipeline.py --model models/jump_only.joblib

Output: pushes continuous class probabilities (float32) to an LSL
stream named 'ClassifierOutput', type 'ClassProb', at a steady rate.
Class labels are embedded directly in the stream's own metadata (channel
labels), so consumers (e.g. game_connector.py) can read the correct order
from the stream itself rather than needing to independently load the same
model file and trust both sides agree.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import scipy.signal as sps

from preprocessing import MAINS_FREQ_HZ
from utils import load_xdf_to_raw


# ============================================================
# Data sources — same interface (get_chunk), different backend
# ============================================================

class LSLSource:
    """Connects to a live LSL EEG stream."""

    def __init__(self, timeout: float = 10.0):
        from pylsl import StreamInlet, resolve_byprop

        print("Looking for an EEG stream...")
        streams = resolve_byprop("type", "EEG", timeout=timeout)
        if not streams:
            raise TimeoutError("No EEG stream found — check your amp/LSL app is running")
        info = streams[0]
        self.sfreq = info.nominal_srate()
        self.ch_names = self._extract_channel_names(info)
        self.inlet = StreamInlet(info)
        print(f"Connected: {len(self.ch_names)} channels @ {self.sfreq} Hz")

    @staticmethod
    def _extract_channel_names(info) -> list[str]:
        names = []
        ch = info.desc().child("channels").child("channel")
        while ch.name() == "channel":
            names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        return names

    def get_chunk(self) -> np.ndarray | None:
        chunk, _timestamps = self.inlet.pull_chunk()
        if not chunk:
            return None
        return np.array(chunk).T  # (channels, n_samples)


class ReplaySource:
    """Replays a recorded .xdf file as if it were a live stream, for testing
    the online loop without hardware. speed=1.0 is real-time; higher values
    replay faster (e.g. speed=20 chews through a session in a fraction of
    the time) for quick iteration."""

    def __init__(self, xdf_path: str | Path, speed: float = 1.0, chunk_size: int = 25):
        raw = load_xdf_to_raw(xdf_path)
        self.data = raw.get_data()  # (n_channels, n_times), volts
        self.sfreq = raw.info["sfreq"]
        self.ch_names = raw.ch_names
        self.speed = speed
        self.chunk_size = chunk_size
        self._pos = 0
        print(f"Replaying {xdf_path}: {len(self.ch_names)} channels @ "
              f"{self.sfreq} Hz, {self.data.shape[1] / self.sfreq:.0f}s of data, "
              f"speed={speed}x")

    def get_chunk(self) -> np.ndarray | None:
        if self._pos >= self.data.shape[1]:
            return None
        end = min(self._pos + self.chunk_size, self.data.shape[1])
        chunk = self.data[:, self._pos:end]
        self._pos = end
        if self.speed < float("inf"):
            time.sleep(self.chunk_size / self.sfreq / self.speed)
        return chunk


# ============================================================
# Causal streaming filter (matches preprocess.py's filter_raw),
# but with persisted state (zi) across chunks instead of one-shot on a
# whole recording.
# ============================================================

class CausalStreamFilter:
    def __init__(self, n_channels: int, sfreq: float,
                 notch_freq: float = MAINS_FREQ_HZ,
                 highpass_freq: float = 0.1, lowpass_freq: float = 60.0,
                 order: int = 4):
        self.b_notch, self.a_notch = sps.iirnotch(w0=notch_freq, Q=30, fs=sfreq)
        nyquist = sfreq / 2
        self.b_bp, self.a_bp = sps.butter(
            order, [highpass_freq / nyquist, lowpass_freq / nyquist], btype="band"
        )
        zi_notch = sps.lfilter_zi(self.b_notch, self.a_notch)
        zi_bp = sps.lfilter_zi(self.b_bp, self.a_bp)
        self.z_notch = np.tile(zi_notch, (n_channels, 1))
        self.z_bp = np.tile(zi_bp, (n_channels, 1))

    def apply(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: (n_channels, n_samples). Returns filtered chunk, same shape."""
        chunk, self.z_notch = sps.lfilter(self.b_notch, self.a_notch, chunk, axis=-1, zi=self.z_notch)
        chunk, self.z_bp = sps.lfilter(self.b_bp, self.a_bp, chunk, axis=-1, zi=self.z_bp)
        return chunk


# ============================================================
# Ring buffer
# ============================================================

#ring buffer is used for storing the last 6 seconds of data for the online pipeline. 
class RingBuffer:
    def __init__(self, n_channels: int, capacity: int):
        self.capacity = capacity
        self.data = np.zeros((n_channels, capacity))
        self.filled = 0

    def push(self, chunk: np.ndarray):
        n = chunk.shape[1]
        if n >= self.capacity:
            self.data[:] = chunk[:, -self.capacity:]
            self.filled = self.capacity
        else:
            self.data = np.roll(self.data, -n, axis=1)
            self.data[:, -n:] = chunk
            self.filled = min(self.filled + n, self.capacity)

    @property
    def is_full(self) -> bool:
        return self.filled >= self.capacity

    def snapshot(self) -> np.ndarray:
        return self.data.copy()


# ============================================================
# Main loop
# ============================================================

def emit_prediction(probs: np.ndarray, outlet=None):
    print(f"  -> probs: {np.round(probs, 3)}")
    if outlet is not None:
        outlet.push_sample(probs.tolist())


def make_output_outlet(class_order: list[str], nominal_srate: float = 10.0):
    """Embeds class_order into the stream's own metadata (channel labels),
    so consumers like game_connector.py can read the correct order directly
    from the stream itself — not by coincidentally loading the same model
    file a second time. Without this, a consumer's metadata-based lookup
    silently fails and falls back to the bundle file, which happens to work
    only as long as both processes are pointed at the same --model path."""
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(name="ClassifierOutput", type="ClassProb",
                       channel_count=len(class_order), nominal_srate=nominal_srate,
                       channel_format="float32", source_id="bci_pipeline_classifier")
    channels = info.desc().append_child("channels")
    for label in class_order:
        channels.append_child("channel").append_child_value("label", label)
    return StreamOutlet(info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="path to a .joblib saved by train_final_model.py")
    parser.add_argument("--replay-file", type=Path, default=None,
                         help="replay this .xdf file instead of connecting to a live LSL stream")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier (only with --replay-file)")
    parser.add_argument("--predict-interval", type=float, default=0.3,
                         help="seconds between predictions once the buffer is full")
    parser.add_argument("--no-output", action="store_true",
                         help="skip creating the LSL output stream (just print predictions)")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    pipeline = bundle["pipeline"]
    model_ch_names = bundle["ch_names"]
    window_n = bundle["window_n_samples"]
    classes = bundle["classes"]  # classes[i] = class name for predicted label i
    print(f"Loaded model: {bundle['feature_name']} + {bundle['classifier_name']}, "
          f"classes: {classes}, window={window_n} samples")

    has_proba = hasattr(pipeline, "predict_proba")
    # pipeline.classes_ holds the ascending label ints (0, 1, ...) the
    # classifier actually saw at fit time — always index into `classes`
    # through this rather than assuming an order, in either branch below.
    class_order = [classes[c] for c in pipeline.classes_]
    if has_proba:
        print(f"Output probability channel order: {class_order}")
    else:
        print(f"NOTE: {bundle['classifier_name']} has no predict_proba — "
              f"falling back to a one-hot vector (1.0/0.0) instead of a real "
              f"probability. Prefer LDA/sLDA/logistic-style classifiers here "
              f"if you want genuine confidence values, not just a hard decision.")
        print(f"Output vector channel order: {class_order}")

    if args.replay_file is not None:
        source = ReplaySource(args.replay_file, speed=args.speed)
    else:
        source = LSLSource()

    if abs(source.sfreq - bundle["sfreq"]) > 1e-3:
        print(f"WARNING: source sfreq ({source.sfreq}) != model's training sfreq "
              f"({bundle['sfreq']}) — filter coefficients will be wrong. Resample "
              f"or retrain before trusting predictions.")

    missing = [ch for ch in model_ch_names if ch not in source.ch_names]
    if missing:
        raise RuntimeError(f"Source is missing channels the model needs: {missing}. "
                            f"Source has: {source.ch_names}")
    channel_idx = [source.ch_names.index(ch) for ch in model_ch_names]

    stream_filter = CausalStreamFilter(len(model_ch_names), source.sfreq)
    buffer = RingBuffer(len(model_ch_names), window_n)

    outlet = None if args.no_output else make_output_outlet(
        class_order=class_order, nominal_srate=1.0 / args.predict_interval
    )

    last_predict_time = 0.0
    print("Starting online loop... (Ctrl+C to stop)")
    try:
        while True:
            chunk = source.get_chunk()
            if chunk is None:
                if isinstance(source, ReplaySource):
                    print("Replay finished.")
                    break
                time.sleep(0.01)
                continue

            chunk = chunk[channel_idx, :]
            chunk = stream_filter.apply(chunk)
            buffer.push(chunk)

            now = time.time()
            if buffer.is_full and (now - last_predict_time) >= args.predict_interval:
                last_predict_time = now
                window = buffer.snapshot()[np.newaxis, :, :]  # (1, n_channels, n_times)

                if has_proba:
                    probs = pipeline.predict_proba(window)[0]
                else:
                    pred = pipeline.predict(window)[0]
                    probs = np.array([1.0 if c == pred else 0.0 for c in pipeline.classes_])

                emit_prediction(probs, outlet)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()