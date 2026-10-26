"""Game connector — bridges online_pipeline.py's LSL 'ClassifierOutput' stream
to the Cortical Peaks Challenge 2026 game server.

Uses the REFERENCE BCINetworkClient and the server's ACTUAL per-game input mapping, read
directly out of shared/connection/server.py in that repo:

    dino_jump (BSJ)         : INPUT_A=jump
    dino      (BSDJ)        : INPUT_A=jump, INPUT_B=duck
    pong / pong_ai (PVP/AI) : INPUT_C=up,   INPUT_D=down
    ski / ski_dyn (SK1/SK2) : INPUT_A=rotate_left, INPUT_B=rotate_right,
                              INPUT_C=forward, INPUT_D=backward

Must be run from inside a checkout of that repo (or with it on PYTHONPATH) -
it needs `shared.connection.protocol` to build a valid CmdMessage.

Usage:
    python game_connector.py --model models/dino.joblib --game dino_jump
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from pylsl import StreamInlet, resolve_byprop

from shared.connection.protocol import (
    AckMessage,
    CmdMessage,
    DeviceType,
    PingMessage,
    PongMessage,
    RegisterMessage,
    ServerMessage,
    ServerPingMessage,
    ServerPongMessage,
    StateMessage,
    parse_server_message,
    to_bytes,
)

SERVER_ADDRESS: tuple[str, int] = ("127.0.0.1", 5000)
BCI_UID: str = "uid-" + str(uuid4())[:4]
BUFFER_SIZE: int = 65535
TIMEOUT: timedelta = timedelta(seconds=3)
SOCKET_TIMEOUT: timedelta = timedelta(seconds=2)

CONFIDENCE_THRESHOLD = 0.6
MIN_RESEND_INTERVAL = 0.5  # seconds - don't re-send an unchanged command faster than this

log = logging.getLogger(__name__)


# ============================================================
# BCINetworkClient — copied verbatim from examples/bci_client.py.
# Do not edit this class; everything project-specific lives below it.
# ============================================================

class BCINetworkClient:
    def __init__(self) -> None:
        self.sock: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(SOCKET_TIMEOUT.total_seconds())

        self.connected: bool = False
        self.session_token: str = ""
        self.last_received_time: float = time.time()

        self._listener_thread = threading.Thread(target=self._network_listener, daemon=True)
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
        self._listener_thread.start()
        self._heartbeat_thread.start()

    def send_command(self, cmd: str) -> None:
        if self.connected:
            msg = CmdMessage(session_token=self.session_token, device=DeviceType.BCI, content=cmd)
            self.sock.sendto(to_bytes(msg), SERVER_ADDRESS)

    def _network_listener(self) -> None:
        while True:
            if not self.connected:
                reg = RegisterMessage(display_name=BCI_UID, device=DeviceType.BCI)
                self.sock.sendto(to_bytes(reg), SERVER_ADDRESS)
                time.sleep(1.0)

            try:
                data, _ = self.sock.recvfrom(BUFFER_SIZE)
                msg = parse_server_message(data)
                self._handle_server_message(msg)
            except TimeoutError:
                pass
            except ValueError:
                log.debug("Dropped malformed server packet")
            except OSError:
                time.sleep(0.1)

    def _handle_server_message(self, msg: ServerMessage) -> None:
        match msg:
            case AckMessage(session_token=token):
                self.session_token = token
                self.connected = True
                self.last_received_time = time.time()
                log.info("Authenticated with token %s", token)

            case ServerPingMessage():
                self.last_received_time = time.time()
                pong = PongMessage(session_token=self.session_token, device=DeviceType.BCI)
                self.sock.sendto(to_bytes(pong), SERVER_ADDRESS)

            case ServerPongMessage():
                self.last_received_time = time.time()

            case StateMessage():
                pass  # BCI clients do not get game state messages

    def _heartbeat_worker(self) -> None:
        while True:
            if self.connected:
                ping = PingMessage(session_token=self.session_token, device=DeviceType.BCI)
                self.sock.sendto(to_bytes(ping), SERVER_ADDRESS)

                if time.time() - self.last_received_time > TIMEOUT.total_seconds():
                    log.warning("Server timed out — reconnecting")
                    self.connected = False
                    self.session_token = ""

            time.sleep(1.0)


# ============================================================
# Server-verified action -> INPUT_x per game. Copied from _DINO_INPUT /
# _PONG_INPUT / _SKI_INPUT in shared/connection/server.py (inverted: we're
# the sender, the server's dicts go INPUT_x -> action, so this direction is
# action -> INPUT_x).
# ============================================================

_ACTION_TO_INPUT: dict[str, dict[str, str]] = {
    "dino_jump": {"jump": "INPUT_A"},
    "dino": {"jump": "INPUT_A", "duck": "INPUT_B"},
    "pong": {"up": "INPUT_C", "down": "INPUT_D"},
    "pong_ai": {"up": "INPUT_C", "down": "INPUT_D"},
    "ski": {
        "rotate_left": "INPUT_A", "rotate_right": "INPUT_B",
        "forward": "INPUT_C", "backward": "INPUT_D"
    },
    "ski_dyn": {
        "rotate_left": "INPUT_A", "rotate_right": "INPUT_B",
        "forward": "INPUT_C", "backward": "INPUT_D"
    },
}

CLASS_TO_ACTION: dict[str, dict[str, str]] = {
    "dino_jump": {"right_hand": "jump"},
    "dino": {"feet": "jump", "right_hand": "duck"},
    "pong": {"right_hand": "up", "feet": "down"},
    "pong_ai": {"right_hand": "up", "feet": "down"},
    "ski": {"left_hand": "rotate_left", "right_hand": "rotate_right", "feet": "forward"},
    "ski_dyn": {"left_hand": "rotate_left", "right_hand": "rotate_right", "feet": "forward"},
}


def connect_classifier_stream(timeout: float = 10.0) -> StreamInlet:
    print("Looking for ClassifierOutput stream...")
    streams = resolve_byprop("name", "ClassifierOutput", timeout=timeout)
    if not streams:
        raise TimeoutError("No ClassifierOutput stream found - is online_pipeline.py running?")
    print("Connected to ClassifierOutput.")
    return StreamInlet(streams[0])


def get_class_order(inlet: StreamInlet, bundle: dict) -> list[str]:
    """Prefer channel labels embedded in the LSL stream's own metadata (per
    the README); fall back to the model bundle's class order if the stream
    doesn't carry labels (e.g. an older online_pipeline.py)."""
    try:
        names: list[str] = []
        ch = inlet.info().desc().child("channels").child("channel")
        while ch.name() == "channel":
            names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        if names and all(names):
            return names
    except Exception:
        pass
    return [bundle["classes"][c] for c in bundle["pipeline"].classes_]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                         help="same .joblib bundle passed to online_pipeline.py")
    parser.add_argument("--game", required=True, choices=list(_ACTION_TO_INPUT),
                         help="dino_jump=BSJ, dino=BSDJ, pong=PVP, pong_ai=AI, ski=SK1, ski_dyn=SK2")
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    bundle = joblib.load(args.model)
    action_map = _ACTION_TO_INPUT[args.game]
    class_map = CLASS_TO_ACTION[args.game]

    bci = BCINetworkClient()
    inlet = connect_classifier_stream()
    class_order = get_class_order(inlet, bundle)
    print(f"Class order: {class_order}")
    print(f"Active mapping for --game {args.game}: "
          f"{ {c: action_map[a] for c, a in class_map.items()} }")

    last_sent_cmd: str | None = None
    last_sent_time = 0.0

    print("Waiting for predictions... (Ctrl+C to stop)")
    try:
        while True:
            sample, _timestamp = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue

            probs = np.array(sample)
            pred_idx = int(np.argmax(probs))
            label = class_order[pred_idx]
            confidence = float(probs[pred_idx])

            if confidence < args.confidence_threshold:
                continue
            action = class_map.get(label)
            if action is None:
                continue  # e.g. "rest", or a class this game doesn't use
            cmd = action_map[action]

            now = time.time()
            if cmd == last_sent_cmd and (now - last_sent_time) < MIN_RESEND_INTERVAL:
                continue  # avoid flooding the server with the same repeated command

            bci.send_command(cmd)
            last_sent_cmd = cmd
            last_sent_time = now
            print(f"-> sent {cmd} ({action}) | label={label} confidence={confidence:.2f}")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()