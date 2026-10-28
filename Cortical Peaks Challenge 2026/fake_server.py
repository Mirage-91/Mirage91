"""To test out if the server connection works.

Only implements what a BCI client actually touches: REGISTER -> ACK,
client PING -> ServerPong, server PING -> expects client PONG back, and
prints every CMD it receives. It does NOT run any actual game logic —
this is purely for checking "does my client speak the protocol
correctly and stay connected", before the real games/ package exists
locally.

Usage:
    python fake_server.py
    # in another terminal:
    python game_connector_2.py --model models/jump_only.joblib --game dino_jump
"""

from __future__ import annotations

import os
import socket
import time

from shared.connection.protocol import (
    AckMessage,
    CmdMessage,
    DeviceType,
    PingMessage,
    PongMessage,
    RegisterMessage,
    ServerPingMessage,
    ServerPongMessage,
    parse_client_message,
    to_bytes,
)

HOST = "0.0.0.0"
PORT = 5000
BUFFER_SIZE = 65535
HEARTBEAT_INTERVAL = 1.0  # how often we send a server-initiated PING to the client

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST, PORT))
sock.settimeout(0.2)

print(f"Fake server listening on {HOST}:{PORT} (Ctrl+C to stop)")

client_addr: tuple[str, int] | None = None
client_token: str | None = None
last_heartbeat_sent = 0.0
cmd_count = 0

try:
    while True:
        now = time.time()

        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
        except TimeoutError:
            data, addr = None, None

        if data is not None:
            try:
                msg = parse_client_message(data)
            except ValueError as exc:
                print(f"  [!] dropped malformed packet from {addr}: {exc}")
                data = None

        if data is not None:
            match msg:
                case RegisterMessage(display_name=name, device=DeviceType.BCI):
                    client_addr = addr
                    client_token = os.urandom(8).hex()
                    sock.sendto(to_bytes(AckMessage(session_token=client_token)), addr)
                    print(f"[REGISTER] '{name}' from {addr} -> issued token {client_token}")

                case PingMessage(session_token=token, device=DeviceType.BCI):
                    if token == client_token:
                        sock.sendto(to_bytes(ServerPongMessage()), addr)
                    else:
                        print(f"  [!] PING with unknown/stale token {token!r} from {addr}")

                case PongMessage(session_token=token, device=DeviceType.BCI):
                    pass  # response to our own server-initiated ping, nothing to do

                case CmdMessage(session_token=token, device=DeviceType.BCI, content=content):
                    cmd_count += 1
                    tag = "OK" if token == client_token else "STALE TOKEN"
                    print(f"[CMD #{cmd_count}] {content!r}  ({tag})")

                case _:
                    print(f"  [?] unhandled/unexpected message: {msg!r}")

        # Server-initiated heartbeat, mirroring what the real server does
        if client_addr and (now - last_heartbeat_sent) >= HEARTBEAT_INTERVAL:
            sock.sendto(to_bytes(ServerPingMessage()), client_addr)
            last_heartbeat_sent = now

except KeyboardInterrupt:
    print("\nStopped.")
