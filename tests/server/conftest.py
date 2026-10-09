"""Fixtures that start nllb-server (or nllb-server-mock) on a free port.

Environment:
    NLLB_SERVER_BIN    server binary   [build/bin/nllb-server-mock]
    NLLB_SERVER_MODEL  -m argument     [mock]  (a real .gguf for -m real)
    NLLB_CLI_BIN       nllb-cli        [build/bin/nllb-cli]   (real tests only)
    NLLB_SERVER_ARGS   extra args appended to every server command line
"""
from __future__ import annotations

import os
import shlex
import socket
import subprocess
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: str) -> str:
    p = os.environ.get(env, default)
    return str((ROOT / p).resolve()) if not os.path.isabs(p) else p


SERVER_BIN = _path("NLLB_SERVER_BIN", "build/bin/nllb-server-mock")
MODEL = os.environ.get("NLLB_SERVER_MODEL", "mock")
CLI_BIN = _path("NLLB_CLI_BIN", "build/bin/nllb-cli")
EXTRA = shlex.split(os.environ.get("NLLB_SERVER_ARGS", ""))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """One running server process. Use .url(path), .get(), .post()."""

    def __init__(self, *args: str, ready_timeout: float = 120.0):
        if not os.path.exists(SERVER_BIN):
            pytest.fail(f"{SERVER_BIN} does not exist - build it first "
                        f"(cmake -B build -DNLLB_BUILD_SERVER_MOCK=ON && cmake --build build)")
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.cmd = [SERVER_BIN, "-m", MODEL, "--port", str(self.port), *EXTRA, *args]
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                pytest.fail(f"server exited with {self.proc.returncode} before it was ready:\n"
                            f"  {' '.join(self.cmd)}\n{out}")
            try:
                if requests.get(self.base + "/health", timeout=1).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.05)
        self.stop()
        pytest.fail(f"server not ready after {ready_timeout:.0f} s: {' '.join(self.cmd)}")

    def url(self, path: str) -> str:
        return self.base + path

    def get(self, path: str, **kw) -> requests.Response:
        return requests.get(self.url(path), timeout=kw.pop("timeout", 30), **kw)

    def post(self, path: str, **kw) -> requests.Response:
        return requests.post(self.url(path), timeout=kw.pop("timeout", 60), **kw)

    def translate(self, text, src="eng_Latn", tgt="deu_Latn", **kw) -> requests.Response:
        return self.post("/translate", json={"text": text, "src": src, "tgt": tgt}, **kw)

    def health(self) -> dict:
        return self.get("/health").json()

    def stop(self, timeout: float = 10.0) -> int | None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        return self.proc.returncode


@pytest.fixture(scope="module")
def server():
    """A server with default settings, shared by the tests of one module.
    Tests that read /health counters must use make_server instead."""
    s = Server()
    yield s
    s.stop()


@pytest.fixture
def make_server():
    """Factory for a fresh server with extra flags: make_server("--max-lines", "4")."""
    started: list[Server] = []

    def _make(*args: str, **kw) -> Server:
        s = Server(*args, **kw)
        started.append(s)
        return s

    yield _make
    for s in started:
        s.stop()


@pytest.fixture(scope="session")
def is_mock() -> bool:
    return MODEL == "mock" or Path(SERVER_BIN).name.startswith("nllb-server-mock")
