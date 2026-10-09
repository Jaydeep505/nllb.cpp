"""HTTP contract of nllb-server. Holds for the mock and for a real model, so
these also run in the real-model job (NLLB_SERVER_BIN=build/bin/nllb-server)."""
from __future__ import annotations

import subprocess

import pytest
import requests

from conftest import MODEL, SERVER_BIN


# ---------------------------------------------------------------- endpoints

def test_health(server):
    r = server.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    for key in ("backend", "queue", "batches", "lines", "tokens", "avg_batch_size", "busy_ms"):
        assert key in j, f"/health lost '{key}'"


def test_languages(server):
    r = server.get("/languages")
    assert r.status_code == 200
    langs = r.json()["languages"]
    assert "eng_Latn" in langs and "deu_Latn" in langs
    assert len(langs) == len(set(langs)), "duplicate language codes"


def test_index_page(server):
    r = server.get("/")
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("text/html")
    assert "nllb-server" in r.text


def test_unknown_path_is_json_404(server):
    r = server.get("/no-such-endpoint")
    assert r.status_code == 404
    assert r.json() == {"error": "not found"}


def test_cors_preflight(server):
    r = requests.options(server.url("/translate"), timeout=10)
    assert r.status_code == 204
    assert r.headers.get("Access-Control-Allow-Origin") == "*"


# ---------------------------------------------------------------- translate

def test_translate_string(server):
    r = server.translate("Hello, world!")
    assert r.status_code == 200, r.text
    j = r.json()
    assert isinstance(j["translation"], str) and j["translation"]
    assert j["src"] == "eng_Latn" and j["tgt"] == "deu_Latn"
    assert j["tokens"] > 0
    assert j["time_ms"] >= 0


def test_translate_array_keeps_length_and_order(server):
    lines = ["One.", "Two.", "Three."]
    j = server.translate(lines).json()
    assert isinstance(j["translation"], list)
    assert len(j["translation"]) == len(lines)


def test_multiline_string_keeps_line_count(server):
    j = server.translate("First line.\nSecond line.\nThird line.").json()
    assert j["translation"].count("\n") == 2


def test_blank_lines_stay_blank(server):
    j = server.translate("First.\n\n   \nLast.").json()
    out = j["translation"].split("\n")
    assert len(out) == 4
    assert out[1] == "" and out[2] == "", "blank lines must not be sent to the model"
    assert out[0] and out[3]


def test_crlf_input(server):
    j = server.translate("First.\r\nSecond.").json()
    assert "\r" not in j["translation"]
    assert j["translation"].count("\n") == 1


def test_only_blank_text(server):
    r = server.translate("")
    assert r.status_code == 200
    assert r.json()["translation"] == ""
    assert r.json()["tokens"] == 0


# ---------------------------------------------------------------- bad input: 400

@pytest.mark.parametrize("body, needle", [
    ("not json",                                                  "not valid JSON"),
    ("[1, 2]",                                                    "JSON object"),
    ('{"text": "x", "tgt": "deu_Latn"}',                          "'src' is required"),
    ('{"text": "x", "src": "eng_Latn"}',                          "'tgt' is required"),
    ('{"text": "x", "src": 5, "tgt": "deu_Latn"}',                "'src' is required"),
    ('{"text": "x", "src": "english", "tgt": "deu_Latn"}',        "unknown language 'english'"),
    ('{"text": "x", "src": "eng_Latn", "tgt": "xxx_Latn"}',       "unknown language 'xxx_Latn'"),
    ('{"src": "eng_Latn", "tgt": "deu_Latn"}',                    "'text' is required"),
    ('{"text": 42, "src": "eng_Latn", "tgt": "deu_Latn"}',        "string or an array"),
    ('{"text": ["a", 1], "src": "eng_Latn", "tgt": "deu_Latn"}',  "only strings"),
    ('{"text": ["a\\nb"], "src": "eng_Latn", "tgt": "deu_Latn"}', "must not contain newlines"),
])
def test_bad_request(server, body, needle):
    r = server.post("/translate", data=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 400, r.text
    assert needle in r.json()["error"]


def test_unknown_language_points_to_languages_endpoint(server):
    r = server.translate("x", src="zzz_Zzzz")
    assert "GET /languages" in r.json()["error"]


# ---------------------------------------------------------------- limits: 413

def test_too_many_lines(make_server):
    s = make_server("--max-lines", "4")
    assert s.translate(["a"] * 4).status_code == 200
    r = s.translate(["a"] * 5)
    assert r.status_code == 413
    assert "too many lines" in r.json()["error"]


def test_line_too_long(make_server):
    s = make_server("--max-chars", "10")
    assert s.translate("a" * 10).status_code == 200
    assert s.translate("a" * 11).status_code == 413


def test_line_length_counts_characters_not_bytes(make_server):
    s = make_server("--max-chars", "10")
    r = s.translate("ü" * 10)            # 10 characters, 20 bytes
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------- process

def test_missing_model_flag_exits_nonzero():
    p = subprocess.run([SERVER_BIN], capture_output=True, text=True, timeout=30)
    assert p.returncode != 0
    assert "-m" in p.stderr


def test_port_in_use_exits_nonzero(server):
    p = subprocess.run([SERVER_BIN, "-m", MODEL, "--port", str(server.port)],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode != 0
    assert "could not bind" in p.stderr


def test_sigterm_shuts_down_cleanly(make_server):
    s = make_server()
    assert s.translate("Hello.").status_code == 200
    assert s.stop(timeout=15) == 0
