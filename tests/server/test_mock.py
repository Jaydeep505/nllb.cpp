"""Queueing and batching behaviour, against nllb-server-mock.

The mock answers "[src->tgt] text" after 40 ms + 2 ms per row, the cost shape
of a batched decode step. That makes ordering, batching, back-pressure and
timeouts deterministic enough to assert without a model.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

pytestmark = pytest.mark.mock


@pytest.fixture(autouse=True)
def _require_mock(is_mock):
    if not is_mock:
        pytest.fail("test_mock.py needs nllb-server-mock; run the real model with -m 'not mock'")


def test_echo_format(server):
    j = server.translate("Hi.", src="eng_Latn", tgt="fra_Latn").json()
    assert j["translation"] == "[eng_Latn->fra_Latn] Hi."


def test_lines_come_back_in_order(server):
    lines = [f"line {i}" for i in range(40)]
    j = server.translate(lines).json()
    assert j["translation"] == [f"[eng_Latn->deu_Latn] line {i}" for i in range(40)]


def test_blank_lines_do_not_reach_the_model(make_server):
    s = make_server()
    s.translate("a\n\n \nb")
    assert s.health()["lines"] == 2


def test_concurrent_requests_are_batched(make_server):
    s = make_server("--batch", "16", "--batch-wait-ms", "100")
    with ThreadPoolExecutor(16) as ex:
        rs = list(ex.map(lambda i: s.translate(f"req {i}"), range(16)))
    assert all(r.status_code == 200 for r in rs)
    for i, r in enumerate(rs):
        assert r.json()["translation"] == f"[eng_Latn->deu_Latn] req {i}", "a reply went to the wrong request"
    h = s.health()
    assert h["lines"] == 16
    assert h["avg_batch_size"] > 1.0, f"16 concurrent requests were never batched: {h}"


def test_mixed_language_pairs_in_one_batch(make_server):
    s = make_server("--batch", "8", "--batch-wait-ms", "100")
    pairs = [("eng_Latn", "deu_Latn"), ("deu_Latn", "eng_Latn"), ("eng_Latn", "hin_Deva"), ("fra_Latn", "jpn_Jpan")] * 2
    with ThreadPoolExecutor(len(pairs)) as ex:
        rs = list(ex.map(lambda p: s.translate("x", src=p[0], tgt=p[1]), pairs))
    for (src, tgt), r in zip(pairs, rs):
        assert r.json()["translation"] == f"[{src}->{tgt}] x"


def test_batch_1_disables_batching(make_server):
    s = make_server("--batch", "1")
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda i: s.translate(f"r{i}"), range(6)))
    assert s.health()["avg_batch_size"] == 1.0


def test_queue_full_is_503_and_all_or_nothing(make_server):
    s = make_server("--max-queue", "2")
    r = s.translate(["a", "b", "c"])          # 3 lines > queue of 2: refused whole
    assert r.status_code == 503
    assert r.headers.get("Retry-After") == "1"
    assert s.health()["lines"] == 0, "a refused request must not leave lines running"
    assert s.translate(["a", "b"]).status_code == 200


def test_timeout_is_504(make_server):
    # --batch 1: 40 lines at ~42 ms each is ~1.7 s, past a 1 s deadline.
    s = make_server("--batch", "1", "--request-timeout", "1")
    r = s.translate([f"l{i}" for i in range(40)], timeout=30)
    assert r.status_code == 504
    assert "timed out" in r.json()["error"]
    # The abandoned lines are skipped, so the server is usable again quickly.
    assert s.translate("after").status_code == 200
