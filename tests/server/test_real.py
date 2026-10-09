"""nllb-server on a real model must say exactly what nllb-cli says.

    NLLB_SERVER_BIN=build/bin/nllb-server NLLB_SERVER_MODEL=models/nllb-600M-f16.gguf \\
        pytest tests/server -m "not mock"

A single line goes through nllb_generate, nllb-cli's own path, so the text must
be byte-identical. A batch goes through nllb_generate_batch; on the CPU its ids
are exact against the single path (nllb-test-batch [pad]), so it must match too.
Both sides are pinned to --backend cpu so a GPU kernel choice cannot differ.
"""
from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import CLI_BIN, MODEL

pytestmark = pytest.mark.real

SENTENCES = [
    "Hello, world!",
    "The weather is nice today.",
    "Machine translation runs offline on this laptop.",
    "Where is the train station?",
    "I would like a cup of coffee, please.",
    "This sentence is a little longer than the others, so the batch has to be padded.",
    "Thank you.",
    "Numbers like 42 and 3.14 should survive.",
]


@pytest.fixture(scope="module", autouse=True)
def _require_real_model():
    if MODEL == "mock" or not os.path.exists(MODEL):
        pytest.fail("set NLLB_SERVER_MODEL to a converted .gguf (and NLLB_SERVER_BIN to nllb-server)")
    if not os.path.exists(CLI_BIN):
        pytest.fail(f"{CLI_BIN} not found - build nllb-cli first")


_cli_cache: dict[tuple[str, str, str], str] = {}


def cli(text: str, src: str = "eng_Latn", tgt: str = "deu_Latn") -> str:
    key = (text, src, tgt)
    if key not in _cli_cache:
        p = subprocess.run([CLI_BIN, "-m", MODEL, "--backend", "cpu", "--src", src, "--tgt", tgt, text],
                           capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr
        _cli_cache[key] = p.stdout.rstrip("\n")
    return _cli_cache[key]


@pytest.fixture(scope="module")
def real(request):
    from conftest import Server
    s = Server("--backend", "cpu", "--batch", "8", "--batch-wait-ms", "50", ready_timeout=300)
    yield s
    s.stop()


def test_all_202_languages(real):
    assert len(real.get("/languages").json()["languages"]) == 202


def test_single_line_matches_cli(real):
    got = real.translate(SENTENCES[0]).json()["translation"]
    assert got == cli(SENTENCES[0])


@pytest.mark.parametrize("tgt", ["fra_Latn", "hin_Deva", "jpn_Jpan"])
def test_other_targets_match_cli(real, tgt):
    assert real.translate(SENTENCES[1], tgt=tgt).json()["translation"] == cli(SENTENCES[1], tgt=tgt)


def test_batched_array_matches_cli_line_by_line(real):
    got = real.translate(SENTENCES, timeout=300).json()["translation"]
    for s, g in zip(SENTENCES, got):
        assert g == cli(s), f"batched output differs from nllb-cli for {s!r}"


def test_concurrent_mixed_pairs_match_cli(real):
    jobs = [(s, "eng_Latn", t) for s, t in zip(SENTENCES[:6], ["deu_Latn", "fra_Latn", "spa_Latn"] * 2)]
    with ThreadPoolExecutor(len(jobs)) as ex:
        rs = list(ex.map(lambda j: real.translate(j[0], src=j[1], tgt=j[2], timeout=300), jobs))
    for (s, src, tgt), r in zip(jobs, rs):
        assert r.status_code == 200, r.text
        assert r.json()["translation"] == cli(s, src, tgt), f"{src}->{tgt} {s!r}"
