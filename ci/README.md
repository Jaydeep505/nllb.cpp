# CI

nllb.cpp has two layers of CI, the same split llama.cpp and whisper.cpp use.

## 1. GitHub Actions (hosted runners, automatic)

Each workflow only starts when files it cares about change. When it starts, it
runs its whole suite - the path filter decides *whether* a workflow runs, not
*which tests* inside it run.

| workflow | triggers on | what it does | needs a model? |
|---|---|---|---|
| `ci.yml` | `src/`, `include/`, `ggml`, `tests/`, `examples/`, `scripts/`, `models/*.py`, CMake | build + every C++ gate on random fixtures, corruption matrices, Metal/CUDA build checks | no |
| `python-lint.yml` | any `*.py`, `.flake8` | flake8 + byte-compile | no |
| `server.yml` | `examples/server/`, `tests/server/` | pytest against `nllb-server-mock` on Linux and macOS | no |
| `ios.yml` | push: `src/`, `include/`, `ggml`, Swift example; **every** PR | xcframework + SwiftUI example | no |
| `download.yml` | download scripts + weekly | download scripts against a fake Hub, then the real Hub | no |
| `real-model.yml` | weekly + manual | full HF-parity run on the real 600M, incl. the server-vs-CLI gate | yes (downloads) |

A README or docs-only change triggers none of them, except `ios.yml` on a PR
(its `pull_request` trigger has no path filter).

## 2. `ci/run.sh` (your machine or a self-hosted runner, manual)

The full real-model suite in one command - everything `real-model.yml` does,
plus f16, batching, the server and an optional GPU backend. Run it before
pushing anything that touches the numerics or a backend:

```bash
bash ci/run.sh ./tmp/results ./tmp/mnt                       # CPU
NLLB_BUILD_METAL=1  bash ci/run.sh ./tmp/results ./tmp/mnt   # Apple silicon
NLLB_BUILD_CUDA=1   bash ci/run.sh ./tmp/results ./tmp/mnt   # NVIDIA
NLLB_LOW_PERF=1     bash ci/run.sh ./tmp/results ./tmp/mnt   # skip the slow gates
NLLB_CI_ONLY="build venv cache batch" bash ci/run.sh ./tmp/results ./tmp/mnt
```

- `./tmp/mnt` is a cache (venv, HF snapshot, GGUFs, goldens) and is reused, so
  only the first run downloads ~2.5 GB.
- `./tmp/results` gets one log per section and `summary.md`.
- The exit code is non-zero if any section failed; every section still runs,
  so one red gate does not hide the others.
- Pin the model with `NLLB_MODEL_REV=<commit sha>` so an upstream re-upload
  cannot change your reference.

Sections, in order: `build venv lint download convert py-gates goldens
tokenizer encoder decoder generate cache f16 batch gpu-<backend> cli server bench`.

## Adding a self-hosted runner

GitHub's hosted runners have no NVIDIA GPU, so CUDA/Vulkan numerics can only be
tested on your own hardware. To make that automatic:

1. Repo Settings -> Actions -> Runners -> New self-hosted runner, and follow
   the steps on the machine. Give it a label such as `cuda`.
2. Add a workflow that runs on it:

   ```yaml
   jobs:
     cuda:
       runs-on: [self-hosted, cuda]
       steps:
         - uses: actions/checkout@v4
           with: { submodules: recursive }
         - run: NLLB_BUILD_CUDA=1 bash ci/run.sh ~/ci/results ~/ci/mnt
   ```

3. Only trigger it on `push` to your own branches or `workflow_dispatch`: a
   self-hosted runner executes the code of whoever opened the PR, so do not
   run it on pull requests from forks.
