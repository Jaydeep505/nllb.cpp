# models/

GGUF files go here. They are git-ignored; only the download scripts and
this README are tracked.

```bash
./models/download-nllb-model.sh            # nllb-600M-f16.gguf, the default
./models/download-nllb-model.sh q8emb      # smaller and faster on CPU
./models/download-nllb-model.sh --list
```

Windows: `.\models\download-nllb-model.ps1` with the same arguments.

See `../MODELS.md` for which file to choose, how to pin a version, the
CC-BY-NC 4.0 license on the weights, and how to convert a checkpoint
yourself.
