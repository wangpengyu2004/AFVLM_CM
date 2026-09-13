# Local model checkpoints

This directory is intentionally excluded from Git except for this file.

Expected layout:

```text
pretrained/
├── llava-v1.5-7b/
└── clip-vit-large-patch14-336/
```

Create both directories with `python tools/download_models.py`. Training uses
these local paths with `local_files_only: true` and never downloads weights.
