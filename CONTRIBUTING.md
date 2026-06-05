# Contributing

This repository is a portfolio/research project for MURA musculoskeletal X-ray abnormality detection. Contributions should improve reproducibility, documentation, or the training/evaluation workflow without adding private data or model artifacts.

## Good Contributions

- Setup fixes for clean environments.
- Documentation clarifying MURA dataset paths or required data access.
- Small training/evaluation bug fixes.
- Reproducible metric or experiment notes.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m compileall data_loader.py evaluate.py explore_data.py model.py train.py
```

The MURA dataset requires a data use agreement. Do not commit dataset files, trained checkpoints, or patient data.

## Pull Requests

Keep changes narrow and include the command you ran. If you change metrics or model behavior, document the split, seed, and dataset path assumptions.
