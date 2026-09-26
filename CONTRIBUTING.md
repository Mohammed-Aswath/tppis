# Contributing

Thank you for considering a contribution. This package implements Tanaka and Matsui (2023)
as closely as the published equations allow. Read `docs/theory.md` before changing a
numerical routine.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Checks

```bash
ruff check tppis tests
ruff format --check tppis tests
mypy --strict
pytest
```

## Rules

- One concept per module. Each numbered equation from the paper is implemented once and named in the docstring.
- Tests that compare against `tests/reference.py` or the identity `TPPIS(alpha=1) == PPIS` must stay exact.
- Do not add a new screening family in a bug-fix or 0.x patch.
