# Contributing to Vinted Hunter

Thanks for your interest in improving Vinted Hunter. This guide covers how to set up a development
environment, run the checks, and send a change.

## Development setup

Requirements: Python 3.12 or newer.

```bash
make install        # creates .venv and installs the package with dev extras
cp .env.example .env
```

Without `make`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Edit `.env` if you want the API backend. The default `claude-cli` backend needs the Claude Code CLI
installed and logged in.

## Running the checks

```bash
make test           # pytest -q
make lint           # ruff check .
```

The test suite is offline. It uses fake Vinted, LLM, and CLI doubles from `tests/`.

Tests marked `live` hit the real vinted.fr and are skipped by default. Run them only when you need to
verify the scraper against the live site:

```bash
.venv/bin/python -m pytest -m live
```

## Code style

- Lint with `ruff check .`. The rule set is in `pyproject.toml`.
- Keep changes focused. Do not reformat unrelated code.
- Add or update tests for behaviour you change.
- Do not commit `.env`, the `data/` directory, or real Vinted or Claude responses that contain personal
  data. Fixtures must be sanitised (no seller ids, addresses, or account names).

## Submitting a change

1. Open an issue first for large changes, so the approach can be agreed on.
2. Create a branch from `main`.
3. Make your change, with tests, and make sure `make test` and `make lint` pass.
4. Open a pull request. Describe what changed and why, and how you tested it.
5. Keep commits readable. A short summary line and a body explaining the reason is enough.

## Vinted and Claude usage

This project reads public pages from vinted.fr and calls Claude. Changes that touch the scraper must keep
request rates polite (the default delay between requests exists for this reason). Do not add features
that bypass Vinted's access controls, and check Vinted's terms of use before changing how the site is
accessed. Avoid pushing usage costs up silently. Mention any new model calls in the pull request.

## Code of conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
