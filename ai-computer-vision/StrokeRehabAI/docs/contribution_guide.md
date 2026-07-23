# Contribution Guide

## Workflow

1. Create a feature branch: `git checkout -b feature/<short-description>`.
2. Make focused commits — one logical change per commit.
3. Run the full check before opening a PR / submitting:
   ```powershell
   pytest
   python main.py check-gpu
   ```
4. Update the relevant doc in `docs/` if you changed a public interface or
   config schema.
5. Remove or update any `TODO` comment your change resolves.

## Code style

- Follow existing patterns in the module you're editing before introducing
  new ones project-wide.
- Prefer explicit config access (`cfg.training.batch_size`) over passing
  raw dicts between functions.
- Keep functions under ~40 lines where reasonable; extract helpers otherwise.
- Docstrings: one-line summary + `Args`/`Returns` for anything non-trivial.

## Commit messages

Short imperative summary line (e.g. "Add ROM scoring for shoulder flexion"),
optionally followed by a blank line and more detail.

## Reporting issues

Include: Python version, OS, GPU (or CPU-only), the exact command run, full
traceback, and — for camera/pose issues — whether `python main.py check-gpu`
and a plain `python -c "import cv2; print(cv2.__version__)"` succeed.

## Academic / research use

If extending this project for a paper or thesis, please keep
`docs/architecture.md` in sync with any structural changes — it's the
canonical source for a "System Architecture" section.
