# Contributing to TFTS

If you are interested in contributing to tfts,

- Feel free to send a Pull Request
- If you want to implement a new feature and are unsure about it, open an issue first

Once you finish implementing a feature a bug-fix, please send a Pull Request to https://github.com/LongxingTan/Time-series-prediction


## Developing TFTS

To develop tfts on your machine, here are some tips:

1. Uninstall existing `tfts` installations.
2. Clone a copy of `tfts` from source.
3. Create a new branch and edit the code.
4. Install the development requirements with `python3 -m pip install -r requirements.txt`.
5. Install the package in editable mode with `python3 -m pip install -e .`.
6. Install the hooks with `pre-commit install`.
7. Run `ruff check .`, `black --check tfts examples tests`, `pylint -E tfts`, and `pytest -q`.
8. Ensure the full test suite passes and the code coverage roughly stays the same.
9. Update `README.md`, `examples/README.md`, and `CHANGELOG.md` when behavior, tooling, or support expectations change.
10. Update and test the documentation.

Python 3.14 support is currently validated against `tf-nightly`. Stable TensorFlow releases still govern the support boundary for older versions.
