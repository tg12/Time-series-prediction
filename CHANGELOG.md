# Release notes

## Unreleased

### Added
- vendor the Air Passengers sample dataset in `tfts/data/air_passengers.csv` so examples and tests do not depend on a live network fetch
- add headless-friendly example flags in `examples/run_prediction_simple.py` with `--plot-path` and `--no-show`

### Changed
- validate Python 3.14 against `tf-nightly` and document the install path in the README and contribution guide
- standardize the local quality gate on `ruff`, `black`, `pytest`, and `pylint -E tfts`
- ignore generated example artifacts and ad-hoc Keras weight files in `.gitignore`

### Fixed
- resolve Python 3.14 and Keras compatibility issues in the trainer, data pipeline, and model wrappers
- remove the main AutoGraph and `gast` warning sources in the Autoformer, MoE, and RWKV layers
- reduce test noise by fixing repo-owned warnings and isolating two upstream-only warnings in pytest config

## v.0.0.15
- fix classification/anomaly detection
- fix from_pretrained


## v0.0.13
- support training style of transformers
- solve pandas version conflict
- support cnn/rnn model for keras3


## v0.0.4 Add models support (21/11/2022)

### Added
- input support
    - tf.Data or array
    - single item or three items

- function support
    - serving

- model support
    - tft
    - nbeats
    - unet
    - informer
    - deepar

## v0.0.3 Add models support (21/10/2022)

### Added
- function support
    - serving

- model support
    - tft
    - nbeats
    - unet

## v0.0.2 Add function support (1/10/2022)

### Added
- function support
    - classification

## v0.0.1 Initial release (15/03/2022)

### Added
- model support
    - rnn
    - tcn
    - bert
    - seq2seq
    - wavenet
    - transformer
- example support
    - sine data
    - air passenger data
- train
    - trainer
    - keras_trainer

### Contributor
- LongxingTan
