# TFTS Examples

Basic scripts:
- [Time Series Prediction](./run_prediction_simple.py): forecast future values from demo data.
- [Time Series Classification](./run_classification.py): classify time series samples.
- [Time Series Anomaly Detection](./run_anomaly.py): detect anomalous behavior in a sequence.
- [AutoML Parameter Tuning](./run_tuner.py): search model hyperparameters.

Run the forecasting example from the repository root after `python -m pip install -e .`:

```shell
MPLBACKEND=Agg python3 examples/run_prediction_simple.py \
  --use_model rnn \
  --use_data sine \
  --epochs 2 \
  --batch_size 8 \
  --plot-path ./artifacts/prediction.png \
  --no-show
```

Notebooks:
- [Single-step Weather Prediction](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/examples/notebooks/single_step_weather_prediction.ipynb)
- [Multi-step Sales Prediction](https://nbviewer.org/github/LongxingTan/Time-series-prediction/blob/master/examples/notebooks/multi_steps_sales_prediction.ipynb)

Benchmark:
- [Kaggle Forecasting Sticker Sales](https://www.kaggle.com/competitions/playground-series-s5e1)

More examples:
- [TFTS-Bert](https://github.com/LongxingTan/KDDCup2022-Baidu): KDD Cup 2022 wind power forecasting.
- [TFTS-Seq2seq](https://github.com/LongxingTan/Data-competitions/tree/master/tianchi-enso-prediction): Tianchi ENSO prediction 2021.
