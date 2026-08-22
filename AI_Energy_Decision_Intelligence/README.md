# AI-Powered Data Analytics and Decision Intelligence for Energy-Efficient Industrial Machines

A reproducible, one-command proof-of-concept experiment for predicting industrial-machine energy consumption and converting predictions into feasible operating recommendations.

## Pipeline
Synthetic industrial sensor data -> XGBoost / Random Forest / PyTorch MLP -> model comparison -> XGBoost SHAP contributions -> constrained decision search -> energy/cost-saving recommendations.

## Run
```bash
python -m pip install -r requirements.txt
bash run.sh
```

Outputs are written to `outputs/`.

> Research caution: the built-in dataset is synthetic and is intended to validate the experimental pipeline. A publication should repeat the experiment on a real industrial dataset and report external validity limitations.
