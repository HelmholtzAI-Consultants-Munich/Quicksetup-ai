# How to do hyperparameter tuning

In this Tutorial, we show you how hyperparameter tuning is performed using the _hydra_optuna_sweeper_ plugin.

## Adjusting the run command

In order for the optuna config file to be read you will need to run your train script with the following configuration:

```python train.py -m hparams_search=search_optuna```

## Edit the config file for tuning the learning rate

In src/configs/hparams_search/mnist_optuna.yaml you may consider adding the *log* key to change the distribution from which your parameter is sampled, from uniform to log uniform. This is particularly useful if you wish to tune the learning rate:

```
search_space:
  model.lr:
    type: float
    log: true
    low: 0.0001
    high: 0.2

```
