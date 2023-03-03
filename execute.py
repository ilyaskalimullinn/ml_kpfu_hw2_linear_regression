from typing import List
from random import Random

import numpy as np

from datasets.linear_regression_dataset import LinRegDataset
from models.linear_regression_model import LinearRegression
from utils.metrics import MSE
from utils.visualisation import Visualisation
from configs.linear_regression_cfg import cfg as lin_reg_cfg

linreg_dataset = LinRegDataset(lin_reg_cfg)()

inputs_test = linreg_dataset['inputs']['test']
inputs_train = linreg_dataset['inputs']['train']
inputs_valid = linreg_dataset['inputs']['valid']

targets_test = linreg_dataset['targets']['test']
targets_train = linreg_dataset['targets']['train']
targets_valid = linreg_dataset['targets']['valid']


def train_model(max_power: int, regularization_coeff: float) -> LinearRegression:
    base_functions = [lambda x, power=i: x ** i for i in range(1, max_power + 1)]
    lin_reg_model = LinearRegression(base_functions, regularization_coeff)
    lin_reg_model.train_model(inputs_train, targets_train)
    return lin_reg_model


def validate_model(lin_reg_model: LinearRegression):
    predictions = lin_reg_model(inputs_valid)
    error = MSE(predictions, targets_valid)
    return error


if __name__ == '__main__':
    random = Random()

    best_experiments = []

    for i in range(300):
        max_power = random.randint(5, 200)
        reg_coeff = 5 * random.random()
        lin_reg_model = train_model(max_power, reg_coeff)
        error = validate_model(lin_reg_model)
        best_experiments.append((max_power, reg_coeff, error))

    best_experiments.sort(key=lambda exp: exp[2])

    visualization = Visualisation()

    visualization.visualise_best_models(best_experiments[:10])

    for exp in best_experiments[:10]:
        print(f"Max polynomial power: {exp[0]}, regularization coefficient: {round(exp[1], 2)}, MSE: {round(exp[2], 2)}")
