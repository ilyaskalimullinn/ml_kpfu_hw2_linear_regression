import os.path

import numpy as np

from datasets.linear_regression_dataset import LinRegDataset
from models.linear_regression_model import LinearRegression, ExperimentResult
from utils.metrics import MSE
from utils.visualisation import Visualisation
from configs.linear_regression_cfg import cfg as lin_reg_cfg

linreg_dataset = LinRegDataset(lin_reg_cfg)()


def train_model(max_power: int, regularization_coeff: float) -> LinearRegression:
    base_functions = [lambda x, power=i: x ** power for i in range(1, max_power + 1)]
    lin_reg_model = LinearRegression(base_functions, regularization_coeff)
    lin_reg_model.train_model(linreg_dataset['inputs']['train'], linreg_dataset['targets']['train'])
    return lin_reg_model


def validate_model(lin_reg_model: LinearRegression, dataset_type: str) -> float:
    predictions = lin_reg_model(linreg_dataset['inputs'][dataset_type])
    error = MSE(predictions, linreg_dataset['targets'][dataset_type])
    return error


if __name__ == '__main__':
    best_experiments = []

    for i in range(300):
        max_power = np.random.randint(5, 200)
        reg_coeff = np.random.uniform(0, 5)
        lin_reg_model = train_model(max_power, reg_coeff)
        valid_error = validate_model(lin_reg_model, dataset_type='valid')
        best_experiments.append(ExperimentResult(max_degree=max_power, reg_coeff=reg_coeff, error_valid=valid_error))

    best_experiments.sort(key=lambda exp: exp.error_valid)
    best_experiments = best_experiments[:10]

    for exp in best_experiments:
        lin_reg_model = train_model(exp.max_degree, exp.reg_coeff)
        test_error = validate_model(lin_reg_model, dataset_type='test')
        exp.error_test = test_error
        print(exp)

    visualization = Visualisation()
    ROOT_DIR = os.path.abspath(os.curdir)
    visualization.visualise_best_models(best_experiments, save_path=os.path.join(ROOT_DIR, 'graphs/best_models.html'))

    model_with_reg = train_model(100, 0)
    model_without_reg = train_model(100, 1e-5)

    predictions_with_reg = model_with_reg(linreg_dataset['inputs']['test'])
    predictions_without_reg = model_without_reg(linreg_dataset['inputs']['test'])

    visualization.visualise_predicted_trace(predictions=[predictions_with_reg, predictions_without_reg],
                                            inputs=linreg_dataset['inputs']['test'],
                                            targets=linreg_dataset['targets']['test'],
                                            labels=['Модель с коэффициентом регуляризации = 0', 'Модель с коэффициентом регуляризации = 1e-5'],
                                            save_path=os.path.join(ROOT_DIR, 'graphs/models_predictions.html'),
                                            plot_title='Предсказания моделей с максимальной степенью полинома 100 и '
                                                       'разными коэффициентами регуляризации')
