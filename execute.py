from datasets.linear_regression_dataset import LinRegDataset
from models.linear_regression_model import LinearRegression
from utils.metrics import MSE
from utils.visualisation import Visualisation


def experiment(lin_reg_cfg, visualise_prediction=True):
    lin_reg_model = LinearRegression(lin_reg_cfg.base_functions, lin_reg_cfg.regularization_coeff)
    linreg_dataset = LinRegDataset(lin_reg_cfg)()

    lin_reg_model.train_model(linreg_dataset['inputs']['train'], linreg_dataset['targets']['train'])

    predictions = lin_reg_model(linreg_dataset['inputs']['test'])
    error = MSE(predictions, linreg_dataset['targets']['test'])

    if visualise_prediction:
        Visualisation.visualise_predicted_trace(predictions,
                                                linreg_dataset['inputs']['test'],
                                                linreg_dataset['targets']['test'],
                                                plot_title=f'Полином степени {len(lin_reg_cfg.base_functions)}; MSE = {round(error, 2)}')


if __name__ == '__main__':
    from configs.linear_regression_cfg import cfg as lin_reg_cfg
    experiment(lin_reg_cfg, visualise_prediction=True)
