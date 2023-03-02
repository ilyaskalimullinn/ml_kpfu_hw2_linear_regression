from easydict import EasyDict

cfg = EasyDict()
cfg.dataframe_path = 'linear_regression_dataset.csv'

cfg.base_functions = [lambda x, power=i: x ** power for i in range(1, 50 + 1)]
cfg.regularization_coeff = 0
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
