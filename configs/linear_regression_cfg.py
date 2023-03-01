from easydict import EasyDict
import numpy as np
cfg = EasyDict()
cfg.dataframe_path = ''

cfg.base_functions = [] # TODO list of basis functions
cfg.regularization_coeff = 0
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1