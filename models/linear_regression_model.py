import numpy as np


class LinearRegression():

    def __init__(self, base_functions: list, reg_coeff: float):
        self.weights = None  # init weights using np.random.randn (normal distribution with mean=0 and variance=1).
        # Last homework, if you haven't done it, you need to do it anyway
        self.base_functions = base_functions
        self.reg_coeff = reg_coeff


    def __pseudoinverse_matrix(self,matrix: np.ndarray) -> np.ndarray:
        """TODO calculate pseudoinverse matrix with regularization using SVD """
        pass

    def __plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        # build Plan matrix using list of lambda functions defined in config. Use only one loop (for base_functions).
        # Last homework, if you haven't done it, you need to do it anyway
        pass

    def __calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        """TODO calculate weights of the model using formula from the lecture"""
        pass

    def calculate_model_prediction(self, plan_matrix) -> np.ndarray:
        # calculate prediction of the model (y) using formula from the lecture.
        # Last homework, if you haven't done it, you need to do it anyway
        pass

    def train_model(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        # prepare data
        plan_matrix = self.__plan_matrix(inputs)
        pseudoinverse_plan_matrix = self.__pseudoinverse_matrix(plan_matrix)

        # train process
        self.__calculate_weights(pseudoinverse_plan_matrix, targets)


    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self.__plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions
