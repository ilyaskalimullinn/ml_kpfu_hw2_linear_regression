import numpy as np
from typing import Tuple


class LinearRegression:

    def __init__(self, base_functions: list, reg_coeff: float):
        self.weights = np.random.randn(len(base_functions) + 1).reshape(-1, 1)
        self.base_functions = base_functions
        self.reg_coeff = reg_coeff

    def __pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        u, s, v_t = np.linalg.svd(matrix)

        sigma_inverse = self.__inverse_diagonal_matrix(diagonal=s, shape_to_be=(v_t.shape[0], u.shape[1]))

        return v_t.T @ sigma_inverse @ u.T

    def __inverse_diagonal_matrix(self, diagonal: np.ndarray, shape_to_be: Tuple):
        min_value = np.finfo(float).eps * np.max(diagonal) * max(shape_to_be)

        diagonal_inverse = np.where(diagonal > min_value, diagonal / (diagonal ** 2 + self.reg_coeff), 0)

        matrix = np.zeros(shape=shape_to_be)
        np.fill_diagonal(matrix, diagonal_inverse)
        return matrix

    def __plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        inputs = inputs.reshape(-1, 1)
        columns = [np.ones_like(inputs)]
        for func in self.base_functions:
            columns.append(func(inputs))
        return np.hstack(columns)

    def __calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        self.weights = pseudoinverse_plan_matrix @ targets.reshape(-1, 1)

    def calculate_model_prediction(self, plan_matrix) -> np.ndarray:
        return (plan_matrix @ self.weights).flatten()

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


class ExperimentResult:
    def __init__(self, max_degree: int, reg_coeff: float, error_valid: float = None, error_test: float = None):
        self.max_degree = max_degree
        self.reg_coeff = reg_coeff
        self.error_valid = error_valid
        self.error_test = error_test

    def __str__(self) -> str:
        s = f"Max degree: {self.max_degree}, reg coefficient: {self.reg_coeff}, validation error: {self.error_valid}"
        if self.error_test:
            s += f", test error: {self.error_test}"
        return s
