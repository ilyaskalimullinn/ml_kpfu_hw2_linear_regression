from typing import List

import numpy as np
import pandas as pd
from plotly import graph_objects as go
import plotly.express as px

from models.linear_regression_model import ExperimentResult


class Visualisation:

    @staticmethod
    def visualise_predicted_trace(prediction: np.ndarray, inputs: np.ndarray, targets: np.ndarray, plot_title=''):
        # visualise predicted trace and targets
        # Last homework, if you haven't done it, you need to do it anyway
        """

        :param prediction: model prediction based on inputs (oy for one trace)
        :param inputs: inputs variables (ox for both)
        :param targets: target variables (oy for one trace)
        :param plot_title: plot title
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=inputs,
            y=targets,
            mode='markers',
            name='Фактические наблюдения'
        ))

        fig.add_trace(go.Scatter(
            x=inputs,
            y=prediction,
            mode='markers',
            name="Предсказания модели",
        ))

        fig.update_layout(
            title=plot_title,
            xaxis_title="X",
            yaxis_title="Y",
            legend_title="Legend",
        )

        fig.show()

    @staticmethod
    def visualise_best_models(experiments: List[ExperimentResult]):

        df = pd.DataFrame({
            "degree": [exp.max_degree for exp in experiments],
            "valid_error": [round(exp.error_valid, 2) for exp in experiments],
            "test_error": [round(exp.error_test, 2) for exp in experiments],
            "reg_coeff": [round(exp.reg_coeff, 2) for exp in experiments]
        })

        df['degree_and_reg_coeff'] = "degree: " + df['degree'].astype(str) + ", reg coeff: " + df['reg_coeff'].astype(str)
        df['valid_error_str'] = df['valid_error'].astype(str)

        fig = px.scatter(data_frame=df,
                         x='degree_and_reg_coeff',
                         y='valid_error_str',
                         hover_data={'degree': True,
                                     'reg_coeff': True,
                                     'valid_error': True,
                                     'valid_error_str': False,
                                     'degree_and_reg_coeff': False,
                                     'test_error': True})

        fig.update_layout(
            title='Лучшие модели',
            xaxis_title="Максимальная степень полинома и коэффициент регуляризации",
            yaxis_title="Ошибка на валидационной выборке",
            legend_title="Legend",
            font_size=14
        )

        fig.show()
