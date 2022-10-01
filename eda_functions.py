import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def moving_average(series: pd.Series, n: int):
    """
        Calculate average of last n observations
    """
    return np.average(series[-n:])


def standart(serie: pd.Series):
    """
    Método de normalização min_max dos dados.

    :param serie: série a ser normalizada
    :return: série normalizada
    """
    min_serie = serie.min()
    max_serie = serie.max()
    return serie.apply(lambda x: (x - min_serie) / (max_serie - min_serie))


def mean_absolute_percentage_error(y_real, y_prev):
    """
    Calcula o erro absoluto médio em porcentagem

    :param y_real: valor observado
    :param y_prev: valor previsto
    :return: MAPE calculado
    """
    return np.mean(np.abs((np.subtract(y_real, y_prev) / y_real))) * 100


def plotMovingAverage(series: pd.Series, window: int, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies

    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title("Média móvel\n tamanho da janela = {}".format(window))
    plt.plot(rolling_mean, "g", label="Tendência da média móvel")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Limite superior / Limite inferior")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Valores observados")
    plt.legend(loc="upper left")
    plt.grid(True)
