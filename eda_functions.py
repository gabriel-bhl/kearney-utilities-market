import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error


def corr_func(x: pd.Series, y: pd.Series):
    """
    Calcula a correlação de Pearson, Kendall e Spearman entre x e y.

    :param x: Primeira série de valores numéricos de mesmo tamanho.
    :param y: Segunda série de valores numéricos de mesmo tamanho.
    :return: dicionário com os valores de cada correlação.
    """

    print('Correlação básica:\n')

    corr_dict = {}
    for method in ['kendall', 'spearman', 'pearson']:
        corr = x.corr(y, method=method)
        corr_dict[method] = [corr]
        print('{}: {:.4f}'.format(method.title(), corr))

    return corr_dict


def corr_movel(x: pd.Series, y: pd.Series, z: pd.Series, method_list=None, window=30):
    """
    Calculo e visualização das correlações desejadas  dentro da janela window.

    :param x: Primeira série de valores numéricos de mesmo tamanho.
    :param y: Segunda série de valores numéricos de mesmo tamanho.
    :param z: Terceira série de valores datetime de mesmo tamanho.
    :param method_list: Lista de métodos a ser aplicada, os pontos de min e max da visualização.
    se referem ao primeiro método da lista. Default=['pearson', 'kendall', 'spearman'].
    :param window: Janela a qual será calculada a correlação. Default=30.
    :return: dataframe com o valor das correlações.
    """

    print('Correlação em janelas:\n')

    # Lista de métodos
    if method_list is None:
        method_list = ['kendall', 'spearman', 'pearson']

    # Criando dataframe das correlações
    df = pd.DataFrame()
    for method in method_list:
        df[method] = x.rolling(window=window).corr(y, method=method)

    # Retirando valores nulos
    df = df.loc[df[method_list[0]].notna()].reset_index(drop=True)
    df.index = z.head(z.shape[0] - window + 1)
    df.reset_index(inplace=True)

    # Visualização
    plt.figure(figsize=(13, 5))

    for method in method_list:
        sns.lineplot(x='index', y=method, data=df, label=method)

    ## Annotate valor mín e max de Pearson
    ### Minimo
    min_value_geral = df[method_list[0]].min()
    min_geral = df.loc[df[method_list[0]] == min_value_geral, 'index'].tolist()[0]
    plt.annotate(' {:.4f}'.format(min_value_geral), xy=(min_geral, min_value_geral),
                 ha='left', va='top')

    ### Maximo
    max_value_geral = df[method_list[0]].max()
    max_geral = df.loc[df[method_list[0]] == max_value_geral, 'index'].tolist()[0]
    plt.annotate(' {:.4f}'.format(max_value_geral), xy=(max_geral, max_value_geral),
                 ha='left', va='bottom')

    plt.xlabel('Janela de {} Periodos'.format(window))
    plt.ylabel('Correlações')
    plt.ylim(-1.1, 1.1)
    plt.show()

    return df[method_list]


def corr_deslocada(x: pd.Series, y: pd.Series, method='kendall', desloc=30, limit=True):
    """
    Calculo e visualização da correlação de séries deslocadas

    :param x: Primeira série de valores numéricos de mesmo tamanho.
    :param y: Série a ser deslocada.
    :param method: Qual tipo de correlação deve ser calculada.
    :param desloc: Quantos Periodos deve se deslocar para ambos os sentido.
    :param limit: Booleano se deve limitar ou não o eixo y do gráfico de correlação.
    :return: Dataframe contendo o número de Periodos deslocados e a correlação
    """

    print('Correlação deslocada:\n')

    # Range de deslocamento
    range_shift = range(-desloc, 0)

    # Calculo da correlação deslocada
    correlacao = pd.Series(map(lambda i: x.corr(y.shift(-i), method=method), range_shift))

    # Dataframe com deslocamente e correlação
    df = pd.DataFrame({'deslocamento': range_shift, method: correlacao})

    # Visualização
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))

    # Dados originais normalizados
    sns.lineplot(x=x.index, y=(x - x.mean()) / x.std(), label='{} normalizado'.format(x.name), ax=ax[0])
    sns.lineplot(x=y.index, y=(y - y.mean()) / y.std(), label='{} normalizado'.format(y.name), ax=ax[0])

    ax[0].set_title('Dados normalizados para comparação')
    ax[0].set_xlabel('Tempo')
    ax[0].set_ylabel('Valores para comparação')

    # Correlação
    sns.lineplot(x='deslocamento', y=method, data=df, ax=ax[1])

    # Demarcação da correlação máxima atingida
    if df[method].max() > abs(df[method].min()):
        maximo = df[method].max()
    else:
        maximo = df[method].min()

    x_maximo = df.loc[df[method] == maximo, 'deslocamento'].tolist()[0]
    plt.vlines(x_maximo, ymin=-0.9, ymax=0.9, color='red', linestyles='dashed')
    plt.annotate('{} Periodos\n{:.4f}'.format(x_maximo, maximo), xy=(x_maximo + desloc / 100, 0.9), ha='left', va='top')

    # Set plot
    ax[1].set_title(f'Correlação dos dados com deslocamento da(o) "{y.name}" \n'
                    f'em {desloc} periodos para trás e para frente')
    ax[1].set_xlabel('Periodos deslocados')
    ax[1].set_ylabel('Correlação')
    ax[1].set_xlim(-desloc - 1, 0)
    if limit:
        ax[1].set_ylim(-1, 1)

    plt.tight_layout()
    plt.show()

    return df


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
