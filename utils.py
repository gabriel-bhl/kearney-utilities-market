import os
from datetime import datetime, time, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
from IPython.display import display
from openpyxl import load_workbook


def overall_check(df: pd.DataFrame) -> None:
    """Printa um diagnóstico geral do DataFrame contendo as seguintes informações:
    1. Números de valores únicos por coluna.
    2. Tipo da coluna.
    3. Valores únicos (caso existam menos do que 100 valores).
    4. Top 10 valores (caso existam menos do que 100 valores).
    """
    for col in df.columns:
        print('####################')
        unicos = df[col].nunique()
        print(f'{col} ({df[col].dtype}): {unicos} valores únicos')
        if unicos < 100:
            print(df[col].unique())
        print()
        number = df[col].value_counts(dropna=False).head(10)
        pct = (df[col].value_counts(dropna=False).head(10) / len(df)).mul(100).round(1)
        print(number.astype(str) + ' (' + pct.astype(str) + '%)')
        print()


def find_key(df: pd.DataFrame) -> None:
    """Testa combinações das colunas do DataFrame até encontrar chave única.

    Execução será interrompida após chave ser encontrada (todos as combinações com n colunas
    serão rodadas, sendo n o número de colunas da chave encontrada)."""
    found_key = False

    for qtd_cols in range(1, len(df.columns) + 1):
        for col_combination in combinations(df.columns, qtd_cols):
            if df.duplicated(subset=col_combination).sum() == 0:
                print(col_combination)
                found_key = True

        if found_key:
            print(f'Chaves achadas com {qtd_cols} elementos!')
            break


def display_all(df: pd.DataFrame, dim: str = 'columns') -> None:
    """Exibe DataFrame completo de acordo com a dimensão passada no parâmetro 'dim'.
    Por padrão, exibirá todas as colunas."""
    with pd.option_context(f'display.max_{dim}', None):
        display(df)


def excel_date_parser(series: pd.Series):
    """Trata datas no Excel para formato datetime."""
    return pd.TimedeltaIndex(series, unit='d') + datetime(1899, 12, 30)


def number_to_time(value: float):
    if np.isnan(value):
        return np.nan
    time_num = value * 24
    h = int(time_num)
    time_num = (time_num - h) * 60
    m = int(time_num)
    time_num = (time_num - m) * 60
    s = int(time_num)
    return time(h, m, s)


def excel_time_parser(series: pd.Series):
    """Trata hora no Excel para formato datetime."""
    return series.apply(number_to_time)


def trata_hora(h: timedelta, total_seconds: bool = True) -> str:
    """Transforma um timedelta para uma string no formato 1d 10h 35m.
    Caso a entrada seja nula, também retornará um nulo.

    Por default, o parâmetro 'total_seconds' é verdadeiro, significando que
    a função irá transformar o timedelta para o valor inteiro dos segundos
    através do método total_seconds(). Caso seja passado como valor o número de
    segundos já convertido, o parâmetro 'total_seconds' deve ser setado como falso.
    """
    if pd.isna(h):
        return np.nan

    if total_seconds:
        total_seconds = h.total_seconds()
    else:
        total_seconds = h

    days = np.floor(total_seconds / (60 * 60 * 24))
    total_seconds = total_seconds - (days * 60 * 60 * 24)

    hours = np.floor(total_seconds / (60 * 60))
    total_seconds = total_seconds - (hours * 60 * 60)

    minutes = np.floor(total_seconds / 60)

    return f'{int(days)}d {int(hours)}h {int(minutes)}m'


def remove_collinear_features(x, threshold=0.9):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)

    return x


def trata_diff(h: int) -> str:
    """Transforma uma diferença entre timedelta em uma string no formato
    1d 10h 35m. O valor passado como parâmetro deve ser o número total de segundos
    da diferença.

    Exemplo de uso:

    segundos = timedelta1.total_seconds() - timedelta2.total_seconds()
    trata_diff(segundos)
    """
    if pd.isnull(h):
        return np.nan
    if h > 0:
        return trata_hora(h, total_seconds=False)
    else:
        h = h * -1
        return '-' + trata_hora(h, total_seconds=False)


def save_excel_tab(file_path: str, tab_name: str, df: pd.DataFrame, **kwargs) -> None:
    """Escreve o DataFrame em 'df' na aba 'tab_name' de um arquivo Excel já
    existente, sem alterar seu conteúdo original.
    """
    if os.path.exists(file_path):
        book = load_workbook(file_path)
        writer = pd.ExcelWriter(file_path, engine='openpyxl')
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        df.to_excel(writer, sheet_name=tab_name, **kwargs)
        writer.close()
    else:
        with pd.ExcelWriter(file_path) as writer:
            df.to_excel(writer, sheet_name=tab_name, **kwargs)


def move_col(df: pd.DataFrame, colname: str, index: int) -> pd.DataFrame:
    """Realoca a coluna passada em 'colname' para a posição 'index' do DataFrame
    'df'.

    Exemplo de uso:

    new_df = move_col(df, 'coluna7', 1)  # irá transformar a coluna7
    na segunda coluna do DataFrame.
    """
    cols = list(df.columns)
    cols.remove(colname)
    cols.insert(index, colname)
    return df.reindex(columns=cols)
