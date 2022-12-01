from math import log2
from typing import Iterable

import pandas as pd


# Minimum Description Length (MDL) for **MPM**


def _get_partitions(table):
    tables = []

    for var_name in table.columns:
        if var_name == 'Ind':
            continue
        df = table[['Ind', var_name]]
        tables.append(df.dropna())

    return tables


def _get_model_complexity(table, partitions):
    summation = 0
    for partition in partitions:
        l_bb_i = len(partition.iloc[0]["Ind"])  # tamanho do bb len(A,B)=2
        summation += 2 ** l_bb_i - 1
    population_size = len(table)  # tamanho da população
    return log2(population_size + 1) * summation


def _get_entropy(partition):
    e = 0  # (A,B)
    for index, row in partition.iterrows():
        pk = row.iloc[1]  # frequencia da atual instancia no bb [00,01,10,11]
        if pk > 0:
            e -= pk * log2(pk)
    return e


def _get_compressed_population_complexity(table, partitions):
    summation = 0
    for partition in partitions:
        summation += _get_entropy(partition)  # adicionando a entropia do atual bb
    return len(table) * summation  # tamanho população * somatorio da entropia


def get_sorted_probability_tables(probability_tables: Iterable[pd.DataFrame]):
    comp_ids = []
    MPMs = []
    idx = 0

    for table in probability_tables:
        ptt = _get_partitions(table)  # [(A,B),(C),(D)]

        mc = _get_model_complexity(table, ptt)
        cpc = _get_compressed_population_complexity(table, ptt)
        complexity = mc + cpc

        MPMs.append(table)
        comp_ids.append([complexity, idx])
        idx += 1

    comp_ids.sort()
    sorted_MPMs = tuple((MPMs[id], cp) for cp, id in comp_ids)

    return sorted_MPMs
