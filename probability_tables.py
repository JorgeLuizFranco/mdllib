from typing import Sequence

import pandas as pd

from mdllib import _utils


# Probabilities Matrix


def _n_var_partition(df, variables, partition_type, individuals):
    mid_points = []
    for var in variables:
        if partition_type == 'median':
            mid_points.append(df[var].median())
        elif partition_type == 'min_max':
            mid_points.append((df[var].min() + df[var].max()) / 2)

    ans = {}

    for ind in individuals:
        if ind[0] == '0':
            filt = (df[variables[0]] < mid_points[0])
        else:
            filt = (df[variables[0]] >= mid_points[0])

        for i in range(1, len(variables)):
            if ind[i] == '0':
                filt = filt & (df[variables[i]] < mid_points[i])
            else:
                filt = filt & (df[variables[i]] >= mid_points[i])

        ans[ind] = len(df[filt])

    return ans


def _calc_probability(df, discrete_dict, variables, part):
    number_of_elements = len(df)  # possible modification

    vars_key = '('
    vars_key += ', '.join(map(str, variables))
    vars_key += ')'

    individuals = []
    for i in range(2 ** len(variables)):  # generating zeros and ones
        individuals.append(_utils.get_bin(i, len(variables)))

    numbers_of_occurrence = _n_var_partition(df, variables, part, individuals)

    discrete_dict[vars_key] = {}

    for ind in individuals:
        discrete_dict[vars_key][ind] = (numbers_of_occurrence[ind] / number_of_elements)


def _partition_k(collection, min_value, k):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in _partition_k(collection[1:], min_value - 1, k):
        if len(smaller) > k:
            continue
        # insert `first` in each of the subpartition's subsets
        if len(smaller) >= min_value:
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        if len(smaller) < k:
            yield [[first]] + smaller


def _subsets_k(collection, k):
    return _partition_k(collection, k, k)


def get_all_partitions(variables: Sequence[str]):
    all_partitions = []

    for i in range(1, len(variables)):
        for n, p in enumerate(_subsets_k(variables, i), 1):
            all_partitions.append(p)

    return all_partitions


def get_probability_tables(partitions, variables):
    probability_tables = []

    for p in partitions:
        discrete_dict = {}
        for group in p:
            _calc_probability(group, discrete_dict, variables, part='min_max')
        probability_tables.append(pd.DataFrame(discrete_dict))

    return probability_tables
