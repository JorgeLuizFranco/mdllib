import pandas as pd

from mdllib import _utils


# Transforming in the classic way


def _get_variables_in_column(column):
    return tuple(column.replace("(", "").replace(")", "").split(", "))


def _get_dict_records(table):
    records = {_get_variables_in_column(col): {} for col in table.columns if col != "Ind"}
    for index, row in table.iterrows():
        ind = row["Ind"]
        for el, col in zip(row, table.columns):
            if col == "Ind":
                continue
            if el != ".":
                records[_get_variables_in_column(col)][ind] = el
    return records


def _get_variables_separated(table):
    old_columns = [col for col in table.columns if col != "Ind"]
    probability_tuples = [_get_variables_in_column(col) for col in old_columns]
    individual_columns = []
    for lista in probability_tuples:
        individual_columns.extend(lista)
    return individual_columns, probability_tuples, old_columns


def _get_counting_df(variables):
    pre_dict = {}
    name_to_index = {}

    for index, variable in enumerate(variables):
        name_to_index[variable] = index

    for variable in variables:
        pre_dict[variable] = []

    for n in range(2 ** len(variables)):
        binary_value = _utils.get_bin(n, len(variables))
        for variable, digit in zip(variables, binary_value):
            pre_dict[variable].append(digit)

    return pd.DataFrame(pre_dict)


def _create_base_df(table):
    individual_columns, probability_tuples, old_columns = _get_variables_separated(table)
    df = _get_counting_df(individual_columns)
    for col in old_columns:
        df[col] = [-1 for _ in range(len(df))]
    return df


def get_probability_tables(table: pd.DataFrame):
    records = _get_dict_records(table)
    df = _create_base_df(table)
    len_old_columns = len(df.columns) - len(records)

    for index, row in df.iterrows():
        for header in df.columns[len_old_columns:]:
            variables = _get_variables_in_column(header)
            ind = ""
            for var in variables:
                ind += str(row[var])
            df.loc[index, header] = records[variables][ind]

    rename_mapper = {col: "P" + col for col in df.columns[len_old_columns:]}
    df.rename(columns=rename_mapper, inplace=True)
    return df
