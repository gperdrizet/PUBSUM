import pandas as pd
import numpy as np
from typing import List, Union

def clean_out_of_memory_errors(
    data: pd.DataFrame, 
    unique_condition_columns: List[str],
    oom_columns: List[str], 
    str_columns: List[str], 
    int_columns: List[str], 
    float_columns: List[str],
    oom_replacement_val: Union[str, int, float]
) -> pd.DataFrame:

    # Pick a colum that could/does contain OOMs and make sure it's string
    data[oom_columns[0]] = data[oom_columns[0]].astype(str)

    # Get rows from the dataframe where that column is OOM
    oom = data[data[oom_columns[0]] == 'OOM']

    # Then grab the just the columns which uniquely specify the condition which
    # caused the out-of-memory error, excluding things like replicate, abstract
    # number etc. and make it a list
    oom = oom[unique_condition_columns]
    oom_conditions = oom.to_numpy().tolist()

    # Loop on data columns which will contain the 'OOM' error string
    for oom_column in oom_columns:

        # Loop on the set(s) of conditions which caused the OOM
        for condition in oom_conditions:

            # Mask and replace values in rows which contain OOM error string
            data[oom_column].loc[
                (data[unique_condition_columns] == condition).all(1)
            ] = oom_replacement_val

    # Fix dtypes
    for column in str_columns:
        data[column] = data[column].astype(str)
    
    for column in int_columns:
        data[column] = data[column].astype('Int64')

    for column in float_columns:
        data[column] = data[column].astype(float)

    return data