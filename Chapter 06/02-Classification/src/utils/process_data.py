""" Utility functions to process dataset """

# imports
from src.utils import check_data

import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split

# function to remove NULLs from dataset
def remove_nulls(dataset: pd.DataFrame):
    """ remove all nulls from dataset, and return dataset """

    # remove nulls
    dataset.dropna(inplace=True)

    # return dataset
    return dataset

# function to normalize all numeric columns
def normalize_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Given an input Pandas dataframe, identify
    the numerical columns in the dataframe from
    a separate utility. Then normalize values in
    each of these columns. Finally return new dataframe
    with all normalised values.
    """

    # get num cols
    num_cols, _ = check_data.get_col_types(dataset)

    # create temp dataframe with
    # the numeric columns only
    df_temp = dataset[num_cols]

    # normalize values of all columns
    # in the temporary dataframe
    df_num_norm = pd.DataFrame(normalize(df_temp), columns=num_cols)

    # return normalized dataframe
    return df_num_norm

# function to apply standard scaling to all numeric columns
def standardize_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Given an input Pandas dataframe, identify
    the numerical columns in the dataframe from
    a separate utility. Then standardize values in
    each of these columns. Finally return new dataframe
    with all standardized values.
    """

    # get num cols
    num_cols, _ = check_data.get_col_types(dataset)

    # create temp dataframe with
    # the numeric columns only
    df_temp = dataset[num_cols]

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler on the data
    scaler.fit(df_temp)

    # Create a copy to avoid modifying the original DataFrame
    df_standard = df_temp.copy()

    # Transform (standardize) the numerical columns
    df_standard[num_cols] = scaler.transform(df_temp[num_cols])

    # return the final dataframe
    return df_standard

# function to concatenate given dataframes
def concat_datasets(dataset1: pd.DataFrame, dataset2: pd.DataFrame):
    """ Returns concatenated dataframe given 2 input dataframes """

    # create empty list of all columns
    # for the final dataset
    all_columns = []

    # set all columns from both the input datasets
    for each_col in list(dataset1.columns):
        all_columns.append(each_col)
    for each_col in list(dataset2.columns):
        all_columns.append(each_col)

    # prepare final dataframe
    df_ready_data = pd.concat([dataset1, dataset2], axis=1, ignore_index=True)

    # set the column names of the final dataset
    df_ready_data.columns = all_columns

    # return the final dataset
    return df_ready_data

# Function to split dataset into train/test
def create_train_test(dataset: pd.DataFrame, target_col):
    """ Given an input dataset, divide data into train and test """

    # create features and target data
    features_data = dataset.drop(target_col, axis=1)
    target_data = dataset[target_col]

    # create train-test split
    x_train, x_test, y_train, y_test = train_test_split(features_data, target_data, test_size=0.3)

    # return the split
    return (x_train, x_test, y_train, y_test)


# End.
