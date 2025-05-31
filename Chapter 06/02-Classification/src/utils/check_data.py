""" Utility to check data and its properties """

# imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# get top 10 rows
def get_top_rows(dataset: pd.DataFrame):
    """ Get top 10 rows of a given dataframe """

    # print top 10 rows of dataframe
    print(dataset.head(10))

# get 5 point statistics
def describe_data(dataset: pd.DataFrame):
    """ Print the statistics of the dataset """

    # Get description as dataframe
    description_df = dataset.describe(include='all')

    # Print description with expanded width
    print(description_df.to_string(max_colwidth=1000))

    # Get dataset information
    info_df = dataset.info()

    # Print Dataset information
    print(info_df)

# get top 5 value counts
def value_counts(dataset: pd.Series):
    """ Print top 5 value counts """

    # print top 5 values
    print(pd.DataFrame(dataset.value_counts()).head(5))

# function to return numeric and string columns
def get_col_types(dataset: pd.DataFrame):
    """
    Identify if the col. is numeric or string.
    Return 2 lists: one for num cols and one
    for string cols to the caller module.
    """

    # define empty lists
    num_cols = []
    str_cols = []

    # identify columns
    for each_col in dataset.columns:
        try:
            _ = dataset[each_col] / 2
            num_cols.append(each_col)
        except TypeError as e:
            _ = e
            str_cols.append(each_col)

    # return lists
    return (num_cols, str_cols)

# function to show distribution plots
def get_distribution(dataset: pd.DataFrame):
    """
    First, get the numerical and string columns from
    another utility function in this script.
    Then for each numerical col, print the displot().
    For each string col, print the countplot().
    """

    # get the num cols and str cols
    num_cols, str_cols = get_col_types(dataset)

    # for each num col, display the displot()
    for each_col in num_cols:
        sns.displot(dataset[each_col])
        plt.plot()

    # for each str col, display the countplot()
    for each_col in str_cols:
        sns.countplot(dataset[each_col])
        plt.plot()


# End
