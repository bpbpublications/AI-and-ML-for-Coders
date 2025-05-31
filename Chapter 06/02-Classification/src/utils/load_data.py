""" Utility to load data directly from UCI ML Repo """

# Imports
from ucimlrepo import fetch_ucirepo

# Utility definition
def get_data_from_uci(info=True):
    """ Fetch heart disease data from UCI ML Repo """
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)
    print("Heart Disease Data downloaded from UCI ML Repo.")

    # data (as pandas dataframes)
    features = heart_disease.data.features
    target = heart_disease.data.targets
    print("Datasets for features and target created.")

    # print dataset info if caller wants
    if info:
        # metadata
        print("Metadata for heart disease dataset: ")
        print(heart_disease.metadata)

        # variable information
        print("Heart Disease dataset's variable information:")
        print(heart_disease.variables)

    # return features and target
    return (features, target)

# End.
