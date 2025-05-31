""" Heart Disease Prediction """

# Imports
import pickle

import pandas as pd

from src.utils import load_data
from src.utils import check_data
from src.utils import process_data
from src.utils import model

# Main utility
def main():
    """ Performs the required Data and ML Tasks """

    # opening log
    print("Started: Heart Disease Prediction.")

    # get features and target
    features, target = load_data.get_data_from_uci(False)
    features = pd.DataFrame(features)
    target = pd.DataFrame(target)
    print("Got features of type: %s", type(features))
    print("Got target of type: %s", type(target))

    # Check features data
    print("Check features dataset:")
    check_data.get_top_rows(features)

    # Check value counts of target
    print("Check target value counts:")
    check_data.value_counts(target.num)

    # get target column
    target_col = target.columns
    print("The target column is: %s", target_col)

    # Describe Features data
    print("Describe features' dataset:")
    check_data.describe_data(features)

    # Get distribution of features
    print("Visualising distribution of each feature:")
    check_data.get_distribution(features)

    # Get distribution of target
    print("[Optional] Visualize distribution of target:")
    check_data.get_distribution(target)

    # Join features + target dataframe
    all_data = process_data.concat_datasets(features, target)
    print("Features and Target concatenated for NULL removal.")

    # remove nulls from combined dataset
    data_wo_nulls = process_data.remove_nulls(all_data)
    print("Nulls removed from combined dataset of features and target.")

    # Creating the training and testing sets
    x_train, x_test, y_train, y_test = process_data.create_train_test(data_wo_nulls, target_col)
    print("Broken the data into Training and Testing sets.")

    # normalize feature data in the train-test sets
    x_train_norm = process_data.normalize_data(x_train)
    x_test_norm = process_data.normalize_data(x_test)
    print("Features normalized in Training and Test sets.")

    # standard scale feature data in train-test sets
    x_train_scaled = process_data.standardize_data(x_train)
    x_test_scaled = process_data.standardize_data(x_test)
    print("Created standard-scaled features in Training and Test sets.")

    # fit model with normalized data
    dt_norm = model.fit_model(x_train_norm, y_train, None)
    print("Model created with Normalized features and as-is target.")

    # fit model with scaled data
    dt_scaled = model.fit_model(x_train_scaled, y_train, None)
    print("Model created with Standard Scaled features and as-is target.")

    # evaluate model with norm data
    class_report_norm, conf_matrix_norm = model.evaluate_model(dt_norm, x_test_norm, y_test)
    print("Classification Report of Model on Normalized Data:")
    print(class_report_norm)
    print("Confusion Matrix of Model on Normalized Data:")
    print(conf_matrix_norm)

    # evaluate model with scaled data
    class_report_sc, conf_matrix_sc = model.evaluate_model(dt_scaled, x_test_scaled, y_test)
    print("Classification Report of Model on Standard-scaled Data:")
    print(class_report_sc)
    print("Confusion Matrix of Model on Standard-scaled Data:")
    print(conf_matrix_sc)

    # tune hyperparameters with norm data
    (
        norm_best_params, norm_best_est
    ) = model.hp_tuning(dt_norm, x_train_norm, y_train)
    print("Performed Hyperparameter tuning of model with normalized features.")
    print(norm_best_params)

    # tune hyperparameters with scaled data
    (
        scaled_best_params, scaled_best_est
    ) = model.hp_tuning(dt_scaled, x_train_scaled, y_train)
    print("Performed Hyperparameter tuning of model with standard-scaled features.")
    print(scaled_best_params)

    # evaluate tuned model with norm data
    class_report_norm, conf_matrix_norm = model.evaluate_model(norm_best_est, x_test_norm, y_test)
    print("Classification Report of Model on Normalized Data:")
    print(class_report_norm)
    print("Confusion Matrix of Model on Normalized Data:")
    print(conf_matrix_norm)

    # evaluate tuned model with scaled data
    class_report_sc, conf_matrix_sc = model.evaluate_model(scaled_best_est, x_test_scaled, y_test)
    print("Classification Report of Model on Normalized Data:")
    print(class_report_sc)
    print("Confusion Matrix of Model on Normalized Data:")
    print(conf_matrix_sc)

    # export model trained with norm. data
    filename = 'improved_dt_norm_model_v1.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(norm_best_est, file)

    # export model trained with norm. data
    filename = 'improved_dt_scaled_model_v1.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(scaled_best_est, file)

# Entrypoint
if __name__ == "__main__":
    main()

# End
