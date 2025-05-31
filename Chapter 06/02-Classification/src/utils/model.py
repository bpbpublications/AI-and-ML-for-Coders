""" Set of utilities to create, evaluate adn tune ML models """

# imports
import warnings

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# function to fit a Decision tree model
def fit_model(x_train: pd.DataFrame, y_train: pd.Series, model_params: dict = None):
    """ Fit model given train data """

    # check if model parameters are passed
    # and fit the model accordingly
    if model_params is None:
        # create model object
        dt_model = DecisionTreeClassifier()

        # fit model with training data
        dt_model.fit(x_train, y_train)

    else:
        # unpack model parameters
        max_depth = model_params['max_depth']
        min_samples_split = model_params['min_samples_split']
        min_samples_leaf = model_params['min_samples_leaf']
        criterion = model_params['criterion']
        max_features = model_params['max_features']

        # create model object with parameters
        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            min_samples_split=min_samples_split
        )

        # fit model with training data
        dt_model.fit(x_train, y_train)

    # return model object
    return dt_model

# function to evaluate model
def evaluate_model(model: DecisionTreeClassifier, x_test:pd.DataFrame, y_test: pd.Series):
    """ Evaluate model given test data """

    # get predictions from the model
    y_pred = model.predict(x_test)

    # generate classification report
    class_report = classification_report(y_pred, y_test)

    # generate confusion matrix
    conf_matrix = confusion_matrix(y_pred, y_test)

    # return the metrics
    return (class_report, conf_matrix)

# function to tune hyperparameters
def hp_tuning(model: DecisionTreeClassifier, x_train: pd.DataFrame, y_train: pd.Series):
    """ Perform hyperparamter tuning, and return the improved model """

    # filter warnings
    warnings.filterwarnings('ignore')

    # define basic set of hyperparameters range
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 5, 10, 15],
        'criterion': ['gini', 'entropy'],  # Splitting criteria
        'max_features': ['auto', 'sqrt', 'log2']  # Feature selection
    }

    # Create a GridSearchCV object
    grid_dtree = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=10)

    # Fit the grid search
    grid_dtree.fit(x_train, y_train)

    # return model, best parameters and best score
    return (grid_dtree.best_params_, grid_dtree.best_estimator_)

# End.
