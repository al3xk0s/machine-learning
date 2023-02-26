from model_result import ModelWrapper
from data_provider import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import pandas as pd


def get_forest_model():
    return RandomForestClassifier()


def get_logistic_regression_model():
    return LogisticRegression()


def get_decision_tree():
    return DecisionTreeClassifier()


def main():
    target_column_name = 'is_hit'

    train, test = split_data(get_data('data/spotify.csv'))
    prepare_data(target_column_name, train, test)
    print(train.head(10))
    y, train = split_data_to_target_and_other(target_column_name, train)
    x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=17)
    print(train.shape, test.shape, y.shape)

    models = [
        ModelWrapper(get_forest_model(), 'Random forest'),
        ModelWrapper(get_logistic_regression_model(), 'Logistic regression'),
        ModelWrapper(get_decision_tree(), 'Decision tree'),
    ]

    for model in models:
        model.fit(x_train, y_train)

    validation_results = [m.validate(x_test, y_test) for m in models]

    for model_result in validation_results:
        print(model_result)
        print()


main()
