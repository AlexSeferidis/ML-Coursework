import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from part2_house_value_regression import Regressor, save_regressor, load_regressor

def perform_hyperparameter_search(x=None, y=None): 
    # Ensure to add whatever inputs you deem necessary to this function
    """

    Performs hyperparameter tuning for the Regressor model using Optuna.

    This function splits the input data into training, validation, and test sets,
    defines a search space for hyperparameters, and optimizes the model's performance
    based on validation scores. The best hyperparameters are then used to train a final model.

    Arguments:
        x {pd.DataFrame} -- Input features.
        y {pd.DataFrame} -- Target values.
        
    Returns:
        tuple -- A tuple containing:
            - {Regressor} -- Trained Regressor model with optimal hyperparameters.
            - {dict} -- Dictionary of the best hyperparameters.
            - {pd.DataFrame} -- Test set features.
            - {pd.DataFrame} -- Test set target values.
            - {pd.DataFrame} -- Training set features.
            - {pd.DataFrame} -- Training set target values.
            - {pd.DataFrame} -- Validation set features.
            - {pd.DataFrame} -- Validation set target values.
    """

    #######################################################################
    #                       ** START OF YOUR CODE **                    #
    #######################################################################
    search_results = []
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)

    def objective(trial):
        layer1_neurons = trial.suggest_categorical('layer1_neurons', [32, 64, 128, 256, 512, 1024])
        layer2_neurons = trial.suggest_categorical('layer2_neurons', [32, 64, 128, 256, 512, 1024])
        learning_rate = trial.suggest_categorical('learning_rate', [0.1, 0.01, 0.001])
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
        dropout_rate = trial.suggest_categorical('dropout_rate', [0.3, 0.4, 0.5, 0.6, 0.7])
        
        regressor = Regressor(x_train, nb_epoch=200,
                              layer1_neurons=layer1_neurons,
                              layer2_neurons=layer2_neurons,
                              batchsize=batch_size, dropout_rate=dropout_rate)
        regressor.optimizer = optim.Adam(regressor.parameters(), lr=learning_rate)
        regressor.fit(x_train, y_train)

        val_score = regressor.score(x_val, y_val)

        trial_data = {
            "layer1_neurons": layer1_neurons,
            "layer2_neurons": layer2_neurons,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout rate": dropout_rate,
            "validation_score": val_score
        }
        search_results.append(trial_data)

        return val_score  # Minimize validation loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    results_df = pd.DataFrame(search_results)
    results_df.to_csv("output_csv", index=False)

    print("Best hyperparameters: ", study.best_params)
    print("Best validation score: ", study.best_value)

    best_params = study.best_params
    x_train_val = pd.concat([x_train, x_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    regressor = Regressor(x_train_val, nb_epoch=200,
                          layer1_neurons=best_params['layer1_neurons'],
                          layer2_neurons=best_params['layer2_neurons'],
                          batchsize=best_params['batch_size'], dropout_rate=best_params['dropout_rate'])
    regressor.optimizer = optim.Adam(regressor.parameters(), lr=best_params['learning_rate'])
    regressor.fit(x_train_val, y_train_val)

    return regressor, study.best_params, x_test, y_test, x_train, y_train, x_val, y_val

def main():
    
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")
    
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    
    x_train, x_temp, y_train, y_temp = train_test_split(x_train, y_train, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
    regressor, best_params, x_test, y_test, x_train, y_train, x_val, y_val = perform_hyperparameter_search(x_train, y_train)
    #regressor = Regressor(x_train)
    #regressor.fit(x_train, y_train)

    # Save the best model
    save_regressor(regressor)

    # Load the model to ensure it works correctly
    loaded_regressor = load_regressor()
    if loaded_regressor:
        test_score = loaded_regressor.score(x_test, y_test)
        print(f"\nTest score of loaded regressor: {test_score}\n")
        train_score = loaded_regressor.score(x_train, y_train)
        print(f"\nTrain score of loaded regressor: {train_score}\n")
        val_score = loaded_regressor.score(x_val, y_val)
        print(f"\nValidation score of loaded regressor: {val_score}\n")

if __name__ == "__main__":
    main()
