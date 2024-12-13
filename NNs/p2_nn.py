import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split

class Regressor(nn.Module):
    def __init__(self, x, nb_epoch = 200, layer1_neurons=256, layer2_neurons=32, batchsize=128, dropout_rate=0.5):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        super(Regressor, self).__init__()
        self.lb = None
        self.impute_x = pd.DataFrame()
        self.scaler_x = None
        self.scaler_y = None

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batchsize = batchsize
        
        self.layers = [
            nn.Linear(self.input_size, layer1_neurons), 
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(layer1_neurons, layer2_neurons),
            nn.ReLU(),
            nn.Linear(layer2_neurons, self.output_size)                                             
        ]
        
        self.model = nn.Sequential(*self.layers)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) #56179.99279558022
        
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None
        
        # Handle missing values
        numeric = x.select_dtypes(include=['float']) 
        x.loc[:, numeric.columns] = numeric.fillna(numeric.mean())

        if y is not None:
            y = y.fillna(y.mean())
            self.scaler_y = StandardScaler()
            y = self.scaler_y.fit_transform(y)


        if not training:
            one_hot = self.lb.transform(x.iloc[:, -1])
            one_hot_df = pd.DataFrame(one_hot, columns=self.lb.classes_)
            x_base = x.iloc[:, :-1].reset_index(drop=True)
            one_hot_df = one_hot_df.reset_index(drop=True)
            x = pd.concat([x_base, one_hot_df], axis=1)
            x = self.scaler_x.fit_transform(x)

        else:
            self.lb = LabelBinarizer()
            one_hot = self.lb.fit_transform(x.iloc[:, -1])
            one_hot_df = pd.DataFrame(one_hot, columns=self.lb.classes_)
            x_base = x.iloc[:, :-1].reset_index(drop=True)
            one_hot_df = one_hot_df.reset_index(drop=True)
            x = pd.concat([x_base, one_hot_df], axis=1)
            self.scaler_x = StandardScaler()
            x = self.scaler_x.fit_transform(x)

        x = torch.tensor(pd.DataFrame(x).values, dtype=torch.float32)
        y = torch.tensor(pd.DataFrame(y).values, dtype=torch.float32) if y is not None else None
        return x, y
    
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size= self.batchsize, shuffle=True) 

        for epoch in range(self.nb_epoch):
            epoch_loss = 0
            for batch_X, batch_Y in dataloader:
                self.optimizer.zero_grad()
                predictions = self.model(batch_X) #in this case model is equivalent to forward
                loss = self.criterion(predictions, batch_Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{self.nb_epoch}, Loss: {epoch_loss:.4f}', f'LR: {current_lr:.6f}')
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X).detach().cpu().numpy()
            predictions = self.scaler_y.inverse_transform(predictions)
            
        return predictions

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        y_predictions = self.predict(x)
        return np.sqrt(np.mean((y-y_predictions) ** 2)) # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
         pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor() -> Regressor: 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
         trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


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

 #######################################################################
 # ** END OF YOUR CODE **
 ##################################################################

def example_main():
    
    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")
    
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]
    
    regressor = Regressor(x_train, nb_epoch=200)
    regressor.fit(x_train, y_train)

    # Save the best model
    save_regressor(regressor)

    error = regressor.score(x_train, y_train)
    print(f"\nRegressor error: {error}\n")

if __name__ == "__main__":
    example_main()

