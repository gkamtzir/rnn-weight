import plac
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def predict(X_test, regressor, scaler):
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_weight = regressor.predict(X_test)
    predicted_weight = scaler.inverse_transform(predicted_weight)
    return predicted_weight

def visualize_results(testing_set, predicted_weight):
    # Visualising the results.
    plt.plot(testing_set, color = 'red', label = 'Real Weight')
    plt.plot(predicted_weight, color = 'blue', label = 'Predicted Weight')
    plt.title('Weight Prediction')
    plt.xlabel('Time (in days)')
    plt.ylabel('Weight (in kg)')
    plt.legend()
    plt.show()
    
def load_data():
    dataset_train = pd.read_csv("weight_train.csv", header = None).T
    training_set = dataset_train.iloc[:, 0].values.reshape(-1, 1)
    
    dataset_test = pd.read_csv("weight_test.csv", header = None).T
    testing_set = dataset_test.iloc[:, 0].values.reshape(-1, 1)
    return dataset_train, training_set, dataset_test, testing_set

def train_rnn(X_train, y_train, epochs, output_weights):
    # Creating RNN.
    regressor = Sequential()
    
    # Adding layers.
    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    
    # Adding a second LSTM layer and some Dropout regularization.
    regressor.add(LSTM(units = 100, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a third LSTM layer and some Dropout regularization.
    regressor.add(LSTM(units = 100, return_sequences = True))
    regressor.add(Dropout(0.2))
    
    # Adding a fourth LSTM layer and some Dropout regularization.
    regressor.add(LSTM(units = 100))
    regressor.add(Dropout(0.2))
    
    # Output layer.
    regressor.add(Dense(units = 1))
    
    # Compiling the RNN.
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set.
    regressor.fit(X_train, y_train, epochs = epochs, batch_size = 32)
    
    # Saving the model.
    regressor.save(f"{output_weights}.h5")
    
    return regressor

@plac.annotations(
    model=("Model name. Defaults to blank 'el' model.", "option", "m", str),
    output_weights=("Weights file name", "option", "o", str),
    window=("LSTM window", "option", "w", int),
    epochs=("Number of training epochs", "option", "e", int),
)
def main(model = None, output_weights = "weights_model", window = 4, epochs = 100):
    # Applying scaling.
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0, 1))
    
    dataset_train, training_set, dataset_test, testing_set = load_data()
    
    # Preparing the training set.
    X_train = []
    y_train = []
    if model is None:
        training_set = scaler.fit_transform(training_set)
        for i in range(window, training_set.shape[0]):
            X_train.append(training_set[i - window:i, 0])
            y_train.append(training_set[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Training the RNN.
        regressor = train_rnn(X_train, y_train, epochs, output_weights)
        
        # Testing.
        dataset_total = pd.concat((dataset_train[0], dataset_test[0]), axis = 0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - window:].values
        inputs = inputs.reshape(-1,1)
        inputs = scaler.transform(inputs)
        X_test = []
        for i in range(window, inputs.shape[0]):
            X_test.append(inputs[i-window:i, 0])
            
        predicted_weight = predict(X_test, regressor, scaler)
        
        visualize_results(testing_set, predicted_weight)
    else:
        # Loading model.
        regressor = load_model(f"{model}.h5")
        dataset_total = pd.concat((dataset_train[0], dataset_test[0]), axis = 0)
        inputs = dataset_total[:].values
        inputs = inputs.reshape(-1,1)
        inputs = scaler.fit_transform(inputs)
        for i in range(window, inputs.shape[0]):
            X_train.append(inputs[i - window:i, 0])
            y_train.append(inputs[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Testing.
        inputs = dataset_total[len(dataset_total) - window:].values
        inputs = inputs.reshape(-1,1)
        inputs = scaler.transform(inputs)
        X_test = [inputs[:, 0]]
        
        predicted_weight = predict(X_test, regressor, scaler)
        print(f"Tomorrow your weight will be: {predicted_weight[0][0]}")
        
if __name__ == "__main__":
    plac.call(main)