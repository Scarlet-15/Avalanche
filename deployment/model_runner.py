import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons: int, n_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast

def make_future_forecasts(values,model,HORIZON_SIZE,WINDOW_SIZE):
    future_forecast = []
    last_window = values[-WINDOW_SIZE:]
    for _ in range(HORIZON_SIZE):
        future_pred = model.predict(tf.expand_dims(last_window, axis=0))
        # print(f"Predicting on:\n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n")
        future_forecast.append(tf.squeeze(future_pred).numpy())
        last_window = np.append(last_window, future_pred)[-WINDOW_SIZE:]
    return future_forecast


model_paths = []

for root, directories, files in os.walk("forecast_models_single_digit"):
    for filename in files:
        # Create the full file path by joining the root and filename
        model_path=os.path.join(root, filename)
        # Replace backslashes with forward slashes and remove the file extension
        model_path = model_path.replace("\\", "/")
        # Append the modified file path to the list
        model_paths.append(model_path)
for root, directories, files in os.walk("forecast_models_double_digit"):
    for filename in files:
        # Create the full file path by joining the root and filename
        model_path=os.path.join(root, filename)
        # Replace backslashes with forward slashes and remove the file extension
        model_path = model_path.replace("\\", "/")
        # Append the modified file path to the list
        model_paths.append(model_path)

data = pd.read_excel('AvalancheTimeseriesData.xlsx')
data=data.drop(["Date","Warning"], axis=1)
forecasted_values={}
forecasted_values_only=[]
try:
    for index,model_path in enumerate(model_paths):
        WINDOW_SIZE = 7
        HORIZON_SIZE = 1
        column_name=data.columns[index]
        column_data_type = data[column_name].dtype
        if column_data_type == "object":
            label_encoder = LabelEncoder()
            data[column_name]=label_encoder.fit_transform(data[column_name])
            data[column_name]=data[column_name].astype('int')
            parameters=np.array(data[data.columns[index]])
            parameter_model = tf.keras.models.load_model(model_path,custom_objects={"NBeatsBlock": NBeatsBlock})
            future_forecast = make_future_forecasts(parameters,parameter_model, HORIZON_SIZE, WINDOW_SIZE)
            # future_forecast=label_encoder.inverse_transform([int(future_forecast[0])])
            
        else:
            parameters=np.array(data[data.columns[index]])
            parameter_model = tf.keras.models.load_model(model_path,custom_objects={"NBeatsBlock": NBeatsBlock})
            future_forecast = make_future_forecasts(parameters,parameter_model, HORIZON_SIZE, WINDOW_SIZE)
        forecasted_values[data.columns[index]]=future_forecast[0]
        forecasted_values_only.append(future_forecast[0])
except Exception as e:
    print(f"An error occurred: {str(e)}")
predicted_values_file=open("predicted_values_file", 'wb')
predicted_values_only_file=open("predicted_values_only_file", 'wb')
pickle.dump(forecasted_values,predicted_values_file)
pickle.dump(forecasted_values_only,predicted_values_only_file)
print(forecasted_values_only)