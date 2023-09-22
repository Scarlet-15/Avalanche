from flask import Flask, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

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
app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def index():
    try:
        with open('predicted_values_only_file', 'rb') as f:
            predicted_values_list=pickle.load(f)
            input_data = np.array(predicted_values_list).reshape(1, len(predicted_values_list))
            model = tf.keras.models.load_model('classification_models/classification_model_1.h5')
            
            # Make predictions
            predictions = model.predict(input_data)
            class_list = ["LOW CHANCE OF AVALANCHE ✅", "RISK OF AVALANCHE ⚠️", "HIGH CHANCE OF AVALANCHE 🚨"]
            prediction_index = np.argmax(predictions[0])
            prediction_text = class_list[prediction_index]

            return render_template("index.html", prediction_text=prediction_text, prediction_index=prediction_index)
    except Exception as e:
        return f"An error occurred: {str(e)}"
if __name__ == '__main__':
    app.run(debug=True,port=8080)

#[[99.188835, 24.952957, 538.39594, -105.65974, 5.5562205, 0.0, -105.23563, 0.0, 1370.941, 20.744682, 0.0, -1233.0911, 46.72705, 0.0, 1.0544688, 0.0]]