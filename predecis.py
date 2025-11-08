import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

modelo = tf.keras.models.load_model("modelo_mnist.h5")

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

predicciones = modelo.predict(x_test[:10])
print(np.argmax(predicciones, axis=1))
print(y_test[:10])
