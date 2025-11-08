import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

modelo = tf.keras.models.load_model("modelo_mnist.h5")

(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

ejemplo = x_test[0].reshape(1, 784)  # Forma correcta para MLP
print("Etiqueta real:", y_test[0])
pred = modelo.predict(ejemplo)
print("Predicción:", np.argmax(pred))
