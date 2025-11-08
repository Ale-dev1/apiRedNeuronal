from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import os
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import confusion_matrix, classification_report
import threading

# -----------------------------
# Forzar Eager Execution
# -----------------------------
tf.config.run_functions_eagerly(True)

app = FastAPI(
    title="API MNIST Interactivo",
    version="1.2",
    description="Entrenamiento automático, predicción, feedback incremental y evaluación de MNIST"
)

# Permitir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

RUTA_MODELO = "modelo_mnist.keras"
modelo = None

# -----------------------------
# Clases para recibir datos
# -----------------------------
class Imagen(BaseModel):
    pixels: list  # Lista de 784 valores 0-1

class Feedback(BaseModel):
    pixels: list
    correcto: int  # Número correcto 0-9

# -----------------------------
# Crear modelo desde cero
# -----------------------------
def crear_modelo():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Entrenamiento inicial en background
# -----------------------------
def entrenar_modelo_inicial():
    global modelo
    print("Entrenando modelo inicial...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0

    modelo = crear_modelo()
    modelo.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=2)
    modelo.save(RUTA_MODELO)
    print("Modelo inicial entrenado y guardado.")

# -----------------------------
# Cargar modelo si existe o entrenar
# -----------------------------
def cargar_o_entrenar_modelo():
    global modelo
    if os.path.exists(RUTA_MODELO):
        modelo = tf.keras.models.load_model(RUTA_MODELO)
        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Modelo cargado desde disco.")
    else:
        # Entrenamiento en hilo para no bloquear la API
        threading.Thread(target=entrenar_modelo_inicial, daemon=True).start()

cargar_o_entrenar_modelo()

# -----------------------------
# Entrenar modelo manual (opcional)
# -----------------------------
@app.post("/train")
def train_model(epochs: int = 5):
    global modelo
    entrenar_modelo_inicial()
    return {"mensaje": f"Modelo entrenado y guardado en {RUTA_MODELO}"}

# -----------------------------
# Predecir dígito
# -----------------------------
@app.post("/predecir")
def predecir(data: Imagen):
    global modelo
    if modelo is None:
        return {"error": "Modelo aún no entrenado. Intenta dentro de unos segundos."}

    x = np.array(data.pixels, dtype=np.float32).reshape(1, 784)
    pred = modelo.predict(x)
    numero = int(np.argmax(pred))
    probabilidades = pred.flatten().tolist()
    return {"prediccion": numero, "probabilidades": probabilidades}

# -----------------------------
# Feedback incremental
# -----------------------------
@app.post("/feedback")
def feedback(data: Feedback):
    global modelo
    if modelo is None:
        return {"error": "Modelo aún no entrenado. Intenta dentro de unos segundos."}

    x = np.array(data.pixels, dtype=np.float32).reshape(1, 784)
    y = np.array([data.correcto], dtype=np.int32)

    modelo.train_on_batch(x, y)
    modelo.save(RUTA_MODELO)
    return {"mensaje": f"Modelo ajustado con el dígito {data.correcto}"}

# -----------------------------
# Evaluar modelo: matriz de confusión y reporte
# -----------------------------
@app.get("/evaluar")
def evaluar():
    global modelo
    if modelo is None:
        return {"error": "Modelo aún no entrenado. Intenta dentro de unos segundos."}

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28*28) / 255.0

    y_pred_probs = modelo.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)
    return {"matriz_confusion": cm, "reporte_clases": report}

# -----------------------------
# -----------------------------
# Obtener dígitos con menor F1-score
# -----------------------------
@app.get("/digitos_problematicos")
def digitos_problematicos():
    global modelo
    if modelo is None:
        if os.path.exists(RUTA_MODELO):
            modelo = tf.keras.models.load_model(RUTA_MODELO)
            modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            return {"error": "Modelo no entrenado. Primero ejecuta /train"}

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28*28) / 255.0

    y_pred_probs = modelo.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test, y_pred, output_dict=True)

    # Extraer F1-score por dígito
    f1_scores = {}
    for clase in report:
        if clase not in ["accuracy", "macro avg", "weighted avg"]:
            f1_scores[int(clase)] = report[clase]['f1-score']

    # Ordenar de menor a mayor F1-score
    problematicos = sorted(f1_scores.items(), key=lambda x: x[1])
    # Devolver los 3 dígitos con menor F1-score
    return {"digitos_problematicos": problematicos[:3]}

# Endpoint raíz
# -----------------------------
@app.get("/")
def root():
    return {"mensaje": "API MNIST Interactivo funcionando"}
