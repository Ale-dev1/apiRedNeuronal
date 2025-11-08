import cv2
import mediapipe as mp
import csv
import os

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Crea carpeta de datos
os.makedirs("datos_manos", exist_ok=True)
archivo_salida = open("datos_manos/dataset_vocales.csv", mode="a", newline="")
csv_writer = csv.writer(archivo_salida)

print("=== Captura de manos iniciada ===")
print("Presiona una tecla (A,E,I,O,U) para grabar esa letra, 'q' para salir")

letra_actual = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(rgb)

    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer coordenadas normalizadas
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y])

            if letra_actual:
                fila = [letra_actual] + coords
                csv_writer.writerow(fila)
                print(f"Guardado: {letra_actual} ({len(coords)} coordenadas)")

    cv2.putText(frame, f"Letra actual: {letra_actual}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Captura de Manos', frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla in [ord('a'), ord('e'), ord('i'), ord('o'), ord('u')]:
        letra_actual = chr(tecla).upper()
        print(f"→ Grabando gestos para: {letra_actual}")

cap.release()
archivo_salida.close()
cv2.destroyAllWindows()
