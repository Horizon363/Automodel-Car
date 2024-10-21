import cv2
import numpy as np


def cargar_imagenes():
    rojo = cv2.imread("traficlights/se_red.jpg")
    verde = cv2.imread("traficlights/se_green.jpg")
    amarillo = cv2.imread("traficlights/se_ye.jpg")

    if rojo is None:
        print("Error al cargar la imagen de semáforo rojo.")
    if verde is None:
        print("Error al cargar la imagen de semáforo verde.")
    if amarillo is None:
        print("Error al cargar la imagen de semáforo amarillo.")

    return rojo, verde, amarillo


def detectar_semaforo(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    rojo_bajo = np.array([0, 100, 100])
    rojo_alto = np.array([10, 255, 255])
    verde_bajo = np.array([40, 100, 100])
    verde_alto = np.array([80, 255, 255])
    amarillo_bajo = np.array([20, 100, 100])
    amarillo_alto = np.array([30, 255, 255])

    mask_rojo = cv2.inRange(hsv_frame, rojo_bajo, rojo_alto)
    mask_verde = cv2.inRange(hsv_frame, verde_bajo, verde_alto)
    mask_amarillo = cv2.inRange(hsv_frame, amarillo_bajo, amarillo_alto)

    contar_rojo = cv2.countNonZero(mask_rojo)
    contar_verde = cv2.countNonZero(mask_verde)
    contar_amarillo = cv2.countNonZero(mask_amarillo)

    max_contar = max(contar_rojo, contar_verde, contar_amarillo)
    if max_contar == contar_rojo:
        return "rojo"
    elif max_contar == contar_verde:
        return "verde"
    elif max_contar == contar_amarillo:
        return "amarillo"

    return "ninguno"


img_rojo, img_verde, img_amarillo = cargar_imagenes()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    estado_semaforo = detectar_semaforo(frame)
    print(f"Semáforo: {estado_semaforo}")

    cv2.imshow("Frame", frame)

    cv2.imshow("Rojo", cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                                   np.array([0, 100, 100]), np.array([10, 255, 255])))
    cv2.imshow("Verde", cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                                    np.array([40, 100, 100]), np.array([80, 255, 255])))
    cv2.imshow("Amarillo", cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                                       np.array([20, 100, 100]), np.array([30, 255, 255])))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
