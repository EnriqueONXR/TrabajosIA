import os
import cv2
import numpy as np

# Ruta del dataset
dataset_path = r"C:\Users\kique\Desktop\TrabajosIA\TrabajosIA\Practica 3\ProcessedDataset"
output_path = os.path.join(dataset_path, "ProcessedImages")

# Crear la carpeta de salida si no existe
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Función para aplicar un filtro de mejora de bordes
def aplicar_filtro_bordes(imagen):
    return cv2.Canny(imagen, 100, 200)

# Función para rotar una imagen
def rotar_imagen(imagen, angulo):
    (h, w) = imagen.shape[:2]
    centro = (w // 2, h // 2)
    matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(imagen, matriz_rotacion, (w, h))

# Función para mejorar los píxeles usando histograma
def mejorar_histograma(imagen):
    if len(imagen.shape) == 2:  # Imagen en escala de grises
        return cv2.equalizeHist(imagen)
    elif len(imagen.shape) == 3:  # Imagen en color
        canales = cv2.split(imagen)
        canales_eq = [cv2.equalizeHist(canal) for canal in canales]
        return cv2.merge(canales_eq)

# Función para invertir horizontalmente
def invertir_horizontal(imagen):
    return cv2.flip(imagen, 1)

# Función para invertir verticalmente
def invertir_vertical(imagen):
    return cv2.flip(imagen, 0)

# Función para invertir colores
def invertir_colores(imagen):
    return cv2.bitwise_not(imagen)

# Función para convertir a blanco y negro
def convertir_blanco_negro(imagen):
    _, imagen_bn = cv2.threshold(imagen, 128, 255, cv2.THRESH_BINARY)
    return imagen_bn

# Recorrer todas las subcarpetas e imágenes en el dataset
for root, _, files in os.walk(dataset_path):
    for archivo in files:
        archivo_path = os.path.join(root, archivo)

        # Verificar que sea un archivo de imagen
        if not archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        # Leer la imagen
        imagen = cv2.imread(archivo_path)

        # Si la imagen es inválida, omitirla
        if imagen is None:
            print(f"No se pudo leer la imagen: {archivo}")
            continue

        # Convertir la imagen a escala de grises
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Aplicar transformaciones
        imagen_bordes = aplicar_filtro_bordes(imagen_gris)
        imagen_rotada = rotar_imagen(imagen, 45)
        imagen_mejorada = mejorar_histograma(imagen)
        imagen_invertida_h = invertir_horizontal(imagen)
        imagen_invertida_v = invertir_vertical(imagen)
        imagen_colores_invertidos = invertir_colores(imagen)
        imagen_blanco_negro = convertir_blanco_negro(imagen_gris)

        # Crear la misma estructura de subcarpetas en la salida
        relativa_ruta = os.path.relpath(root, dataset_path)
        carpeta_salida = os.path.join(output_path, relativa_ruta)
        if not os.path.exists(carpeta_salida):
            os.makedirs(carpeta_salida)

        # Guardar las imágenes procesadas
        base_nombre = os.path.splitext(archivo)[0]
        cv2.imwrite(os.path.join(carpeta_salida, f"{base_nombre}_bordes.jpg"), imagen_bordes)
        cv2.imwrite(os.path.join(carpeta_salida, f"{base_nombre}_rotada.jpg"), imagen_rotada)
        cv2.imwrite(os.path.join(carpeta_salida, f"{base_nombre}_mejorada.jpg"), imagen_mejorada)
        cv2.imwrite(os.path.join(carpeta_salida, f"{base_nombre}_inv_horizontal.jpg"), imagen_invertida_h)
        cv2.imwrite(os.path.join(carpeta_salida, f"{base_nombre}_inv_vertical.jpg"), imagen_invertida_v)
        cv2.imwrite(os.path.join(carpeta_salida, f"{base_nombre}_inv_colores.jpg"), imagen_colores_invertidos)
        cv2.imwrite(os.path.join(carpeta_salida, f"{base_nombre}_blanco_negro.jpg"), imagen_blanco_negro)

        print(f"Imagen procesada y guardada: {archivo}")

print("Procesamiento completado.")
