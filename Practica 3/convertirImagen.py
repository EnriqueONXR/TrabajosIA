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

        # Aplicar el filtro de mejora de bordes
        imagen_bordes = aplicar_filtro_bordes(imagen_gris)

        # Rotar la imagen
        imagen_rotada = rotar_imagen(imagen, 45)

        # Mejorar píxeles (histograma)
        imagen_mejorada = mejorar_histograma(imagen)

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

        print(f"Imagen procesada y guardada: {archivo}")

print("Procesamiento completado.")
