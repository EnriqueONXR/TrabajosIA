import os
import cv2
import shutil

def preprocess_images(input_dir, output_dir, img_size=(150, 150)):
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"La carpeta de entrada no existe: {input_dir}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # Procesar las imágenes de la clase actual
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(class_path, filename)
                try:
                    # Leer y redimensionar la imagen
                    image = cv2.imread(image_path)
                    if image is not None:
                        resized_image = cv2.resize(image, img_size)
                        # Guardar la imagen procesada
                        output_path = os.path.join(class_output_dir, filename)
                        cv2.imwrite(output_path, resized_image)
                except Exception as e:
                    print(f"Error procesando la imagen {image_path}: {e}")

    print(f"Preprocesamiento completado. Imágenes guardadas en: {output_dir}")


input_base_dir = r"C:\Users\kique\Desktop\TrabajosIA\TrabajosIA\predataset\cars" 
output_base_dir = r"C:\Users\kique\Desktop\TrabajosIA\TrabajosIA\Practica 3\ProcessedDataset"

# Ejecutar el preprocesamiento
preprocess_images(input_base_dir, output_base_dir, img_size=(150, 150))