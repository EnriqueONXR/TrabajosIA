import tensorflow as tf 
import numpy as np
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog


model = tf.keras.models.load_model('cnn_cars_split.h5')

class_labels = {
    0: "chevrolet camaro v6 2005",
    1: "Chevrolet Suburban 2007",
    2: "nissan tsuru v16 2017",
    3: "volkswagen beetle 1970",
    4: "volkswagen jetta 2012 exterior"
}

# Función para cargar y procesar la imagen
def process_image(img_path, target_size=(150, 150)):
  
    img = image.load_img(img_path, target_size=target_size)  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0  # Normaliza los valores de píxeles
    return img_array

def predict_image(img_path):
    
    processed_img = process_image(img_path)
    predictions = model.predict(processed_img)  
    class_index = np.argmax(predictions[0])  # Índice de la clase con mayor probabilidad
    confidence = predictions[0][class_index]  
    class_name = class_labels[class_index] 
    return class_name, confidence


def select_image():
    
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True)  

    file_path = filedialog.askopenfilename(
        title="Seleccione una imagen de un automóvil",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path

# Programa principal
def main():
    print("Clasificador de automóviles basado en modelo entrenado.")
    while True:
        img_path = select_image()  # Seleccionar imagen
        if not img_path:
            print("No se seleccionó ningún archivo. Intente nuevamente.")
            continue

        # Realizar predicción
        class_name, confidence = predict_image(img_path)
        print(f"\nLa imagen pertenece al modelo: {class_name} con una confianza del {confidence * 100:.2f}%.\n")

       
        repeat = input(" ").strip().lower()
        if repeat != 's':
            print("Gracias por usar el clasificador. ¡Hasta pronto!")
            break

if __name__ == "__main__":
    main()