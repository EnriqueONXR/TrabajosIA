import pygame
import random
import pandas as pd
import numpy as np
import graphviz
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import os


# Inicializar Pygame
pygame.init()
os.chdir(os.path.dirname(__file__))
# Dimensiones de la pantalla
w, h = 800, 400
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables del jugador, bala, nave, fondo, etc.
jugador = None
bala = None
fondo = None
nave = None
menu = None

# Variables de salto
salto = False
salto_altura = 15  # Velocidad inicial de salto
gravedad = 1
en_suelo = True

# Variables de pausa y menú
pausa = False
fuente = pygame.font.SysFont('Arial', 24)
menu_activo = True
modo_auto = False  # Indica si el modo de juego es automático

# Lista para guardar los datos de velocidad, distancia y salto (target)
datos_modelo = []
model = None

# Cargar las imágenes
jugador_frames = [
    pygame.image.load('assets/sprites/mono_frame_1.png'),
    pygame.image.load('assets/sprites/mono_frame_2.png'),
    pygame.image.load('assets/sprites/mono_frame_3.png'),
    pygame.image.load('assets/sprites/mono_frame_4.png')
]

bala_img = pygame.image.load('assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('assets/game/fondo2.png')
nave_img = pygame.image.load('assets/game/ufo.png')
menu_img = pygame.image.load('assets/game/menu.png')

# Escalar la imagen de fondo para que coincida con el tamaño de la pantalla
fondo_img = pygame.transform.scale(fondo_img, (w, h))

# Crear el rectángulo del jugador y de la bala
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 16)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
menu_rect = pygame.Rect(w // 2 - 135, h // 2 - 90, 270, 180)  # Tamaño del menú

# Variables para la animación del jugador
current_frame = 0
frame_speed = 10  # Cuántos frames antes de cambiar a la siguiente imagen
frame_count = 0

# Variables para la bala
velocidad_bala = -10  # Velocidad de la bala hacia la izquierda
bala_disparada = False

# Variables para el fondo en movimiento
fondo_x1 = 0
fondo_x2 = w

# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-15, -10)  # Velocidad aleatoria negativa para la bala
        bala_disparada = True

# Función para reiniciar la posición de la bala
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50  # Reiniciar la posición de la bala
    bala_disparada = False

def reset_model():
    tf.keras.backend.clear_session()

 #Función para manejar el salto
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura   # Mover al jugador hacia arriba
        salto_altura -= gravedad   # Aplicar gravedad (reduce la velocidad del salto)

        #  Si el jugador llega al suelo, detener el salto
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 15   # Restablecer la velocidad de salto
            en_suelo = True

def manejar_autosalto():
    global jugador, salto, salto_altura, gravedad, en_suelo

    if salto:
        jugador.y -= salto_altura  # Mover al jugador hacia arriba
        salto_altura -= gravedad  # Aplicar gravedad (reduce la velocidad del salto)

        # Si el jugador llega al suelo (o ligeramente debajo), detener el salto
        if jugador.y >= h - 100:
            jugador.y = h - 100  # Forzar al jugador al suelo exacto
            salto = False
            salto_altura = 15  # Reiniciar la velocidad inicial del salto
            en_suelo = True


# Función para actualizar el juego
def update():
    global bala, velocidad_bala, current_frame, frame_count, fondo_x1, fondo_x2

    # Mover el fondo
    fondo_x1 -= 1
    fondo_x2 -= 1

    # Si el primer fondo sale de la pantalla, lo movemos detrás del segundo
    if fondo_x1 <= -w:
        fondo_x1 = w

    # Si el segundo fondo sale de la pantalla, lo movemos detrás del primero
    if fondo_x2 <= -w:
        fondo_x2 = w

    # Dibujar los fondos
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0

    # Dibujar el jugador con la animación
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    # Dibujar la nave
    pantalla.blit(nave_img, (nave.x, nave.y))

    # Mover y dibujar la bala
    if bala_disparada:
        bala.x += velocidad_bala

    # Si la bala sale de la pantalla, reiniciar su posición
    if bala.x < 0:
        reset_bala()

    pantalla.blit(bala_img, (bala.x, bala.y))

    # Colisión entre la bala y el jugador
    if jugador.colliderect(bala):
        print("Colisión detectada!")
        # train_model()
        reiniciar_juego()  # Terminar el juego y mostrar el menú


# Función para guardar datos del modelo en modo manual
def guardar_datos():
    global jugador, bala, velocidad_bala, salto
    distancia = abs(jugador.x - bala.x)
    salto_hecho = 1 if salto else 0  # 1 si saltó, 0 si no saltó
    # Guardar velocidad de la bala, distancia al jugador y si saltó o no
    datos_modelo.append((velocidad_bala, distancia, salto_hecho))

# Funcion para graficar los datos
def graficar_datos():
    # Separar datos según el valor de 'salto_hecho'
    x1 = [x for x, y, z in datos_modelo if z == 0]
    x2 = [y for x, y, z in datos_modelo if z == 0]
    target0 = [z for x, y, z in datos_modelo if z == 0]

    x3 = [x for x, y, z in datos_modelo if z == 1]
    x4 = [y for x, y, z in datos_modelo if z == 1]
    target1 = [z for x, y, z in datos_modelo if z == 1]

    # Crear el gráfico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar los puntos con salto_hecho=0 en azul
    ax.scatter(x1, x2, target0, c='blue', marker='o', label='Target=0')

    # Graficar los puntos con salto_hecho=1 en rojo
    ax.scatter(x3, x4, target1, c='red', marker='x', label='Target=1')

    # Etiquetas y leyenda
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('target')
    ax.legend()

    plt.show()

#Funcion para hacer el decision tree
def graficar_arbol():
    # Separar datos
    x1 = [x for x, y, z in datos_modelo]
    x2 = [y for x, y, z in datos_modelo]
    target0 = [z for x, y, z in datos_modelo]

    # Definir características (X) y etiquetas (y)
    X = list(zip(x1, x2))  # Las dos primeras columnas son las características
    y = target0  # La tercera columna es la etiqueta

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el clasificador de Árbol de Decisión
    clf = DecisionTreeClassifier()

    # # Exportar el árbol de decisión en formato DOT para su visualización
    # dot_data = export_graphviz(clf, out_file=None, 
    #                         feature_names=['Feature 1', 'Feature 2'],  
    #                         class_names=['Clase 0', 'Clase 1'],  
    #                         filled=True, rounded=True,  
    #                         special_characters=True)  

    # # Crear el gráfico con graphviz
    # graph = graphviz.Source(dot_data)

    # # Mostrar el gráfico
    # graph.view()
    
def train_tree():
    global model_tree
    print('entrenando con arbol')
    if len(datos_modelo) < 10:  # Requerir al menos 10 datos para el árbol
        print("No hay datos suficientes para entrenar el árbol de decisión.")
        return

    x1 = [x for x, y, z in datos_modelo]
    x2 = [y for x, y, z in datos_modelo]
    target0 = [z for x, y, z in datos_modelo]

    X = list(zip(x1, x2))
    y = target0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)  # Limitar profundidad para evitar sobreajuste
    model_tree = clf.fit(X_train, y_train)

    # Evaluar el modelo
    accuracy = clf.score(X_test, y_test)
    print(f"Precisión del Árbol de Decisión: {accuracy:.2f}")

# Entrenar modelo    
def train_model():
    global model

    if not datos_modelo:
        print("No hay datos suficientes para entrenar el modelo.")
        return

    # Separar datos
    x1 = [x for x, y, z in datos_modelo]
    x2 = [y for x, y, z in datos_modelo]
    target0 = [z for x, y, z in datos_modelo]

    # Definir características (X) y etiquetas (y)
    X = np.array(list(zip(x1, x2)))  # Convertir a numpy array
    y = np.array(target0)  # Convertir a numpy array

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
    ])

    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrecisión en el conjunto de prueba: {accuracy:.2f}")
    
# Función para pausar el juego y guardar los datos
def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados hasta ahora:", datos_modelo)
    else:
        print("Juego reanudado.")

# Función para mostrar el menú y seleccionar el modo de juego
def mostrar_menu():
    global menu_activo, modo_auto, modo_auto_tree
    pantalla.fill(NEGRO)
    texto = fuente.render("Presiona 'A' para Auto, 'M' para Manual, 'G' para Graficar o 'Q' para Salir", False, BLANCO)
    pantalla.blit(texto, (10 , h // 2))
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    modo_auto = True
                    menu_activo = False
                    if len(datos_modelo) < 10:
                        print("No hay suficientes datos para entrenar el árbol de decisión.")
                    else:
                        train_tree()  # Entrenar el árbol de decisión si no se ha hecho antes
                elif evento.key == pygame.K_m:
                    reset_model()
                    modo_auto = False
                    menu_activo = False
                elif evento.key == pygame.K_g:
                    modo_auto = False
                    menu_activo = False
                    graficar_arbol()
                    graficar_datos()
                elif evento.key == pygame.K_q:
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

# Función para reiniciar el juego tras la colisión
def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    menu_activo = True  # Activar de nuevo el menú
    jugador.x, jugador.y = 50, h - 100  # Reiniciar posición del jugador
    bala.x = w - 50  # Reiniciar posición de la bala
    nave.x, nave.y = w - 100, h - 100  # Reiniciar posición de la nave
    bala_disparada = False
    salto = False
    en_suelo = True
    # Mostrar los datos recopilados hasta el momento
    print("Datos recopilados para el modelo: ", datos_modelo)
    mostrar_menu()  # Mostrar el menú de nuevo para seleccionar modo

def main():
    global salto, en_suelo, bala_disparada

    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al inicio
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:  # Detectar la tecla espacio para saltar
                    salto = True
                    en_suelo = False
                if evento.key == pygame.K_p:  # Presiona 'p' para pausar el juego
                    pausa_juego()
                if evento.key == pygame.K_q:  # Presiona 'q' para terminar el juego
                    print("Juego terminado. Datos recopilados:", datos_modelo)
                    pygame.quit()
                    exit()

        if not pausa:
            if modo_auto:
                # Obtener las características actuales
                distancia = abs(jugador.x - bala.x)
                velocidad = velocidad_bala

                # Hacer una predicción
                entrada = np.array([[velocidad, distancia]])
                prediccion = model_tree.predict(entrada)[0] > 0.5

                if prediccion == 1 and en_suelo:
                    print("¡Saltando automáticamente!")  # Mensaje de depuración
                    salto = True
                    en_suelo = False

            # Manejar el salto si está activado
            if salto:
                manejar_autosalto()

            # Modo manual: el jugador controla el salto
            if not modo_auto:
                if salto:
                    salto = True
                    en_suelo = False
                    manejar_salto()
                # Guardar los datos si estamos en modo manual
                guardar_datos()

            # Actualizar el juego
            if not bala_disparada:
                disparar_bala()
            update()

        # Actualizar la pantalla
        pygame.display.flip()
        reloj.tick(30)  # Limitar el juego a 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()