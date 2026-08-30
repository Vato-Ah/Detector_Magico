import cv2
import numpy as np
import tensorflow as tf
import os

# 1. Configuración inicial
modelo_path = 'detector_magico.h5'

if not os.path.exists(modelo_path):
    print(f"Error: No se encuentra el archivo {modelo_path}")
    print("Asegúrate de haberlo movido a la carpeta del proyecto.")
    exit()

# Cargar el modelo entrenado
print("Cargando modelo...")
model = tf.keras.models.load_model(modelo_path)

# Las categorías deben estar en el mismo orden alfabético
categorias = ["audifonos", "celular", "cuaderno", "mouse", "platano"]

# 2. Configurar la cámara web
print("Iniciando cámara web. Presiona la tecla 'q' para salir.")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara web.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video")
        break
    
    # 3. Preprocesar la imagen para el modelo
  
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Normalizar y añadir la dimensión del batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    
    # 4. Hacer la predicción
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    
    # 5. Preparar el texto a mostrar
    nombre_clase = categorias[class_idx].capitalize()
    porcentaje = confidence * 100
    label = f"{nombre_clase}: {porcentaje:.1f}%"
    
    # 6. Dibujar el resultado en el frame original
    # Fondo negro semitransparente para que el texto sea legible
    cv2.rectangle(frame, (10, 10), (380, 60), (0, 0, 0), -1)
    
    # Cambiar color del texto según la confianza (Verde > 80%, Amarillo 50-80%, Rojo < 50%)
    color_texto = (0, 255, 0) if porcentaje > 80 else (0, 255, 255) if porcentaje > 50 else (0, 0, 255)
    
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)
    
    # 7. Mostrar la ventana de video
    cv2.imshow('Detector Magico - Proyecto Final IA', frame)
    
    # 8. Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Liberar recursos
cap.release()
cv2.destroyAllWindows()
print("Aplicación cerrada correctamente.")