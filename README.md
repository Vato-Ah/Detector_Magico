# Detector Mágico - Proyecto Final de Inteligencia Artificial

## 1. Descripción del Proyecto
Sistema de visión por computadora capaz de identificar 5 categorías de objetos cotidianos (celular, plátano, cuaderno, mouse, audífonos) en tiempo real a través de la cámara web. El proyecto implementa Transfer Learning utilizando la arquitectura MobileNetV2 pre-entrenada en ImageNet.

## 2. Dataset
El conjunto de datos es una combinación de imágenes descargadas de internet y fotografías propias para garantizar la robustez del modelo en entornos reales.
- **Estructura:**
  - Train: ~340 imágenes (entrenamiento y aumento de datos).
  - Validation: ~85 imágenes (ajuste de hiperparámetros).
  - Test (Fotos propias): 92 imágenes (evaluación final).
- **Preprocesamiento:** Redimensionamiento a 224x224, normalización de píxeles (1/255), conversión a RGB y aplicación de pesos de clase (class weights) para balancear el aprendizaje.

## 3. Arquitectura del Modelo
- **Base:** MobileNetV2 (congelada para Feature Extraction).
- **Capas personalizadas:**
  - GlobalAveragePooling2D
  - Dropout (0.5)
  - Dense (128 neuronas, activación ReLU)
  - Dropout (0.3)
  - Dense (5 neuronas, activación Softmax)

## 4. Métricas y Resultados
El modelo fue evaluado utilizando un conjunto de prueba de 130 fotografías propias tomadas en condiciones de iluminación y ángulos variables.

- **Accuracy en Validación**: ~99%
- **Accuracy en Test (Fotos propias)**: 82.00% (Supera el requisito mínimo del 80%).

### Reporte de Clasificación (Test):
| Clase | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Audífonos | 0.91 | 0.91 | 0.91 | 32 |
| Celular | 0.31 | 1.00 | 0.47 | 9 |
| Cuaderno | 0.97 | 0.85 | 0.90 | 39 |
| Mouse | 1.00 | 0.61 | 0.75 | 33 |
| Plátano | 1.00 | 0.88 | 0.94 | 17 |
| **Overall** | **0.92** | **0.82** | **0.84** | **130** |

### Análisis de Errores:
Se identificaron dos patrones de error que demuestran cómo el modelo aprende características visuales locales:
1. **Falsos positivos en "Celular"**: El modelo tiene un Recall de 1.00, pero una Precision de 0.31. Clasifica cuadernos cerrados de portada oscura como "celular" debido a la similitud en la forma rectangular y textura.
2. **Falsos negativos en "Mouse"**: Con una Precision de 1.00 pero Recall de 0.61, el modelo a veces no detecta el mouse o lo confunde con audífonos debido a la similitud en colores (azul/negro) y formas curvas ergonómicas.

## 5. Instrucciones de Ejecución

### Requisitos Previos
- Python 3.10 o superior.
- Cámara web conectada.

### Instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Vato-Ah/Detector_Magico.git
   cd Detector_Magico
   
2. Crear y activar el entono virtual:
    py -m venv venv
    venv\Scripts\activate

3. Instalar dependencias: 
    py -m pip install -r requirements.txt

4. Ejecucuion:
    py app_tiempo_real.py

## 6. Archivos del Proyecto
app_tiempo_real.py: Script principal para la inferencia en tiempo real con OpenCV.
detector_magico.h5: Pesos del modelo entrenado.
matriz_confusion.png: Gráfica de la matriz de confusión del conjunto de prueba.
historial_entrenamiento.png: Gráficas de Accuracy y Loss durante el entrenamiento.

## 7. Entrenamiento del Modelo
El modelo fue entrenado utilizando Google Colab con GPU T4. El notebook completo está disponible en el archivo `entrenamiento_colab.ipynb` y contiene:
- Carga y preprocesamiento del dataset
- Implementación de Transfer Learning con MobileNetV2
- Cálculo y aplicación de class weights
- Entrenamiento y evaluación
- Generación de métricas y gráficas