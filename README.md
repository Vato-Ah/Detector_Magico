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
El modelo fue evaluado utilizando un conjunto de prueba compuesto exclusivamente por fotografías propias tomadas en condiciones de iluminación y ángulos variables.

- **Accuracy en Validación:** 99.39%
- **Accuracy en Test (Fotos propias):** 84.78% (Cumple y supera el requisito mínimo del 80%).

### Reporte de Clasificación (Test):
| Clase | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| Audífonos | 0.94 | 0.94 | 0.94 |
| Celular | 0.43 | 1.00 | 0.60 |
| Cuaderno | 1.00 | 0.72 | 0.84 |
| Mouse | 0.94 | 0.89 | 0.92 |
| Plátano | 1.00 | 0.82 | 0.90 |

### Análisis de Errores:
Se identificaron dos patrones de error interesantes que demuestran cómo el modelo aprende características visuales locales en lugar de conceptos semánticos:
1. **Confusión por forma geométrica:** El modelo clasifica cuadernos cerrados de portada oscura como "celular" (98.2% de confianza) debido a la similitud en la forma rectangular y la textura lisa.
2. **Confusión por color y textura:** En ciertas condiciones de luz, el modelo confunde el mouse con los audífonos, ya que ambos comparten tonos azules, superficies brillantes y formas curvas ergonómicas.

## 5. Instrucciones de Ejecución

### Requisitos Previos
- Python 3.10 o superior.
- Cámara web conectada.

### Instalación
1. Clonar el repositorio:
   ```bash
   git clone [URL_DE_TU_REPOSITORIO]
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
