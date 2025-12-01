# Proyecto de Clasificación de Neumonía en Rayos X

Este proyecto implementa un sistema completo para la clasificación de neumonía en imágenes de rayos X de tórax utilizando descriptores clásicos de forma y textura, junto con algoritmos de machine learning tradicionales y deep learning.

## 📋 Contenido

1. [Descripción](#descripción)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Uso](#uso)
6. [Parte 1: Análisis y Preprocesamiento](#parte-1-análisis-y-preprocesamiento)
7. [Parte 2: Extracción de Descriptores](#parte-2-extracción-de-descriptores)
8. [Parte 3: Clasificación](#parte-3-clasificación)
9. [Notebooks](#notebooks)
10. [Resultados](#resultados)

## 📖 Descripción

Este proyecto está dividido en tres partes principales:

1. **Análisis y Preprocesamiento**: Realiza análisis exploratorio del dataset de rayos X, visualiza la distribución de clases y dimensiones, e implementa un pipeline de preprocesamiento con normalización de tamaño y ecualización de contraste (CLAHE).

2. **Extracción de Descriptores**: Extrae descriptores clásicos de forma y textura de las imágenes:
   - **Forma**: HOG, Momentos de Hu, Descriptores de Contorno, Descriptores de Fourier
   - **Textura**: LBP, GLCM, Filtros de Gabor, Estadísticas de Primer Orden

3. **Clasificación**: Implementa y compara múltiples algoritmos de clasificación:
   - **Métodos Clásicos**: SVM (Linear, RBF, Polynomial), Random Forest, k-NN, Regresión Logística
   - **Deep Learning**: CNN con PyTorch

## 🔧 Requisitos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- Dataset de rayos X de tórax (disponible en [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia))

## 📦 Instalación

### 1. Clonar o descargar el proyecto

```bash
cd trabajo3
```

### 2. Crear entorno virtual (recomendado)

```bash
python3 -m venv venv
# Mac/Linux
source venv/bin/activate  
# Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación

```bash
python -c "import cv2, numpy, pandas, sklearn, torch; print('Instalación correcta')"
```

## 📁 Estructura del Proyecto

```
trabajo3/
├── data/
│   ├── chest_xray/          # Dataset de rayos X (Kaggle)
│   │   ├── train/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   ├── test/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   └── val/
│   ├── metadata.csv          # Metadatos del dataset
│   ├── features_sample.csv  # Características extraídas
│   └── statistics.json      # Estadísticas del dataset
├── notebooks/               # Jupyter notebooks interactivos
│   ├── 1_analisis_preprocesamiento.ipynb
│   ├── 2_extraccion_descriptores.ipynb
│   └── 3_calsificacion_descriptores.ipynb
├── results/                 # Resultados y visualizaciones
│   ├── *.png                # Gráficos y visualizaciones
│   └── final_comparison.csv # Comparación de modelos
├── src/                     # Scripts principales
│   ├── analisis_preprocesamiento.py
│   ├── extraer_descriptores.py
│   └── clasificaccion_descriptores.py
├── venv/                    # Entorno virtual (no incluido en git)
├── requirements.txt         # Dependencias del proyecto
└── README.md               # Este archivo
```

## 🚀 Uso

### Ejecución Secuencial (Recomendado)

Los scripts están diseñados para ejecutarse en orden:

```bash
# 1. Análisis y preprocesamiento
python src/analisis_preprocesamiento.py

# 2. Extracción de descriptores
python src/extraer_descriptores.py

# 3. Clasificación
python src/clasificaccion_descriptores.py
```

### Ejecución Individual

Cada script puede ejecutarse de forma independiente si se cumplen los requisitos previos.

## 📊 Parte 1: Análisis y Preprocesamiento

### Descripción

Esta parte realiza el análisis exploratorio del dataset y prepara las imágenes para el procesamiento posterior.

### Funcionalidades

- **Carga de datos**: Carga y organiza las imágenes del dataset
- **Análisis exploratorio**: 
  - Distribución de clases (NORMAL vs PNEUMONIA)
  - Análisis de dimensiones de imágenes
  - Visualización de ejemplos
- **Preprocesamiento**:
  - Redimensionamiento a tamaño estándar (224x224)
  - Ecualización de contraste (CLAHE)
  - Normalización

### Ejecución

```bash
python src/analisis_preprocesamiento.py
```

### Qué Genera

#### Archivos en `data/`:
- **`metadata.csv`**: Metadatos de todas las imágenes (ruta, split, clase, dimensiones)
- **`statistics.json`**: Estadísticas del dataset (conteos, dimensiones promedio, etc.)

#### Imágenes en `results/`:
- **`ejemplos_imagenes.png`**: Muestras aleatorias de cada clase
- **`distribucion_clases.png`**: Gráficos de distribución de clases
- **`analisis_dimensiones.png`**: Análisis de dimensiones de imágenes
- **`comparacion_redimensionamiento.png`**: Comparación de métodos de redimensionamiento
- **`comparacion_ecualizacion.png`**: Comparación de métodos de ecualización
- **`pipeline_preprocesamiento.png`**: Visualización del pipeline completo

### Ejemplo de Salida

```
✅ Directorio encontrado: ../data/chest_xray
📊 Distribución de datos:
  Train - Normal: 1341
  Train - Pneumonia: 3875
  Test - Normal: 234
  Test - Pneumonia: 390

✅ Dataset creado: 5840 imágenes totales

📊 Estadísticas de distribución:
class  NORMAL  PNEUMONIA
split                   
test      234        390
train    1341       3875

Total imágenes: 5840
Balance de clases: 26.97% NORMAL vs 73.03% PNEUMONIA
```

## 🔍 Parte 2: Extracción de Descriptores

### Descripción

Extrae descriptores clásicos de forma y textura de las imágenes preprocesadas.

### Descriptores Implementados

#### Descriptores de Forma:
1. **HOG (Histogram of Oriented Gradients)**: Captura la distribución de gradientes locales
2. **Momentos de Hu**: 7 momentos invariantes a traslación, rotación y escala
3. **Descriptores de Contorno**: Área, perímetro, circularidad, excentricidad, solidez
4. **Descriptores de Fourier**: Representación del contorno en el dominio de la frecuencia

#### Descriptores de Textura:
1. **LBP (Local Binary Patterns)**: Patrones binarios locales para textura
2. **GLCM (Gray Level Co-occurrence Matrix)**: Matriz de co-ocurrencia de niveles de gris
3. **Filtros de Gabor**: Respuestas a diferentes frecuencias y orientaciones
4. **Estadísticas de Primer Orden**: Media, varianza, asimetría, curtosis, entropía

### Ejecución

```bash
python src/extraer_descriptores.py
```

### Qué Genera

#### Archivos en `data/`:
- **`features_sample.csv`**: Dataset con todas las características extraídas (26,338 características por imagen)

#### Imágenes en `results/`:
- **`hog_visualization.png`**: Visualización de características HOG
- **`hu_moments.png`**: Visualización de momentos de Hu
- **`contour_features.png`**: Contornos detectados
- **`fourier_descriptors.png`**: Descriptores de Fourier
- **`lbp_features.png`**: Visualización de LBP
- **`glcm_features.png`**: Matriz GLCM
- **`gabor_features.png`**: Respuestas de filtros de Gabor
- **`first_order_stats.png`**: Estadísticas de primer orden

### Ejemplo de Salida

```
✅ HOG extraído: 26244 características
✅ Momentos de Hu calculados: 7 características
✅ Descriptores de contorno extraídos: 5 características
✅ Descriptores de Fourier extraídos: 20 coeficientes
✅ LBP extraído: 26 características
✅ Características GLCM extraídas: 6 propiedades
✅ Características de Gabor extraídas: 24 características
✅ Estadísticas de primer orden extraídas: 6 estadísticas

Total: 26338 características
```

## 🤖 Parte 3: Clasificación

### Descripción

Implementa y compara múltiples algoritmos de clasificación usando los descriptores extraídos.

### Algoritmos Implementados

1. **SVM (Support Vector Machine)**:
   - Kernel Linear
   - Kernel RBF
   - Kernel Polynomial

2. **Random Forest**: Clasificador basado en árboles de decisión

3. **k-NN (k-Nearest Neighbors)**: Clasificador basado en vecinos más cercanos

4. **Regresión Logística**: Modelo lineal probabilístico

5. **CNN (Convolutional Neural Network)**: Red neuronal convolucional con PyTorch

### Ejecución

```bash
python src/clasificaccion_descriptores.py
```

### Qué Genera

#### Archivos en `results/`:
- **`pca_analysis.png`**: Análisis de componentes principales
- **`cm_*.png`**: Matrices de confusión para cada modelo
- **`model_comparison.png`**: Comparación visual de modelos
- **`roc_curves.png`**: Curvas ROC de todos los modelos
- **`rf_importance.png`**: Importancia de características (Random Forest)
- **`final_comparison.csv`**: Tabla comparativa de todos los modelos
- **`final_comparison.png`**: Visualización final de comparación

### Métricas Evaluadas

- **Accuracy**: Precisión general
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica de precisión y recall
- **ROC AUC**: Área bajo la curva ROC
- **Validación Cruzada**: 5-fold cross-validation

### Ejemplo de Salida

```
📊 Comparación de Modelos:
         Classifier  Accuracy  Precision  Recall  F1-Score  CV Mean   CV Std  ROC AUC
         SVM Linear      0.80   0.842105    0.80  0.745098   0.8625 0.082916 0.973333
            SVM RBF      0.75   0.562500    0.75  0.642857   0.7750 0.030619 0.920000
     SVM Polynomial      0.75   0.562500    0.75  0.642857   0.7750 0.030619 0.093333
      Random Forest      0.75   0.562500    0.75  0.642857   0.7625 0.025000 0.940000
         k-NN (k=3)      0.80   0.842105    0.80  0.745098   0.8250 0.082916 0.693333
Logistic Regression      0.80   0.842105    0.80  0.745098   0.8750 0.079057 0.986667
      CNN (PyTorch)      0.65   0.422500    0.65  0.512121   0.7802      NaN 0.780220
```

## 📓 Notebooks

Los notebooks de Jupyter proporcionan una versión interactiva de cada script, ideal para experimentación y análisis detallado.

### Ejecutar Notebooks

```bash
# Desde el directorio del proyecto
jupyter notebook notebooks/

# O abrir directamente
jupyter notebook notebooks/1_analisis_preprocesamiento.ipynb
```

### Ventajas de los Notebooks

- Ejecución celda por celda
- Visualización interactiva de resultados
- Fácil modificación de parámetros
- Análisis paso a paso
- Documentación integrada

## 📈 Resultados

### Estructura de Resultados

Todos los resultados se guardan automáticamente en:

- **`results/`**: Imágenes generadas, gráficos y comparaciones
- **`data/`**: Datasets procesados y características extraídas

### Archivos de Salida

Cada ejecución genera:

1. **Archivos CSV**: Datasets con características y resultados
2. **Archivos JSON**: Estadísticas y metadatos estructurados
3. **Imágenes PNG**: Visualizaciones y gráficos de análisis

### Interpretación de Resultados

#### Parte 1 (Preprocesamiento)
- **Distribución de clases**: Balance del dataset (típicamente desbalanceado en datasets médicos)
- **Dimensiones**: Variabilidad en tamaños de imágenes (requiere normalización)
- **Preprocesamiento**: Mejora del contraste y normalización para mejor extracción de características

#### Parte 2 (Extracción)
- **Número de características**: Total de descriptores extraídos (26,338 en este proyecto)
- **Visualizaciones**: Permiten entender qué capturan los descriptores
- **Tiempo de procesamiento**: Depende del tamaño del dataset

#### Parte 3 (Clasificación)
- **Accuracy > 0.75**: Buen rendimiento para dataset médico
- **ROC AUC > 0.85**: Excelente capacidad de discriminación
- **F1-Score**: Balance entre precisión y recall (importante en datasets desbalanceados)
- **Validación cruzada**: Confiabilidad del modelo

## 🔍 Solución de Problemas

### Error: "No se encontró metadata.csv"
- **Solución**: Ejecuta primero `analisis_preprocesamiento.py`

### Error: "No se encontró features_sample.csv"
- **Solución**: Ejecuta primero `extraer_descriptores.py`

### Error: "Dataset no encontrado"
- **Solución**: Descarga el dataset de [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) y colócalo en `data/chest_xray/`

### Memoria insuficiente al extraer características
- **Solución**: Usa `sample_size` en `extract_features_batch()` para procesar una muestra más pequeña

### Modelos con bajo rendimiento
- **Soluciones**:
  - Aumenta el tamaño de la muestra
  - Ajusta hiperparámetros de los modelos
  - Considera usar PCA para reducción de dimensionalidad
  - Prueba diferentes combinaciones de descriptores

## 📝 Notas Técnicas

### Algoritmos y Técnicas Utilizadas

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Ecualización adaptativa de histograma, ideal para radiografías
- **HOG**: Descriptor robusto para detección de objetos
- **Momentos de Hu**: Invariantes a transformaciones geométricas
- **LBP**: Descriptor eficiente para textura
- **GLCM**: Análisis de textura basado en estadísticas de segundo orden
- **PCA (Principal Component Analysis)**: Reducción de dimensionalidad
- **RANSAC**: Eliminación de outliers en matching de características
- **Cross-Validation**: Validación robusta de modelos

### Parámetros Ajustables

En cada script puedes modificar:

- **Parte 1**: Tamaño de redimensionamiento, parámetros CLAHE
- **Parte 2**: Parámetros de cada descriptor (orientaciones HOG, radio LBP, etc.)
- **Parte 3**: Hiperparámetros de modelos, número de componentes PCA, épocas de CNN

### Descriptores y sus Invarianzas

- **Momentos de Hu**: Invariantes a traslación, rotación y escala
- **Descriptores de Fourier**: Invariantes a rotación (solo magnitud)
- **HOG**: Parcialmente invariante a iluminación
- **LBP**: Invariante a cambios de iluminación monotónicos

## 📊 Dataset

### Información del Dataset

- **Fuente**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total de imágenes**: 5,840
- **Clases**: 
  - NORMAL: 1,575 imágenes (27%)
  - PNEUMONIA: 4,265 imágenes (73%)
- **División**:
  - Train: 5,216 imágenes
  - Test: 624 imágenes
  - Val: 16 imágenes

### Características del Dataset

- **Formato**: JPEG
- **Dimensiones**: Variables (promedio ~970x1327 píxeles)
- **Balance**: Desbalanceado (más casos de neumonía)
- **Calidad**: Imágenes médicas reales con variabilidad en calidad y orientación

## 📄 Licencia

Este proyecto es parte de un trabajo académico de la Universidad Nacional.

## 👥 Autor

Daniela Buitrago, estudiante de la Universidad Nacional.

---

**Última actualización**: 2025

Para más información sobre los descriptores y algoritmos, consulta la documentación en cada módulo.
