"""
Parte 2: Extracción de Descriptores Clásicos

Este módulo implementa la extracción de descriptores de forma y textura 
para las imágenes de rayos X.
"""

# Importación de librerías
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from scipy import ndimage
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# Funciones de Preprocesamiento
# ============================================================================

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Aplica CLAHE a la imagen"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


# ============================================================================
# Parte A: Descriptores de Forma
# ============================================================================

def extract_hog_features(img, orientations=9, pixels_per_cell=(8, 8), 
                        cells_per_block=(2, 2), visualize=False):
    """
    Extrae características HOG (Histogram of Oriented Gradients).
    
    Args:
        img: Imagen en escala de grises
        orientations: Número de bins de orientación
        pixels_per_cell: Tamaño de celda en píxeles
        cells_per_block: Número de celdas por bloque
        visualize: Si True, retorna también la visualización
    
    Returns:
        Características HOG (y visualización si visualize=True)
    """
    result = hog(
        img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=visualize,
        feature_vector=True
    )
    
    if visualize:
        return result[0], result[1]  # features, hog_image
    else:
        return result  # solo features


def calculate_hu_moments(img):
    """
    Calcula los 7 momentos invariantes de Hu.
    
    Los momentos de Hu son descriptores de forma que poseen tres invarianzas fundamentales:
    1. **Traslación**: No dependen de la posición del objeto en la imagen
    2. **Rotación**: No cambian cuando el objeto se rota
    3. **Escala**: No dependen del tamaño del objeto (están normalizados)
    
    Propiedades de cada momento:
    - Hu₁: Dispersión espacial (relacionado con varianza)
    - Hu₂: Asimetría y elongación
    - Hu₃: Asimetría (skewness)
    - Hu₄: Kurtosis (curtosis)
    - Hu₅-Hu₇: Invariantes adicionales para formas complejas
    
    Nota: Los momentos Hu₁-Hu₄ son los más estables y utilizados.
          Los momentos Hu₅-Hu₇ pueden ser muy pequeños y sensibles al ruido.
    
    Args:
        img: Imagen binaria o en escala de grises
    
    Returns:
        Array con los 7 momentos de Hu (aplicada transformación logarítmica)
    """
    # Calcular momentos
    moments = cv2.moments(img)
    
    # Calcular momentos de Hu
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Aplicar transformación logarítmica para hacer los valores más manejables
    # (los momentos de Hu pueden ser muy pequeños)
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments


def extract_contour_features(img, threshold_method='otsu'):
    """
    Extrae descriptores de contorno de la imagen.
    Requiere segmentación previa.
    
    Args:
        img: Imagen en escala de grises
        threshold_method: Método de umbralización ('otsu' o 'adaptive')
    
    Returns:
        Diccionario con descriptores de contorno
    """
    # Umbralización
    if threshold_method == 'otsu':
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {
            'area': 0,
            'perimeter': 0,
            'circularity': 0,
            'eccentricity': 0,
            'solidity': 0
        }
    
    # Tomar el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calcular descriptores
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Circularidad: 4π*área/perímetro² (1 = círculo perfecto)
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-10)
    
    # Excentricidad (usando elipse ajustada)
    if len(largest_contour) >= 5:
        ellipse = cv2.fitEllipse(largest_contour)
        a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
        if a > b:
            eccentricity = np.sqrt(1 - (b**2 / a**2))
        else:
            eccentricity = np.sqrt(1 - (a**2 / b**2))
    else:
        eccentricity = 0
    
    # Solidez (área del contorno / área del casco convexo)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-10)
    
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'eccentricity': eccentricity,
        'solidity': solidity
    }


def extract_fourier_descriptors(img, n_coefficients=20):
    """
    Extrae descriptores de Fourier del contorno.
    Representa el contorno en el dominio de la frecuencia.
    
    Args:
        img: Imagen en escala de grises
        n_coefficients: Número de coeficientes a retornar
    
    Returns:
        Array con los primeros N coeficientes de Fourier
    """
    # Umbralizar y encontrar contorno
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros(n_coefficients, dtype=complex)
    
    # Tomar el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Convertir contorno a array complejo (x + iy)
    contour_complex = largest_contour[:, 0, 0] + 1j * largest_contour[:, 0, 1]
    
    # Aplicar FFT
    fft_result = np.fft.fft(contour_complex)
    
    # Normalizar por el primer coeficiente (DC component) para hacerlo invariante a escala
    if fft_result[0] != 0:
        fft_result = fft_result / fft_result[0]
    
    # Tomar los primeros N coeficientes (excluyendo el DC)
    descriptors = fft_result[1:n_coefficients+1]
    
    # Retornar magnitud y fase (o solo magnitud para invarianza a rotación)
    return np.abs(descriptors)  # Solo magnitud para invarianza a rotación


# ============================================================================
# Parte B: Descriptores de Textura
# ============================================================================

def extract_lbp_features(img, radius=3, n_points=24, method='uniform'):
    """
    Extrae características LBP (Local Binary Patterns).
    
    Args:
        img: Imagen en escala de grises
        radius: Radio del patrón LBP
        n_points: Número de puntos vecinos
        method: Método de LBP ('default', 'ror', 'uniform', 'var')
    
    Returns:
        Tupla (histograma de patrones LBP, imagen LBP)
    """
    # Calcular LBP
    lbp = local_binary_pattern(img, n_points, radius, method=method)
    
    # Calcular histograma
    n_bins = n_points + 2  # Para uniform patterns
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist, lbp


def extract_glcm_features(img, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extrae características GLCM (Gray Level Co-occurrence Matrix).
    
    Args:
        img: Imagen en escala de grises
        distances: Lista de distancias para calcular GLCM
        angles: Lista de ángulos en radianes
    
    Returns:
        Diccionario con propiedades GLCM
    """
    # Convertir a enteros de 0-255 si es necesario
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    # Calcular GLCM
    glcm = graycomatrix(img, distances=distances, angles=angles, 
                       levels=256, symmetric=True, normed=True)
    
    # Calcular propiedades
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = {}
    
    for prop in properties:
        values = graycoprops(glcm, prop)
        # Promediar sobre todas las distancias y ángulos
        features[prop] = np.mean(values)
    
    return features


def extract_gabor_features(img, frequencies=[0.1, 0.3, 0.5], 
                          orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extrae características usando filtros de Gabor.
    Crea un banco de filtros con diferentes frecuencias y orientaciones.
    
    Args:
        img: Imagen en escala de grises
        frequencies: Lista de frecuencias
        orientations: Lista de orientaciones en radianes
    
    Returns:
        Array con estadísticas de respuesta (media, std) para cada filtro
    """
    features = []
    
    for frequency in frequencies:
        for theta in orientations:
            # Aplicar filtro de Gabor
            real, imag = gabor(img, frequency=frequency, theta=theta)
            
            # Calcular magnitud de la respuesta
            magnitude = np.sqrt(real**2 + imag**2)
            
            # Estadísticas de respuesta
            features.append(np.mean(magnitude))
            features.append(np.std(magnitude))
    
    return np.array(features)


def extract_first_order_statistics(img):
    """
    Extrae estadísticas de primer orden de la imagen.
    
    Args:
        img: Imagen en escala de grises
    
    Returns:
        Diccionario con estadísticas
    """
    # Asegurar que la imagen está en el rango correcto
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    # Calcular estadísticas
    mean = np.mean(img)
    variance = np.var(img)
    std = np.std(img)
    
    # Skewness (asimetría)
    skewness = np.mean(((img - mean) / (std + 1e-10)) ** 3)
    
    # Kurtosis (curtosis)
    kurtosis = np.mean(((img - mean) / (std + 1e-10)) ** 4) - 3
    
    # Entropía
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)  # Normalizar
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    return {
        'mean': mean,
        'variance': variance,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'entropy': entropy
    }


# ============================================================================
# Función de Extracción Completa
# ============================================================================

def extract_all_features(img_path, target_size=(224, 224)):
    """
    Extrae todos los descriptores de una imagen.
    
    Args:
        img_path: Ruta a la imagen
        target_size: Tamaño objetivo para preprocesamiento
    
    Returns:
        Diccionario con todos los descriptores
    """
    # Cargar y preprocesar imagen
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(img, target_size)
    img_clahe = apply_clahe(img)
    
    features = {}
    
    # Descriptores de forma
    # 1. HOG
    hog_feat = extract_hog_features(img_clahe)
    features['hog'] = hog_feat
    
    # 2. Momentos de Hu
    hu_moments = calculate_hu_moments(img_clahe)
    features['hu_moments'] = hu_moments
    
    # 3. Descriptores de contorno
    contour_feat = extract_contour_features(img_clahe)
    features.update({f'contour_{k}': v for k, v in contour_feat.items()})
    
    # 4. Descriptores de Fourier
    fourier_feat = extract_fourier_descriptors(img_clahe, n_coefficients=20)
    features['fourier'] = fourier_feat
    
    # Descriptores de textura
    # 5. LBP
    lbp_hist, _ = extract_lbp_features(img_clahe, radius=3, n_points=24)
    features['lbp'] = lbp_hist
    
    # 6. GLCM
    glcm_feat = extract_glcm_features(img_clahe)
    features.update({f'glcm_{k}': v for k, v in glcm_feat.items()})
    
    # 7. Gabor
    gabor_feat = extract_gabor_features(img_clahe)
    features['gabor'] = gabor_feat
    
    # 8. Estadísticas de primer orden
    first_order = extract_first_order_statistics(img_clahe)
    features.update({f'first_order_{k}': v for k, v in first_order.items()})
    
    return features


# ============================================================================
# Extracción de Características para Todo el Dataset
# ============================================================================

def extract_features_batch(df, sample_size=None, save_path=None):
    """
    Extrae características para un lote de imágenes.
    
    Args:
        df: DataFrame con rutas de imágenes
        sample_size: Si se especifica, solo procesa una muestra
        save_path: Ruta para guardar las características
    
    Returns:
        DataFrame con características y etiquetas
    """
    # Tomar muestra si se especifica
    if sample_size and sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"📝 Procesando muestra de {sample_size} imágenes...")
    else:
        df_sample = df.copy()
        print(f"📝 Procesando {len(df_sample)} imágenes...")
    
    features_list = []
    labels = []
    paths = []
    
    for idx, row in df_sample.iterrows():
        try:
            feat = extract_all_features(row['path'])
            if feat is not None:
                # Convertir diccionario a array plano
                feature_vector = []
                for key in sorted(feat.keys()):
                    value = feat[key]
                    if isinstance(value, np.ndarray):
                        feature_vector.extend(value.tolist())
                    else:
                        feature_vector.append(value)
                
                features_list.append(feature_vector)
                labels.append(row['class'])
                paths.append(row['path'])
                
                if (len(features_list) % 50 == 0):
                    print(f"   Procesadas: {len(features_list)}/{len(df_sample)}")
        except Exception as e:
            print(f"   Error procesando {row['path']}: {e}")
            continue
    
    # Crear DataFrame
    feature_df = pd.DataFrame(features_list)
    feature_df['label'] = labels
    feature_df['path'] = paths
    
    # Guardar si se especifica
    if save_path:
        feature_df.to_csv(save_path, index=False)
        print(f"\n✅ Características guardadas en: {save_path}")
    
    print(f"\n✅ Extracción completada: {len(feature_df)} imágenes procesadas")
    print(f"   Dimensiones: {feature_df.shape[1] - 2} características + 2 columnas (label, path)")
    
    return feature_df


# ============================================================================
# Funciones de Carga de Datos
# ============================================================================

def load_dataset(metadata_path='../data/metadata.csv', data_dir='../data/chest_xray'):
    """
    Carga el dataset desde metadata.csv o desde el directorio.
    
    Args:
        metadata_path: Ruta al archivo metadata.csv
        data_dir: Directorio base del dataset (si no existe metadata.csv)
    
    Returns:
        DataFrame con información del dataset
    """
    try:
        df = pd.read_csv(metadata_path)
        print(f"✅ Dataset cargado: {len(df)} imágenes")
    except:
        print("⚠️  No se encontró metadata.csv. Cargando desde directorio...")
        DATA_DIR = Path(data_dir)
        TRAIN_DIR = DATA_DIR / 'train'
        TEST_DIR = DATA_DIR / 'test'
        
        def load_image_paths(base_dir, class_name):
            class_dir = base_dir / class_name
            if class_dir.exists():
                return list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.jpg'))
            return []
        
        train_normal = load_image_paths(TRAIN_DIR, 'NORMAL')
        train_pneumonia = load_image_paths(TRAIN_DIR, 'PNEUMONIA')
        test_normal = load_image_paths(TEST_DIR, 'NORMAL')
        test_pneumonia = load_image_paths(TEST_DIR, 'PNEUMONIA')
        
        data = []
        for paths, split, label in [(train_normal, 'train', 'NORMAL'), 
                                    (train_pneumonia, 'train', 'PNEUMONIA'),
                                    (test_normal, 'test', 'NORMAL'), 
                                    (test_pneumonia, 'test', 'PNEUMONIA')]:
            for path in paths:
                data.append({'path': str(path), 'split': split, 'class': label})
        
        df = pd.DataFrame(data)
        print(f"✅ Dataset creado: {len(df)} imágenes")
    
    # Mostrar distribución
    print("\n📊 Distribución:")
    print(df.groupby(['split', 'class']).size())
    
    return df


# ============================================================================
# Funciones de Visualización
# ============================================================================

def visualize_hog(img_path, save_path=None):
    """Visualiza características HOG de una imagen"""
    sample_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    sample_img = cv2.resize(sample_img, (224, 224))
    
    hog_features, hog_image = extract_hog_features(sample_img, visualize=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(sample_img, cmap='gray')
    axes[0].set_title('Imagen Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(hog_image, cmap='hot')
    axes[1].set_title(f'HOG Visualization\n{len(hog_features)} características', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ HOG extraído: {len(hog_features)} características")
    print(f"   Parámetros: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)")


def visualize_hu_moments(img_path, save_path=None):
    """Visualiza los momentos de Hu de una imagen"""
    sample_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    sample_img = cv2.resize(sample_img, (224, 224))
    
    hu_moments = calculate_hu_moments(sample_img)
    
    print("✅ Momentos de Hu calculados:")
    print("   Los 7 momentos invariantes de Hu y sus invarianzas:")
    print("\n   📐 INVARIANZAS GENERALES (todos los momentos):")
    print("      • Traslación: No dependen de la posición del objeto en la imagen")
    print("      • Rotación: No cambian al rotar el objeto")
    print("      • Escala: No dependen del tamaño del objeto (normalizados)")
    print("\n   🔍 PROPIEDADES ESPECÍFICAS DE CADA MOMENTO:")
    print("   1. Hu₁: Medida de dispersión espacial (relacionado con varianza)")
    print("   2. Hu₂: Medida de asimetría y elongación")
    print("   3. Hu₃: Medida de asimetría (skewness)")
    print("   4. Hu₄: Medida de kurtosis (curtosis)")
    print("   5. Hu₅: Invariante adicional para formas complejas")
    print("   6. Hu₆: Invariante adicional para formas complejas")
    print("   7. Hu₇: Invariante adicional para formas complejas (más robusto)")
    print("\n   💡 NOTA: Los momentos Hu₁-Hu₄ son los más utilizados y estables.")
    print("      Los momentos Hu₅-Hu₇ pueden ser muy pequeños y sensibles al ruido.")
    print(f"\n   Valores (transformación logarítmica): {hu_moments}")
    
    # Visualizar distribución de momentos
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(range(1, 8), hu_moments, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Momento de Hu', fontsize=12)
    ax.set_ylabel('Valor (log)', fontsize=12)
    ax.set_title('Momentos Invariantes de Hu', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 8))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_contours(img_path, save_path=None):
    """Visualiza los contornos detectados en una imagen"""
    sample_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    sample_img = cv2.resize(sample_img, (224, 224))
    
    contour_features = extract_contour_features(sample_img)
    
    print("✅ Descriptores de contorno extraídos:")
    for key, value in contour_features.items():
        print(f"   {key}: {value:.4f}")
    
    # Visualizar contornos
    _, binary = cv2.threshold(sample_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(sample_img, cmap='gray')
    axes[0].set_title('Imagen Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    img_with_contours = sample_img.copy()
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(img_with_contours, [largest], -1, 255, 2)
    axes[1].imshow(img_with_contours, cmap='gray')
    axes[1].set_title('Contorno Detectado', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_fourier_descriptors(img_path, n_coefficients=20, save_path=None):
    """Visualiza los descriptores de Fourier de una imagen"""
    sample_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    sample_img = cv2.resize(sample_img, (224, 224))
    
    fourier_descriptors = extract_fourier_descriptors(sample_img, n_coefficients=n_coefficients)
    
    print(f"✅ Descriptores de Fourier extraídos: {len(fourier_descriptors)} coeficientes")
    print(f"   Primeros 5 valores: {fourier_descriptors[:5]}")
    
    # Visualizar
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(range(1, len(fourier_descriptors) + 1), fourier_descriptors, 
            marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Coeficiente de Fourier', fontsize=12)
    ax.set_ylabel('Magnitud', fontsize=12)
    ax.set_title('Descriptores de Fourier (Magnitud)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_lbp(img_path, save_path=None):
    """Visualiza características LBP de una imagen"""
    sample_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    sample_img = cv2.resize(sample_img, (224, 224))
    
    lbp_hist_1, lbp_img_1 = extract_lbp_features(sample_img, radius=1, n_points=8)
    lbp_hist_2, lbp_img_2 = extract_lbp_features(sample_img, radius=3, n_points=24)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].imshow(sample_img, cmap='gray')
    axes[0, 0].set_title('Imagen Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(lbp_img_1, cmap='gray')
    axes[0, 1].set_title(f'LBP (r=1, n=8)\n{len(lbp_hist_1)} bins', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(lbp_img_2, cmap='gray')
    axes[1, 0].set_title(f'LBP (r=3, n=24)\n{len(lbp_hist_2)} bins', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].bar(range(len(lbp_hist_2)), lbp_hist_2, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Histograma LBP (r=3, n=24)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Patrón LBP')
    axes[1, 1].set_ylabel('Frecuencia Normalizada')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ LBP extraído: {len(lbp_hist_2)} características (r=3, n=24)")


def visualize_glcm(img_path, save_path=None):
    """Visualiza características GLCM de una imagen"""
    sample_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    sample_img = cv2.resize(sample_img, (224, 224))
    
    glcm_features = extract_glcm_features(sample_img)
    
    print("✅ Características GLCM extraídas:")
    for key, value in glcm_features.items():
        print(f"   {key}: {value:.4f}")
    
    # Visualizar GLCM
    img_uint8 = (sample_img).astype(np.uint8) if sample_img.max() <= 1.0 else sample_img.astype(np.uint8)
    glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(sample_img, cmap='gray')
    axes[0].set_title('Imagen Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Visualizar GLCM (solo una distancia y ángulo para visualización)
    axes[1].imshow(glcm[:, :, 0, 0], cmap='hot', interpolation='nearest')
    axes[1].set_title('GLCM (distancia=1, ángulo=0°)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Nivel de gris j')
    axes[1].set_ylabel('Nivel de gris i')
    plt.colorbar(axes[1].images[0], ax=axes[1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_gabor(img_path, save_path=None):
    """Visualiza características de Gabor de una imagen"""
    sample_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    sample_img = cv2.resize(sample_img, (224, 224))
    
    frequencies = [0.1, 0.3, 0.5]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    gabor_features = extract_gabor_features(sample_img)
    
    print(f"✅ Características de Gabor extraídas: {len(gabor_features)} características")
    print(f"   (2 estadísticas × {len(frequencies)} frecuencias × {len(orientations)} orientaciones)")
    
    # Visualizar respuestas de filtros de Gabor
    fig, axes = plt.subplots(len(frequencies), len(orientations), figsize=(16, 12))
    
    for i, freq in enumerate(frequencies):
        for j, theta in enumerate(orientations):
            real, imag = gabor(sample_img, frequency=freq, theta=theta)
            magnitude = np.sqrt(real**2 + imag**2)
            
            axes[i, j].imshow(magnitude, cmap='hot')
            axes[i, j].set_title(f'f={freq:.1f}, θ={theta*180/np.pi:.0f}°', 
                               fontsize=10, fontweight='bold')
            axes[i, j].axis('off')
    
    plt.suptitle('Respuestas de Filtros de Gabor', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_first_order_stats(img_path, save_path=None):
    """Visualiza estadísticas de primer orden de una imagen"""
    sample_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    sample_img = cv2.resize(sample_img, (224, 224))
    
    first_order_stats = extract_first_order_statistics(sample_img)
    
    print("✅ Estadísticas de primer orden extraídas:")
    for key, value in first_order_stats.items():
        print(f"   {key}: {value:.4f}")
    
    # Visualizar histograma con estadísticas
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    hist, bins = np.histogram(sample_img.flatten(), bins=256, range=(0, 256))
    axes[0].bar(bins[:-1], hist, color='steelblue', alpha=0.7, edgecolor='black', width=1)
    axes[0].axvline(first_order_stats['mean'], color='red', linestyle='--', 
                   label=f"Media: {first_order_stats['mean']:.2f}", linewidth=2)
    axes[0].axvline(first_order_stats['mean'] + first_order_stats['std'], 
                   color='orange', linestyle='--', label=f"±1 std", linewidth=1)
    axes[0].axvline(first_order_stats['mean'] - first_order_stats['std'], 
                   color='orange', linestyle='--', linewidth=1)
    axes[0].set_title('Histograma de Intensidades', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Intensidad de píxel')
    axes[0].set_ylabel('Frecuencia')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Gráfico de barras de estadísticas
    stats_names = list(first_order_stats.keys())
    stats_values = list(first_order_stats.values())
    axes[1].bar(stats_names, stats_values, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_title('Estadísticas de Primer Orden', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Valor')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# Función Principal
# ============================================================================

def main(metadata_path='../data/metadata.csv', sample_size=100, 
         save_features_path='../data/features_sample.csv'):
    """
    Ejecuta el pipeline completo de extracción de descriptores.
    
    Args:
        metadata_path: Ruta al archivo metadata.csv
        sample_size: Tamaño de muestra para procesar (None para todo el dataset)
        save_features_path: Ruta para guardar las características extraídas
    """
    # Cargar dataset
    df = load_dataset(metadata_path)
    
    # Extraer características
    print("\n🚀 Iniciando extracción de características...")
    if sample_size:
        print(f"   (Usando muestra de {sample_size} imágenes para prueba rápida)")
        print("   Para procesar todo el dataset, cambiar sample_size=None\n")
    else:
        print("   Procesando todo el dataset...\n")
    
    features_df = extract_features_batch(df, sample_size=sample_size, 
                                        save_path=save_features_path)
    
    # Mostrar resumen
    print("\n📊 Resumen del dataset de características:")
    print(features_df.head())
    print(f"\nDistribución de clases:")
    print(features_df['label'].value_counts())
    
    return features_df


if __name__ == '__main__':
    # Ejecutar pipeline completo
    features_df = main()

