"""
Parte 3: Clasificación con Descriptores Clásicos

Este módulo implementa la clasificación de imágenes usando los descriptores extraídos 
y diferentes algoritmos de machine learning.
"""

# Importación de librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, auc, roc_auc_score)

# PyTorch para CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 1. Carga y Preparación de Datos
# ============================================================================

def load_features(features_path='../data/features_sample.csv'):
    """
    Carga las características extraídas.
    
    Args:
        features_path: Ruta al archivo CSV con características
    
    Returns:
        tuple: (X, y, y_encoded, label_encoder, features_df)
    """
    try:
        features_df = pd.read_csv(features_path)
        print(f"✅ Características cargadas: {features_df.shape}")
    except:
        print("⚠️  No se encontró features_sample.csv")
        print("   Por favor, ejecuta primero el módulo de extracción de descriptores")
        raise
    
    # Separar características y etiquetas
    X = features_df.drop(['label', 'path'], axis=1).values
    y = features_df['label'].values
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\n📊 Dataset:")
    print(f"   Muestras: {X.shape[0]}")
    print(f"   Características: {X.shape[1]}")
    print(f"   Clases: {le.classes_}")
    print(f"   Distribución: {np.bincount(y_encoded)}")
    
    return X, y, y_encoded, le, features_df


def clean_data(X_train, X_test):
    """
    Limpia los datos detectando y manejando valores infinitos y NaN.
    
    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de prueba
    
    Returns:
        tuple: (X_train_clean, X_test_clean)
    """
    print("\n🔍 Verificando calidad de datos...")
    
    # Contar valores infinitos y NaN
    inf_count_train = np.isinf(X_train).sum()
    nan_count_train = np.isnan(X_train).sum()
    inf_count_test = np.isinf(X_test).sum()
    nan_count_test = np.isnan(X_test).sum()
    
    print(f"   Valores infinitos - Train: {inf_count_train}, Test: {inf_count_test}")
    print(f"   Valores NaN - Train: {nan_count_train}, Test: {nan_count_test}")
    
    # Reemplazar infinitos y NaN
    if inf_count_train > 0 or nan_count_train > 0 or inf_count_test > 0 or nan_count_test > 0:
        print("   ⚠️  Limpiando valores problemáticos...")
        
        # Reemplazar infinitos con NaN primero
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_test = np.where(np.isinf(X_test), np.nan, X_test)
        
        # Calcular mediana por columna (ignorando NaN) para reemplazar
        valid_cols = ~np.isnan(X_train).all(axis=0)
        if valid_cols.any():
            medians = np.nanmedian(X_train[:, valid_cols], axis=0)
            
            # Reemplazar NaN con la mediana de cada columna
            for i, col_idx in enumerate(np.where(valid_cols)[0]):
                col_median = medians[i]
                if not np.isnan(col_median) and not np.isinf(col_median):
                    X_train[:, col_idx] = np.where(np.isnan(X_train[:, col_idx]), col_median, X_train[:, col_idx])
                    X_test[:, col_idx] = np.where(np.isnan(X_test[:, col_idx]), col_median, X_test[:, col_idx])
        
        # Para columnas que son todas NaN o que aún tienen NaN, reemplazar con 0
        nan_cols = np.isnan(X_train).any(axis=0)
        if nan_cols.any():
            X_train[:, nan_cols] = np.nan_to_num(X_train[:, nan_cols], nan=0.0, posinf=0.0, neginf=0.0)
            X_test[:, nan_cols] = np.nan_to_num(X_test[:, nan_cols], nan=0.0, posinf=0.0, neginf=0.0)
            print(f"   ⚠️  {nan_cols.sum()} columnas con NaN fueron reemplazadas con 0")
        
        # Verificar que no queden valores problemáticos
        remaining_inf = np.isinf(X_train).sum() + np.isinf(X_test).sum()
        remaining_nan = np.isnan(X_train).sum() + np.isnan(X_test).sum()
        
        if remaining_inf == 0 and remaining_nan == 0:
            print("   ✅ Datos limpiados exitosamente")
        else:
            print(f"   ⚠️  Aún quedan {remaining_inf} infinitos y {remaining_nan} NaN")
            # Última medida: reemplazar cualquier valor problemático restante con 0
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            print("   ✅ Valores problemáticos reemplazados con 0")
    else:
        print("   ✅ No se encontraron valores problemáticos")
    
    return X_train, X_test


def prepare_data(X, y_encoded, test_size=0.2, random_state=42):
    """
    Divide los datos y normaliza.
    
    Args:
        X: Características
        y_encoded: Etiquetas codificadas
        test_size: Proporción de datos de prueba
        random_state: Semilla aleatoria
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    print(f"📊 División de datos:")
    print(f"   Train: {X_train.shape[0]} muestras")
    print(f"   Test: {X_test.shape[0]} muestras")
    
    # Limpiar datos
    X_train, X_test = clean_data(X_train, X_test)
    
    # Normalización
    print("\n📊 Normalizando datos...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✅ Datos normalizados:")
    print(f"   StandardScaler: media={X_train_scaled.mean():.4f}, std={X_train_scaled.std():.4f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ============================================================================
# 2. Reducción de Dimensionalidad
# ============================================================================

def analyze_pca(X_train_scaled, n_components_max=50, variance_threshold=0.95, save_path=None):
    """
    Analiza PCA y encuentra el número óptimo de componentes.
    
    Args:
        X_train_scaled: Datos de entrenamiento normalizados
        n_components_max: Máximo número de componentes a considerar
        variance_threshold: Umbral de varianza explicada
        save_path: Ruta para guardar la visualización
    
    Returns:
        tuple: (pca, X_train_pca, n_components)
    """
    # PCA - Análisis de varianza explicada
    pca_full = PCA()
    pca_full.fit(X_train_scaled)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Encontrar número de componentes para umbral de varianza
    n_components_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Varianza explicada
    n_plot = min(n_components_max, len(cumulative_variance))
    axes[0].plot(range(1, n_plot + 1), 
                cumulative_variance[:n_plot], marker='o', linewidth=2)
    axes[0].axhline(variance_threshold, color='r', linestyle='--', 
                   label=f'{variance_threshold*100}% varianza')
    axes[0].axvline(n_components_threshold, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Número de Componentes', fontsize=12)
    axes[0].set_ylabel('Varianza Explicada Acumulada', fontsize=12)
    axes[0].set_title('Análisis PCA - Varianza Explicada', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Varianza por componente
    n_plot_components = min(20, len(pca_full.explained_variance_ratio_))
    axes[1].bar(range(1, n_plot_components + 1),
               pca_full.explained_variance_ratio_[:n_plot_components], 
               alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Componente Principal', fontsize=12)
    axes[1].set_ylabel('Varianza Explicada', fontsize=12)
    axes[1].set_title('Varianza por Componente (Top 20)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Análisis PCA:")
    print(f"   Componentes para {variance_threshold*100}% varianza: {n_components_threshold}")
    print(f"   Reducción: {X_train_scaled.shape[1]} → {n_components_threshold} características")
    
    # Aplicar PCA
    n_components = min(n_components_threshold, n_components_max)
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    print(f"   Aplicado PCA con {n_components} componentes")
    
    return pca, X_train_pca, n_components


def apply_pca(pca, X_train_scaled, X_test_scaled):
    """
    Aplica PCA a los datos de entrenamiento y prueba.
    
    Args:
        pca: Objeto PCA entrenado
        X_train_scaled: Datos de entrenamiento normalizados
        X_test_scaled: Datos de prueba normalizados
    
    Returns:
        tuple: (X_train_pca, X_test_pca)
    """
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca


# ============================================================================
# 3. Evaluación de Clasificadores
# ============================================================================

def evaluate_classifier(clf, X_train, X_test, y_train, y_test, 
                        classifier_name="Classifier", cv=5):
    """
    Evalúa un clasificador con múltiples métricas.
    
    Args:
        clf: Clasificador a evaluar
        X_train: Datos de entrenamiento
        X_test: Datos de prueba
        y_train: Etiquetas de entrenamiento
        y_test: Etiquetas de prueba
        classifier_name: Nombre del clasificador
        cv: Número de folds para validación cruzada
    
    Returns:
        dict: Diccionario con resultados de evaluación
    """
    # Entrenar
    clf.fit(X_train, y_train)
    
    # Predicciones
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Validación cruzada
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC (si hay probabilidades)
    roc_auc = None
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results = {
        'classifier': classifier_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results


def plot_confusion_matrix(cm, class_names, title="Matriz de Confusión", save_path=None):
    """
    Visualiza matriz de confusión.
    
    Args:
        cm: Matriz de confusión
        class_names: Nombres de las clases
        title: Título del gráfico
        save_path: Ruta para guardar la imagen
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# 4. Clasificadores Específicos
# ============================================================================

def train_svm_models(X_train, X_test, y_train, y_test, save_paths=None):
    """
    Entrena modelos SVM con diferentes kernels.
    
    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de prueba
        y_train: Etiquetas de entrenamiento
        y_test: Etiquetas de prueba
        save_paths: Diccionario con rutas para guardar matrices de confusión
    
    Returns:
        dict: Diccionario con resultados de cada modelo SVM
    """
    svm_models = {
        'SVM Linear': SVC(kernel='linear', probability=True, random_state=42),
        'SVM RBF': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM Polynomial': SVC(kernel='poly', degree=3, probability=True, random_state=42)
    }
    
    svm_results = {}
    
    for name, model in svm_models.items():
        print(f"\n🔄 Entrenando {name}...")
        result = evaluate_classifier(model, X_train, X_test, 
                                     y_train, y_test, classifier_name=name)
        svm_results[name] = result
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   F1-Score: {result['f1_score']:.4f}")
        print(f"   CV Accuracy: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
        
        # Guardar matriz de confusión si se especifica
        if save_paths and name in save_paths:
            plot_confusion_matrix(result['confusion_matrix'], ['NORMAL', 'PNEUMONIA'], 
                                 title=f'{name}\nAcc: {result["accuracy"]:.3f}',
                                 save_path=save_paths[name])
    
    return svm_results


def train_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, 
                       save_paths=None):
    """
    Entrena un modelo Random Forest.
    
    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de prueba
        y_train: Etiquetas de entrenamiento
        y_test: Etiquetas de prueba
        n_estimators: Número de árboles
        save_paths: Diccionario con rutas para guardar visualizaciones
    
    Returns:
        dict: Resultados del modelo Random Forest
    """
    print("🔄 Entrenando Random Forest...")
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf_result = evaluate_classifier(rf, X_train, X_test, 
                                   y_train, y_test, classifier_name='Random Forest')
    
    print(f"   Accuracy: {rf_result['accuracy']:.4f}")
    print(f"   F1-Score: {rf_result['f1_score']:.4f}")
    print(f"   CV Accuracy: {rf_result['cv_mean']:.4f} (±{rf_result['cv_std']:.4f})")
    
    # Importancia de características
    feature_importance = rf.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    
    if save_paths:
        # Matriz de confusión
        if 'cm' in save_paths:
            plot_confusion_matrix(rf_result['confusion_matrix'], ['NORMAL', 'PNEUMONIA'], 
                                 title=f'Random Forest\nAcc: {rf_result["accuracy"]:.3f}',
                                 save_path=save_paths['cm'])
        
        # Importancia de características
        if 'importance' in save_paths:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.barh(range(len(top_features_idx)), feature_importance[top_features_idx], 
                    color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(top_features_idx)))
            ax.set_yticklabels([f'Feature {i}' for i in top_features_idx])
            ax.set_xlabel('Importancia', fontsize=12)
            ax.set_title('Top 10 Características Más Importantes', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(save_paths['importance'], dpi=150, bbox_inches='tight')
            plt.show()
    
    return rf_result


def train_knn_models(X_train, X_test, y_train, y_test, k_values=[3, 5, 7, 9, 11],
                     save_paths=None):
    """
    Entrena modelos k-NN con diferentes valores de k.
    
    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de prueba
        y_train: Etiquetas de entrenamiento
        y_test: Etiquetas de prueba
        k_values: Lista de valores de k a probar
        save_paths: Diccionario con rutas para guardar visualizaciones
    
    Returns:
        tuple: (knn_results, best_knn_result)
    """
    knn_results = {}
    
    for k in k_values:
        print(f"\n🔄 Entrenando k-NN (k={k})...")
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        result = evaluate_classifier(knn, X_train, X_test, 
                                    y_train, y_test, classifier_name=f'k-NN (k={k})')
        knn_results[k] = result
        print(f"   Accuracy: {result['accuracy']:.4f}")
    
    # Encontrar mejor k
    k_accuracies = [knn_results[k]['accuracy'] for k in k_values]
    best_k = k_values[np.argmax(k_accuracies)]
    best_knn_result = knn_results[best_k]
    
    if save_paths:
        # Visualizar efecto de k
        if 'k_effect' in save_paths:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(k_values, k_accuracies, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('k (Número de Vecinos)', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('k-NN: Efecto del Parámetro k', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(k_values)
            plt.tight_layout()
            plt.savefig(save_paths['k_effect'], dpi=150, bbox_inches='tight')
            plt.show()
        
        # Matriz de confusión del mejor modelo
        if 'cm' in save_paths:
            plot_confusion_matrix(best_knn_result['confusion_matrix'], ['NORMAL', 'PNEUMONIA'],
                                 title=f'k-NN (k={best_k})\nAcc: {best_knn_result["accuracy"]:.3f}',
                                 save_path=save_paths['cm'])
    
    print(f"\n✅ Mejor k: {best_k} con accuracy: {best_knn_result['accuracy']:.4f}")
    
    return knn_results, best_knn_result


def train_logistic_regression(X_train, X_test, y_train, y_test, max_iter=1000,
                              save_path=None):
    """
    Entrena un modelo de Regresión Logística.
    
    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de prueba
        y_train: Etiquetas de entrenamiento
        y_test: Etiquetas de prueba
        max_iter: Número máximo de iteraciones
        save_path: Ruta para guardar la matriz de confusión
    
    Returns:
        dict: Resultados del modelo
    """
    print("🔄 Entrenando Logistic Regression...")
    lr = LogisticRegression(max_iter=max_iter, random_state=42, n_jobs=-1)
    lr_result = evaluate_classifier(lr, X_train, X_test, 
                                    y_train, y_test, classifier_name='Logistic Regression')
    
    print(f"   Accuracy: {lr_result['accuracy']:.4f}")
    print(f"   F1-Score: {lr_result['f1_score']:.4f}")
    print(f"   CV Accuracy: {lr_result['cv_mean']:.4f} (±{lr_result['cv_std']:.4f})")
    
    if save_path:
        plot_confusion_matrix(lr_result['confusion_matrix'], ['NORMAL', 'PNEUMONIA'],
                             title=f'Logistic Regression\nAcc: {lr_result["accuracy"]:.3f}',
                             save_path=save_path)
    
    return lr_result


# ============================================================================
# 5. Comparación y Visualización
# ============================================================================

def compare_models(all_results, save_path=None):
    """
    Compara múltiples modelos y crea visualizaciones.
    
    Args:
        all_results: Lista de diccionarios con resultados de modelos
        save_path: Ruta para guardar la visualización
    
    Returns:
        pd.DataFrame: DataFrame con comparación de modelos
    """
    # Crear DataFrame comparativo
    comparison_df = pd.DataFrame({
        'Classifier': [r['classifier'] for r in all_results],
        'Accuracy': [r['accuracy'] for r in all_results],
        'Precision': [r['precision'] for r in all_results],
        'Recall': [r['recall'] for r in all_results],
        'F1-Score': [r['f1_score'] for r in all_results],
        'CV Mean': [r['cv_mean'] for r in all_results],
        'CV Std': [r['cv_std'] for r in all_results],
        'ROC AUC': [r['roc_auc'] if r['roc_auc'] is not None else np.nan for r in all_results]
    })
    
    print("\n📊 Comparación de Modelos:")
    print(comparison_df.to_string(index=False))
    
    # Visualizar comparación
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy
    axes[0, 0].barh(comparison_df['Classifier'], comparison_df['Accuracy'], 
                   color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Comparación de Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # F1-Score
    axes[0, 1].barh(comparison_df['Classifier'], comparison_df['F1-Score'], 
                   color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('F1-Score', fontsize=12)
    axes[0, 1].set_title('Comparación de F1-Score', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # CV Accuracy con barras de error
    axes[1, 0].errorbar(comparison_df['CV Mean'], comparison_df['Classifier'], 
                       xerr=comparison_df['CV Std'], fmt='o', capsize=5, capthick=2)
    axes[1, 0].set_xlabel('CV Accuracy', fontsize=12)
    axes[1, 0].set_title('Validación Cruzada (5-fold)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # ROC AUC (si está disponible)
    roc_available = comparison_df['ROC AUC'].notna()
    if roc_available.any():
        axes[1, 1].barh(comparison_df[roc_available]['Classifier'], 
                      comparison_df[roc_available]['ROC AUC'], 
                      color='green', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('ROC AUC', fontsize=12)
        axes[1, 1].set_title('Comparación de ROC AUC', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
    else:
        axes[1, 1].text(0.5, 0.5, 'ROC AUC no disponible\npara todos los modelos', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return comparison_df


def plot_roc_curves(all_results, y_test, save_path=None):
    """
    Grafica curvas ROC para modelos con predict_proba.
    
    Args:
        all_results: Lista de diccionarios con resultados de modelos
        y_test: Etiquetas de prueba
        save_path: Ruta para guardar la visualización
    """
    plt.figure(figsize=(10, 8))
    
    for result in all_results:
        if result['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f"{result['classifier']} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Clasificador Aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title('Curvas ROC - Comparación de Modelos', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# 6. CNN con PyTorch
# ============================================================================

class ChestXRayDataset(Dataset):
    """Dataset personalizado para imágenes de rayos X"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        
        # Cargar imagen
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)  # Convertir a RGB
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        
        # Aplicar transformaciones
        if self.transform:
            img = self.transform(torch.from_numpy(img))
        
        # Codificar etiqueta
        label_encoded = 1 if label == 'PNEUMONIA' else 0
        
        return img, label_encoded


class SimpleCNN(nn.Module):
    """CNN simple para clasificación de rayos X"""
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_cnn(features_df, metadata_path='../data/metadata.csv', 
             num_epochs=3, batch_size=32, learning_rate=0.001,
             save_path=None):
    """
    Entrena una CNN para clasificación.
    
    Args:
        features_df: DataFrame con características (para obtener paths)
        metadata_path: Ruta al archivo metadata.csv
        num_epochs: Número de épocas
        batch_size: Tamaño del batch
        learning_rate: Tasa de aprendizaje
        save_path: Ruta para guardar la matriz de confusión
    
    Returns:
        dict: Resultados de la CNN
    """
    print("🔄 Entrenando CNN...")
    print("   (Nota: Esto es una versión simplificada. Para entrenamiento completo,")
    print("    ajustar epochs, learning rate, y usar GPU si está disponible)")
    
    # Preparar datos para CNN
    try:
        metadata_df = pd.read_csv(metadata_path)
        train_df_cnn = features_df[features_df['path'].isin(metadata_df['path'])].sample(
            frac=0.8, random_state=42)
        test_df_cnn = features_df[~features_df.index.isin(train_df_cnn.index)]
    except:
        # Si no hay metadata, usar división simple
        train_df_cnn = features_df.sample(frac=0.8, random_state=42)
        test_df_cnn = features_df[~features_df.index.isin(train_df_cnn.index)]
    
    train_dataset = ChestXRayDataset(train_df_cnn)
    test_dataset = ChestXRayDataset(test_df_cnn)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Datasets CNN creados:")
    print(f"   Train: {len(train_dataset)} muestras")
    print(f"   Test: {len(test_dataset)} muestras")
    
    # Modelo
    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Entrenamiento
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluar CNN
    model.eval()
    y_pred_cnn = []
    y_true_cnn = []
    y_pred_proba_cnn = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            
            y_pred_cnn.extend(predicted.cpu().numpy())
            y_true_cnn.extend(labels.cpu().numpy())
            y_pred_proba_cnn.extend(probabilities[:, 1].cpu().numpy())
    
    y_pred_cnn = np.array(y_pred_cnn)
    y_true_cnn = np.array(y_true_cnn)
    y_pred_proba_cnn = np.array(y_pred_proba_cnn)
    
    # Calcular métricas CNN
    cnn_accuracy = accuracy_score(y_true_cnn, y_pred_cnn)
    cnn_precision = precision_score(y_true_cnn, y_pred_cnn, average='weighted', zero_division=0)
    cnn_recall = recall_score(y_true_cnn, y_pred_cnn, average='weighted', zero_division=0)
    cnn_f1 = f1_score(y_true_cnn, y_pred_cnn, average='weighted', zero_division=0)
    cnn_roc_auc = roc_auc_score(y_true_cnn, y_pred_proba_cnn)
    
    print(f"\n✅ CNN Evaluado:")
    print(f"   Accuracy: {cnn_accuracy:.4f}")
    print(f"   F1-Score: {cnn_f1:.4f}")
    print(f"   ROC AUC: {cnn_roc_auc:.4f}")
    
    # Visualizar resultados CNN
    cm_cnn = confusion_matrix(y_true_cnn, y_pred_cnn)
    if save_path:
        plot_confusion_matrix(cm_cnn, ['NORMAL', 'PNEUMONIA'],
                             title=f'CNN\nAcc: {cnn_accuracy:.3f}',
                             save_path=save_path)
    
    cnn_result = {
        'classifier': 'CNN (PyTorch)',
        'accuracy': cnn_accuracy,
        'precision': cnn_precision,
        'recall': cnn_recall,
        'f1_score': cnn_f1,
        'roc_auc': cnn_roc_auc,
        'confusion_matrix': cm_cnn,
        'y_pred': y_pred_cnn,
        'y_pred_proba': y_pred_proba_cnn
    }
    
    return cnn_result


# ============================================================================
# 7. Función Principal
# ============================================================================

def main(features_path='../data/features_sample.csv', 
         use_pca=False, n_components_max=50,
         train_cnn_flag=True, num_epochs=3,
         save_results=True):
    """
    Ejecuta el pipeline completo de clasificación.
    
    Args:
        features_path: Ruta al archivo CSV con características
        use_pca: Si True, usa PCA para reducción de dimensionalidad
        n_components_max: Máximo número de componentes PCA
        train_cnn_flag: Si True, entrena también la CNN
        num_epochs: Número de épocas para CNN
        save_results: Si True, guarda los resultados y visualizaciones
    """
    print("="*80)
    print("🚀 INICIANDO PIPELINE DE CLASIFICACIÓN")
    print("="*80)
    
    # 1. Cargar datos
    X, y, y_encoded, le, features_df = load_features(features_path)
    
    # 2. Preparar datos
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(X, y_encoded)
    
    # 3. Reducción de dimensionalidad (opcional)
    if use_pca:
        pca, X_train_pca, n_components = analyze_pca(
            X_train_scaled, n_components_max=n_components_max,
            save_path='../results/pca_analysis.png' if save_results else None
        )
        X_train_pca, X_test_pca = apply_pca(pca, X_train_scaled, X_test_scaled)
        X_train_final, X_test_final = X_train_pca, X_test_pca
    else:
        X_train_final, X_test_final = X_train_scaled, X_test_scaled
    
    # 4. Entrenar modelos clásicos
    print("\n" + "="*80)
    print("📊 ENTRENANDO MODELOS CLÁSICOS")
    print("="*80)
    
    # SVM
    svm_results = train_svm_models(
        X_train_final, X_test_final, y_train, y_test,
        save_paths={
            'SVM Linear': '../results/cm_svm_linear.png' if save_results else None,
            'SVM RBF': '../results/cm_svm_rbf.png' if save_results else None,
            'SVM Polynomial': '../results/cm_svm_polynomial.png' if save_results else None
        } if save_results else None
    )
    
    # Random Forest
    rf_result = train_random_forest(
        X_train_final, X_test_final, y_train, y_test,
        save_paths={
            'cm': '../results/cm_random_forest.png' if save_results else None,
            'importance': '../results/rf_importance.png' if save_results else None
        } if save_results else None
    )
    
    # k-NN
    knn_results, best_knn_result = train_knn_models(
        X_train_final, X_test_final, y_train, y_test,
        save_paths={
            'k_effect': '../results/knn_k_effect.png' if save_results else None,
            'cm': '../results/cm_knn.png' if save_results else None
        } if save_results else None
    )
    
    # Logistic Regression
    lr_result = train_logistic_regression(
        X_train_final, X_test_final, y_train, y_test,
        save_path='../results/cm_logistic_regression.png' if save_results else None
    )
    
    # 5. Comparar modelos clásicos
    all_classical_results = list(svm_results.values()) + [rf_result, best_knn_result, lr_result]
    comparison_df = compare_models(
        all_classical_results,
        save_path='../results/model_comparison.png' if save_results else None
    )
    
    # 6. Curvas ROC
    plot_roc_curves(
        all_classical_results, y_test,
        save_path='../results/roc_curves.png' if save_results else None
    )
    
    # 7. CNN (opcional)
    cnn_result = None
    if train_cnn_flag:
        print("\n" + "="*80)
        print("🧠 ENTRENANDO CNN")
        print("="*80)
        cnn_result = train_cnn(
            features_df, num_epochs=num_epochs,
            save_path='../results/cm_cnn.png' if save_results else None
        )
    
    # 8. Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL - COMPARACIÓN DE MÉTODOS")
    print("="*80)
    
    if cnn_result:
        final_results = all_classical_results + [cnn_result]
    else:
        final_results = all_classical_results
    
    final_comparison = pd.DataFrame({
        'Classifier': [r['classifier'] for r in final_results],
        'Accuracy': [r['accuracy'] for r in final_results],
        'Precision': [r['precision'] for r in final_results],
        'Recall': [r['recall'] for r in final_results],
        'F1-Score': [r['f1_score'] for r in final_results],
        'ROC AUC': [r['roc_auc'] if r['roc_auc'] is not None else np.nan 
                   for r in final_results]
    })
    
    print(final_comparison.to_string(index=False))
    print("="*80)
    
    if save_results:
        final_comparison.to_csv('../results/final_comparison.csv', index=False)
        print("\n✅ Resultados guardados en '../results/final_comparison.csv'")
        
        # Visualización final
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        x = np.arange(len(final_comparison))
        width = 0.2
        
        ax.bar(x - 1.5*width, final_comparison['Accuracy'], width, label='Accuracy', alpha=0.8)
        ax.bar(x - 0.5*width, final_comparison['F1-Score'], width, label='F1-Score', alpha=0.8)
        ax.bar(x + 0.5*width, final_comparison['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x + 1.5*width, final_comparison['Recall'], width, label='Recall', alpha=0.8)
        
        ax.set_xlabel('Clasificador', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comparación Final de Métodos de Clasificación', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(final_comparison['Classifier'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig('../results/final_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return final_comparison, final_results


if __name__ == '__main__':
    # Ejecutar pipeline completo
    print(f"🔧 Dispositivo: {device}")
    final_comparison, final_results = main()

