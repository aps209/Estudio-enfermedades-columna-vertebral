import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

# --- Modelos Estándar de Sklearn ---
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KernelDensity, NearestNeighbors
from ucimlrepo import fetch_ucirepo

# =============================================================================
# UTILIDADES: CÁLCULO DE MEAN IR Y BALANCEO (TAREA 2)
# =============================================================================

def compute_mean_ir(y):
    """
    [cite_start]Calcula el Mean Imbalance Ratio según la fórmula del PDF[cite: 17].
    """
    counts = Counter(y)
    max_count = max(counts.values())
    n_classes = len(counts)
    
    sum_ir = 0
    classes = list(counts.keys())
    
    for i in classes:
        # Suma del IR de cada clase respecto a la mayoritaria
        sum_ir += max_count / counts[i]
        
    return sum_ir / n_classes

def adjust_dataset_ir(X, y, target_threshold=1.15, mode='balance'):
    """
    Ajusta el dataset eliminando muestras para cumplir el umbral de MeanIR.
    mode 'balance': Busca MeanIR < 1.15 (elimina de la clase mayoritaria).
    mode 'imbalance': Busca MeanIR > 1.15 (elimina de clases aleatorias/minoritarias).
    """
    X_curr, y_curr = X.copy(), y.copy()
    rng = np.random.default_rng(42)
    
    # Límite de iteraciones para evitar bucles infinitos
    max_iter = 5000
    iter_count = 0
    
    while iter_count < max_iter:
        current_ir = compute_mean_ir(y_curr)
        counts = Counter(y_curr)
        
        # Condición de parada
        if mode == 'balance':
            if current_ir < target_threshold: break
            # Estrategia: Eliminar 1 muestra de la clase más poblada
            class_to_reduce = max(counts, key=counts.get)
            
        else: # mode == 'imbalance'
            if current_ir > target_threshold: break
            # Estrategia: Eliminar 1 muestra de una clase aleatoria que NO sea la mayoritaria
            # (para aumentar la diferencia)
            majority_class = max(counts, key=counts.get)
            candidates = [c for c in counts.keys() if c != majority_class and counts[c] > 5]
            if not candidates: break # No podemos desbalancear más sin vaciar clases
            class_to_reduce = rng.choice(candidates)

        # Encontrar índices de esa clase y eliminar uno al azar
        indices = np.where(y_curr == class_to_reduce)[0]
        idx_to_remove = rng.choice(indices)
        
        X_curr = np.delete(X_curr, idx_to_remove, axis=0)
        y_curr = np.delete(y_curr, idx_to_remove)
        iter_count += 1

    print(f"--> Ajuste ({mode}): MeanIR final = {compute_mean_ir(y_curr):.4f} | Muestras: {len(y_curr)}")
    return X_curr, y_curr

# =============================================================================
# CLASIFICADORES PERSONALIZADOS (TAREA 3)
# =============================================================================

class NaiveHistogramClassifier(BaseEstimator, ClassifierMixin):
    """
    Estimador de densidad por Histograma (Multivariante asumiendo independencia).
    """
    def __init__(self, bins=10):
        self.bins = bins

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.priors_ = {}
        self.hists_ = {}
        self.edges_ = {}
        self.n_features_ = X.shape[1]
        
        # Calcular bordes globales para cada feature
        for f in range(self.n_features_):
            self.edges_[f] = np.linspace(X[:, f].min(), X[:, f].max(), self.bins + 1)

        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = len(X_c) / len(X)
            self.hists_[c] = []
            
            # Histograma por característica (Naive)
            for f in range(self.n_features_):
                h, _ = np.histogram(X_c[:, f], bins=self.edges_[f], density=True)
                # Suavizado simple para evitar ceros
                h = np.maximum(h, 1e-7)
                self.hists_[c].append(h)
        return self

    def predict_proba(self, X):
        probs = []
        for x in X:
            class_probs = []
            for c in self.classes_:
                likelihood = 1.0
                for f in range(self.n_features_):
                    # Encontrar bin
                    bin_idx = np.digitize(x[f], self.edges_[f]) - 1
                    bin_idx = np.clip(bin_idx, 0, self.bins - 1)
                    likelihood *= self.hists_[c][f][bin_idx]
                
                class_probs.append(likelihood * self.priors_[c])
            probs.append(class_probs)
        
        probs = np.array(probs)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

class ParzenWindowClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en Ventana de Parzen (Multivariante usando KernelDensity).
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.priors_ = {}
        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = len(X_c) / len(X)
            # Kernel Density maneja multivariante automáticamente
            self.models_[c] = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(X_c)
        return self

    def predict(self, X):
        log_probs = []
        for c in self.classes_:
            # score_samples devuelve log(p(x|c))
            log_likelihood = self.models_[c].score_samples(X)
            log_posterior = log_likelihood + np.log(self.priors_[c])
            log_probs.append(log_posterior)
        
        log_probs = np.array(log_probs).T
        return self.classes_[np.argmax(log_probs, axis=1)]

class KnnDensityClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en densidad k_n Nearest Neighbor.
    P(x|c) ~ k / (N_c * V_k), donde V_k es el volumen hasta el k-vecino.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.counts_ = {}
        self.total_N_ = len(X)
        
        for c in self.classes_:
            X_c = X[y == c]
            self.counts_[c] = len(X_c)
            # Validar que k <= muestras en clase
            k_safe = min(self.n_neighbors, len(X_c))
            self.models_[c] = NearestNeighbors(n_neighbors=k_safe).fit(X_c)
        return self

    def predict(self, X):
        posteriors = []
        epsilon = 1e-10 # Evitar división por cero
        
        # Precalcular distancias para todas las clases
        class_scores = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, c in enumerate(self.classes_):
            k_safe = min(self.n_neighbors, self.counts_[c])
            dists, _ = self.models_[c].kneighbors(X, n_neighbors=k_safe)
            
            # Distancia al k-ésimo vecino
            d_k = dists[:, -1] 
            # En d-dimensiones, Volumen ~ radio^d. Usamos d_k como proxy de volumen.
            # P(C|x) ∝ (N_c / N_total) * (k / (N_c * d_k^d)) ∝ 1 / d_k^d aprox
            # Simplificación robusta: Score = Prior / Distancia
            
            score = (self.counts_[c] / self.total_N_) / (d_k + epsilon)
            class_scores[:, i] = score
            
        return self.classes_[np.argmax(class_scores, axis=1)]

# =============================================================================
# FLUJO PRINCIPAL
# =============================================================================

# 1. Carga de Datos (T1)
print("--- T1: Cargando Dataset (Vertebral Column) ---")
dataset = fetch_ucirepo(id=212)
X_raw = dataset.data.features.values
y_raw = LabelEncoder().fit_transform(dataset.data.targets.values.ravel())

# Normalización (Crucial para KNN y Parzen)
scaler = StandardScaler()
X_raw = scaler.fit_transform(X_raw)

print(f"Original: {X_raw.shape}, MeanIR: {compute_mean_ir(y_raw):.4f}")

# 2. Preparar Surtidos (T2)
print("\n--- T2: Generando Versiones Balanced y Imbalanced ---")
X_bal, y_bal = adjust_dataset_ir(X_raw, y_raw, 1.15, 'balance')
X_imb, y_imb = adjust_dataset_ir(X_raw, y_raw, 1.15, 'imbalance')

assortments = {
    "Balanced (MeanIR < 1.15)": (X_bal, y_bal),
    "Imbalanced (MeanIR > 1.15)": (X_imb, y_imb)
}

# 3. Configuración de Modelos y Parámetros (T3, T4)
models_config = {
    'MLE (QDA)': (QuadraticDiscriminantAnalysis(), {'reg_param': [0.0, 0.1, 0.5]}),
    'Naive Bayes': (GaussianNB(), {'var_smoothing': [1e-9, 1e-5]}),
    'Histogram Est.': (NaiveHistogramClassifier(), {'bins': [5, 10, 20]}),
    'Parzen Window': (ParzenWindowClassifier(), {'bandwidth': [0.1, 0.5, 1.0, 2.0]}),
    'kn-NN (Density)': (KnnDensityClassifier(), {'n_neighbors': [3, 5, 9, 15]}),
    'k-NN (Rule)': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 9, 15]})
}

# 4. Ejecución Experimental (Cross-Validation)
results = []

print("\n--- T3-T5: Ejecutando Experimentos y Optimización ---")

for dataset_name, (X_data, y_data) in assortments.items():
    print(f"\nProcesando: {dataset_name}...")
    
    # [cite_start]Outer CV: Separa Train+Val de Test [cite: 24]
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, (model, params) in models_config.items():
        fold_accuracies = []
        
        for train_idx, test_idx in outer_cv.split(X_data, y_data):
            X_train_full, X_test = X_data[train_idx], X_data[test_idx]
            y_train_full, y_test = y_data[train_idx], y_data[test_idx]
            
            # [cite_start]Inner CV: Separa Train de Validation para optimizar hiperparámetros [cite: 36]
            # GridSearchCV hace esto internamente (split train/val)
            clf = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
            clf.fit(X_train_full, y_train_full)
            
            # Evaluar en partición de Test (que el modelo nunca vio)
            best_model = clf.best_estimator_
            y_pred = best_model.predict(X_test)
            fold_accuracies.append(accuracy_score(y_test, y_pred))
            
        # [cite_start]Reportar media y std [cite: 41]
        results.append({
            "Assortment": dataset_name,
            "Method": model_name,
            "Accuracy Mean": np.mean(fold_accuracies),
            "Accuracy Std": np.std(fold_accuracies),
            "Best Params (Ex)": clf.best_params_
        })

# 5. Mostrar Resultados
print("\n" + "="*60)
print("RESUMEN DE RESULTADOS (T5)")
print("="*60)
df_res = pd.DataFrame(results)
# Ordenar por Dataset y luego por Accuracy
df_res = df_res.sort_values(by=['Assortment', 'Accuracy Mean'], ascending=[True, False])

print(df_res[['Assortment', 'Method', 'Accuracy Mean', 'Accuracy Std']].to_string(index=False))

print("\n" + "="*60)
print("Listo para redactar el informe T6.")