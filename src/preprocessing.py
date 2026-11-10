"""
Módulo de Preprocesamiento y Preparación de Datos para Deep Retention.
Maneja carga, limpieza, partición estratificada y transformaciones (OHE y Scaling)
sin fuga de información (data leakage).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Definición de columnas
NUMERICAL_FEATURES = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]

CATEGORICAL_FEATURES = [
    'Geography',
    'Gender'
]

DROP_COLUMNS = ['RowNumber', 'CustomerId', 'Surname']
TARGET_COLUMN = 'Exited'


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Carga el dataset bancario crudo desde archivo CSV."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"El archivo no existe en la ruta: {data_path}")
    df = pd.read_csv(data_path)
    return df


def split_data(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """
    Realiza una partición estratificada en 3 conjuntos: Train (70%), Val (15%), Test (15%).
    Mantiene la proporción exacta de desbalance de clientes que abandonan.
    """
    clean_df = df.drop(columns=DROP_COLUMNS, errors='ignore').copy()
    
    X = clean_df.drop(columns=[TARGET_COLUMN])
    y = clean_df[TARGET_COLUMN].values.astype(np.float32)

    # Primera partición: Train+Val vs Test
    total_val_test = test_size + val_size
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Segunda partición: Train vs Val
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_relative_size,
        stratify=y_train_val,
        random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_and_fit_preprocessor(X_train: pd.DataFrame, save_path: str = None) -> ColumnTransformer:
    """
    Construye y ajusta el preprocesador ÚNICAMENTE con los datos de entrenamiento.
    Aplica StandardScaler a variables continuas y OneHotEncoder a variables categóricas.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    preprocessor.fit(X_train)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(preprocessor, save_path)
        print(f"Preprocesador guardado exitosamente en: {save_path}")
        
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Obtiene los nombres legibles de todas las columnas de salida tras la transformación."""
    num_cols = NUMERICAL_FEATURES
    cat_cols = list(preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES))
    return num_cols + cat_cols


def transform_data(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Transforma un conjunto de datos usando el preprocesador ya ajustado."""
    return preprocessor.transform(X).astype(np.float32)


def preprocess_single_client(client_data: dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """Prepara un único registro de cliente para inferencia en tiempo real."""
    df_single = pd.DataFrame([client_data])
    return transform_data(preprocessor, df_single)

